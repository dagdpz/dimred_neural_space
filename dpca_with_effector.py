from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from dPCA import dPCA

from preprocess import PROCESSED_TRIALS_PATH
from scripts.plotting import *
from scripts.utils import *


def load_processed_trials(path=PROCESSED_TRIALS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python preprocess.py` first."
        )
    return pd.read_pickle(path)


def main(seed=0):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run dPCA.
    """
    rng = np.random.default_rng(seed)

    plots_dir = Path("plots/dpca")
    df = load_processed_trials()

    # --- Align spikes and event times to cue onset (state 6)
    cue_state = 6
    align = df.apply(
        lambda row: trial_alignment_to_cue(row, cue_state),
        axis=1,
    )
    df = pd.concat([df, align], axis=1)

    # --- Spike density functions (Gaussian-smoothed Hz), time axis aligned to cue (state 6) ---
    t_start = -0.5
    t_end = 2
    bin_size = 0.001  # 1 ms bins
    sigma = 0.05  # 50 ms Gaussian smoothing
    sdf_results = df["arrival_times_rel"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=t_start,
            t_end=t_end,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    df["sdf_time"] = sdf_results.apply(lambda x: x[0])
    df["sdf_rate"] = sdf_results.apply(lambda x: x[1])
    
    condition_cols = ["effector", "reach_hand", "target_hemifield"]
    # plot_sdf_per_condition("effector", condition_cols, data=df, t_start=-0.5, t_end=2)
    # plot_sdf_per_condition("reach_hand", condition_cols, data=df, t_start=-0.5, t_end=2)
    # plot_sdf_per_condition(
    #     "target_hemifield", condition_cols, data=df, t_start=-0.5, t_end=2
    # )

    # ---------------------------------------------------------------------------------------------
    # --- Build balanced trial tensor R[pseudotrial, unit, time, effector, hand, target]; dPCA fit ---
    # ---------------------------------------------------------------------------------------------
    unit_cols = ["session", "unit_ID"]
    cond_cols = ["effector", "reach_hand", "target_hemifield"]

    analysis_df = df.copy()

    units = (
        analysis_df[unit_cols]
        .drop_duplicates()
        .sort_values(unit_cols)
        .itertuples(index=False, name=None)
    )
    units = list(units)
    
    cond_levels = [np.sort(analysis_df[col].unique()) for col in cond_cols]
    t_ref = np.asarray(analysis_df.iloc[0]["sdf_time"], dtype=float)
    T = len(t_ref)

    trial_counts = (
        analysis_df.groupby([*unit_cols, *cond_cols]).size().reset_index(name="n_trials")
    )

    # units are (session, unit_ID) pairs — not a Cartesian product of separate levels
    full_index = pd.MultiIndex.from_tuples(
        [(*u, *c) for u in units for c in product(*cond_levels)],
        names=[*unit_cols, *cond_cols],
    )
    trial_counts_full = (
        trial_counts.set_index([*unit_cols, *cond_cols])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    unit_to_i = {u: i for i, u in enumerate(units)}
    cond_to_i = [{level: i for i, level in enumerate(levels)} for levels in cond_levels]

    n_pseudotrials = int(trial_counts_full["n_trials"].min())
    trial_data = np.full(
        (n_pseudotrials, len(units), T, *[len(levels) for levels in cond_levels]),
        np.nan,
        dtype=float,
    )
    
    for keys, g in analysis_df.groupby([*unit_cols, *cond_cols]):
        unit_key = tuple(keys[: len(unit_cols)])
        cond_values = keys[len(unit_cols) :]

        u_idx = unit_to_i[unit_key]
        c_idx = tuple(cond_to_i[j][cond_values[j]] for j in range(len(cond_cols)))
        rates = np.stack(
            g["sdf_rate"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
        )

        trial_data[(slice(None), u_idx, slice(None), *c_idx)] = rates[:n_pseudotrials]

    trial_data_dpca = np.transpose(trial_data, (0, 1, 3, 4, 5, 2))
    print(trial_data_dpca.shape)
    exit()
    R_dpca = np.nanmean(trial_data_dpca, axis=0)

    dpca = dPCA.dPCA(labels="ehst", regularizer="auto")
    dpca.protect = ["t"]
    Z = dpca.fit_transform(R_dpca, trial_data_dpca)

    print("n_pseudotrials:", n_pseudotrials)
    print("R_dpca shape:", R_dpca.shape)
    print("trial_data_dpca shape:", trial_data_dpca.shape)

    print(trial_counts_full["n_trials"].describe())
    print(trial_counts_full.sort_values("n_trials").head(20))

    plot_dpca_results(
        dpca,
        Z,
        t_ref,
        cond_levels,
        cond_cols,
        plots_dir,
    )


if __name__ == "__main__":
    main()
