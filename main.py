from pathlib import Path
import pandas as pd
import numpy as np
from dPCA import dPCA
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

from scripts.plotting import *
from scripts.utils import *


def clean_mat_value(x):
    """
    Some fields have just one item or array in them
    """
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        if x.size == 1:
            return x.item()
        return x
    return x


def decode_reach_hand_label(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    if x == 1:
        return "left"
    elif x == 2:
        return "right"
    else:
        return np.nan


def ipsi_contra(side, reference_side):
    if pd.isna(side) or pd.isna(reference_side):
        return np.nan
    return "ipsi" if side == reference_side else "contra"


def decode_effector_label(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    if x == 3:
        return "saccade"
    elif x == 4:
        return "reach"
    elif x == 6:
        return "saccade_reach"
    else:
        return np.nan


def tar_pos_to_side(x):
    """
    tar_pos is complex: X + iY.
    Real part = X position.
    negative X = left
    positive X = right
    """
    if pd.isna(x):
        return np.nan

    x_real = np.real(x)

    if x_real < 0:
        return "left"
    elif x_real > 0:
        return "right"
    else:
        return "center"


def pulvinar_to_side(x):
    if x == "dPulv_r":
        return "right"
    elif x == "dPulv_l":
        return "left"
    else:
        return np.nan


def get_state_onset(states_onset, states, state_id):
    """Return onset time for the first occurrence of state_id, or NaN if missing."""
    if states_onset is None or states is None:
        return np.nan
    s = np.asarray(states, dtype=float).ravel()
    t = np.asarray(states_onset, dtype=float).ravel()
    mask = s == float(state_id)
    return float(t[np.where(mask)[0][0]])


def trial_alignment_to_cue(row, cue_state):
    """Absolute cue time and spike times relative to cue for one trial."""
    t_cue = get_state_onset(row["states_onset"], row["states"], cue_state)
    spike_arrival_times = np.asarray(row["arrival_times"], dtype=float).ravel()
    spikes_arrival_times_rel = spike_arrival_times - t_cue
    return pd.Series(
        {
            "t_cue": t_cue,
            "arrival_times_rel": spikes_arrival_times_rel,
        }
    )


def spike_times_to_sdf(spike_times, t_start, t_end, bin_size=0.001, sigma=0.05):
    """Bin spikes, Gaussian-smooth counts, convert to Hz; returns (time_axis, rate) or NaN rate if spikes missing."""
    if isinstance(spike_times, float) and np.isnan(spike_times):
        time = np.arange(t_start, t_end, bin_size)
        return time, np.full_like(time, np.nan, dtype=float)
    spike_times = np.asarray(spike_times, dtype=float).ravel()
    time_edges = np.arange(t_start, t_end + bin_size, bin_size)
    time = time_edges[:-1] + bin_size / 2
    counts, _ = np.histogram(spike_times, bins=time_edges)
    sigma_bins = sigma / bin_size
    smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=sigma_bins)
    rate = smoothed_counts / bin_size
    return time, rate


def load_population_spike_data(filepath):
    # --- Load MATLAB struct: one row per unit, nested trials per unit ---
    mat = loadmat("data/population_Linus_20160518.mat")
    population = mat["population"]

    # --- Unpack selected top-level fields per unit (IDs, site, trial struct array) ---
    data = {}
    selected_keys = ["unit_ID", "target", "trial"]
    for field in population.dtype.names:
        if field not in selected_keys:
            continue
        values = []
        for item in population[field][0]:
            try:
                values.append(clean_mat_value(item[0]))
            except Exception:
                values.append(clean_mat_value(item))
        data[field] = values

    # --- One DataFrame row per (unit, trial); copy trial-level fields from nested structs ---
    rows = []
    for unit_idx, unit_id in enumerate(data["unit_ID"]):
        # Convert recorded pulvinar side to hemifield
        recorded_pulvinar = clean_mat_value(data["target"][unit_idx])
        pulvinar_hemifield = (
            "left" if pulvinar_to_side(recorded_pulvinar) == "right" else "right"
        )

        # extract trial data
        unit_trials = data["trial"][unit_idx]
        trial_fields = unit_trials.dtype.names
        for trial_idx in range(unit_trials.shape[0]):
            row = {
                "unit_index": unit_idx,
                "unit_ID": unit_id,
                "pulvinar_hemifield": pulvinar_hemifield,
                "trial_index": trial_idx,  # Use an ID to track trial for a neuron
            }
            for field in trial_fields:
                value = unit_trials[field][trial_idx]
                row[field] = clean_mat_value(value)
            rows.append(row)
    return pd.DataFrame(rows)


def process_labels_and_filter(df):
    # --- Keep trials with correct choice and success; drop unused columns ---
    df = df[df["type"] == 4]
    df = df[df["choice"] == 0]
    df = df[df["success"] == 1]

    # Convert reach hand from numeric to string
    df["reach_hand"] = df["reach_hand"].apply(decode_reach_hand_label)
    df["reach_hand"] = df.apply(
        lambda row: ipsi_contra(row["reach_hand"], row["pulvinar_hemifield"]),
        axis=1,
    )

    # Dont use joint saccadereach condition for now
    df["effector"] = df["effector"].apply(decode_effector_label)
    df = df[df["effector"] != "saccade_reach"]

    # Target position
    df["target_side"] = df["tar_pos"].apply(tar_pos_to_side)
    df["target_hemifield"] = df.apply(
        lambda row: ipsi_contra(row["target_side"], row["pulvinar_hemifield"]),
        axis=1,
    )

    select_columns = [
        "unit_index",
        "trial_index",
        "unit_ID",
        "pulvinar_hemifield",
        "reach_hand",
        "effector",
        "target_hemifield",
        "trial_onset_time",
        "run_onset_time",
        "states_onset",
        "states",
        "arrival_times",
    ]
    df = df[select_columns]
    return df


def main(seed=0):
    """
    End-to-end pipeline: load pulvinar population .mat, flatten to trials, filter and derive labels,
    align spikes to cue, compute SDFs, save per-unit figures, then build a tensor and run dPCA.

    Parameters
    ----------
    seed : int
        RNG seed for subsampling trials when balancing counts for the dPCA tensor (time axis ``t``).
    """
    rng = np.random.default_rng(seed)

    plots_dir = Path("plots/dpca")
    filepath = "data/population_Linus_20160518.mat"
    df = load_population_spike_data(filepath)
    df = process_labels_and_filter(df)

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
    plot_sdf_per_condition("effector", condition_cols, data=df, t_start=-0.5, t_end=2)
    plot_sdf_per_condition("reach_hand", condition_cols, data=df, t_start=-0.5, t_end=2)
    plot_sdf_per_condition(
        "target_hemifield", condition_cols, data=df, t_start=-0.5, t_end=2
    )

    # ---------------------------------------------------------------------------------------------
    # --- Build balanced trial tensor R[pseudotrial, unit, time, effector, hand, target]; dPCA fit ---
    # ---------------------------------------------------------------------------------------------
    unit_col = "unit_index"
    cond_cols = ["effector", "reach_hand", "target_hemifield"]

    analysis_df = df.copy()

    units = np.sort(analysis_df[unit_col].unique())
    cond_levels = []
    for col in cond_cols:
        cond_levels.append(np.sort(analysis_df[col].unique()))
    t_ref = np.asarray(analysis_df.iloc[0]["sdf_time"], dtype=float)
    T = len(t_ref)

    trial_counts = (
        analysis_df.groupby([unit_col, *cond_cols]).size().reset_index(name="n_trials")
    )
    full_index = pd.MultiIndex.from_product(
        [units, *cond_levels],
        names=[unit_col, *cond_cols],
    )
    trial_counts_full = (
        trial_counts.set_index([unit_col, *cond_cols])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    unit_to_i = {int(u): i for i, u in enumerate(units)}
    cond_to_i = [{level: i for i, level in enumerate(levels)} for levels in cond_levels]

    n_pseudotrials = int(trial_counts_full["n_trials"].min())
    trial_data = np.full(
        (n_pseudotrials, len(units), T, *[len(levels) for levels in cond_levels]),
        np.nan,
        dtype=float,
    )
    for keys, g in analysis_df.groupby([unit_col, *cond_cols]):
        unit = keys[0]
        cond_values = keys[1:]

        u_idx = unit_to_i[unit]
        c_idx = tuple(cond_to_i[j][cond_values[j]] for j in range(len(cond_cols)))
        rates = np.stack(
            g["sdf_rate"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
        )

        """ # Can be used for balancing
        sampled_idx = rng.choice(
            rates.shape[0],
            size=n_pseudotrials,
            replace=False,
        )
        sampled_rates = rates[sampled_idx] """

        trial_data[(slice(None), u_idx, slice(None), *c_idx)] = rates[:n_pseudotrials]

    # dPCA expects protected axes (time) at the end for stable CV indexing.
    # R: [trial, unit, time, effector, hand, target] -> [trial, unit, effector, hand, target, time]
    trial_data_dpca = np.transpose(trial_data, (0, 1, 3, 4, 5, 2))
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
