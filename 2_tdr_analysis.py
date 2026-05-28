from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from dPCA import dPCA
from sklearn.linear_model import LinearRegression

from scripts.preprocess import *
from scripts.plotting import *
from scripts.utils import *
from scripts.state_space_functions import *
from scripts.tdr_functions import *
from scripts.tdr_plotting import *


def main(seed=0):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    rng = np.random.default_rng(seed)
    df = load_processed_trials()

    # ------------------------------------------------------------
    # Align spikes to cue and movement
    # ------------------------------------------------------------
    cue_state = 6
    mov_state = 68
    cue_align = df.apply(
        lambda row: trial_alignment_to_state(row, cue_state),
        axis=1,
    )
    cue_align = cue_align.rename(
        columns={
            "t_state": "t_cue",
            "arrival_times_rel": "arrival_times_cue",
        }
    )
    mov_align = df.apply(
        lambda row: trial_alignment_to_state(row, mov_state),
        axis=1,
    )
    mov_align = mov_align.rename(
        columns={
            "t_state": "t_mov",
            "arrival_times_rel": "arrival_times_mov",
        }
    )
    df = pd.concat([df, cue_align, mov_align], axis=1)

    go_state = 4
    df["t_go"] = df.apply(
        lambda r: get_state_onset(r["states_onset"], r["states"], go_state),
        axis=1,
    )
    
    # ------------------------------------------------------------
    # SDFs
    # ------------------------------------------------------------
    bin_size = 0.001  # 1 ms bins
    sigma = 0.02  # 50 ms Gaussian smoothing
    cue_sdf = df["arrival_times_cue"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=-0.5,
            t_end=0.8,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    mov_sdf = df["arrival_times_mov"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=-0.8,
            t_end=0.5,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    df["sdf_time_cue"] = cue_sdf.apply(lambda x: x[0])
    df["sdf_rate_cue"] = cue_sdf.apply(lambda x: x[1])
    df["sdf_time_mov"] = mov_sdf.apply(lambda x: x[0])
    df["sdf_rate_mov"] = mov_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Stitch cue and movement times
    # ------------------------------------------------------------
    mov_start = -0.8
    mov_end = 0.5
    cue_start = -0.5
    cue_end = 0.8

    df["stitched_rate"] = df.apply(
        lambda row: make_stitched_sdf(
            row,
            cue_start=cue_start,
            cue_end=cue_end,
            mov_start=mov_start,
            mov_end=mov_end,
        ),
        axis=1,
    )
    stitched_time, stitch_x, cue_t, mov_t = make_stitched_time(
        df.iloc[0],
        cue_start=cue_start,
        cue_end=cue_end,
        mov_start=mov_start,
        mov_end=mov_end,
    )

    df["stitched_time"] = [stitched_time] * len(df)

    # ------------------------------------------------------------
    # Remove units with low firing rates
    # ------------------------------------------------------------

    threshold = 2.0
    cue_rates = (
        df.groupby(["session", "unit_ID"])["sdf_rate_cue"]
        .apply(mean_rate_from_series)
        .rename("cue_rate")
    )
    mov_rates = (
        df.groupby(["session", "unit_ID"])["sdf_rate_mov"]
        .apply(mean_rate_from_series)
        .rename("mov_rate")
    )
    unit_stats = pd.concat([cue_rates, mov_rates], axis=1).reset_index()
    unit_stats["mean_rate"] = (unit_stats["cue_rate"] + unit_stats["mov_rate"]) / 2

    good_units = unit_stats[unit_stats["mean_rate"] >= threshold][
        ["session", "unit_ID"]
    ]
    df = df.merge(good_units, on=["session", "unit_ID"], how="inner")

    plt.figure(figsize=(6, 4))
    plt.hist(unit_stats["mean_rate"], bins=50, edgecolor="black", alpha=0.75)
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"threshold = {threshold} Hz",
    )
    plt.xlabel("Mean firing rate (Hz)")
    plt.ylabel("Number of units")
    plt.title("Distribution of unit mean firing rates")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()

    # ------------------------------------------------------------
    # Regressors
    # ------------------------------------------------------------
    df = add_tdr_regressors(df)
    df = df.dropna(
        subset=[
            "E",
            "T",
            "H",
            "t_cue",
            "t_mov",
            "sdf_rate_cue",
            "sdf_rate_mov",
        ]
    ).copy()

    # ------------------------------------------------------------
    # Fit TDR axes
    # ------------------------------------------------------------
    main_effect_regressors = ("E", "T", "H")

    axes_raw, axes_ortho, units = fit_tdr_axes(
        df, regressors=main_effect_regressors
    )

    # ------------------------------------------------------------
    # Condition-averaged trajectories
    # ------------------------------------------------------------
    condition_pop = condition_mean_population(
        df,
        units,
        condition_cols=("effector",),
    )

    projections, axis_names = project_trajectories(
        condition_pop,
        axes_ortho,
        main_effect_regressors,
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_tdr_reach_vs_saccade_3d(
        projections,
        stitched_time,
        axis_names,
        x_axis="E",
        y_axis="T",
        z_axis="H",
        out_path=Path("plots/tdr/tdr_reach_vs_saccade_3d.html"),
        downsample=2,
    )

    plot_tdr_timecolor_3d(
        projections,
        stitched_time,
        axis_names,
        x_axis="E",
        y_axis="T",
        z_axis="H",
        out_path=Path("plots/tdr/tdr_reach_saccade_side_by_side_timecolor.html"),
        downsample=2,
    )


if __name__ == "__main__":
    main()
