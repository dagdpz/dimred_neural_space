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
    stitched_time, stitched_segment, stitch_x, cue_t, mov_t = make_stitched_time(
        df.iloc[0],
        cue_start=cue_start,
        cue_end=cue_end,
        mov_start=mov_start,
        mov_end=mov_end,
    )

    df["stitched_time"] = [stitched_time] * len(df)
    df["stitched_segment"] = [stitched_segment] * len(df)

    # ------------------------------------------------------------
    # Fit TDR axes
    # ------------------------------------------------------------
    main_effect_regressors = ("E", "T", "H")

    axes_raw, axes_ortho, units = fit_tdr_axes(
        df,
        regressors=main_effect_regressors,
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
