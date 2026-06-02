from pathlib import Path
import numpy as np
import pandas as pd
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
    df = load_processed_trials(normalized=True)

    # ------------------------------------------------------------
    # Align spikes to cue and movement
    # ------------------------------------------------------------
    cue_state = 6
    mov_state = 68
    go_state = 4
    df["t_cue"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], cue_state),
        axis=1,
    )

    df["t_mov"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], mov_state),
        axis=1,
    )
    df["t_go"] = df.apply(
        lambda r: get_state_onset(r["states_onset"], r["states"], go_state),
        axis=1,
    )

    # ------------------------------------------------------------
    # Split existing SDF into cue- and movement-aligned windows
    # ------------------------------------------------------------
    bin_size = 0.001
    cue_sdf = df.apply(
        lambda row: slice_sdf_to_event(
            row,
            event_time_col="t_cue",
            t_start=-0.5,
            t_end=0.8,
            bin_size=bin_size,
        ),
        axis=1,
    )

    mov_sdf = df.apply(
        lambda row: slice_sdf_to_event(
            row,
            event_time_col="t_mov",
            t_start=-0.8,
            t_end=0.5,
            bin_size=bin_size,
        ),
        axis=1,
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

    df = df[
        df["stitched_rate"].apply(
            lambda x: np.all(np.isfinite(np.asarray(x, dtype=float)))
        )
    ].copy()

    stitched_time, stitch_x, cue_t, mov_t = make_stitched_time(
        df.iloc[0],
        cue_start=cue_start,
        cue_end=cue_end,
        mov_start=mov_start,
        mov_end=mov_end,
    )

    df["stitched_time"] = [stitched_time] * len(df)

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

    axes_raw, axes_ortho, units = fit_tdr_axes(df, regressors=main_effect_regressors)

    # ------------------------------------------------------------
    # Condition-averaged trajectories + projections
    # ------------------------------------------------------------

    # 2 x 2 x 2 = 8 conditions:
    # effector: reach / saccade
    # reach_hand: ipsi / contra
    # target_hemifield: ipsi / contra
    condition_factors = {
        "all_conditions": ("effector", "reach_hand", "target_hemifield"),
    }

    condition_pops = {}
    projections_by = {}
    axis_names = list(main_effect_regressors)

    for name, cols in condition_factors.items():
        condition_pops[name] = condition_mean_population(
            df,
            units,
            condition_cols=cols,
        )

        projections_by[name], _ = project_trajectories(
            condition_pops[name],
            axes_ortho,
            main_effect_regressors,
        )

    # aliases for full 8-condition plots
    condition_pop = condition_pops["all_conditions"]
    projections = projections_by["all_conditions"]

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_tdr_time_hand_target_3d(
        projections,
        stitched_time,
        axis_names,
        hand_axis="H",
        target_axis="T",
        out_path=Path("plots/tdr/tdr_time_hand_target_8_conditions.html"),
        downsample=2,
    )
    exit()

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

    cond_label = lambda c: c[0] if isinstance(c, tuple) else str(c)

    axis_timecourse_configs = [
        ("E", "effector", [("reach",), ("saccade",)]),
        ("T", "target_hemifield", [("contra",), ("ipsi",)]),
        ("H", "reach_hand", [("contra",), ("ipsi",)]),
    ]
    for axis, factor, cond_order in axis_timecourse_configs:
        plot_tdr_axis_timecourses(
            projections_by[factor],
            stitched_time,
            axis_names,
            axes_to_plot=(axis,),
            cond_order=cond_order,
            cond_label_fn=cond_label,
            cue_time=0.0,
            mov_time=1.6,
            lw=2.0,
            out_path=Path(f"plots/tdr/tdr_axis_timecourse_{axis}_by_{factor}.png"),
        )


if __name__ == "__main__":
    main()
