from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse

from scripts.preprocess import *
from scripts.plotting import *
from scripts.utils import *
from scripts.state_space_functions import *
from scripts.tdr_functions import *
from scripts.tdr_plotting import *
from scripts.decoding_functions import *


def main(
    use_pca_denoising=False,
    plot=False,
    plots_dir=Path("plots/epoched_tdr_cue"),
):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_trials()

    # Analyze only one monkey
    df = df[df["session"].str.contains("Flaffus", na=False)].copy()

    # ------------------------------------------------------------
    # Get time of onset of CUE, MOV and GO
    # ------------------------------------------------------------
    cue_state = 6
    mov_state = 68
    mov_end_state = 69
    go_state = 4
    df["t_cue"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], cue_state),
        axis=1,
    )
    df["t_go"] = df.apply(
        lambda r: get_state_onset(r["states_onset"], r["states"], go_state),
        axis=1,
    )
    df["t_mov"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], mov_state),
        axis=1,
    )
    df["t_mov_end"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], mov_end_state),
        axis=1,
    )

    # ------------------------------------------------------------
    # Cue-aligned analysis window: -0.5 to 2.0 s after cue
    # ------------------------------------------------------------
    bin_size = 0.001
    cue_sdf = df.apply(
        lambda row: slice_sdf_to_event(
            row,
            event_time_col="t_cue",
            t_start=-0.5,
            t_end=2.0,
            bin_size=bin_size,
        ),
        axis=1,
    )

    df["analysis_time"] = cue_sdf.apply(lambda x: x[0])
    df["analysis_rate"] = cue_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Align other event times to cue
    # ------------------------------------------------------------
    df["t_mov"] = df["t_mov"].to_numpy(dtype=float) - df["t_cue"].to_numpy(dtype=float)

    df["t_mov_end"] = df["t_mov_end"].to_numpy(dtype=float) - df["t_cue"].to_numpy(
        dtype=float
    )

    df["t_go"] = df["t_go"].to_numpy(dtype=float) - df["t_cue"].to_numpy(dtype=float)

    df["t_cue"] = 0.0

    # ------------------------------------------------------------
    # Remove rows with NaNs in cue or movement SDFs
    # ------------------------------------------------------------
    before = len(df)
    df = df[
        df["analysis_rate"].apply(is_valid_array)
        & df["analysis_time"].apply(is_valid_array)
    ].reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} rows with NaNs in SDFs")
    print(f"Remaining rows: {after}")
    analysis_time = np.asarray(df["analysis_time"].iloc[0], dtype=float)

    trials_per_effector = count_rows_per_unit_condition(
        df,
        unit_cols=("unit_ID",),
        condition_cols=("effector",),
    )

    trials_per_effector.to_csv(
        plots_dir / "rows_per_unit_per_effector.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Keep only units with enough trials in every 8-condition cell
    # effector x reach_hand x target_hemifield
    # ------------------------------------------------------------
    min_trials_per_condition = 5
    # You can increase this later, e.g. 5 or 10, if you want stricter averaging.

    condition_cols_8cond = ("effector", "reach_hand", "target_hemifield")
    condition_levels_8cond = {
        "effector": ["reach", "saccade"],
        "reach_hand": ["ipsi", "contra"],
        "target_hemifield": ["ipsi", "contra"],
    }

    trials_per_8cond = count_rows_per_unit_condition(
        df,
        unit_cols=("unit_ID",),
        condition_cols=condition_cols_8cond,
        condition_levels=condition_levels_8cond,
    )

    trials_per_8cond.to_csv(
        plots_dir / "rows_per_unit_per_8condition.csv",
        index=False,
    )

    good_units = trials_per_8cond.groupby("unit_ID")["n_rows"].min().reset_index()

    good_units = good_units[good_units["n_rows"] >= min_trials_per_condition][
        ["unit_ID"]
    ]

    n_units_before = df["unit_ID"].nunique()

    df = df.merge(
        good_units,
        on="unit_ID",
        how="inner",
    ).reset_index(drop=True)

    n_units_after = df["unit_ID"].nunique()

    print(f"Units before 8-condition filtering: {n_units_before}")
    print(
        f"Units after requiring >= {min_trials_per_condition} trial(s) "
        f"in every effector x hand x target condition: {n_units_after}"
    )
    print(f"Rows after unit filtering: {len(df)}")

    # ------------------------------------------------------------
    # Common event markers for plots
    # ------------------------------------------------------------
    event_times = [
        0.0,
        np.nanmedian(df["t_go"]),
    ]
    event_labels = ["Cue", "GO"]
    event_colors = ["black", "black"]
    event_opacities = [0.04, 0.04]
    event_linestyles = ["--", ":"]
    event_linewidths = [1.2, 1.2]
    event_alphas = [0.7, 0.8]

    # ------------------------------------------------------------
    # Plot 10 random SDFs, vertically shifted
    # ------------------------------------------------------------
    if plot:
        plot_random_shifted_sdfs(
            df,
            analysis_time,
            event_times=event_times,
            event_labels=event_labels,
            event_linestyles=event_linestyles,
            event_colors=event_colors,
            event_linewidths=event_linewidths,
            event_alphas=event_alphas,
            out_path=plots_dir / "random_10_sdfs_shifted.png",
            xlabel="Time relative to cue onset (s)",
        )

    columns_to_keep = [
        "unit_ID",
        "trial_index",
        "pulvinar_hemifield",
        "reach_hand",
        "effector",
        "tar_pos",
        "fix_pos",
        "recorded_side",
        "target_hemifield",
        "t_cue",
        "t_mov",
        "t_go",
        "t_mov_end",
        "analysis_rate",
        "analysis_time",
    ]
    df = df[columns_to_keep].copy()

    # ------------------------------------------------------------
    # Regressors
    # ------------------------------------------------------------

    # Condition-independent regressors
    df = add_condition_independent_regressors(df)

    if plot:
        plot_first_row_ci_windows(
            df,
            out_path=plots_dir / "ci_windows_first_row.png",
        )

    # ------------------------------------------------------------
    # Hand Regressor
    # ------------------------------------------------------------
    df["hand"] = df["reach_hand"].map({"contra": 1.0, "ipsi": -1.0})
    df["hand"] = [
        float(h) * np.ones_like(np.asarray(t, dtype=float))
        for h, t in zip(df["hand"], df["analysis_time"])
    ]

    # ------------------------------------------------------------
    # Space regressors: target x/y
    # ------------------------------------------------------------
    df = add_target_xy_regressors(df)

    # Print unique target positions after ipsi/contra x-sign correction
    unique_positions = (
        df[["space_x", "space_y"]]
        .drop_duplicates()
        .sort_values(["space_x", "space_y"])
        .reset_index(drop=True)
    )

    print("\nUnique target positions after space regressor transform:")
    print(unique_positions)
    print(f"\nNumber of unique target positions: {len(unique_positions)}")

    # Space as cue-defined regressors
    df = mask_regressors(
        df,
        regressors=("space_x", "space_y"),
        mask_col="cueCI",
        suffix="_cue",
    )

    target_counts = (
        df.dropna(subset=["space_x", "space_y"])
        .groupby(["space_x", "space_y"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["space_x", "space_y"])
    )

    print("\nUnique target positions:")
    print(target_counts)
    print(f"\nNumber of unique target positions: {len(target_counts)}")

    target_counts.to_csv(
        plots_dir / "target_position_counts.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Effector/action regressors
    # ------------------------------------------------------------
    df = add_effector_regressors(df)

    df = add_effector_masks(df)

    action_regs = ("saccade", "ipsi_hand", "contra_hand")
    space_regs = ("space_x", "space_y")
    regs = []

    for action in action_regs:
        for space in space_regs:
            col = f"{action}_{space}"
            df[col] = df[action] * df[space]
            regs.append(col)

    # Planning action regressors
    df = mask_regressors(
        df,
        regressors=regs,
        mask_col="eff_plan_mask",
        suffix="_plan",
    )

    # Movement action regressors
    df = mask_regressors(
        df,
        regressors=regs,
        mask_col="eff_mov_mask",
        suffix="_mov",
    )

    drop_cols = [
        "unit_ID",
        "trial_index",
        "t_cue",
        "t_mov",
        "t_go",
        "t_mov_end",
        "analysis_rate",
        "analysis_time",
        # CI regressors
        "cueCI",
        "planCI",
        "goCI",
        "movCI",
        # Regressors
        "hand",
        "space_x_cue",
        "space_y_cue",
        "saccade_space_x_plan",
        "saccade_space_y_plan",
        "ipsi_hand_space_x_plan",
        "ipsi_hand_space_y_plan",
        "contra_hand_space_x_plan",
        "contra_hand_space_y_plan",
        "saccade_space_x_mov",
        "saccade_space_y_mov",
        "ipsi_hand_space_x_mov",
        "ipsi_hand_space_y_mov",
        "contra_hand_space_x_mov",
        "contra_hand_space_y_mov",
    ]

    df = df.dropna(subset=drop_cols).copy()
    df = df[
        df["analysis_rate"].apply(is_valid_array)
        & df["analysis_time"].apply(is_valid_array)
    ].reset_index(drop=True)

    # ------------------------------------------------------------
    # Fit TDR axes
    # ------------------------------------------------------------
    regressors = (
        # CI regressors
        "cueCI",
        "planCI",
        "goCI",
        "movCI",
        # Regressors
        "hand",
        "space_x_cue",
        "space_y_cue",
        "saccade_space_x_plan",
        "saccade_space_y_plan",
        "ipsi_hand_space_x_plan",
        "ipsi_hand_space_y_plan",
        "contra_hand_space_x_plan",
        "contra_hand_space_y_plan",
        "saccade_space_x_mov",
        "saccade_space_y_mov",
        "ipsi_hand_space_x_mov",
        "ipsi_hand_space_y_mov",
        "contra_hand_space_x_mov",
        "contra_hand_space_y_mov",
    )

    axes_raw, axes_ortho, units, task_regressors = fit_tdr_axes(
        df,
        regressors=regressors,
        rate_col="analysis_rate",
        time_col="analysis_time",
    )

    # ------------------------------------------------------------
    # TDR beta/effect-size statistics
    # ------------------------------------------------------------
    beta_unit_stats, beta_summary, beta_axis_summary = summarize_tdr_beta_effect_sizes(
        axes_raw,
        task_regressors,
        units=units,
    )

    beta_unit_stats.to_csv(
        plots_dir / "tdr_beta_effect_sizes_by_unit.csv",
        index=False,
    )

    beta_summary.to_csv(
        plots_dir / "tdr_beta_effect_size_summary.csv",
        index=False,
    )

    beta_axis_summary.to_csv(
        plots_dir / "tdr_beta_axis_strength_summary.csv",
        index=False,
    )

    print("\nTDR beta effect-size summary:")
    print(beta_summary)

    print("\nTDR raw axis strength:")
    print(beta_axis_summary)

    # ------------------------------------------------------------
    # Condition-averaged trajectories + projections
    # ------------------------------------------------------------
    condition_pops_target_pos = condition_mean_population(
        df,
        units,
        condition_cols=("space_x", "space_y"),
        unit_cols=("unit_ID",),  # use ("session", "unit_ID") if you keep session
        rate_col="analysis_rate",
    )

    projections_target_pos, _ = project_trajectories(
        condition_pops_target_pos,
        axes_ortho,
        task_regressors,
    )
    axis_names = list(task_regressors)
    cond_order = sorted(projections_target_pos.keys())

    plot_target_position_axis_timecourse(
        projections_target_pos,
        analysis_time,
        axis_names,
        axis="space_x_cue",
        cond_order=cond_order,
        out_path=plots_dir / "target_position_time_space_x_cue_colored_by_xy.png",
        downsample=5,
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    plot_target_position_axis_timecourse(
        projections_target_pos,
        analysis_time,
        axis_names,
        axis="space_y_cue",
        cond_order=cond_order,
        out_path=plots_dir / "target_position_time_space_y_cue_colored_by_xy.png",
        downsample=5,
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    # ------------------------------------------------------------
    # Debug 8-condition counts before condition averaging
    # ------------------------------------------------------------
    expected_8cond = [
        ("reach", "ipsi", "ipsi"),
        ("reach", "ipsi", "contra"),
        ("reach", "contra", "ipsi"),
        ("reach", "contra", "contra"),
        ("saccade", "ipsi", "ipsi"),
        ("saccade", "ipsi", "contra"),
        ("saccade", "contra", "ipsi"),
        ("saccade", "contra", "contra"),
    ]

    counts_8cond = (
        df.groupby(["effector", "reach_hand", "target_hemifield"])
        .size()
        .rename("n_rows")
        .reindex(
            pd.MultiIndex.from_tuples(
                expected_8cond,
                names=["effector", "reach_hand", "target_hemifield"],
            ),
            fill_value=0,
        )
        .reset_index()
    )

    print("\n8-condition row counts before condition averaging:")
    print(counts_8cond)

    counts_8cond.to_csv(
        plots_dir / "8_condition_row_counts_before_averaging.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 8-condition averaged trajectories:
    # effector x hand x target hemifield
    # ------------------------------------------------------------
    condition_pops_8cond = condition_mean_population(
        df,
        units,
        condition_cols=("effector", "reach_hand", "target_hemifield"),
        unit_cols=("unit_ID",),
        rate_col="analysis_rate",
    )

    projections_8cond, _ = project_trajectories(
        condition_pops_8cond,
        axes_ortho,
        task_regressors,
    )

    axis_names = list(task_regressors)

    cond_order_8cond = [
        ("reach", "ipsi", "ipsi"),
        ("reach", "ipsi", "contra"),
        ("reach", "contra", "ipsi"),
        ("reach", "contra", "contra"),
        ("saccade", "ipsi", "ipsi"),
        ("saccade", "ipsi", "contra"),
        ("saccade", "contra", "ipsi"),
        ("saccade", "contra", "contra"),
    ]

    # Keep only conditions that actually exist in the projections dict
    cond_order_8cond = [cond for cond in cond_order_8cond if cond in projections_8cond]

    print("\n8-condition averaged trajectories:")
    for cond in cond_order_8cond:
        print(cond, projections_8cond[cond].shape)

    # ------------------------------------------------------------
    # Plot 8 projected conditions on all 15 condition-dependent axes
    # ------------------------------------------------------------
    condition_dependent_axes = [
        # Hand / cue-space axes
        "hand",
        "space_x_cue",
        "space_y_cue",
        # Planning-period action-specific spatial axes
        "saccade_space_x_plan",
        "saccade_space_y_plan",
        "ipsi_hand_space_x_plan",
        "ipsi_hand_space_y_plan",
        "contra_hand_space_x_plan",
        "contra_hand_space_y_plan",
        # Movement-period action-specific spatial axes
        "saccade_space_x_mov",
        "saccade_space_y_mov",
        "ipsi_hand_space_x_mov",
        "ipsi_hand_space_y_mov",
        "contra_hand_space_x_mov",
        "contra_hand_space_y_mov",
    ]

    out_dir_8cond = plots_dir / "8_condition_axis_timecourses"
    out_dir_8cond.mkdir(parents=True, exist_ok=True)

    for axis in condition_dependent_axes:
        if axis not in axis_names:
            print(f"Skipping {axis}: not found in axis_names")
            continue
        plot_tdr_axis_timecourse(
            projections_8cond,
            analysis_time,
            axis_names,
            axis=axis,
            cond_order=cond_order_8cond,
            out_path=out_dir_8cond / f"time_{axis}.png",
            downsample=5,
            event_times=event_times,
            event_labels=event_labels,
            event_linestyles=event_linestyles,
            event_colors=event_colors,
            event_linewidths=event_linewidths,
            event_alphas=event_alphas,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate preprocessing plots.",
    )
    parser.add_argument(
        "--use_pca_denoising",
        action="store_true",
        default=False,
        help="PCA Denoising.",
    )
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=Path("plots/epoched_tdr_cue"),
        help="Directory where TDR output plots and CSV files are saved.",
    )
    args = parser.parse_args()
    main(
        use_pca_denoising=args.use_pca_denoising,
        plot=args.plot,
        plots_dir=args.plots_dir,
    )
