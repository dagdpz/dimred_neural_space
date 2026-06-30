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
    data_dir="data/new_data/flaffus",
    use_pca_denoising=False,
    plot=False,
    plots_dir=Path("plots/epoched_tdr_cue"),
):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    data_dir = Path(data_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_trials(path=data_dir)

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
    # Summary after all filtering
    # ------------------------------------------------------------
    n_units = df[["unit_ID"]].drop_duplicates().shape[0]
    n_trials_total = len(df)

    trials_per_unit = df.groupby("unit_ID").size().rename("n_trials").reset_index()

    mean_trials_per_unit = trials_per_unit["n_trials"].mean()
    sd_trials_per_unit = trials_per_unit["n_trials"].std(ddof=1)

    print("\nAfter analysis-rate filtering:")
    print(f"  Units: {n_units}")
    print(f"  Rows / unit-trials: {n_trials_total}")
    print("\nTrials per unit:")
    print(f"  Mean: {mean_trials_per_unit:.2f}")
    print(f"  SD:   {sd_trials_per_unit:.2f}")
    print(f"  Min:  {trials_per_unit['n_trials'].min()}")
    print(f"  Max:  {trials_per_unit['n_trials'].max()}")

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
    df = add_target_y_position_label(df)

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

    condition_dependent_axes = [
        "cueCI",
        "planCI",
        "goCI",
        "movCI",
        # Hand / cue-space axes
        "hand",
        "space_x_cue",
        # Planning-period action-specific spatial axes
        "saccade_space_x_plan",
        "ipsi_hand_space_x_plan",
        "contra_hand_space_x_plan",
        # Movement-period action-specific spatial axes
        "saccade_space_x_mov",
        "ipsi_hand_space_x_mov",
        "contra_hand_space_x_mov",
        "space_y_cue",
        "saccade_space_y_plan",
        "ipsi_hand_space_y_plan",
        "contra_hand_space_y_plan",
        "saccade_space_y_mov",
        "ipsi_hand_space_y_mov",
        "contra_hand_space_y_mov",
    ]

    # ------------------------------------------------------------
    # Train/test split control with repeated splits
    # ------------------------------------------------------------
    n_repeats = 30
    test_frac = 0.3

    projection_repeats = []
    axis_names = None
    task_regressors_ref = None

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

    for repeat_idx in range(n_repeats):
        print(f"\nTrain/test repeat {repeat_idx + 1}/{n_repeats}")

        train_df, test_df = train_test_split_within_unit_condition(
            df,
            unit_col="unit_ID",
            condition_cols=("effector", "reach_hand", "target_hemifield"),
            test_frac=test_frac,
            min_train_trials=2,
            min_test_trials=2,
            random_state=repeat_idx,
        )

        axes_raw_train, axes_ortho_train, units_train, task_regressors_train = (
            fit_tdr_axes(
                train_df,
                regressors=regressors,
                rate_col="analysis_rate",
                time_col="analysis_time",
                unit_cols=("unit_ID",),
            )
        )

        if axis_names is None:
            axis_names = list(task_regressors_train)
            task_regressors_ref = task_regressors_train
        else:
            if list(task_regressors_train) != axis_names:
                raise ValueError("Axis order changed across train/test repeats.")

        condition_pops_8cond_test = condition_mean_population(
            test_df,
            units_train,
            condition_cols=("effector", "reach_hand", "target_hemifield"),
            unit_cols=("unit_ID",),
            rate_col="analysis_rate",
        )

        projections_8cond_test, _ = project_trajectories(
            condition_pops_8cond_test,
            axes_ortho_train,
            task_regressors_train,
        )

        projection_repeats.append(projections_8cond_test)

    # Keep only conditions present in all / most repeats
    cond_order_8cond = [
        cond
        for cond in cond_order_8cond
        if any(cond in rep for rep in projection_repeats)
    ]

    # ------------------------------------------------------------
    # Mean ± SEM across repeated train/test splits
    # ------------------------------------------------------------
    projections_8cond_mean, projections_8cond_sem = stack_projection_repeats(
        projection_repeats,
        cond_order_8cond,
        axis_names,
    )

    out_dir_8cond_test = (
        plots_dir / "train_test_split" / "test_8_condition_axis_timecourses_sem"
    )
    out_dir_8cond_test.mkdir(parents=True, exist_ok=True)

    for axis in condition_dependent_axes:
        if axis not in axis_names:
            print(f"Skipping {axis}: not found in axis_names")
            continue

        plot_tdr_axis_timecourse_with_sem(
            projections_8cond_mean,
            projections_8cond_sem,
            analysis_time,
            axis_names,
            axis=axis,
            cond_order=cond_order_8cond,
            out_path=out_dir_8cond_test / f"test_time_{axis}_mean_sem.png",
            downsample=5,
            event_times=event_times,
            event_labels=event_labels,
            event_linestyles=event_linestyles,
            event_colors=event_colors,
            event_linewidths=event_linewidths,
            event_alphas=event_alphas,
        )

    out_dir_plan_mov_2d = (
        plots_dir / "train_test_split" / "test_8_condition_plan_vs_mov_2d"
    )
    out_dir_plan_mov_2d.mkdir(parents=True, exist_ok=True)

    plan_mov_axis_pairs = [
        (
            "saccade_space_x_plan",
            "saccade_space_x_mov",
        ),
        (
            "ipsi_hand_space_x_plan",
            "ipsi_hand_space_x_mov",
        ),
        (
            "contra_hand_space_x_plan",
            "contra_hand_space_x_mov",
        ),
        (
            "saccade_space_y_plan",
            "saccade_space_y_mov",
        ),
        (
            "ipsi_hand_space_y_plan",
            "ipsi_hand_space_y_mov",
        ),
        (
            "contra_hand_space_y_plan",
            "contra_hand_space_y_mov",
        ),
    ]

    for plan_axis, mov_axis in plan_mov_axis_pairs:
        if plan_axis not in axis_names or mov_axis not in axis_names:
            print(f"Skipping {plan_axis} vs {mov_axis}: axis not found")
            continue

        plot_tdr_plan_vs_movement_2d(
            projections_8cond_mean,
            analysis_time,
            axis_names,
            plan_axis=plan_axis,
            mov_axis=mov_axis,
            cond_order=cond_order_8cond,
            out_path=out_dir_plan_mov_2d / f"{plan_axis}_vs_{mov_axis}.png",
            downsample=10,
            event_times=event_times,
            event_labels=event_labels,
            title=f"{plan_axis} vs {mov_axis}",
        )

    # ------------------------------------------------------------
    # Train/test split control for vertical target position
    # effector x reach_hand x target_y_position
    # ------------------------------------------------------------
    df_y = df.dropna(subset=["target_y_position"]).copy()

    condition_cols_y = ("effector", "reach_hand", "target_y_position")

    condition_levels_y = {
        "effector": ["reach", "saccade"],
        "reach_hand": ["ipsi", "contra"],
        "target_y_position": ["up", "down"],
    }

    # ------------------------------------------------------------
    # keep only units with enough trials in every vertical condition
    # ------------------------------------------------------------
    min_trials_per_y_condition = 5

    trials_per_ycond = count_rows_per_unit_condition(
        df_y,
        unit_cols=("unit_ID",),
        condition_cols=condition_cols_y,
        condition_levels=condition_levels_y,
    )

    trials_per_ycond.to_csv(
        plots_dir / "rows_per_unit_per_y_condition.csv",
        index=False,
    )

    good_units_y = trials_per_ycond.groupby("unit_ID")["n_rows"].min().reset_index()

    good_units_y = good_units_y[good_units_y["n_rows"] >= min_trials_per_y_condition][
        ["unit_ID"]
    ]

    n_units_y_before = df_y["unit_ID"].nunique()

    df_y = df_y.merge(
        good_units_y,
        on="unit_ID",
        how="inner",
    ).reset_index(drop=True)

    n_units_y_after = df_y["unit_ID"].nunique()

    print("\nVertical up/down condition filtering:")
    print(f"  Units before y-condition filtering: {n_units_y_before}")
    print(
        f"  Units after requiring >= {min_trials_per_y_condition} trial(s) "
        "in every effector x hand x up/down condition: "
        f"{n_units_y_after}"
    )
    print(f"  Rows after y-condition filtering: {len(df_y)}")

    n_repeats = 30
    test_frac = 0.3

    projection_repeats_y = []
    axis_names_y = None

    cond_order_8cond_y = [
        ("reach", "ipsi", "up"),
        ("reach", "ipsi", "down"),
        ("reach", "contra", "up"),
        ("reach", "contra", "down"),
        ("saccade", "ipsi", "up"),
        ("saccade", "ipsi", "down"),
        ("saccade", "contra", "up"),
        ("saccade", "contra", "down"),
    ]

    for repeat_idx in range(n_repeats):
        print(f"\nVertical train/test repeat {repeat_idx + 1}/{n_repeats}")

        train_df_y, test_df_y = train_test_split_within_unit_condition(
            df_y,
            unit_col="unit_ID",
            condition_cols=condition_cols_y,
            test_frac=test_frac,
            min_train_trials=3,
            min_test_trials=2,
            random_state=repeat_idx,
        )

        # Fit TDR axes only on training trials
        axes_raw_train_y, axes_ortho_train_y, units_train_y, task_regressors_train_y = (
            fit_tdr_axes(
                train_df_y,
                regressors=regressors,
                rate_col="analysis_rate",
                time_col="analysis_time",
                unit_cols=("unit_ID",),
            )
        )

        if axis_names_y is None:
            axis_names_y = list(task_regressors_train_y)
        else:
            if list(task_regressors_train_y) != axis_names_y:
                raise ValueError(
                    "Axis order changed across vertical train/test repeats."
                )

        # Build held-out test condition averages:
        # effector x reach_hand x up/down
        condition_pops_y_test = condition_mean_population(
            test_df_y,
            units_train_y,
            condition_cols=condition_cols_y,
            unit_cols=("unit_ID",),
            rate_col="analysis_rate",
        )

        # Project held-out test trajectories onto train-defined axes
        projections_y_test, _ = project_trajectories(
            condition_pops_y_test,
            axes_ortho_train_y,
            task_regressors_train_y,
        )

        projection_repeats_y.append(projections_y_test)

    # Keep only conditions present in at least one repeat
    cond_order_8cond_y = [
        cond
        for cond in cond_order_8cond_y
        if any(cond in rep for rep in projection_repeats_y)
    ]

    # ------------------------------------------------------------
    # Mean ± SEM across repeated train/test splits
    # ------------------------------------------------------------
    projections_y_mean, projections_y_sem = stack_projection_repeats(
        projection_repeats_y,
        cond_order_8cond_y,
        axis_names_y,
    )

    out_dir_y_test = (
        plots_dir / "train_test_split" / "test_8_condition_axis_timecourses_y_mean_sem"
    )
    out_dir_y_test.mkdir(parents=True, exist_ok=True)

    # Axes especially relevant for vertical position
    condition_dependent_axes_y = [
        "space_y_cue",
        "saccade_space_y_plan",
        "ipsi_hand_space_y_plan",
        "contra_hand_space_y_plan",
        "saccade_space_y_mov",
        "ipsi_hand_space_y_mov",
        "contra_hand_space_y_mov",
        # optional context axes
        "cueCI",
        "planCI",
        "goCI",
        "movCI",
        "hand",
    ]

    for axis in condition_dependent_axes_y:
        if axis not in axis_names_y:
            print(f"Skipping {axis}: not found in axis_names_y")
            continue

        plot_tdr_axis_timecourse_with_sem(
            projections_y_mean,
            projections_y_sem,
            analysis_time,
            axis_names_y,
            axis=axis,
            cond_order=cond_order_8cond_y,
            out_path=out_dir_y_test / f"test_time_{axis}_up_down_mean_sem.png",
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
        "--data_dir",
        type=str,
        default="data/new_data/flaffus",
        help="Folder containing new-format population_*.mat and trials_*.mat files.",
    )
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
        data_dir=args.data_dir,
        use_pca_denoising=args.use_pca_denoising,
        plot=args.plot,
        plots_dir=args.plots_dir,
    )
