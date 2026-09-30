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
import scripts.config as cfg


def main(
    data_dir="data/new_data/flaffus",
    plots_dir=Path("plots/TDR_CUE_VARIANCE"),
):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    data_dir = Path(data_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_trials(path=data_dir, filename="processed_trials.pkl")

    # ------------------------------------------------------------
    # Mean Centering
    # ------------------------------------------------------------
    df = mean_center_rates(
        df,
        rate_col="sdf_rate",
    )

    # ------------------------------------------------------------
    # Unit Filtering
    # ------------------------------------------------------------
    unit_info = pd.read_excel(
        Path("data/unit_info.xlsx"),
        usecols=[0],
        dtype=str,
        header=None,
    )
    allowed_unit_ids = set(unit_info.iloc[:, 0].dropna().str.strip())
    df = df[df["unit_ID"].astype(str).str.strip().isin(allowed_unit_ids)].reset_index(
        drop=True
    )

    # ------------------------------------------------------------
    # Get time of onset of CUE, MOV and GO
    # ------------------------------------------------------------
    cue_state = 6
    mov_state = 68
    mov_end_state = 69
    go_state = 4
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
            event_time_col="t_mov",
            t_start=-0.5,
            t_end=0.5,
            bin_size=bin_size,
        ),
        axis=1,
    )
    df["analysis_time"] = cue_sdf.apply(lambda x: x[0])
    df["analysis_rate"] = cue_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Align other event times to cue
    # ------------------------------------------------------------
    df["t_mov_end"] = df["t_mov_end"].to_numpy(dtype=float) - df["t_mov"].to_numpy(
        dtype=float
    )
    df["t_go"] = df["t_go"].to_numpy(dtype=float) - df["t_mov"].to_numpy(dtype=float)
    df["t_mov"] = 0.0

    # ------------------------------------------------------------
    # Remove rows with NaNs in cue or movement SDFs
    # ------------------------------------------------------------
    df = df[
        df["analysis_rate"].apply(is_valid_array)
        & df["analysis_time"].apply(is_valid_array)
    ].reset_index(drop=True)
    analysis_time = np.asarray(df["analysis_time"].iloc[0], dtype=float)

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

    def time_window(time, start_time, end_time):
        time = np.asarray(time, dtype=float)
        if not np.all(np.isfinite([start_time, end_time])):
            return np.zeros_like(time, dtype=float)
        return ((time >= start_time) & (time < end_time)).astype(float)

    # Movement CI
    saccade_ci_window = (0.0, 0.30)
    reach_ci_window = (0.0, 0.30)

    def make_effector_movement_ci(row, effector, window):
        time = np.asarray(row["analysis_time"], dtype=float)

        # The regressor is zero for the other effector
        if row["effector"] != effector:
            return np.zeros_like(time, dtype=float)

        start_offset, end_offset = window

        return time_window(
            time,
            start_time=row["t_mov"] + start_offset,
            end_time=row["t_mov"] + end_offset,
        )

    df["sacCI"] = df.apply(
        lambda row: make_effector_movement_ci(
            row,
            effector="saccade",
            window=saccade_ci_window,
        ),
        axis=1,
    )

    df["reaCI"] = df.apply(
        lambda row: make_effector_movement_ci(
            row,
            effector="reach",
            window=reach_ci_window,
        ),
        axis=1,
    )

    # ------------------------------------------------------------
    # Hand Regressor
    # ------------------------------------------------------------
    df["hand"] = df["reach_hand"].map({"contra": 1.0, "ipsi": -1.0})
    df["hand_mask"] = [
        (np.asarray(t, dtype=float) < float(t_mov)).astype(float)
        for t, t_mov in zip(
            df["analysis_time"],
            df["t_mov"],
        )
    ]

    df = mask_regressors(
        df,
        regressors=("hand",),
        mask_col="hand_mask",
        suffix="",
    )

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

    # ------------------------------------------------------------
    # Effector regressors
    # ------------------------------------------------------------
    df = add_effector_regressors(df)

    df = add_effector_masks(df)

    effector_regs = ("saccade", "ipsi_hand", "contra_hand")
    space_regs = ("space_x", "space_y")
    regs = []

    for effector in effector_regs:
        for space in space_regs:
            col = f"{effector}_{space}"
            df[col] = df[effector] * df[space]
            regs.append(col)

    # Movement effector regressors
    df = mask_regressors(
        df,
        regressors=regs,
        mask_col="eff_mov_mask",
        suffix="_mov",
    )

    drop_cols = [
        "unit_ID",
        "trial_index",
        "t_mov",
        "t_go",
        "t_mov_end",
        "analysis_rate",
        "analysis_time",
        # CI regressors
        "sacCI",
        "reaCI",
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
        "sacCI",
        "reaCI",
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

    # Separate combined trials from axis-training trials
    combined_df = df.loc[df["effector"].eq("saccade_reach")].copy()
    df = df.loc[df["effector"].isin(["saccade", "reach"])].copy()

    df["effector"] = np.where(
        df["effector"].eq("saccade"),
        "saccade",
        df["reach_hand"].map(
            {
                "ipsi": "ipsi_hand",
                "contra": "contra_hand",
            }
        ),
    )

    combined_df["effector"] = combined_df["reach_hand"].map(
        {
            "ipsi": "saccade_reach_ipsi_hand",
            "contra": "saccade_reach_contra_hand",
        }
    )
    combined_df = combined_df.dropna(
        subset=["effector", "space_x", "space_y"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Keep units with enough trials in every effector-target condition
    # ------------------------------------------------------------
    min_trials_per_condition = 5
    trajectory_condition_cols = (
        "effector",
        "space_x",
        "space_y",
    )
    df, _, _ = filter_units_by_min_condition_trials(
        df,
        unit_col="unit_ID",
        condition_cols=trajectory_condition_cols,
        min_trials_per_condition=min_trials_per_condition,
        counts_out_path=(plots_dir / "rows_per_unit_per_effector_target.csv"),
    )

    # ------------------------------------------------------------
    # Repeated train/test TDR
    #
    # For every repeat:
    #   1. split real trials within each unit-condition cell
    #   2. fit TDR axes using training trials only
    #   3. compute condition averages from held-out test trials only
    #   4. project the test averages onto the training-derived axes
    # ------------------------------------------------------------
    n_repeats = 10
    test_frac = 0.3
    split_condition_cols = [
        "effector",
        "space_x",
        "space_y",
    ]

    # Store all the test projections and residuals here
    test_projection_repeats = []
    combined_projection_repeats = []
    residual_repeats = []
    condition_variance_repeats = []
    axes_raw_repeats = []

    pairwise_angle_repeats = []
    axis_change_repeats = []
    raw_angle_matrix_repeats = []
    raw_to_ortho_alignment_repeats = []

    for repeat_idx in range(n_repeats):
        print(f"\nTrain/test TDR repeat {repeat_idx + 1}/{n_repeats}")

        train_df, test_df = train_test_split_within_unit_condition(
            df,
            unit_col="unit_ID",
            condition_cols=tuple(split_condition_cols),
            test_frac=test_frac,
            min_train_trials=2,
            min_test_trials=1,
            random_state=repeat_idx,
        )

        # Equalize the number of training trials for each condition.
        # Oversampling occurs only after splitting, so no duplicated training
        # observation can leak into the held-out test partition.
        train_df = oversample_trials_within_unit_condition(
            train_df,
            unit_col="unit_ID",
            condition_cols=tuple(trajectory_condition_cols),
            random_state=10_000 + repeat_idx,
        )

        axes_raw_train, axes_ortho_train, units_train, regressors_train = fit_tdr_axes(
            train_df,
            regressors=regressors,
            unit_col="unit_ID",
            rate_col="analysis_rate",
            time_col="analysis_time",
        )

        pairwise_raw_angles, axis_changes, raw_angle_matrix, raw_to_ortho_alignment = (
            compute_tdr_axis_angles(
                axes_raw_train,
                axes_ortho_train,
                regressors,
            )
        )
        pairwise_raw_angles["repeat"] = repeat_idx
        axis_changes["repeat"] = repeat_idx

        pairwise_angle_repeats.append(pairwise_raw_angles)
        axis_change_repeats.append(axis_changes)
        raw_angle_matrix_repeats.append(raw_angle_matrix.to_numpy())
        raw_to_ortho_alignment_repeats.append(raw_to_ortho_alignment.to_numpy())

        test_condition_trajectories = condition_mean_population(
            test_df,
            units_train,
            condition_cols=tuple(trajectory_condition_cols),
            unit_col="unit_ID",
            rate_col="analysis_rate",
        )

        test_projections, _ = project_trajectories(
            test_condition_trajectories,
            axes_ortho_train,
            regressors_train,
        )

        combined_condition_trajectories = condition_mean_population(
            combined_df,
            units_train,
            condition_cols=("effector", "space_x", "space_y"),
            unit_col="unit_ID",
            rate_col="analysis_rate",
        )

        combined_projections, _ = project_trajectories(
            combined_condition_trajectories,
            axes_ortho_train,
            regressors_train,
        )

        observed, reconstructed, residuals = compute_tdr_subspace_residuals(
            test_condition_trajectories,
            test_projections,
            axes_ortho_train,
        )
        condition_variance = condition_mean_subspace_variance(
            observed,
            reconstructed,
            residuals,
            analysis_time,
        )
        condition_variance["repeat"] = repeat_idx

        test_projection_repeats.append(test_projections)
        combined_projection_repeats.append(combined_projections)
        residual_repeats.append(residuals)
        axes_raw_repeats.append(axes_raw_train)
        condition_variance_repeats.append(condition_variance)

    units = list(units_train)
    task_regressors = tuple(regressors_train)
    axis_names = list(task_regressors)
    cond_order = sorted(
        set.intersection(*[set(proj.keys()) for proj in test_projection_repeats])
    )

    # ------------------------------------------------------------
    # Angle Plots
    # ------------------------------------------------------------
    mean_raw_angle_matrix = pd.DataFrame(
        np.mean(
            np.stack(raw_angle_matrix_repeats, axis=0),
            axis=0,
        ),
        index=axis_names,
        columns=axis_names,
    )

    sd_raw_angle_matrix = pd.DataFrame(
        np.std(
            np.stack(raw_angle_matrix_repeats, axis=0),
            axis=0,
            ddof=1,
        ),
        index=axis_names,
        columns=axis_names,
    )

    mean_raw_to_ortho_alignment = pd.DataFrame(
        np.mean(
            np.stack(raw_to_ortho_alignment_repeats, axis=0),
            axis=0,
        ),
        index=axis_names,
        columns=axis_names,
    )

    sd_raw_to_ortho_alignment = pd.DataFrame(
        np.std(
            np.stack(raw_to_ortho_alignment_repeats, axis=0),
            axis=0,
            ddof=1,
        ),
        index=axis_names,
        columns=axis_names,
    )

    pairwise_angle_repeats = pd.concat(
        pairwise_angle_repeats,
        ignore_index=True,
    )

    axis_change_repeats = pd.concat(
        axis_change_repeats,
        ignore_index=True,
    )

    axis_change_summary = (
        axis_change_repeats.groupby("axis", sort=False)
        .agg(
            axis_rotation_deg=("axis_rotation_deg", "mean"),
            rotation_sd=("axis_rotation_deg", "std"),
            cosine_raw_vs_ortho=(
                "cosine_raw_vs_ortho",
                "mean",
            ),
            cosine_sd=(
                "cosine_raw_vs_ortho",
                "std",
            ),
            raw_norm=("raw_norm", "mean"),
            ortho_norm=("ortho_norm", "mean"),
        )
        .reset_index()
    )

    plot_tdr_axis_angle_diagnostics(
        mean_raw_angle_matrix,
        mean_raw_to_ortho_alignment,
        axis_change_summary,
        out_path=plots_dir / "tdr_axis_angle_diagnostics_mean.png",
    )

    # ------------------------------------------------------------
    # Trajectory Plotting
    # ------------------------------------------------------------
    trajectory_projections, trajectory_projection_sd = stack_projection_repeats(
        test_projection_repeats,
        cond_order,
        axis_names,
    )

    combined_cond_order = sorted(
        set.intersection(*[set(proj.keys()) for proj in combined_projection_repeats])
    )

    combined_trajectory_projections, combined_trajectory_projection_sd = (
        stack_projection_repeats(
            combined_projection_repeats,
            combined_cond_order,
            axis_names,
        )
    )
    if n_repeats == 1:
        trajectory_projection_sd = None
        combined_trajectory_projection_sd = None

    # ------------------------------------------------------------
    # Condition-mean subspace variance across repeated splits
    # ------------------------------------------------------------
    condition_variance_by_repeat = pd.concat(
        condition_variance_repeats,
        ignore_index=True,
    )
    variance_group_cols = ["effector", "space_x", "space_y", "time"]
    variance_metric_cols = [
        "observed_variance",
        "reconstructed_variance",
        "residual_variance",
        "observed_energy",
        "reconstructed_energy",
        "residual_energy",
        "reconstruction_r2",
    ]

    missing_variance_cols = set(
        variance_group_cols + variance_metric_cols + ["repeat"]
    ).difference(condition_variance_by_repeat.columns)
    if missing_variance_cols:
        raise ValueError(
            "condition_mean_subspace_variance returned an unexpected schema; "
            f"missing columns: {sorted(missing_variance_cols)}"
        )

    condition_variance_mean = condition_variance_by_repeat.groupby(
        variance_group_cols,
        as_index=False,
        observed=True,
    )[variance_metric_cols].mean()
    variance_repeat_counts = (
        condition_variance_by_repeat.groupby(
            variance_group_cols,
            as_index=False,
            observed=True,
        )["repeat"]
        .nunique()
        .rename(columns={"repeat": "n_repeats"})
    )
    condition_variance_mean = condition_variance_mean.merge(
        variance_repeat_counts,
        on=variance_group_cols,
        how="left",
        validate="one_to_one",
    )

    # Average the training-derived raw beta vectors across repeated splits
    axes_raw = np.mean(np.stack(axes_raw_repeats, axis=0), axis=0)

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

    plot_condition_mean_variance_grid(
        condition_variance_mean,
        out_path=plots_dir / "condition_mean_subspace_variance.png",
        event_times=event_times,
        event_linestyles=event_linestyles,
        downsample=3,
    )

    # ------------------------------------------------------------
    # Plot with 6 subplots
    # ------------------------------------------------------------

    downsample = 3
    row_axes = ("saccade_space_x_plan", "saccade_space_y_plan")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "saccade_planning_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("ipsi_hand_space_x_plan", "ipsi_hand_space_y_plan")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "ipsi_hand_planning_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("contra_hand_space_x_plan", "contra_hand_space_y_plan")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "contra_hand_planning_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("saccade_space_x_mov", "saccade_space_y_mov")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "saccade_movement_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("ipsi_hand_space_x_mov", "ipsi_hand_space_y_mov")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "ipsi_hand_movement_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("contra_hand_space_x_mov", "contra_hand_space_y_mov")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "contra_hand_movement_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("space_x_cue", "space_y_cue")
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "space_cue_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = (
        "cueCI",
        "planCI",
        "sacCI",
        "reaCI",
    )
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "CI_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    row_axes = ("hand",)
    plot_tdr_grid(
        trajectory_projections,
        row_axes,
        analysis_time,
        axis_names,
        projections_sem=trajectory_projection_sd,
        out_path=plots_dir / "hand_axes_by_effector.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=downsample,
    )

    combined_effectors = (
        "saccade_reach_contra_hand",
        "saccade_reach_ipsi_hand",
    )

    combined_effector_labels = {
        "saccade_reach_contra_hand": "Combined: contra-hand",
        "saccade_reach_ipsi_hand": "Combined: ipsi-hand",
    }

    combined_plot_specs = [
        (
            "combined_saccade_planning_axes.png",
            ("saccade_space_x_plan", "saccade_space_y_plan"),
        ),
        (
            "combined_ipsi_hand_planning_axes.png",
            ("ipsi_hand_space_x_plan", "ipsi_hand_space_y_plan"),
        ),
        (
            "combined_contra_hand_planning_axes.png",
            ("contra_hand_space_x_plan", "contra_hand_space_y_plan"),
        ),
        (
            "combined_saccade_movement_axes.png",
            ("saccade_space_x_mov", "saccade_space_y_mov"),
        ),
        (
            "combined_ipsi_hand_movement_axes.png",
            ("ipsi_hand_space_x_mov", "ipsi_hand_space_y_mov"),
        ),
        (
            "combined_contra_hand_movement_axes.png",
            ("contra_hand_space_x_mov", "contra_hand_space_y_mov"),
        ),
        (
            "combined_space_cue_axes.png",
            ("space_x_cue", "space_y_cue"),
        ),
        (
            "combined_CI_axes.png",
            ("cueCI", "planCI", "sacCI", "reaCI"),
        ),
        (
            "combined_hand_axis.png",
            ("hand",),
        ),
    ]

    for filename, row_axes in combined_plot_specs:
        plot_tdr_grid(
            combined_trajectory_projections,
            row_axes,
            analysis_time,
            axis_names,
            projections_sem=combined_trajectory_projection_sd,
            effectors=combined_effectors,
            effector_labels=combined_effector_labels,
            out_path=plots_dir / filename,
            event_times=event_times,
            event_labels=event_labels,
            event_linestyles=event_linestyles,
            event_colors=event_colors,
            event_linewidths=event_linewidths,
            event_alphas=event_alphas,
            downsample=downsample,
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
        "--plots_dir",
        type=Path,
        default=Path("plots/TDR_CUE_VARIANCE"),
        help="Directory where TDR output plots and CSV files are saved.",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        plots_dir=args.plots_dir,
    )
