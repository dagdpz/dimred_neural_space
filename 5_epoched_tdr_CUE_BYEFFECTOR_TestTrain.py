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
    plot=False,
    plots_dir=Path("plots/TDR_CUE_NEW/train_test_split"),
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
        unit_cols=("session", "unit_ID"),
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
    df = df[
        df["analysis_rate"].apply(is_valid_array)
        & df["analysis_time"].apply(is_valid_array)
    ].reset_index(drop=True)
    analysis_time = np.asarray(df["analysis_time"].iloc[0], dtype=float)

    # ------------------------------------------------------------
    # Keep only units with enough trials in every 8-condition cell
    # effector x reach_hand x target_hemifield
    # ------------------------------------------------------------
    min_trials_per_condition = 5
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
    df = df.merge(
        good_units,
        on="unit_ID",
        how="inner",
    ).reset_index(drop=True)

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
    target_counts.to_csv(
        plots_dir / "target_position_counts.csv",
        index=False,
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

    # Planning effector regressors
    df = mask_regressors(
        df,
        regressors=regs,
        mask_col="eff_plan_mask",
        suffix="_plan",
    )

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

    # ------------------------------------------------------------
    # Keep units with enough trials in every effector-target condition
    # ------------------------------------------------------------
    min_trials_per_condition = 5
    trajectory_condition_cols = [
        "effector",
        "space_x",
        "space_y",
    ]
    valid_conditions = (
        df[trajectory_condition_cols].dropna().drop_duplicates().reset_index(drop=True)
    )
    required_counts = (
        df[["unit_ID"]].drop_duplicates().merge(valid_conditions, how="cross")
    )
    observed_counts = (
        df.groupby(
            ["unit_ID", *trajectory_condition_cols],
            observed=True,
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )
    trials_per_target_condition = required_counts.merge(
        observed_counts,
        on=["unit_ID", *trajectory_condition_cols],
        how="left",
    )
    trials_per_target_condition["n_rows"] = (
        trials_per_target_condition["n_rows"].fillna(0).astype(int)
    )
    trials_per_target_condition.to_csv(
        plots_dir / "rows_per_unit_per_effector_target.csv",
        index=False,
    )

    good_units = (
        trials_per_target_condition.groupby("unit_ID")["n_rows"]
        .min()
        .loc[lambda counts: counts >= min_trials_per_condition]
        .index
    )
    print(f"Units before effector-target filtering: " f"{df['unit_ID'].nunique()}")
    print(f"Units after effector-target filtering: " f"{len(good_units)}")
    df = df[df["unit_ID"].isin(good_units)].reset_index(drop=True)

    # ------------------------------------------------------------
    # Repeated train/test TDR
    #
    # For every repeat:
    #   1. split real trials within each unit-condition cell
    #   2. fit TDR axes using training trials only
    #   3. compute condition averages from held-out test trials only
    #   4. project the test averages onto the training-derived axes
    # ------------------------------------------------------------
    n_repeats = 60
    test_frac = 0.3
    split_condition_cols = [
        "effector",
        "space_x",
        "space_y",
    ]

    # Store all the test projections here
    projection_repeats = []

    # Store all the raw axes here
    axes_raw_repeats = []
    task_regressors = None

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

        # Equalize the number of training trials contributed by every unit.
        # Oversampling occurs only after splitting, so no duplicated training
        # observation can leak into the held-out test partition.
        train_df = oversample_units_to_equal_trials(
            train_df,
            unit_col="unit_ID",
            target_n_trials=None,  # use the largest training-unit count
            random_state=10_000 + repeat_idx,
        )

        axes_raw_train, axes_ortho_train, units_train, regressors_train = fit_tdr_axes(
            train_df,
            regressors=regressors,
            rate_col="analysis_rate",
            time_col="analysis_time",
            unit_cols=("unit_ID",),
        )

        test_condition_trajectories = condition_mean_population(
            test_df,
            units_train,
            condition_cols=tuple(trajectory_condition_cols),
            unit_cols=("unit_ID",),
            rate_col="analysis_rate",
        )

        test_projections, _ = project_trajectories(
            test_condition_trajectories,
            axes_ortho_train,
            regressors_train,
        )

        projection_repeats.append(test_projections)
        axes_raw_repeats.append(axes_raw_train)

    units = list(units_train)
    task_regressors = tuple(regressors_train)
    axis_names = list(task_regressors)
    cond_order = sorted(
        set.intersection(*[set(proj.keys()) for proj in projection_repeats])
    )

    trajectory_projections, trajectory_projection_sd = stack_projection_repeats(
        projection_repeats,
        cond_order,
        axis_names,
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
        "goCI",
        "movCI",
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
        "--plots_dir",
        type=Path,
        default=Path("plots/TDR_CUE/train_test_split"),
        help="Directory where TDR output plots and CSV files are saved.",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        plot=args.plot,
        plots_dir=args.plots_dir,
    )
