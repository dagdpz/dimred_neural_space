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

    print("\nUnique target positions:")
    print(target_counts)
    print(f"\nNumber of unique target positions: {len(target_counts)}")

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
        out_path=plots_dir / "target_position_x_cue.png",
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
        out_path=plots_dir / "target_position_y_cue.png",
        downsample=5,
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    # ------------------------------------------------------------
    # Shuffled TDR control: same plots with shuffled task regressors
    # ------------------------------------------------------------

    shuffle_dir = plots_dir / "shuffled_tdr"
    shuffle_dir.mkdir(parents=True, exist_ok=True)

    ci_regressors = {
        "cueCI",
        "planCI",
        "goCI",
        "movCI",
    }

    regressors_to_shuffle = [reg for reg in regressors if reg not in ci_regressors]

    df_shuf = shuffle_regressor_columns_within_unit(
        df,
        regressors_to_shuffle=regressors_to_shuffle,
        unit_cols=("unit_ID",),
        random_state=0,
    )

    # Fit TDR axes using shuffled regressors
    axes_raw_shuf, axes_ortho_shuf, units_shuf, task_regressors_shuf = fit_tdr_axes(
        df_shuf,
        regressors=regressors,
        rate_col="analysis_rate",
        time_col="analysis_time",
        unit_cols=("unit_ID",),
    )

    axis_names_shuf = list(task_regressors_shuf)

    # Important:
    # Build condition averages using the real df, not df_shuf.
    # This keeps the real condition labels for plotting.
    condition_pop_shuf = condition_mean_population(
        df,
        units_shuf,
        condition_cols=("effector", "reach_hand", "target_hemifield"),
        unit_cols=("unit_ID",),
        rate_col="analysis_rate",
    )

    projections_shuf, axis_names_shuf = project_trajectories(
        condition_pop_shuf,
        axes_ortho_shuf,
        task_regressors_shuf,
    )

    # ------------------------------------------------------------
    # Plot the same axis time courses, but using shuffled TDR axes
    # ------------------------------------------------------------
    plot_all_tdr_axes_timecourses_separate(
        projections_shuf,
        analysis_time,
        axis_names_shuf,
        axes_to_plot=[
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
        ],
        out_dir=shuffle_dir / "axis_timecourses",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        lw=2.0,
        downsample=2,
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
