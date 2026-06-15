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
    no_interaction=False,
    use_pca_denoising=False,
    plot=False,
    plots_dir=Path("plots/tdr"),
):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_trials(normalized=True)

    # ------------------------------------------------------------
    # Get time of onset of MOV
    # ------------------------------------------------------------
    mov_state = 68
    go_state = 4
    df["t_mov"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], mov_state),
        axis=1,
    )
    df["t_go"] = df.apply(
        lambda r: get_state_onset(r["states_onset"], r["states"], go_state),
        axis=1,
    )

    # ------------------------------------------------------------
    # Aligned analysis window: -0.5 to 0.5 s after MVO
    # ------------------------------------------------------------
    bin_size = 0.001
    mov_sdf = df.apply(
        lambda row: slice_sdf_to_event(
            row,
            event_time_col="t_mov",
            t_start=-0.5,
            t_end=0.5,
            bin_size=bin_size,
        ),
        axis=1,
    )

    df["analysis_time"] = mov_sdf.apply(lambda x: x[0])
    df["analysis_rate"] = mov_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Remove rows with NaNs in movement SDFs
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

    trials_per_condition = count_rows_per_unit_condition(df)
    trials_per_condition.to_csv(
        plots_dir / "rows_per_unit_per_condition.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Plot 10 random SDFs, vertically shifted
    # ------------------------------------------------------------
    if plot:
        plot_random_shifted_sdfs(
            df,
            analysis_time,
            event_times=[0.0],
            event_labels=["MOV"],
            event_linestyles=["--"],
            event_colors=["k"],
            event_linewidths=[1.2],
            event_alphas=[0.7],
            out_path=plots_dir / "random_10_sdfs_shifted.png",
            xlabel="Time relative to MOV onset (s)",
        )

    # ------------------------------------------------------------
    # Keep only units with at least one trial in every condition
    # ------------------------------------------------------------
    complete_units = (
        trials_per_condition.groupby(["session", "unit_ID"])["n_rows"]
        .min()
        .reset_index()
    )

    complete_units = complete_units[complete_units["n_rows"] > 0][
        ["session", "unit_ID"]
    ]

    print(
        f"Units before complete-condition filtering: {df[['session', 'unit_ID']].drop_duplicates().shape[0]}"
    )
    print(f"Units after complete-condition filtering: {len(complete_units)}")

    df = df.merge(
        complete_units,
        on=["session", "unit_ID"],
        how="inner",
    )

    columns_to_keep = [
        "session",
        "unit_ID",
        "trial_index",
        "pulvinar_hemifield",
        "reach_hand",
        "effector",
        "target_hemifield",
        "t_mov",
        "t_go",
        "analysis_rate",
        "analysis_time",
    ]
    df = df[columns_to_keep].copy()

    # ------------------------------------------------------------
    # Regressors
    # ------------------------------------------------------------
    df = add_tdr_regressors(df, interaction=True)

    drop_cols = [
        "E",
        "T",
        "H",
        "t_mov",
        "t_go",
        "analysis_rate",
        "analysis_time",
    ]
    if not no_interaction:
        drop_cols += ["EH", "ET"]

    df = df.dropna(subset=drop_cols).copy()
    df = df[
        df["analysis_rate"].apply(is_valid_array)
        & df["analysis_time"].apply(is_valid_array)
    ].reset_index(drop=True)

    # ------------------------------------------------------------
    # Fit TDR axes
    # ------------------------------------------------------------
    main_regressors = ("E", "T", "H")
    interaction_regressors = ("EH", "ET")

    axes_raw, axes_ortho, units, task_regressors = fit_tdr_axes(
        df,
        main_regressors=main_regressors,
        interaction_regressors=interaction_regressors,
        no_interaction=no_interaction,
        cue_time_col=None,
    )

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
    axis_names = list(task_regressors)

    for name, cols in condition_factors.items():
        condition_pops[name] = condition_mean_population(
            df,
            units,
            condition_cols=cols,
            rate_col="analysis_rate",
        )

        if use_pca_denoising:
            pca_explained_variance = 0.8
            pca_max_components = None
            condition_pops[name], pca_info = pca_denoise_condition_population(
                condition_pops[name],
                explained_variance=pca_explained_variance,
                max_components=pca_max_components,
            )
            print(
                f"PCA denoising [{name}]: kept {pca_info['n_components']} PCs "
                f"explaining {100 * pca_info['selected_explained_variance']:.2f}% "
                "of condition-trajectory variance"
            )

        projections_by[name], _ = project_trajectories(
            condition_pops[name],
            axes_ortho,
            task_regressors,
        )

    # aliases for full 8-condition plots
    condition_pop = condition_pops["all_conditions"]
    projections = projections_by["all_conditions"]

    """ tdr_var_time = time_resolved_variance_explained_by_tdr_axes(
        condition_pop,
        axes_ortho,
        axis_names,
        analysis_time,
    )

    tdr_var_time.to_csv(
        plots_dir / "tdr_axis_variance_explained_time_resolved.csv",
        index=False,
    ) """

    # ------------------------------------------------------------
    # Simple population averages of model input
    # ------------------------------------------------------------

    # Common event markers for plots
    event_times = [0.0]
    event_labels = ["MOV"]
    event_colors = ["0.25"]
    event_opacities = [0.04]
    event_linestyles = ["--"]
    event_linewidths = [1.0]
    event_alphas = [0.7]

    plot_population_average_tdr_input(
        df,
        rate_col="analysis_rate",
        time_col="analysis_time",
        out_path=plots_dir / "population_average_tdr_input_8_conditions.png",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        xlabel="Time relative to cue onset (s)",
    )

    cond_stats, overall_stats = summarize_tdr_input_data(df)
    stats_path = plots_dir / "population_average_tdr_input_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    cond_stats.to_csv(stats_path, index=False)

    print("\nTDR input summary:")
    for k, v in overall_stats.items():
        print(f"  {k}: {v}")
    print("\nTrials per unit per condition:")
    print(cond_stats)

    # ------------------------------------------------------------
    # Euclidean distance between condition-averaged trajectories
    # ------------------------------------------------------------

    # Space: target ipsi vs target contra, averaged over effector and hand
    traj_space_ipsi = average_projected_trajectory_by_factor(
        projections,
        factor="target_hemifield",
        level="ipsi",
    )
    traj_space_contra = average_projected_trajectory_by_factor(
        projections,
        factor="target_hemifield",
        level="contra",
    )
    d_space = trajectory_euclidean_distance(
        traj_space_ipsi,
        traj_space_contra,
        axis_names=axis_names,
    )

    # Reach hand: hand ipsi vs hand contra, averaged over effector and target
    traj_hand_ipsi = average_projected_trajectory_by_factor(
        projections,
        factor="reach_hand",
        level="ipsi",
    )
    traj_hand_contra = average_projected_trajectory_by_factor(
        projections,
        factor="reach_hand",
        level="contra",
    )
    d_hand = trajectory_euclidean_distance(
        traj_hand_ipsi,
        traj_hand_contra,
        axis_names=axis_names,
    )

    # Effector: reach vs saccade, averaged over hand and target
    traj_reach = average_projected_trajectory_by_factor(
        projections,
        factor="effector",
        level="reach",
    )
    traj_saccade = average_projected_trajectory_by_factor(
        projections,
        factor="effector",
        level="saccade",
    )
    d_effector = trajectory_euclidean_distance(
        traj_reach,
        traj_saccade,
        axis_names=axis_names,
    )

    plot_tdr_trajectory_distance(
        [d_effector, d_hand, d_space],
        analysis_time,
        labels=["Effector", "Reach hand", "Space"],
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        title="Factor distances between condition-averaged TDR trajectories",
        ylabel="Euclidean distance in TDR space",
        xlabel="Stitched time (s)",
        out_path=plots_dir / "factor_distances_space_hand_effector.png",
        downsample=2,
    )

    event_colors = ["rgba(80,80,80,1)"]
    event_opacities = [0.04]

    plot_tdr_time_y_z_3d(
        projections,
        analysis_time,
        axis_names,
        y_axis="H",
        z_axis="T",
        y_label="Hand",
        z_label="Space",
        out_path=plots_dir / "tdr_time_H_S.html",
        downsample=2,
        event_times=event_times,
        event_labels=event_labels,
        event_colors=event_colors,
        event_opacities=event_opacities,
        x_label="Time (s)",
    )

    plot_tdr_time_y_z_3d(
        projections,
        analysis_time,
        axis_names,
        y_axis="H",
        z_axis="E",
        y_label="Hand",
        z_label="Effector",
        out_path=plots_dir / "tdr_time_H_E.html",
        downsample=2,
        event_times=event_times,
        event_labels=event_labels,
        event_colors=event_colors,
        event_opacities=event_opacities,
        x_label="Time (s)",
    )

    plot_tdr_time_y_z_3d(
        projections,
        analysis_time,
        axis_names,
        y_axis="H",
        z_axis="E",
        y_label="Space",
        z_label="Effector",
        out_path=plots_dir / "tdr_time_S_E.html",
        downsample=2,
        event_times=event_times,
        event_labels=event_labels,
        event_colors=event_colors,
        event_opacities=event_opacities,
        x_label="Time (s)",
    )

    if not no_interaction:
        # [time, space, effector x space]
        plot_tdr_time_y_z_3d(
            projections,
            analysis_time,
            axis_names,
            y_axis="T",
            z_axis="ET",
            y_label="Space",
            z_label="Effector x Space",
            out_path=plots_dir / "tdr_time_S_ES.html",
            downsample=2,
            event_times=event_times,
            event_labels=event_labels,
            event_colors=event_colors,
            event_opacities=event_opacities,
            x_label="Time (s)",
        )

        # [time, hand, effector x hand]
        plot_tdr_time_y_z_3d(
            projections,
            analysis_time,
            axis_names,
            y_axis="H",
            z_axis="EH",
            y_label="Hand",
            z_label="Effector x Hand",
            out_path=plots_dir / "tdr_time_H_EH.html",
            downsample=2,
            event_times=event_times,
            event_labels=event_labels,
            event_colors=event_colors,
            event_opacities=event_opacities,
            x_label="Time (s)",
        )

        # [time, effector x space, effector x hand]
        plot_tdr_time_y_z_3d(
            projections,
            analysis_time,
            axis_names,
            y_axis="ET",
            z_axis="EH",
            y_label="Effector x Space",
            z_label="Effector x Hand",
            out_path=plots_dir / "tdr_time_ES_EH.html",
            downsample=2,
            event_times=event_times,
            event_labels=event_labels,
            event_colors=event_colors,
            event_opacities=event_opacities,
            x_label="Time (s)",
        )

        # [time, effector, effector x hand]
        plot_tdr_time_y_z_3d(
            projections,
            analysis_time,
            axis_names,
            y_axis="E",
            z_axis="EH",
            y_label="Effector",
            z_label="Effector x Hand",
            out_path=plots_dir / "tdr_time_E_EH.html",
            downsample=2,
            event_times=event_times,
            event_labels=event_labels,
            event_colors=event_colors,
            event_opacities=event_opacities,
            x_label="Time (s)",
        )

        # [time, effector, effector x space]
        plot_tdr_time_y_z_3d(
            projections,
            analysis_time,
            axis_names,
            y_axis="E",
            z_axis="ET",
            y_label="Effector",
            z_label="Effector x Space",
            out_path=plots_dir / "tdr_time_E_ES.html",
            downsample=2,
            event_times=event_times,
            event_labels=event_labels,
            event_colors=event_colors,
            event_opacities=event_opacities,
            x_label="Time (s)",
        )

    # ------------------------------------------------------------
    # 2D projections of 8 condition-averaged trajectories
    # onto each regression/TDR axis over time
    # ------------------------------------------------------------
    event_times = [0.0]
    event_labels = ["MOV"]
    event_colors = ["k"]
    event_opacities = [0.04]
    event_linestyles = ["--"]
    event_linewidths = [1.0]
    event_alphas = [0.7]

    plot_all_tdr_axes_timecourses_separate(
        projections,
        analysis_time,
        axis_names,
        axes_to_plot=None,
        out_dir=plots_dir / "axis_timecourses",
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
        downsample=2,
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
        "--no_interaction",
        action="store_true",
        default=False,
        help="No interaction.",
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
        default=Path("plots/tdr_int_mov"),
        help="Directory where TDR output plots and CSV files are saved.",
    )
    args = parser.parse_args()
    main(
        no_interaction=args.no_interaction,
        use_pca_denoising=args.use_pca_denoising,
        plot=args.plot,
        plots_dir=args.plots_dir,
    )
