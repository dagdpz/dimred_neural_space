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
    plot=False,
    plots_dir=Path("plots/tdr_int_go"),
):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    trials_per_condition = count_rows_per_unit_condition(df)
    trials_per_condition.to_csv(
        plots_dir / "rows_per_unit_per_condition.csv",
        index=False,
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
        "t_cue",
        "t_mov",
        "t_go",
        "analysis_rate",
        "analysis_time",
    ]
    df = df[columns_to_keep].copy()

    # ------------------------------------------------------------
    # Interaction labels for temporal generalization decoding
    # ------------------------------------------------------------
    df["E_code"] = df["effector"].map({"reach": 1, "saccade": -1})
    df["H_code"] = df["reach_hand"].map({"contra": 1, "ipsi": -1})
    df["T_code"] = df["target_hemifield"].map({"contra": 1, "ipsi": -1})

    # Binary interaction contrasts
    df["EH_code"] = df["E_code"] * df["H_code"]
    df["ET_code"] = df["E_code"] * df["T_code"]

    df["effector_x_hand"] = df["EH_code"].map(
        {
            1: "same_EH",
            -1: "opposite_EH",
        }
    )

    df["effector_x_space"] = df["ET_code"].map(
        {
            1: "same_ET",
            -1: "opposite_ET",
        }
    )

    # ------------------------------------------------------------
    # Time-resolved decoding: effector
    # ------------------------------------------------------------
    tgm_effector, info_tgm_effector = temporal_generalization_logistic_decoding(
        df,
        label_col="effector",
        unit_col="unit_ID",
        rate_col="analysis_rate",
        time_col="analysis_time",
        n_pseudotrials_per_class=200,
        n_splits=5,
        random_state=0,
        time_step=10,  # use every 10th time bin; change to 1 for full matrix
    )
    tgm_effector.to_csv(
        plots_dir / "tgm_effector_accuracy.csv",
    )

    # ------------------------------------------------------------
    # Time-resolved decoding: target hemifield / space
    # ------------------------------------------------------------
    tgm_space, info_tgm_space = temporal_generalization_logistic_decoding(
        df,
        label_col="target_hemifield",
        unit_col="unit_ID",
        rate_col="analysis_rate",
        time_col="analysis_time",
        n_pseudotrials_per_class=200,
        n_splits=5,
        random_state=0,
        time_step=10,  # use every 10th time bin; change to 1 for full matrix
    )
    tgm_space.to_csv(
        plots_dir / "tgm_space_accuracy.csv",
    )

    # ------------------------------------------------------------
    # Time-resolved decoding: reach hand
    # ------------------------------------------------------------

    tgm_hand, info_tgm_hand = temporal_generalization_logistic_decoding(
        df,
        label_col="reach_hand",
        unit_col="unit_ID",
        rate_col="analysis_rate",
        time_col="analysis_time",
        n_pseudotrials_per_class=200,
        n_splits=5,
        random_state=0,
        time_step=10,  # use every 10th time bin; change to 1 for full matrix
    )

    tgm_hand.to_csv(
        plots_dir / "tgm_hand_accuracy.csv",
    )

    # ------------------------------------------------------------
    # Temporal generalization decoding: effector x hand interaction
    # ------------------------------------------------------------
    tgm_effector_x_hand, info_tgm_effector_x_hand = (
        temporal_generalization_logistic_decoding(
            df,
            label_col="effector_x_hand",
            unit_col="unit_ID",
            rate_col="analysis_rate",
            time_col="analysis_time",
            n_pseudotrials_per_class=200,
            n_splits=5,
            random_state=0,
            time_step=10,
        )
    )

    tgm_effector_x_hand.to_csv(
        plots_dir / "tgm_effector_x_hand_accuracy.csv",
    )

    # ------------------------------------------------------------
    # Temporal generalization decoding: effector x space interaction
    # ------------------------------------------------------------
    tgm_effector_x_space, info_tgm_effector_x_space = (
        temporal_generalization_logistic_decoding(
            df,
            label_col="effector_x_space",
            unit_col="unit_ID",
            rate_col="analysis_rate",
            time_col="analysis_time",
            n_pseudotrials_per_class=200,
            n_splits=5,
            random_state=0,
            time_step=10,
        )
    )

    tgm_effector_x_space.to_csv(
        plots_dir / "tgm_effector_x_space_accuracy.csv",
    )

    # ------------------------------------------------------------
    # Combined decoding plot
    # ------------------------------------------------------------
    plot_temporal_generalization_matrix(
        tgm_effector,
        out_path=plots_dir / "tgm_effector_accuracy.png",
        title="Temporal generalization: effector",
        chance=info_tgm_effector["chance"],
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
    )

    plot_temporal_generalization_matrix(
        tgm_space,
        out_path=plots_dir / "tgm_space_accuracy.png",
        title="Temporal generalization: space",
        chance=info_tgm_space["chance"],
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
    )

    plot_temporal_generalization_matrix(
        tgm_hand,
        out_path=plots_dir / "tgm_reach_hand_accuracy.png",
        title="Temporal generalization: reach hand",
        chance=info_tgm_hand["chance"],
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
    )

    plot_temporal_generalization_matrix(
        tgm_effector_x_hand,
        out_path=plots_dir / "tgm_effector_x_hand_accuracy.png",
        title="Temporal generalization: effector x hand",
        chance=info_tgm_effector_x_hand["chance"],
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
    )

    plot_temporal_generalization_matrix(
        tgm_effector_x_space,
        out_path=plots_dir / "tgm_effector_x_space_accuracy.png",
        title="Temporal generalization: effector x space",
        chance=info_tgm_effector_x_space["chance"],
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
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
        "--plots_dir",
        type=Path,
        default=Path("plots/tgm_cue_aligned"),
        help="Directory where TDR output plots and CSV files are saved.",
    )
    args = parser.parse_args()
    main(
        plot=args.plot,
        plots_dir=args.plots_dir,
    )
