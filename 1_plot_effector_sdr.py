from pathlib import Path
import numpy as np
import argparse

from scripts.preprocess import *
from scripts.plotting import *
from scripts.utils import *


def main(
    plot=False,
    plots_dir=Path("plots/sdf"),
    cue_window=(-0.5, 0.8),
    mov_window=(-0.8, 0.5),
    bin_size=0.001,
):
    """
    Load preprocessed trials, align SDFs to cue and movement,
    and plot two-panel cue/movement SDFs.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_trials()

    # ------------------------------------------------------------
    # Event times
    # ------------------------------------------------------------
    cue_state = 6
    go_state = 4
    mov_state = 68

    df["t_cue"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], cue_state),
        axis=1,
    )
    df["t_go"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], go_state),
        axis=1,
    )
    df["t_mov"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], mov_state),
        axis=1,
    )

    # ------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------
    if plot:
        plot_event_diagnostics(
            df,
            plots_dir=plots_dir,
        )

    # ------------------------------------------------------------
    # Cue-aligned window
    # ------------------------------------------------------------
    cue_sdf = df.apply(
        lambda row: slice_sdf_to_event(
            row,
            event_time_col="t_cue",
            t_start=cue_window[0],
            t_end=cue_window[1],
            bin_size=bin_size,
        ),
        axis=1,
    )

    df["sdf_time_cue"] = cue_sdf.apply(lambda x: x[0])
    df["sdf_rate_cue"] = cue_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Movement-aligned window
    # ------------------------------------------------------------
    mov_sdf = df.apply(
        lambda row: slice_sdf_to_event(
            row,
            event_time_col="t_mov",
            t_start=mov_window[0],
            t_end=mov_window[1],
            bin_size=bin_size,
        ),
        axis=1,
    )

    df["sdf_time_mov"] = mov_sdf.apply(lambda x: x[0])
    df["sdf_rate_mov"] = mov_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Remove invalid rows
    # ------------------------------------------------------------
    before = len(df)

    df = df[
        df["sdf_rate_cue"].apply(is_valid_array)
        & df["sdf_time_cue"].apply(is_valid_array)
        & df["sdf_rate_mov"].apply(is_valid_array)
        & df["sdf_time_mov"].apply(is_valid_array)
    ].reset_index(drop=True)

    after = len(df)

    print(f"Removed {before - after} rows with invalid cue/movement SDFs")
    print(f"Remaining rows: {after}")

    # ------------------------------------------------------------
    # Plot two-panel cue/movement SDFs
    # ------------------------------------------------------------
    if plot:
        for reach_hand in ("ipsi", "contra"):
            for target_hemifield in ("ipsi", "contra"):
                sub = df[
                    (df["reach_hand"] == reach_hand)
                    & (df["target_hemifield"] == target_hemifield)
                ].dropna(subset=["effector"])

                plot_effector_sdf(
                    data=sub,
                    reach_hand=reach_hand,
                    target_hemifield=target_hemifield,
                    plots_dir=plots_dir / "units",
                    analysis_label=f"{target_hemifield}_{reach_hand}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate plots.",
    )

    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=Path("plots/sdf"),
        help="Directory where SDF plots are saved.",
    )

    parser.add_argument(
        "--cue_start",
        type=float,
        default=-0.5,
        help="Start of cue-aligned window in seconds.",
    )

    parser.add_argument(
        "--cue_end",
        type=float,
        default=0.8,
        help="End of cue-aligned window in seconds.",
    )

    parser.add_argument(
        "--mov_start",
        type=float,
        default=-0.8,
        help="Start of movement-aligned window in seconds.",
    )

    parser.add_argument(
        "--mov_end",
        type=float,
        default=0.5,
        help="End of movement-aligned window in seconds.",
    )

    parser.add_argument(
        "--bin_size",
        type=float,
        default=0.001,
        help="SDF bin size in seconds.",
    )

    args = parser.parse_args()

    main(
        plot=args.plot,
        plots_dir=args.plots_dir,
        cue_window=(args.cue_start, args.cue_end),
        mov_window=(args.mov_start, args.mov_end),
        bin_size=args.bin_size,
    )
