from pathlib import Path
import numpy as np
import pandas as pd

from scripts.preprocess import *
from scripts.plotting import *
from scripts.utils import *


def main():
    """
    Load preprocessed trials, align spikes to cue, compute SDFs
    """
    df = load_processed_trials(normalized=True)

    # ------------------------------------------------------------
    # Event times
    # ------------------------------------------------------------
    cue_state = 6
    mov_state = 68

    df["t_cue"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], cue_state),
        axis=1,
    )

    df["t_mov"] = df.apply(
        lambda row: get_state_onset(row["states_onset"], row["states"], mov_state),
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
    # Plot
    # ------------------------------------------------------------
    for reach_hand in ("ipsi", "contra"):
        for target_hemifield in ("ipsi", "contra"):
            sub = df[
                (df["reach_hand"] == reach_hand)
                & (df["target_hemifield"] == target_hemifield)
            ].dropna(subset=["effector"])
            sdf_dir = Path("plots/sdf")
            plot_effector_sdf(
                data=sub,
                reach_hand=reach_hand,
                target_hemifield=target_hemifield,
                plots_dir=sdf_dir,
                analysis_label=f"{target_hemifield}_{reach_hand}",
            )


if __name__ == "__main__":
    main()
