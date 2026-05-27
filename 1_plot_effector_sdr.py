from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from dPCA import dPCA

from scripts.preprocess import *
from scripts.plotting import *
from scripts.utils import *


def main(seed=0):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run dPCA.
    """
    df = load_processed_trials()

    # --- Align spikes and event times to states
    cue_state = 6
    mov_state = 68
    cue_align = df.apply(
        lambda row: trial_alignment_to_state(row, cue_state),
        axis=1,
    )
    cue_align = cue_align.rename(
        columns={
            "t_state": "t_cue",
            "arrival_times_rel": "arrival_times_cue",
        }
    )
    mov_align = df.apply(
        lambda row: trial_alignment_to_state(row, mov_state),
        axis=1,
    )
    mov_align = mov_align.rename(
        columns={
            "t_state": "t_mov",
            "arrival_times_rel": "arrival_times_mov",
        }
    )
    df = pd.concat([df, cue_align, mov_align], axis=1)
    print(df.columns)

    # --- Spike density functions (Gaussian-smoothed Hz), time axis aligned to cue (state 6) ---
    bin_size = 0.001  # 1 ms bins
    sigma = 0.02  # 50 ms Gaussian smoothing
    cue_sdf = df["arrival_times_cue"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=-0.5,
            t_end=0.8,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    mov_sdf = df["arrival_times_mov"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=-0.8,
            t_end=0.5,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    df["sdf_time_cue"] = cue_sdf.apply(lambda x: x[0])
    df["sdf_rate_cue"] = cue_sdf.apply(lambda x: x[1])
    df["sdf_time_mov"] = mov_sdf.apply(lambda x: x[0])
    df["sdf_rate_mov"] = mov_sdf.apply(lambda x: x[1])

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
