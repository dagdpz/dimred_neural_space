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
    plots_dir=Path("plots/tgm_cue_aligned"),
):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    plots_dir = Path(plots_dir)

    tgm_space = pd.read_csv(plots_dir / "tgm_space_accuracy.csv", index_col=0)
    tgm_effector = pd.read_csv(plots_dir / "tgm_effector_accuracy.csv", index_col=0)
    tgm_hand = pd.read_csv(plots_dir / "tgm_hand_accuracy.csv", index_col=0)

    # ------------------------------------------------------------
    # Combined decoding plot
    # ------------------------------------------------------------
    plot_temporal_generalization_matrix(
        tgm_effector,
        out_path=plots_dir / "tgm_effector_accuracy.png",
        title="Temporal generalization: effector",
        chance=0.50,
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        colorbar_fontsize=16,
        legend_fontsize=14,
    )

    plot_temporal_generalization_matrix(
        tgm_space,
        out_path=plots_dir / "tgm_space_accuracy.png",
        title="Temporal generalization: space",
        chance=0.50,
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        colorbar_fontsize=16,
        legend_fontsize=14,
    )

    plot_temporal_generalization_matrix(
        tgm_hand,
        out_path=plots_dir / "tgm_reach_hand_accuracy.png",
        title="Temporal generalization: reach hand",
        chance=0.50,
        event_times=[0.0, 1.3],
        event_labels=["Cue", "GO"],
        title_fontsize=20,
        label_fontsize=18,
        tick_fontsize=15,
        colorbar_fontsize=16,
        legend_fontsize=14,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots_dir",
        type=Path,
        default=Path("plots/tgm_cue_aligned"),
        help="Directory where TDR output plots and CSV files are saved.",
    )
    args = parser.parse_args()
    main(
        plots_dir=args.plots_dir,
    )
