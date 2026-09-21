# script/config.py
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_DIR = Path("data/new_data/flaffus")
UNIT_INFO_PATH = Path("data/unit_info.xlsx")
PLOTS_DIR = Path("plots/TDR_CUE_NEW")

# ------------------------------------------------------------
# Task-state codes
# ------------------------------------------------------------

CUE_STATE = 6
GO_STATE = 4
MOV_STATE = 68
MOV_END_STATE = 69

# ------------------------------------------------------------
# Time settings
# ------------------------------------------------------------

BIN_SIZE = 0.001

CUE_WINDOW = (-0.5, 2.0)

# ------------------------------------------------------------
# Unit-selection settings
# ------------------------------------------------------------

MIN_TRIALS_8COND = 1
MIN_TRIALS_TARGET_CONDITION = 1

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

DOWNSAMPLE = 3

EVENT_LABELS = ("Cue", "GO")
EVENT_COLORS = ("black", "black")
EVENT_LINESTYLES = ("--", ":")
EVENT_LINEWIDTHS = (1.2, 1.2)
EVENT_ALPHAS = (0.7, 0.8)
EVENT_OPACITIES = (0.04, 0.04)
