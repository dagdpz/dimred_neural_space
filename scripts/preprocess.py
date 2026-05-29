from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts.plotting import *
from scripts.utils import *

PROCESSED_TRIALS_PATH = Path("data/processed_trials.pkl")


def load_population_spike_data(filepath):
    # --- Load MATLAB struct: one row per unit, nested trials per unit ---
    mat = loadmat(filepath)
    population = mat["population"]

    # --- Unpack selected top-level fields per unit (IDs, site, trial struct array) ---
    data = {}
    selected_keys = ["unit_ID", "target", "trial"]
    for field in population.dtype.names:
        if field not in selected_keys:
            continue
        values = []
        for item in population[field][0]:
            try:
                values.append(clean_mat_value(item[0]))
            except Exception:
                values.append(clean_mat_value(item))
        data[field] = values

    # --- One DataFrame row per (unit, trial); copy trial-level fields from nested structs ---
    rows = []
    for unit_idx, unit_id in enumerate(data["unit_ID"]):
        # Convert recorded pulvinar side to hemifield
        recorded_pulvinar = clean_mat_value(data["target"][unit_idx])
        pulvinar_hemifield = (
            "left" if pulvinar_to_side(recorded_pulvinar) == "right" else "right"
        )

        # extract trial data
        unit_trials = data["trial"][unit_idx]
        trial_fields = unit_trials.dtype.names
        for trial_idx in range(unit_trials.shape[0]):
            row = {
                "unit_ID": unit_id,
                "pulvinar_hemifield": pulvinar_hemifield,
                "trial_index": trial_idx,  # Use an ID to track trial for a neuron
            }
            for field in trial_fields:
                value = clean_mat_value(unit_trials[field][trial_idx])
                if np.iscomplexobj(value):
                    value = np.real(value)
                row[field] = value
            rows.append(row)
    return pd.DataFrame(rows)


def process_labels_and_filter(df):
    # --- Keep trials with correct choice and success; drop unused columns ---
    df = df[df["type"] == 4]
    df = df[df["choice"] == 0]
    df = df[df["success"] == 1]

    # Convert reach hand from numeric to string
    df["reach_hand"] = df["reach_hand"].apply(decode_reach_hand_label)
    df["reach_hand"] = df.apply(
        lambda row: ipsi_contra(row["reach_hand"], row["pulvinar_hemifield"]),
        axis=1,
    )

    # Dont use joint saccadereach condition for now
    df["effector"] = df["effector"].apply(decode_effector_label)
    df = df[df["effector"] != "saccade_reach"]

    # Target position
    df["target_side"] = df["tar_pos"].apply(tar_pos_to_side)
    df["target_hemifield"] = df.apply(
        lambda row: ipsi_contra(row["target_side"], row["pulvinar_hemifield"]),
        axis=1,
    )

    # Remove data with no spikes
    df = df[~df["arrival_times"].isna()]

    select_columns = [
        "session",
        "unit_ID",
        "trial_index",
        "pulvinar_hemifield",
        "reach_hand",
        "effector",
        "target_hemifield",
        "trial_onset_time",
        "run_onset_time",
        "states_onset",
        "states",
        "arrival_times",
    ]
    return df[select_columns]


def build_processed_trials(data_dir=Path("data"), normalize=True):
    population_files = sorted(data_dir.glob("*population*.mat"))
    if not population_files:
        raise FileNotFoundError(f"No *population*.mat files in {data_dir}")

    df = pd.DataFrame()
    for filepath in population_files:
        part = load_population_spike_data(filepath)
        part["session"] = filepath.stem
        df = pd.concat([df, part], ignore_index=True)
    df = process_labels_and_filter(df)
    if normalize:
        df = normalize_rates(df)
    return df


def normalize_rates(
    df,
    *,
    unit_cols=("session", "unit_ID"),
):
    """
    Normalize rates across units and trials
    """
    units = (
        df[list(unit_cols)]
        .drop_duplicates()
        .sort_values(list(unit_cols))
        .itertuples(
            index=False, name=None
        )  # create tuples from each row from dataframe
    )
    units = list(units)
    for unit in units:
        unit_df = df.copy()

        # Select only the rows for the current unit
        for col, val in zip(unit_cols, unit):
            unit_df = unit_df[unit_df[col] == val]

        unit_df[["sdf_time", "sdf_rate"]] = unit_df.apply(
            lambda r: spike_times_to_sdf(
                r["arrival_times"],
                t_start=float(np.min(r["arrival_times"])),
                t_end=float(np.max(r["arrival_times"])),
                bin_size=0.001,
                sigma=0.02,
            ),
            axis=1,
            result_type="expand",
        )

        rates = np.concatenate(unit_df["sdf_rate"].to_numpy()).astype(float)
        rates = np.sqrt(np.clip(rates, 0.0, None))
        mu = np.mean(rates)
        sd = np.std(rates, ddof=1)

        def _norm_trial(arr):
            arr = np.asarray(arr, dtype=float)
            arr = np.sqrt(np.clip(arr, 0.0, None))
            return (arr - mu) / sd

        unit_df["sdf_rate"] = unit_df["sdf_rate"].apply(_norm_trial)

        df.loc[unit_df.index, "sdf_rate"] = unit_df["sdf_rate"]
        df.loc[unit_df.index, "sdf_time"] = unit_df["sdf_time"]

    return df

def save_processed_trials(df, normalize=True, path=PROCESSED_TRIALS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    if normalize:
        path = path.with_suffix(path.suffix + "_normalized.pkl")
        df.to_pickle(path)
    print(f"Wrote {len(df)} rows to {path}")


def load_processed_trials(normalized=True, path=PROCESSED_TRIALS_PATH):
    path = Path(path)
    if normalized:
        path = path.with_suffix(path.suffix + "_normalized.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python preprocess.py` first.")
    return pd.read_pickle(path)
