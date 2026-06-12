from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts.plotting import *
from scripts.utils import *

PROCESSED_TRIALS_PATH = Path("data/processed_trials.pkl")


def _extract_population_field(population, field):
    """
    Extract one field from the MATLAB population struct.
    """
    values = []

    for item in population[field][0]:
        try:
            value = item[0]
        except Exception:
            value = item

        values.append(clean_mat_value(value))

    return values


def load_population_spike_data(filepath):
    """
    Load one MATLAB population file and return one row per unit-trial pair.

    Output rows contain:
        - unit-level info: unit_ID, pulvinar_hemifield
        - trial-level info: states, states_onset, arrival_times, labels, etc.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Missing {filepath}.")
    mat = loadmat(filepath)
    population = mat["population"]

    # Extract unit-level fields from MATLAB struct
    unit_ids = _extract_population_field(population, "unit_ID")
    targets = _extract_population_field(population, "target")
    trials = _extract_population_field(population, "trial")

    rows = []

    for unit_id, target, unit_trials in zip(unit_ids, targets, trials):
        recorded_side = pulvinar_to_side(clean_mat_value(target))

        # Convert recorded pulvinar side to contralateral visual hemifield
        pulvinar_hemifield = "left" if recorded_side == "right" else "right"

        trial_fields = unit_trials.dtype.names
        for trial_idx in range(unit_trials.shape[0]):
            row = {
                "unit_ID": unit_id,
                "recorded_side": recorded_side,
                "pulvinar_hemifield": pulvinar_hemifield,
                "trial_index": trial_idx,
            }
            for field in trial_fields:
                value = clean_mat_value(unit_trials[field][trial_idx])
                # Convert complex values like tar_pos to real part
                if np.iscomplexobj(value):
                    value = np.real(value)
                row[field] = value
            rows.append(row)
    return pd.DataFrame(rows)


def process_labels_and_filter(df):
    """
    Clean trial labels and keep only usable trials.

    Keeps:
        - type == 4
        - choice == 0
        - success == 1
        - reach or saccade trials only
        - trials with spike times

    Adds:
        - reach_hand as ipsi/contra
        - target_hemifield as ipsi/contra
    """
    df = df.copy()

    # Keep only correct/successful trials of the desired type
    df = df.query("type == 4 and choice == 0 and success == 1").copy()

    # Decode reach hand and convert to ipsi/contra relative to recorded pulvinar side
    reach_side = df["reach_hand"].apply(decode_reach_hand_label)
    df["reach_hand"] = [
        ipsi_contra(side, ref) for side, ref in zip(reach_side, df["recorded_side"])
    ]

    # Decode effector and remove joint saccade-reach trials
    df["effector"] = df["effector"].apply(decode_effector_label)
    df = df[df["effector"].isin(["reach", "saccade"])].copy()

    # Convert target position to ipsi/contra hemifield
    target_side = df["tar_pos"].apply(tar_pos_to_side)
    df["target_hemifield"] = [
        ipsi_contra(side, ref) for side, ref in zip(target_side, df["recorded_side"])
    ]
    # Remove trials without spike times
    df = df.dropna(subset=["arrival_times"])

    columns = [
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
    return df[columns].reset_index(drop=True)


def build_processed_trials(
    data_dir=Path("data"),
    *,
    normalize=True,
    plot=False,
):
    """
    Load all population .mat files, convert them into one trial-level DataFrame,
    process labels, filter trials, and optionally normalize firing rates.
    """
    population_files = sorted(data_dir.glob("*population*.mat"))
    if not population_files:
        raise FileNotFoundError(f"No *population*.mat files in {data_dir}")

    # Load each session separately
    sessions = []
    for filepath in population_files:
        session_df = load_population_spike_data(filepath)
        session_df["session"] = filepath.stem
        sessions.append(session_df)

    # Combine all sessions into one DataFrame
    df = pd.concat(sessions, ignore_index=True)

    # ------------------------------------------------------------
    # Plot trial_index counts
    # ------------------------------------------------------------
    trial_counts = df.groupby(by="unit_ID").count()["trial_index"]
    avg_trials_per_unit = trial_counts.mean()
    std_trials_per_unit = trial_counts.std()
    median_trials_per_unit = trial_counts.median()

    # print(f"Average number of trials per unit: {avg_trials_per_unit:.2f}")
    # print(f"STD number of trials per unit: {std_trials_per_unit:.2f}")
    # print(f"Median number of trials per unit: {median_trials_per_unit:.2f}")

    if plot:
        plots_dir = Path("plots/preprocessing")
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.bar(trial_counts.index, trial_counts.values)
        ax.set_xticks(np.arange(len(trial_counts.index)))
        ax.set_xticklabels(
            trial_counts.index,
            rotation=90,
            fontsize=6,
        )
        ax.set_xlabel("Unit")
        ax.set_ylabel("Trial counts")
        ax.set_title("Number of trials per unit")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "trial_counts.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Process labels and filter ---
    df = process_labels_and_filter(df)

    df = add_sdfs(
        df,
        spike_col="arrival_times",
        time_col="sdf_time",
        rate_col="sdf_rate",
        bin_size=0.001,
        sigma=0.05,
    )

    # ------------------------------------------------------------
    # Plot 10 random SDFs
    # ------------------------------------------------------------
    if plot:
        rng = np.random.default_rng(0)
        valid_sdf = df[
            df["sdf_time"].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)
            & df["sdf_rate"].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)
        ].copy()
        n_plot = min(10, len(valid_sdf))
        sample_idx = rng.choice(valid_sdf.index, size=n_plot, replace=False)
        plots_dir = Path("plots/preprocessing")
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx in sample_idx:
            row = valid_sdf.loc[idx]
            t = np.asarray(row["sdf_time"], dtype=float)
            r = np.asarray(row["sdf_rate"], dtype=float)
            ax.plot(
                t,
                r,
                lw=1.2,
                alpha=0.8,
                label=f"unit {row['unit_ID']}, trial {row['trial_index']}",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title("Random example SDFs")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / "random_10_sdfs.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Remove rows with invalid/NaN/empty rates
    df = remove_invalid_rate_rows(
        df,
        rate_col="sdf_rate",
        time_col="sdf_time",
    )

    # Remove low-firing units based on raw rates
    df, unit_stats = remove_low_firing_units(
        df,
        unit_cols=("session", "unit_ID"),
        rate_col="sdf_rate",
        threshold=2.0,
    )

    # Optional per-unit rate normalization
    if normalize:
        df = normalize_rates(
            df,
            unit_cols=("session", "unit_ID"),
            sqrt_transform=True,
        )

    return df


def remove_low_firing_units(
    df,
    *,
    unit_cols=("session", "unit_ID"),
    rate_col="sdf_rate",
    threshold=2.0,
):
    """
    Remove units whose mean raw firing rate is below threshold.

    Assumes rate_col contains raw firing-rate arrays in Hz.
    """
    unit_stats = (
        df.groupby(list(unit_cols))[rate_col]
        .apply(mean_rate_from_series)
        .rename("mean_rate")
        .reset_index()
    )

    good_units = unit_stats.loc[
        unit_stats["mean_rate"] >= threshold,
        list(unit_cols),
    ]

    df = df.merge(good_units, on=list(unit_cols), how="inner")

    return df, unit_stats


def remove_invalid_rate_rows(
    df,
    *,
    rate_col="sdf_rate",
    time_col="sdf_time",
):
    """
    Remove rows where rate/time arrays are missing, empty, or contain non-finite values.
    """
    df = df.copy()

    valid = df[rate_col].apply(is_valid_array)

    if time_col in df.columns:
        valid = valid & df[time_col].apply(is_valid_array)

    return df.loc[valid].reset_index(drop=True)


def remove_low_firing_units(
    df,
    *,
    unit_cols=("session", "unit_ID"),
    rate_col="sdf_rate",
    threshold=2.0,
):
    """
    Remove units whose mean raw firing rate is below threshold.

    Assumes rate_col contains raw firing-rate arrays in Hz.
    """
    unit_stats = (
        df.groupby(list(unit_cols))[rate_col]
        .apply(mean_rate_from_series)
        .rename("mean_rate")
        .reset_index()
    )

    good_units = unit_stats.loc[
        unit_stats["mean_rate"] >= threshold,
        list(unit_cols),
    ]

    df = df.merge(good_units, on=list(unit_cols), how="inner")

    return df.reset_index(drop=True), unit_stats


def normalize_trial(rate, *, mu, sd, sqrt_transform=True):
    """Normalize one trial SDF."""
    rate = np.asarray(rate, dtype=float)

    if sqrt_transform:
        rate = np.sqrt(np.clip(rate, 0.0, None))

    return (rate - mu) / sd


def is_nonempty_array(x):
    """Return True if x is an array-like object with at least one value."""
    try:
        return len(x) > 0
    except TypeError:
        return False


def add_sdfs(
    df,
    *,
    spike_col="arrival_times",
    time_col="sdf_time",
    rate_col="sdf_rate",
    bin_size=0.001,
    sigma=0.05,
):
    """
    Compute raw SDFs for each row.
    Does not normalize or filter.
    """
    df = df.copy()

    sdf = df[spike_col].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=float(np.min(spikes)),
            t_end=float(np.max(spikes)),
            bin_size=bin_size,
            sigma=sigma,
        )
    )

    df[time_col] = sdf.apply(lambda x: x[0])
    df[rate_col] = sdf.apply(lambda x: x[1])

    return df


def normalize_rates(
    df,
    *,
    unit_cols=("session", "unit_ID"),
    sqrt_transform=True,
    plot=False,
    plots_dir=Path("plots/preprocessing"),
):
    """
    Compute SDFs for each trial, then normalize firing rates per unit.

    Filtering:
        removes units with mean raw firing rate < min_mean_rate Hz

    Steps per unit:
    1. Convert spike times to SDFs.
    2. Concatenate all trial SDFs for that unit.
    3. Apply sqrt transform.
    4. Z-score using that unit's mean and std.

    If plot=True, saves firing-rate distributions before and after normalization.
    """
    df = df.copy()

    good_indices = []
    raw_rates_for_plot = []
    sqrt_rates_for_plot = []
    norm_raw_rates_for_plot = []
    norm_sqrt_rates_for_plot = []

    # Process one unit at a time
    for _, unit_df in df.groupby(list(unit_cols), sort=True):
        idx = unit_df.index

        # ------------------------------------------------------------
        # Pool raw rates for this unit
        # ------------------------------------------------------------
        sdf_rate = unit_df["sdf_rate"]
        raw_rates = np.concatenate(sdf_rate.to_numpy()).astype(float)
        sqrt_rates = np.sqrt(np.clip(raw_rates, 0.0, None))

        # ------------------------------------------------------------
        # Compute normalization statistics
        # ------------------------------------------------------------
        raw_mu = np.nanmean(raw_rates)
        raw_sd = np.nanstd(raw_rates, ddof=1)

        sqrt_mu = np.nanmean(sqrt_rates)
        sqrt_sd = np.nanstd(sqrt_rates, ddof=1)

        # Remove units with invalid variance
        if (
            not np.isfinite(raw_sd)
            or raw_sd < 1e-8
            or not np.isfinite(sqrt_sd)
            or sqrt_sd < 1e-8
        ):
            continue

        # ------------------------------------------------------------
        # Normalize rates
        # ------------------------------------------------------------

        # For plotting only: z-score without sqrt
        normalized_raw_rates = sdf_rate.apply(
            normalize_trial,
            mu=raw_mu,
            sd=raw_sd,
            sqrt_transform=False,
        )

        # Actual saved rate: sqrt + z-score
        normalized_sqrt_rates = sdf_rate.apply(
            normalize_trial,
            mu=sqrt_mu,
            sd=sqrt_sd,
            sqrt_transform=True,
        )
        df.loc[idx, "sdf_rate"] = normalized_sqrt_rates

        # Keep only rows from valid units/trials
        good_indices.extend(idx)

        if plot:
            raw_rates_for_plot.append(raw_rates)
            sqrt_rates_for_plot.append(sqrt_rates)
            norm_raw_rates_for_plot.append(
                np.concatenate(normalized_raw_rates.to_numpy()).astype(float)
            )
            norm_sqrt_rates_for_plot.append(
                np.concatenate(normalized_sqrt_rates.to_numpy()).astype(float)
            )

    # Remove rows with empty SDFs, invalid units, and low-firing units
    df = df.loc[good_indices].reset_index(drop=True)
    if plot:
        plot_rate_distributions_before_after(
            raw_rates_for_plot,
            sqrt_rates_for_plot,
            norm_raw_rates_for_plot,
            norm_sqrt_rates_for_plot,
            plots_dir=plots_dir,
        )
    return df


def save_processed_trials(
    df,
    *,
    normalize=False,
    path=PROCESSED_TRIALS_PATH,
):
    """
    Save processed trials to disk.

    If normalize=True, the file name gets '_normalized' before the extension.
    Example:
        processed_trials.pkl
        processed_trials_normalized.pkl
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalize:
        save_path = path.with_name(f"{path.stem}_normalized{path.suffix}")
    else:
        save_path = path
    df.to_pickle(save_path, compression="gzip")
    print(f"Wrote {len(df)} rows to {save_path}")
    return save_path


def load_processed_trials(
    *,
    normalized=False,
    path=PROCESSED_TRIALS_PATH,
):
    """
    Load processed trials from disk.

    If normalized=True, loads:
        processed_trials_normalized.pkl

    Otherwise loads:
        processed_trials.pkl
    """
    path = Path(path)

    if normalized:
        path = path.with_name(f"{path.stem}_normalized{path.suffix}")

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python 0_preprocess.py` first.")
    return pd.read_pickle(path, compression="gzip")
