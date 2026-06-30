from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *

PROCESSED_TRIALS_PATH = Path("data/processed_trials.pkl")


def _extract_population_field_old(population, field):
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


def load_population_spike_data_old(filepath):
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
    unit_ids = _extract_population_field_old(population, "unit_ID")
    targets = _extract_population_field_old(population, "target")
    trials = _extract_population_field_old(population, "trial")

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


def build_processed_trials_old(
    data_dir=Path("data/old_data"),
    *,
    sqrt_transform=True,
    zscore=False,
    min_mean_rate=1.0,
    bin_size=0.001,
    sigma=0.05,
    plot=False,
    plots_dir=Path("plots/preprocessing"),
):
    """
    Load old-format population .mat files, convert them into one trial-level
    DataFrame, process labels, filter trials, compute SDFs, and optionally
    transform / normalize firing rates.

    This keeps compatibility with the old one-file population format.

    Parameters
    ----------
    data_dir : Path
        Directory containing old-format *population*.mat files.

    sqrt_transform : bool
        If True, apply sqrt transform to SDF firing rates.

    zscore : bool
        If True, z-score firing rates per unit after optional sqrt transform.

    min_mean_rate : float
        Minimum mean raw firing rate in Hz required to keep a unit.

    plot : bool
        If True, save preprocessing diagnostic plots.

    plots_dir : Path
        Output directory for diagnostic plots.
    """
    data_dir = Path(data_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    population_files = sorted(data_dir.glob("*population*.mat"))
    if not population_files:
        raise FileNotFoundError(f"No *population*.mat files in {data_dir}")

    # ------------------------------------------------------------
    # Load each session separately
    # ------------------------------------------------------------
    sessions = []

    for filepath in population_files:
        session_df = load_population_spike_data_old(filepath)
        session_df["session"] = filepath.stem
        sessions.append(session_df)

    df = pd.concat(sessions, ignore_index=True)

    # ------------------------------------------------------------
    # Optional trial-count plot before filtering
    # ------------------------------------------------------------
    if plot:
        plot_trial_counts(
            df,
            unit_cols=("session", "unit_ID"),
            plots_dir=plots_dir,
            filename="trial_counts_old_format.png",
        )

    # ------------------------------------------------------------
    # Process labels and keep usable trials
    # ------------------------------------------------------------
    df = process_labels_and_filter(df)

    # ------------------------------------------------------------
    # Compute raw SDFs
    # ------------------------------------------------------------
    df = add_sdfs(
        df,
        spike_col="arrival_times",
        time_col="sdf_time",
        rate_col="sdf_rate",
        bin_size=bin_size,
        sigma=sigma,
    )

    # ------------------------------------------------------------
    # Optional random SDF plot before rate filtering / normalization
    # ------------------------------------------------------------
    if plot:
        plot_random_sdfs(
            df,
            time_col="sdf_time",
            rate_col="sdf_rate",
            plots_dir=plots_dir,
            filename="random_10_sdfs_old_format.png",
        )

    # ------------------------------------------------------------
    # Remove invalid SDF rows
    # ------------------------------------------------------------
    before = len(df)

    df = remove_invalid_rate_rows(
        df,
        rate_col="sdf_rate",
        time_col="sdf_time",
    )

    print(f"Removed {before - len(df)} rows with invalid SDFs.")
    print(f"Rows after SDF filtering: {len(df)}")

    # ------------------------------------------------------------
    # Remove low-firing units based on raw rates
    # Important: do this before sqrt/z-score.
    # ------------------------------------------------------------
    df, unit_stats = remove_low_firing_units(
        df,
        unit_cols=("session", "unit_ID"),
        rate_col="sdf_rate",
        threshold=min_mean_rate,
    )

    unit_stats.to_csv(
        plots_dir / "unit_mean_firing_rates_old_format.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Optional rate transform / normalization
    # ------------------------------------------------------------
    if sqrt_transform:
        df = sqrt_transform_rates(
            df,
            rate_col="sdf_rate",
        )

    if zscore:
        df = zscore_rates(
            df,
            unit_cols=("session", "unit_ID"),
            rate_col="sdf_rate",
        )

    return df


def session_key_from_path(path):
    """
    Extract subject + date from names like:
        population_Linus_20160513.mat
        trials_Linus_20160513.mat
        trials_Flaffus_20160608.mat

    Returns:
        Linus_20160513
        Flaffus_20160608
    """
    stem = Path(path).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Expected filename like population_NAME_DATE.mat or trials_NAME_DATE.mat, "
            f"got {Path(path).name}"
        )

    return "_".join(parts[1:])


def load_population_file(filepath):
    filepath = Path(filepath)
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    population = np.asarray(mat["population"]).ravel()

    rows = []

    for unit in population:
        unit_id = clean_mat_value(unit.unit_ID)
        target = clean_mat_value(unit.target)

        recorded_side = pulvinar_to_side(target)
        pulvinar_hemifield = "left" if recorded_side == "right" else "right"

        unit_trials = np.asarray(unit.trial, dtype=object).ravel()
        n_trials = len(unit_trials)

        accepted = np.asarray(clean_mat_value(unit.accepted), dtype=float).ravel()
        block = np.asarray(clean_mat_value(unit.block), dtype=float).ravel()
        run = np.asarray(clean_mat_value(unit.run), dtype=float).ravel()
        n = np.asarray(clean_mat_value(unit.n), dtype=float).ravel()

        for trial_idx, trial in enumerate(unit_trials):
            accepted_trial = int(accepted[trial_idx])

            arrival_times = clean_mat_value(trial).arrival_times
            arrival_times = clean_mat_value(arrival_times)

            # Make spike times a clean 1D float array
            if isinstance(arrival_times, np.ndarray):
                arrival_times = np.asarray(arrival_times, dtype=float).ravel()
            elif pd.isna(arrival_times):
                arrival_times = np.array([], dtype=float)
            else:
                arrival_times = np.asarray([arrival_times], dtype=float)

            rows.append(
                {
                    "unit_ID": unit_id,
                    "recorded_side": recorded_side,
                    "pulvinar_hemifield": pulvinar_hemifield,
                    "trial_index": trial_idx,
                    "block": int(block[trial_idx]),
                    "run": int(run[trial_idx]),
                    "n": int(n[trial_idx]),
                    "accepted": int(accepted[trial_idx]),
                    "arrival_times": arrival_times,
                }
            )

    return pd.DataFrame(rows)


def load_trials_file(filepath):
    """
    Load one trials_*.mat file.

    Output:
        one row per behavioral trial.

    Important:
        block, run, n are the merge keys used to match population unit-trials.
    """
    filepath = Path(filepath)
    mat = loadmat(
        filepath,
        squeeze_me=True,
        struct_as_record=False,
    )
    trials = np.asarray(mat["trials"]).ravel()

    rows = []
    for trial_file_index, trial in enumerate(trials):
        row = {
            "trial_file_index": trial_file_index,
        }
        for field in trial._fieldnames:
            value = clean_mat_value(getattr(trial, field))
            row[field] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["block", "run", "n"]:
        df[col] = df[col].astype(int)
    return df


def has_nonempty_arrival_times(x):
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except Exception:
        return False

    return arr.size > 0 and np.any(np.isfinite(arr))


def load_session(population_file, trial_file):
    """
    Load one matched population/trials session and merge them.
    """
    population_file = Path(population_file)
    trial_file = Path(trial_file)
    session = session_key_from_path(population_file)

    pop_df = load_population_file(population_file)
    trial_df = load_trials_file(trial_file)

    df = pop_df.merge(
        trial_df,
        on=["block", "run", "n"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_trial"),
    )
    df["session"] = session

    # Filter accepted unit-trials
    df = df[df["accepted"] == 1].reset_index(drop=True)

    df = df[df["arrival_times"].apply(has_nonempty_arrival_times)].reset_index(
        drop=True
    )

    columns_to_keep = [
        "session",
        "unit_ID",
        "trial_index",
        "recorded_side",
        "pulvinar_hemifield",
        "arrival_times",
        "type",
        "choice",
        "success",
        "effector",
        "reach_hand",
        "tar_pos",
        "fix_pos",
        "trial_onset_time",
        "run_onset_time",
        "states_onset",
        "states",
    ]
    return df[columns_to_keep].copy()


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
        "tar_pos",
        "fix_pos",
        "target_hemifield",
        "recorded_side",
        "trial_onset_time",
        "run_onset_time",
        "states_onset",
        "states",
        "arrival_times",
    ]
    return df[columns].reset_index(drop=True)


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
    threshold=1.0,
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


def sqrt_transform_trial(rate):
    """
    Apply square-root transform to one firing-rate array.
    """
    rate = np.asarray(rate, dtype=float)
    return np.sqrt(np.clip(rate, 0.0, None))


def sqrt_transform_rates(
    df,
    *,
    rate_col="sdf_rate",
):
    """
    Apply sqrt transform to firing-rate arrays.
    """
    df = df.copy()
    df[rate_col] = df[rate_col].apply(sqrt_transform_trial)
    return df


def zscore_rates(
    df,
    *,
    unit_cols=("session", "unit_ID"),
    rate_col="sdf_rate",
):
    """
    Z-score firing-rate arrays per unit.

    This uses all trials and all time bins for each unit:
        z = (rate - unit_mean) / unit_sd
    """
    df = df.copy()

    good_indices = []

    for _, unit_df in df.groupby(list(unit_cols), sort=True):
        idx = unit_df.index

        rates = np.concatenate(unit_df[rate_col].to_numpy()).astype(float)

        mu = np.nanmean(rates)
        sd = np.nanstd(rates, ddof=1)

        if not np.isfinite(sd) or sd < 1e-8:
            continue

        df.loc[idx, rate_col] = unit_df[rate_col].apply(
            lambda r: (np.asarray(r, dtype=float) - mu) / sd
        )

        good_indices.extend(idx)

    return df.loc[good_indices].reset_index(drop=True)


def build_processed_trials(
    data_dir=Path("data/new_data/flaffus"),
    *,
    use_old_data=False,
    old_data_dir=Path("data/old_data"),
    sqrt_transform=True,
    zscore=False,
    min_mean_rate=1.0,
    bin_size=0.001,
    sigma=0.05,
    plot=False,
    plots_dir=Path("plots/preprocessing"),
):
    """
    Build processed trial DataFrame.

    Default:
        Use the new two-file data format:
            population_*.mat
            trials_*.mat

    If use_old_data=True:
        Use the old one-file population format in old_data_dir.

    Processing steps:
        1. Load session data.
        2. Process behavioral labels and keep valid trials.
        3. Compute raw SDFs.
        4. Remove invalid SDF rows.
        5. Remove low-firing units based on raw firing rates.
        6. Optionally apply sqrt transform.
        7. Optionally apply per-unit z-scoring.
    """
    data_dir = Path(data_dir)
    old_data_dir = Path(old_data_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Old-format data path
    # ------------------------------------------------------------
    if use_old_data:
        return build_processed_trials_old(
            data_dir=old_data_dir,
            sqrt_transform=sqrt_transform,
            zscore=zscore,
            min_mean_rate=min_mean_rate,
            bin_size=bin_size,
            sigma=sigma,
            plot=plot,
            plots_dir=plots_dir,
        )

    # ------------------------------------------------------------
    # New-format data path
    # ------------------------------------------------------------
    population_files = sorted(data_dir.glob("*population*.mat"))
    trial_files = sorted(data_dir.glob("*trials*.mat"))

    if not population_files:
        raise FileNotFoundError(f"No *population*.mat files in {data_dir}")

    if not trial_files:
        raise FileNotFoundError(f"No *trials*.mat files in {data_dir}")

    pop_by_session = {session_key_from_path(f): f for f in population_files}
    trial_by_session = {session_key_from_path(f): f for f in trial_files}

    common_sessions = sorted(set(pop_by_session) & set(trial_by_session))

    missing_trials = sorted(set(pop_by_session) - set(trial_by_session))
    missing_population = sorted(set(trial_by_session) - set(pop_by_session))

    if missing_trials:
        print(f"Population files without trial files: {missing_trials}")

    if missing_population:
        print(f"Trial files without population files: {missing_population}")

    if not common_sessions:
        raise FileNotFoundError(
            f"No matched population/trials sessions found in {data_dir}."
        )

    # ------------------------------------------------------------
    # Load matched sessions
    # ------------------------------------------------------------
    sessions = []

    for session in common_sessions:
        print(f"Loading {session}")

        session_df = load_session(
            pop_by_session[session],
            trial_by_session[session],
        )

        sessions.append(session_df)

    df = pd.concat(sessions, ignore_index=True)

    print("\nLoaded raw new-format data:")
    print(f"  Rows: {len(df)}")
    print(f"  Units: {df[['session', 'unit_ID']].drop_duplicates().shape[0]}")
    print(f"  Sessions: {df['session'].nunique()}")

    trials_per_unit = (
        df.groupby(["session", "unit_ID"]).size().rename("n_trials").reset_index()
    )

    mean_trials = trials_per_unit["n_trials"].mean()
    sd_trials = trials_per_unit["n_trials"].std(ddof=1)

    print("\nTrials per unit:")
    print(f"  Mean: {mean_trials:.2f}")
    print(f"  SD:   {sd_trials:.2f}")

    # ------------------------------------------------------------
    # Optional trial-count diagnostic before filtering
    # ------------------------------------------------------------
    if plot:
        plot_trial_counts(
            df,
            unit_cols=("session", "unit_ID"),
            plots_dir=plots_dir,
            filename="trial_counts.png",
        )

    # ------------------------------------------------------------
    # Process labels and filter behavioral trials
    # ------------------------------------------------------------
    before = len(df)

    df = process_labels_and_filter(df)

    print("\nAfter label/trial filtering:")
    print(f"  Removed rows: {before - len(df)}")
    print(f"  Remaining rows: {len(df)}")
    print(f"  Units: {df['unit_ID'].drop_duplicates().shape[0]}")

    # ------------------------------------------------------------
    # Compute raw SDFs
    # ------------------------------------------------------------
    df = add_sdfs(
        df,
        spike_col="arrival_times",
        time_col="sdf_time",
        rate_col="sdf_rate",
        bin_size=bin_size,
        sigma=0.05,
        # sigma=sigma,
    )

    # ------------------------------------------------------------
    # Optional raw SDF diagnostic plot
    # ------------------------------------------------------------
    if plot:
        plot_random_sdfs(
            df,
            time_col="sdf_time",
            rate_col="sdf_rate",
            plots_dir=plots_dir,
            filename="random_10_sdfs.png",
        )

    # ------------------------------------------------------------
    # Remove invalid SDF rows
    # ------------------------------------------------------------
    before = len(df)

    df = remove_invalid_rate_rows(
        df,
        rate_col="sdf_rate",
        time_col="sdf_time",
    )

    print("\nAfter invalid-SDF filtering:")
    print(f"  Removed rows: {before - len(df)}")
    print(f"  Remaining rows: {len(df)}")
    print(f"  Units: {df['unit_ID'].drop_duplicates().shape[0]}")

    # ------------------------------------------------------------
    # Remove low-firing units using raw rates
    # ------------------------------------------------------------
    before_units = df["unit_ID"].drop_duplicates().shape[0]

    df, unit_stats = remove_low_firing_units(
        df,
        unit_cols=("session", "unit_ID"),
        rate_col="sdf_rate",
        threshold=min_mean_rate,
    )

    after_units = df["unit_ID"].drop_duplicates().shape[0]

    print("\nAfter low-firing unit filtering:")
    print(f"  Minimum mean rate: {min_mean_rate} Hz")
    print(f"  Removed units: {before_units - after_units}")
    print(f"  Remaining units: {after_units}")
    print(f"  Remaining rows: {len(df)}")

    unit_stats.to_csv(
        plots_dir / "unit_mean_firing_rates_new_format.csv",
        index=False,
    )

    trials_per_unit = (
        df.groupby(["session", "unit_ID"]).size().rename("n_trials").reset_index()
    )

    mean_trials = trials_per_unit["n_trials"].mean()
    sd_trials = trials_per_unit["n_trials"].std(ddof=1)

    print("\nTrials per unit:")
    print(f"  Mean: {mean_trials:.2f}")
    print(f"  SD:   {sd_trials:.2f}")
    print(f"  Units: {df[['session', 'unit_ID']].drop_duplicates().shape[0]}")

    # ------------------------------------------------------------
    # Optional sqrt transform
    # ------------------------------------------------------------
    if sqrt_transform:

        raw_rates_before_sqrt = df["sdf_rate"].copy()

        df = sqrt_transform_rates(
            df,
            rate_col="sdf_rate",
        )

        sqrt_rates_after = df["sdf_rate"].copy()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

            raw_vals = np.concatenate(raw_rates_before_sqrt.to_numpy()).astype(float)
            sqrt_vals = np.concatenate(sqrt_rates_after.to_numpy()).astype(float)

            raw_vals = raw_vals[np.isfinite(raw_vals)]
            sqrt_vals = sqrt_vals[np.isfinite(sqrt_vals)]

            axes[0].hist(raw_vals, bins=100, edgecolor="black", alpha=0.75)
            axes[0].set_title("Before sqrt transform")
            axes[0].set_xlabel("Firing rate (Hz)")
            axes[0].set_ylabel("Count")
            axes[0].grid(alpha=0.3)

            axes[1].hist(sqrt_vals, bins=100, edgecolor="black", alpha=0.75)
            axes[1].set_title("After sqrt transform")
            axes[1].set_xlabel("sqrt(firing rate)")
            axes[1].set_ylabel("Count")
            axes[1].grid(alpha=0.3)

            fig.suptitle("Firing-rate distributions before and after sqrt transform")

            fig.savefig(
                plots_dir / "rate_distribution_before_after_sqrt.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    # ------------------------------------------------------------
    # Optional per-unit z-scoring
    # ------------------------------------------------------------
    if zscore:
        df = zscore_rates(
            df,
            unit_cols=("unit_ID",),
            rate_col="sdf_rate",
        )

    print("\nFinal processed data:")
    print(f"  Rows: {len(df)}")
    print(f"  Units: {df['unit_ID'].drop_duplicates().shape[0]}")
    print(f"  sqrt_transform: {sqrt_transform}")
    print(f"  zscore: {zscore}")
    print(df.columns)

    return df


def save_processed_trials(
    df,
    *,
    path=Path("data/new_data/flaffus"),
):
    """
    Save processed trials to disk.

    Example:
        processed_trials.pkl
        processed_trials_normalized.pkl
    """
    path = Path(path) / "processed_trials.pkl"
    df.to_pickle(path, compression="gzip")
    print(f"Wrote {len(df)} rows to {path}")
    return path


def load_processed_trials(
    *,
    path=PROCESSED_TRIALS_PATH,
):
    """
    Load processed trials from disk.

    Loads:
        processed_trials.pkl
    """
    path = Path(path) / "processed_trials.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python 0_preprocess.py` first.")
    return pd.read_pickle(path, compression="gzip")
