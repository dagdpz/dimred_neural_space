from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from scripts.utils import *


def get_good_units_for_decoding(
    df,
    *,
    label_col,
    unit_col="unit_ID",
    min_trials_per_label=5,
):
    """
    Return units that have enough trials for every label.

    A unit is kept only if it has at least `min_trials_per_label`
    trials for each label. This is needed before KFold decoding,
    because each unit must contribute train/test trials for every class.
    """
    labels = sorted(df[label_col].dropna().unique())

    good_units = []
    for unit, unit_df in df.groupby(unit_col):
        n_per_label = [len(unit_df[unit_df[label_col] == label]) for label in labels]

        if min(n_per_label) >= min_trials_per_label:
            good_units.append(unit)

    return sorted(good_units)


def build_pseudopopulation_from_pools(
    pools,
    *,
    units,
    labels,
    n_pseudotrials_per_class,
    rng,
):
    """
    pools[(unit, label)] = array of shape n_real_trials x n_time

    Returns:
        X: n_pseudotrials x n_units x n_time
        y: n_pseudotrials
    """
    X_list = []
    y_list = []

    for label in labels:
        for _ in range(n_pseudotrials_per_class):
            pseudo = []

            for unit in units:
                rates = pools[(unit, label)]
                idx = rng.integers(0, rates.shape[0])
                pseudo.append(rates[idx])

            pseudo = np.stack(pseudo, axis=0)
            X_list.append(pseudo)
            y_list.append(label)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)

    return X, y


def make_groupwise_kfold_splits(
    df,
    *,
    group_col,
    n_splits=5,
    random_state=0,
):
    """
    Make KFold train/test splits separately for each group level.

    Returns
    -------
    splits : dict
        splits[group][fold_idx] = (train_indices, test_indices)
    """
    splits = {}

    for group, group_df in df.groupby(group_col):
        row_indices = group_df.index.to_numpy()

        kfold = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        folds = []

        for train_local, test_local in kfold.split(row_indices):
            train_idx = row_indices[train_local]
            test_idx = row_indices[test_local]
            folds.append((train_idx, test_idx))

        splits[group] = folds

    return splits


def time_resolved_logistic_decoding(
    df,
    *,
    label_col,
    unit_col="unit_ID",
    rate_col="analysis_rate",
    time_col="analysis_time",
    n_pseudotrials_per_class=100,
    n_splits=5,
    random_state=0,
    C=1.0,
    max_iter=1000,
    shuffle_labels=False,
):
    """
    Time-resolved pseudo-population decoding.

    Input dataframe:
        one row = one unit-trial
        rate_col = 1D activity array over time

    Goal:
        At each time point, train logistic regression:

            X_t = pseudotrials x units
            y   = task labels

    Important:
        We split real trials into train/test BEFORE making pseudotrials.
        This avoids the same real trial appearing in both train and test.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------------------
    # Remove rows missing the label, unit ID, rate array, or time array.
    df = df.dropna(subset=[label_col, unit_col, rate_col, time_col]).copy()

    if len(df) == 0:
        raise ValueError("No rows left after dropping NaNs.")

    # Use the first remaining row as the common time axis.
    # All rate arrays must match this length.
    analysis_time = np.asarray(df[time_col].iloc[0], dtype=float)
    n_time = len(analysis_time)

    # Keep only rows where the rate vector is a valid 1D finite array.
    df = df[df[rate_col].apply(is_valid_array)].copy()

    # Keep only rows where the rate vector has the same length as the time axis.
    df = df[
        df[rate_col].apply(lambda x: np.asarray(x, dtype=float).size == n_time)
    ].copy()

    if len(df) == 0:
        raise ValueError("No valid rate arrays left.")

    # ------------------------------------------------------------
    # Optional shuffled-label control
    # ------------------------------------------------------------

    # If requested, randomly shuffle labels across rows.
    # This destroys the true relationship between neural activity and condition.
    # Decoding accuracy should then be close to chance.
    if shuffle_labels:
        label_col_used = f"{label_col}_shuffled"
        df[label_col_used] = rng.permutation(df[label_col].to_numpy())
    else:
        label_col_used = label_col

    # Get the two class labels used for binary decoding.
    labels = sorted(df[label_col_used].dropna().unique())
    if len(labels) != 2:
        raise ValueError(
            f"Only binary decoding is supported here. "
            f"{label_col_used} has labels: {labels}"
        )

    # ------------------------------------------------------------
    # Keep only units with enough trials for both labels
    # ------------------------------------------------------------
    good_units = get_good_units_for_decoding(
        df,
        label_col=label_col_used,
        unit_col=unit_col,
        min_trials_per_label=n_splits,
    )

    if len(good_units) == 0:
        raise ValueError(f"No units have at least {n_splits} trials for every label.")

    df = df[df[unit_col].isin(good_units)].copy()

    print(
        f"Decoding {label_col}"
        f"{' with shuffled labels' if shuffle_labels else ''}: "
        f"{len(good_units)} units, labels={labels}"
    )

    # ------------------------------------------------------------
    # Precompute train/test splits separately for each unit and label
    # ------------------------------------------------------------
    # For each unit and each label, store the KFold train/test row indices.
    # This ensures that real trials are split before pseudotrials are built,
    # so the same real trial cannot appear in both train and test data.
    #
    # Structure:
    #   split_indices[(unit, label)][fold_idx] = (train_indices, test_indices)
    split_indices = {}

    for unit in good_units:
        unit_df = df[df[unit_col] == unit]

        label_splits = make_groupwise_kfold_splits(
            unit_df,
            group_col=label_col_used,
            n_splits=n_splits,
            random_state=random_state,
        )

        for label, folds in label_splits.items():
            split_indices[(unit, label)] = folds

    # ------------------------------------------------------------
    # Decode independently at every time bin
    # ------------------------------------------------------------
    fold_accuracy = np.full((n_splits, n_time), np.nan)
    for fold_idx in tqdm(
        range(n_splits),
        desc=f"Decoding {label_col}",
    ):
        train_pools = {}
        test_pools = {}

        # Build train/test pools for this fold
        for unit in good_units:
            for label in labels:
                train_idx, test_idx = split_indices[(unit, label)][fold_idx]

                train_rates = np.stack(df.loc[train_idx, rate_col].to_numpy()).astype(
                    float
                )

                test_rates = np.stack(df.loc[test_idx, rate_col].to_numpy()).astype(
                    float
                )

                train_pools[(unit, label)] = train_rates
                test_pools[(unit, label)] = test_rates

        # Build train and test pseudopopulations
        X_train, y_train = build_pseudopopulation_from_pools(
            train_pools,
            units=good_units,
            labels=labels,
            n_pseudotrials_per_class=n_pseudotrials_per_class,
            rng=rng,
        )

        X_test, y_test = build_pseudopopulation_from_pools(
            test_pools,
            units=good_units,
            labels=labels,
            n_pseudotrials_per_class=n_pseudotrials_per_class,
            rng=rng,
        )

        # Decode each time point separately
        for t_idx in range(n_time):
            X_train_t = X_train[:, :, t_idx]
            X_test_t = X_test[:, :, t_idx]

            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    l1_ratio=0,
                    C=C,
                    solver="liblinear",
                    max_iter=max_iter,
                ),
            )

            clf.fit(X_train_t, y_train)
            y_pred = clf.predict(X_test_t)

            fold_accuracy[fold_idx, t_idx] = accuracy_score(y_test, y_pred)

    # ------------------------------------------------------------
    # Average across folds
    # ------------------------------------------------------------
    accuracy = np.nanmean(fold_accuracy, axis=0)
    sem = np.nanstd(fold_accuracy, axis=0, ddof=1) / np.sqrt(n_splits)
    results = pd.DataFrame(
        {
            "time": analysis_time,
            "accuracy": accuracy,
            "sem": sem,
            "chance": 1.0 / len(labels),
        }
    )

    info = {
        "label_col": label_col,
        "labels": labels,
        "units": good_units,
        "n_units": len(good_units),
        "n_splits": n_splits,
        "n_pseudotrials_per_class": n_pseudotrials_per_class,
        "fold_accuracy": fold_accuracy,
    }

    return results, info


def temporal_generalization_logistic_decoding(
    df,
    *,
    label_col,
    unit_col="unit_ID",
    rate_col="analysis_rate",
    time_col="analysis_time",
    n_pseudotrials_per_class=100,
    n_splits=5,
    random_state=0,
    C=1.0,
    max_iter=1000,
    shuffle_labels=False,
    time_step=1,
):
    """
    Temporal generalization decoding.

    Instead of training and testing at the same time point, this trains
    a decoder at one time point and tests it at every other time point.

    Output:
        accuracy[train_time, test_time]

    The diagonal is equivalent to normal time-resolved decoding.
    Off-diagonal accuracy shows whether a code learned at one time
    generalizes to other times.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------------------
    # Remove rows missing the label, unit ID, rate array, or time array.
    df = df.dropna(subset=[label_col, unit_col, rate_col, time_col]).copy()

    if len(df) == 0:
        raise ValueError("No rows left after dropping NaNs.")

    # Use the first remaining row as the common time axis.
    # All rate arrays must match this length.
    analysis_time = np.asarray(df[time_col].iloc[0], dtype=float)
    n_time = len(analysis_time)

    # Keep only valid 1D finite rate arrays.
    df = df[df[rate_col].apply(is_valid_array)].copy()

    # Keep only rate arrays with the same length as the time axis.
    df = df[
        df[rate_col].apply(lambda x: np.asarray(x, dtype=float).size == n_time)
    ].copy()

    if len(df) == 0:
        raise ValueError("No valid rate arrays left.")

    # ------------------------------------------------------------
    # Optional shuffled-label control
    # ------------------------------------------------------------
    # If labels are shuffled, true neural-label structure is destroyed.
    # The TGM should then be close to chance.
    if shuffle_labels:
        label_col_used = f"{label_col}_shuffled"
        df[label_col_used] = rng.permutation(df[label_col].to_numpy())
    else:
        label_col_used = label_col

    # Only binary decoding is supported here.
    labels = sorted(df[label_col_used].dropna().unique())

    if len(labels) != 2:
        raise ValueError(
            f"Only binary decoding is supported here. "
            f"{label_col_used} has labels: {labels}"
        )

    # ------------------------------------------------------------
    # Keep only units with enough trials for both labels
    # ------------------------------------------------------------
    good_units = get_good_units_for_decoding(
        df,
        label_col=label_col_used,
        unit_col=unit_col,
        min_trials_per_label=n_splits,
    )

    if len(good_units) == 0:
        raise ValueError(f"No units have at least {n_splits} trials for every label.")

    df = df[df[unit_col].isin(good_units)].copy()

    print(
        f"Temporal generalization decoding {label_col}"
        f"{' with shuffled labels' if shuffle_labels else ''}: "
        f"{len(good_units)} units, labels={labels}"
    )

    # ------------------------------------------------------------
    # Choose evaluated time points
    # ------------------------------------------------------------
    # With 1 ms bins, the full matrix can be very large.
    # time_step=10 uses every 10th time point.
    if time_step is None or time_step < 1:
        time_step = 1

    time_indices = np.arange(0, n_time, time_step)
    tgm_time = analysis_time[time_indices]
    n_eval_time = len(time_indices)

    # ------------------------------------------------------------
    # Precompute train/test splits separately for each unit and label
    # ------------------------------------------------------------
    # Real trials are split before pseudotrials are built.
    # This prevents the same real trial from appearing in both train
    # and test pseudopopulations.
    #
    # Structure:
    #   split_indices[(unit, label)][fold_idx] = (train_indices, test_indices)
    split_indices = {}

    for unit in good_units:
        unit_df = df[df[unit_col] == unit]

        label_splits = make_groupwise_kfold_splits(
            unit_df,
            group_col=label_col_used,
            n_splits=n_splits,
            random_state=random_state,
        )

        for label, folds in label_splits.items():
            split_indices[(unit, label)] = folds

    # ------------------------------------------------------------
    # Temporal generalization decoding
    # ------------------------------------------------------------
    # fold_accuracy[fold, train_time, test_time]
    fold_accuracy = np.full(
        (n_splits, n_eval_time, n_eval_time),
        np.nan,
    )

    for fold_idx in tqdm(
        range(n_splits),
        desc=f"Decoding {label_col}",
    ):
        train_pools = {}
        test_pools = {}

        # Build train/test pools for this fold.
        for unit in good_units:
            for label in labels:
                train_idx, test_idx = split_indices[(unit, label)][fold_idx]

                train_rates = np.stack(df.loc[train_idx, rate_col].to_numpy()).astype(
                    float
                )

                test_rates = np.stack(df.loc[test_idx, rate_col].to_numpy()).astype(
                    float
                )

                train_pools[(unit, label)] = train_rates
                test_pools[(unit, label)] = test_rates

        # Build train and test pseudopopulations.
        # Shape:
        #   X_train: n_pseudotrials x n_units x n_time
        #   X_test:  n_pseudotrials x n_units x n_time
        X_train, y_train = build_pseudopopulation_from_pools(
            train_pools,
            units=good_units,
            labels=labels,
            n_pseudotrials_per_class=n_pseudotrials_per_class,
            rng=rng,
        )

        X_test, y_test = build_pseudopopulation_from_pools(
            test_pools,
            units=good_units,
            labels=labels,
            n_pseudotrials_per_class=n_pseudotrials_per_class,
            rng=rng,
        )

        # Train at one time point, test at all time points.
        for i_train, train_t_idx in enumerate(time_indices):
            X_train_t = X_train[:, :, train_t_idx]
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=max_iter,
                ),
            )

            clf.fit(X_train_t, y_train)

            for i_test, test_t_idx in enumerate(time_indices):
                X_test_t = X_test[:, :, test_t_idx]
                y_pred = clf.predict(X_test_t)

                fold_accuracy[fold_idx, i_train, i_test] = accuracy_score(
                    y_test,
                    y_pred,
                )

    # ------------------------------------------------------------
    # Average across folds
    # ------------------------------------------------------------
    accuracy = np.nanmean(fold_accuracy, axis=0)
    sem = np.nanstd(fold_accuracy, axis=0, ddof=1) / np.sqrt(n_splits)

    # Store matrix also as a DataFrame for convenient CSV saving.
    accuracy_df = pd.DataFrame(
        accuracy,
        index=tgm_time,
        columns=tgm_time,
    )
    accuracy_df.index.name = "train_time"
    accuracy_df.columns.name = "test_time"

    sem_df = pd.DataFrame(
        sem,
        index=tgm_time,
        columns=tgm_time,
    )
    sem_df.index.name = "train_time"
    sem_df.columns.name = "test_time"

    info = {
        "label_col": label_col,
        "labels": labels,
        "units": good_units,
        "n_units": len(good_units),
        "n_splits": n_splits,
        "n_pseudotrials_per_class": n_pseudotrials_per_class,
        "chance": 1.0 / len(labels),
        "time": tgm_time,
        "time_indices": time_indices,
        "fold_accuracy": fold_accuracy,
        "sem": sem,
    }

    return accuracy_df, info


def plot_time_resolved_decoding(
    results,
    *,
    out_path,
    title=None,
    ylabel="Decoding accuracy",
    xlabel="Time (s)",
    event_times=None,
    event_labels=None,
    show_sem=True,
):
    """
    Plot one decoding accuracy curve over time.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = results["time"].to_numpy(dtype=float)
    acc = results["accuracy"].to_numpy(dtype=float)
    chance = np.nanmean(results["chance"].to_numpy(dtype=float))

    finite = np.isfinite(t) & np.isfinite(acc)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    ax.plot(
        t[finite],
        acc[finite],
        lw=2.0,
        label="Accuracy",
    )

    if show_sem and "sem" in results.columns:
        sem = results["sem"].to_numpy(dtype=float)
        ax.fill_between(
            t[finite],
            acc[finite] - sem[finite],
            acc[finite] + sem[finite],
            alpha=0.2,
            linewidth=0,
        )

    ax.axhline(
        chance,
        color="0.5",
        linestyle="--",
        lw=1.2,
        label="Chance",
    )

    if event_times is not None:
        if event_labels is None:
            event_labels = [None] * len(event_times)

        for event_time, event_label in zip(event_times, event_labels):
            ax.axvline(
                event_time,
                color="k",
                linestyle=":",
                lw=1.2,
                alpha=0.8,
                label=event_label,
            )

    if title is None:
        title = "Time-resolved decoding"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_multiple_time_resolved_decoding(
    decoding_results,
    *,
    out_path,
    title="Time-resolved decoding",
    ylabel="Decoding accuracy",
    xlabel="Time (s)",
    event_times=None,
    event_labels=None,
    show_sem=True,
):
    """
    Plot several decoding curves in one figure.

    decoding_results:
        {
            "Effector": dec_effector,
            "Space": dec_space,
            "Reach hand": dec_hand,
        }
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    chance_values = []

    for label, results in decoding_results.items():
        t = results["time"].to_numpy(dtype=float)
        acc = results["accuracy"].to_numpy(dtype=float)
        chance = np.nanmean(results["chance"].to_numpy(dtype=float))

        finite = np.isfinite(t) & np.isfinite(acc)

        line = ax.plot(
            t[finite],
            acc[finite],
            lw=2.0,
            label=label,
        )[0]

        if show_sem and "sem" in results.columns:
            sem = results["sem"].to_numpy(dtype=float)
            ax.fill_between(
                t[finite],
                acc[finite] - sem[finite],
                acc[finite] + sem[finite],
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )

        chance_values.append(chance)

    unique_chance = np.unique(np.round(chance_values, 6))

    if len(unique_chance) == 1:
        ax.axhline(
            unique_chance[0],
            color="0.5",
            linestyle="--",
            lw=1.2,
            label="Chance",
        )
    else:
        for label, results in decoding_results.items():
            chance = np.nanmean(results["chance"].to_numpy(dtype=float))
            ax.axhline(
                chance,
                linestyle="--",
                lw=1.0,
                alpha=0.5,
                label=f"{label} chance",
            )

    if event_times is not None:
        if event_labels is None:
            event_labels = [None] * len(event_times)

        for event_time, event_label in zip(event_times, event_labels):
            ax.axvline(
                event_time,
                color="k",
                linestyle=":",
                lw=1.2,
                alpha=0.8,
                label=event_label,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_temporal_generalization_matrix(
    accuracy_df,
    *,
    out_path,
    title="Temporal generalization matrix",
    xlabel="Test time (s)",
    ylabel="Train time (s)",
    chance=None,
    event_times=None,
    event_labels=None,
    vmin=None,
    vmax=1.0,
    title_fontsize=18,
    label_fontsize=16,
    tick_fontsize=13,
    colorbar_fontsize=14,
    legend_fontsize=12,
):
    """
    Plot temporal generalization accuracy matrix.

    Rows:
        train time

    Columns:
        test time

    Diagonal:
        normal time-resolved decoding accuracy
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    acc = accuracy_df.to_numpy(dtype=float)
    train_time = accuracy_df.index.to_numpy(dtype=float)
    test_time = accuracy_df.columns.to_numpy(dtype=float)

    if vmin is None:
        vmin = chance if chance is not None else np.nanmin(acc)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    im = ax.imshow(
        acc,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            test_time[0],
            test_time[-1],
            train_time[0],
            train_time[-1],
        ],
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Decoding accuracy", fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Diagonal: train time = test time
    ax.plot(
        [test_time[0], test_time[-1]],
        [train_time[0], train_time[-1]],
        color="white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="train = test",
    )

    event_color = "k"

    # Optional event markers on both train and test axes.
    if event_times is not None:
        if event_labels is None:
            event_labels = [None] * len(event_times)

        for event_time, event_label in zip(event_times, event_labels):
            ax.axvline(
                event_time,
                color=event_color,
                linestyle=":",
                linewidth=2.0,
                alpha=0.8,
            )
            ax.axhline(
                event_time,
                color=event_color,
                linestyle=":",
                linewidth=2.0,
                alpha=0.8,
                label=event_label,
            )

    if chance is not None:
        title = f"{title} | chance = {chance:.2f}"

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    legend = ax.legend(
        frameon=True,
        fontsize=14,
        loc="upper left",
        facecolor="white",
        edgecolor="0.8",
        framealpha=0.7,
        fancybox=True,
    )

    for text in legend.get_texts():
        if text.get_text() != "train = test":
            text.set_color(event_color)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path
