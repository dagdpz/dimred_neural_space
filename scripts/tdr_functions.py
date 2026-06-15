import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from scripts.utils import *


def add_condition_independent_regressors(
    df,
    *,
    time_col="analysis_time",
    cue_time=0.0,
    go_time=1.3,
    mov_time_col=None,
    cue_duration=0.150,
):
    """
    Add condition-independent, time-dependent regressors.

    Each new regressor is stored as a 1D array with the same length as
    `analysis_time`.

    cueCI:
        cue period only.

    prepCI:
        after cue period until movement onset.

    movCI:
        from movement onset onward.
    """
    df = df.copy()

    if len(df) == 0:
        raise ValueError("Cannot add CI regressors to an empty dataframe.")

    if mov_time_col is None:
        raise ValueError("mov_time_col is required to define prepCI and movCI.")

    cueCI = []
    goCI = []
    prepCI = []
    movCI = []

    for _, row in df.iterrows():
        t = np.asarray(row[time_col], dtype=float)

        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            cueCI.append(np.full(0, np.nan))
            prepCI.append(np.full(0, np.nan))
            movCI.append(np.full(0, np.nan))
            continue

        t_mov = row[mov_time_col]

        if not np.isfinite(t_mov):
            cueCI.append(np.zeros_like(t, dtype=float))
            prepCI.append(np.zeros_like(t, dtype=float))
            movCI.append(np.zeros_like(t, dtype=float))
            continue

        t_mov = float(t_mov)

        cueCI.append(((t >= cue_time) & (t < cue_time + cue_duration)).astype(float))
        goCI.append(((t >= go_time) & (t < t_mov)).astype(float))

        prepCI.append(((t >= cue_time + cue_duration) & (t < t_mov)).astype(float))

        movCI.append((t >= t_mov).astype(float))

    df["cueCI"] = cueCI
    df["goCI"] = goCI
    df["prepCI"] = prepCI
    df["movCI"] = movCI

    return df


def add_effector_regressors(df):
    """
    Add effector regressors.

    Columns:
        saccade:
            1 for saccade trials, 0 otherwise

        ipsi_hand:
            1 for reach trials with ipsi hand, 0 otherwise

        contra_hand:
            1 for reach trials with contra hand, 0 otherwise

    These are scalar trial-level regressors.
    """
    df = df.copy()

    required_cols = ["effector", "reach_hand"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for action regressors: {missing}")

    df["saccade"] = (df["effector"] == "saccade").astype(float)

    df["ipsi_hand"] = (
        (df["effector"] == "reach") & (df["reach_hand"] == "ipsi")
    ).astype(float)

    df["contra_hand"] = (
        (df["effector"] == "reach") & (df["reach_hand"] == "contra")
    ).astype(float)

    return df


def mask_regressors(
    df,
    *,
    regressors,
    ci_col,
    suffix=None,
):
    """
    Convert scalar trial-level regressors into time-dependent regressors
    by multiplying each scalar value by a CI mask.

    Example:
        saccade_masked[t] = saccade * movCI[t]

    Parameters
    ----------
    regressors : list or tuple
        Scalar regressor columns to mask.

    ci_col : str
        Column containing 1D CI mask arrays.

    suffix : str or None
        If None, overwrite the original columns.
        If given, save as f"{reg}{suffix}".
    """
    df = df.copy()

    required_cols = list(regressors) + [ci_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for masking: {missing}")

    for reg in regressors:
        out_col = reg if suffix is None else f"{reg}{suffix}"

        masked = []
        for _, row in df.iterrows():
            mask = np.asarray(row[ci_col], dtype=float)
            value = float(row[reg])
            masked.append(value * mask)

        df[out_col] = masked

    return df


def lowdin_orthogonalization(A, tol=1e-12):
    """
    Löwdin symmetric orthogonalization.

    Parameters
    ----------
    A : array, shape (m, n)
        Columns are the vectors to orthogonalize.
        Requires columns to be linearly independent.
    tol : float
        Small eigenvalue cutoff for numerical stability.

    Returns
    -------
    Q : array, shape (m, n)
        Orthonormalized vectors as columns.
    """
    A = np.asarray(A, dtype=float)

    # Overlap / Gram matrix
    S = A.T @ A

    # Eigendecomposition of S
    eigvals, eigvecs = np.linalg.eigh(S)

    # Construct S^{-1/2}
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    Q = A @ S_inv_sqrt
    return Q


def fit_tdr_axes(
    df,
    *,
    unit_cols=("unit_ID",),
    regressors=("E", "T", "H"),
    rate_col="analysis_rate",
    time_col="analysis_time",
):
    """
    Fit one multilinear regression per unit.

    Each regressor can be either:
        - array-valued: one vector per trial, length n_time
        - scalar-valued: one value per trial, repeated over time

    Returns
    -------
    axes_raw : ndarray, shape n_units x n_regressors
        Raw regression beta vectors.

    axes_ortho : ndarray, shape n_units x n_regressors
        Orthogonalized TDR axes.

    units_used : list
        Units retained in the regression.

    task_regressors : tuple
        Names of returned axes.
    """
    task_regressors = tuple(regressors)

    units = (
        df[list(unit_cols)]
        .drop_duplicates()
        .sort_values(list(unit_cols))
        .itertuples(index=False, name=None)
    )
    units = list(units)

    betas = []
    units_used = []

    for unit in units:
        unit_df = df.copy()

        for col, val in zip(unit_cols, unit):
            unit_df = unit_df[unit_df[col] == val]

        required_cols = list(task_regressors) + [rate_col, time_col]
        unit_df = unit_df.dropna(subset=required_cols)

        if len(unit_df) == 0:
            continue

        # ------------------------------------------------------------
        # Response matrix: trials x time
        # ------------------------------------------------------------
        rates = np.stack(unit_df[rate_col].to_numpy()).astype(float)

        if rates.ndim != 2 or not np.all(np.isfinite(rates)):
            continue

        n_trials, n_time = rates.shape

        # Regression target:
        # one row per trial-timepoint
        y = rates.reshape(-1)

        # ------------------------------------------------------------
        # Design matrix
        # ------------------------------------------------------------
        X_cols = []

        for reg in task_regressors:
            first_val = unit_df[reg].iloc[0]

            # Array-valued time-dependent regressor
            if isinstance(first_val, (np.ndarray, list, tuple)):
                X_reg = np.stack(unit_df[reg].to_numpy()).astype(float)
                if X_reg.shape != rates.shape:
                    raise ValueError(
                        f"Regressor {reg!r} has shape {X_reg.shape}, "
                        f"but rates have shape {rates.shape}. "
                        "Each time-dependent regressor must have one array per trial "
                        "with the same length as analysis_rate."
                    )
                if not np.all(np.isfinite(X_reg)):
                    raise ValueError(f"Regressor {reg!r} contains non-finite values.")
                X_cols.append(X_reg.reshape(-1))
            # Scalar trial-level regressor
            else:
                values = unit_df[reg].to_numpy(dtype=float)

                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Regressor {reg!r} contains non-finite values.")

                X_cols.append(np.repeat(values, n_time))

        X = np.column_stack(X_cols)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Design matrix has {X.shape[0]} rows, " f"but y has {y.shape[0]} rows."
            )

        # Skip units with rank-deficient or all-zero design only if needed.
        # Usually LinearRegression can still fit, but beta interpretation
        # may be poor if a regressor is always zero for this unit.
        if not np.all(np.isfinite(X)):
            continue

        # ------------------------------------------------------------
        # Fit regression for this unit
        # ------------------------------------------------------------
        model = LinearRegression()
        model.fit(X, y)

        beta = np.asarray(model.coef_, dtype=float)
        betas.append(beta)
        units_used.append(unit)

    if len(betas) == 0:
        raise ValueError("No units were successfully fit.")

    axes_raw = np.stack(betas, axis=0)

    # ------------------------------------------------------------
    # Orthogonalize axes
    # ------------------------------------------------------------
    axes_ortho = lowdin_orthogonalization(axes_raw)

    return axes_raw, axes_ortho, units_used, task_regressors


def summarize_tdr_beta_effect_sizes(
    axes_raw,
    task_regressors,
    units=None,
):
    """
    Summarize TDR beta/effect sizes across units.

    axes_raw : array, shape n_units x n_regressors
        Raw regression coefficients from fit_tdr_axes.
        Since rates are normalized, beta magnitudes are in normalized-rate units.

    task_regressors : list or tuple
        Names of TDR regressors, e.g. E, T, H, EH, ET.

    units : list or None
        Optional unit identifiers returned by fit_tdr_axes.

    Returns
    -------
    beta_unit_stats : pd.DataFrame
        One row per unit x regressor.

    beta_summary : pd.DataFrame
        Summary statistics per regressor.

    beta_axis_summary : pd.DataFrame
        One row per regressor with population axis norm.
    """
    axes_raw = np.asarray(axes_raw, dtype=float)
    task_regressors = list(task_regressors)

    if axes_raw.ndim != 2:
        raise ValueError("axes_raw must be 2D: n_units x n_regressors.")

    if axes_raw.shape[1] != len(task_regressors):
        raise ValueError(
            f"axes_raw has {axes_raw.shape[1]} columns, "
            f"but task_regressors has {len(task_regressors)} entries."
        )

    n_units, n_regressors = axes_raw.shape

    if units is None:
        units = list(range(n_units))

    rows = []
    for unit, beta_row in zip(units, axes_raw):
        for reg, beta in zip(task_regressors, beta_row):
            rows.append(
                {
                    "unit": unit,
                    "regressor": reg,
                    "beta": beta,
                    "abs_beta": abs(beta),
                    "beta_squared": beta**2,
                }
            )

    beta_unit_stats = pd.DataFrame(rows)

    beta_summary = (
        beta_unit_stats.groupby("regressor")
        .agg(
            n_units=("beta", "count"),
            mean_beta=("beta", "mean"),
            median_beta=("beta", "median"),
            sd_beta=("beta", "std"),
            sem_beta=("beta", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            mean_abs_beta=("abs_beta", "mean"),
            median_abs_beta=("abs_beta", "median"),
            rms_beta=("beta_squared", lambda x: np.sqrt(np.mean(x))),
        )
        .reset_index()
    )

    # Population-level raw TDR axis strength:
    # length of beta vector across units.
    axis_norms = np.linalg.norm(axes_raw, axis=0)

    beta_axis_summary = pd.DataFrame(
        {
            "regressor": task_regressors,
            "axis_norm": axis_norms,
            "axis_norm_per_unit": axis_norms / np.sqrt(n_units),
        }
    )

    return beta_unit_stats, beta_summary, beta_axis_summary


def summarize_tdr_input_data(
    df,
    rate_col="analysis_rate",
    *,
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    unit_cols=("session", "unit_ID"),
):
    """
    Basic summary statistics for the population-average TDR input.

    Assumes one row = one unit-trial and rate_col is already
    sqrt-transformed + z-scored per unit.
    """
    plot_df = df.dropna(subset=list(condition_cols) + [rate_col]).copy()

    # Number of units
    n_units = plot_df[list(unit_cols)].drop_duplicates().shape[0]

    # Trial count per unit per condition
    unit_cond_counts = (
        plot_df.groupby(list(unit_cols) + list(condition_cols))
        .size()
        .rename("n_trials")
        .reset_index()
    )

    # Summary per condition
    cond_stats = (
        unit_cond_counts.groupby(list(condition_cols))["n_trials"]
        .agg(
            mean_n_trials_per_unit="mean",
            sd_n_trials_per_unit="std",
            min_n_trials_per_unit="min",
            max_n_trials_per_unit="max",
            n_units_present="count",
        )
        .reset_index()
    )

    cond_stats.insert(0, "total_n_units", n_units)

    # Overall mean number of trials per condition, across units and conditions
    overall_stats = {
        "n_units": n_units,
        "mean_n_trials_per_condition_per_unit": unit_cond_counts["n_trials"].mean(),
        "median_n_trials_per_condition_per_unit": unit_cond_counts["n_trials"].median(),
        "min_n_trials_per_condition_per_unit": unit_cond_counts["n_trials"].min(),
        "max_n_trials_per_condition_per_unit": unit_cond_counts["n_trials"].max(),
    }

    return cond_stats, overall_stats


def time_resolved_var_by_tdr_axes(
    condition_pop,
    axes_ortho,
    axis_names,
    time,
    *,
    center_across_conditions=False,
    normalize_across_axes=True,
):
    """
    Time-resolved variance explained by each TDR axis.

    At each time point:
        Xbar_t = units x conditions
        Xhat_t = beta_dim @ beta_dim.T @ Xbar_t
        R2_t = 1 - SSE/SST
    """
    time = np.asarray(time, dtype=float)
    rows = []

    conditions = list(condition_pop.keys())

    for t_idx, t in enumerate(time):
        # Xbar_t: units x conditions
        Xbar_t = np.column_stack(
            [
                np.asarray(condition_pop[cond], dtype=float)[:, t_idx]
                for cond in conditions
            ]
        )

        valid = np.all(np.isfinite(Xbar_t), axis=0)
        Xbar_t = Xbar_t[:, valid]

        if Xbar_t.shape[1] < 2:
            continue

        # Center across conditions for each unit
        if center_across_conditions:
            Xbar_t = Xbar_t - np.mean(Xbar_t, axis=1, keepdims=True)

        ss_total = np.sum(Xbar_t**2)

        if ss_total <= 0 or not np.isfinite(ss_total):
            continue

        # ------------------------------------------------------------
        # First compute raw variance explained for all axes
        # ------------------------------------------------------------
        axis_rows = []
        for axis_idx, axis_name in enumerate(axis_names):
            beta_dim = axes_ortho[:, [axis_idx]]

            # Reconstruction using only this axis
            Xhat_t = beta_dim @ (beta_dim.T @ Xbar_t)

            # Because beta_dim is orthonormal, this is the variance captured
            # by the projection onto this single axis.
            ss_explained = np.sum(Xhat_t**2)

            variance_explained = ss_explained / ss_total

            axis_rows.append(
                {
                    "time": t,
                    "axis": axis_name,
                    "variance_explained": variance_explained,
                    "percent_variance_explained": 100.0 * variance_explained,
                }
            )

        # ------------------------------------------------------------
        # Normalize across axes at this time point
        # ------------------------------------------------------------
        var_sum = np.sum([row["variance_explained"] for row in axis_rows])
        for row in axis_rows:
            row["variance_explained_sum_across_axes"] = var_sum

            row["normalized_variance_explained"] = row["variance_explained"] / var_sum

            row["percent_normalized_variance_explained"] = (
                100.0 * row["normalized_variance_explained"]
            )

            rows.append(row)
    return pd.DataFrame(rows)
