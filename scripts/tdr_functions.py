import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from scripts.utils import *


def add_tdr_regressors(df, interaction=False):
    """
    Binary coding:
        effector: reach=+1, saccade=-1
        target/space: ipsi=-1, contra=1
        hand: ipsi=-1, contra=1

    Interactions:
        EH = effector x hand
        ET = effector x target/space
    """
    df = df.copy()

    df["E"] = df["effector"].map({"reach": 1.0, "saccade": -1.0})
    df["T"] = df["target_hemifield"].map({"contra": 1.0, "ipsi": -1.0})
    df["H"] = df["reach_hand"].map({"contra": 1.0, "ipsi": -1.0})

    if interaction:
        df["EH"] = df["E"] * df["H"]
        df["ET"] = df["E"] * df["T"]

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
    unit_cols=("session", "unit_ID"),
    no_interaction=False,
    main_regressors=("E", "T", "H"),
    interaction_regressors=("EH", "ET"),
    rate_col="analysis_rate",
    time_col="analysis_time",
    cue_time_col="t_cue",
    go_time_col="t_go",
    include_ci_axes=False,
):
    """
    Fits one regression per unit.

    If include_ci_axes=False:
        returns only task axes: E, T, H, optionally EH, ET

    If include_ci_axes=True:
        returns task axes plus condition-independent axes:
            E, T, H, EH, ET, cueCI, goCI
    """
    task_regressors = tuple(main_regressors)
    if not no_interaction:
        task_regressors = task_regressors + tuple(interaction_regressors)
    units = (
        df[list(unit_cols)]
        .drop_duplicates()
        .sort_values(list(unit_cols))
        .itertuples(index=False, name=None)
    )
    units = list(units)

    betas = []
    units_used = []
    returned_regressors = None

    for unit in units:
        unit_df = df.copy()

        for col, val in zip(unit_cols, unit):
            unit_df = unit_df[unit_df[col] == val]

        required_cols = list(task_regressors) + [rate_col, time_col]
        if cue_time_col is not None and cue_time_col in unit_df.columns:
            required_cols.append(cue_time_col)
        if go_time_col is not None and go_time_col in unit_df.columns:
            required_cols.append(go_time_col)

        unit_df = unit_df.dropna(subset=required_cols)
        if len(unit_df) == 0:
            continue

        # Shape: n_trials x n_timepoints
        rates = np.stack(unit_df[rate_col].to_numpy()).astype(float)
        if not np.all(np.isfinite(rates)):
            continue

        # Regression target:
        # one row per trial-timepoint
        y = rates.reshape(-1)

        # Trial-level task regressors repeated across time
        n_trials, n_time = rates.shape

        # ------------------------------------------------------------
        # Condition-Independent Regressors
        # ------------------------------------------------------------
        t = np.asarray(unit_df[time_col].iloc[0], dtype=float)

        ci_regressors = []
        regressor_values = {}

        if cue_time_col is not None and cue_time_col in unit_df.columns:
            t_cue_analysis = unit_df[cue_time_col].to_numpy(dtype=float)
            cue_window = 0.150  # Have to define concretely using TGM
            cueCI = (
                (t[None, :] >= t_cue_analysis[:, None])
                & (t[None, :] < t_cue_analysis[:, None] + cue_window)
            ).astype(float)
            print(cueCI)

            plt.plot(cueCI[0])
            plt.show()
            exit()
            cueCI[~np.isfinite(t_cue_analysis), :] = 0.0

            regressor_values["cueCI"] = cueCI.reshape(-1)
            ci_regressors.append("cueCI")

        if go_time_col is not None and go_time_col in unit_df.columns:
            t_go_analysis = unit_df[go_time_col].to_numpy(dtype=float)

            goCI = (t[None, :] >= t_go_analysis[:, None]).astype(float)
            goCI[~np.isfinite(t_go_analysis), :] = 0.0

            regressor_values["goCI"] = goCI.reshape(-1)
            ci_regressors.append("goCI")

        # ------------------------------------------------------------
        # Time-dependent Masks
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Design matrix
        # ------------------------------------------------------------
        all_regressors = task_regressors + tuple(ci_regressors)

        for reg in task_regressors:
            regressor_values[reg] = np.repeat(
                unit_df[reg].to_numpy(dtype=float),
                n_time,
            )

        X = np.column_stack([regressor_values[reg] for reg in all_regressors])

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)

        # ------------------------------------------------------------
        # Keep task axes only, or task + CI axes
        # ------------------------------------------------------------
        coef = pd.Series(
            model.coef_,
            index=all_regressors,
            dtype=float,
        )
        if include_ci_axes:
            returned_regressors_this_unit = all_regressors
        else:
            returned_regressors_this_unit = task_regressors

        if returned_regressors is None:
            returned_regressors = returned_regressors_this_unit
        elif returned_regressors != returned_regressors_this_unit:
            raise ValueError(
                "Different units produced different regressor sets. "
                f"Expected {returned_regressors}, got {returned_regressors_this_unit}."
            )

        betas.append(coef.loc[list(returned_regressors)].to_numpy())
        units_used.append(unit)

    axes_raw = np.asarray(betas, dtype=float)

    if axes_raw.size == 0:
        raise ValueError("No valid units were available for fitting TDR axes.")

    axes_ortho = lowdin_orthogonalization(axes_raw)

    return axes_raw, axes_ortho, units_used, returned_regressors


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
