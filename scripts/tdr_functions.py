import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from scripts.utils import *
from scripts.plotting import *


def add_tdr_int_regressors(df, interaction=False):
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


def add_condition_independent_regressors(df):
    """
    Add condition-independent, time-dependent regressors.

    Each new regressor is stored as a 1D array with the same length as
    `analysis_time`.
    """
    time_col = "analysis_time"

    df = df.copy()

    if len(df) == 0:
        raise ValueError("Cannot add CI regressors to an empty dataframe.")

    required_cols = [time_col, "t_cue", "t_go", "t_mov", "t_mov_end"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    cueCI = []
    planCI = []
    goCI = []
    movCI = []

    for _, row in df.iterrows():
        t = np.asarray(row[time_col], dtype=float)

        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            cueCI.append(np.full(0, np.nan))
            planCI.append(np.full(0, np.nan))
            goCI.append(np.full(0, np.nan))
            movCI.append(np.full(0, np.nan))
            continue

        t_mov = float(row["t_mov"])
        t_cue = float(row["t_cue"])
        t_go = float(row["t_go"])
        t_mov_end = float(row["t_mov_end"])

        if not np.isfinite(t_mov):
            cueCI.append(np.zeros_like(t, dtype=float))
            planCI.append(np.zeros_like(t, dtype=float))
            goCI.append(np.zeros_like(t, dtype=float))
            movCI.append(np.zeros_like(t, dtype=float))
            continue

        cueCI.append(((t >= t_cue + 0.05) & (t < t_cue + 0.2)).astype(float))
        planCI.append(((t >= t_cue + 0.2) & (t < t_go + 0.05)).astype(float))
        goCI.append(((t >= t_go + 0.05) & (t < t_go + 0.2)).astype(float))
        movCI.append(((t >= t_mov) & (t < t_mov + 0.3)).astype(float))

    df["cueCI"] = cueCI
    df["planCI"] = planCI
    df["goCI"] = goCI
    df["movCI"] = movCI

    return df


def add_condition_independent_regressors_nogo(df):
    """
    Add condition-independent, time-dependent regressors.

    Each new regressor is stored as a 1D array with the same length as
    `analysis_time`.
    """
    time_col = "analysis_time"

    df = df.copy()

    if len(df) == 0:
        raise ValueError("Cannot add CI regressors to an empty dataframe.")

    required_cols = [time_col, "t_cue", "t_mov", "t_mov_end"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    cueCI = []
    planCI = []
    movCI = []

    for _, row in df.iterrows():
        t = np.asarray(row[time_col], dtype=float)

        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            cueCI.append(np.full(0, np.nan))
            planCI.append(np.full(0, np.nan))
            movCI.append(np.full(0, np.nan))
            continue

        t_mov = float(row["t_mov"])
        t_cue = float(row["t_cue"])
        t_mov_end = float(row["t_mov_end"])

        if not np.isfinite(t_mov):
            cueCI.append(np.zeros_like(t, dtype=float))
            planCI.append(np.zeros_like(t, dtype=float))
            movCI.append(np.zeros_like(t, dtype=float))
            continue

        cueCI.append(((t >= t_cue + 0.05) & (t < t_cue + 0.2)).astype(float))
        planCI.append(((t >= t_cue + 0.2) & (t < t_mov)).astype(float))
        movCI.append(((t >= t_mov) & (t < t_mov + 0.3)).astype(float))

    df["cueCI"] = cueCI
    df["planCI"] = planCI
    df["movCI"] = movCI

    return df


def add_condition_independent_regressors_mov(df):
    """
    Add condition-independent, time-dependent regressors.

    Each new regressor is stored as a 1D array with the same length as
    `analysis_time`.
    """
    time_col = "analysis_time"

    df = df.copy()

    if len(df) == 0:
        raise ValueError("Cannot add CI regressors to an empty dataframe.")

    required_cols = [
        time_col,
        "t_go",
        "t_mov",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    goCI = []
    movCI = []

    for _, row in df.iterrows():
        t = np.asarray(row[time_col], dtype=float)

        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            goCI.append(np.full(0, np.nan))
            movCI.append(np.full(0, np.nan))
            continue

        t_mov = float(row["t_mov"])
        t_go = float(row["t_go"])

        if not np.isfinite(t_mov):
            goCI.append(np.zeros_like(t, dtype=float))
            movCI.append(np.zeros_like(t, dtype=float))
            continue

        goCI.append(((t >= t_go + 0.05) & (t < t_go + 0.2)).astype(float))
        movCI.append(((t >= t_mov) & (t < t_mov + 0.3)).astype(float))

    df["goCI"] = goCI
    df["movCI"] = movCI

    return df


def add_target_xy_regressors(df):
    """
    Add continuous target-position regressors.

    space_x:
        horizontal target position, signed relative to pulvinar:
            ipsi   = negative
            contra = positive

    space_y:
        vertical target position relative to fixation position:
            target_y - fixation_y
    """
    df = df.copy()

    if "tar_pos" not in df.columns:
        raise ValueError("Missing required column: tar_pos")

    raw_x = df["tar_pos"].apply(lambda z: np.real(z) if pd.notna(z) else np.nan)
    raw_y = df["tar_pos"].apply(lambda z: np.imag(z) if pd.notna(z) else np.nan)

    # Change sign of x position for ipsi values (they should be negative)
    abs_x = raw_x.abs()
    side_sign = df["recorded_side"].map(
        {
            "left": 1.0,
            "right": -1.0,
        }
    )
    df["space_x"] = side_sign * raw_x

    # Raw fixation coordinates
    fix_y = df["fix_pos"].apply(lambda z: np.imag(z) if pd.notna(z) else np.nan)
    df["space_y"] = raw_y - fix_y

    df.loc[df["space_x"].abs() < 1e-3, "space_x"] = 0.0
    df.loc[df["space_y"].abs() < 1e-3, "space_y"] = 0.0

    df["space_x"] = df["space_x"].round(2)
    df["space_y"] = df["space_y"].round(2)

    return df


def add_effector_masks(
    df,
    *,
    time_col="analysis_time",
):
    """
    Add masks used only for effector/action regressors.
    """
    df = df.copy()

    required_cols = [time_col, "t_cue", "t_mov", "t_mov_end"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eff_plan_mask = []
    eff_mov_mask = []

    for _, row in df.iterrows():
        t = np.asarray(row[time_col], dtype=float)

        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            eff_plan_mask.append(np.full(0, np.nan))
            eff_mov_mask.append(np.full(0, np.nan))
            continue

        t_cue = float(row["t_cue"])
        t_mov = float(row["t_mov"])
        t_mov_end = float(row["t_mov_end"])

        if not np.all(np.isfinite([t_cue, t_mov, t_mov_end])):
            eff_plan_mask.append(np.zeros_like(t, dtype=float))
            eff_mov_mask.append(np.zeros_like(t, dtype=float))
            continue

        eff_plan_mask.append(((t >= t_cue + 0.2) & (t < t_mov)).astype(float))
        eff_mov_mask.append(((t >= t_mov) & (t < t_mov_end)).astype(float))

    df["eff_plan_mask"] = eff_plan_mask
    df["eff_mov_mask"] = eff_mov_mask

    return df


def add_effector_masks_mov(
    df,
    *,
    time_col="analysis_time",
):
    """
    Add masks used only for effector/action regressors.
    """
    df = df.copy()

    required_cols = [time_col, "t_mov", "t_mov_end"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eff_mov_mask = []

    for _, row in df.iterrows():
        t = np.asarray(row[time_col], dtype=float)

        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            eff_mov_mask.append(np.full(0, np.nan))
            continue

        t_mov = float(row["t_mov"])
        t_mov_end = float(row["t_mov_end"])
        if not np.all(np.isfinite([t_mov, t_mov_end])):
            eff_mov_mask.append(np.zeros_like(t, dtype=float))
            continue

        eff_mov_mask.append(((t >= t_mov) & (t < t_mov_end)).astype(float))

    df["eff_mov_mask"] = eff_mov_mask

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
    mask_col,
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

    mask_col : str
        Column containing 1D mask arrays.

    suffix : str or None
        If None, overwrite the original columns.
        If given, save as f"{reg}{suffix}".
    """
    df = df.copy()

    required_cols = list(regressors) + [mask_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for masking: {missing}")

    for reg in regressors:
        out_col = reg if suffix is None else f"{reg}{suffix}"

        masked = []
        for _, row in df.iterrows():
            mask = np.asarray(row[mask_col], dtype=float)
            value = float(row[reg])
            masked.append(value * mask)

        df[out_col] = masked

    return df


def filter_units_by_min_condition_trials(
    df,
    *,
    unit_col="unit_ID",
    condition_cols=("effector", "space_x", "space_y"),
    min_trials_per_condition=5,
    counts_out_path=None,
):
    """
    Keep only units having at least `min_trials_per_condition` trials
    in every observed condition.

    Returns
    -------
    filtered_df : pd.DataFrame
        Input dataframe restricted to qualifying units.

    trials_per_condition : pd.DataFrame
        Trial counts for every unit x condition combination. Missing
        combinations appear with n_rows=0.

    good_units : pd.DataFrame
        Unit identifiers that passed the criterion.
    """
    condition_cols = list(condition_cols)

    required_cols = [unit_col, *condition_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if min_trials_per_condition < 1:
        raise ValueError("min_trials_per_condition must be at least 1.")

    # Conditions that exist somewhere in the dataset.
    valid_conditions = (
        df[condition_cols].dropna().drop_duplicates().reset_index(drop=True)
    )
    units = df[[unit_col]].drop_duplicates().reset_index(drop=True)
    if valid_conditions.empty:
        raise ValueError("No valid conditions were found.")
    if units.empty:
        raise ValueError("No units were found.")

    # Every unit is required to possess every observed condition.
    required_counts = units.merge(valid_conditions, how="cross")

    observed_counts = (
        df.dropna(subset=condition_cols)
        .groupby(
            [unit_col, *condition_cols],
            observed=True,
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )

    trials_per_condition = required_counts.merge(
        observed_counts,
        on=[unit_col, *condition_cols],
        how="left",
    )

    trials_per_condition["n_rows"] = (
        trials_per_condition["n_rows"].fillna(0).astype(int)
    )

    good_units = (
        trials_per_condition.groupby(unit_col)["n_rows"]
        .min()
        .loc[lambda counts: counts >= min_trials_per_condition]
        .index
    )

    filtered_df = df[df[unit_col].isin(good_units)].reset_index(drop=True)

    if counts_out_path is not None:
        counts_out_path = Path(counts_out_path)
        counts_out_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        trials_per_condition.to_csv(
            counts_out_path,
            index=False,
        )

    print(f"Units before condition filtering: {len(units)}")
    print(f"Units after condition filtering: {len(good_units)}")

    return filtered_df, trials_per_condition, good_units


def oversample_trials_within_unit_condition(
    df,
    *,
    unit_col="unit_ID",
    condition_cols=("effector", "space_x", "space_y"),
    random_state=0,
):
    """
    Balance condition counts separately within each unit.

    For every unit, smaller condition cells are sampled with replacement
    until they equal that unit's largest condition cell.

    Different units may still have different total row counts.
    """
    condition_cols = list(condition_cols)
    group_cols = [unit_col, *condition_cols]

    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing grouping columns: {missing_cols}")
    if df.empty:
        return df.copy()
    if df[group_cols].isna().any().any():
        raise ValueError(
            "Unit and condition columns must not contain "
            "missing values before oversampling."
        )

    rng = np.random.default_rng(random_state)
    balanced_groups = []

    for _, unit_df in df.groupby(
        unit_col,
        sort=False,
        observed=True,
    ):
        condition_groups = list(
            unit_df.groupby(
                condition_cols,
                sort=False,
                observed=True,
            )
        )
        target_count = max(len(condition_df) for _, condition_df in condition_groups)

        for _, condition_df in condition_groups:
            balanced_groups.append(condition_df.copy())
            n_extra = target_count - len(condition_df)
            if n_extra > 0:
                sampled_positions = rng.choice(
                    len(condition_df),
                    size=n_extra,
                    replace=True,
                )
                balanced_groups.append(condition_df.iloc[sampled_positions].copy())

    return pd.concat(balanced_groups, ignore_index=True)


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
    regressors,
    *,
    unit_col="unit_ID",
    rate_col="analysis_rate",
    time_col="analysis_time",
):
    """
    Fit one multilinear regression independently for every unit.

    Each regressor may be either:

    - array-valued: one time series per trial
    - scalar-valued: one value per trial, repeated across time

    Returns
    -------
    axes_raw : ndarray
        Shape: n_units x n_regressors.

    axes_ortho : ndarray
        Orthogonalized axes with shape n_units x n_regressors.

    units_used : list
        Unit IDs in the same order as the rows of the axis matrices.

    task_regressors : tuple
        Regressor names in the same order as the axis columns.
    """
    df = df.copy()
    task_regressors = tuple(regressors)
    if not task_regressors:
        raise ValueError("At least one regressor is required.")

    missing_cols = [
        col
        for col in (
            unit_col,
            *task_regressors,
            rate_col,
            time_col,
        )
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    betas = []
    units_used = []

    # ------------------------------------------------------------
    # Fit one regression per unit
    # ------------------------------------------------------------
    for unit, unit_df in df.groupby(
        unit_col,
        sort=True,
        observed=True,
    ):
        required_cols = [
            *task_regressors,
            rate_col,
            time_col,
        ]

        unit_df = unit_df.dropna(subset=required_cols).copy()

        if unit_df.empty:
            continue

        # ------------------------------------------------------------
        # Response matrix: trials x time
        # ------------------------------------------------------------
        rates = np.stack(unit_df[rate_col].to_numpy()).astype(float)
        if rates.ndim != 2:
            continue
        if not np.all(np.isfinite(rates)):
            continue

        n_trials, n_time = rates.shape
        y = rates.reshape(-1)
        time = np.asarray(
            unit_df[time_col].iloc[0],
            dtype=float,
        )
        if time.ndim != 1 or time.size != n_time or not np.all(np.isfinite(time)):
            continue

        # ------------------------------------------------------------
        # Design matrix
        # ------------------------------------------------------------
        X_cols = []

        for regressor in task_regressors:
            first_value = unit_df[regressor].iloc[0]

            if isinstance(
                first_value,
                (np.ndarray, list, tuple),
            ):
                # Time-dependent regressor.
                X_reg = np.stack(unit_df[regressor].to_numpy()).astype(float)
                if X_reg.shape != rates.shape:
                    raise ValueError(
                        f"Regressor {regressor!r} has shape "
                        f"{X_reg.shape}, but rates have shape "
                        f"{rates.shape}."
                    )
                if not np.all(np.isfinite(X_reg)):
                    raise ValueError(
                        f"Regressor {regressor!r} contains " "non-finite values."
                    )
                X_cols.append(X_reg.reshape(-1))

            else:
                # Scalar trial-level regressor.
                values = unit_df[regressor].to_numpy(dtype=float)

                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Regressor {regressor!r} contains " "non-finite values."
                    )

                X_cols.append(np.repeat(values, n_time))

        X = np.column_stack(X_cols)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Design matrix has {X.shape[0]} rows, "
                f"but response has {y.shape[0]} rows."
            )

        if not np.all(np.isfinite(X)):
            continue

        # ------------------------------------------------------------
        # Fit regression for this unit
        # ------------------------------------------------------------
        model = LinearRegression(
            fit_intercept=True,
        )
        model.fit(X, y)

        beta = np.asarray(
            model.coef_,
            dtype=float,
        )
        betas.append(beta)
        units_used.append(unit)

    if not betas:
        raise ValueError("No units were successfully fitted.")

    axes_raw = np.stack(
        betas,
        axis=0,
    )

    axes_ortho = lowdin_orthogonalization(axes_raw)

    return (
        axes_raw,
        axes_ortho,
        units_used,
        task_regressors,
    )


def compute_tdr_subspace_residuals(
    condition_trajectories,
    condition_projections,
    axes_ortho,
):
    """
    Reconstruct condition trajectories from the TDR subspace.

    Returns
    -------
    observed : dict
        Condition -> observed activity, n_units x n_time.

    reconstructed : dict
        Condition -> TDR-reconstructed activity, n_units x n_time.

    residuals : dict
        Condition -> observed minus reconstructed activity,
        n_units x n_time.
    """
    axes_ortho = np.asarray(axes_ortho, dtype=float)

    if axes_ortho.ndim != 2:
        raise ValueError("axes_ortho must have shape n_units x n_axes.")

    if not np.all(np.isfinite(axes_ortho)):
        raise ValueError("axes_ortho contains non-finite values.")

    n_units, n_axes = axes_ortho.shape

    trajectory_conditions = set(condition_trajectories)
    projection_conditions = set(condition_projections)

    if trajectory_conditions != projection_conditions:
        missing_projections = trajectory_conditions - projection_conditions
        missing_trajectories = projection_conditions - trajectory_conditions

        raise ValueError(
            "Condition mismatch between trajectories and projections. "
            f"Missing projections: {sorted(missing_projections)}; "
            f"missing trajectories: {sorted(missing_trajectories)}."
        )

    observed = {}
    reconstructed = {}
    residuals = {}

    for condition in condition_trajectories:
        condition_observed = np.asarray(
            condition_trajectories[condition],
            dtype=float,
        )

        condition_projection = np.asarray(
            condition_projections[condition],
            dtype=float,
        )

        if condition_observed.ndim != 2:
            raise ValueError(
                f"Condition {condition!r}: observed activity must "
                "have shape n_units x n_time."
            )

        if condition_projection.ndim != 2:
            raise ValueError(
                f"Condition {condition!r}: projection must "
                "have shape n_axes x n_time."
            )

        if condition_observed.shape[0] != n_units:
            raise ValueError(
                f"Condition {condition!r}: observed activity has "
                f"{condition_observed.shape[0]} units; expected "
                f"{n_units}."
            )

        if condition_projection.shape[0] != n_axes:
            raise ValueError(
                f"Condition {condition!r}: projection has "
                f"{condition_projection.shape[0]} axes; expected "
                f"{n_axes}."
            )

        if condition_observed.shape[1] != condition_projection.shape[1]:
            raise ValueError(
                f"Condition {condition!r}: observed and projected "
                "trajectories have different time lengths."
            )

        if not np.all(np.isfinite(condition_observed)):
            raise ValueError(
                f"Condition {condition!r}: observed activity "
                "contains non-finite values."
            )

        if not np.all(np.isfinite(condition_projection)):
            raise ValueError(
                f"Condition {condition!r}: projection contains " "non-finite values."
            )

        condition_reconstructed = axes_ortho @ condition_projection
        condition_residuals = condition_observed - condition_reconstructed

        observed[condition] = condition_observed
        reconstructed[condition] = condition_reconstructed
        residuals[condition] = condition_residuals

    return observed, reconstructed, residuals


def stack_curve_repeats(curve_repeats, cond_order):
    """
    Average one-dimensional time-resolved curves across repetitions.

    Each curve_repeats element is:
        condition -> array of shape (n_time,)

    Returns
    -------
    mean_curves : dict
        condition -> mean curve, shape (n_time,)

    sd_curves : dict
        condition -> across-repeat SD, shape (n_time,)
    """
    mean_curves = {}
    sd_curves = {}

    for condition in cond_order:
        curves = np.stack(
            [repeat[condition] for repeat in curve_repeats],
            axis=0,
        )  # n_repeats x n_time

        mean_curves[condition] = np.nanmean(curves, axis=0)

        if curves.shape[0] > 1:
            sd_curves[condition] = np.nanstd(curves, axis=0, ddof=1)
        else:
            sd_curves[condition] = np.zeros_like(curves[0])

    return mean_curves, sd_curves


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
    center_across_conditions=True,
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
        # ------------------------------------------------------------
        # Condition-averaged population activity at this time point
        # Shape: units x conditions
        # ------------------------------------------------------------
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
        # Raw variance explained by each axis
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

            rows.append(
                {
                    "time": t,
                    "axis": axis_name,
                    "variance_explained": variance_explained,
                }
            )
    return pd.DataFrame(rows)

    """
    Plot real axis separation against shuffled null distribution.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(analysis_time, dtype=float)
    real_sep = np.asarray(real_sep, dtype=float)
    null_sep = np.asarray(null_sep, dtype=float)

    null_mean = np.nanmean(null_sep, axis=0)
    null_lo = np.nanpercentile(null_sep, 2.5, axis=0)
    null_hi = np.nanpercentile(null_sep, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    ax.fill_between(
        t,
        null_lo,
        null_hi,
        alpha=0.25,
        linewidth=0,
        label="Shuffle 95% interval",
    )

    ax.plot(
        t,
        null_mean,
        lw=1.5,
        linestyle="--",
        label="Shuffle mean",
    )

    ax.plot(
        t,
        real_sep,
        lw=2.2,
        label="Real separation",
    )

    if p_values is not None:
        sig = np.asarray(p_values) < alpha

        if np.any(sig):
            y_sig = np.nanmax([np.nanmax(real_sep), np.nanmax(null_hi)])
            y_sig = y_sig + 0.05 * np.abs(y_sig)

            ax.plot(
                t[sig],
                np.full(np.sum(sig), y_sig),
                linestyle="None",
                marker=".",
                markersize=3,
                label=f"p < {alpha}",
            )

    if event_times is not None:
        if event_labels is None:
            event_labels = [None] * len(event_times)

        for x, label in zip(event_times, event_labels):
            ax.axvline(
                x,
                color="k",
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def shuffle_regressor_columns_within_unit(
    df,
    *,
    regressors_to_shuffle,
    unit_cols=("unit_ID",),
    random_state=0,
):
    """
    Shuffle selected regressor columns within each unit.

    Neural activity and condition labels are unchanged.
    Only the relationship between task regressors and neural activity is destroyed.

    This works for both scalar regressors and array-valued regressors.
    """
    rng = np.random.default_rng(random_state)
    df_shuf = df.copy()

    for _, unit_df in df_shuf.groupby(list(unit_cols), sort=False):
        idx = unit_df.index.to_numpy()

        if len(idx) <= 1:
            continue

        # Use one permutation for all shuffled regressors.
        # This preserves correlations among the shuffled task regressors.
        perm = rng.permutation(len(idx))

        for reg in regressors_to_shuffle:
            values = df_shuf.loc[idx, reg].to_numpy(dtype=object)
            shuffled_values = values[perm]

            df_shuf.loc[idx, reg] = pd.Series(
                list(shuffled_values),
                index=idx,
                dtype=object,
            )

    return df_shuf


def add_target_y_position_label(
    df,
    *,
    y_col="space_y",
    out_col="target_y_position",
    eps=1e-3,
):
    """
    Add a binary up/down target-position label from vertical target position.

    up:
        target is above fixation / center.

    down:
        target is below fixation / center.

    Values close to zero are set to NaN, because they are neither up nor down.
    """
    df = df.copy()

    if y_col not in df.columns:
        raise ValueError(f"Missing required column: {y_col}")

    y = df[y_col].to_numpy(dtype=float)

    df[out_col] = pd.Series(np.nan, index=df.index, dtype="object")
    df.loc[y > eps, out_col] = "up"
    df.loc[y < -eps, out_col] = "down"

    return df


def train_test_split_within_unit_condition(
    df,
    *,
    unit_col="unit_ID",
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    test_frac=0.5,
    min_train_trials=2,
    min_test_trials=2,
    random_state=0,
):
    """
    Split rows into train/test separately within each unit x condition cell.

    This ensures that:
        - every unit contributes train and test trials
        - every condition is represented in train and test
        - TDR axes are fit only on train trials
        - test condition averages are independent of axis fitting
    """
    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    group_cols = [unit_col] + list(condition_cols)

    for _, group_df in df.groupby(group_cols, sort=False):
        idx = group_df.index.to_numpy()
        n = len(idx)

        n_test = int(np.floor(test_frac * n))
        n_test = max(min_test_trials, n_test)
        n_test = min(n_test, n - min_train_trials)

        if n_test < min_test_trials or (n - n_test) < min_train_trials:
            # Skip this unit-condition cell if it cannot support the split.
            continue

        shuffled = rng.permutation(idx)

        test_idx = shuffled[:n_test]
        train_idx = shuffled[n_test:]

        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    train_df = df.loc[train_indices].copy().reset_index(drop=True)
    test_df = df.loc[test_indices].copy().reset_index(drop=True)

    # Keep only units that survived in both train and test.
    train_units = set(train_df[unit_col].unique())
    test_units = set(test_df[unit_col].unique())
    common_units = sorted(train_units & test_units)

    train_df = train_df[train_df[unit_col].isin(common_units)].reset_index(drop=True)
    test_df = test_df[test_df[unit_col].isin(common_units)].reset_index(drop=True)

    print("\nTrain/test split:")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Test rows:  {len(test_df)}")
    print(f"  Common units: {len(common_units)}")

    return train_df, test_df


def oversample_units_to_equal_trials(
    df,
    *,
    unit_col="unit_ID",
    target_n_trials=None,
    random_state=0,
):
    """Randomly oversample units to an equal number of rows.

    Every original row is retained. For units with fewer than the target
    number of trials, the missing rows are sampled from that unit with
    replacement. By default, the target is the largest unit trial count in
    the supplied dataframe.

    This function should be applied to training data only, after the original
    trials have been divided into train and test partitions.
    """
    if unit_col not in df.columns:
        raise ValueError(f"Missing required unit column: {unit_col!r}")
    if df.empty:
        raise ValueError("Cannot oversample an empty dataframe.")

    counts_before = df.groupby(unit_col, sort=True).size().rename("n_trials")

    if target_n_trials is None:
        target_n_trials = int(counts_before.max())
    else:
        target_n_trials = int(target_n_trials)

    if target_n_trials < int(counts_before.max()):
        raise ValueError(
            "target_n_trials must be at least the largest existing unit "
            f"count ({int(counts_before.max())}); otherwise this would also "
            "perform subsampling."
        )

    rng = np.random.default_rng(random_state)
    sampled_groups = []
    for _, unit_df in df.groupby(unit_col, sort=True):
        n_extra = target_n_trials - len(unit_df)
        if n_extra > 0:
            extra_indices = rng.choice(
                unit_df.index.to_numpy(),
                size=n_extra,
                replace=True,
            )
            extra_df = df.loc[extra_indices].copy()
            unit_df = pd.concat([unit_df, extra_df], ignore_index=False)

        sampled_groups.append(unit_df)
    oversampled_df = pd.concat(sampled_groups, ignore_index=True)
    counts_after = oversampled_df.groupby(unit_col).size()

    if not (counts_after == target_n_trials).all():
        raise RuntimeError("Oversampling failed to equalize unit trial counts.")

    print("\nTraining-unit oversampling:")
    print(f"  Units: {len(counts_before)}")
    print(
        f"  Trials per unit before: "
        f"min={int(counts_before.min())}, "
        f"median={counts_before.median():.1f}, "
        f"max={int(counts_before.max())}"
    )
    print(f"  Trials per unit after:  {target_n_trials}")
    print(f"  Training rows before:   {len(df)}")
    print(f"  Training rows after:    {len(oversampled_df)}")

    return oversampled_df


def stack_projection_repeats(
    projection_repeats,
    cond_order,
    axis_names,
):
    """
    Average projected trajectories across repetitions.

    With one repetition, the mean is still defined, but an
    across-repeat SD is not. A zero array is returned as a
    shape-compatible placeholder.
    """
    mean_proj = {}
    sd_proj = {}

    for cond in cond_order:
        arr = np.stack(
            [rep[cond] for rep in projection_repeats if cond in rep],
            axis=0,
        )
        # n_repeats x n_axes x n_time

        mean_proj[cond] = np.nanmean(arr, axis=0)

        if arr.shape[0] > 1:
            sd_proj[cond] = np.nanstd(
                arr,
                axis=0,
                ddof=1,
            )
        else:
            sd_proj[cond] = np.zeros_like(mean_proj[cond])

    return mean_proj, sd_proj


def bootstrap_resample_units(
    df,
    *,
    unit_col="unit_ID",
    boot_unit_col="bootstrap_unit_ID",
    random_state=0,
):
    """
    Resample units with replacement.

    Important:
        If the same unit is sampled multiple times, each copy is given
        a new bootstrap unit ID. This lets the population contain duplicate
        units, as required for unit bootstrap.
    """
    rng = np.random.default_rng(random_state)

    units = np.asarray(sorted(df[unit_col].dropna().unique()))
    n_units = len(units)

    sampled_units = rng.choice(units, size=n_units, replace=True)

    boot_dfs = []

    for boot_idx, unit in enumerate(sampled_units):
        unit_df = df[df[unit_col] == unit].copy()

        # Give each sampled copy a unique unit identity
        unit_df[boot_unit_col] = f"boot{boot_idx:04d}_unit{unit}"

        boot_dfs.append(unit_df)

    boot_df = pd.concat(boot_dfs, ignore_index=True)

    return boot_df, sampled_units


def stack_projection_bootstraps(
    projection_bootstraps,
    cond_order,
):
    """
    Convert bootstrap projection dictionaries into mean and 95% CI.

    projection_bootstraps[bootstrap][condition] = n_axes x n_time

    Returns:
        mean_proj[cond]
        lower_proj[cond]
        upper_proj[cond]
    """
    mean_proj = {}
    lower_proj = {}
    upper_proj = {}

    for cond in cond_order:
        arr = np.stack(
            [boot[cond] for boot in projection_bootstraps if cond in boot],
            axis=0,
        )
        # arr shape: n_bootstrap x n_axes x n_time

        mean_proj[cond] = np.nanmean(arr, axis=0)
        lower_proj[cond] = np.nanpercentile(arr, 2.5, axis=0)
        upper_proj[cond] = np.nanpercentile(arr, 97.5, axis=0)

    return mean_proj, lower_proj, upper_proj


def condition_mean_subspace_variance(
    observed,
    reconstructed,
    residuals,
    analysis_time,
):
    rows = []
    for cond in observed.keys():
        # Variance across units at every time point.
        observed_variance = np.var(observed[cond], axis=0, ddof=1)
        reconstructed_variance = np.var(reconstructed[cond], axis=0, ddof=1)
        residual_variance = np.var(residuals[cond], axis=0, ddof=1)

        # Population energy has an exact additive decomposition for an
        # orthogonal projection; ordinary across-unit variance need not.
        observed_energy = np.mean(observed[cond] ** 2, axis=0)
        reconstructed_energy = np.mean(reconstructed[cond] ** 2, axis=0)
        residual_energy = np.mean(residuals[cond] ** 2, axis=0)

        # Conventional reconstruction R^2 across units. It is deliberately
        # kept separate from the energy fraction and may be negative.
        sse = np.sum(residuals[cond] ** 2, axis=0)
        centered = observed[cond] - np.mean(
            observed[cond],
            axis=0,
            keepdims=True,
        )
        sst = np.sum(centered**2, axis=0)
        reconstruction_r2 = 1.0 - np.divide(
            sse,
            sst,
            out=np.full(analysis_time.size, np.nan),
            where=sst > 0,
        )

        effector, space_x, space_y = cond
        for t_idx, t in enumerate(analysis_time):
            rows.append(
                {
                    "effector": effector,
                    "space_x": space_x,
                    "space_y": space_y,
                    "time": t,
                    "observed_variance": observed_variance[t_idx],
                    "reconstructed_variance": reconstructed_variance[t_idx],
                    "residual_variance": residual_variance[t_idx],
                    "observed_energy": observed_energy[t_idx],
                    "reconstructed_energy": reconstructed_energy[t_idx],
                    "residual_energy": residual_energy[t_idx],
                    "reconstruction_r2": reconstruction_r2[t_idx],
                }
            )

    return pd.DataFrame(rows)
