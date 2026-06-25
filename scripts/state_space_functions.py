import numpy as np
from itertools import product
import pandas as pd


def stitch_sdfs(
    row,
    *,
    period1=(-0.5, 0.8),
    period2=(-0.8, 0.5),
    col1=None,
    col2=None,
    time_col1=None,
    time_col2=None,
):
    """
    Stitch two event-aligned SDF segments into one vector.

    Example use:
        period1 = cue-aligned window, e.g. (-0.5, 0.8)
        period2 = movement-aligned window, e.g. (-0.8, 0.5)

    The output is:
        [SDF during period1, SDF during period2]

    Important:
        The two periods are concatenated for analysis/plotting convenience.
        They are not assumed to be continuous in real time.
    """
    t_1 = np.asarray(row[time_col1], dtype=float)
    r_1 = np.asarray(row[col1], dtype=float)

    t_2 = np.asarray(row[time_col2], dtype=float)
    r_2 = np.asarray(row[col2], dtype=float)

    mask1 = (t_1 >= period1[0]) & (t_1 <= period1[1])
    mask2 = (t_2 >= period2[0]) & (t_2 <= period2[1])

    stitched_rate = np.concatenate([r_1[mask1], r_2[mask2]])

    return stitched_rate


def stitch_time(
    row, *, period1=(-0.5, 0.8), period2=(-0.8, 0.5), time_col1=None, time_col2=None
):
    """
    Create an artificial time axis for stitched SDFs.

    The first period keeps its original event-aligned time.
    The second period is shifted so that it starts immediately after period1.

    Example:
        period1: cue-aligned time from -0.5 to 0.8 s
        period2: movement-aligned time from -0.8 to 0.5 s

    The returned stitched time is useful for plotting and regression, but it
    should not be interpreted as a continuous real trial time axis.

    Returns
    -------
    stitched_t : array
        Artificial stitched time axis.
    stitch_x : float
        Boundary between period1 and period2.
    t_1 : array
        Selected time points from period1.
    t_2_shifted : array
        Selected and shifted time points from period2.
    """
    t_1 = np.asarray(row[time_col1], dtype=float)
    t_2 = np.asarray(row[time_col2], dtype=float)

    mask1 = (t_1 >= period1[0]) & (t_1 <= period1[1])
    mask2 = (t_2 >= period2[0]) & (t_2 <= period2[1])

    t_1 = t_1[mask1]
    t_2 = t_2[mask2]

    if t_1.size == 0:
        raise ValueError(f"No time points found in period1={period1}")
    if t_2.size == 0:
        raise ValueError(f"No time points found in period2={period2}")
    if t_1.size < 2:
        raise ValueError("period1 needs at least two time points to estimate dt.")

    # Estimate time-bin spacing from the first segment
    dt = np.nanmedian(np.diff(t_1))

    # Shift period2 so that it starts right after period1
    t_2_shifted = t_2 - t_2[0] + t_1[-1] + dt

    # Concatenate the original period1 time and shifted period2 time
    stitched_t = np.concatenate([t_1, t_2_shifted])
    stitch_x = t_1[-1]
    return stitched_t, stitch_x, t_1, t_2_shifted


def condition_mean_population(
    df,
    units,
    *,
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    unit_cols=("unit_ID",),
    rate_col="analysis_rate",
):
    """
    Builds condition-averaged pseudo-population trajectories.

    For each condition:
        1. average trials within each unit
        2. stack units into a population matrix

    Output:
        dict mapping condition tuple -> array of shape n_units x n_time
    """
    out = {}
    df = df.dropna(subset=list(condition_cols) + [rate_col]).copy()
    grouped = df.groupby(list(condition_cols), sort=True)

    # Get time length from first valid stitched_rate
    example_rate = np.asarray(df[rate_col].iloc[0], dtype=float)
    n_time = example_rate.size

    for cond, cond_df in grouped:
        pop = []
        for unit in units:
            unit_df = cond_df.copy()
            for col, val in zip(unit_cols, unit):
                unit_df = unit_df[unit_df[col] == val]

            # If this unit has no trials in this condition,
            # fill with NaNs instead of crashing.
            if len(unit_df) == 0:
                pop.append(np.full(n_time, np.nan))
                continue
            rates = np.stack(unit_df[rate_col].to_numpy()).astype(float)

            # Mean across trials for this unit-condition
            pop.append(np.nanmean(rates, axis=0))
        out[cond] = np.stack(pop, axis=0)
    return out


def count_rows_per_unit_condition(
    df,
    *,
    unit_cols=("session", "unit_ID"),
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    condition_levels=None,
):
    """
    Count rows/trials for every unit x condition combination.

    Includes rows with n_rows = 0 when a unit has no trials
    for a condition.

    Works for:
        condition_cols=("effector",)
        condition_cols=("effector", "reach_hand", "target_hemifield")
        etc.
    """
    if condition_levels is None:
        default_levels = {
            "effector": ("reach", "saccade"),
            "reach_hand": ("ipsi", "contra"),
            "target_hemifield": ("ipsi", "contra"),
            "space": ("ipsi", "contra"),
        }

        condition_levels = {col: default_levels[col] for col in condition_cols}

    # Existing unit list
    units = df[list(unit_cols)].drop_duplicates().sort_values(list(unit_cols))

    # All requested condition combinations
    conditions = pd.DataFrame(
        list(product(*[condition_levels[col] for col in condition_cols])),
        columns=list(condition_cols),
    )

    # Cartesian product: every unit x every requested condition
    full_index = (
        units.assign(_key=1)
        .merge(conditions.assign(_key=1), on="_key")
        .drop(columns="_key")
    )

    # Actual counts
    counts = (
        df.groupby(list(unit_cols) + list(condition_cols))
        .size()
        .rename("n_rows")
        .reset_index()
    )

    # Fill missing unit-condition rows with 0
    counts_long = full_index.merge(
        counts,
        on=list(unit_cols) + list(condition_cols),
        how="left",
    ).fillna({"n_rows": 0})

    counts_long["n_rows"] = counts_long["n_rows"].astype(int)

    return counts_long


def project_trajectories(condition_pop, axes_q, regressors):
    """
    Project condition-averaged population trajectories onto axes.

    condition_pop[cond]: units x time
    axes_q: units x n_axes

    Returns:
        projections[cond]: n_axes x time
    """
    projections = {}
    for cond, R in condition_pop.items():
        projections[cond] = axes_q.T @ R

    axis_names = list(regressors)
    return projections, axis_names


def pca_denoise_condition_population(
    condition_pop,
    *,
    explained_variance=0.95,
    max_components=None,
):
    """
    PCA-denoise condition-averaged population trajectories.

    Parameters
    ----------
    condition_pop : dict
        Maps condition tuple -> array of shape units x time.
    explained_variance : float
        Cumulative variance threshold used to choose the number of PCs.
        Example: 0.95 keeps the smallest number of PCs explaining >=95% variance.
    max_components : int or None
        Optional upper bound on the number of PCs.

    Returns
    -------
    denoised_pop : dict
        Same structure as condition_pop, but reconstructed from retained PCs.
    info : dict
        PCA diagnostics: n_components, explained_variance_ratio,
        cumulative_explained_variance, selected_explained_variance.

    Notes
    -----
    PCA is fit across all condition-time points, with units as features:
        observations = condition-time samples
        features = units

    This is appropriate for denoising the condition-averaged pseudo-population
    trajectories used for plotting/projection. It does not mix time bins within
    a trajectory; it only reconstructs population activity through a lower-
    dimensional neural subspace.
    """
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    conds = list(condition_pop.keys())
    if len(conds) == 0:
        raise ValueError("condition_pop is empty.")

    shapes = [np.asarray(condition_pop[c], dtype=float).shape for c in conds]
    if len(set(shapes)) != 1:
        raise ValueError(f"All condition arrays must have same shape. Got {shapes}.")

    n_units, n_time = shapes[0]

    # Stack as observations x units: (conditions*time) x units
    X = np.concatenate(
        [np.asarray(condition_pop[c], dtype=float).T for c in conds],
        axis=0,
    )

    if not np.all(np.isfinite(X)):
        raise ValueError("PCA denoising requires finite values in condition_pop.")

    # Center each unit before PCA.
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # SVD PCA. Xc = U S Vt. PCs are rows of Vt.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    eigvals = (S**2) / max(Xc.shape[0] - 1, 1)
    total_var = eigvals.sum()
    if total_var <= 0 or not np.isfinite(total_var):
        raise ValueError(
            "Cannot run PCA denoising because total variance is zero/non-finite."
        )

    explained_ratio = eigvals / total_var
    cumulative = np.cumsum(explained_ratio)

    n_components = int(np.searchsorted(cumulative, explained_variance) + 1)
    if max_components is not None:
        n_components = min(n_components, int(max_components))
    n_components = max(1, min(n_components, Vt.shape[0]))

    # Reconstruct using retained PCs.
    scores = Xc @ Vt[:n_components].T
    X_hat = scores @ Vt[:n_components] + mu

    denoised_pop = {}
    start = 0
    for cond in conds:
        stop = start + n_time
        denoised_pop[cond] = X_hat[start:stop].T
        start = stop

    info = {
        "n_components": n_components,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance": cumulative,
        "selected_explained_variance": float(cumulative[n_components - 1]),
    }
    return denoised_pop, info


def trajectory_euclidean_distance(traj1, traj2, *, axis=None, axis_names=None):
    """
    Compute Euclidean distance between two trajectories in TDR space.

    Parameters
    ----------
    traj1, traj2 : array-like
        Trajectories with shape:
            n_axes x n_time

    axis_names : list[str] or None
        Required if axis is not None.

    summary : {"mean", "sum", "max", None}

    Returns
    -------
    dist : float or ndarray
        array of shape n_time
    """
    traj1 = np.asarray(traj1, dtype=float)
    traj2 = np.asarray(traj2, dtype=float)
    if traj1.shape != traj2.shape:
        raise ValueError(
            f"traj1 and traj2 must have same shape. "
            f"Got {traj1.shape} and {traj2.shape}."
        )

    if traj1.ndim != 2:
        raise ValueError(
            f"Trajectories must have shape n_axes x n_time. Got {traj1.shape}."
        )

    diff = traj1 - traj2
    dist_t = np.sqrt(np.nansum(diff**2, axis=0))
    return dist_t


def average_projected_trajectory_by_factor(
    projections,
    *,
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    factor="effector",
    level="saccade",
):
    """
    Average projected trajectories across all conditions matching one factor level.
    Example:
        factor="target_hemifield", level="ipsi"
        averages all trajectories with target_hemifield == "ipsi",
        across effector and reach_hand.

    Returns
    -------
    mean_traj : ndarray
        Mean projected trajectory for this factor level.
    """
    factor_idx = condition_cols.index(factor)
    matched = []
    for cond, traj in projections.items():
        if cond[factor_idx] == level:
            matched.append(np.asarray(traj, dtype=float))
    return np.nanmean(np.stack(matched, axis=0), axis=0)
