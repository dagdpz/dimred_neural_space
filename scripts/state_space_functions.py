import numpy as np


def make_stitched_sdf(
    row,
    *,
    cue_start=-0.5,
    cue_end=0.6,
    mov_start=-0.8,
    mov_end=0.5,
):
    """
    Returns one stitched SDF vector:
        cue-aligned segment + movement-aligned segment

    The two pieces are not truly continuous in real time, but this gives
    a common trial-length vector for TDR projection.
    """
    t_cue = np.asarray(row["sdf_time_cue"], dtype=float)
    r_cue = np.asarray(row["sdf_rate_cue"], dtype=float)

    t_mov = np.asarray(row["sdf_time_mov"], dtype=float)
    r_mov = np.asarray(row["sdf_rate_mov"], dtype=float)

    cue_mask = (t_cue >= cue_start) & (t_cue <= cue_end)
    mov_mask = (t_mov >= mov_start) & (t_mov <= mov_end)

    stitched_rate = np.concatenate([r_cue[cue_mask], r_mov[mov_mask]])

    return stitched_rate


def make_stitched_time(
    example_row, *, cue_start=-0.5, cue_end=0.6, mov_start=-0.8, mov_end=0.5
):
    """
    Creates artificial plot time.

    The movement segment is shifted to appear after the cue segment.
    Use a visual gap so the plot does not imply true continuity.
    """
    t_cue = np.asarray(example_row["sdf_time_cue"], dtype=float)
    t_mov = np.asarray(example_row["sdf_time_mov"], dtype=float)

    cue_mask = (t_cue >= cue_start) & (t_cue <= cue_end)
    mov_mask = (t_mov >= mov_start) & (t_mov <= mov_end)

    cue_t = t_cue[cue_mask]
    mov_t_raw = t_mov[mov_mask]
    dt = np.nanmedian(np.diff(cue_t))

    mov_t = mov_t_raw - mov_t_raw[0] + cue_t[-1] + dt
    stitched_t = np.concatenate([cue_t, mov_t])
    stitch_x = cue_t[-1]
    return stitched_t, stitch_x, cue_t, mov_t


def condition_mean_population(
    df,
    units,
    *,
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    unit_cols=("session", "unit_ID"),
):
    """
    Builds condition-averaged pseudo-population trajectories.

    Output:
        dict mapping condition tuple -> array of shape units x time
    """
    out = {}
    grouped = df.dropna(subset=list(condition_cols) + ["stitched_rate"]).groupby(
        list(condition_cols)
    )

    for cond, cond_df in grouped:
        pop = []

        for unit in units:
            unit_df = cond_df.copy()
            for col, val in zip(unit_cols, unit):
                unit_df = unit_df[unit_df[col] == val]
            rates = np.stack(unit_df["stitched_rate"].to_numpy())
            pop.append(np.nanmean(rates, axis=0))
        out[cond] = np.stack(pop, axis=0)
    return out


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
