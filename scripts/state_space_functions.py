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
