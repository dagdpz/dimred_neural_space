from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from dPCA import dPCA
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from preprocess import PROCESSED_TRIALS_PATH
from scripts.plotting import *
from scripts.utils import *


def load_processed_trials(path=PROCESSED_TRIALS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python preprocess.py` first.")
    return pd.read_pickle(path)


def add_tdr_regressors(df):
    """
    Binary coding:
        effector: reach=+1, saccade=-1
        target: contra=+1, ipsi=-1
        hand: contra=+1, ipsi=-1
    """
    df = df.copy()

    df["E"] = df["effector"].map({"reach": 1.0, "saccade": -1.0})
    df["T"] = df["target_hemifield"].map({"contra": 1.0, "ipsi": -1.0})
    df["H"] = df["reach_hand"].map({"contra": 1.0, "ipsi": -1.0})

    df["E_T"] = df["E"] * df["T"]
    df["E_H"] = df["E"] * df["H"]
    df["T_H"] = df["T"] * df["H"]

    return df


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
    segment = np.array(["cue"] * len(cue_t) + ["movement"] * len(mov_t))
    stitch_x = cue_t[-1]
    return stitched_t, segment, stitch_x, cue_t, mov_t


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
    regressors=("E", "T", "H"),
):
    """
    Fits one regression per unit.

    For each unit, the dependent variable is mean firing rate in a chosen
    preparatory window. The beta coefficients across units become TDR axes.

    Returns:
        axes_raw: units x regressors beta matrix
        axes_q: orthonormalized TDR axes from QR decomposition
        units: list of unit keys
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

    betas = []

    for unit in units:
        unit_df = df.copy()
        for col, val in zip(unit_cols, unit):
            unit_df = unit_df[unit_df[col] == val]
        unit_df = unit_df.dropna(subset=list(regressors) + ["stitched_rate"])

        rates = np.stack(unit_df["stitched_rate"].to_numpy())
        y = np.nanmean(rates, axis=1)

        X = unit_df[list(regressors)].to_numpy(dtype=float)

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)

        betas.append(model.coef_)

    axes_raw = np.asarray(betas, dtype=float)

    # QR orthogonalization, like standard TDR usage.
    # Columns of Q are orthonormal population axes.
    axes_ortho = lowdin_orthogonalization(axes_raw)

    return axes_raw, axes_ortho, units


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
    Project condition-averaged population trajectories onto TDR axes.

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


def plot_tdr_reach_vs_saccade_3d(
    projections,
    stitched_time,
    axis_names,
    *,
    x_axis="E",
    y_axis="T",
    z_axis="H",
    out_path=Path("plots/tdr/tdr_reach_vs_saccade_3d.html"),
    downsample=2,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_idx = axis_names.index(x_axis)
    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    colors = {
        "reach": "blue",
        "saccade": "orange",
    }

    fig = go.Figure()
    stitched_time = np.asarray(stitched_time)

    for cond, Z in projections.items():
        # cond is now only ("reach",) or ("saccade",)
        effector = cond[0] if isinstance(cond, tuple) else cond

        x = np.asarray(Z[x_idx], dtype=float)
        y = np.asarray(Z[y_idx], dtype=float)
        z = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[finite]
        y = y[finite]
        z = z[finite]
        t = stitched_time[finite]

        x_plot = x[::downsample]
        y_plot = y[::downsample]
        z_plot = z[::downsample]
        t_plot = t[::downsample]

        hover_text = [
            (
                f"{effector}<br>"
                f"time={t_plot[i]:.3f} s<br>"
                f"{x_axis}={x_plot[i]:.3f}<br>"
                f"{y_axis}={y_plot[i]:.3f}<br>"
                f"{z_axis}={z_plot[i]:.3f}"
            )
            for i in range(len(x_plot))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="lines+markers",
                name=effector,
                line=dict(
                    color=colors.get(effector, "gray"),
                    width=5,
                ),
                marker=dict(
                    size=2,
                    color=colors.get(effector, "gray"),
                    opacity=0.6,
                ),
                text=hover_text,
                hoverinfo="text",
                connectgaps=True,
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                name=f"{effector} start",
                marker=dict(
                    size=7,
                    color=colors.get(effector, "gray"),
                    symbol="circle",
                ),
                showlegend=False,
                text=[f"{effector}<br>start"],
                hoverinfo="text",
            )
        )

        # End marker
        fig.add_trace(
            go.Scatter3d(
                x=[x[-1]],
                y=[y[-1]],
                z=[z[-1]],
                mode="markers",
                name=f"{effector} end",
                marker=dict(
                    size=7,
                    color=colors.get(effector, "gray"),
                    symbol="x",
                ),
                showlegend=False,
                text=[f"{effector}<br>end"],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="TDR trajectory: reach vs saccade",
        scene=dict(
            xaxis_title=f"TDR axis: {x_axis}",
            yaxis_title=f"TDR axis: {y_axis}",
            zaxis_title=f"TDR axis: {z_axis}",
        ),
        width=950,
        height=750,
    )

    fig.write_html(out_path)


def plot_tdr_trajectories_3d(
    projections,
    stitched_time,
    axis_names,
    *,
    x_axis="E",
    y_axis="T",
    z_axis="H",
    out_path=Path("plots/tdr/tdr_trajectory_E_T_H.html"),
    downsample=2,
):
    """
    Interactive 3D TDR state-space trajectory using Plotly.

    Each condition trajectory is plotted through time in the space:
        x = effector axis
        y = target axis
        z = hand axis
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_idx = axis_names.index(x_axis)
    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    colors = {
        "reach": "blue",
        "saccade": "orange",
    }

    dash_styles = {
        "ipsi": "dash",
        "contra": "solid",
    }

    fig = go.Figure()
    stitched_time = np.asarray(stitched_time)

    for cond, Z in projections.items():
        effector, hand, target = cond

        x = np.asarray(Z[x_idx], dtype=float)
        y = np.asarray(Z[y_idx], dtype=float)
        z = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[finite]
        y = y[finite]
        z = z[finite]
        t = stitched_time[finite]

        x_plot = x[::downsample]
        y_plot = y[::downsample]
        z_plot = z[::downsample]
        t_plot = t[::downsample]

        label = f"{effector}, hand={hand}, target={target}"
        color = colors.get(effector, "gray")

        hover_text = [
            (
                f"{label}<br>"
                f"time={t_plot[i]:.3f} s<br>"
                f"{x_axis}={x_plot[i]:.3f}<br>"
                f"{y_axis}={y_plot[i]:.3f}<br>"
                f"{z_axis}={z_plot[i]:.3f}"
            )
            for i in range(len(x_plot))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="lines+markers",
                name=label,
                line=dict(
                    color=color,
                    width=5,
                ),
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.6,
                ),
                text=hover_text,
                hoverinfo="text",
                connectgaps=True,
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                marker=dict(
                    size=7,
                    color=color,
                    symbol="circle",
                ),
                name=f"{label} start",
                showlegend=False,
                text=[f"{label}<br>start"],
                hoverinfo="text",
            )
        )

        # End marker
        fig.add_trace(
            go.Scatter3d(
                x=[x[-1]],
                y=[y[-1]],
                z=[z[-1]],
                mode="markers",
                marker=dict(
                    size=7,
                    color=color,
                    symbol="x",
                ),
                name=f"{label} end",
                showlegend=False,
                text=[f"{label}<br>end"],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="3D TDR state-space trajectory",
        scene=dict(
            xaxis_title=f"TDR axis: {x_axis}",
            yaxis_title=f"TDR axis: {y_axis}",
            zaxis_title=f"TDR axis: {z_axis}",
        ),
        width=950,
        height=750,
    )

    fig.write_html(out_path)


def plot_tdr_timecolor_3d(
    projections,
    stitched_time,
    axis_names,
    *,
    x_axis="E",
    y_axis="T",
    z_axis="H",
    out_path=Path("plots/tdr/tdr_reach_saccade_side_by_side_timecolor.html"),
    downsample=2,
):
    """
    Plot reach and saccade trajectories in separate 3D Plotly panels.

    Time is represented by color along each trajectory.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_idx = axis_names.index(x_axis)
    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    stitched_time = np.asarray(stitched_time, dtype=float)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Reach", "Saccade"),
        horizontal_spacing=0.02,
    )

    effector_to_col = {
        "reach": 1,
        "saccade": 2,
    }

    all_x = []
    all_y = []
    all_z = []

    for cond, Z in projections.items():
        effector = cond[0] if isinstance(cond, tuple) else cond

        if effector not in effector_to_col:
            continue

        col = effector_to_col[effector]

        x = np.asarray(Z[x_idx], dtype=float)
        y = np.asarray(Z[y_idx], dtype=float)
        z = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[finite]
        y = y[finite]
        z = z[finite]
        t = stitched_time[finite]

        all_x.append(x[finite])
        all_y.append(y[finite])
        all_z.append(z[finite])

        x_plot = x[::downsample]
        y_plot = y[::downsample]
        z_plot = z[::downsample]
        t_plot = t[::downsample]

        hover_text = [
            (
                f"{effector}<br>"
                f"time={t_plot[i]:.3f} s<br>"
                f"{x_axis}={x_plot[i]:.3f}<br>"
                f"{y_axis}={y_plot[i]:.3f}<br>"
                f"{z_axis}={z_plot[i]:.3f}"
            )
            for i in range(len(x_plot))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="lines+markers",
                name=effector,
                line=dict(
                    color=t_plot,
                    colorscale="Viridis",
                    width=6,
                    cmin=np.nanmin(stitched_time),
                    cmax=np.nanmax(stitched_time),
                ),
                marker=dict(
                    size=3,
                    color=t_plot,
                    colorscale="Viridis",
                    cmin=np.nanmin(stitched_time),
                    cmax=np.nanmax(stitched_time),
                    colorbar=dict(title="Time (s)") if col == 2 else None,
                ),
                text=hover_text,
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[x_plot[0]],
                y=[y_plot[0]],
                z=[z_plot[0]],
                mode="markers",
                marker=dict(
                    size=7,
                    color="black",
                    symbol="circle",
                ),
                name=f"{effector} start",
                text=[f"{effector}<br>start<br>time={t_plot[0]:.3f} s"],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        # End marker
        fig.add_trace(
            go.Scatter3d(
                x=[x_plot[-1]],
                y=[y_plot[-1]],
                z=[z_plot[-1]],
                mode="markers",
                marker=dict(
                    size=7,
                    color="black",
                    symbol="x",
                ),
                name=f"{effector} end",
                text=[f"{effector}<br>end<br>time={t_plot[-1]:.3f} s"],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    def padded_range(v, pad_frac=0.08):
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)
        pad = pad_frac * (vmax - vmin)

        if pad == 0:
            pad = 1.0

        return [vmin - pad, vmax + pad]

    x_range = padded_range(all_x)
    y_range = padded_range(all_y)
    z_range = padded_range(all_z)

    scene_settings = dict(
        xaxis=dict(
            title=f"TDR axis: {x_axis}",
            range=x_range,
        ),
        yaxis=dict(
            title=f"TDR axis: {y_axis}",
            range=y_range,
        ),
        zaxis=dict(
            title=f"TDR axis: {z_axis}",
            range=z_range,
        ),
        aspectmode="cube",
    )

    fig.update_layout(
        title="TDR trajectories: reach vs saccade",
        width=1200,
        height=650,
        scene=scene_settings,
        scene2=scene_settings,
    )

    fig.write_html(out_path)


def main(seed=0, plot_sdf=False):
    """
    Load preprocessed trials, align spikes to cue, compute SDFs, then run TDR.
    """
    rng = np.random.default_rng(seed)
    df = load_processed_trials()

    # ------------------------------------------------------------
    # Align spikes to cue and movement
    # ------------------------------------------------------------
    cue_state = 6
    mov_state = 68
    cue_align = df.apply(
        lambda row: trial_alignment_to_state(row, cue_state),
        axis=1,
    )
    cue_align = cue_align.rename(
        columns={
            "t_state": "t_cue",
            "arrival_times_rel": "arrival_times_cue",
        }
    )
    mov_align = df.apply(
        lambda row: trial_alignment_to_state(row, mov_state),
        axis=1,
    )
    mov_align = mov_align.rename(
        columns={
            "t_state": "t_mov",
            "arrival_times_rel": "arrival_times_mov",
        }
    )
    df = pd.concat([df, cue_align, mov_align], axis=1)

    # ------------------------------------------------------------
    # SDFs
    # ------------------------------------------------------------
    bin_size = 0.001  # 1 ms bins
    sigma = 0.02  # 50 ms Gaussian smoothing
    cue_sdf = df["arrival_times_cue"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=-0.5,
            t_end=0.8,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    mov_sdf = df["arrival_times_mov"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=-0.8,
            t_end=0.5,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    df["sdf_time_cue"] = cue_sdf.apply(lambda x: x[0])
    df["sdf_rate_cue"] = cue_sdf.apply(lambda x: x[1])
    df["sdf_time_mov"] = mov_sdf.apply(lambda x: x[0])
    df["sdf_rate_mov"] = mov_sdf.apply(lambda x: x[1])

    # ------------------------------------------------------------
    # Plot SDFs
    # ------------------------------------------------------------
    if plot_sdf:
        for reach_hand in ("ipsi", "contra"):
            for target_hemifield in ("ipsi", "contra"):
                sub = df[
                    (df["reach_hand"] == reach_hand)
                    & (df["target_hemifield"] == target_hemifield)
                ].dropna(subset=["effector"])
                sdf_dir = Path("plots/sdf")
                plot_effector_sdf(
                    data=sub,
                    reach_hand=reach_hand,
                    target_hemifield=target_hemifield,
                    plots_dir=sdf_dir,
                    analysis_label=f"{target_hemifield}_{reach_hand}",
                )

    # ------------------------------------------------------------
    # Regressors
    # ------------------------------------------------------------
    df = add_tdr_regressors(df)
    df = df.dropna(
        subset=[
            "E",
            "T",
            "H",
            "t_cue",
            "t_mov",
            "sdf_rate_cue",
            "sdf_rate_mov",
        ]
    ).copy()

    # ------------------------------------------------------------
    # Choose cue-end to reduce stitch discontinuity
    # ------------------------------------------------------------
    mov_start = -0.8
    mov_end = 0.5
    cue_start = -0.5
    cue_end = 0.8

    df["stitched_rate"] = df.apply(
        lambda row: make_stitched_sdf(
            row,
            cue_start=cue_start,
            cue_end=cue_end,
            mov_start=mov_start,
            mov_end=mov_end,
        ),
        axis=1,
    )
    stitched_time, stitched_segment, stitch_x, cue_t, mov_t = make_stitched_time(
        df.iloc[0],
        cue_start=cue_start,
        cue_end=cue_end,
        mov_start=mov_start,
        mov_end=mov_end,
    )

    df["stitched_time"] = [stitched_time] * len(df)
    df["stitched_segment"] = [stitched_segment] * len(df)

    # ------------------------------------------------------------
    # Fit TDR axes
    # ------------------------------------------------------------
    main_effect_regressors = ("E", "T", "H")

    axes_raw, axes_ortho, units = fit_tdr_axes(
        df,
        regressors=main_effect_regressors,
    )

    # ------------------------------------------------------------
    # Condition-averaged trajectories
    # ------------------------------------------------------------
    condition_pop = condition_mean_population(
        df,
        units,
        condition_cols=("effector",),
    )

    projections, axis_names = project_trajectories(
        condition_pop,
        axes_ortho,
        main_effect_regressors,
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_tdr_reach_vs_saccade_3d(
        projections,
        stitched_time,
        axis_names,
        x_axis="E",
        y_axis="T",
        z_axis="H",
        out_path=Path("plots/tdr/tdr_reach_vs_saccade_3d.html"),
        downsample=2,
    )

    plot_tdr_timecolor_3d(
        projections,
        stitched_time,
        axis_names,
        x_axis="E",
        y_axis="T",
        z_axis="H",
        out_path=Path("plots/tdr/tdr_reach_saccade_side_by_side_timecolor.html"),
        downsample=2,
    )


if __name__ == "__main__":
    main(plot_sdf=False)
