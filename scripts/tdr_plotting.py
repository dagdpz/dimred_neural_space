from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from itertools import product

from scripts.plotting import *


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

    cue_time = 0.0
    mov_time = 1.6
    cue_idx = np.nanargmin(np.abs(stitched_time - cue_time))
    mov_idx = np.nanargmin(np.abs(stitched_time - mov_time))

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

        # -----------------------------------
        # ADD CUE + GO MARKERS
        # -----------------------------------
        fig.add_trace(
            go.Scatter3d(
                x=[x[cue_idx]],
                y=[y[cue_idx]],
                z=[z[cue_idx]],
                mode="markers",
                marker=dict(size=6, color="orange", symbol="square"),
                name="Cue (0s)",
                showlegend=(col == 1),
                hovertext=[f"Cue: t=0"],
                hoverinfo="text",
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Scatter3d(
                x=[x[mov_idx]],
                y=[y[mov_idx]],
                z=[z[mov_idx]],
                mode="markers",
                marker=dict(size=6, color="red", symbol="square"),
                name="MOV (1.6s)",
                showlegend=(col == 1),
                hovertext=[f"MOV: t=1.6s"],
                hoverinfo="text",
            ),
            row=1,
            col=col,
        )

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

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


def target_pos_label(cond):
    x, y = cond
    return f"x={x:.2f}, y={y:.2f}"


def plot_tdr_axis_timecourse(
    projections,
    analysis_time,
    axis_names,
    *,
    axis,
    cond_order=None,
    cond_label_fn=None,
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    lw=2.0,
    downsample=2,
    out_path=None,
):
    """
    Plot one TDR/regression axis over time.

    Saves one separate file for one axis:
        time vs selected axis projection

    Works with or without interaction axes, because `axis`
    only needs to exist in `axis_names`.
    """
    if axis not in axis_names:
        raise ValueError(f"axis={axis!r} not found in axis_names={axis_names}")

    if out_path is None:
        out_path = Path(f"plots/axis_timecourses/tdr_time_{axis}.png")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(analysis_time, dtype=float)
    axis_idx = axis_names.index(axis)

    conds = list(projections.keys()) if cond_order is None else list(cond_order)

    if cond_label_fn is None:

        def cond_label_fn(cond):
            if isinstance(cond, tuple) and len(cond) == 3:
                effector, hand, target = cond
                return f"{effector}, hand={hand}, target={target}"
            return str(cond)

    label_map = {
        "E": "Effector",
        "T": "Space",
        "H": "Hand",
        "EH": "Effector × Hand",
        "ET": "Effector × Space",
    }

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    # Color cycle for conditions that are not in CONDITION_COLORS
    cmap = plt.get_cmap("tab20")
    fallback_colors = [cmap(i) for i in np.linspace(0, 1, max(len(conds), 2))]

    for i, cond in enumerate(conds):
        Y = np.asarray(projections[cond], dtype=float)
        y = np.asarray(Y[axis_idx, :], dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)

        if isinstance(cond, tuple) and len(cond) == 3:
            color = CONDITION_COLORS.get(cond, fallback_colors[i])
        else:
            color = fallback_colors[i]

        ax.plot(
            t[finite][::downsample],
            y[finite][::downsample],
            color=color,
            lw=lw,
            label=cond_label_fn(cond),
        )

    add_vertical_event_lines(
        ax,
        event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    ax.axhline(0.0, color="0.75", lw=0.8)

    axis_label = label_map.get(axis, axis)

    ax.set_title(f"Condition-averaged projections on {axis_label} axis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Projection on {axis_label}")
    ax.grid(alpha=0.25)

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_all_tdr_axes_timecourses_separate(
    projections,
    analysis_time,
    axis_names,
    *,
    axes_to_plot=None,
    out_dir=Path("plots/axis_timecourses"),
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    lw=2.0,
    downsample=2,
):
    """
    Save one separate timecourse plot per available TDR/regression axis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if axes_to_plot is None:
        axes_to_plot = list(axis_names)
    else:
        # keep only requested axes that actually exist
        axes_to_plot = [a for a in axes_to_plot if a in axis_names]

    out_paths = []

    for axis in axes_to_plot:
        out_path = out_dir / f"tdr_time_{axis}.png"

        path = plot_tdr_axis_timecourse(
            projections,
            analysis_time,
            axis_names,
            axis=axis,
            event_times=event_times,
            event_labels=event_labels,
            event_linestyles=event_linestyles,
            event_colors=event_colors,
            event_linewidths=event_linewidths,
            event_alphas=event_alphas,
            lw=lw,
            downsample=downsample,
            out_path=out_path,
        )

        out_paths.append(path)

    return out_paths


def plot_tdr_time_y_z_3d(
    projections,
    analysis_time,
    axis_names,
    *,
    y_axis="H",
    z_axis="T",
    y_label=None,
    z_label=None,
    out_path=Path("plots/tdr/tdr_time_y_z_3d.html"),
    downsample=2,
    event_times=None,
    event_labels=None,
    event_colors=None,
    event_opacities=None,
    event_line_dash="dot",
    title=None,
    x_label="Time (s)",
):
    """
    3D Plotly trajectory where:
        x-axis = time
        y-axis = selected TDR axis
        z-axis = selected TDR axis

    Parameters
    ----------
    projections : dict
        condition -> array of shape n_axes x n_time

    analysis_time : array-like
        Time axis. Can be stitched time, cue-aligned time, movement-aligned time, etc.

    event_times : list[float] or None
        Times at which to draw vertical planes.

    event_labels : list[str] or None
        Names for event planes.

    event_colors : list[str] or None
        Plotly rgba strings, e.g. "rgba(80,80,80,1)".

    event_opacities : list[float] or None
        Opacity for each plane.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    analysis_time = np.asarray(analysis_time, dtype=float)

    if y_label is None:
        y_label = y_axis
    if z_label is None:
        z_label = z_axis

    fig = go.Figure()
    all_y = []
    all_z = []
    for cond, Z in projections.items():
        Z = np.asarray(Z, dtype=float)

        if isinstance(cond, tuple) and len(cond) == 3:
            effector, reach_hand, target_hemifield = cond
            label = f"{effector}, hand={reach_hand}, target={target_hemifield}"
            color = CONDITION_COLORS.get(cond, "gray")
        else:
            label = str(cond)
            color = "gray"

        y_proj = np.asarray(Z[y_idx], dtype=float)
        z_proj = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(analysis_time) & np.isfinite(y_proj) & np.isfinite(z_proj)

        time = analysis_time[finite]
        y_proj = y_proj[finite]
        z_proj = z_proj[finite]

        all_y.append(y_proj)
        all_z.append(z_proj)

        time_plot = time[::downsample]
        y_plot = y_proj[::downsample]
        z_plot = z_proj[::downsample]

        hover_text = [
            (
                f"{label}<br>"
                f"time={time_plot[i]:.3f} s<br>"
                f"{y_label}={y_plot[i]:.3f}<br>"
                f"{z_label}={z_plot[i]:.3f}"
            )
            for i in range(len(time_plot))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=time_plot,
                y=y_plot,
                z=z_plot,
                mode="lines+markers",
                name=label,
                line=dict(
                    color=color,
                    width=2,
                ),
                marker=dict(
                    size=1.75,
                    color=color,
                    opacity=0.75,
                ),
                text=hover_text,
                hoverinfo="text",
                connectgaps=True,
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[time_plot[0]],
                y=[y_plot[0]],
                z=[z_plot[0]],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color,
                    symbol="circle",
                ),
                name=f"{label} start",
                showlegend=False,
                text=[f"{label}<br>start<br>time={time_plot[0]:.3f} s"],
                hoverinfo="text",
            )
        )

        # End marker
        fig.add_trace(
            go.Scatter3d(
                x=[time_plot[-1]],
                y=[y_plot[-1]],
                z=[z_plot[-1]],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color,
                    symbol="x",
                ),
                name=f"{label} end",
                showlegend=False,
                text=[f"{label}<br>end<br>time={time_plot[-1]:.3f} s"],
                hoverinfo="text",
            )
        )

    # ------------------------------------------------------------
    # Add cue and movement time planes
    # ------------------------------------------------------------
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
    z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

    y_pad = 0.08 * (y_max - y_min)
    z_pad = 0.08 * (z_max - z_min)

    if y_pad == 0:
        y_pad = 1.0
    if z_pad == 0:
        z_pad = 1.0

    y_min -= y_pad
    y_max += y_pad
    z_min -= z_pad
    z_max += z_pad

    def add_time_plane(
        x_time,
        name,
        *,
        color="rgba(80,80,80,1)",
        opacity=0.04,
        line_dash="dot",
    ):
        if x_time is None or not np.isfinite(x_time):
            return

        fig.add_trace(
            go.Surface(
                x=np.array(
                    [
                        [x_time, x_time],
                        [x_time, x_time],
                    ]
                ),
                y=np.array(
                    [
                        [y_min, y_max],
                        [y_min, y_max],
                    ]
                ),
                z=np.array(
                    [
                        [z_min, z_min],
                        [z_max, z_max],
                    ]
                ),
                showscale=False,
                opacity=opacity,
                colorscale=[[0, color], [1, color]],
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Dotted outline
        corners_y = [y_min, y_max, y_max, y_min, y_min]
        corners_z = [z_min, z_min, z_max, z_max, z_min]

        fig.add_trace(
            go.Scatter3d(
                x=[x_time] * len(corners_y),
                y=corners_y,
                z=corners_z,
                mode="lines",
                line=dict(
                    color=color,
                    width=2,
                    dash=line_dash,
                ),
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if event_times is not None:
        event_times = list(event_times)
        n_events = len(event_times)

        if event_labels is None:
            event_labels = [
                f"Event {i + 1} ({t:g} s)" for i, t in enumerate(event_times)
            ]

        if event_colors is None:
            event_colors = ["rgba(80,80,80,1)"] * n_events

        if event_opacities is None:
            event_opacities = [0.04] * n_events

        if not (
            len(event_labels) == len(event_times)
            and len(event_colors) == len(event_times)
            and len(event_opacities) == len(event_times)
        ):
            raise ValueError(
                "event_times, event_labels, event_colors, and event_opacities "
                "must all have the same length."
            )

        for x_time, label, color, opacity in zip(
            event_times,
            event_labels,
            event_colors,
            event_opacities,
        ):
            add_time_plane(
                x_time,
                label,
                color=color,
                opacity=opacity,
                line_dash=event_line_dash,
            )

    if title is None:
        title = f"TDR trajectories over time: {y_label} axis vs {z_label} axis"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=f"TDR axis: {y_label}",
            zaxis_title=f"TDR axis: {z_label}",
            aspectmode="manual",
            aspectratio=dict(
                x=1.6,
                y=1.0,
                z=1.0,
            ),
        ),
        width=1000,
        height=750,
    )

    fig.write_html(out_path)

    return out_path


def plot_tdr_time_y_2d(
    projections,
    stitched_time,
    axis_names,
    *,
    axis,
    axis_label=None,
    out_path=Path("plots/tdr_int/tdr_time_y_2d.png"),
    cue_time=0.0,
    mov_time=1.6,
    downsample=2,
):
    """
    Plot 8 condition-averaged trajectories projected onto one TDR axis over time.

    x-axis: stitched time
    y-axis: projection onto selected TDR axis
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    axis_idx = axis_names.index(axis)
    t = np.asarray(stitched_time, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for cond, Z in projections.items():
        Z = np.asarray(Z, dtype=float)
        y = Z[axis_idx]

        finite = np.isfinite(t) & np.isfinite(y)

        if isinstance(cond, tuple) and len(cond) == 3:
            effector, hand, target = cond
            label = f"{effector}, hand={hand}, target={target}"
            color = CONDITION_COLORS.get(cond, "0.4")
        else:
            label = str(cond)
            color = "0.4"

        ax.plot(
            t[finite][::downsample],
            y[finite][::downsample],
            color=color,
            lw=2.0,
            label=label,
        )
    ax.axvline(cue_time, color="0.25", linestyle="--", lw=1.2, label="Cue")
    ax.axvline(mov_time, color="0.25", linestyle=":", lw=1.4, label="Movement")

    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)

    ax.set_xlabel("Stitched time (s)")
    ax.set_ylabel(f"Projection onto {axis_label or axis} axis")
    ax.set_title(f"8 condition-averaged projections on {axis_label or axis} axis")

    ax.grid(alpha=0.25)

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_population_average_tdr_input(
    df,
    rate_col="analysis_rate",
    time_col="analysis_time",
    *,
    condition_cols=("effector", "reach_hand", "target_hemifield"),
    unit_cols=("session", "unit_ID"),
    out_path=Path("plots/tdr/population_average_tdr_input_8_conditions.png"),
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    xlabel="Time (s)",
    title="Population-average TDR input: 8 conditions",
):
    """
    Plot simple population averages of the TDR input.

    For each condition:
        1. average trials within each unit
        2. average those unit means across units

    This avoids units with more trials dominating the population average.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.dropna(subset=list(condition_cols) + [rate_col, time_col]).copy()

    # Use a common time axis
    analysis_time = np.asarray(plot_df[time_col].dropna().iloc[0], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    cond_order = list(
        product(
            ["reach", "saccade"],
            ["ipsi", "contra"],
            ["ipsi", "contra"],
        )
    )

    for cond in cond_order:
        effector, reach_hand, target_hemifield = cond

        cond_df = plot_df[
            (plot_df["effector"] == effector)
            & (plot_df["reach_hand"] == reach_hand)
            & (plot_df["target_hemifield"] == target_hemifield)
        ]

        if cond_df.empty:
            continue

        unit_means = []

        for _, unit_df in cond_df.groupby(list(unit_cols), sort=True):
            rates = np.stack(unit_df[rate_col].to_numpy()).astype(float)
            unit_means.append(np.nanmean(rates, axis=0))

        unit_means = np.stack(unit_means, axis=0)

        pop_mean = np.nanmean(unit_means, axis=0)
        pop_sem = np.nanstd(unit_means, axis=0, ddof=1) / np.sqrt(unit_means.shape[0])

        color = CONDITION_COLORS.get(cond, "0.4")
        label = f"{effector}, hand={reach_hand}, target={target_hemifield}"

        ax.plot(
            analysis_time,
            pop_mean,
            color=color,
            lw=2,
            label=label,
        )

        ax.fill_between(
            analysis_time,
            pop_mean - pop_sem,
            pop_mean + pop_sem,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    # ------------------------------------------------------------
    # General event lines
    # ------------------------------------------------------------
    if event_times is not None:
        event_times = list(event_times)
        n_events = len(event_times)

        if event_labels is None:
            event_labels = [f"Event {i + 1}" for i in range(n_events)]

        if event_linestyles is None:
            event_linestyles = ["--"] * n_events

        if event_colors is None:
            event_colors = ["0.25"] * n_events

        if event_linewidths is None:
            event_linewidths = [1.2] * n_events

        if event_alphas is None:
            event_alphas = [0.8] * n_events

        if not (
            len(event_labels) == n_events
            and len(event_linestyles) == n_events
            and len(event_colors) == n_events
            and len(event_linewidths) == n_events
            and len(event_alphas) == n_events
        ):
            raise ValueError(
                "event_times, event_labels, event_linestyles, event_colors, "
                "event_linewidths, and event_alphas must all have the same length."
            )

        for x_time, label, linestyle, color, linewidth, alpha in zip(
            event_times,
            event_labels,
            event_linestyles,
            event_colors,
            event_linewidths,
            event_alphas,
        ):
            if x_time is None or not np.isfinite(x_time):
                continue

            ax.axvline(
                float(x_time),
                color=color,
                linestyle=linestyle,
                lw=linewidth,
                alpha=alpha,
                label=label,
            )

    ax.set_title("Population-average TDR input: 8 conditions")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean z-scored activity")
    ax.grid(alpha=0.25)

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_tdr_trajectory_distance(
    distance,
    analysis_time,
    *,
    labels=None,
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    title="Euclidean distance between trajectories in TDR space",
    ylabel="Euclidean distance",
    xlabel="Time (s)",
    out_path=Path("plots/tdr_int/trajectory_distance.png"),
    lw=2.0,
    downsample=1,
):
    """
    Plot one or more time-varying Euclidean distances.

    Parameters
    ----------
    distance : array-like or list of array-like
        Either:
            - one 1D array of shape n_time
            - list of 1D arrays, each of shape n_time

    analysis_time : array-like
        1D time axis of length n_time.
        Can be stitched time, cue-aligned time, movement-aligned time, etc.

    labels : list[str] or None
        Labels for distance curves.

    event_times : list[float] or None
        Times at which to draw vertical event lines.

    event_labels : list[str] or None
        Labels for event lines.

    xlabel : str
        X-axis label.
    """
    t = np.asarray(analysis_time, dtype=float).ravel()

    # ------------------------------------------------------------
    # Allow either one distance array or a list of distance arrays
    # ------------------------------------------------------------
    if isinstance(distance, (list, tuple)):
        distances = [np.asarray(d, dtype=float).ravel() for d in distance]
    else:
        distances = [np.asarray(distance, dtype=float).ravel()]

    if labels is None:
        labels = [f"Distance {i + 1}" for i in range(len(distances))]

    if len(labels) != len(distances):
        raise ValueError(
            f"`labels` must have same length as `distance`. "
            f"Got {len(labels)} labels and {len(distances)} distances."
        )

    if downsample is None or downsample < 1:
        downsample = 1

    for label, d in zip(labels, distances):
        if d.shape != t.shape:
            raise ValueError(
                f"`distance` and `analysis_time` must have same shape for {label}. "
                f"Got {d.shape} and {t.shape}."
            )

    # ------------------------------------------------------------
    # Event-line defaults
    # ------------------------------------------------------------
    if event_times is not None:
        event_times = list(event_times)
        n_events = len(event_times)

        if event_labels is None:
            event_labels = [f"Event {i + 1}" for i in range(n_events)]

        if event_linestyles is None:
            event_linestyles = ["--"] * n_events

        if event_colors is None:
            event_colors = ["k"] * n_events

        if event_linewidths is None:
            event_linewidths = [1.0] * n_events

        if event_alphas is None:
            event_alphas = [0.75] * n_events

        if not (
            len(event_labels) == n_events
            and len(event_linestyles) == n_events
            and len(event_colors) == n_events
            and len(event_linewidths) == n_events
            and len(event_alphas) == n_events
        ):
            raise ValueError(
                "event_times, event_labels, event_linestyles, event_colors, "
                "event_linewidths, and event_alphas must all have the same length."
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

    for label, d in zip(labels, distances):
        finite = np.isfinite(d) & np.isfinite(t)

        ax.plot(
            t[finite][::downsample],
            d[finite][::downsample],
            lw=lw,
            label=label,
        )

    # ------------------------------------------------------------
    # Draw general event lines
    # ------------------------------------------------------------
    if event_times is not None:
        for x_time, event_label, ls, color, line_width, alpha in zip(
            event_times,
            event_labels,
            event_linestyles,
            event_colors,
            event_linewidths,
            event_alphas,
        ):
            if x_time is None or not np.isfinite(x_time):
                continue

            ax.axvline(
                float(x_time),
                color=color,
                ls=ls,
                lw=line_width,
                alpha=alpha,
                label=event_label,
            )

    ax.axhline(0.0, color="0.75", lw=0.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_tdr_axis_var_time_resolved(
    results,
    *,
    out_path,
    title="Time-resolved variance explained by TDR axes",
    xlabel="Time (s)",
    ylabel="Variance explained",
    y_col="variance_explained",
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    downsample=1,
    axis_order=None,
):
    """
    Plot time-resolved variance explained by TDR axes as line plots.

    Parameters
    ----------
    results : pd.DataFrame
        Expected columns:
            - "time"
            - "axis"
            - y_col

    y_col : str or None
        Column to plot.

    axis_order : list[str] or None
        Optional order of plotted axes, e.g.
            ["T", "H", "E", "ET", "EH", "cueCI", "goCI"]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {"time", "axis", y_col}
    missing = required_cols - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plot_df = results[["time", "axis", y_col]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

    if plot_df.empty:
        raise ValueError("No finite data available for plotting.")

    # Wide format: rows=time, columns=axis, values=variance explained
    wide = plot_df.pivot_table(
        index="time",
        columns="axis",
        values=y_col,
        aggfunc="mean",
    ).sort_index()

    # Optional axis ordering
    if axis_order is not None:
        axis_order = [axis for axis in axis_order if axis in wide.columns]
        remaining = [axis for axis in wide.columns if axis not in axis_order]
        wide = wide[axis_order + remaining]

    # Downsample after pivoting, so all axes stay aligned
    if downsample is None or downsample < 1:
        downsample = 1

    wide = wide.iloc[::downsample]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for axis in wide.columns:
        y = wide[axis].to_numpy(dtype=float)
        t = wide.index.to_numpy(dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)

        if not np.any(finite):
            continue

        ax.plot(
            t[finite],
            y[finite],
            lw=2.0,
            label=axis,
        )

    add_vertical_event_lines(
        ax,
        event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    ax.axhline(0.0, color="0.75", lw=0.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_target_position_trajectories_3d(
    projections,
    analysis_time,
    axis_names,
    *,
    y_axis="space_x_cue",
    z_axis="space_y_cue",
    out_path=Path("plots/tdr/target_position_trajectories_3d.html"),
    downsample=5,
    event_times=None,
    event_labels=None,
):
    """
    Plot condition-averaged neural trajectories for each target position.

    3D axes:
        x = time
        y = projection onto space-x TDR axis
        z = projection onto space-y TDR axis

    projections:
        dict mapping (target_x, target_y) -> projected trajectory
        where each value has shape n_axes x n_time.
    """
    import plotly.graph_objects as go
    import numpy as np
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(analysis_time, dtype=float)

    if y_axis not in axis_names:
        raise ValueError(f"{y_axis=} not found in axis_names={axis_names}")
    if z_axis not in axis_names:
        raise ValueError(f"{z_axis=} not found in axis_names={axis_names}")

    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    fig = go.Figure()

    for cond, Z in projections.items():
        target_x, target_y = cond

        y = np.asarray(Z[y_idx], dtype=float)
        z = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(t) & np.isfinite(y) & np.isfinite(z)

        t_plot = t[finite][::downsample]
        y_plot = y[finite][::downsample]
        z_plot = z[finite][::downsample]

        label = f"x={target_x:.2f}, y={target_y:.2f}"

        hover_text = [
            (
                f"target {label}<br>"
                f"time={t_plot[i]:.3f} s<br>"
                f"{y_axis}={y_plot[i]:.3f}<br>"
                f"{z_axis}={z_plot[i]:.3f}"
            )
            for i in range(len(t_plot))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=t_plot,
                y=y_plot,
                z=z_plot,
                mode="lines+markers",
                name=label,
                line=dict(width=5),
                marker=dict(size=2, opacity=0.7),
                text=hover_text,
                hoverinfo="text",
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[t_plot[0]],
                y=[y_plot[0]],
                z=[z_plot[0]],
                mode="markers",
                marker=dict(size=6, symbol="circle"),
                name=f"{label} start",
                showlegend=False,
            )
        )

        # End marker
        fig.add_trace(
            go.Scatter3d(
                x=[t_plot[-1]],
                y=[y_plot[-1]],
                z=[z_plot[-1]],
                mode="markers",
                marker=dict(size=6, symbol="x"),
                name=f"{label} end",
                showlegend=False,
            )
        )

    # Optional event markers as transparent planes in time
    if event_times is not None:
        y_all = []
        z_all = []

        for Z in projections.values():
            y_all.append(np.asarray(Z[y_idx], dtype=float))
            z_all.append(np.asarray(Z[z_idx], dtype=float))

        y_all = np.concatenate(y_all)
        z_all = np.concatenate(z_all)

        y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
        z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)

        if event_labels is None:
            event_labels = [f"event {i}" for i in range(len(event_times))]

        for event_time, event_label in zip(event_times, event_labels):
            fig.add_trace(
                go.Surface(
                    x=np.array(
                        [
                            [event_time, event_time],
                            [event_time, event_time],
                        ]
                    ),
                    y=np.array(
                        [
                            [y_min, y_max],
                            [y_min, y_max],
                        ]
                    ),
                    z=np.array(
                        [
                            [z_min, z_min],
                            [z_max, z_max],
                        ]
                    ),
                    opacity=0.12,
                    showscale=False,
                    name=event_label,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="Condition-averaged neural trajectories by target position",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title=f"TDR axis: {y_axis}",
            zaxis_title=f"TDR axis: {z_axis}",
            aspectmode="cube",
        ),
        width=1000,
        height=750,
    )

    fig.write_html(out_path)


def plot_target_position_axis_timecourse(
    projections,
    analysis_time,
    axis_names,
    *,
    axis,
    cond_order=None,
    out_path=None,
    downsample=5,
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    lw=2.0,
):
    """
    Plot condition-averaged trajectories for target positions on one TDR axis.

    This makes one 2D timecourse plot:

        x = time
        y = projection on selected TDR axis

    Each line is one target position.

    Expected condition format:
        cond = (target_x, target_y)

    Color encodes physical 2D target position:
        target_x controls red/blue balance
        target_y controls green/brightness
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    if axis not in axis_names:
        raise ValueError(f"{axis=} not found in axis_names={axis_names}")

    if out_path is None:
        out_path = Path(f"plots/tdr/target_position_time_{axis}.png")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(analysis_time, dtype=float)
    axis_idx = axis_names.index(axis)

    conds = list(projections.keys()) if cond_order is None else list(cond_order)

    # ------------------------------------------------------------
    # Color each trajectory by physical target x/y
    # ------------------------------------------------------------
    target_color = make_target_color_fn(conds)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for cond in conds:
        target_x, target_y = cond

        Y = np.asarray(projections[cond], dtype=float)
        y = np.asarray(Y[axis_idx, :], dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)

        color = target_color(cond)
        label = f"x={target_x:.2f}, y={target_y:.2f}"

        ax.plot(
            t[finite][::downsample],
            y[finite][::downsample],
            color=color,
            lw=lw,
            label=label,
        )

    add_vertical_event_lines(
        ax,
        event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    ax.axhline(0.0, color="0.75", lw=0.8)

    ax.set_title(f"Target-position trajectories on {axis}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Projection on {axis}")
    ax.grid(alpha=0.25)

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path


from matplotlib.colors import to_hex


def make_target_color_fn(conds, *, min_brightness=0.35, max_brightness=1.00):
    """
    Color target positions with:
        x-position: blue -> red gradient
        y-position: brightness

    cond format:
        (target_x, target_y)
    """
    target_xs = np.asarray([cond[0] for cond in conds], dtype=float)
    target_ys = np.asarray([cond[1] for cond in conds], dtype=float)

    x_min, x_max = np.nanmin(target_xs), np.nanmax(target_xs)
    y_min, y_max = np.nanmin(target_ys), np.nanmax(target_ys)

    def norm(v, vmin, vmax):
        if vmax == vmin:
            return 0.5
        return (v - vmin) / (vmax - vmin)

    def color_fn(cond):
        target_x, target_y = cond

        x01 = norm(float(target_x), x_min, x_max)
        y01 = norm(float(target_y), y_min, y_max)

        # x controls hue: blue -> red
        base_r = x01
        base_g = 0.0
        base_b = 1.0 - x01

        # y controls brightness
        brightness = min_brightness + y01 * (max_brightness - min_brightness)

        r = brightness * base_r
        g = brightness * base_g
        b = brightness * base_b

        return to_hex((r, g, b))

    return color_fn


def plot_action_subspace_3d_time(
    projections,
    analysis_time,
    axis_names,
    *,
    y_axis,
    z_axis,
    y_label=None,
    z_label=None,
    out_path,
    title=None,
    downsample=5,
    event_times=None,
    event_labels=None,
    event_colors=None,
    event_opacities=None,
    x_label="Time relative to cue onset (s)",
):
    """
    Plot 3 combined action trajectories in a 3D time x axis1 x axis2 plot.

    x-axis:
        time

    y-axis:
        projection onto y_axis

    z-axis:
        projection onto z_axis

    Expected projection conditions:
        ("saccade",)
        ("ipsi_hand",)
        ("contra_hand",)

    This is useful for plots like:
        time x saccade_space_x_mov x saccade_space_y_mov
        time x ipsi_hand_space_x_mov x ipsi_hand_space_y_mov
        time x contra_hand_space_x_mov x contra_hand_space_y_mov
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if y_axis not in axis_names:
        raise ValueError(f"y_axis={y_axis!r} not found in axis_names.")
    if z_axis not in axis_names:
        raise ValueError(f"z_axis={z_axis!r} not found in axis_names.")

    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    if y_label is None:
        y_label = y_axis
    if z_label is None:
        z_label = z_axis

    analysis_time = np.asarray(analysis_time, dtype=float)

    action_order = ["saccade", "ipsi_hand", "contra_hand"]

    action_labels = {
        "saccade": "Saccade",
        "ipsi_hand": "Ipsi-hand reach",
        "contra_hand": "Contra-hand reach",
    }

    action_colors = {
        "saccade": "#bf5f00",
        "ipsi_hand": "#005fbf",
        "contra_hand": "#007fff",
    }

    fig = go.Figure()

    all_y = []
    all_z = []

    for action in action_order:
        key = (action,)

        if key in projections:
            Z = projections[key]
        elif action in projections:
            Z = projections[action]
        else:
            print(f"Skipping {action}: not found in projections.")
            continue

        Z = np.asarray(Z, dtype=float)

        y_proj = np.asarray(Z[y_idx], dtype=float)
        z_proj = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(analysis_time) & np.isfinite(y_proj) & np.isfinite(z_proj)

        t = analysis_time[finite]
        y = y_proj[finite]
        z = z_proj[finite]

        if len(t) == 0:
            print(f"Skipping {action}: no finite points.")
            continue

        all_y.append(y)
        all_z.append(z)

        t_plot = t[::downsample]
        y_plot = y[::downsample]
        z_plot = z[::downsample]

        label = action_labels.get(action, action)
        color = action_colors.get(action, "gray")

        hover_text = [
            (
                f"{label}<br>"
                f"time={t_plot[i]:.3f} s<br>"
                f"{y_axis}={y_plot[i]:.3f}<br>"
                f"{z_axis}={z_plot[i]:.3f}"
            )
            for i in range(len(t_plot))
        ]

        fig.add_trace(
            go.Scatter3d(
                x=t_plot,
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
                    opacity=0.75,
                ),
                text=hover_text,
                hoverinfo="text",
                connectgaps=True,
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[t_plot[0]],
                y=[y_plot[0]],
                z=[z_plot[0]],
                mode="markers",
                name=f"{label} start",
                marker=dict(
                    size=7,
                    color=color,
                    symbol="circle",
                ),
                showlegend=False,
                text=[f"{label}<br>start<br>time={t_plot[0]:.3f} s"],
                hoverinfo="text",
            )
        )

        # End marker
        fig.add_trace(
            go.Scatter3d(
                x=[t_plot[-1]],
                y=[y_plot[-1]],
                z=[z_plot[-1]],
                mode="markers",
                name=f"{label} end",
                marker=dict(
                    size=7,
                    color=color,
                    symbol="x",
                ),
                showlegend=False,
                text=[f"{label}<br>end<br>time={t_plot[-1]:.3f} s"],
                hoverinfo="text",
            )
        )

    if len(all_y) == 0:
        raise ValueError("No action trajectories were plotted.")

    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
    z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)

    y_pad = 0.08 * (y_max - y_min)
    z_pad = 0.08 * (z_max - z_min)

    if y_pad == 0:
        y_pad = 1.0
    if z_pad == 0:
        z_pad = 1.0

    y_min -= y_pad
    y_max += y_pad
    z_min -= z_pad
    z_max += z_pad

    # ------------------------------------------------------------
    # Add event planes, e.g. cue and GO
    # ------------------------------------------------------------
    def add_time_plane(
        x_time,
        name,
        *,
        color="rgba(80,80,80,1)",
        opacity=0.04,
    ):
        if x_time is None or not np.isfinite(x_time):
            return

        fig.add_trace(
            go.Surface(
                x=np.array(
                    [
                        [x_time, x_time],
                        [x_time, x_time],
                    ]
                ),
                y=np.array(
                    [
                        [y_min, y_max],
                        [y_min, y_max],
                    ]
                ),
                z=np.array(
                    [
                        [z_min, z_min],
                        [z_max, z_max],
                    ]
                ),
                showscale=False,
                opacity=opacity,
                colorscale=[[0, color], [1, color]],
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if event_times is not None:
        event_times = list(event_times)
        n_events = len(event_times)

        if event_labels is None:
            event_labels = [f"Event {i + 1}" for i in range(n_events)]

        if event_colors is None:
            event_colors = ["rgba(80,80,80,1)"] * n_events

        if event_opacities is None:
            event_opacities = [0.04] * n_events

        for event_time, event_label, event_color, event_opacity in zip(
            event_times,
            event_labels,
            event_colors,
            event_opacities,
        ):
            add_time_plane(
                event_time,
                event_label,
                color=event_color,
                opacity=event_opacity,
            )

    if title is None:
        title = f"Action trajectories: time × {y_axis} × {z_axis}"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=f"TDR axis: {y_label}",
            zaxis_title=f"TDR axis: {z_label}",
            aspectmode="manual",
            aspectratio=dict(
                x=1.6,
                y=1.0,
                z=1.0,
            ),
        ),
        width=1000,
        height=750,
    )

    fig.write_html(out_path)

    return out_path


def plot_three_action_movement_subspaces_3d(
    projections,
    analysis_time,
    axis_names,
    *,
    out_dir,
    event_times=None,
    event_labels=None,
    event_colors=None,
    event_opacities=None,
    downsample=5,
):
    """
    Make the three action-specific movement-space 3D plots:

        time x saccade_space_x_mov x saccade_space_y_mov
        time x ipsi_hand_space_x_mov x ipsi_hand_space_y_mov
        time x contra_hand_space_x_mov x contra_hand_space_y_mov
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        {
            "name": "saccade",
            "y_axis": "saccade_space_x_mov",
            "z_axis": "saccade_space_y_mov",
            "title": "Action trajectories in saccade movement-space",
            "out_name": "time_saccade_space_x_mov_y_mov.html",
        },
        {
            "name": "ipsi_hand",
            "y_axis": "ipsi_hand_space_x_mov",
            "z_axis": "ipsi_hand_space_y_mov",
            "title": "Action trajectories in ipsi-hand reach movement-space",
            "out_name": "time_ipsi_hand_space_x_mov_y_mov.html",
        },
        {
            "name": "contra_hand",
            "y_axis": "contra_hand_space_x_mov",
            "z_axis": "contra_hand_space_y_mov",
            "title": "Action trajectories in contra-hand reach movement-space",
            "out_name": "time_contra_hand_space_x_mov_y_mov.html",
        },
    ]

    out_paths = []

    for spec in plot_specs:
        y_axis = spec["y_axis"]
        z_axis = spec["z_axis"]

        if y_axis not in axis_names or z_axis not in axis_names:
            print(f"Skipping {spec['name']}: missing {y_axis} or {z_axis}")
            continue

        out_path = plot_action_subspace_3d_time(
            projections,
            analysis_time,
            axis_names,
            y_axis=y_axis,
            z_axis=z_axis,
            y_label=y_axis,
            z_label=z_axis,
            out_path=out_dir / spec["out_name"],
            title=spec["title"],
            event_times=event_times,
            event_labels=event_labels,
            event_colors=event_colors,
            event_opacities=event_opacities,
            downsample=downsample,
        )

        out_paths.append(out_path)

    return out_paths
