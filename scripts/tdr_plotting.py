from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

CONDITION_COLORS = {
    # effector, reach_hand, target_hemifield
    ("reach", "ipsi", "ipsi"): "#005fbf",
    ("reach", "ipsi", "contra"): "#bf00bf",
    ("reach", "contra", "ipsi"): "#00bf00",
    ("reach", "contra", "contra"): "#bf5f00",
    ("saccade", "ipsi", "ipsi"): "#007fff",
    ("saccade", "ipsi", "contra"): "#ff00ff",
    ("saccade", "contra", "ipsi"): "#00ff00",
    ("saccade", "contra", "contra"): "#ff7f00",
}


def padded_range(v, pad_frac=0.08):
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    pad = pad_frac * (vmax - vmin)

    if pad == 0:
        pad = 1.0

    return [vmin - pad, vmax + pad]


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


def plot_tdr_axis_timecourses(
    projections,
    stitched_time,
    axis_names,
    *,
    axes_to_plot=("E"),
    cond_order=None,
    cond_label_fn=str,
    cue_time=0.0,
    mov_time=1.6,
    lw=2.0,
    out_path=Path("plots/tdr/tdr_axis_timecourses_E_T_H.html"),
):
    t = np.asarray(stitched_time, dtype=float)
    conds = list(projections.keys()) if cond_order is None else list(cond_order)
    axis_idx = [axis_names.index(a) for a in axes_to_plot]

    fig, axs = plt.subplots(
        len(axis_idx), 1, figsize=(10, 2.8 * len(axis_idx)), sharex=True, sharey=False
    )

    if len(axis_idx) == 1:
        axs = [axs]

    for row, (ax, a_name, ai) in enumerate(zip(axs, axes_to_plot, axis_idx)):
        for cond in conds:
            Y = np.asarray(projections[cond], dtype=float)
            y = Y[ai, :]
            ax.plot(t, y, lw=lw, label=cond_label_fn(cond))
        if cue_time is not None:
            ax.axvline(float(cue_time), color="k", ls="--", lw=1, alpha=0.7)
        if mov_time is not None:
            ax.axvline(float(mov_time), color="k", ls=":", lw=1.2, alpha=0.8)
        ax.set_ylabel(a_name)
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(frameon=False, ncol=2)

    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tdr_time_hand_target_3d(
    projections,
    stitched_time,
    axis_names,
    *,
    y_axis="H",
    z_axis="T",
    y_label=None,
    z_label=None,
    out_path=Path("plots/tdr/tdr_time_hand_target_3d.html"),
    downsample=2,
    cue_time=0.0,
    mov_time=1.6,
):
    """
    3D Plotly trajectory where:
        x-axis = time
        y-axis = selected TDR axis
        z-axis = selected TDR axis

    Example:
        y_axis="H", z_axis="T"
        y_axis="E", z_axis="T"
        y_axis="E", z_axis="H"
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_idx = axis_names.index(y_axis)
    z_idx = axis_names.index(z_axis)

    stitched_time = np.asarray(stitched_time, dtype=float)

    fig = go.Figure()
    all_y = []
    all_z = []
    for cond, Z in projections.items():
        Z = np.asarray(Z, dtype=float)

        # Expected full 8-condition key:
        # cond = (effector, reach_hand, target_hemifield)
        if isinstance(cond, tuple) and len(cond) == 3:
            effector, reach_hand, target_hemifield = cond
            label = f"{effector}, hand={reach_hand}, target={target_hemifield}"
            color = CONDITION_COLORS.get(cond, "gray")
        else:
            label = str(cond)
            color = "gray"

        y_proj = np.asarray(Z[y_idx], dtype=float)
        z_proj = np.asarray(Z[z_idx], dtype=float)

        finite = np.isfinite(stitched_time) & np.isfinite(y_proj) & np.isfinite(z_proj)

        time = stitched_time[finite]
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
                    width=3,
                ),
                marker=dict(
                    size=2.5,
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
                    size=7,
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
                    size=7,
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

    def add_time_plane(x_time, name, color="rgba(80,80,80,1)"):
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
                opacity=0.04,
                colorscale=[[0, color], [1, color]],
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Add dotted outline so the time plane is visible
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
                    dash="dot",
                ),
                name=name,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    add_time_plane(cue_time, f"Cue ({cue_time:g} s)")
    add_time_plane(mov_time, f"MOV ({mov_time:g} s)")

    fig.update_layout(
        title="TDR trajectories over time: hand axis vs target axis",
        scene=dict(
            xaxis_title="Time (s)",
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
