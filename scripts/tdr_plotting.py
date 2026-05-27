from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    go_time = 1.6
    cue_idx = np.nanargmin(np.abs(stitched_time - cue_time))
    go_idx = np.nanargmin(np.abs(stitched_time - go_time))

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
                x=[x[go_idx]],
                y=[y[go_idx]],
                z=[z[go_idx]],
                mode="markers",
                marker=dict(size=6, color="red", symbol="square"),
                name="GO (1.6s)",
                showlegend=(col == 1),
                hovertext=[f"GO: t=1.6s"],
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
