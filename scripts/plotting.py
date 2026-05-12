from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from itertools import product

from scripts.utils import *


def plot_condition_subplot(
    ax,
    sdf_unit,
    time,
    condition_level,
    *,
    condition_cols,
    marker_handles,
    t_start,
    t_end,
):
    """Plot trial-averaged SDFs by condition and GO markers; return True if curves were drawn."""
    condition_sdf = (
        sdf_unit.groupby(condition_cols)["sdf_rate"].apply(mean_sdf).reset_index()
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cond_handles = []
    for i, (_, row) in enumerate(condition_sdf.iterrows()):
        c = colors[i % len(colors)]

        label = " | ".join(f"{col}={row[col]}" for col in condition_cols)

        (line,) = ax.plot(
            time,
            row["sdf_rate"],
            label=label,
            color=c,
        )

        cond_handles.append(line)

    ax.axvline(
        0.0,
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        zorder=0,
    )

    ax.legend(
        handles=cond_handles + marker_handles,
        fontsize=8,
        ncol=2,
        loc="upper left",
    )

    ax.set_title(f"{condition_level}, trial-averaged SDF")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlim(t_start, t_end)
    ax.grid(True, alpha=0.3)

    return True


def plot_sdf_per_condition(plot_condition, condition_cols, data, t_start, t_end):
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # The plotted line conditions are all condition columns except the subplot condition
    line_condition_cols = [c for c in condition_cols if c != plot_condition]

    # Setup plot
    marker_handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="--",
            label="Cue onset (state 6)",
        ),
    ]
    plot_kw = dict(
        condition_cols=line_condition_cols,
        marker_handles=marker_handles,
        t_start=t_start,
        t_end=t_end,
    )

    required_cols = [plot_condition] + line_condition_cols + ["sdf_rate", "sdf_time"]
    plot_df = data.dropna(subset=required_cols).copy()
    plot_df = plot_df[np.isfinite(plot_df["t_cue"])]

    # Unique subplot levels, e.g. ["saccade", "reach"]
    condition_levels = sorted(plot_df[plot_condition].dropna().unique())

    for unit_idx, unit_df in plot_df.groupby("unit_index", sort=True):
        unit_id = unit_df["unit_ID"].iloc[0]
        time_series = unit_df["sdf_time"].dropna().iloc[0]
        n_levels = len(condition_levels)

        fig, axes = plt.subplots(
            n_levels,
            1,
            figsize=(7, 3 * n_levels),
            sharex=True,
            constrained_layout=True,
        )
        if n_levels == 1:
            axes = [axes]
        any_plotted = False

        for ax, condition_level in zip(axes, condition_levels):
            sdf_level = unit_df[unit_df[plot_condition] == condition_level].dropna(
                subset=line_condition_cols + ["sdf_rate"]
            )
            ok = plot_condition_subplot(
                ax,
                sdf_level,
                time_series,
                str(condition_level),
                **plot_kw,
            )
            any_plotted = any_plotted or ok
        if not any_plotted:
            plt.close(fig)
            continue
        axes[-1].set_xlabel("Time relative to cue onset (state 6) (s)")
        fig.suptitle(
            f"Unit {unit_id} grouped by {plot_condition}",
            fontsize=12,
        )
        out_path = (
            plots_dir / f"unit_{safe_filename_part(unit_id)}_by_{plot_condition}.png"
        )
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_dpca_results(
    dpca_obj,
    Z,
    time_s,
    cond_levels,
    cond_col_names,
    plots_dir,
    *,
    marginalizations=("t", "e", "h", "s", "eh", "hs"),
    max_components=3,
    dpi=150,
):
    """
    Save dPCA summary figures: time courses per marginalization and a 2D state-space trajectory
    (dPC1 vs dPC2) for marginalization ``t`` if at least two components exist.
    """
    time_s = np.asarray(time_s, dtype=float).ravel()
    cond_dims_list = [list(levels) for levels in cond_levels]
    n_cond = len(cond_dims_list)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plots_dir.mkdir(parents=True, exist_ok=True)
    evr_dict = getattr(dpca_obj, "explained_variance_ratio_", {}) or {}
    print(evr_dict)

    for key in marginalizations:
        if key not in Z:
            continue
        arr = np.asarray(Z[key], dtype=float)
        if arr.ndim < 3:
            continue
        cond_sizes = arr.shape[1:-1]
        expected = tuple(len(levels) for levels in cond_levels)
        if cond_sizes != expected:
            continue
        n_comp = min(max_components, arr.shape[0])
        T_axis = arr.shape[-1]
        t_plot = time_s[:T_axis] if time_s.size >= T_axis else np.arange(T_axis)
        rngs = [range(n) for n in cond_sizes]

        fig, axes = plt.subplots(
            n_comp,
            1,
            figsize=(8, max(3.0, 2.2 * n_comp)),
            sharex=True,
            constrained_layout=True,
        )
        if n_comp == 1:
            axes = [axes]

        for c in range(n_comp):
            ax = axes[c]
            li = 0
            for idxs in product(*rngs):
                y = arr[(c, *idxs, slice(None))]
                label_parts = [
                    f"{cond_col_names[i]}={cond_dims_list[i][idxs[i]]}"
                    for i in range(n_cond)
                ]
                label = ", ".join(label_parts)
                ax.plot(
                    t_plot,
                    y,
                    color=colors[li % len(colors)],
                    label=label,
                )
                li += 1
            ax.set_ylabel("Amplitude")
            ax.set_title(_dpca_component_title(key, c, evr_dict))
            ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("Time relative to cue onset (s)")
        fig.suptitle(f"dPCA marginalization «{key}»", fontsize=12)
        out = plots_dir / f"dpca_timecourses_{key}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 2D trajectory: first two time-marginalized components across conditions
    if "t" in Z and np.asarray(Z["t"]).shape[0] >= 2:
        arr = np.asarray(Z["t"], dtype=float)
        cond_sizes = arr.shape[1:-1]
        expected = tuple(len(levels) for levels in cond_levels)
        if cond_sizes == expected:
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            li = 0
            rngs = [range(n) for n in cond_sizes]
            for idxs in product(*rngs):
                x = arr[(0, *idxs, slice(None))]
                y = arr[(1, *idxs, slice(None))]
                ccol = colors[li % len(colors)]
                lab_parts = [
                    f"{cond_col_names[i]}={cond_dims_list[i][idxs[i]]}"
                    for i in range(n_cond)
                ]
                lab = ", ".join(lab_parts)
                ax.plot(x, y, color=ccol, alpha=0.85, label=lab)
                ax.scatter(x[0], y[0], color=ccol, s=28, marker="o", zorder=5)
                ax.scatter(x[-1], y[-1], color=ccol, s=28, marker="x", zorder=5)
                li += 1
            ax.set_xlabel(_dpca_component_title("t", 0, evr_dict))
            ax.set_ylabel(_dpca_component_title("t", 1, evr_dict))
            ax.set_title("dPC1 vs dPC2 (time marginalization); ○ start, × end")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
            fig.savefig(
                plots_dir / "dpca_trajectory_t_dpc1_dpc2.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)


def _dpca_component_title(key, comp_idx, evr_dict):
    """Subtitle line with optional relative variance for this component (within marginalization)."""
    title = f"dPC{comp_idx + 1} ({key})"
    ratios = evr_dict.get(key)
    if ratios is not None and comp_idx < len(ratios):
        title += f" — var. expl. {ratios[comp_idx]:.3f}"
    return title
