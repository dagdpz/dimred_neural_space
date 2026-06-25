from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from itertools import product

from scripts.utils import *

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)

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


def plot_trial_counts(
    df,
    *,
    unit_cols=("session", "unit_ID"),
    plots_dir=Path("plots/preprocessing"),
    filename="trial_counts.png",
):
    """
    Plot number of rows/trials per unit.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    trial_counts = df.groupby(list(unit_cols)).size().rename("n_trials").reset_index()

    x_labels = [
        "_".join(str(row[col]) for col in unit_cols)
        for _, row in trial_counts.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.bar(np.arange(len(trial_counts)), trial_counts["n_trials"].to_numpy())

    ax.set_xticks(np.arange(len(trial_counts)))
    ax.set_xticklabels(
        x_labels,
        rotation=90,
        fontsize=6,
    )

    ax.set_xlabel("Unit")
    ax.set_ylabel("Trial count")
    ax.set_title("Number of trials per unit")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nTrial count summary:")
    print(f"  Mean: {trial_counts['n_trials'].mean():.2f}")
    print(f"  SD: {trial_counts['n_trials'].std(ddof=1):.2f}")
    print(f"  Median: {trial_counts['n_trials'].median():.2f}")


def plot_random_sdfs(
    df,
    *,
    time_col="sdf_time",
    rate_col="sdf_rate",
    plots_dir=Path("plots/preprocessing"),
    filename="random_10_sdfs.png",
    n_examples=10,
    random_state=0,
):
    """
    Plot random example SDFs from valid rows.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    valid_sdf = df[
        df[time_col].apply(is_valid_array) & df[rate_col].apply(is_valid_array)
    ].copy()

    n_plot = min(n_examples, len(valid_sdf))

    if n_plot == 0:
        print("No valid SDFs available for random SDF plot.")
        return None

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(valid_sdf.index, size=n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx in sample_idx:
        row = valid_sdf.loc[idx]

        t = np.asarray(row[time_col], dtype=float)
        r = np.asarray(row[rate_col], dtype=float)

        label = f"unit {row['unit_ID']}, trial {row['trial_index']}"

        if "session" in row.index:
            label = f"{row['session']}, {label}"

        ax.plot(
            t,
            r,
            lw=1.2,
            alpha=0.8,
            label=label,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Random example SDFs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plots_dir / filename


def plot_event_diagnostics(
    df,
    *,
    plots_dir=Path("plots/sdf"),
):
    """
    Plot basic event-time diagnostics:
        1. cue and movement onset distributions
        2. cue-to-movement delay by effector
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Event-time distributions
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))

    for col, label in [
        ("t_cue", "Cue"),
        ("t_mov", "Movement"),
    ]:
        if col not in df.columns:
            continue

        values = df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        ax.hist(
            values,
            bins=80,
            alpha=0.5,
            label=label,
        )

    ax.set_xlabel("Event time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Event-time distributions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        plots_dir / "event_time_distributions.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Cue-to-movement delay by effector
    # ------------------------------------------------------------
    required = {"t_cue", "t_mov", "effector"}

    if not required.issubset(df.columns):
        return

    plot_df = df.dropna(subset=["t_cue", "t_mov", "effector"]).copy()
    plot_df["cue_to_mov_delay"] = plot_df["t_mov"].to_numpy(dtype=float) - plot_df[
        "t_cue"
    ].to_numpy(dtype=float)

    plot_df = plot_df[np.isfinite(plot_df["cue_to_mov_delay"])]

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    effectors = sorted(plot_df["effector"].dropna().unique())

    data = [
        plot_df.loc[
            plot_df["effector"] == effector,
            "cue_to_mov_delay",
        ].to_numpy(dtype=float)
        for effector in effectors
    ]

    ax.boxplot(
        data,
        labels=effectors,
        showfliers=False,
    )

    ax.set_xlabel("Effector")
    ax.set_ylabel("Cue to movement delay (s)")
    ax.set_title("Cue-to-movement delay by effector")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        plots_dir / "cue_to_movement_delay_by_effector.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def padded_range(v, pad_frac=0.08):
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    pad = pad_frac * (vmax - vmin)

    if pad == 0:
        pad = 1.0

    return [vmin - pad, vmax + pad]


def add_vertical_event_lines(
    ax,
    event_times=None,
    *,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
):
    """
    Draw one or more vertical event lines on an axis.

    Parameters
    ----------
    event_times : list[float] or None
        Times at which to draw vertical lines.

    event_labels : list[str] or None
        Labels for legend. Must match length of event_times if given.

    event_linestyles : list[str] or None
        Line styles, e.g. ["--", ":"]. Defaults to "--".

    event_colors : list[str] or None
        Line colors. Defaults to "k".

    event_linewidths : list[float] or None
        Line widths. Defaults to 1.0.

    event_alphas : list[float] or None
        Transparencies. Defaults to 0.8.
    """
    if event_times is None:
        return

    n = len(event_times)

    if event_labels is None:
        event_labels = [None] * n
    if event_linestyles is None:
        event_linestyles = ["--"] * n
    if event_colors is None:
        event_colors = ["k"] * n
    if event_linewidths is None:
        event_linewidths = [1.0] * n
    if event_alphas is None:
        event_alphas = [0.8] * n

    for name, values in {
        "event_labels": event_labels,
        "event_linestyles": event_linestyles,
        "event_colors": event_colors,
        "event_linewidths": event_linewidths,
        "event_alphas": event_alphas,
    }.items():
        if len(values) != n:
            raise ValueError(
                f"{name} must have same length as event_times. "
                f"Got {len(values)} and {n}."
            )

    for t, label, ls, color, lw, alpha in zip(
        event_times,
        event_labels,
        event_linestyles,
        event_colors,
        event_linewidths,
        event_alphas,
    ):
        if t is None or not np.isfinite(t):
            continue

        ax.axvline(
            float(t),
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            label=label,
        )


def plot_rate_distributions_before_after(
    raw_rates,
    sqrt_rates,
    norm_raw_rates,
    norm_sqrt_rates,
    *,
    plots_dir=Path("plots/preprocessing"),
):
    """
    Plot firing-rate distribution before and after sqrt + z-score normalization.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_rates = np.concatenate(raw_rates).astype(float)
    sqrt_rates = np.concatenate(sqrt_rates).astype(float)
    norm_raw_rates = np.concatenate(norm_raw_rates).astype(float)
    norm_sqrt_rates = np.concatenate(norm_sqrt_rates).astype(float)

    raw_rates = raw_rates[np.isfinite(raw_rates)]
    sqrt_rates = sqrt_rates[np.isfinite(sqrt_rates)]
    norm_raw_rates = norm_raw_rates[np.isfinite(norm_raw_rates)]
    norm_sqrt_rates = norm_sqrt_rates[np.isfinite(norm_sqrt_rates)]

    norm_bins = np.linspace(
        min(np.nanmin(norm_raw_rates), np.nanmin(norm_sqrt_rates)),
        max(np.nanmax(norm_raw_rates), np.nanmax(norm_sqrt_rates)),
        500,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    axes[0, 0].hist(raw_rates, bins=100, edgecolor="black", alpha=0.75)
    axes[0, 0].set_title("Raw firing rates")
    axes[0, 0].set_xlabel("Firing rate (Hz)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(sqrt_rates, bins=100, edgecolor="black", alpha=0.75)
    axes[0, 1].set_title("After sqrt transform")
    axes[0, 1].set_xlabel("sqrt(rate)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(norm_raw_rates, bins=norm_bins, edgecolor="black", alpha=0.75)
    axes[1, 0].set_title("Z-score without sqrt")
    axes[1, 0].set_xlabel("Z-scored rate")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xlim(-10, 10)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(norm_sqrt_rates, bins=norm_bins, edgecolor="black", alpha=0.75)
    axes[1, 1].set_title("Sqrt + z-score")
    axes[1, 1].set_xlabel("Z-scored sqrt(rate)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xlim(-10, 10)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Firing-rate distributions before and after normalization")

    out_path = plots_dir / "rate_distribution_normalization_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_condition_color(row, condition_cols, fallback_color):
    """
    Order: effector, reach_hand, target_hemifield
    """
    key = tuple(
        row[col]
        for col in ["effector", "reach_hand", "target_hemifield"]
        if col in condition_cols
    )

    # If all three variables are plotted together
    if len(key) == 3 and key in CONDITION_COLORS:
        return CONDITION_COLORS[key]

    return fallback_color


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
        fallback = colors[i % len(colors)]
        c = get_condition_color(row, condition_cols, fallback)

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


def plot_effector_sdf(
    data,
    reach_hand,
    target_hemifield,
    *,
    plots_dir=None,
    analysis_label=None,
):
    plots_dir = Path("plots") if plots_dir is None else Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = f"{analysis_label} | " if analysis_label else ""

    effector_colors = {
        "reach": CONDITION_COLORS[("reach", f"{reach_hand}", f"{target_hemifield}")],
        "saccade": CONDITION_COLORS[
            ("saccade", f"{reach_hand}", f"{target_hemifield}")
        ],
    }

    marker_handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="--",
            label="Cue onset (state 6)",
        ),
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle=":",
            label="Movement onset (state 68)",
        ),
    ]
    required_cols = [
        "effector",
        "sdf_rate_cue",
        "sdf_time_cue",
        "sdf_rate_mov",
        "sdf_time_mov",
    ]
    plot_df = data.dropna(subset=required_cols).copy()
    plot_df = plot_df[np.isfinite(plot_df["t_cue"])]
    unit_group_cols = ["session", "unit_ID"]

    for _, unit_df in plot_df.groupby(unit_group_cols, sort=True):
        unit_id = unit_df["unit_ID"].iloc[0]
        session = unit_df["session"].iloc[0]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(6, 5), sharey=True, constrained_layout=True
        )

        cue_time = unit_df["sdf_time_cue"].dropna().iloc[0]
        mov_time = unit_df["sdf_time_mov"].dropna().iloc[0]
        cue_sdf = (
            unit_df.groupby("effector")["sdf_rate_cue"].apply(mean_sdf).reset_index()
        )
        mov_sdf = (
            unit_df.groupby("effector")["sdf_rate_mov"].apply(mean_sdf).reset_index()
        )

        cond_handles = []
        for _, row in cue_sdf.iterrows():
            effector = row["effector"]
            color = effector_colors.get(effector, "0.4")
            (line,) = ax1.plot(
                cue_time,
                row["sdf_rate_cue"],
                color=color,
                label=f"effector={effector}",
            )
            cond_handles.append(line)
        for _, row in mov_sdf.iterrows():
            effector = row["effector"]
            color = effector_colors.get(effector, "0.4")
            (line,) = ax2.plot(
                mov_time,
                row["sdf_rate_mov"],
                color=color,
                label=f"effector={effector}",
            )
        ax1.axvline(
            0.0,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=0,
        )
        ax2.axvline(
            0.0,
            color="0.35",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            zorder=0,
        )

        fig.legend(
            handles=cond_handles + marker_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(cond_handles) + len(marker_handles),
            fontsize=8,
            frameon=False,
        )

        ax1.set_ylabel("Firing rate (Hz)")
        ax1.set_xlabel("Time relative to cue onset (s)")
        ax2.set_xlabel("Time relative to movement onset (s)")

        ax1.set_title("Cue-aligned")
        ax2.set_title("Movement-aligned")
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"unit {unit_id} | hand={reach_hand}, target={target_hemifield}",
            fontsize=12,
        )

        out_path = (
            plots_dir / f"{safe_filename_part(unit_id)}"
            f"_hand_{reach_hand}_target_{target_hemifield}_effector_sdf.png"
        )
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_sdf_per_condition(
    plot_condition,
    condition_cols,
    data,
    t_start,
    t_end,
    *,
    plots_dir=None,
    analysis_label=None,
):
    plots_dir = Path("plots") if plots_dir is None else Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = f"{analysis_label} | " if analysis_label else ""

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

    unit_group_cols = ["session", "unit_ID"]
    for _, unit_df in plot_df.groupby(unit_group_cols, sort=True):
        unit_id = unit_df["unit_ID"].iloc[0]
        session = unit_df["session"].iloc[0]
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
            f"{title_prefix}{session} | unit {unit_id} grouped by {plot_condition}",
            fontsize=12,
        )
        out_path = (
            plots_dir
            / f"{safe_filename_part(session)}_unit_{safe_filename_part(unit_id)}_by_{plot_condition}.png"
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
    analysis_label=None,
    marginalizations=("t", "h", "s", "hs", "ht", "st"),
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
    title_prefix = f"{analysis_label} — " if analysis_label else ""
    print(f"[{analysis_label or plots_dir}] {evr_dict}")

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
        fig.suptitle(f"{title_prefix}dPCA marginalization «{key}»", fontsize=12)
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
            ax.set_title(
                f"{title_prefix}dPC1 vs dPC2 (time marginalization); ○ start, × end"
            )
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


def plot_random_shifted_sdfs(
    df,
    analysis_time,
    *,
    rate_col="analysis_rate",
    unit_col="unit_ID",
    trial_col="trial_index",
    effector_col="effector",
    n_plot=10,
    seed=0,
    event_times=None,
    event_labels=None,
    event_linestyles=None,
    event_colors=None,
    event_linewidths=None,
    event_alphas=None,
    out_path=Path("plots/tdr/random_10_sdfs_shifted.png"),
    title="10 random SDFs, vertically shifted",
    xlabel="Time (s)",
    ylabel="Shifted normalized firing rate",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_time = np.asarray(analysis_time, dtype=float)
    n_time = len(analysis_time)

    rng = np.random.default_rng(seed)

    valid_df = df[
        df[rate_col].apply(
            lambda x: (is_valid_array(x) and np.asarray(x, dtype=float).size == n_time)
        )
    ].copy()

    if len(valid_df) == 0:
        raise ValueError(f"No valid arrays found in {rate_col!r}.")

    n_plot = min(n_plot, len(valid_df))
    sample_idx = rng.choice(valid_df.index, size=n_plot, replace=False)

    all_rates = np.stack(
        [np.asarray(valid_df.loc[idx, rate_col], dtype=float) for idx in sample_idx]
    )

    y_range = np.nanmax(all_rates) - np.nanmin(all_rates)
    offset_step = 0.45 * y_range if y_range > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, idx in enumerate(sample_idx):
        row = valid_df.loc[idx]
        r = np.asarray(row[rate_col], dtype=float)
        r_shifted = r + i * offset_step

        label_parts = []
        if unit_col in row:
            label_parts.append(f"unit {row[unit_col]}")
        if trial_col in row:
            label_parts.append(f"trial {row[trial_col]}")
        if effector_col in row:
            label_parts.append(str(row[effector_col]))

        ax.plot(
            analysis_time,
            r_shifted,
            lw=1.2,
            alpha=0.9,
            label=", ".join(label_parts),
        )

    add_vertical_event_lines(
        ax,
        event_times=event_times,
        event_labels=event_labels,
        event_linestyles=event_linestyles,
        event_colors=event_colors,
        event_linewidths=event_linewidths,
        event_alphas=event_alphas,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_first_row_ci_windows(
    df,
    *,
    time_col="analysis_time",
    ci_cols=("cueCI", "planCI", "goCI", "movCI"),
    out_path=Path("plots/tdr_cue/ci_windows_first_row.png"),
):
    """
    Plot CI regressor windows from the first row of df.

    This is mainly a sanity check that cueCI, planCI, goCI, and movCI
    are active at the expected times.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        raise ValueError("Cannot plot CI windows from an empty dataframe.")

    required_cols = [time_col] + list(ci_cols)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    row = df.iloc[0]
    t = np.asarray(row[time_col], dtype=float)

    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError(f"{time_col} is not a valid 1D finite array.")

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    offset_step = 1.25

    for i, col in enumerate(ci_cols):
        y = np.asarray(row[col], dtype=float)

        if y.shape != t.shape:
            raise ValueError(
                f"{col} has shape {y.shape}, but {time_col} has shape {t.shape}."
            )

        ax.plot(
            t,
            y + i * offset_step,
            lw=2,
            label=col,
        )

    # Event markers from the same row
    event_specs = [
        ("Cue", "t_cue", "--"),
        ("GO", "t_go", ":"),
        ("Movement", "t_mov", "-."),
        ("Movement end", "t_mov_end", "-"),
    ]

    for label, col, linestyle in event_specs:
        if col in df.columns:
            event_time = row[col]
            if np.isfinite(event_time):
                ax.axvline(
                    float(event_time),
                    color="k",
                    linestyle=linestyle,
                    linewidth=1.0,
                    alpha=0.7,
                    label=label,
                )

    ax.set_xlabel("Time relative to cue onset (s)")
    ax.set_ylabel("CI regressor value, vertically shifted")
    ax.set_title("Condition-independent regressor windows")
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return out_path
