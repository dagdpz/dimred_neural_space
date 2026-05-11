from pathlib import Path

import pandas as pd
import numpy as np
from dPCA import dPCA
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


def clean_mat_value(x):
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        if x.size == 1:
            return x.item()
        return x
    return x


def reach_hand_to_side(x):
    if pd.isna(x):
        return np.nan

    x = int(x)

    if x == 1:
        return "left"
    elif x == 2:
        return "right"
    else:
        return np.nan


def ipsi_contra(side, reference_side):
    if pd.isna(side) or pd.isna(reference_side):
        return np.nan

    return "ipsi" if side == reference_side else "contra"


def effector_to_label(x):
    if pd.isna(x):
        return np.nan

    x = int(x)

    if x == 3:
        return "saccade"
    elif x == 4:
        return "reach"
    elif x == 6:
        return "saccade_reach"
    else:
        return np.nan


def tar_pos_to_side(x):
    """
    tar_pos is complex: X + iY.
    Real part = X position.
    negative X = left
    positive X = right
    """
    if pd.isna(x):
        return np.nan

    x_real = np.real(x)

    if x_real < 0:
        return "left"
    elif x_real > 0:
        return "right"
    else:
        return "center"


def pulvinar_to_side(x):
    if x == "dPulv_r":
        return "right"
    elif x == "dPulv_l":
        return "left"
    else:
        return np.nan


def get_state_onset(states_onset, states, state_id):
    """Return onset time for the first occurrence of state_id, or NaN if missing."""
    if states_onset is None or states is None:
        return np.nan
    s = np.asarray(states, dtype=float).ravel()
    t = np.asarray(states_onset, dtype=float).ravel()
    if s.size == 0 or t.size == 0 or s.size != t.size:
        return np.nan
    mask = np.isfinite(s) & np.isfinite(t) & (s == float(state_id))
    if not np.any(mask):
        return np.nan
    return float(t[np.where(mask)[0][0]])


def spike_times_to_sdf(spike_times, t_start, t_end, bin_size=0.001, sigma=0.05):
    if isinstance(spike_times, float) and np.isnan(spike_times):
        time = np.arange(t_start, t_end, bin_size)
        return time, np.full_like(time, np.nan, dtype=float)
    spike_times = np.asarray(spike_times, dtype=float).ravel()
    time_edges = np.arange(t_start, t_end + bin_size, bin_size)
    time = time_edges[:-1] + bin_size / 2
    counts, _ = np.histogram(spike_times, bins=time_edges)
    sigma_bins = sigma / bin_size
    smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=sigma_bins)
    rate = smoothed_counts / bin_size
    return time, rate


def main(seed=0):
    rng = np.random.default_rng(seed)

    mat = loadmat("data/population_Linus_20160518.mat")
    population = mat["population"]

    data = {}
    selected_keys = ["unit_ID", "target", "trial"]

    for field in population.dtype.names:
        if field not in selected_keys:
            continue
        values = []
        for item in population[field][0]:
            try:
                values.append(clean_mat_value(item[0]))
            except Exception:
                values.append(clean_mat_value(item))
        data[field] = values

    rows = []
    n_units = len(data["unit_ID"])
    for unit_idx in range(n_units):
        unit_id = str(clean_mat_value(data["unit_ID"][unit_idx]))
        pulvinar = clean_mat_value(data["target"][unit_idx])

        pulvinar_side = pulvinar_to_side(pulvinar)
        pulvinar_hemifield = "left" if pulvinar_side == "right" else "right"

        unit_trials = data["trial"][unit_idx]
        trial_fields = unit_trials.dtype.names

        for trial_idx in range(unit_trials.shape[0]):
            row = {
                "unit_index": unit_idx,
                "unit_ID": unit_id,
                "pulvinar": pulvinar,
                "pulvinar_side": pulvinar_side,
                "pulvinar_hemifield": pulvinar_hemifield,
                "trial_index": trial_idx,
            }

            for field in trial_fields:
                value = unit_trials[field][trial_idx]
                row[field] = clean_mat_value(value)

            rows.append(row)

    df = pd.DataFrame(rows)

    df["reach_hand"] = df["reach_hand"].apply(reach_hand_to_side)
    # Cue color indicates which hand is relevant for the trial. `reach_hand` encodes that side;
    # ipsi/contra is relative to the recorded pulvinar. On reach trials the monkey reaches with that
    # hand; on saccade trials only the eyes may move, but the same hand rule is still cued (color).
    df["reach_hand_relative"] = df.apply(
        lambda row: ipsi_contra(row["reach_hand"], row["pulvinar_side"]),
        axis=1,
    )
    df["effector"] = df["effector"].apply(effector_to_label)
    df["target_x"] = df["tar_pos"].apply(
        lambda x: np.real(x) if not pd.isna(x) else np.nan
    )
    df["target_side"] = df["tar_pos"].apply(tar_pos_to_side)
    # Target side (from tar_pos) vs recording site: ipsi/contra in *target* space (not "hemifield" naming).
    df["target_region_relative"] = df.apply(
        lambda row: ipsi_contra(row["target_side"], row["pulvinar_side"]),
        axis=1,
    )

    df = df[df["type"] == 4]
    df = df[df["choice"] == 0]
    df = df[df["success"] == 1]

    select_columns = [
        "unit_index",
        "trial_index",
        "unit_ID",
        "pulvinar",
        "pulvinar_side",
        "pulvinar_hemifield",
        "reach_hand",
        "reach_hand_relative",
        "effector",
        "target_side",
        "target_region_relative",
        "trial_onset_time",
        "run_onset_time",
        "states_onset",
        "states",
        "arrival_times",
    ]
    df = df[select_columns]

    # Align spikes and event times to cue onset (state 6). GO marker: state 4 relative to cue.
    cue_state, go_state = 6, 4

    def trial_alignment(row):
        t_cue = get_state_onset(row["states_onset"], row["states"], cue_state)
        t_go = get_state_onset(row["states_onset"], row["states"], go_state)
        spikes = np.asarray(row["arrival_times"], dtype=float).ravel()
        if np.isfinite(t_cue):
            spikes_rel = spikes - t_cue
            go_rel = (t_go - t_cue) if np.isfinite(t_go) else np.nan
        else:
            spikes_rel = np.full_like(spikes, np.nan)
            go_rel = np.nan
        return pd.Series(
            {
                "t_cue": t_cue,
                "t_go_rel": go_rel,
                "arrival_times_rel": spikes_rel,
            }
        )

    align = df.apply(trial_alignment, axis=1)
    df = pd.concat([df, align], axis=1)

    # -------------------------------------------------------
    # Converting spikes to spike density (aligned to cue onset, state 6)
    # -------------------------------------------------------
    t_start = -0.5
    t_end = 2
    bin_size = 0.001  # 1 ms bins
    sigma = 0.05  # 50 ms Gaussian smoothing
    sdf_results = df["arrival_times_rel"].apply(
        lambda spikes: spike_times_to_sdf(
            spikes,
            t_start=t_start,
            t_end=t_end,
            bin_size=bin_size,
            sigma=sigma,
        )
    )
    df["sdf_time"] = sdf_results.apply(lambda x: x[0])
    df["sdf_rate"] = sdf_results.apply(lambda x: x[1])

    # -------------------------------------------------------
    # Per unit: one figure, two rows — saccade (top), reach (bottom);
    # trial-averaged SDF per condition + markers (states 6 and 4)
    # -------------------------------------------------------
    condition_cols = ["reach_hand_relative", "target_region_relative"]
    marker_cols = ["t_go_rel"]
    plot_df = df.dropna(subset=condition_cols + ["effector"]).copy()
    plot_df = plot_df[np.isfinite(plot_df["t_cue"])]
    plot_df = plot_df[plot_df["effector"].isin(["saccade", "reach"])]

    def mean_sdf(series):
        arr = np.stack(series.to_numpy())
        return np.nanmean(arr, axis=0)

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    def safe_filename_part(s):
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))

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
            label="GO signal (state 4)",
        ),
    ]

    def plot_effector_subplot(ax, sdf_unit, time, effector_name):
        """Trial-averaged SDFs for hand label × target region; returns True if any curves drawn."""
        if sdf_unit.empty:
            ax.text(
                0.5,
                0.5,
                "No trials",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(f"{effector_name} (no data)")
            ax.set_ylabel("Firing rate (Hz)")
            ax.set_xlim(t_start, t_end)
            ax.grid(True, alpha=0.3)
            return False

        condition_sdf = (
            sdf_unit.groupby(condition_cols)["sdf_rate"].apply(mean_sdf).reset_index()
        )
        go_mean = (
            sdf_unit.groupby(condition_cols)[marker_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        merged = condition_sdf.merge(go_mean, on=condition_cols, how="left")
        if merged.empty:
            ax.text(
                0.5,
                0.5,
                "No trials",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(f"{effector_name} (no data)")
            ax.set_ylabel("Firing rate (Hz)")
            ax.set_xlim(t_start, t_end)
            ax.grid(True, alpha=0.3)
            return False

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cond_handles = []
        for i, (_, row) in enumerate(merged.iterrows()):
            c = colors[i % len(colors)]
            label = (
                f"hand={row['reach_hand_relative']} | "
                f"target={row['target_region_relative']}"
            )
            (line,) = ax.plot(time, row["sdf_rate"], label=label, color=c)
            cond_handles.append(line)
            if np.isfinite(row["t_go_rel"]):
                ax.axvline(
                    row["t_go_rel"],
                    color=c,
                    alpha=0.45,
                    linestyle=":",
                    linewidth=1.2,
                )

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
        ax.set_title(f"{effector_name}, trial-averaged SDF per condition")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlim(t_start, t_end)
        ax.grid(True, alpha=0.3)
        return True

    for unit_idx, unit_df in plot_df.groupby("unit_index", sort=True):
        unit_id = unit_df["unit_ID"].iloc[0]
        time_series = unit_df["sdf_time"].dropna()
        if time_series.empty:
            continue
        time = time_series.iloc[0]

        fig, (ax_sac, ax_reach) = plt.subplots(
            2,
            1,
            figsize=(7, 6),
            sharex=True,
            constrained_layout=True,
        )
        sdf_sac = unit_df[unit_df["effector"] == "saccade"].dropna(
            subset=condition_cols + ["sdf_rate"]
        )
        sdf_reach = unit_df[unit_df["effector"] == "reach"].dropna(
            subset=condition_cols + ["sdf_rate"]
        )

        ok_sac = plot_effector_subplot(ax_sac, sdf_sac, time, "Saccade")
        ok_reach = plot_effector_subplot(ax_reach, sdf_reach, time, "Reach")
        if not (ok_sac or ok_reach):
            plt.close(fig)
            continue

        ax_reach.set_xlabel("Time relative to cue onset (state 6) (s)")
        fig.suptitle(f"Unit {unit_idx} ({unit_id})", fontsize=12)

        out_path = plots_dir / f"unit_{unit_idx:04d}_{safe_filename_part(unit_id)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
