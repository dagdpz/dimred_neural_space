import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def clean_mat_value(x):
    """
    Some fields have just one item or array in them
    """
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        if x.size == 1:
            return x.item()
        return x
    return x


def decode_reach_hand_label(x):
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


def decode_effector_label(x):
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
    mask = s == float(state_id)
    return float(t[np.where(mask)[0][0]])


def trial_alignment_to_state(row, state):
    """Returns absolute state time and spike times relative to that state for one trial."""
    t_state = get_state_onset(row["states_onset"], row["states"], state)
    spike_arrival_times = np.asarray(row["arrival_times"], dtype=float).ravel()
    spikes_arrival_times_rel = spike_arrival_times - t_state
    return pd.Series(
        {
            "t_state": t_state,
            "arrival_times_rel": spikes_arrival_times_rel,
        }
    )


def spike_times_to_sdf(spike_times, t_start, t_end, bin_size=0.001, sigma=0.05):
    """Bin spikes, Gaussian-smooth counts, convert to Hz; returns (time_axis, rate) or NaN rate if spikes missing."""
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


def mean_sdf(series):
    """Mean SDF across trials in a group (same time bins); ignores NaNs per time point."""
    arr = np.stack(series.to_numpy())
    return np.nanmean(arr, axis=0)


def safe_filename_part(s):
    """Replace characters unsafe in file names with underscores; keeps alphanumerics, hyphen, underscore."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))
