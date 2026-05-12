import numpy as np


def mean_sdf(series):
    """Mean SDF across trials in a group (same time bins); ignores NaNs per time point."""
    arr = np.stack(series.to_numpy())
    return np.nanmean(arr, axis=0)


def safe_filename_part(s):
    """Replace characters unsafe in file names with underscores; keeps alphanumerics, hyphen, underscore."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))
