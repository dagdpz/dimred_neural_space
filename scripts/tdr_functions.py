import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.utils import get_state_onset


def add_tdr_regressors(df):
    """
    Binary coding:
        effector: reach=+1, saccade=-1
        target: contra=+1, ipsi=-1
        hand: contra=+1, ipsi=-1
    """
    df = df.copy()

    df["E"] = df["effector"].map({"reach": 1.0, "saccade": -1.0})
    df["T"] = df["target_hemifield"].map({"contra": -1.0, "ipsi": 1.0})
    df["H"] = df["reach_hand"].map({"contra": -1.0, "ipsi": 1.0})

    return df


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
        axes_ortho: orthonormalized TDR axes from QR decomposition
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

        # Select only the rows for the current unit
        for col, val in zip(unit_cols, unit):
            unit_df = unit_df[unit_df[col] == val]
        unit_df = unit_df.dropna(
            subset=list(regressors)
            + [
                "stitched_rate",
                "stitched_time",
                "t_mov",
                "t_go",
            ]
        )
        if len(unit_df) == 0:
            continue

        # Already sqrt-transformed and normalized
        # Shape: n_trials x n_timepoints
        rates = np.stack(unit_df["stitched_rate"].to_numpy()).astype(float)

        # Regression target:
        # one row per trial-timepoint
        y = rates.reshape(-1)

        # Trial-level task regressors repeated across time
        n_trials, n_time = rates.shape

        E = np.repeat(unit_df["E"].to_numpy(dtype=float), n_time)
        H = np.repeat(unit_df["H"].to_numpy(dtype=float), n_time)
        T = np.repeat(unit_df["T"].to_numpy(dtype=float), n_time)

        # Time-dependent condition-independent regressors
        t = np.asarray(unit_df["stitched_time"].iloc[0], dtype=float)

        cueCI_t = (t >= 0.0).astype(float)
        cueCI = np.tile(cueCI_t, n_trials)

        dt_mov_go = unit_df["t_mov"].to_numpy(dtype=float) - unit_df["t_go"].to_numpy(
            dtype=float
        )

        t_go_stitched = 1.6 - dt_mov_go
        goCI = (t[None, :] >= t_go_stitched[:, None]).astype(float)
        goCI[~np.isfinite(t_go_stitched), :] = 0.0
        goCI = goCI.reshape(-1)

        # Design matrix
        X = np.column_stack([E, H, T, cueCI, goCI])

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)

        colnames = ["E", "H", "T", "cueCI", "goCI"]
        keep = [colnames.index(k) for k in regressors]

        betas.append(model.coef_[keep])

    axes_raw = np.asarray(betas, dtype=float)

    axes_ortho = lowdin_orthogonalization(axes_raw)

    return axes_raw, axes_ortho, units
