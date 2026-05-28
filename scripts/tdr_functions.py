import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
    interaction=False,
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
    interaction_terms = [
        ("E", "H"),
    ]

    for unit in units:
        unit_df = df.copy()
        for col, val in zip(unit_cols, unit):
            unit_df = unit_df[unit_df[col] == val]
        unit_df = unit_df.dropna(subset=list(regressors) + ["stitched_rate"])

        rates = np.stack(unit_df["stitched_rate"].to_numpy())
        y = np.nanmean(rates, axis=1)

        X = unit_df[list(regressors)].to_numpy(dtype=float)
        print(unit_df[list(regressors)])
        exit()

        if interaction:
            base_X = unit_df[list(regressors)].to_numpy(dtype=float)
            interaction_X = np.column_stack(
                [
                    unit_df[a].to_numpy(dtype=float) * unit_df[b].to_numpy(dtype=float)
                    for a, b in interaction_terms
                ]
            )
            X = np.hstack([base_X, interaction_X])

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)

        betas.append(model.coef_)

    axes_raw = np.asarray(betas, dtype=float)

    # QR orthogonalization, like standard TDR usage.
    # Columns of Q are orthonormal population axes.
    axes_ortho = lowdin_orthogonalization(axes_raw)

    return axes_raw, axes_ortho, units
