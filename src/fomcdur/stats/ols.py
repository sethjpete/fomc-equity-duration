"""Small, dependency-free OLS with heteroskedasticity-robust (HC1) errors.

Kept in-house so the event regressions are transparent and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    nobs: int
    r2: float

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse


def ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True) -> OLSResult:
    """OLS of y on X with HC1 standard errors. Rows with any NaN are dropped."""
    X = X.copy()
    if add_const:
        X.insert(0, "const", 1.0)
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = df["_y"].to_numpy(float)
    Xv = df.drop(columns="_y").to_numpy(float)
    n, k = Xv.shape
    if n <= k:
        raise ValueError(f"Not enough observations ({n}) for {k} regressors")
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    meat = (Xv * resid[:, None] ** 2).T @ Xv
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)
    tss = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - (resid**2).sum() / tss if tss > 0 else np.nan
    cols = df.columns.drop("_y")
    return OLSResult(
        params=pd.Series(beta, index=cols),
        bse=pd.Series(np.sqrt(np.diag(cov)), index=cols),
        nobs=n,
        r2=float(r2),
    )
