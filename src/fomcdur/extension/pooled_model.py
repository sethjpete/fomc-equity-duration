"""Pooled characteristic model for stock-level duration.

r_{i,t} = a + (b0 + b' z_{i,t-1}) (-s_t) + e_{i,t};   D_hat_{i,m} = b0 + b' z_{i,m}

At each month end m the model is fit on all scheduled announcements strictly
before m, so the fitted durations are available in real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.stats.ols import ols


def design(panel: pd.DataFrame, zcols: list[str], scale: float = 0.01) -> pd.DataFrame:
    neg_s = -panel["s"] * scale
    X = pd.DataFrame({"neg_s": neg_s}, index=panel.index)
    for c in zcols:
        X[f"neg_s:{c}"] = neg_s * panel[c]
    return X


def fit_pooled(panel: pd.DataFrame, zcols: list[str], cutoff: pd.Timestamp,
               ret_col: str = "ret", scale: float = 0.01) -> pd.Series:
    """panel: one row per (permno, event) with s, ret and lagged z's. Returns [b0, b...]."""
    p = panel[panel["event"] < cutoff]
    res = ols(p[ret_col], design(p, zcols, scale))
    b = res.params.drop("const")
    b.index = ["b0", *zcols]
    return b


def fitted_duration(chars: pd.DataFrame, b: pd.Series) -> pd.Series:
    zcols = list(b.index[1:])
    return b["b0"] + chars[zcols].fillna(0).to_numpy() @ b[zcols].to_numpy()


def rolling_durations(panel: pd.DataFrame, chars_monthly: pd.DataFrame, zcols: list[str],
                      min_events: int = 40, ret_col: str = "ret",
                      scale: float = 0.01) -> pd.DataFrame:
    """Fitted duration for every (permno, month) once `min_events` meetings are available."""
    events = np.sort(panel["event"].unique())
    out, coefs = [], {}
    for m, g in chars_monthly.groupby("month"):
        if (events < m).sum() < min_events:
            continue
        b = fit_pooled(panel, zcols, m, ret_col, scale)
        coefs[m] = b
        out.append(g[["permno", "month"]].assign(D_hat=fitted_duration(g, b)))
    res = pd.concat(out, ignore_index=True)
    res.attrs["coefs"] = pd.DataFrame(coefs).T
    return res
