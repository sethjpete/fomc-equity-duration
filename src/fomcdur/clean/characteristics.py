"""Firm characteristics for the pooled duration model.

Timing: an annual Compustat record becomes usable 6 months after its fiscal
year end (datadate), and stays in use until the next record is usable. All
characteristics are z-scored across stocks within each month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AVAILABILITY_LAG_MONTHS = 6


def book_equity(f: pd.DataFrame) -> pd.Series:
    """Fama-French book equity: SE + deferred taxes - preferred stock."""
    se = f["seq"].fillna(f["ceq"] + f["pstk"].fillna(0)).fillna(f["at"] - f["lt"])
    ps = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0)
    be = se + f["txditc"].fillna(0) - ps
    return be.where(be > 0)


def compustat_characteristics(funda: pd.DataFrame) -> pd.DataFrame:
    f = funda.copy()
    f["be"] = book_equity(f)
    f["op_prof"] = (f["revt"] - f["cogs"].fillna(0) - f["xsga"].fillna(0)
                    - f["xint"].fillna(0)) / f["be"]
    f["leverage"] = (f["dltt"].fillna(0) + f["dlc"].fillna(0)) / f["at"]
    f["available"] = (f["datadate"] + pd.DateOffset(months=AVAILABILITY_LAG_MONTHS)
                      + pd.offsets.MonthEnd(0))
    return f[["gvkey", "datadate", "available", "be", "op_prof", "leverage", "sich"]]


def link_to_permno(chars: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Attach permno using the CCM link valid at datadate."""
    link = link.copy()
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    df = chars.merge(link, on="gvkey", how="inner")
    ok = (df["datadate"] >= df["linkdt"]) & (df["datadate"] <= df["linkenddt"])
    return df.loc[ok].drop(columns=["linkdt", "linkenddt", "linktype", "linkprim"])


def monthly_panel(chars: pd.DataFrame, me_month_end: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: for each (permno, month) take the latest record already available."""
    left = me_month_end.sort_values("month")
    right = chars.sort_values("available")
    df = pd.merge_asof(left, right, left_on="month", right_on="available",
                       by="permno", direction="backward")
    df["log_me"] = np.log(df["me"])
    df["log_bm"] = np.log(df["be"] / df["me"])
    return df


def zscore_by_month(df: pd.DataFrame, cols: list[str], month_col: str = "month",
                    winsor: float = 0.01) -> pd.DataFrame:
    """Winsorize at (winsor, 1-winsor) then z-score within each month."""
    out = df.copy()
    for c in cols:
        g = out.groupby(month_col)[c]
        lo, hi = g.transform(lambda x: x.quantile(winsor)), g.transform(lambda x: x.quantile(1 - winsor))
        x = out[c].clip(lo, hi)
        g2 = x.groupby(out[month_col])
        out[f"z_{c}"] = (x - g2.transform("mean")) / g2.transform("std")
    return out
