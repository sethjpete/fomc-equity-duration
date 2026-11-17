"""Stock universe: CRSP common shares, delisting-adjusted returns, microcap flag."""

from __future__ import annotations

import pandas as pd


def adjust_for_delisting(dsf: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    """Combine ret with dlret on the delisting date: (1+ret)(1+dlret)-1."""
    df = dsf.merge(delist[["permno", "date", "dlret"]], on=["permno", "date"], how="left")
    has = df["dlret"].notna()
    df.loc[has, "ret"] = (1 + df.loc[has, "ret"].fillna(0)) * (1 + df.loc[has, "dlret"]) - 1
    return df.drop(columns="dlret")


def add_market_equity(dsf: pd.DataFrame) -> pd.DataFrame:
    """ME in $ millions (CRSP prc is negative for bid/ask midpoints; shrout in 000s)."""
    dsf = dsf.copy()
    dsf["me"] = dsf["prc"].abs() * dsf["shrout"] / 1000.0
    return dsf


def microcap_flag(me_month_end: pd.DataFrame, breakpoints: pd.DataFrame,
                  pct: int = 20) -> pd.Series:
    """True if a stock's month-end ME is below the NYSE `pct` percentile.

    me_month_end: columns [permno, month, me]; breakpoints from french.load_me_breakpoints().
    """
    bp = breakpoints[f"p{pct}"].rename("bp")
    df = me_month_end.merge(bp, left_on="month", right_index=True, how="left")
    return (df["me"] < df["bp"]).set_axis(me_month_end.index)
