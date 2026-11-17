"""Portfolio construction shared by the replication and the extension."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_quantiles(df: pd.DataFrame, signal: str, n: int, by: str = "month",
                     breakpoint_mask: str | None = None) -> pd.Series:
    """Quantile labels 1..n of `signal` within each `by` group.

    breakpoint_mask: optional boolean column; breakpoints are computed only from
    rows where it is True (e.g. NYSE stocks) and applied to everyone.
    """
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(by):
        base = g.loc[g[breakpoint_mask].astype(bool), signal] if breakpoint_mask else g[signal]
        base = base.dropna()
        if len(base) < n:
            continue
        edges = np.quantile(base, np.linspace(0, 1, n + 1)[1:-1])
        q = np.searchsorted(edges, g[signal].to_numpy(), side="right") + 1
        out.loc[g.index] = np.where(g[signal].isna(), np.nan, q)
    return out


def neutralize(df: pd.DataFrame, signal: str, group: str, by: str = "month") -> pd.Series:
    """Z-score `signal` within (by, group), for the industry-neutral sort."""
    g = df.groupby([by, group])[signal]
    return (df[signal] - g.transform("mean")) / g.transform("std")


def value_weighted_returns(daily: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """Daily value-weighted portfolio returns.

    daily:    [permno, date, ret]
    holdings: [permno, month, port, me]  formed at the end of `month`, held next month.
    Returns dates x ports. Weights are fixed at formation (no intra-month drift
    adjustment; a buy-and-hold refinement is a robustness item).
    """
    h = holdings.copy()
    h["hold_month"] = h["month"] + pd.offsets.MonthEnd(1)
    d = daily.copy()
    d["hold_month"] = d["date"] + pd.offsets.MonthEnd(0)
    m = d.merge(h[["permno", "hold_month", "port", "me"]], on=["permno", "hold_month"])
    m = m.dropna(subset=["ret", "me"])
    m["wr"] = m["ret"] * m["me"]
    g = m.groupby(["date", "port"])
    return (g["wr"].sum() / g["me"].sum()).unstack("port").sort_index()


def high_minus_low(port_returns: pd.DataFrame) -> pd.Series:
    cols = sorted(port_returns.columns)
    return (port_returns[cols[-1]] - port_returns[cols[0]]).rename("HL")
