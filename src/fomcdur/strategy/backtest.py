"""Event strategy: after each announcement, short HL after tightening, long after easing,
entering at the day-0 close and holding for `h` trading days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.stats.ols import ols


def event_positions(dates: pd.DatetimeIndex, events: pd.DatetimeIndex, s: pd.Series,
                    hold: int, size_by_surprise: bool = False) -> pd.Series:
    """Daily position in HL (+1 long, -1 short, 0 flat) for days 1..hold after each event."""
    pos = pd.Series(0.0, index=dates)
    loc = dates.get_indexer(events)
    for e, p in zip(events, loc):
        if p < 0 or s.loc[e] == 0:
            continue
        w = -np.sign(s.loc[e]) * (abs(s.loc[e]) if size_by_surprise else 1.0)
        pos.iloc[p + 1: p + 1 + hold] = w
    return pos


def strategy_returns(hl: pd.Series, pos: pd.Series, cost_bps_per_side: float = 0.0) -> pd.DataFrame:
    """Gross and net daily returns. Cost is charged on |change in position| for both legs."""
    gross = pos * hl.reindex(pos.index).fillna(0)
    trades = pos.diff().abs().fillna(pos.abs())
    cost = trades * 2 * cost_bps_per_side / 1e4  # two legs (long and short side of HL)
    return pd.DataFrame({"pos": pos, "gross": gross, "net": gross - cost, "turnover": trades})


def summarize(ret: pd.Series, factors: pd.DataFrame | None = None,
              periods_per_year: int = 252) -> dict:
    ret = ret.dropna()
    out = {
        "ann_mean": ret.mean() * periods_per_year,
        "ann_vol": ret.std() * np.sqrt(periods_per_year),
    }
    out["sharpe"] = out["ann_mean"] / out["ann_vol"] if out["ann_vol"] > 0 else np.nan
    if factors is not None:
        cols = [c for c in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in factors]
        res = ols(ret, factors.reindex(ret.index)[cols])
        out["alpha_ann"] = res.params["const"] * periods_per_year
        out["alpha_t"] = res.tvalues["const"]
    return out


def by_decade(ret: pd.Series, factors: pd.DataFrame | None = None) -> pd.DataFrame:
    dec = (ret.index.year // 10) * 10
    return pd.DataFrame({f"{d}s": summarize(r, factors) for d, r in ret.groupby(dec)}).T
