"""FOMC event calendar and post-announcement windows.

Convention: day 0 is the announcement day (statements come out at ~2:00-2:15pm ET,
before the close, so the day-0 close-to-close return contains the news). If an
announcement date is not a trading day it maps to the next trading day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_to_trading_days(event_dates: pd.Series | pd.DatetimeIndex,
                          trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    idx = trading_days.searchsorted(pd.DatetimeIndex(event_dates))
    if (idx >= len(trading_days)).any():
        raise ValueError("Event after the last trading day")
    return trading_days[idx]


def cumulative_returns(returns: pd.DataFrame, events: pd.DatetimeIndex,
                       max_h: int) -> pd.DataFrame:
    """Compounded return from the close before day 0 through the close of day h.

    `returns` is dates x assets (simple returns). Output has a MultiIndex
    (event, h) and the same columns. Windows running past the sample end are NaN.
    """
    returns = returns.sort_index()
    pos = returns.index.get_indexer(events)
    if (pos < 0).any():
        raise ValueError("Align events to trading days first")
    log1p = np.log1p(returns.to_numpy(float))
    out = []
    for e, p in zip(events, pos):
        block = np.full((max_h + 1, returns.shape[1]), np.nan)
        avail = min(max_h + 1, len(returns) - p)
        block[:avail] = np.expm1(np.cumsum(log1p[p:p + avail], axis=0))
        out.append(pd.DataFrame(block, columns=returns.columns,
                                index=pd.MultiIndex.from_product([[e], range(max_h + 1)],
                                                                 names=["event", "h"])))
    return pd.concat(out)


def post_event_changes(levels: pd.DataFrame, events: pd.DatetimeIndex,
                       max_h: int) -> pd.DataFrame:
    """Change in a level series (e.g. yields) from the close of day 0 to day h.

    This is the yield control dy_{t+1 -> t+h}; it is zero at h = 0 by construction.
    """
    levels = levels.sort_index().ffill()
    pos = levels.index.get_indexer(events)
    if (pos < 0).any():
        raise ValueError("Events missing from the level series' dates")
    vals = levels.to_numpy(float)
    rows = []
    for e, p in zip(events, pos):
        for h in range(max_h + 1):
            q = p + h
            rows.append((e, h, *(vals[q] - vals[p] if q < len(vals) else [np.nan] * vals.shape[1])))
    df = pd.DataFrame(rows, columns=["event", "h", *levels.columns])
    return df.set_index(["event", "h"])
