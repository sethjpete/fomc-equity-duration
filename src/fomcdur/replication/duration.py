"""Chen (2022) FOMC-day effective duration: r_{p,t} = a_p - D_p s_t + e_{p,t}."""

from __future__ import annotations

import pandas as pd

from fomcdur.stats.ols import ols


def event_duration(r: pd.Series, s: pd.Series, scale: float = 0.01) -> dict:
    """Regress event-day returns on the (scaled) surprise; D is minus the slope.

    s is in percentage points; `scale` converts to decimals so D is in years.
    """
    x = (-s * scale).rename("neg_s")
    res = ols(r, x.to_frame())
    return {"D": res.params["neg_s"], "se": res.bse["neg_s"], "n": res.nobs, "r2": res.r2}


def portfolio_durations(event_returns: pd.DataFrame, s: pd.Series,
                        scale: float = 0.01) -> pd.DataFrame:
    """event_returns: events x portfolios (day-0 returns). Returns one row per portfolio."""
    rows = {p: event_duration(event_returns[p], s, scale) for p in event_returns.columns}
    return pd.DataFrame(rows).T
