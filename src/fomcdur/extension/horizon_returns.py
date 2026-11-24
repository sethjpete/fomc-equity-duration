"""M4: post-FOMC cumulative returns anchored at the pre-announcement intraday price.

Design (Seth, 2026-11-24): every horizon starts from the TAQ NBBO midpoint 30 minutes before
the statement (`mid_pre`), so
- h = 0 is the 40-minute response [s - 30, s + 10] (the M3 day-0 measure);
- "0c" runs from s - 30 to the day-0 CRSP close (the rest of the afternoon, including the
  press conference since 2011);
- h >= 1 compounds daily CRSP returns of days 1..h onto the 0c return.
So D(h) - D(0) is the drift after the 40-minute window, and nothing before s - 30 enters.

Stock returns: the day-0 close is |prc| (CRSP; a negative price is a bid/ask midpoint), the
anchor-to-close return is |prc_0| / mid_pre - 1 (dropped if beyond +/-50%, as in
`intraday.MAX_ABS_RET`). Missing daily returns after day 0 (a delisting or a halt) count as
zero, so the position is held as cash. Portfolios are buy-and-hold: weights are ME at the
month end before the meeting, applied to each stock's cumulative return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.replication.intraday import MAX_ABS_RET, match_permno

CLOSE0 = -1  # internal code for the "0c" horizon (anchor to the day-0 close)


def anchor_prices(nbbo: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """[permno, date, mid_pre, ret40]: anchor midpoint and the 40-minute return."""
    m = match_permno(nbbo, link)
    m = m[(m["mid_pre"] > 0) & (m["mid_post"] > 0)].copy()
    m["mid_pre"] = m["mid_pre"].astype(float)
    m["ret40"] = m["mid_post"].astype(float) / m["mid_pre"] - 1
    m = m[m["ret40"].abs() <= MAX_ABS_RET]
    return m[["permno", "date", "mid_pre", "ret40"]].reset_index(drop=True)


def event_windows(day0: pd.DatetimeIndex, trading_days: pd.DatetimeIndex,
                  max_h: int) -> pd.DataFrame:
    """[event, h, date] for h = 0..max_h trading days after each day 0."""
    td = pd.DatetimeIndex(trading_days).sort_values()
    pos = td.get_indexer(pd.DatetimeIndex(day0))
    if (pos < 0).any():
        raise ValueError("Every day 0 must be a trading day")
    rows = [(e, h, td[p + h]) for e, p in zip(day0, pos) for h in range(max_h + 1)
            if p + h < len(td)]
    return pd.DataFrame(rows, columns=["event", "h", "date"])


def stock_cumulative(anchor: pd.DataFrame, daily: pd.DataFrame, windows: pd.DataFrame,
                     max_h: int) -> pd.DataFrame:
    """Cumulative stock returns from the anchor, long format [permno, event, h, cum].

    anchor: [permno, date (= day 0), mid_pre, ret40]
    daily:  CRSP [permno, date, ret, prc] covering the windows
    h = CLOSE0 holds the anchor-to-day-0-close return; h = 0 the 40-minute return.
    """
    a = anchor.rename(columns={"date": "event"})
    d0 = daily.rename(columns={"date": "event"})[["permno", "event", "prc"]]
    a = a.merge(d0, on=["permno", "event"], how="inner")
    a["c0"] = a["prc"].abs() / a["mid_pre"] - 1
    a = a[(a["prc"].abs() > 0) & (a["c0"].abs() <= MAX_ABS_RET)]

    w = windows[windows["h"] >= 1]
    dr = daily[["permno", "date", "ret"]].merge(w, on="date")
    dr = dr.merge(a[["permno", "event"]], on=["permno", "event"])
    wide = dr.pivot_table(index=["permno", "event"], columns="h", values="ret",
                          aggfunc="first")
    wide = wide.reindex(columns=range(1, max_h + 1))
    base = a.set_index(["permno", "event"])
    wide = wide.reindex(base.index)
    # Windows that run past the end of the CRSP data stay NaN; other gaps count as zero.
    last = daily["date"].max()
    ends = windows.groupby("event")["date"].max()
    complete = ends.reindex(base.index.get_level_values("event")).to_numpy() <= last
    g = np.log1p(wide.fillna(0.0).to_numpy(float))
    cum = np.expm1(np.log1p(base["c0"].to_numpy())[:, None] + np.cumsum(g, axis=1))
    have_h = windows.groupby("event")["h"].max()
    hmax = have_h.reindex(base.index.get_level_values("event")).to_numpy()
    cols = np.arange(1, max_h + 1)
    cum[(cols[None, :] > hmax[:, None]) | ~complete[:, None]] = np.nan
    out = pd.DataFrame(cum, index=base.index, columns=cols)
    out[CLOSE0] = base["c0"].to_numpy()
    out[0] = base["ret40"].to_numpy()
    long = out.stack(future_stack=True).rename("cum").reset_index()
    long.columns = ["permno", "event", "h", "cum"]
    return long


def portfolio_cumulative(stock_cum: pd.DataFrame, monthly: pd.DataFrame,
                         port: str | None) -> pd.DataFrame:
    """Buy-and-hold VW cumulative returns, index (event, h), columns = portfolios."""
    h = monthly[["permno", "month", "me"] + ([port] if port else [])].copy()
    h["port"] = h[port] if port else "MKT"
    h = h.dropna(subset=["port", "me"])
    h = h[h["me"] > 0]
    h["hold_month"] = h["month"] + pd.offsets.MonthEnd(1)
    c = stock_cum.assign(hold_month=stock_cum["event"] + pd.offsets.MonthEnd(0))
    m = c.merge(h[["permno", "hold_month", "port", "me"]], on=["permno", "hold_month"])
    m = m.dropna(subset=["cum"])
    m["wc"] = m["cum"] * m["me"]
    g = m.groupby(["event", "h", "port"])
    out = (g["wc"].sum() / g["me"].sum()).unstack("port").sort_index()
    if port:
        out.columns = out.columns.astype(int)
    return out


def level_changes_from(levels: pd.DataFrame, windows: pd.DataFrame,
                       base_offset: int = 0) -> pd.DataFrame:
    """Change in a daily level series (e.g. yields, pp) from the close of day `base_offset`
    (relative to day 0; -1 = the day before) to the close of day h. Index (event, h)."""
    lv = levels.sort_index().ffill()
    td = lv.index
    rows = []
    for e, g in windows.groupby("event"):
        p0 = td.searchsorted(e) + base_offset
        if p0 < 0:
            continue
        base = lv.iloc[p0]
        for r in g.itertuples():
            v = lv.reindex([r.date]).iloc[0] - base
            rows.append((e, r.h, *v.to_numpy()))
    return pd.DataFrame(rows, columns=["event", "h", *lv.columns]).set_index(["event", "h"])


def cumulative_factors(factors: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Factor returns compounded over days 1..h (zero at h = 0). Index (event, h)."""
    f = windows.merge(factors, left_on="date", right_index=True, how="left")
    f = f.sort_values(["event", "h"])
    cols = list(factors.columns)
    f.loc[f["h"] == 0, cols] = 0.0
    f[cols] = np.expm1(np.log1p(f[cols].fillna(0.0)).groupby(f["event"]).cumsum())
    return f.set_index(["event", "h"])[cols]
