"""M2: FOMC day-0 durations for the market and size / book-to-market deciles, vs Chen (2022).

What Chen (2022, JFQA) does, from the paper and its internet appendix (read 2026-11-19):
- Each estimate is a ratio for one stock at one announcement, not a regression:
  D_it = -(dP/P) / [beta_it * dER_M,t + (1 - beta_it) * dRf_t]   (eq. 14).
  The return is the TAQ midpoint over [-30, +10] minutes around the statement; dRf is the
  fed funds futures surprise; dER_M comes from a VAR; beta from the past year of daily
  returns, shrunk to 100 beta-sorted portfolios.
- 1995-2016, 47 announcements whose futures surprise is at least 0.2bp in absolute value.
- Common stocks (shrcd 10/11) on NYSE/AMEX/Nasdaq, ex financials and utilities; the main
  tables also drop NYSE-20th-percentile microcaps. Estimates are trimmed to [1, 300] years.
- Portfolio durations are value-weighted averages of stock estimates.

What we do (the roadmap's daily, regression version): value-weighted day-0 close-to-close
returns of each portfolio, regressed on the SF Fed MPS surprise across scheduled meetings
(`replication.duration`). The two measures differ in window (40 minutes vs one day), in the
shock (dER vs MPS) and in estimator (ratio vs slope), so levels are not comparable. M2's check
is whether deciles **rank** the same way, using the same portfolios: we place Chen's
stock-level estimates into our deciles and value-weight them exactly as we weight returns.

Portfolio conventions:
- Formed at the end of month m-1 and used for announcements in month m; weights are ME at
  that month end.
- Size deciles use the NYSE ME breakpoints from Ken French (p10..p90).
- Book-to-market deciles use NYSE breakpoints of BE/ME computed from our panel. BE is the
  latest Compustat record available 6 months after fiscal year end (at most 18 months old);
  ME is at the formation month end. This is a monthly-updated B/M, not the June-rebalanced
  Fama-French sort.
- Deciles exclude financials (SIC 6000-6999) and utilities (4900-4999), as Chen does; the
  market portfolio keeps every common stock.
- Day-0 returns are not delisting-adjusted: a delisting rarely falls on an FOMC day, and the
  portfolios are value-weighted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.portfolios import assign_quantiles, value_weighted_returns
from fomcdur.replication.duration import portfolio_durations

SIZE_BREAKPOINTS = [f"p{q}" for q in range(10, 100, 10)]
MAX_BE_AGE_MONTHS = 12  # after the 6-month availability lag, i.e. at most 18 months old


def is_fin_util(sic: pd.Series) -> pd.Series:
    sic = pd.to_numeric(sic, errors="coerce")
    return sic.between(6000, 6999) | sic.between(4900, 4999)


# ---------------------------------------------------------------- CRSP extracts

def extract_from_daily(dsf: pd.DataFrame, day0: pd.DatetimeIndex
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From a daily CRSP chunk, keep what M2 needs.

    Returns
    - monthly: [permno, month, me, exchcd, siccd], the last day in each month with a price
      (ME in $ millions, month = month end);
    - event: [permno, date, ret] on the day-0 dates only.
    """
    d = dsf.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["me"] = d["prc"].abs() * d["shrout"] / 1000.0
    d["month"] = d["date"] + pd.offsets.MonthEnd(0)
    priced = d[d["me"] > 0].sort_values(["permno", "date"])
    monthly = priced.groupby(["permno", "month"], as_index=False).tail(1)
    monthly = monthly[["permno", "month", "me", "exchcd", "siccd"]].reset_index(drop=True)
    event = d.loc[d["date"].isin(pd.DatetimeIndex(day0)), ["permno", "date", "ret"]]
    return monthly, event.reset_index(drop=True)


def load_panels(day0: pd.DatetimeIndex, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `extract_from_daily` year by year over the CRSP parquet files (keeps memory low)."""
    from fomcdur.data import wrds_pull

    cols = ["permno", "date", "ret", "prc", "shrout", "exchcd", "siccd"]
    months, events = [], []
    for y in years:
        try:
            dsf = wrds_pull.load_crsp([y], columns=cols)
        except FileNotFoundError:
            continue
        m, e = extract_from_daily(dsf, day0)
        months.append(m)
        events.append(e)
    if not months:
        raise FileNotFoundError("No CRSP files; run `make data-wrds` first")
    monthly = pd.concat(months, ignore_index=True)
    monthly["permno"] = monthly["permno"].astype("int64")
    monthly["exchcd"] = pd.to_numeric(monthly["exchcd"], errors="coerce")
    monthly["siccd"] = pd.to_numeric(monthly["siccd"], errors="coerce")
    event = pd.concat(events, ignore_index=True)
    event["permno"] = event["permno"].astype("int64")
    return monthly, event


# ---------------------------------------------------------------- sorts

def add_book_to_market(monthly: pd.DataFrame, chars: pd.DataFrame) -> pd.DataFrame:
    """Attach BE (linked Compustat characteristics with permno) and bm = BE / ME."""
    right = chars[["permno", "available", "be"]].dropna(subset=["be"]).copy()
    right["permno"] = right["permno"].astype("int64")
    left = monthly.sort_values("month")
    df = pd.merge_asof(left, right.sort_values("available"), left_on="month",
                       right_on="available", by="permno", direction="backward")
    stale = df["month"] > df["available"] + pd.DateOffset(months=MAX_BE_AGE_MONTHS)
    df.loc[stale, "be"] = np.nan
    df["bm"] = df["be"] / df["me"]
    return df.drop(columns="available")


def add_size_decile(monthly: pd.DataFrame, breakpoints: pd.DataFrame) -> pd.Series:
    """Size decile 1 (small) .. 10 (big) from the French NYSE ME breakpoints of that month."""
    bp = breakpoints[SIZE_BREAKPOINTS]
    edges = bp.reindex(monthly["month"]).to_numpy()
    me = monthly["me"].to_numpy()
    dec = 1 + (me[:, None] >= edges).sum(axis=1).astype(float)
    dec[np.isnan(edges).any(axis=1) | np.isnan(me)] = np.nan
    return pd.Series(dec, index=monthly.index, name="size_dec")


def add_bm_decile(monthly: pd.DataFrame) -> pd.Series:
    """B/M decile 1 (growth) .. 10 (value) with NYSE breakpoints, within each month."""
    df = monthly.assign(nyse=monthly["exchcd"] == 1)
    return assign_quantiles(df, "bm", 10, by="month", breakpoint_mask="nyse").rename("bm_dec")


# ---------------------------------------------------------------- our durations

def _holdings(monthly: pd.DataFrame, port: str | None) -> pd.DataFrame:
    h = monthly[["permno", "month", "me"] + ([port] if port else [])].copy()
    h["port"] = h[port] if port else "MKT"
    return h.dropna(subset=["port", "me"])[["permno", "month", "port", "me"]]


def event_portfolio_returns(event: pd.DataFrame, monthly: pd.DataFrame,
                            port: str | None) -> pd.DataFrame:
    """Day-0 VW returns: day0 x portfolios. port=None gives the all-stock market."""
    out = value_weighted_returns(event, _holdings(monthly, port))
    if port:
        out.columns = out.columns.astype(int)
    return out


def durations_with_spread(event_ret: pd.DataFrame, s: pd.Series,
                          scale: float = 0.01) -> pd.DataFrame:
    """Durations per portfolio plus the top-minus-bottom ('10-1') spread portfolio."""
    r = event_ret.reindex(s.index)
    cols = sorted(r.columns)
    if len(cols) > 1:
        r = r.assign(**{"10-1": r[cols[-1]] - r[cols[0]]})
    return portfolio_durations(r, s, scale)


# ---------------------------------------------------------------- Chen's estimates

def _sas_date(col: pd.Series) -> pd.Series:
    """read_sas usually returns datetimes; unformatted SAS dates are days since 1960-01-01."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="D", origin="1960-01-01")
    return pd.to_datetime(col)


def chen_portfolio_durations(chen: pd.DataFrame, monthly: pd.DataFrame,
                             port: str | None) -> pd.DataFrame:
    """Chen's stock-level durations aggregated into our portfolios.

    For each announcement (eventdate) and portfolio: value-weighted mean (Chen's own
    convention), equal-weighted mean and median of the stock estimates, then averaged
    equally across announcements. A '10-1' row holds the spread of each statistic.
    """
    c = chen[["permno", "eventdate", "duration"]].dropna().copy()
    c["permno"] = c["permno"].astype("int64")
    c["eventdate"] = _sas_date(c["eventdate"])
    c["form_month"] = c["eventdate"] + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)
    h = _holdings(monthly, port).rename(columns={"month": "form_month"})
    m = c.merge(h, on=["permno", "form_month"], how="inner")
    m["wd"] = m["duration"] * m["me"]
    g = m.groupby(["eventdate", "port"])
    per_event = pd.DataFrame({
        "vw": g["wd"].sum() / g["me"].sum(),
        "ew": g["duration"].mean(),
        "median": g["duration"].median(),
        "n_stocks": g["duration"].size(),
    })
    out = per_event.groupby(level="port").mean()
    out["n_events"] = per_event.groupby(level="port").size()
    if port:
        out.index = out.index.astype(int)
        lo, hi = out.index.min(), out.index.max()
        spread = (out.loc[hi, ["vw", "ew", "median"]] - out.loc[lo, ["vw", "ew", "median"]])
        out.loc["10-1"] = spread
    return out


def compare(ours: pd.DataFrame, theirs: pd.DataFrame) -> dict:
    """Rank agreement across deciles (excluding the spread row) and spread signs."""
    deciles = [i for i in ours.index if i != "10-1" and i in theirs.index]
    a = ours.loc[deciles, "D"].astype(float)
    b = theirs.loc[deciles, "vw"].astype(float)
    return {
        "spearman_vw": float(a.rank().corr(b.rank())),
        "spearman_median": float(a.rank().corr(theirs.loc[deciles, "median"].astype(float).rank())),
        "spread_ours": float(ours.loc["10-1", "D"]) if "10-1" in ours.index else np.nan,
        "spread_ours_t": (float(ours.loc["10-1", "D"] / ours.loc["10-1", "se"])
                          if "10-1" in ours.index else np.nan),
        "spread_chen_vw": float(theirs.loc["10-1", "vw"]) if "10-1" in theirs.index else np.nan,
    }
