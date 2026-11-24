"""M3: FOMC day-0 durations of portfolios (French 49 industries, characteristic deciles).

Design (Seth, 2026-11-20 and 2026-11-24):
- Durations are estimated on portfolios, not stocks: one 40-minute return per stock and
  meeting is mostly noise (M2), and value-weighted portfolios average it out.
- Day-0 return = intraday TAQ NBBO midpoint change over [s - 30, s + 10] minutes around the
  statement (Chen's window). Sample: scheduled meetings from 2003-09-10 (first Millisecond
  TAQ day) to 2023-12 with an SF Fed MPS surprise. No daily or pre-2003 measure.
- D_p is minus the slope of r_{p,t} on MPS_t x 0.01 (years), HC1 errors (`duration.py`).
- Portfolios are formed at the end of month m-1 and held for meetings in month m, weighted
  by ME at that month end. Industries: FF49 from the CRSP header SIC (`french.assign_ff49`),
  all common stocks. Characteristic deciles: NYSE breakpoints within each month, excluding
  financials and utilities (as in M2).
- Every result comes with the robustness set: all meetings, without the 5 largest |MPS|,
  without 2008-09 and 2020, and 2003-2007 vs 2008-2023 (M2 showed results lean on a few
  large surprises and on the post-2008 period).

Is there real cross-sectional dispersion in duration, or just noise? Three checks:
- signal share: 1 - mean(se^2) / var(D) across portfolios (share of the dispersion in
  estimated durations that is not sampling noise);
- split-half reliability: correlate durations estimated on odd vs even meetings across
  portfolios, Spearman-Brown corrected to the full sample;
- permutation test of equal durations: under H0 all D_p equal the market's, so returns in
  excess of the market have no slope on the surprise. Statistic = sum of squared t-stats of
  those slopes; null distribution by permuting surprises across meetings (keeps the
  cross-portfolio correlation of returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.portfolios import assign_quantiles
from fomcdur.replication.duration import event_duration

MAX_CHAR_AGE_MONTHS = 12  # after the 6-month availability lag, as in M2
SAMPLE_ORDER = ["all", "ex top-5 |MPS|", "ex 2008-09, 2020", "2003-2007", "2008-2023"]


# ---------------------------------------------------------------- sorts

def add_characteristics(monthly: pd.DataFrame, chars: pd.DataFrame,
                        cols: tuple[str, ...] = ("be", "op_prof", "leverage")) -> pd.DataFrame:
    """As-of merge of linked Compustat characteristics (latest record available at the month
    end, at most MAX_CHAR_AGE_MONTHS past availability); adds bm = BE / ME."""
    right = chars[["permno", "available", *cols]].dropna(subset=["permno"]).copy()
    right["permno"] = right["permno"].astype("int64")
    right = right.sort_values("available").drop_duplicates(["permno", "available"], keep="last")
    df = pd.merge_asof(monthly.sort_values("month"), right, left_on="month",
                       right_on="available", by="permno", direction="backward")
    stale = df["month"] > df["available"] + pd.DateOffset(months=MAX_CHAR_AGE_MONTHS)
    df.loc[stale, list(cols)] = np.nan
    df["bm"] = df["be"] / df["me"]
    return df.drop(columns="available")


def add_deciles(monthly: pd.DataFrame, signals: dict[str, str], exclude: pd.Series
                ) -> pd.DataFrame:
    """Decile columns (1..10, NYSE breakpoints within month) for each {new_col: signal};
    rows flagged in `exclude` get no decile."""
    df = monthly.assign(nyse=monthly["exchcd"] == 1)
    base = df[~exclude]
    for col, sig in signals.items():
        df[col] = assign_quantiles(base, sig, 10, by="month", breakpoint_mask="nyse"
                                   ).reindex(df.index)
    return df.drop(columns="nyse")


# ---------------------------------------------------------------- returns

def event_portfolio_returns(event: pd.DataFrame, monthly: pd.DataFrame, port: str | None,
                            min_stocks: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VW day-0 returns (meetings x portfolios) and stock counts. port=None: all stocks.

    Portfolio-meetings with fewer than `min_stocks` stocks are set to NaN.
    """
    h = monthly[["permno", "month", "me"] + ([port] if port else [])].copy()
    h["port"] = h[port] if port else "MKT"
    h = h.dropna(subset=["port", "me"])
    h = h[h["me"] > 0]
    h["hold_month"] = h["month"] + pd.offsets.MonthEnd(1)
    e = event.assign(hold_month=event["date"] + pd.offsets.MonthEnd(0))
    m = e.merge(h[["permno", "hold_month", "port", "me"]], on=["permno", "hold_month"])
    m = m.dropna(subset=["ret"])
    m["wr"] = m["ret"] * m["me"]
    g = m.groupby(["date", "port"])
    ret = (g["wr"].sum() / g["me"].sum()).unstack("port").sort_index()
    n = g.size().unstack("port").reindex_like(ret)
    ret = ret.where(n >= min_stocks)
    if port:
        ret.columns = ret.columns.astype(int)
        n.columns = n.columns.astype(int)
    return ret, n


def add_spread(r: pd.DataFrame, label: str = "10-1") -> pd.DataFrame:
    cols = sorted(c for c in r.columns if not isinstance(c, str))
    return r.assign(**{label: r[cols[-1]] - r[cols[0]]})


# ---------------------------------------------------------------- estimation

def robustness_samples(s: pd.Series, n_top: int = 5) -> dict[str, pd.Series]:
    """The standard robustness set of surprise series (index = day 0)."""
    y = s.index.year
    return {
        "all": s,
        f"ex top-{n_top} |MPS|": s.drop(s.abs().nlargest(n_top).index),
        "ex 2008-09, 2020": s[~np.isin(y, [2008, 2009, 2020])],
        "2003-2007": s[y <= 2007],
        "2008-2023": s[y >= 2008],
    }


def duration_table(r: pd.DataFrame, s: pd.Series, scale: float = 0.01) -> pd.DataFrame:
    """Long table: one row per (sample, portfolio) with D, se, t, n, r2."""
    rows = []
    for name, ss in robustness_samples(s).items():
        for p in r.columns:
            y = r[p].reindex(ss.index)
            ok = y.notna()
            if ok.sum() < 5:
                continue
            d = event_duration(y[ok], ss[ok], scale)
            rows.append({"sample": name, "portfolio": p, "D": d["D"], "se": d["se"],
                         "t": d["D"] / d["se"], "n": d["n"], "r2": d["r2"]})
    return pd.DataFrame(rows)


def wide(tab: pd.DataFrame, fmt: str = "{:.1f} ({:.1f})") -> pd.DataFrame:
    """Portfolio x sample cells 'D (se)', main sample first."""
    t = tab.assign(cell=[fmt.format(d, s) for d, s in zip(tab["D"], tab["se"])])
    out = t.pivot(index="portfolio", columns="sample", values="cell")
    return out[[c for c in SAMPLE_ORDER if c in out.columns]]


def signal_share(D: pd.Series, se: pd.Series) -> float:
    """1 - mean(se^2) / var(D): share of cross-portfolio dispersion that is not noise."""
    return float(1 - (se**2).mean() / D.var(ddof=1))


def split_half_reliability(r: pd.DataFrame, s: pd.Series, scale: float = 0.01) -> dict:
    """Correlation across portfolios of D on odd vs even meetings (chronological), and the
    Spearman-Brown full-sample reliability 2c / (1 + c)."""
    s = s.sort_index()
    halves = [s.iloc[0::2], s.iloc[1::2]]
    D = [pd.Series({p: event_duration(r[p].reindex(h.index), h, scale)["D"]
                    for p in r.columns}) for h in halves]
    c = float(D[0].corr(D[1]))
    return {"corr_odd_even": c, "spearman_brown": 2 * c / (1 + c),
            "rank_corr": float(D[0].rank().corr(D[1].rank()))}


def _slopes_t(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """t-stats (HC1) of each column of R regressed on x with a constant; R: T x N."""
    T = len(x)
    xd = x - x.mean()
    sxx = (xd**2).sum()
    Rd = R - R.mean(axis=0)
    b = xd @ Rd / sxx
    e = Rd - np.outer(xd, b)
    var = (xd[:, None] ** 2 * e**2).sum(axis=0) / sxx**2 * T / (T - 2)
    return b / np.sqrt(var)


def equal_duration_test(r: pd.DataFrame, r_mkt: pd.Series, s: pd.Series,
                        n_perm: int = 5000, seed: int = 0) -> dict:
    """Permutation test of H0: every portfolio has the market's duration.

    Uses r_p - r_mkt on meetings where all portfolios have returns. Statistic: sum of squared
    t-stats of the slopes on -s. p-value: share of permutations of s with a larger statistic.
    """
    X = r.sub(r_mkt, axis=0).reindex(s.index).dropna(axis=0, how="any")
    x = -s.reindex(X.index).to_numpy(float)
    R = X.to_numpy(float)
    stat = float((_slopes_t(R, x) ** 2).sum())
    rng = np.random.default_rng(seed)
    null = np.array([(_slopes_t(R, rng.permutation(x)) ** 2).sum() for _ in range(n_perm)])
    return {"stat": stat, "df": R.shape[1], "null_mean": float(null.mean()),
            "p_perm": float((1 + (null >= stat).sum()) / (1 + n_perm)), "n_meetings": len(x)}
