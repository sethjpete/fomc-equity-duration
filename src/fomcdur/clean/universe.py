"""Stock universe: CRSP common shares, delisting-adjusted returns, microcap flag."""

from __future__ import annotations

import pandas as pd

ACTIVE_CODE = 100  # dsedelist row for a security still trading: not a delisting


def is_performance_delisting(code: pd.Series) -> pd.Series:
    """CRSP codes 500 and 520-584: dropped for poor performance (Shumway 1997)."""
    return (code == 500) | code.between(520, 584)


def impute_delisting_returns(delist: pd.DataFrame, exchcd: pd.Series | None = None,
                             nyse_amex: float = -0.30, nasdaq: float = -0.55) -> pd.DataFrame:
    """Fill missing dlret for performance delistings.

    Shumway (1997) for NYSE/AMEX (-30%) and Shumway and Warther (1999) for Nasdaq (-55%).
    Other missing dlret (mergers, exchange moves, active code 100) are left missing, which
    means no extra return. Adds `dlret_imputed`.
    """
    d = delist.copy()
    perf = is_performance_delisting(d["dlstcd"]) & d["dlret"].isna()
    if exchcd is None:
        fill = pd.Series(nyse_amex, index=d.index)
    else:
        fill = pd.Series(nyse_amex, index=d.index).where(exchcd.reindex(d.index) != 3, nasdaq)
    d["dlret_imputed"] = perf
    d.loc[perf, "dlret"] = fill[perf]
    return d


def adjust_for_delisting(dsf: pd.DataFrame, delist: pd.DataFrame,
                         tolerance_days: int = 31) -> pd.DataFrame:
    """Compound the delisting return into the stock's last daily return: (1+ret)(1+dlret)-1.

    The delisting date (dlstdt) is often after the last day the stock traded, so dlret is
    attached to the last dsf row on or before dlstdt (within `tolerance_days`). Missing
    dlret for performance delistings is imputed first (see impute_delisting_returns).
    Code 100 rows (still active) are ignored.
    """
    d = delist[delist["dlstcd"] != ACTIVE_CODE]
    last = (dsf.sort_values("date").groupby("permno")[["permno", "date", "exchcd"]].tail(1)
            if "exchcd" in dsf.columns else
            dsf.sort_values("date").groupby("permno")[["permno", "date"]].tail(1))
    d = d.merge(last.rename(columns={"date": "last_date"}), on="permno", how="inner")
    d = d[(d["last_date"] <= d["date"])
          & (d["date"] - d["last_date"] <= pd.Timedelta(days=tolerance_days))]
    d = impute_delisting_returns(d, d["exchcd"] if "exchcd" in d.columns else None)
    d = d.dropna(subset=["dlret"])[["permno", "last_date", "dlret"]]
    df = dsf.merge(d.rename(columns={"last_date": "date"}), on=["permno", "date"], how="left")
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
