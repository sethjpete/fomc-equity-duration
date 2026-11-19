"""Intraday day-0 stock returns from TAQ NBBO midpoints, matched to CRSP permnos.

ret = mid_post / mid_pre - 1 over [s - 30, s + 10] minutes around the statement (Chen 2022).
The output has the same layout as the daily event returns ([permno, date, ret]), so the
portfolio and duration code in `chen_compare` works unchanged.
"""

from __future__ import annotations

import pandas as pd

MAX_ABS_RET = 0.5  # 40-minute moves beyond +/-50% are treated as data errors


def _sfx(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def match_permno(nbbo: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Attach permno by (date, sym_root, sym_suffix); keep the best match per symbol and,
    if a permno has several symbols, the one with the most quotes."""
    need = {"date", "sym_root", "sym_suffix", "permno"}
    if need - set(link.columns):
        raise KeyError(f"TAQ link table lacks {need - set(link.columns)}; has {list(link.columns)}")
    lk = link.copy()
    lk["date"] = pd.to_datetime(lk["date"])
    lk["sym_suffix"] = _sfx(lk["sym_suffix"])
    lk = lk.dropna(subset=["permno"])
    if "match_lvl" in lk.columns:
        lk = lk.sort_values("match_lvl")
    lk = lk.drop_duplicates(["date", "sym_root", "sym_suffix"])
    q = nbbo.copy()
    q["date"] = pd.to_datetime(q["date"])
    q["sym_suffix"] = _sfx(q["sym_suffix"])
    m = q.merge(lk[["date", "sym_root", "sym_suffix", "permno"]],
                on=["date", "sym_root", "sym_suffix"], how="inner")
    m["permno"] = m["permno"].astype("int64")
    m = m.sort_values("n_quotes", ascending=False).drop_duplicates(["date", "permno"])
    return m.reset_index(drop=True)


def event_returns(nbbo: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """[permno, date, ret] intraday returns; drops missing or implausible midpoints."""
    m = match_permno(nbbo, link)
    m = m[(m["mid_pre"] > 0) & (m["mid_post"] > 0)]
    m["ret"] = m["mid_post"] / m["mid_pre"] - 1
    m = m[m["ret"].abs() <= MAX_ABS_RET]
    return m[["permno", "date", "ret"]].reset_index(drop=True)
