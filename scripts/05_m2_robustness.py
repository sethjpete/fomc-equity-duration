"""M2 sanity checks: are our day-0 durations robust, and what do they mean in yield units?

Usage: python scripts/05_m2_robustness.py
Writes results/tables/m2_robustness.md.

1. Units: day-0 change in the 2y and 10y Treasury yields on MPS. Dividing D by the 2y
   pass-through turns "% per pp of MPS" into "% per pp of the 2y yield" (years).
2. Robustness of the market, size 10-1 and B/M 10-1 durations: ex 2008-09 and 2020,
   ex the 5 largest |MPS| meetings, 1994-2007 vs 2008-2023, equal-weighted portfolios.
3. Influence: the single meeting whose removal moves each estimate most.
4. Asymmetry: tightening vs easing surprises.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fomcdur.clean.characteristics import compustat_characteristics, link_to_permno
from fomcdur.clean.events import build_calendar
from fomcdur.config import RESULTS, raw_dir, settings
from fomcdur.data import fred, french, sffed
from fomcdur.replication import chen_compare as cc
from fomcdur.replication.duration import event_duration

S = settings()
END = pd.Timestamp(S["sample"]["end"])
SCALE = S["duration"]["surprise_scale"]


def ew_returns(event: pd.DataFrame, monthly: pd.DataFrame, port: str) -> pd.DataFrame:
    h = monthly[["permno", "month", port]].dropna().rename(columns={port: "port"})
    h["hold_month"] = h["month"] + pd.offsets.MonthEnd(1)
    e = event.assign(hold_month=event["date"] + pd.offsets.MonthEnd(0))
    m = e.merge(h[["permno", "hold_month", "port"]], on=["permno", "hold_month"]).dropna()
    out = m.groupby(["date", "port"])["ret"].mean().unstack("port")
    out.columns = out.columns.astype(int)
    return out


def spread(r: pd.DataFrame) -> pd.Series:
    return r[r.columns.max()] - r[r.columns.min()]


def est(r: pd.Series, s: pd.Series) -> str:
    idx = s.index.intersection(r.dropna().index)
    d = event_duration(r.loc[idx], s.loc[idx], SCALE)
    return f"{d['D']:.1f} ({d['se']:.1f}) n={d['n']}"


def main() -> None:
    out = ["# M2 robustness and units", f"\nGenerated {date.today()}. Cells: D (HC1 se) n."]
    td = french.load_factors().index
    surp = sffed.load(scheduled_only=True)
    surp = surp[surp["date"] <= END]
    cal = build_calendar(surp, td).set_index("day0")
    s = cal["mps"]

    monthly, event = cc.load_panels(cal.index, list(range(1993, END.year + 1)))
    funda = pd.read_parquet(raw_dir("compustat") / "funda.parquet")
    link = pd.read_parquet(raw_dir("compustat") / "ccm_link.parquet")
    monthly = cc.add_book_to_market(monthly, link_to_permno(compustat_characteristics(funda),
                                                            link))
    monthly["size_dec"] = cc.add_size_decile(monthly, french.load_me_breakpoints())
    monthly["bm_dec"] = cc.add_bm_decile(monthly)
    monthly.loc[cc.is_fin_util(monthly["siccd"]), ["size_dec", "bm_dec"]] = np.nan

    series = {
        "Market (VW)": cc.event_portfolio_returns(event, monthly, None)["MKT"],
        "Size 10-1 (VW)": spread(cc.event_portfolio_returns(event, monthly, "size_dec")),
        "B/M 10-1 (VW)": spread(cc.event_portfolio_returns(event, monthly, "bm_dec")),
        "Size 10-1 (EW)": spread(ew_returns(event, monthly, "size_dec")),
        "B/M 10-1 (EW)": spread(ew_returns(event, monthly, "bm_dec")),
    }

    # 1. Units
    y = fred.load().sort_index().ffill()
    prev = y.shift(1)
    dy = (y - prev).reindex(s.index)
    out.append("\n## 1. Yield pass-through on day 0 (pp of yield per pp of MPS)\n")
    rows = {}
    for c in dy.columns:
        d = event_duration(-dy[c], s, scale=1.0)  # slope of dy on s
        rows[c] = {"slope": d["D"], "se": d["se"], "R2": d["r2"], "n": d["n"]}
    pt = pd.DataFrame(rows).T
    out.append(pt.to_markdown(floatfmt=".3g"))
    b2 = float(pt.loc["DGS2", "slope"])
    mk = event_duration(series["Market (VW)"].reindex(s.index), s, SCALE)
    out.append(f"\nMarket D per pp of MPS = {mk['D']:.2f}; per pp of the 2y yield = "
               f"{mk['D'] / b2:.2f} (D / {b2:.3f}).")

    # 2. Robustness
    yr = s.index.year
    variants = {
        "baseline": s,
        "ex 2008-09 and 2020": s[~np.isin(yr, [2008, 2009, 2020])],
        "ex 5 largest |MPS|": s.drop(s.abs().nlargest(5).index),
        "1994-2007": s[yr <= 2007],
        "2008-2023": s[yr >= 2008],
        "tightening (MPS > 0)": s[s > 0],
        "easing (MPS < 0)": s[s < 0],
    }
    tab = pd.DataFrame({k: {name: est(r, v) for name, r in series.items()}
                        for k, v in variants.items()}).T
    out.append("\n## 2. Robustness\n")
    out.append(tab.to_markdown())

    # 3. Influence
    rows = {}
    for name, r in series.items():
        base = event_duration(r.reindex(s.index), s, SCALE)["D"]
        deltas = {d: event_duration(r.reindex(s.index).drop(d), s.drop(d), SCALE)["D"] - base
                  for d in s.index}
        dmax = max(deltas, key=lambda k: abs(deltas[k]))
        rows[name] = {"baseline D": base, "most influential meeting": dmax.date(),
                      "MPS there (pp)": s[dmax], "D without it": base + deltas[dmax]}
    out.append("\n## 3. Most influential single meeting\n")
    out.append(pd.DataFrame(rows).T.to_markdown(floatfmt=".3g"))
    top = s.abs().nlargest(8)
    out.append("\nLargest |MPS| meetings: " + ", ".join(f"{d.date()} ({s[d]:+.3f})"
                                                       for d in top.index))

    path = RESULTS / "tables" / "m2_robustness.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
