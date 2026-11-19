"""Day-0 durations from intraday TAQ returns vs daily CRSP returns vs Chen (2022).

Usage: python scripts/06_intraday_durations.py
Needs the TAQ pull (`python scripts/01_pull_data.py taq`) plus the M2 inputs.
Writes results/tables/m2_intraday.md.

Same meetings for both measures (scheduled, from 2003-09 with TAQ data), same portfolios
(formed at the prior month end, VW by ME), same regression on MPS. Only the return window
differs: [s-30, s+10] minutes (Chen's) vs close to close.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fomcdur.clean.characteristics import compustat_characteristics, link_to_permno
from fomcdur.clean.events import align_to_trading_days
from fomcdur.config import RESULTS, raw_dir, settings
from fomcdur.data import chen, french, taq
from fomcdur.replication import chen_compare as cc
from fomcdur.replication import intraday
from fomcdur.replication.duration import event_duration

S = settings()
SCALE = S["duration"]["surprise_scale"]
LO, HI = 1.0, 300.0


def fmt(d: dict) -> str:
    return f"{d['D']:.2f} ({d['se']:.2f})"


def main() -> None:
    out = ["# Intraday (TAQ) vs daily day-0 durations",
           (f"\nGenerated {date.today()}. D (HC1 se) from regressing VW portfolio returns on "
           f"minus MPS x {SCALE}. Intraday: NBBO midpoints [-30, +10] min around the "
           "statement. Daily: CRSP close to close. Same meetings and portfolios.")]

    meet = taq.fomc_meetings().set_index("day0")
    ev_i = intraday.event_returns(taq.load_nbbo(), taq.load_link())
    days = pd.DatetimeIndex(sorted(ev_i["date"].unique()))
    s = meet.loc[meet.index.isin(days), "mps"].dropna()
    days = s.index

    monthly, ev_d = cc.load_panels(days, list(range(days.min().year - 1, days.max().year + 1)))
    funda = pd.read_parquet(raw_dir("compustat") / "funda.parquet")
    link = pd.read_parquet(raw_dir("compustat") / "ccm_link.parquet")
    monthly = cc.add_book_to_market(monthly, link_to_permno(compustat_characteristics(funda),
                                                            link))
    monthly["size_dec"] = cc.add_size_decile(monthly, french.load_me_breakpoints())
    monthly["bm_dec"] = cc.add_bm_decile(monthly)
    monthly.loc[cc.is_fin_util(monthly["siccd"]), ["size_dec", "bm_dec"]] = np.nan

    # Coverage
    univ = ev_d.merge(ev_i, on=["permno", "date"], how="left", suffixes=("_d", "_i"))
    out.append(f"\n## Sample\n\n{len(days)} scheduled meetings, {days.min():%Y-%m-%d} to "
               f"{days.max():%Y-%m-%d}. Stock-meetings with a daily CRSP return: {len(univ):,}; "
               f"with an intraday return too: {univ['ret_i'].notna().mean():.1%}. "
               f"Corr(intraday, daily) across stock-meetings: "
               f"{univ[['ret_i', 'ret_d']].corr().iloc[0, 1]:.2f}.")

    # Durations
    sorts = {"Market": None, "Size": "size_dec", "B/M": "bm_dec"}
    yr = days.year
    samples = {"all": s, "ex 2008-09 and 2020": s[~np.isin(yr, [2008, 2009, 2020])],
               "ex 5 largest |MPS|": s.drop(s.abs().nlargest(5).index)}
    for name, port in sorts.items():
        r_i = cc.event_portfolio_returns(ev_i, monthly, port)
        r_d = cc.event_portfolio_returns(ev_d, monthly, port)
        if port:
            for r in (r_i, r_d):
                r["10-1"] = r[r.columns.max()] - r[r.columns.min()]
        rows = []
        for col in r_i.columns:
            row = {"portfolio": col}
            for lab, r in (("intraday", r_i), ("daily", r_d)):
                d = event_duration(r[col].reindex(s.index), s, SCALE)
                row[f"{lab} D (se)"] = fmt(d)
                row[f"{lab} R2"] = round(d["r2"], 3)
            for k, v in samples.items():
                if k != "all":
                    row[f"intraday, {k}"] = fmt(event_duration(r_i[col].reindex(v.index), v,
                                                               SCALE))
            rows.append(row)
        out.append(f"\n## {name}\n")
        out.append(pd.DataFrame(rows).set_index("portfolio").to_markdown())

    # Chen on the overlapping meetings
    ch = chen.load("all")
    ch["eventdate"] = cc._sas_date(ch["eventdate"])
    ch["day0"] = align_to_trading_days(ch["eventdate"], french.trading_days())
    ch_o = ch[ch["day0"].isin(days)]
    s_o = s[s.index.isin(ch_o["day0"].unique())]
    out.append(f"\n## Against Chen on the {len(s_o)} overlapping meetings\n")
    rows = []
    for name, port in (("Size", "size_dec"), ("B/M", "bm_dec")):
        theirs = cc.chen_portfolio_durations(ch_o.drop(columns="day0"), monthly, port)
        for lab, ev in (("intraday", ev_i), ("daily", ev_d)):
            ours = cc.durations_with_spread(cc.event_portfolio_returns(ev, monthly, port),
                                            s_o, SCALE)
            res = cc.compare(ours, theirs)
            rows.append({"sort": name, "our returns": lab, **res})
    out.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".2f"))

    # Per-stock ratios: how much noise does Chen's trimming face with each window?
    out.append("\n## Per-stock ratio -r / (MPS x scale), Chen's [1, 300] rule, by size decile\n")
    rows = []
    h = monthly[["permno", "month", "me", "size_dec"]].dropna()
    h = h.rename(columns={"month": "form_month"})
    for lab, ev in (("intraday", ev_i), ("daily", ev_d)):
        e = ev[ev["date"].isin(days)].copy()
        e["form_month"] = e["date"] + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)
        e = e.merge(h, on=["permno", "form_month"])
        e["ratio"] = -e["ret"] / (e["date"].map(s) * SCALE)
        e["kept"] = e["ratio"].between(LO, HI)
        for k, g in e.groupby("size_dec"):
            kept = g[g["kept"]]
            vw = ((kept["ratio"] * kept["me"]).groupby(kept["date"]).sum()
                  / kept.groupby("date")["me"].sum()).mean()
            rows.append({"returns": lab, "size decile": int(k),
                         "share ratio < 0": (g["ratio"] < 0).mean(),
                         "share kept": g["kept"].mean(), "VW mean of kept": vw,
                         "sd of return": g.groupby("date")["ret"].std().mean()})
    t = pd.DataFrame(rows).pivot(index="size decile", columns="returns")
    t.columns = [f"{a}, {b}" for a, b in t.columns]
    out.append(t.to_markdown(floatfmt=".3g"))

    path = RESULTS / "tables" / "m2_intraday.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
