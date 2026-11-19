"""M2 follow-up: why Chen's size-decile durations are U-shaped while ours are flat.

Usage: python scripts/04_size_gap_diagnostics.py [--chen-file all|table1|table2]
Writes results/tables/m2_size_gap.md.

Hypothesis: Chen's per-stock ratio D = -r / dER is noisiest for small stocks, and keeping
only estimates inside [1, 300] years selects on the noise, so noisy deciles average higher.
Tests:
1. Bounds: is Chen's file clipped (mass at 1 and 300) or truncated (no mass there)?
2. Coverage: share of our universe (stock x Chen meeting) that has a Chen estimate, by size
   decile. Truncation drops more of the noisy deciles.
3. Simulation on our own data: the per-stock daily ratio D_it = -r_it / (MPS_t x 0.01) at
   Chen's meetings, then (a) no trimming, aggregated as the pooled slope, and (b) Chen's
   rule: keep 1 <= D <= 300, VW mean within each meeting, averaged across meetings.
   If (b) is U-shaped while (a) is flat, the gap comes from the estimator, not the data.
4. Noise: cross-sectional dispersion of day-0 returns by size decile at those meetings.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from fomcdur.clean.events import align_to_trading_days, build_calendar
from fomcdur.config import RESULTS, settings
from fomcdur.data import chen, french, sffed
from fomcdur.replication import chen_compare as cc

S = settings()
END = pd.Timestamp(S["sample"]["end"])
SCALE = S["duration"]["surprise_scale"]
LO, HI = 1.0, 300.0


def main(chen_file: str) -> None:
    out = ["# M2 follow-up: the size-decile gap with Chen (2022)",
           f"\nGenerated {date.today()}; Chen file `{chen.FILES[chen_file]}`."]

    td = french.load_factors().index
    surp = sffed.load(scheduled_only=False)
    surp = surp[surp["date"] <= END]
    cal = build_calendar(surp, td).drop_duplicates("day0").set_index("day0")

    ch = chen.load(chen_file)
    ch["eventdate"] = cc._sas_date(ch["eventdate"])
    ch["day0"] = align_to_trading_days(ch["eventdate"], td)
    chen_days = pd.DatetimeIndex(ch["day0"].unique()).sort_values()

    # 1. Bounds
    d = ch["duration"].dropna()
    out.append("\n## 1. Bounds in Chen's file\n")
    out.append(pd.Series({
        "estimates": len(d), "min": d.min(), "max": d.max(),
        "share == 1 (to 1e-6)": (d - LO).abs().lt(1e-6).mean(),
        "share in [1, 1.05]": d.between(1, 1.05).mean(),
        "share == 300 (to 1e-6)": (d - HI).abs().lt(1e-6).mean(),
        "share in [295, 300]": d.between(295, 300).mean(),
        "p1": d.quantile(.01), "p5": d.quantile(.05), "p50": d.median(),
        "p95": d.quantile(.95), "p99": d.quantile(.99),
    }, name="value").to_markdown(floatfmt=".4g"))
    hist = np.histogram(d, bins=[1, 2, 5, 10, 20, 50, 100, 200, 300.001])
    out.append("\nHistogram:\n")
    out.append(pd.Series(hist[0], index=[f"[{a:g}, {b:g})" for a, b in
                                         zip(hist[1][:-1], hist[1][1:])],
                         name="count").to_markdown())

    # Panels
    years = sorted({y for y in range(chen_days.min().year - 1, chen_days.max().year + 1)})
    monthly, event = cc.load_panels(chen_days, years)
    monthly["size_dec"] = cc.add_size_decile(monthly, french.load_me_breakpoints())
    monthly.loc[cc.is_fin_util(monthly["siccd"]), "size_dec"] = np.nan
    h = monthly[["permno", "month", "me", "size_dec"]].dropna()
    h = h.rename(columns={"month": "form_month"})
    ev = event.copy()
    ev["form_month"] = ev["date"] + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)
    ev = ev.merge(h, on=["permno", "form_month"], how="inner").dropna(subset=["ret"])
    ev["mps"] = ev["date"].map(cal["mps"])
    ev = ev.dropna(subset=["mps"])

    # 2. Coverage
    c = ch[["permno", "day0", "duration"]].dropna().copy()
    c["permno"] = c["permno"].astype("int64")
    cov = ev.merge(c.rename(columns={"day0": "date"}), on=["permno", "date"], how="left")
    g = cov.groupby("size_dec")
    tab2 = pd.DataFrame({
        "our stock-meetings": g.size(),
        "with Chen estimate": g["duration"].count(),
    })
    tab2["coverage"] = tab2["with Chen estimate"] / tab2["our stock-meetings"]
    tab2["ME-weighted coverage"] = (cov.assign(w=cov["me"] * cov["duration"].notna())
                                    .groupby("size_dec")["w"].sum() / g["me"].sum())
    out.append("\n## 2. Coverage: share of our stock-meetings with a Chen estimate\n")
    out.append("Our universe: common stocks, NYSE/AMEX/Nasdaq, ex financials and utilities, "
               "with a day-0 return, at the meetings in Chen's file.\n")
    tab2.index = tab2.index.astype(int)
    out.append(tab2.to_markdown(floatfmt=".3g"))

    # 3. Simulation with our daily ratio
    ev["ratio"] = -ev["ret"] / (ev["mps"] * SCALE)
    ev["kept"] = ev["ratio"].between(LO, HI)
    ev["wr"] = ev["ret"] * ev["me"]
    rows = {}
    for k, gk in ev.groupby("size_dec"):
        # (a) pooled slope of VW day-0 return on -MPS, no trimming
        per = gk.groupby("date").apply(lambda x: pd.Series({
            "r": x["wr"].sum() / x["me"].sum(), "s": x["mps"].iloc[0]}))
        x = -per["s"] * SCALE
        slope = float(((x - x.mean()) * (per["r"] - per["r"].mean())).sum()
                      / ((x - x.mean()) ** 2).sum())
        kept = gk[gk["kept"]]
        vw = (kept.assign(w=kept["ratio"] * kept["me"]).groupby("date")["w"].sum()
              / kept.groupby("date")["me"].sum())
        rows[int(k)] = {
            "(a) slope, no trimming": slope,
            "(b) Chen rule: VW mean of kept ratios": vw.mean(),
            "(b) median of kept ratios": kept.groupby("date")["ratio"].median().mean(),
            "share kept": gk["kept"].mean(),
            "share ratio < 0": (gk["ratio"] < 0).mean(),
            "sd of day-0 return": gk.groupby("date")["ret"].std().mean(),
        }
    tab3 = pd.DataFrame(rows).T
    chen_vw = cc.chen_portfolio_durations(ch.drop(columns="day0"), monthly, "size_dec")["vw"]
    tab3["Chen VW (for reference)"] = chen_vw.reindex(tab3.index).astype(float)
    out.append("\n## 3. Chen's trimming applied to our daily ratio, same meetings\n")
    out.append(f"{len(ev['date'].unique())} meetings. Ratio = -r / (MPS x {SCALE}). "
               "(a) is the untrimmed estimate; (b) applies Chen's keep-if-in-[1, 300] rule "
               "and his VW aggregation. Section 4 is the last column: cross-sectional sd "
               "of day-0 returns, averaged over meetings.\n")
    out.append(tab3.to_markdown(floatfmt=".3g"))
    sp = tab3.rank().corr().loc["(b) Chen rule: VW mean of kept ratios",
                                 "Chen VW (for reference)"]
    sp_a = tab3.rank().corr().loc["(a) slope, no trimming", "Chen VW (for reference)"]
    out.append(f"\nSpearman with Chen VW across deciles: (a) {sp_a:.2f}, (b) {sp:.2f}.")

    path = RESULTS / "tables" / "m2_size_gap.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chen-file", default="all", choices=list(chen.FILES))
    main(p.parse_args().chen_file)
