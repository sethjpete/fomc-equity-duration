"""M2: replicate Chen (2022) FOMC-day durations with daily CRSP and SF Fed surprises.

Usage: python scripts/03_replicate_chen.py [--chen-file all|table1|table2]
Needs: CRSP daily, Compustat + CCM link, SF Fed surprises, French factors and ME breakpoints,
Chen's files (`make data`).
Writes results/tables/m2_chen_replication.md, m2_durations.csv, m2_chen_aggregates.csv and
results/figures/m2_ours_vs_chen.png. Method notes: src/fomcdur/replication/chen_compare.py.
"""

from __future__ import annotations

import argparse
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fomcdur.clean.characteristics import compustat_characteristics, link_to_permno
from fomcdur.clean.events import align_to_trading_days, build_calendar
from fomcdur.config import RESULTS, raw_dir, settings
from fomcdur.data import chen, french, sffed
from fomcdur.replication import chen_compare as cc
from fomcdur.replication.duration import portfolio_durations

S = settings()
START, END = pd.Timestamp(S["sample"]["start"]), pd.Timestamp(S["sample"]["end"])
SCALE = S["duration"]["surprise_scale"]
CHEN_START, CHEN_END = pd.Timestamp("1995-01-01"), pd.Timestamp("2016-12-31")
TABLES, FIGS = RESULTS / "tables", RESULTS / "figures"


def fmt_d(row) -> str:
    return f"{row['D']:.1f} ({row['se']:.1f})"


def main(chen_file: str) -> None:
    lines = ["# M2: Chen (2022) replication, FOMC day-0 durations",
             (f"\nGenerated {date.today()} (Chen file: `{chen.FILES[chen_file]}`). "
             "Our D: day-0 VW portfolio return regressed on minus the SF Fed MPS (pp x "
             f"{SCALE}), HC1 standard errors in parentheses. Chen: his stock-level ratio "
             "estimates (TAQ [-30,+10] min, trimmed to [1, 300] years) placed in the same "
             "portfolios, VW within each announcement, then averaged over announcements. "
             "Levels differ by construction; the check is the ranking across deciles.")]

    # Surprises and day 0
    factors = french.load_factors()
    td = factors.index
    surp = sffed.load(scheduled_only=True)
    surp = surp[surp["date"] <= END]
    cal = build_calendar(surp, td).set_index("day0")
    day0 = cal.index

    # Chen's estimates
    ch = chen.load(chen_file)
    ch["eventdate"] = cc._sas_date(ch["eventdate"])
    chen_days = align_to_trading_days(ch["eventdate"].drop_duplicates(), td)

    # Panels
    years = list(range(START.year - 1, END.year + 1))
    all_days = day0.union(chen_days)
    monthly, event = cc.load_panels(all_days, years)
    funda = pd.read_parquet(raw_dir("compustat") / "funda.parquet")
    link = pd.read_parquet(raw_dir("compustat") / "ccm_link.parquet")
    chars = link_to_permno(compustat_characteristics(funda), link)
    monthly = cc.add_book_to_market(monthly, chars)
    monthly["size_dec"] = cc.add_size_decile(monthly, french.load_me_breakpoints())
    monthly["bm_dec"] = cc.add_bm_decile(monthly)
    fu = cc.is_fin_util(monthly["siccd"])
    monthly.loc[fu, ["size_dec", "bm_dec"]] = float("nan")

    variants = {
        "main": (cal["mps"], f"MPS, {len(cal)} scheduled meetings {START.year}-{END.year}"),
        "orth": (cal["mps_orth"].dropna(),
                 "MPS_ORTH (Bauer-Swanson orthogonalized; 2020 Mar-Dec dropped)"),
        "chen_window": (cal.loc[CHEN_START:CHEN_END, "mps"], "MPS, 1995-2016 (Chen's window)"),
        "chen_events": (cal.loc[cal.index.isin(chen_days), "mps"],
                        "MPS, only meetings in Chen's file"),
    }
    lines.append("\n## Samples\n")
    for k, (s, desc) in variants.items():
        lines.append(f"- `{k}`: {desc}; n = {len(s)}")
    lines.append(f"- Chen's file: {ch['eventdate'].nunique()} announcements "
                 f"({ch['eventdate'].min():%Y-%m-%d} to {ch['eventdate'].max():%Y-%m-%d}), "
                 f"{len(ch):,} stock-event estimates; duration range "
                 f"[{ch['duration'].min():.2f}, {ch['duration'].max():.2f}].")

    # Market
    lines.append("\n## Market\n")
    mkt = cc.event_portfolio_returns(event, monthly, None)
    ff_mkt = (factors["Mkt_RF"] + factors["RF"]).rename("FF")
    rows = {}
    for k, (s, _) in variants.items():
        d1 = portfolio_durations(mkt.reindex(s.index), s, SCALE).loc["MKT"]
        d2 = portfolio_durations(ff_mkt.reindex(s.index).to_frame(), s, SCALE).loc["FF"]
        rows[k] = {"ours, CRSP VW": fmt_d(d1), "ours, French Mkt": fmt_d(d2),
                   "R2 (CRSP VW)": f"{d1['r2']:.2f}"}
    ch_mkt = cc.chen_portfolio_durations(ch, monthly, None).loc["MKT"]
    lines.append(pd.DataFrame(rows).T.to_markdown())
    lines.append(f"\nChen, same stocks: VW {ch_mkt['vw']:.1f}, EW {ch_mkt['ew']:.1f}, "
                 f"median {ch_mkt['median']:.1f} years "
                 f"({ch_mkt['n_stocks']:.0f} stocks per announcement).")

    # Deciles
    long_rows, chen_rows, summary = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (sort, label) in zip(axes, [("size_dec", "Size (1 = small)"),
                                        ("bm_dec", "Book-to-market (1 = growth)")]):
        rets = cc.event_portfolio_returns(event, monthly, sort)
        ours = {k: cc.durations_with_spread(rets, s, SCALE) for k, (s, _) in variants.items()}
        theirs = cc.chen_portfolio_durations(ch, monthly, sort)
        tab = pd.DataFrame({f"ours {k}": o.apply(fmt_d, axis=1) for k, o in ours.items()})
        tab["Chen VW"] = theirs["vw"].map("{:.1f}".format)
        tab["Chen EW"] = theirs["ew"].map("{:.1f}".format)
        tab["Chen median"] = theirs["median"].map("{:.1f}".format)
        tab["stocks/event"] = theirs["n_stocks"].map(
            lambda x: "" if pd.isna(x) else f"{x:.0f}")
        lines.append(f"\n## {label} deciles (ex financials and utilities)\n")
        lines.append(tab.to_markdown())
        for k, o in ours.items():
            summary.append({"sort": sort, "variant": k, **cc.compare(o, theirs)})
            long_rows.append(o.assign(sort=sort, variant=k).rename_axis("port").reset_index())
        chen_rows.append(theirs.assign(sort=sort).rename_axis("port").reset_index())

        dec = [i for i in ours["main"].index if i != "10-1"]
        x, y = theirs.loc[dec, "vw"].astype(float), ours["main"].loc[dec, "D"].astype(float)
        ax.plot(x, y, "o", color="#2a6ebb", ms=7)
        for i in dec:
            ax.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(5, 4),
                        fontsize=9, color="#333333")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Chen (2022): VW mean duration, years")
        ax.set_ylabel("Ours: day-0 MPS duration")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    lines.append("\n## Agreement\n")
    lines.append("Spearman correlation of the 10 decile durations, ours vs Chen; "
                 "`10-1` spreads (ours with t-stat).\n")
    lines.append(pd.DataFrame(summary).to_markdown(index=False, floatfmt=".2f"))

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    (TABLES / "m2_chen_replication.md").write_text("\n".join(lines) + "\n")
    pd.concat(long_rows).to_csv(TABLES / "m2_durations.csv", index=False)
    pd.concat(chen_rows).to_csv(TABLES / "m2_chen_aggregates.csv", index=False)
    fig.tight_layout()
    fig.savefig(FIGS / "m2_ours_vs_chen.png", dpi=200)
    print("\n".join(lines))
    print(f"\nWrote {TABLES / 'm2_chen_replication.md'} and {FIGS / 'm2_ours_vs_chen.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chen-file", default="all", choices=list(chen.FILES))
    main(p.parse_args().chen_file)
