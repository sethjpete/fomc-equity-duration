"""M1 data report: coverage, counts by year, missingness, event calendar.

Usage: python scripts/02_data_report.py
Writes results/tables/data_report.md and results/tables/fomc_calendar.csv.
Each section runs only if its raw data is present, so this works before the WRDS pull.
"""

from __future__ import annotations

import traceback
from datetime import date

import pandas as pd
import pyarrow.parquet as pq

from fomcdur.clean.events import build_calendar
from fomcdur.config import RESULTS, raw_dir, settings
from fomcdur.data import chen, fred, french, sffed, wrds_pull

OUT = RESULTS / "tables"
lines: list[str] = []


def h(title: str) -> None:
    lines.extend(["", f"## {title}", ""])


def table(df: pd.DataFrame | pd.Series, floatfmt: str = ".4g") -> None:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    lines.append(df.to_markdown(floatfmt=floatfmt))


def section(fn):
    try:
        fn()
    except FileNotFoundError as e:
        lines.append(f"_Skipped: {e}_")
    except Exception:  # noqa: BLE001 - keep going; the report should show every problem at once
        lines.append("```\n" + traceback.format_exc() + "```")


TODAY = date.today()  # noqa: DTZ011
S = settings()
START, END = pd.Timestamp(S["sample"]["start"]), pd.Timestamp(S["sample"]["end"])
H = S["sample"]["max_horizon"]
state: dict = {}


def factors():
    h("Ken French daily factors")
    f = french.load_factors()
    state["td"] = f.index
    lines.append(f"{f.index.min().date()} to {f.index.max().date()}, {len(f):,} trading days.")
    win = f.loc[START:END]
    lines.append(f"In sample window: {len(win):,} days; missing values: {int(win.isna().sum().sum())}.")
    table(win.describe().T[["mean", "std", "min", "max"]])
    bp = french.load_me_breakpoints()
    lines.append(f"\nNYSE ME breakpoints: {bp.index.min():%Y-%m} to {bp.index.max():%Y-%m}, "
                 f"{len(bp)} months. Latest p20 = ${bp['p20'].iloc[-1]:,.0f}m.")
    sic = french.load_siccodes49()
    lines.append(f"FF49 SIC table: {sic['ind49'].nunique()} industries, {len(sic)} SIC ranges.")


def surprises():
    h("SF Fed monetary policy surprises")
    allrows = sffed.load(scheduled_only=False, start="1900-01-01", drop_missing=False)
    end = allrows["date"].max()
    lines.append(f"File covers {allrows['date'].min().date()} to {end.date()} "
                 f"({len(allrows)} announcements).")
    s = sffed.load(scheduled_only=False, drop_missing=False)
    s = s[s["date"] <= END]
    td = state.get("td")
    if td is None:
        td = pd.bdate_range(START, END + pd.Timedelta(days=60))
        lines.append("_Trading days approximated by weekdays (French factors not loaded)._")
    cal = build_calendar(s, td)
    state["cal"] = cal
    by = cal.groupby([cal["date"].dt.year.rename("year"), "scheduled"]).size().unstack(fill_value=0)
    by.columns = ["unscheduled" if not c else "scheduled" for c in by.columns]
    sched = cal[cal["scheduled"] & cal["mps"].notna()]
    lines.append(f"\nScheduled announcements with MPS, {START.year}-{END.year}: **{len(sched)}**. "
                 f"Missing MPS: {', '.join(str(d.date()) for d in cal.loc[cal['mps'].isna(), 'date'])}. "
                 f"Scheduled with MPS_ORTH missing: {int(sched['mps_orth'].isna().sum())} "
                 "(Mar-Dec 2020 by design).")
    if end < END:
        gap_years = range(end.year + 1, END.year + 1)
        lines.append(f"\n**Coverage gap:** the file stops at {end.date()}, so scheduled meetings in "
                     f"{', '.join(map(str, gap_years))} (8 a year) have no surprise yet.")
    elif end < pd.Timestamp(S["sample"]["data_end"]):
        lines.append(f"\nMain sample ends {END.date()} with the SF Fed file; meetings after "
                     f"{end.date()} wait for an SF Fed update.")
    shifted = cal[cal["shifted"]]
    if len(shifted):
        lines.append("\nAnnouncements whose day 0 is a later trading day: "
                     + ", ".join(f"{r.date.date()} {r.time} -> {r.day0.date()}"
                                 for r in shifted.itertuples()))
    lines.append("\nCounts by year:\n")
    table(by)
    lines.append("\nScheduled-meeting surprises (pp):\n")
    table(sched[["mps", "mps_orth"]].describe().T)
    lines.append(f"\ncorr(MPS, MPS_ORTH) = {sched['mps'].corr(sched['mps_orth']):.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    cal[["date", "time", "scheduled", "day0", "shifted"]].to_csv(OUT / "fomc_calendar.csv",
                                                                 index=False)


def yields():
    h("FRED Treasury yields")
    y = fred.load()
    lines.append(f"{y.index.min().date()} to {y.index.max().date()}.")
    win = y.loc[START:END]
    table(win.describe().T[["count", "mean", "min", "max"]])
    cal = state.get("cal")
    if cal is not None:
        sched = cal[cal["scheduled"]]
        miss = {c: [str(d.date()) for d in sched["day0"] if pd.isna(y[c].get(d))] for c in y}
        for c, m in miss.items():
            lines.append(f"- {c} missing on {len(m)} scheduled day-0 dates "
                         f"{'(' + ', '.join(m[:10]) + ')' if m else ''}; forward-filled downstream.")


def chen_files():
    h("Chen (2022) posted estimates")
    for k in chen.FILES:
        df = chen.load(k)
        lines.append(f"\n**{chen.FILES[k]}**: {len(df):,} rows; columns "
                     f"{', '.join(f'`{c}` ({t})' for c, t in df.dtypes.astype(str).items())}")
        num = df.select_dtypes("number")
        if len(num.columns):
            table(num.describe().T[["count", "mean", "std", "min", "max"]])


def crsp():
    h("CRSP daily")
    files = sorted(raw_dir("crsp").glob("dsf_*.parquet"))
    if not files:
        raise FileNotFoundError("no CRSP files yet (run `make data-wrds`)")
    rows = []
    for f in files:
        cols = ["permno", "date", "ret", "prc", "shrout"]
        d = pd.read_parquet(f, columns=cols + (["source"] if "source" in pq.read_schema(f).names
                                               else []))
        rows.append({"year": int(f.stem.split("_")[1]),
                     "source": d["source"].iloc[0] if "source" in d else "legacy",
                     "rows": len(d), "permnos": d["permno"].nunique(), "days": d["date"].nunique(),
                     "ret_missing": d["ret"].isna().mean(), "prc_missing": d["prc"].isna().mean()})
    table(pd.DataFrame(rows).set_index("year"), floatfmt=".3g")
    dl = raw_dir("crsp") / "dsedelist.parquet"
    if dl.exists():
        dd = pd.read_parquet(dl)
        lines.append(f"\nDelistings: {len(dd):,} events, dlret missing {dd['dlret'].isna().mean():.1%}.")
    ov = sorted((raw_dir("crsp") / "overlap").glob("dsf_ciz_*.parquet"))
    for f in ov:
        y = int(f.stem.split("_")[-1])
        leg = wrds_pull.load_crsp([y])
        stats = wrds_pull.compare_overlap(leg, pd.read_parquet(f))
        lines.append(f"\nLegacy vs CIZ overlap, {y}:\n")
        table(pd.Series(stats, name="value"), floatfmt=".6g")


def compustat():
    h("Compustat annual and CCM link")
    p = raw_dir("compustat") / "funda.parquet"
    if not p.exists():
        raise FileNotFoundError("no Compustat file yet (run `make data-wrds`)")
    f = pd.read_parquet(p)
    from fomcdur.clean.characteristics import compustat_characteristics
    c = compustat_characteristics(f)
    by = c.groupby(c["datadate"].dt.year).agg(firms=("gvkey", "nunique"),
                                              be_missing=("be", lambda x: x.isna().mean()),
                                              op_missing=("op_prof", lambda x: x.isna().mean()),
                                              lev_missing=("leverage", lambda x: x.isna().mean()))
    table(by, floatfmt=".3g")
    link = pd.read_parquet(raw_dir("compustat") / "ccm_link.parquet")
    lines.append(f"\nCCM links: {len(link):,} rows, {link['gvkey'].nunique():,} gvkeys, "
                 f"{link['permno'].nunique():,} permnos.")


if __name__ == "__main__":
    lines.append("# M1 data report")
    lines.append(f"\nGenerated {TODAY} for the sample {START.date()} to {END.date()}, "
                 f"horizons 0-{H}.")
    for fn in [factors, surprises, yields, chen_files, crsp, compustat]:
        section(fn)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT / 'data_report.md'}")
