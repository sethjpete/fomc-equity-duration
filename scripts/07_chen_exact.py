"""Chen (2022) exact replication on his announcements with TAQ coverage (2004-2016).

Usage: python scripts/07_chen_exact.py
Needs: `python scripts/01_pull_data.py dsi fred_chen taq_chen` plus CRSP, Compustat, French,
Chen's files. Writes results/tables/m2_chen_exact.md and data/processed/chen_exact_D.parquet.
Method: src/fomcdur/replication/chen_exact.py.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from fomcdur.clean.characteristics import compustat_characteristics, link_to_permno
from fomcdur.config import PROCESSED, RESULTS, raw_dir
from fomcdur.data import chen, fred, french, taq, wrds_pull
from fomcdur.replication import chen_compare as cc
from fomcdur.replication import chen_exact as cx
from fomcdur.replication import intraday

ER_SCALES = {"daily": 1.0, "annualized": 252.0}
CRSP_COLS = ["permno", "date", "ret", "prc", "shrout", "exchcd", "siccd", "shrcd"]


class CrspCache:
    def __init__(self):
        self.years: dict[int, pd.DataFrame] = {}

    def get(self, years) -> pd.DataFrame:
        for y in years:
            if y not in self.years:
                cols = list(CRSP_COLS)
                try:
                    self.years[y] = wrds_pull.load_crsp([y], columns=cols)
                except Exception:  # noqa: BLE001 - shrcd may be absent in some files
                    self.years[y] = wrds_pull.load_crsp([y], columns=cols[:-1])
            num = ["ret", "prc", "shrout", "exchcd", "siccd", "shrcd"]
            df = self.years[y]
            for c in num:
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            if len(self.years) > 3:
                self.years.pop(min(self.years))
        return pd.concat([self.years[y] for y in years], ignore_index=True)


def main() -> None:
    out = ["# Chen (2022) exact replication, 2004-2016 announcements",
           (f"\nGenerated {date.today()}. D = 9.75 x -(dP/P) / [beta dER_M + (1-beta) dRf], "
           "dP/P from window-average NBBO midpoints, dRf from his Table A.1, dER_M from a daily "
           "VAR(1), beta from 100 beta portfolios; kept if 1 <= D <= 300.")]

    ev_all = cx.load_events()
    nbbo = taq.load_chen_nbbo()
    nbbo["n_quotes"] = nbbo["n_pre"] + nbbo["n_post"]
    nbbo = nbbo.rename(columns={"avg_pre": "mid_pre", "avg_post": "mid_post"})
    rets = intraday.event_returns(nbbo, taq.load_link())
    ev = ev_all[ev_all["date"].isin(rets["date"].unique())].reset_index(drop=True)

    # dER_M
    dsi = pd.read_parquet(raw_dir("crsp") / "dsi.parquet")
    dsi["date"] = pd.to_datetime(dsi["date"])
    dsi[["vwretd", "vwretx"]] = dsi[["vwretd", "vwretx"]].astype("float64")
    ylds = fred.load(("DGS10", "DTB3", "DBAA", "DAAA"))
    hml = french.load_factors()["HML"]
    z = cx.var_states(dsi, ylds, hml)
    der = cx.delta_er_m(z, pd.DatetimeIndex(ev_all["date"]))
    ev_all["der_m"] = ev_all["date"].map(der)
    ev = ev.merge(ev_all[["date", "der_m"]], on="date")
    tab = ev_all.assign(drf_bp=ev_all["drf"] * 1e4, der_m_bp=ev_all["der_m"] * 1e4,
                        der_m_ann_bp=ev_all["der_m"] * 252 * 1e4,
                        taq=ev_all["date"].isin(ev["date"]))
    out.append(f"\n## Announcements\n\nVAR sample from {z.index.min():%Y-%m-%d}. "
               f"{len(ev)} of {len(ev_all)} announcements have TAQ data.\n")
    out.append(tab[["date", "time", "drf_bp", "der_m_bp", "der_m_ann_bp", "taq"]]
               .to_markdown(index=False, floatfmt=".3g"))

    # Betas and universe, event by event
    bp = french.load_me_breakpoints()
    crsp = CrspCache()
    pieces = []
    for e in ev.itertuples():
        d = crsp.get([e.date.year - 1, e.date.year])
        d["date"] = pd.to_datetime(d["date"])
        days = np.sort(d.loc[d["date"] < e.date, "date"].unique())[-cx.BETA_WINDOW:]
        win = d[d["date"].isin(days)]
        R = win.pivot_table(index="date", columns="permno", values="ret")
        mkt = dsi.set_index("date")["vwretd"]
        last = win[win["date"] == days[-1]].copy()
        last["me"] = last["prc"].abs() * last["shrout"] / 1000.0
        me = last.set_index("permno")["me"]
        b = cx.portfolio_betas(R, mkt, me)
        # universe at t-1
        u = last[["permno", "me", "siccd", "exchcd"]].copy()
        if "shrcd" in last:
            u = u[last["shrcd"].isin([10, 11]).to_numpy()]
        u = u[~cc.is_fin_util(u["siccd"])]
        prior_me = e.date + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)
        p20 = bp["p20"].get(prior_me, np.nan)
        me_m = (d[(d["date"] <= prior_me)].sort_values("date").groupby("permno").tail(1))
        me_m = (me_m["prc"].abs() * me_m["shrout"] / 1000.0).set_axis(me_m["permno"])
        u["micro"] = u["permno"].map(me_m) < p20
        x = (rets[rets["date"] == e.date].merge(b[["permno", "beta"]], on="permno")
             .merge(u[["permno", "me", "micro"]], on="permno"))
        x["drf"], x["der_m"] = e.drf, e.der_m
        pieces.append(x)
        print(f"[chen_exact] {e.date:%Y-%m-%d}: {len(x):,} stocks", flush=True)
    base = pd.concat(pieces, ignore_index=True)

    # Chen's file
    ch = chen.load("all")
    ch["date"] = cc._sas_date(ch["eventdate"])
    ch = ch[ch["date"].isin(ev["date"])][["permno", "date", "duration"]]
    ch["permno"] = ch["permno"].astype("int64")
    ch2 = chen.load("table2")
    ch2["date"] = cc._sas_date(ch2["eventdate"])
    in_t2 = set(zip(ch2["permno"].astype("int64"), ch2["date"]))

    out.append("\n## Stock-level match with Chen's posted estimates (same announcements)\n")
    rows, kept_frames = [], {}
    for lab, sc in ER_SCALES.items():
        for micro in (False, True):
            dd = cx.durations(base, sc)
            if micro:
                dd = dd[~dd["micro"].fillna(True)]
            k = dd[dd["kept"]]
            m = k.merge(ch, on=["permno", "date"], how="outer", indicator=True)
            both = m[m["_merge"] == "both"]
            rows.append({
                "dER_M units": lab, "ex microcaps": micro,
                "our stock-events": len(dd), "ours kept": len(k), "Chen's": len(ch),
                "matched": len(both),
                "share of Chen's matched": len(both) / max(len(ch), 1),
                "corr(log D)": np.log(both["D"]).corr(np.log(both["duration"])),
                "Spearman": both["D"].rank().corr(both["duration"].rank()),
                "median D ours / Chen": float((both["D"] / both["duration"]).median()),
                "mean D ours": k["D"].mean(), "mean D Chen": ch["duration"].mean(),
            })
            kept_frames[(lab, micro)] = dd
    out.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))
    out.append(f"\nChen's table2 file (his main sample) has "
               f"{sum(1 for p, d in in_t2 if d in set(ev['date']))} of these stock-events.")

    # Reverse-engineer his denominators
    mt = base.merge(ch, on=["permno", "date"])
    imp = cx.implied_denominators(mt)
    imp["dRf (Table A.1)"] = imp.index.map(ev.set_index("date")["drf"])
    imp["our dER_M daily"] = imp.index.map(ev.set_index("date")["der_m"])
    imp["our dER_M x 252"] = imp["our dER_M daily"] * 252
    out.append("\n## His implied denominator vs ours\n\nFor each stock he reports, denom = "
               "9.75 x -(our ret) / his D. Regressed on our beta within each announcement: the "
               "intercept should be dRf and intercept + slope his dER_M (decimals). Low R2 means "
               "our returns or betas differ from his.\n")
    out.append(imp.to_markdown(floatfmt=".3g"))

    # Deciles with the exact D (better-matching units)
    best = max(rows, key=lambda r: (r["corr(log D)"] if pd.notna(r["corr(log D)"]) else -9))
    dd = kept_frames[(best["dER_M units"], best["ex microcaps"])]
    monthly, _ = cc.load_panels(pd.DatetimeIndex(ev["date"]),
                                list(range(ev["date"].min().year - 1,
                                           ev["date"].max().year + 1)))
    funda = pd.read_parquet(raw_dir("compustat") / "funda.parquet")
    link = pd.read_parquet(raw_dir("compustat") / "ccm_link.parquet")
    monthly = cc.add_book_to_market(monthly, link_to_permno(compustat_characteristics(funda),
                                                            link))
    monthly["size_dec"] = cc.add_size_decile(monthly, bp)
    monthly["bm_dec"] = cc.add_bm_decile(monthly)
    monthly.loc[cc.is_fin_util(monthly["siccd"]), ["size_dec", "bm_dec"]] = np.nan
    ours = dd[dd["kept"]].rename(columns={"date": "eventdate", "D": "duration"})
    chx = ch.rename(columns={"date": "eventdate"})
    out.append(f"\n## Deciles: our exact D vs Chen's (dER_M {best['dER_M units']}, ex microcaps "
               f"{best['ex microcaps']})\n")
    for sort in ("size_dec", "bm_dec"):
        a = cc.chen_portfolio_durations(ours, monthly, sort)
        b = cc.chen_portfolio_durations(chx, monthly, sort)
        t = pd.DataFrame({"ours VW": a["vw"], "Chen VW": b["vw"], "ours median": a["median"],
                          "Chen median": b["median"], "ours n": a["n_stocks"],
                          "Chen n": b["n_stocks"]})
        dec = [i for i in t.index if i != "10-1"]
        sp = t.loc[dec, "ours VW"].astype(float).rank().corr(
            t.loc[dec, "Chen VW"].astype(float).rank())
        out.append(f"\n### {sort} (Spearman of VW: {sp:.2f})\n")
        out.append(t.to_markdown(floatfmt=".3g"))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    dd.to_parquet(PROCESSED / "chen_exact_D.parquet", index=False)
    path = RESULTS / "tables" / "m2_chen_exact.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
