"""M3 step 1: day-0 durations of FF49 industry and characteristic-decile portfolios.

Usage: python scripts/08_portfolio_durations.py   (or `make portfolios`)
Needs the TAQ pull (`make data-taq`), CRSP daily, Compustat + CCM, French files, SF Fed.
Writes results/tables/m3_portfolio_durations.md, m3_portfolio_durations.csv and
results/figures/m3_industry_durations.png.

Sample: scheduled meetings with TAQ data (2003-09 to 2023-12) and an MPS surprise. Day-0
return: NBBO midpoints [-30, +10] min around the statement. Method in
`fomcdur.extension.portfolio_durations`.
"""

from __future__ import annotations

from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fomcdur.config import RESULTS, settings
from fomcdur.data import french
from fomcdur.extension import portfolio_durations as pdur

S = settings()
SCALE = S["duration"]["surprise_scale"]
MIN_STOCKS = 5  # industry-meetings with fewer stocks are dropped
SORTS = {"Size (ME)": "size_dec", "Book-to-market": "bm_dec",
         "Operating profitability": "op_dec", "Leverage": "lev_dec"}


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(**kw)


def industry_figure(tab: pd.DataFrame, names: pd.Series, d_mkt: float, path) -> None:
    t = tab[tab["sample"] == "all"].set_index("portfolio").sort_values("D")
    fig, ax = plt.subplots(figsize=(7, 9.5))
    y = np.arange(len(t))
    ax.errorbar(t["D"], y, xerr=1.96 * t["se"], fmt="o", ms=3.5, color="#2a5d9f",
                ecolor="#9db6d6", elinewidth=1.2, capsize=0)
    ax.axvline(d_mkt, color="#c0392b", lw=1, ls="--", label=f"Market D = {d_mkt:.1f}")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(y, [names.get(int(p), str(p)) for p in t.index], fontsize=7.5)
    ax.set_xlabel("Day-0 duration (years), 95% CI")
    ax.set_title("FOMC duration of the 49 French industries\n"
                 "intraday [-30, +10] min, 2003-2023", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    s, ev, monthly = pdur.build_inputs()
    names = french.load_siccodes49().drop_duplicates("ind49").set_index("ind49")["ind_name"]
    out = ["# M3: FOMC day-0 durations of industry and characteristic portfolios",
           (f"\nGenerated {date.today()}. D in years (HC1 se): minus the slope of VW portfolio "  # noqa: DTZ011
           f"returns on MPS x {SCALE}. Day-0 return: TAQ NBBO midpoints [-30, +10] min around "
           "the statement. Portfolios formed at the prior month end, weights = ME then. "
           "Robustness columns: without the 5 largest |MPS|, without 2008-09 and 2020, and "
           "the two subperiods.")]
    samples = pdur.robustness_samples(s)
    out.append(f"\n## Sample\n\n{len(s)} scheduled meetings, {s.index.min():%Y-%m-%d} to "
               f"{s.index.max():%Y-%m-%d}. Meetings per column: "
               + ", ".join(f"{k} {len(v)}" for k, v in samples.items()) + ". "
               "Dropped as largest |MPS|: "
               + ", ".join(f"{d:%Y-%m-%d} ({s[d]:+.2f}pp)"
                           for d in s.abs().nlargest(5).index) + ".")

    all_tabs = []
    r_m, n_m = pdur.event_portfolio_returns(ev, monthly, None)
    tm = pdur.duration_table(r_m, s, SCALE)
    all_tabs.append(tm.assign(sort="Market"))
    d_mkt = float(tm.loc[tm["sample"] == "all", "D"].iloc[0])
    out.append("\n## Market (all matched common stocks, VW)\n")
    out.append(md(pdur.wide(tm)))
    out.append(f"\nStocks per meeting: median {int(n_m['MKT'].median()):,}.")

    # Industries
    r_i, n_i = pdur.event_portfolio_returns(ev, monthly, "ind49", MIN_STOCKS)
    ti = pdur.duration_table(r_i, s, SCALE)
    all_tabs.append(ti.assign(sort="FF49 industry"))
    w = pdur.wide(ti)
    main_ = ti[ti["sample"] == "all"].set_index("portfolio")
    w.insert(0, "industry", [names.get(int(p), "") for p in w.index])
    w["R2"] = main_["r2"].round(2)
    w["median stocks"] = n_i.median().reindex(w.index).astype(int)
    w = w.loc[main_["D"].sort_values(ascending=False).index]
    out.append("\n## FF49 industries (sorted by full-sample D)\n")
    out.append(md(w))

    out.append("\n### Is the industry dispersion signal or noise?\n")
    rows = []
    for name, ss in samples.items():
        t = ti[ti["sample"] == name]
        r_sub = r_i.reindex(ss.index)
        full = r_sub.columns[r_sub.notna().all()]
        rel = pdur.split_half_reliability(r_sub[full], ss, SCALE)
        test = pdur.equal_duration_test(r_sub[full], r_m["MKT"], ss, n_perm=2000)
        rows.append({"sample": name, "sd of D": t["D"].std(),
                     "mean se": t["se"].mean(),
                     "signal share": pdur.signal_share(t["D"], t["se"]),
                     "odd/even corr": rel["corr_odd_even"],
                     "Spearman-Brown": rel["spearman_brown"],
                     "equal-D stat": test["stat"], "null mean": test["null_mean"],
                     "perm p": test["p_perm"], "industries": test["df"]})
    out.append(md(pd.DataFrame(rows).set_index("sample"), floatfmt=".2f"))
    out.append("\nSignal share = 1 - mean(se^2)/var(D). Odd/even: correlation across "
               "industries of D estimated on alternate meetings; Spearman-Brown scales it to "
               "the full sample. Equal-D: sum of squared t-stats of industry-minus-market "
               "slopes, p-value from 2,000 permutations of the surprise across meetings "
               "(industries with a return at every meeting in that sample).")

    # Characteristic deciles
    for title, col in SORTS.items():
        r, n = pdur.event_portfolio_returns(ev, monthly, col)
        r = pdur.add_spread(r)
        t = pdur.duration_table(r, s, SCALE)
        all_tabs.append(t.assign(sort=title))
        w = pdur.wide(t)
        w["R2"] = t[t["sample"] == "all"].set_index("portfolio")["r2"].round(2)
        w["median stocks"] = [f"{v:.0f}" if pd.notna(v) else "" for v in
                              n.median().reindex(w.index)]
        w.index = [str(i) for i in w.index]
        out.append(f"\n## {title} deciles (NYSE breakpoints, ex financials and utilities)\n")
        out.append(md(w))

    tab = pd.concat(all_tabs, ignore_index=True)
    tab["portfolio"] = tab["portfolio"].astype(str)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / "tables" / "m3_portfolio_durations.csv", index=False,
               float_format="%.4f")
    industry_figure(ti, names, d_mkt, RESULTS / "figures" / "m3_industry_durations.png")
    path = RESULTS / "tables" / "m3_portfolio_durations.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
