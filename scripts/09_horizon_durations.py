"""M4: horizon durations D(h), anchored at the pre-announcement TAQ midpoint.

Usage: python scripts/09_horizon_durations.py   (or `make horizons`)
Needs the M3 inputs plus CRSP daily through 2024, FRED DGS2/DGS10 and French factors.
Writes results/tables/m4_horizon_durations.{md,csv} and results/figures/m4_dh_curves.png.

R_{p,t}(h) = a_h - D(h) MPS_t x 0.01 + controls. Horizons: 0 = 40-minute window, 0c = to the
day-0 close, 1..20 = to the close of trading day h. Construction in
`fomcdur.extension.horizon_returns`.
"""

from __future__ import annotations

from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fomcdur.config import RESULTS, settings
from fomcdur.data import fred, french, taq, wrds_pull
from fomcdur.extension import horizon_returns as hr
from fomcdur.extension import portfolio_durations as pdur
from fomcdur.extension.horizon import bootstrap_same_day_share, horizon_durations

S = settings()
SCALE = S["duration"]["surprise_scale"]
H = S["sample"]["max_horizon"]
SHOW_H = [0, hr.CLOSE0, 1, 2, 5, 10, 15, 20]
FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def hlabel(h: int) -> str:
    return "0c" if h == hr.CLOSE0 else str(h)


def order(h: pd.Index) -> list:
    """Display order 0, 0c, 1, 2, ..."""
    return sorted(h, key=lambda x: 0.5 if x == hr.CLOSE0 else x)


def curve(cum: pd.Series, s: pd.Series, dy=None, ctrl=None) -> pd.DataFrame:
    c = horizon_durations(cum.dropna(), s, dy, ctrl, SCALE)
    return c.loc[order(c.index)]


def cell(c: pd.DataFrame, h: int) -> str:
    return f"{c.loc[h, 'D']:.1f} ({c.loc[h, 'se']:.1f})"


def load_daily(windows: pd.DataFrame) -> pd.DataFrame:
    dates = set(windows["date"])
    years = sorted({d.year for d in dates})
    parts = []
    for y in years:
        d = wrds_pull.load_crsp([y], columns=["permno", "date", "ret", "prc"])
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"].isin(dates)]
        d["permno"] = d["permno"].astype("int64")
        d[["ret", "prc"]] = d[["ret", "prc"]].astype(float)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    s, _, monthly = pdur.build_inputs()
    td = french.trading_days()
    win = hr.event_windows(s.index, td, H)
    anchor = hr.anchor_prices(taq.load_nbbo(), taq.load_link())
    anchor = anchor[anchor["date"].isin(s.index)]
    daily = load_daily(win)
    stock = hr.stock_cumulative(anchor, daily, win, H)

    yields = fred.load(("DGS2", "DGS10"))
    # Controls: yield changes from the day-0 close to the day-h close (0 for h <= 0).
    dy = hr.level_changes_from(yields, win, base_offset=0)
    dy_close = dy.xs(0, level="h").assign(h=hr.CLOSE0).set_index("h", append=True) * 0
    dy = pd.concat([dy, dy_close]).sort_index()
    fac = hr.cumulative_factors(french.load_factors()[FACTORS], win)
    fac = pd.concat([fac, fac.xs(0, level="h").assign(h=hr.CLOSE0)
                     .set_index("h", append=True)]).sort_index()

    out = ["# M4: horizon durations D(h), anchored at the pre-announcement price",
           (f"\nGenerated {date.today()}. D(h) in years (HC1 se): minus the slope of the "  # noqa: DTZ011
            f"portfolio's cumulative return on MPS x {SCALE}. Every horizon starts at the "
            "TAQ NBBO midpoint 30 minutes before the statement. h = 0: 40-minute window "
            "[-30, +10] min (= M3). 0c: to the day-0 close. h >= 1: to the close of trading "
            "day h. VW buy-and-hold, weights at the prior month end. Yield controls: change "
            "in the 2y and 10y Treasury yields from the day-0 close to the day-h close. "
            "Factor controls: FF5 + momentum compounded over days 1..h.")]
    out.append(f"\n{len(s)} meetings, {s.index.min():%Y-%m-%d} to {s.index.max():%Y-%m-%d}. "
               f"Stock-meetings with an anchor and a day-0 close: "
               f"{stock.loc[stock['h'] == 0, 'cum'].notna().sum():,}.")

    tabs = []
    # ---------------------------------------------------------------- yields
    lv = hr.level_changes_from(yields, win, base_offset=-1)
    rows = []
    for col in lv.columns:
        c = curve(lv[col], s)
        rows.append({"series": col, **{f"h={hlabel(h)}": f"{-c.loc[h, 'D'] * SCALE:.2f} "
                                       f"({c.loc[h, 'se'] * SCALE:.2f})"
                                       for h in c.index if h in SHOW_H}})
    out.append("\n## How much do Treasury yields move per 1pp of MPS?\n")
    out.append("Slope of the yield change (pp) from the close before day 0 to the close of "
               "day h on MPS (pp). Daily data, so h = 0 is close to close.\n")
    out.append(pd.DataFrame(rows).set_index("series").to_markdown())

    # ---------------------------------------------------------------- market
    cm = hr.portfolio_cumulative(stock, monthly, None)["MKT"]
    specs = {"no controls": (None, None), "yield controls": (dy, None)}
    curves = {k: curve(cm, s, *v) for k, v in specs.items()}
    for k, c in curves.items():
        tabs.append(c.reset_index().assign(portfolio="MKT", spec=k))
    t = pd.DataFrame({k: [cell(c, h) for h in SHOW_H] for k, c in curves.items()},
                     index=[hlabel(h) for h in SHOW_H])
    t.index.name = "h"
    out.append("\n## Market (VW, all matched common stocks)\n")
    out.append(t.to_markdown())

    rows = []
    for name, ss in pdur.robustness_samples(s).items():
        c = curve(cm, ss)
        rows.append({"sample": name, "n": int(c.loc[0, "n"]),
                     **{f"h={hlabel(h)}": cell(c, h) for h in [0, hr.CLOSE0, 5, 20]},
                     "D(0)/D(20)": c.loc[0, "D"] / c.loc[20, "D"]})
    out.append("\n### Market, robustness set (no controls)\n")
    out.append(pd.DataFrame(rows).set_index("sample").to_markdown(floatfmt=".2f"))

    out.append("\n### Same-day share, market (999 meeting-bootstrap draws)\n")
    rows = []
    for k, (d, _) in specs.items():
        c = curves[k]
        cm0 = cm.drop(index=hr.CLOSE0, level="h")
        draws = bootstrap_same_day_share(cm0.dropna(), s, d, None, h_end=20, n_boot=999)
        rows.append({"spec": k, "D(0)/D(20)": c.loc[0, "D"] / c.loc[20, "D"],
                     "95% CI": f"[{np.nanpercentile(draws, 2.5):.2f}, "
                               f"{np.nanpercentile(draws, 97.5):.2f}]",
                     "D(0c)/D(20)": c.loc[hr.CLOSE0, "D"] / c.loc[20, "D"]})
    out.append(pd.DataFrame(rows).set_index("spec").to_markdown(floatfmt=".2f"))
    out.append("\nThe bootstrap ratio is unstable when D(20) is near zero in a draw; read the "
               "interval together with the D(20) standard error above.")

    # ---------------------------------------------------------------- spreads
    spread_curves = {}
    for title, col in {"Size 10-1": "size_dec", "B/M 10-1": "bm_dec",
                       "Profitability 10-1": "op_dec", "Leverage 10-1": "lev_dec"}.items():
        cp = hr.portfolio_cumulative(stock, monthly, col)
        sp = cp[10] - cp[1]
        cs = {"no controls": curve(sp, s), "yields": curve(sp, s, dy),
              "yields + FF5 + Mom": curve(sp, s, dy, fac)}
        spread_curves[title] = cs["no controls"]
        for k, c in cs.items():
            tabs.append(c.reset_index().assign(portfolio=title, spec=k))
        t = pd.DataFrame({k: [cell(c, h) for h in SHOW_H] for k, c in cs.items()},
                         index=[hlabel(h) for h in SHOW_H])
        t.index.name = "h"
        rob = [f"{k}: " + ", ".join(f"h={hlabel(h)} {cell(curve(sp, ss), h)}"
                                    for h in [0, 5, 20])
               for k, ss in pdur.robustness_samples(s).items() if k != "all"]
        out.append(f"\n## {title} (deciles ex financials and utilities)\n")
        out.append(t.to_markdown())
        out.append("\nRobustness (no controls): " + "; ".join(rob) + ".")
        if title == "B/M 10-1":
            out.append("\nCaution: the factor controls include HML compounded over days 1..h, "
                       "which can itself respond to the surprise, so that column may absorb "
                       "part of the drift it is meant to isolate.")

    # ---------------------------------------------------------------- industries
    ci = hr.portfolio_cumulative(stock, monthly, "ind49")
    names = french.load_siccodes49().drop_duplicates("ind49").set_index("ind49")["ind_name"]
    rows = []
    for p in ci.columns:
        c = curve(ci[p], s)
        rows.append({"industry": names.get(int(p), str(p)), "ind": int(p),
                     **{f"D({hlabel(h)})": c.loc[h, "D"] for h in [0, hr.CLOSE0, 5, 20]},
                     **{f"se({hlabel(h)})": c.loc[h, "se"] for h in [0, 20]}})
        tabs.append(c.reset_index().assign(portfolio=f"ind{int(p)}", spec="no controls"))
    ind = pd.DataFrame(rows).set_index("industry").sort_values("D(0)", ascending=False)
    rel20 = pdur.signal_share(ind["D(20)"], ind["se(20)"])
    out.append("\n## FF49 industries\n")
    out.append(f"Across industries: corr(D(0), D(0c)) = {ind['D(0)'].corr(ind['D(0c)']):.2f}, "
               f"corr(D(0), D(5)) = {ind['D(0)'].corr(ind['D(5)']):.2f}, "
               f"corr(D(0), D(20)) = {ind['D(0)'].corr(ind['D(20)']):.2f}. Signal share at "
               f"h = 20: {rel20:.2f} (mean se {ind['se(20)'].mean():.1f} vs sd of D(20) "
               f"{ind['D(20)'].std():.1f}).\n")
    out.append(ind.drop(columns="ind").to_markdown(floatfmt=".1f"))

    # ---------------------------------------------------------------- outputs
    tab = pd.concat(tabs, ignore_index=True)
    tab["h"] = tab["h"].map(hlabel)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / "tables" / "m4_horizon_durations.csv", index=False,
               float_format="%.4f")
    figure(curves, spread_curves, RESULTS / "figures" / "m4_dh_curves.png")
    path = RESULTS / "tables" / "m4_horizon_durations.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {path}")


def figure(market: dict, spreads: dict, path) -> None:
    def x_of(idx):
        return [0.5 if h == hr.CLOSE0 else h for h in idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("Market", market["no controls"], market["yield controls"]),
              ("Size 10-1", spreads["Size 10-1"], None),
              ("B/M 10-1", spreads["B/M 10-1"], None)]
    for ax, (title, c, c2) in zip(axes, panels):
        x = x_of(c.index)
        ax.fill_between(x, c["D"] - 1.96 * c["se"], c["D"] + 1.96 * c["se"],
                        color="#2a5d9f", alpha=0.15, lw=0)
        ax.plot(x, c["D"], color="#2a5d9f", marker="o", ms=2.5, lw=1.4, label="no controls")
        if c2 is not None:
            ax.plot(x_of(c2.index), c2["D"], color="#c0392b", lw=1.2, ls="--",
                    label="yield controls")
            ax.legend(fontsize=8, frameon=False)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0.5, color="0.8", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("h (trading days; 0 = 40 min, 0.5 = day-0 close)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("D(h), years, 95% CI")
    fig.suptitle("Post-FOMC horizon durations, 2003-2023 (anchored 30 min before the "
                 "statement)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
