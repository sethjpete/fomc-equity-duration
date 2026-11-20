"""Chen (2022, JFQA) stock-level effective duration, implemented as in the paper.

    D_it = (390/40) * -(dP/P)_it / [beta_it * dER_M,t + (1 - beta_it) * dRf_t]     (eq. 16)

Pieces (page numbers refer to the JFQA article; read 2026-11-20):
- Announcements and dRf: his Internet Appendix Table A.1 (config/chen_table_a1.csv): 47
  announcements 1995-2016 with fed funds futures surprises (U/(U-u) scaled, [s-60, s+5] min).
- dP/P: mean NBBO midpoint over (s, s+10] over the mean over [s-30, s), minus 1 (p. 678);
  `data.taq.pull_chen`. Only from 2003-09 (Millisecond TAQ); 25 of the 47 announcements.
- beta: daily returns over the past year ending t-1 on the CRSP VW index; stocks sorted into
  100 portfolios on that beta, VW portfolio returns over the same year, post-ranking portfolio
  beta assigned to members (p. 679). Weights: ME at t-1 (not stated in the paper).
- dER_M: daily first-order VAR (eq. 15) on market return, dividend yield, TERM (10y - 3m),
  DEF (BAA - AAA) and HML, extending window from 1990 (p. 681). E_t[r_M,t+1] = e1'(a + G z_t),
  so dER_M,t = e1' G (z_t - z_{t-1}), with coefficients estimated on data through t-1.
- Filters: shrcd 10/11, NYSE/AMEX/Nasdaq, ex SIC 4900-4999 and 6000-6999, ex stocks below the
  NYSE 20th percentile ME (prior month end). Estimates outside [1, 300] years are dropped
  (his posted files have no mass at the bounds, so they are not clipped).

Not stated in the paper and therefore set here, to be checked against his posted estimates:
units of dER_M (daily vs annualized: `er_scale`), the 100-portfolio weights, the treatment of
negative denominators (the ratio is computed as is; the trim drops negative D).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.config import ROOT

SCALE_40MIN = 390 / 40
LO, HI = 1.0, 300.0
BETA_WINDOW = 252
BETA_MIN_OBS = 126
N_BETA_PORTS = 100
VAR_START = "1990-01-01"


def load_events() -> pd.DataFrame:
    """Chen's 47 announcements: date, time ('HH:MM:SS' ET), drf (decimal, e.g. 0.00065)."""
    df = pd.read_csv(ROOT / "config" / "chen_table_a1.csv", comment="#")
    df["date"] = pd.to_datetime(df["date"])
    df["drf"] = df["surprise_pct"] / 100.0
    return df[["date", "time", "drf"]]


# ---------------------------------------------------------------- VAR

def var_states(dsi: pd.DataFrame, yields: pd.DataFrame, hml: pd.Series) -> pd.DataFrame:
    """Daily state vector z = [rm, dy, term, def, hml].

    dsi: [date, vwretd, vwretx]; yields: DGS10, DTB3, DBAA, DAAA in percent (date index);
    hml: daily HML in decimals. Dividend yield = dividends over the past 252 days / index level.
    """
    # wrds returns nullable Float64 columns; mixing them with float64 gives object arrays
    d = dsi.sort_values("date").set_index("date")[["vwretd", "vwretx"]].astype("float64")
    level = (1 + d["vwretx"]).cumprod()
    div = (d["vwretd"] - d["vwretx"]) * level.shift(1)
    dy = div.rolling(252, min_periods=200).sum() / level
    y = yields.sort_index().ffill().reindex(d.index).ffill()
    z = pd.DataFrame({
        "rm": d["vwretd"],
        "dy": dy,
        "term": (y["DGS10"] - y["DTB3"]) / 100,
        "def": (y["DBAA"] - y["DAAA"]) / 100,
        "hml": hml.reindex(d.index),
    }).astype("float64")
    return z.dropna()


def monthly_states(z: pd.DataFrame) -> pd.DataFrame:
    """Daily states -> month-end states: rm and hml compounded, dy/term/def at month end."""
    g = z.groupby(z.index.to_period("M"))
    out = pd.DataFrame({
        "rm": g["rm"].apply(lambda x: (1 + x).prod() - 1),
        "dy": g["dy"].last(), "term": g["term"].last(), "def": g["def"].last(),
        "hml": g["hml"].apply(lambda x: (1 + x).prod() - 1),
    })
    out.index = out.index.to_timestamp("M")
    return out[z.columns]


def der_m_window_monthly(zm: pd.DataFrame, dates: pd.DatetimeIndex,
                         mkt_win: pd.Series) -> pd.DataFrame:
    """Monthly VAR(1) fit on months before the event month; the announcement moves the
    return state by the 40-min market return and the dividend yield by -dy * r (prices up,
    yield down). Returns g_rm, g_dy and der (monthly units) per event."""
    rows = {}
    for t in dates:
        hist = zm.loc[:t + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)]
        if len(hist) < 60 or t not in mkt_win.index:
            continue
        _, G = fit_var1(hist)
        r = float(mkt_win[t])
        dy = float(hist["dy"].iloc[-1])
        rows[t] = {"g_rm_m": G[0, 0], "g_dy_m": G[0, 1],
                   "der_m_month": G[0, 0] * r + G[0, 1] * (-dy * r)}
    return pd.DataFrame(rows).T


def fit_var1(z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """OLS VAR(1): z_t = a + G z_{t-1} + u_t. Returns (a, G)."""
    Y = z.iloc[1:].to_numpy()
    X = np.column_stack([np.ones(len(z) - 1), z.iloc[:-1].to_numpy()])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return B[0], B[1:].T


def delta_er_m(z: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """dER_M on each event date: e1' G (z_t - z_{t-1}), G fitted on data through t-1."""
    out = {}
    for t in dates:
        hist = z.loc[:t - pd.Timedelta(days=1)]
        if t not in z.index or len(hist) < 500:
            out[t] = np.nan
            continue
        _, G = fit_var1(hist)
        p = z.index.get_loc(t)
        out[t] = float(G[0] @ (z.iloc[p].to_numpy() - z.iloc[p - 1].to_numpy()))
    return pd.Series(out, name="der_m")


def delta_er_m_parts(z: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per event date: the VAR's return-equation coefficients g_k (G[0, k]), the state changes
    dz_k = z_t - z_{t-1}, their products c_k and the total der_m = sum c_k (as `delta_er_m`)."""
    rows = {}
    for t in dates:
        hist = z.loc[:t - pd.Timedelta(days=1)]
        if t not in z.index or len(hist) < 500:
            continue
        _, G = fit_var1(hist)
        p = z.index.get_loc(t)
        dz = z.iloc[p].to_numpy() - z.iloc[p - 1].to_numpy()
        r = {}
        for k, name in enumerate(z.columns):
            r[f"g_{name}"], r[f"dz_{name}"], r[f"c_{name}"] = G[0, k], dz[k], G[0, k] * dz[k]
        r["der_m"] = float(G[0] @ dz)
        rows[t] = r
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- betas

def _betas(R: np.ndarray, m: np.ndarray, min_obs: int) -> np.ndarray:
    """Column-wise OLS slope of R (T x N, NaN = missing) on m (T), using each column's rows."""
    mask = ~np.isnan(R)
    M = np.where(mask, m[:, None], np.nan)
    n = mask.sum(0)
    rm = np.nanmean(R, 0)
    mm = np.nanmean(M, 0)
    cov = np.nansum((R - rm) * (M - mm), 0)
    var = np.nansum((M - mm) ** 2, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = cov / var
    b[n < min_obs] = np.nan
    return b


def portfolio_betas(rets: pd.DataFrame, mkt: pd.Series, me: pd.Series,
                    n_ports: int = N_BETA_PORTS, min_obs: int = BETA_MIN_OBS) -> pd.DataFrame:
    """rets: days x permnos over the estimation year; mkt: index returns on the same days;
    me: ME per permno (weights). Returns [permno, beta_pre, port, beta]."""
    rets = rets.astype("float64")
    m = mkt.astype("float64").reindex(rets.index).to_numpy()
    pre = pd.Series(_betas(rets.to_numpy(), m, min_obs), index=rets.columns).dropna()
    port = pd.qcut(pre.rank(method="first"), n_ports, labels=False)
    w = me.astype("float64").reindex(pre.index).fillna(0.0)
    post = {}
    for k, members in pre.groupby(port).groups.items():
        sub = rets[members]
        ww = w[members].to_numpy()
        valid = ~np.isnan(sub.to_numpy())
        num = np.nansum(sub.to_numpy() * ww, 1)
        den = (valid * ww).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = np.where(den > 0, num / den, np.nan)
        post[k] = _betas(pr[:, None], m, min_obs)[0]
    out = pd.DataFrame({"permno": pre.index, "beta_pre": pre.to_numpy(), "port": port.to_numpy()})
    out["beta"] = out["port"].map(post)
    return out


# ---------------------------------------------------------------- durations

def durations(ev: pd.DataFrame, er_scale: float = 1.0, sign: int = 1) -> pd.DataFrame:
    """ev: [permno, date, ret, beta, der_m, drf]. dER_M is multiplied by `er_scale`
    (1 = daily units as estimated, 252 = annualized). sign = -1 flips the denominator, i.e.
    discount-rate changes measured as decreases (his Table 1 reports dER as positive
    "decreases" while the mean event return is positive). Adds denom, D, kept."""
    out = ev.copy()
    out["denom"] = sign * (out["beta"] * out["der_m"] * er_scale
                           + (1 - out["beta"]) * out["drf"])
    with np.errstate(invalid="ignore", divide="ignore"):
        out["D"] = SCALE_40MIN * -out["ret"] / out["denom"]
    out["kept"] = out["D"].between(LO, HI)
    return out


def implied_denominators(matched: pd.DataFrame) -> pd.DataFrame:
    """Back out Chen's denominator from his D and our return: denom = 9.75 * -ret / D_chen.
    Within an announcement it should be linear in beta: a + b * beta with a = dRf and
    a + b = dER_M. Returns per-event intercept, slope, implied dER_M, R2 and n."""
    rows = {}
    for d, g in matched.groupby("date"):
        g = g.dropna(subset=["beta", "duration", "ret"])
        if len(g) < 30:
            continue
        y = SCALE_40MIN * -g["ret"] / g["duration"]
        X = np.column_stack([np.ones(len(g)), g["beta"]])
        b, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
        fit = X @ b
        r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows[d] = {"a (dRf?)": b[0], "b": b[1], "a+b (dER_M?)": b[0] + b[1], "R2": r2,
                   "n": len(g)}
    return pd.DataFrame(rows).T
