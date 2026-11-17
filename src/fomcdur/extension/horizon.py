"""Horizon durations D(h): R_{t->t+h} = a_h - D(h) s_t + delta_h' dy_{t+1->t+h} + controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fomcdur.stats.ols import ols


def horizon_durations(cumret: pd.Series, s: pd.Series, dy: pd.DataFrame | None = None,
                      controls: pd.DataFrame | None = None, scale: float = 0.01) -> pd.DataFrame:
    """D(h) curve for one portfolio.

    cumret:   MultiIndex (event, h) cumulative returns (from events.cumulative_returns)
    s:        surprise indexed by event (pp)
    dy:       MultiIndex (event, h) post-event yield changes (pp), e.g. DGS2, DGS10
    controls: MultiIndex (event, h) cumulative factor returns over the window
    """
    rows = []
    for h, y in cumret.groupby(level="h"):
        y = y.droplevel("h")
        X = (-s.reindex(y.index) * scale).rename("neg_s").to_frame()
        for extra in (dy, controls):
            if extra is not None:
                e = extra.xs(h, level="h").reindex(y.index)
                e = e.loc[:, e.std() > 0]  # dy is identically 0 at h = 0
                X = X.join(e)
        res = ols(y, X)
        rows.append({"h": h, "D": res.params["neg_s"], "se": res.bse["neg_s"], "n": res.nobs})
    return pd.DataFrame(rows).set_index("h")


def same_day_share(curve: pd.DataFrame, h_end: int = 20) -> float:
    return float(curve.loc[0, "D"] / curve.loc[h_end, "D"])


def _design(h, idx, s, dy, controls, scale):
    X = (-s.reindex(idx) * scale).rename("neg_s").to_frame()
    for extra in (dy, controls):
        if extra is not None:
            e = extra.xs(h, level="h").reindex(idx)
            X = X.join(e.loc[:, e.std() > 0])
    return X


def bootstrap_same_day_share(cumret: pd.Series, s: pd.Series, dy=None, controls=None,
                             h_end: int = 20, n_boot: int = 999, seed: int = 0,
                             scale: float = 0.01) -> np.ndarray:
    """Resample events with replacement; returns bootstrap draws of D(0)/D(h_end).

    The same event draw is used at both horizons, so the ratio's sampling
    correlation is preserved.
    """
    rng = np.random.default_rng(seed)
    y0, yH = cumret.xs(0, level="h"), cumret.xs(h_end, level="h")
    idx = y0.index.intersection(yH.dropna().index)
    y0, yH = y0.loc[idx], yH.loc[idx]
    X0 = _design(0, idx, s, dy, controls, scale)
    XH = _design(h_end, idx, s, dy, controls, scale)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        d0 = ols(y0.iloc[pick].reset_index(drop=True), X0.iloc[pick].reset_index(drop=True))
        dH = ols(yH.iloc[pick].reset_index(drop=True), XH.iloc[pick].reset_index(drop=True))
        draws[b] = d0.params["neg_s"] / dH.params["neg_s"]
    return draws
