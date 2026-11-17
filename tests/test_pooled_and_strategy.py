import numpy as np
import pandas as pd

from fomcdur.extension.pooled_model import fit_pooled
from fomcdur.portfolios import assign_quantiles, high_minus_low, value_weighted_returns
from fomcdur.strategy.backtest import event_positions, strategy_returns


def test_pooled_model_recovers_loadings():
    rng = np.random.default_rng(1)
    n_ev, n_st = 120, 300
    events = pd.date_range("2000-01-31", periods=n_ev, freq="45D")
    s = rng.normal(0, 5, n_ev)
    z = rng.normal(size=(n_ev, n_st))
    D = 5 + 2 * z  # b0 = 5, b_z = 2
    r = -D * (s[:, None] * 0.01) + rng.normal(0, 0.01, (n_ev, n_st))
    panel = pd.DataFrame({
        "event": np.repeat(events, n_st), "s": np.repeat(s, n_st),
        "z_x": z.ravel(), "ret": r.ravel(),
    })
    b = fit_pooled(panel, ["z_x"], cutoff=events[-1] + pd.Timedelta(days=1))
    assert abs(b["b0"] - 5) < 0.3 and abs(b["z_x"] - 2) < 0.3


def test_quantiles_and_vw_hl():
    df = pd.DataFrame({"month": pd.Timestamp("2020-01-31"), "permno": range(10),
                       "sig": range(10), "me": 1.0})
    df["port"] = assign_quantiles(df, "sig", 5)
    assert sorted(df["port"].unique()) == [1, 2, 3, 4, 5]
    daily = pd.DataFrame({"permno": range(10), "date": pd.Timestamp("2020-02-03"),
                          "ret": [0.0] * 2 + [0.01] * 6 + [0.05] * 2})
    hl = high_minus_low(value_weighted_returns(daily, df))
    assert np.isclose(hl.iloc[0], 0.05)


def test_strategy_shorts_after_tightening():
    d = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series([+2.0, -1.0], index=d[[1, 5]])
    pos = event_positions(d, s.index, s, hold=2)
    assert list(pos) == [0, 0, -1, -1, 0, 0, 1, 1, 0, 0]
    out = strategy_returns(pd.Series(0.01, index=d), pos, cost_bps_per_side=10)
    assert np.isclose(out["gross"].sum(), 0.0)
    assert out["net"].sum() < 0
