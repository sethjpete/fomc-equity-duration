import numpy as np
import pandas as pd
import pytest

from fomcdur.extension import portfolio_durations as pdur


def _panel(D, T=120, noise=0.002, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2004-01-05", periods=T * 5)[::5]
    s = pd.Series(rng.normal(0, 0.05, T), index=days)  # pp
    r = pd.DataFrame({p: -d * s * 0.01 + rng.normal(0, noise, T) for p, d in enumerate(D)},
                     index=days)
    return r, s


def test_event_portfolio_returns_vw_and_min_stocks():
    monthly = pd.DataFrame({"permno": [1, 2, 3], "month": pd.Timestamp("2010-01-31"),
                            "me": [1.0, 3.0, 2.0], "ind": [7, 7, 9]})
    ev = pd.DataFrame({"permno": [1, 2, 3], "date": pd.Timestamp("2010-02-10"),
                       "ret": [0.01, 0.03, -0.02]})
    r, n = pdur.event_portfolio_returns(ev, monthly, "ind")
    assert r.loc["2010-02-10", 7] == pytest.approx((0.01 + 3 * 0.03) / 4)
    assert n.loc["2010-02-10", 9] == 1
    r2, _ = pdur.event_portfolio_returns(ev, monthly, "ind", min_stocks=2)
    assert np.isnan(r2.loc["2010-02-10", 9]) and r2[7].notna().all()
    m, _ = pdur.event_portfolio_returns(ev, monthly, None)
    assert m["MKT"].iloc[0] == pytest.approx((0.01 + 0.09 - 0.04) / 6)


def test_robustness_samples():
    s = pd.Series([0.3, -0.1, 0.05, 0.2, -0.4, 0.01, 0.02],
                  index=pd.to_datetime(["2004-01-28", "2007-03-21", "2008-03-18", "2009-01-28",
                                        "2015-12-16", "2020-04-29", "2022-03-16"]))
    out = pdur.robustness_samples(s, n_top=2)
    assert len(out["ex top-2 |MPS|"]) == 5 and 0.3 not in out["ex top-2 |MPS|"].values
    assert list(out["ex 2008-09, 2020"].index.year) == [2004, 2007, 2015, 2022]
    assert len(out["2003-2007"]) == 2 and len(out["2008-2023"]) == 5


def test_duration_table_recovers_D():
    r, s = _panel([2.0, 8.0], noise=1e-5)
    t = pdur.duration_table(r, s)
    main = t[t["sample"] == "all"].set_index("portfolio")["D"]
    assert main[0] == pytest.approx(2.0, abs=0.05) and main[1] == pytest.approx(8.0, abs=0.05)


def test_dispersion_checks_separate_signal_from_noise():
    D = np.linspace(2, 10, 20)
    r, s = _panel(D, T=160, noise=0.0005)
    mkt = r.mean(axis=1)
    t = pdur.duration_table(r, s)
    t = t[t["sample"] == "all"]
    assert pdur.signal_share(t["D"], t["se"]) > 0.8
    assert pdur.split_half_reliability(r, s)["spearman_brown"] > 0.8
    assert pdur.equal_duration_test(r, mkt, s, n_perm=200)["p_perm"] < 0.01
    # Equal true durations: no detectable dispersion
    r0, s0 = _panel([5.0] * 20, T=160, noise=0.002, seed=3)
    test = pdur.equal_duration_test(r0, r0.mean(axis=1), s0, n_perm=200)
    assert test["p_perm"] > 0.05
