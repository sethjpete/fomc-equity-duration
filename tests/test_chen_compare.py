import numpy as np
import pandas as pd

from fomcdur.replication import chen_compare as cc


def test_extract_from_daily_keeps_month_end_and_event_rows():
    dates = pd.to_datetime(["2000-01-28", "2000-01-31", "2000-02-01", "2000-02-29"])
    dsf = pd.DataFrame({
        "permno": [1, 1, 1, 1], "date": dates, "ret": [0.01, 0.02, -0.01, 0.0],
        "prc": [10.0, -11.0, 12.0, np.nan], "shrout": [1000, 1000, 1000, 1000],
        "exchcd": 1, "siccd": 2000,
    })
    monthly, event = cc.extract_from_daily(dsf, pd.DatetimeIndex(["2000-02-01"]))
    # January: last priced day 01-31 (negative prc = bid/ask midpoint); February: only 02-01
    assert list(monthly["month"].dt.strftime("%Y-%m-%d")) == ["2000-01-31", "2000-02-29"]
    assert np.allclose(monthly["me"], [11.0, 12.0])
    assert list(event["date"]) == [pd.Timestamp("2000-02-01")]


def test_size_decile_uses_breakpoints():
    month = pd.Timestamp("2000-01-31")
    bp = pd.DataFrame([np.arange(10, 100, 10, dtype=float)], index=[month],
                      columns=cc.SIZE_BREAKPOINTS)
    m = pd.DataFrame({"month": month, "me": [5.0, 10.0, 55.0, 95.0, np.nan]})
    assert cc.add_size_decile(m, bp).tolist()[:4] == [1, 2, 6, 10]
    assert np.isnan(cc.add_size_decile(m, bp).iloc[4])
    m2 = m.assign(month=pd.Timestamp("2000-02-29"))  # no breakpoints that month
    assert cc.add_size_decile(m2, bp).isna().all()


def test_book_to_market_respects_availability_and_staleness():
    monthly = pd.DataFrame({
        "permno": 1, "me": 100.0,
        "month": pd.to_datetime(["2000-05-31", "2000-06-30", "2001-06-30", "2001-07-31"]),
    })
    chars = pd.DataFrame({"permno": [1], "available": [pd.Timestamp("2000-06-30")],
                          "be": [50.0]})
    out = cc.add_book_to_market(monthly, chars).set_index("month")
    assert np.isnan(out.loc["2000-05-31", "bm"])         # not yet available
    assert out.loc["2000-06-30", "bm"] == 0.5
    assert out.loc["2001-06-30", "bm"] == 0.5             # 12 months after: still used
    assert np.isnan(out.loc["2001-07-31", "bm"])         # stale


def _world(seed=0, n_events=150, n_per_dec=20):
    """Deciles with true day-0 duration D_k = 2k; Chen durations ~ 10k per stock."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1995-01-02", periods=n_events * 25)
    day0 = days[10::25][:n_events]
    s = pd.Series(rng.normal(0, 0.05, n_events), index=day0)  # pp
    permnos = np.arange(10 * n_per_dec)
    dec = permnos // n_per_dec + 1
    months = pd.date_range("1994-12-31", day0.max() + pd.offsets.MonthEnd(0), freq="ME")
    monthly = pd.DataFrame([(p, m, float(rng.uniform(1, 100)), d)
                            for m in months for p, d in zip(permnos, dec)],
                           columns=["permno", "month", "me", "size_dec"])
    event = pd.DataFrame([(p, e, -2 * d * s[e] * 0.01 + rng.normal(0, 0.002))
                          for e in day0 for p, d in zip(permnos, dec)],
                         columns=["permno", "date", "ret"])
    chen = pd.DataFrame([(p, e, 10 * d + rng.normal(0, 3))
                         for e in day0[::5] for p, d in zip(permnos, dec)],
                        columns=["permno", "eventdate", "duration"])
    return s, monthly, event, chen


def test_decile_durations_recover_truth_and_rank_agreement():
    s, monthly, event, chen = _world()
    rets = cc.event_portfolio_returns(event, monthly, "size_dec")
    ours = cc.durations_with_spread(rets, s)
    D = ours["D"].astype(float)
    assert np.allclose(D.loc[list(range(1, 11))], 2 * np.arange(1, 11), atol=0.8)
    assert abs(D.loc["10-1"] - 18) < 1.0
    theirs = cc.chen_portfolio_durations(chen, monthly, "size_dec")
    assert np.allclose(theirs.loc[list(range(1, 11)), "vw"], 10 * np.arange(1, 11), atol=1.5)
    assert theirs.loc[1, "n_stocks"] == 20
    res = cc.compare(ours, theirs)
    assert res["spearman_vw"] > 0.95 and res["spread_ours_t"] > 5


def test_market_portfolio_and_fin_util():
    _s, monthly, event, chen = _world(n_events=60)
    mkt = cc.event_portfolio_returns(event, monthly, None)
    assert list(mkt.columns) == ["MKT"]
    assert cc.chen_portfolio_durations(chen, monthly, None).index.tolist() == ["MKT"]
    sic = pd.Series([4900, 4999, 5000, 6000, 6999, 7000, np.nan])
    assert cc.is_fin_util(sic).tolist() == [True, True, False, True, True, False, False]
