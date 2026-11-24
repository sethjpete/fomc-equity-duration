import numpy as np
import pandas as pd
import pytest

from fomcdur.extension import horizon_returns as hr


def test_stock_cumulative_anchors_at_pre_price():
    td = pd.bdate_range("2010-03-01", periods=10)
    e = td[2]
    win = hr.event_windows(pd.DatetimeIndex([e]), td, 3)
    assert list(win["date"]) == list(td[2:6])
    anchor = pd.DataFrame({"permno": [1], "date": [e], "mid_pre": [100.0], "ret40": [0.01]})
    daily = pd.DataFrame({"permno": 1, "date": td,
                          "ret": [0, 0, 0.5, 0.02, np.nan, -0.01, 0, 0, 0, 0],
                          "prc": [-102.0] * 10})  # negative = bid/ask midpoint
    out = hr.stock_cumulative(anchor, daily, win, 3).set_index("h")["cum"]
    assert out[0] == pytest.approx(0.01)            # 40-minute return
    assert out[hr.CLOSE0] == pytest.approx(0.02)    # |prc| / mid_pre - 1, not CRSP ret
    assert out[1] == pytest.approx(1.02 * 1.02 - 1)
    assert out[2] == pytest.approx(out[1])          # missing return counts as zero
    assert out[3] == pytest.approx(1.02 * 1.02 * 0.99 - 1)


def test_window_past_data_end_is_nan():
    td = pd.bdate_range("2010-03-01", periods=6)
    e = td[3]
    win = hr.event_windows(pd.DatetimeIndex([e]), td, 4)  # only h = 0..2 exist
    anchor = pd.DataFrame({"permno": [1], "date": [e], "mid_pre": [10.0], "ret40": [0.0]})
    daily = pd.DataFrame({"permno": 1, "date": td, "ret": 0.01, "prc": 10.0})
    out = hr.stock_cumulative(anchor, daily, win, 4).set_index("h")["cum"]
    assert out[2] == pytest.approx(1.01**2 - 1) and np.isnan(out[3]) and np.isnan(out[4])


def test_portfolio_cumulative_buy_and_hold_weights():
    e = pd.Timestamp("2010-03-17")
    stock = pd.DataFrame({"permno": [1, 2, 1, 2], "event": e, "h": [0, 0, 1, 1],
                          "cum": [0.01, 0.03, 0.02, np.nan]})
    monthly = pd.DataFrame({"permno": [1, 2], "month": pd.Timestamp("2010-02-28"),
                            "me": [1.0, 3.0]})
    out = hr.portfolio_cumulative(stock, monthly, None)["MKT"]
    assert out[(e, 0)] == pytest.approx(0.025) and out[(e, 1)] == pytest.approx(0.02)


def test_level_changes_and_factors():
    td = pd.bdate_range("2010-03-01", periods=6)
    win = hr.event_windows(pd.DatetimeIndex([td[2]]), td, 2)
    lv = pd.DataFrame({"y": [1.0, 1.1, 1.3, 1.2, 1.5, 1.5]}, index=td)
    ch = hr.level_changes_from(lv, win, base_offset=-1)["y"]
    assert list(ch.round(10)) == [0.2, 0.1, 0.4]
    f = pd.DataFrame({"Mkt": [0.0, 0.0, 0.05, 0.01, 0.02, 0.0]}, index=td)
    cf = hr.cumulative_factors(f, win)["Mkt"]
    assert list(cf.round(10)) == [0.0, 0.01, round(1.01 * 1.02 - 1, 10)]
