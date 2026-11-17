import numpy as np
import pandas as pd

from fomcdur.clean.events import align_to_trading_days, cumulative_returns, post_event_changes


def test_align_moves_weekend_to_next_trading_day():
    td = pd.bdate_range("2020-01-01", "2020-01-31")
    out = align_to_trading_days(pd.to_datetime(["2020-01-04"]), td)  # Saturday
    assert out[0] == pd.Timestamp("2020-01-06")


def test_cumulative_returns_compound():
    d = pd.bdate_range("2020-01-01", periods=5)
    r = pd.DataFrame({"a": [0.1, 0.1, 0.0, 0.0, 0.0]}, index=d)
    c = cumulative_returns(r, d[[0]], max_h=2)["a"]
    assert np.allclose(c.to_numpy(), [0.1, 0.21, 0.21])


def test_post_event_changes_zero_at_h0_and_nan_past_end():
    d = pd.bdate_range("2020-01-01", periods=3)
    y = pd.DataFrame({"DGS2": [1.0, 1.1, 1.3]}, index=d)
    dy = post_event_changes(y, d[[1]], max_h=2)["DGS2"]
    assert dy.iloc[0] == 0 and np.isclose(dy.iloc[1], 0.2) and np.isnan(dy.iloc[2])
