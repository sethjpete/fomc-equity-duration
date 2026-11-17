import numpy as np
import pandas as pd
import pytest

from fomcdur.clean.universe import adjust_for_delisting, impute_delisting_returns


def dsf():
    d = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    return pd.DataFrame({
        "permno": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "date": list(d) * 3,
        "ret": [0.01, 0.02, 0.10] * 3,
        "exchcd": [1] * 3 + [3] * 3 + [1] * 3,
    })


def test_delisting_attached_to_last_trading_day_with_imputation():
    delist = pd.DataFrame({
        "permno": [1, 2, 3],
        "date": pd.to_datetime(["2020-01-09", "2020-01-08", "2020-01-06"]),  # after last day
        "dlret": [np.nan, np.nan, 0.05],
        "dlstcd": [552, 560, 233],  # NYSE perf., Nasdaq perf., merger with dlret
    })
    out = adjust_for_delisting(dsf(), delist).set_index(["permno", "date"])["ret"]
    last = pd.Timestamp("2020-01-06")
    assert out[(1, last)] == pytest.approx(1.10 * 0.70 - 1)
    assert out[(2, last)] == pytest.approx(1.10 * 0.45 - 1)
    assert out[(3, last)] == pytest.approx(1.10 * 1.05 - 1)
    assert out[(1, pd.Timestamp("2020-01-03"))] == 0.02  # earlier days untouched


def test_active_and_nonperformance_missing_are_left_alone():
    delist = pd.DataFrame({"permno": [1, 2], "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
                           "dlret": [np.nan, np.nan], "dlstcd": [100, 231]})
    out = adjust_for_delisting(dsf(), delist)
    assert np.allclose(out["ret"], dsf()["ret"])
    imp = impute_delisting_returns(delist)
    assert not imp["dlret_imputed"].any()


def test_stale_delisting_beyond_tolerance_ignored():
    delist = pd.DataFrame({"permno": [1], "date": pd.to_datetime(["2020-06-30"]),
                           "dlret": [-0.5], "dlstcd": [552]})
    assert np.allclose(adjust_for_delisting(dsf(), delist)["ret"], dsf()["ret"])
