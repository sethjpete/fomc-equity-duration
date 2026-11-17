import numpy as np

from fomcdur.clean.events import cumulative_returns
from fomcdur.extension.horizon import bootstrap_same_day_share, horizon_durations, same_day_share
from fomcdur.replication.duration import event_duration


def test_day0_duration(drift_world):
    w = drift_world
    est = event_duration(w["ret"].loc[w["events"]], w["s"])
    assert abs(est["D"] - w["D0"]) < 3 * est["se"]


def test_horizon_curve_and_share(drift_world):
    w = drift_world
    cr = cumulative_returns(w["ret"].to_frame("HL"), w["events"], w["H"])["HL"]
    curve = horizon_durations(cr, w["s"])
    assert list(curve.index) == list(range(w["H"] + 1))
    assert abs(curve.loc[w["H"], "D"] - w["DH"]) < 3 * curve.loc[w["H"], "se"]
    share = same_day_share(curve, w["H"])
    assert abs(share - w["D0"] / w["DH"]) < 0.1
    draws = bootstrap_same_day_share(cr, w["s"], h_end=w["H"], n_boot=50)
    assert np.percentile(draws, 2.5) < w["D0"] / w["DH"] < np.percentile(draws, 97.5)
