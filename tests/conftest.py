import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def drift_world():
    """250 events; HL portfolio has same-day duration 6 rising linearly to 10 by h=20."""
    rng = np.random.default_rng(42)
    n_ev, H, gap = 250, 20, 30
    dates = pd.bdate_range("1994-01-03", periods=n_ev * gap + H + 5)
    events = dates[5::gap][:n_ev]
    s = pd.Series(rng.normal(0, 5, n_ev), index=events)  # pp; sd 5bp
    ret = pd.Series(rng.normal(0, 0.002, len(dates)), index=dates)
    D0, DH = 6.0, 10.0
    for e in events:
        p = dates.get_loc(e)
        ret.iloc[p] += -D0 * s[e] * 0.01
        for k in range(1, H + 1):  # (DH-D0)/H of extra response per day
            ret.iloc[p + k] += -(DH - D0) / H * s[e] * 0.01
    return dict(dates=dates, events=events, s=s, ret=ret, D0=D0, DH=DH, H=H)
