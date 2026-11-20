import numpy as np
import pandas as pd

from fomcdur.data import taq
from fomcdur.replication import intraday


def test_window_and_sql():
    w = taq.window(pd.Timestamp("2008-12-16"), 14.25)  # 2:15pm
    assert w == {"t0": "13:45:00", "t1": "14:25:00", "lo": "13:30:00"}
    sql = taq.nbbo_sql(pd.Timestamp("2022-03-16"), 14.0)
    assert "taqm_2022.complete_nbbo_20220316" in sql
    assert "'13:30:00'" in sql and "'14:10:00'" in sql and "'13:15:00'" in sql


def test_event_returns_matches_and_filters():
    d = pd.Timestamp("2008-12-16")
    nbbo = pd.DataFrame({
        "sym_root": ["AAA", "BBB", "BBB", "CCC", "DDD"],
        "sym_suffix": [None, None, "A", None, None],
        "mid_pre": [10.0, 20.0, 5.0, 0.0, 10.0],
        "mid_post": [10.5, 19.0, 5.1, 3.0, 30.0],
        "n_quotes": [100, 50, 5, 10, 10],
        "date": d,
    })
    link = pd.DataFrame({
        "date": [d] * 6,
        "sym_root": ["AAA", "AAA", "BBB", "BBB", "CCC", "DDD"],
        "sym_suffix": ["", "", None, "A", None, None],
        "permno": [1, 9, 2, 2, 3, 4],
        "match_lvl": [1, 2, 1, 1, 1, 1],
    })
    out = intraday.event_returns(nbbo, link).set_index("permno")
    assert sorted(out.index) == [1, 2]          # CCC: zero mid; DDD: +200% dropped
    assert np.isclose(out.loc[1, "ret"], 0.05)  # best match_lvl wins
    assert np.isclose(out.loc[2, "ret"], -0.05)  # BBB (more quotes) kept over BBB.A
