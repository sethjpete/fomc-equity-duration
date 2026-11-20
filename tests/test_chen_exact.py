import numpy as np
import pandas as pd

from fomcdur.replication import chen_exact as cx


def test_table_a1_events():
    ev = cx.load_events()
    assert len(ev) == 47
    assert (ev["date"] >= "2003-09-10").sum() == 25
    row = ev.set_index("date").loc["2007-09-18"]
    assert np.isclose(row["drf"], -0.00224230769) and row["time"] == "14:15:01"


def test_var1_recovers_coefficients_and_der_m():
    rng = np.random.default_rng(0)
    G = np.diag([0.2, 0.9, 0.95, 0.97, 0.1])
    G[0, 1] = 0.05
    T = 4000
    z = np.zeros((T, 5))
    for t in range(1, T):
        z[t] = G @ z[t - 1] + rng.normal(0, 0.01, 5)
    zz = pd.DataFrame(z, index=pd.bdate_range("1990-01-01", periods=T))
    _, Gh = cx.fit_var1(zz)
    assert np.allclose(Gh, G, atol=0.05)
    t = zz.index[3000]
    d = cx.delta_er_m(zz, pd.DatetimeIndex([t])).iloc[0]
    _, Gt = cx.fit_var1(zz.loc[:zz.index[2999]])
    assert np.isclose(d, Gt[0] @ (zz.iloc[3000] - zz.iloc[2999]).to_numpy())


def test_portfolio_betas():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2005-01-03", periods=252)
    m = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    true = np.linspace(0.2, 2.0, 400)
    R = pd.DataFrame(np.outer(m, true) + rng.normal(0, 0.01, (len(days), 400)),
                     index=days, columns=np.arange(400))
    R.iloc[:200, 0] = np.nan  # too few observations -> no beta
    out = cx.portfolio_betas(R, m, pd.Series(1.0, index=R.columns), n_ports=20)
    assert 0 not in set(out["permno"])
    assert out["port"].nunique() == 20
    assert np.corrcoef(out["beta"], true[out["permno"]])[0, 1] > 0.95
    assert out.groupby("port")["beta"].nunique().max() == 1  # one beta per portfolio


def test_durations_formula_and_trim():
    ev = pd.DataFrame({"ret": [-0.001, 0.001, -0.01], "beta": [1.0, 1.0, 0.0],
                       "der_m": [0.0005, 0.0005, 0.0], "drf": [0.0001, 0.0001, 0.0001]})
    d = cx.durations(ev)
    assert np.isclose(d.loc[0, "D"], 9.75 * 0.001 / 0.0005)   # 19.5
    assert d["kept"].tolist() == [True, False, False]          # negative; 975 > 300
    d252 = cx.durations(ev, er_scale=252)
    assert np.isclose(d252.loc[0, "denom"], 0.0005 * 252)


def test_delta_er_m_parts_sum_to_total():
    rng = np.random.default_rng(2)
    zz = pd.DataFrame(rng.normal(0, 0.01, (800, 3)), columns=["rm", "dy", "term"],
                      index=pd.bdate_range("1990-01-01", periods=800))
    t = pd.DatetimeIndex([zz.index[700]])
    p = cx.delta_er_m_parts(zz, t)
    assert np.isclose(p[["c_rm", "c_dy", "c_term"]].sum(1).iloc[0], p["der_m"].iloc[0])
    assert np.isclose(p["der_m"].iloc[0], cx.delta_er_m(zz, t).iloc[0])
