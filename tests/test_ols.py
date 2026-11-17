import numpy as np
import pandas as pd

from fomcdur.stats.ols import ols


def test_recovers_coefficients_and_hc1():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n) * (1 + np.abs(x))  # heteroskedastic
    res = ols(pd.Series(y), pd.DataFrame({"x": x}))
    assert abs(res.params["x"] - 2.0) < 0.1
    # HC1 by hand
    X = np.column_stack([np.ones(n), x])
    e = y - X @ res.params.to_numpy()
    bread = np.linalg.inv(X.T @ X)
    V = bread @ (X.T * e**2) @ X @ bread * n / (n - 2)
    assert np.allclose(res.bse.to_numpy(), np.sqrt(np.diag(V)))
