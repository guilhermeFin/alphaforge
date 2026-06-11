import numpy as np
import pandas as pd

from research import data, factors


def test_momentum_positive_on_uptrend():
    idx = pd.bdate_range("2020-01-01", periods=200)
    close = pd.DataFrame({"A": np.linspace(100, 300, 200)}, index=idx)
    mom = factors.momentum(close, lookback=126, skip=21)
    assert mom["A"].dropna().iloc[-1] > 0


def test_zscore_rows_demeaned():
    panel = data.make_synthetic_panel([f"S{i}" for i in range(10)], periods=150, seed=1).close
    z = factors.cross_sectional_zscore(panel)
    row_means = z.mean(axis=1).dropna()
    assert np.allclose(row_means.values, 0.0, atol=1e-9)


def test_long_short_weights_dollar_neutral():
    panel = data.make_synthetic_panel([f"S{i}" for i in range(12)], periods=200, seed=2).close
    score = factors.cross_sectional_zscore(factors.momentum(panel))
    w = factors.long_short_weights(score, gross=1.0)
    active = w.loc[w.abs().sum(axis=1) > 0]
    np.testing.assert_allclose(active.sum(axis=1).values, 0.0, atol=1e-9)        # market-neutral
    np.testing.assert_allclose(active.abs().sum(axis=1).values, 1.0, atol=1e-9)  # 1$ gross
