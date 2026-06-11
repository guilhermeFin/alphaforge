"""The most important tests in the repo: prove the honesty guards hold."""
import numpy as np
import pandas as pd

from research import data, factors
from research.backtest import backtest


def _toy():
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 121.0, 121.0], index=idx)
    signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
    return close, signal


def test_position_is_lagged_no_costs():
    close, signal = _toy()
    res = backtest(close, signal, cost_bps=0.0, execution_lag=1)
    # position held = yesterday's signal
    assert list(res.positions.values) == [0.0, 1.0, 1.0, 0.0]
    # returns = pos * pct_change
    np.testing.assert_allclose(res.returns.values, [0.0, 0.10, 0.10, 0.0], atol=1e-12)
    np.testing.assert_allclose(res.equity.values, [1.0, 1.10, 1.21, 1.21], atol=1e-12)


def test_costs_reduce_return_monotonically():
    close, signal = _toy()
    free = backtest(close, signal, cost_bps=0.0).equity.iloc[-1]
    cheap = backtest(close, signal, cost_bps=10.0).equity.iloc[-1]
    pricey = backtest(close, signal, cost_bps=100.0).equity.iloc[-1]
    assert free > cheap > pricey


def test_zero_turnover_pays_no_costs():
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series(np.linspace(100, 120, 5), index=idx)
    signal = pd.Series(1.0, index=idx)  # constant -> no turnover after entry
    no_cost = backtest(close, signal, cost_bps=0.0)
    with_cost = backtest(close, signal, cost_bps=50.0)
    # only the single entry on day 1 differs; days 2+ identical
    np.testing.assert_allclose(no_cost.returns.values[2:], with_cost.returns.values[2:], atol=1e-12)


def test_execution_lag_zero_is_rejected():
    close, signal = _toy()
    try:
        backtest(close, signal, execution_lag=0)
    except ValueError:
        return
    raise AssertionError("execution_lag=0 should raise (look-ahead)")


def test_point_in_time_consistency():
    """THE no-look-ahead proof: running the full pipeline on data truncated at k
    must reproduce the first k points of the equity curve exactly. If any step
    peeked into the future, the prefixes would diverge."""
    panel = data.make_synthetic_panel([f"S{i}" for i in range(8)], periods=400, seed=7)
    close = panel.close

    def pipeline(px):
        score = factors.cross_sectional_zscore(factors.momentum(px))
        w = factors.long_short_weights(score)
        return backtest(px, w, cost_bps=5.0).equity

    full = pipeline(close)
    for k in (200, 300, 350):
        trunc = pipeline(close.iloc[:k])
        np.testing.assert_allclose(full.iloc[:k].values, trunc.values, atol=1e-12)
