import numpy as np
import pandas as pd

from research import metrics


def test_max_drawdown_known():
    eq = pd.Series([1.0, 2.0, 1.5, 3.0])
    assert abs(metrics.max_drawdown(eq) - (-0.25)) < 1e-12


def test_sharpe_signs():
    pos = pd.Series([0.01] * 100)            # constant positive, zero vol-> handled as 0
    assert metrics.annualised_sharpe(pos) == 0.0  # zero std -> 0 by convention
    rng = np.random.default_rng(0)
    good = pd.Series(0.001 + 0.005 * rng.standard_normal(1000))
    assert metrics.annualised_sharpe(good) > 0
    zero = pd.Series(0.005 * rng.standard_normal(1000))
    assert abs(metrics.annualised_sharpe(zero)) < 1.0


def test_psr_bounds_and_behaviour():
    rng = np.random.default_rng(1)
    strong = pd.Series(0.001 + 0.003 * rng.standard_normal(2000))
    psr = metrics.probabilistic_sharpe_ratio(strong, 0.0)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.95  # a genuine, long, positive-Sharpe record clears the bar


def test_deflated_sr_discounts_multiple_trials():
    rng = np.random.default_rng(2)
    r = pd.Series(0.0006 + 0.004 * rng.standard_normal(1500))
    psr = metrics.probabilistic_sharpe_ratio(r, 0.0)
    dsr_many = metrics.deflated_sharpe_ratio(r, n_trials=100)
    assert dsr_many <= psr  # trying 100 strategies must lower confidence


def test_expected_max_sharpe_grows_with_trials():
    a = metrics.expected_max_sharpe_pp(10, 0.05)
    b = metrics.expected_max_sharpe_pp(1000, 0.05)
    assert b > a > 0


def test_summarize_keys():
    rng = np.random.default_rng(3)
    r = pd.Series(0.0005 + 0.01 * rng.standard_normal(500))
    s = metrics.summarize(r, n_trials=5)
    for key in ("ann_sharpe", "max_drawdown", "psr_vs_0", "deflated_sr",
                "skew", "excess_kurtosis", "returns_are_normal"):
        assert key in s
