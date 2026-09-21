"""Hand-checkable reference tests for Module 3 tail and distribution analytics."""
import numpy as np
import pandas as pd
from scipy import stats

from research import tail_risk


def test_historical_var_and_expected_shortfall_hand_check():
    returns = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
    threshold = np.quantile(returns, 0.20, method="linear")
    assert np.isclose(tail_risk.historical_var(returns, confidence=0.80), abs(threshold))
    assert tail_risk.historical_expected_shortfall(returns, confidence=0.80) == -returns[returns <= threshold].mean()


def test_parametric_var_and_es_hand_check_for_standard_normal_sample_parameters():
    returns = np.array([-1.0, 0.0, 1.0])
    mean, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(0.05)
    expected_var = -(mean + sigma * z)
    expected_es = -(mean - sigma * stats.norm.pdf(z) / 0.05)
    assert np.isclose(tail_risk.parametric_var(returns), expected_var)
    assert np.isclose(tail_risk.parametric_expected_shortfall(returns), expected_es)


def test_tail_ratio_skew_and_kurtosis_hand_check():
    returns = np.array([-0.10, 0.0, 0.10, 0.20, 0.30])
    expected = np.quantile(returns, 0.95) / abs(np.quantile(returns, 0.05))
    assert np.isclose(tail_risk.tail_ratio(returns), expected)
    out = tail_risk.tail_risk_summary(returns)
    assert np.isclose(out["skewness"], stats.skew(returns, bias=False))
    assert np.isclose(out["excess_kurtosis"], stats.kurtosis(returns, fisher=True, bias=False))


def test_worst_periods_and_drawdown_duration_hand_check():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.Series([0.0, -0.10, 0.0, 1 / 0.9 - 1, -0.20, 0.10], index=index)
    worst = tail_risk.worst_periods(returns)
    assert worst["worst_day"]["return"] == -0.20
    episodes = tail_risk.drawdown_duration_distribution(returns)
    assert episodes[0]["duration_periods"] == 2
    assert episodes[-1]["max_drawdown"] <= -0.20 + 1e-12


def test_distribution_chart_data_and_specs_are_complete():
    data = tail_risk.distribution_chart_data(np.linspace(-0.02, 0.03, 50), bins=5)
    assert len(data["histogram"]) == 5
    assert len(data["qq"]) == 50
    assert {"return_histogram", "normal_qq", "underwater_duration"} <= tail_risk.chart_specs().keys()
