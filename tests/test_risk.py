"""Hand-checkable reference tests for Module 1 downside-risk analytics."""
import math

import numpy as np
import pandas as pd
from scipy import stats

from research import risk


def test_annualized_volatility_matches_sample_std_formula():
    returns = [0.01, -0.01]
    expected = np.std(returns, ddof=1) * math.sqrt(252)
    assert risk.annualized_volatility(returns) == expected


def test_downside_deviation_and_sortino_hand_check():
    # DD = sqrt((0^2 + (-0.01)^2)/2) and mean return is 0.005.
    returns = [0.02, -0.01]
    expected_dd = math.sqrt(0.0001 / 2.0)
    assert risk.downside_deviation(returns) == expected_dd
    assert risk.sortino_ratio(returns, periods_per_year=1) == 0.005 / expected_dd


def test_ulcer_and_pain_index_hand_check():
    # Equity: 1.0, 0.9, 0.9, 1.0 -> drawdowns 0, -10%, -10%, 0.
    returns = [0.0, -0.10, 0.0, 1 / 0.9 - 1]
    assert np.isclose(risk.ulcer_index(returns), math.sqrt(0.005))
    assert np.isclose(risk.pain_index(returns), 0.05)
    curve = risk.underwater_curve(returns)
    assert curve["time_under_water_periods"].tolist() == [0, 1, 2, 0]


def test_calmar_and_mar_ratio_hand_check():
    # CAGR=10% with P=3; deepest drawdown=20%; Calmar=MAR=0.5.
    returns = [0.10, -0.20, 0.25]
    assert np.isclose(risk.calmar_ratio(returns, periods_per_year=3), 0.5)
    assert np.isclose(risk.mar_ratio(returns, periods_per_year=3), 0.5)


def test_omega_and_modified_sterling_hand_check():
    assert np.isclose(risk.omega_ratio([-0.10, 0.10, 0.20]), 3.0)
    # Same 10% CAGR / (20% annual max DD - 10% adjustment) = 1.0.
    assert np.isclose(risk.sterling_ratio([0.10, -0.20, 0.25], periods_per_year=3), 1.0)


def test_k_ratio_matches_ols_slope_over_standard_error():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])
    equity = np.cumprod(1.0 + returns.to_numpy())
    fit = stats.linregress(np.arange(len(equity)), np.log(equity))
    assert np.isclose(risk.k_ratio(returns), fit.slope / fit.stderr)


def test_summary_and_chart_specs_are_complete():
    summary = risk.downside_risk_summary([0.01, -0.02, 0.01, 0.03])
    assert {"annualized_volatility", "ulcer_index", "pain_index", "omega_ratio", "k_ratio"} <= summary.keys()
    assert {"underwater", "drawdown_distribution"} <= risk.chart_specs().keys()
