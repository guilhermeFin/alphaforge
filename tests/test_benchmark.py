"""Hand-checkable reference tests for Module 2 benchmark-relative analytics."""
import math

import numpy as np
import pandas as pd

from research import benchmark


def test_information_ratio_and_tracking_error_hand_check():
    strategy = [0.02, 0.04]
    reference = [0.01, 0.02]
    active = np.array([0.01, 0.02])
    assert benchmark.tracking_error(strategy, reference, periods_per_year=1) == active.std(ddof=1)
    assert benchmark.information_ratio(strategy, reference, periods_per_year=1) == active.mean() / active.std(ddof=1)


def test_beta_and_jensens_alpha_hand_check():
    reference = pd.Series([-0.01, 0.0, 0.01, 0.02])
    strategy = 0.001 + 1.5 * reference
    result = benchmark.jensens_alpha(strategy, reference, periods_per_year=1)
    assert np.isclose(benchmark.beta(strategy, reference), 1.5)
    assert np.isclose(result["alpha_per_period"], 0.001)
    assert np.isclose(result["alpha_annualized"], 0.001)


def test_capture_ratios_and_correlation_hand_check():
    reference = pd.Series([0.01, -0.02, 0.03, -0.01])
    strategy = 2.0 * reference
    captures = benchmark.capture_ratios(strategy, reference)
    assert captures["up_capture"] == 2.0
    assert captures["down_capture"] == 2.0
    assert np.isclose(strategy.corr(reference), 1.0)


def test_rolling_metrics_have_declared_window_and_confidence_bands():
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    reference = pd.Series(np.linspace(-0.02, 0.02, 8), index=index)
    strategy = 0.001 + 1.2 * reference
    rolling = benchmark.rolling_benchmark_metrics(strategy, reference, window=4, periods_per_year=1)
    assert rolling.iloc[:3].isna().all().all()
    assert np.isclose(rolling.iloc[-1]["rolling_beta"], 1.2)
    assert rolling.iloc[-1]["beta_lower"] <= 1.2 <= rolling.iloc[-1]["beta_upper"]
    ic = benchmark.rolling_ic_metrics(pd.Series([0.1, 0.2, 0.0, 0.1], index=index[:4]), window=3)
    assert np.isclose(ic.iloc[-1]["rolling_ic"], 0.1)


def test_summary_returns_compact_chart_records():
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(0.0, 0.01, 80), index=index)
    strategy = 0.0005 + 0.8 * reference + rng.normal(0.0, 0.001, 80)
    out = benchmark.benchmark_relative_summary(strategy, reference, periods_per_year=252)
    assert {"information_ratio", "jensens_alpha", "capture_ratios", "rolling_chart_data"} <= out.keys()
    assert out["rolling_chart_data"][-1]["date"]


# --- degenerate inputs must be named, not silently reduced to NaN -----------


def _flat_and_moving(n=40):
    index = pd.bdate_range("2024-01-02", periods=n)
    flat = pd.Series(np.zeros(n), index=index)
    moving = pd.Series(np.linspace(0.001, 0.004, n), index=index)
    return flat, moving


def test_constant_strategy_returns_give_an_undefined_correlation_with_a_reason():
    flat, moving = _flat_and_moving()
    value, reason = benchmark.return_correlation(flat, moving)
    assert math.isnan(value)
    assert "constant" in reason and "undefined rather than zero" in reason


def test_constant_benchmark_returns_are_also_named():
    flat, moving = _flat_and_moving()
    value, reason = benchmark.return_correlation(moving, flat)
    assert math.isnan(value)
    assert "benchmark return series" in reason


def test_valid_inputs_report_no_reason():
    index = pd.bdate_range("2024-01-02", periods=40)
    a = pd.Series(np.linspace(-0.01, 0.02, 40), index=index)
    b = pd.Series(np.linspace(0.02, -0.01, 40), index=index)
    value, reason = benchmark.return_correlation(a, b)
    assert reason is None
    assert math.isclose(value, -1.0, abs_tol=1e-9)


def test_correlation_on_constant_input_emits_no_numpy_warning(recwarn):
    """The 0/0 that produced the original RuntimeWarning must not happen."""
    flat, moving = _flat_and_moving()
    benchmark.return_correlation(flat, moving)
    assert [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)] == []


def test_summary_surfaces_undefined_statistics_instead_of_bare_nans():
    flat, moving = _flat_and_moving(80)
    summary = benchmark.benchmark_relative_summary(flat, moving, rolling_window=10)
    assert summary["has_undefined_statistics"] is True
    named = {item["statistic"] for item in summary["statistic_diagnostics"]}
    assert "correlation" in named
    assert all(item["status"] == "undefined" for item in summary["statistic_diagnostics"])
    assert all(item["reason"] for item in summary["statistic_diagnostics"])
    assert "not a weak result" in summary["diagnostics_note"]


def test_summary_on_healthy_inputs_reports_no_undefined_statistics():
    index = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(7)
    strategy = pd.Series(rng.normal(0.0005, 0.01, 80), index=index)
    reference = pd.Series(rng.normal(0.0003, 0.008, 80), index=index)
    summary = benchmark.benchmark_relative_summary(strategy, reference, rolling_window=10)
    assert summary["has_undefined_statistics"] is False
    assert summary["statistic_diagnostics"] == []
    assert "diagnostics_note" not in summary
    assert np.isfinite(summary["correlation"])
