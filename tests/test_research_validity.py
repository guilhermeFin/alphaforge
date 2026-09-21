"""Data-health diagnostics: a degenerate sample must not read as a result."""
import json

import numpy as np
import pandas as pd
import pytest

from research import research_validity as rv
from research.strategy_report import build_strategy_report

DATES = pd.bdate_range("2024-01-02", periods=60)


def _flat_panel(names=("AAA", "BBB", "CCC")):
    """Every name identical on every date, as proportional-scaling fixtures produce."""
    return pd.DataFrame({name: np.full(len(DATES), 0.26) for name in names}, index=DATES)


def _varying_panel(names=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(3)
    return pd.DataFrame({name: rng.normal(0, 1, len(DATES)) for name in names}, index=DATES)


# ------------------------------------------------------------- returns


def test_return_health_counts_flat_and_moving_observations():
    health = rv.return_health(np.zeros(40))
    assert health["effective_observations"] == 40
    assert health["non_flat_returns"] == 0
    assert health["flat_returns"] == 40
    assert health["distinct_return_values"] == 1
    assert health["is_flat"] is True


def test_return_health_ignores_non_finite_values():
    health = rv.return_health([0.01, np.nan, 0.02, np.inf, -0.01])
    assert health["effective_observations"] == 3
    assert health["is_flat"] is False


def test_return_health_on_a_single_observation_is_not_called_flat():
    health = rv.return_health([0.01])
    assert health["effective_observations"] == 1
    assert health["is_flat"] is False
    assert health["return_variance"] is None


# ---------------------------------------------------------- dispersion


def test_identical_names_give_zero_cross_sectional_dispersion():
    dispersion = rv.signal_dispersion(_flat_panel())
    assert dispersion["available"] is True
    assert dispersion["mean_dispersion"] == pytest.approx(0.0)
    assert dispersion["zero_dispersion_share"] == pytest.approx(1.0)
    assert dispersion["is_degenerate"] is True


def test_varying_names_are_not_degenerate():
    dispersion = rv.signal_dispersion(_varying_panel())
    assert dispersion["is_degenerate"] is False
    assert dispersion["mean_dispersion"] > 0


def test_single_name_panel_cannot_support_a_cross_sectional_check():
    dispersion = rv.signal_dispersion(pd.DataFrame({"AAA": [1.0, 2.0]}))
    assert dispersion["available"] is False
    assert "at least two names" in dispersion["reason"]


def test_series_signal_uses_its_own_spread():
    flat = rv.signal_dispersion(pd.Series([0.5] * 10))
    assert flat["shape"] == "series" and flat["is_degenerate"] is True
    moving = rv.signal_dispersion(pd.Series(np.linspace(-1, 1, 10)))
    assert moving["is_degenerate"] is False


def test_missing_signal_is_reported_as_unavailable_not_degenerate():
    dispersion = rv.signal_dispersion(None)
    assert dispersion["available"] is False
    assert dispersion.get("is_degenerate") is None


# ----------------------------------------------------------- positions


def test_position_health_reports_active_names_and_concentration():
    book = pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [-0.5, 0.0], "CCC": [0.0, 0.0]}, index=DATES[:2])
    health = rv.position_health(book)
    assert health["average_active_names"] == pytest.approx(1.5)
    assert health["minimum_active_names"] == 1
    assert health["average_gross_exposure"] == pytest.approx(0.75)
    # Date 1 is two equal names (0.5), date 2 is one name (1.0): mean 0.75.
    assert health["concentration_herfindahl"] == pytest.approx(0.75)
    assert health["is_degenerate"] is False


def test_an_empty_book_is_degenerate_and_named():
    book = pd.DataFrame({"AAA": [0.0, 0.0], "BBB": [0.0, 0.0]}, index=DATES[:2])
    health = rv.position_health(book)
    assert health["is_degenerate"] is True
    assert health["dates_with_no_positions"] == 2
    assert health["average_active_names"] == 0.0


def test_position_health_emits_no_warning_on_an_all_zero_book(recwarn):
    """The gross-weight division must not reintroduce a 0/0 RuntimeWarning."""
    rv.position_health(pd.DataFrame({"AAA": [0.0, 0.0], "BBB": [0.0, 0.0]}))
    assert [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)] == []


# -------------------------------------------------------------- summary


def test_flat_returns_with_no_positions_blocks_a_performance_claim():
    book = pd.DataFrame(0.0, index=DATES, columns=["AAA", "BBB", "CCC"])
    summary = rv.research_validity_summary(np.zeros(len(DATES)), positions=book, signal=_flat_panel())
    assert summary["status"] == rv.STATUS_INSUFFICIENT
    assert summary["supports_performance_claim"] is False
    assert "insufficient variation for a performance claim" in summary["conclusion"]
    assert "no position was ever taken" in summary["conclusion"]


def test_flat_returns_without_positions_blame_the_signal():
    summary = rv.research_validity_summary(np.zeros(len(DATES)), signal=_flat_panel())
    assert summary["status"] == rv.STATUS_INSUFFICIENT
    assert "identical across every name" in summary["conclusion"]


def test_degenerate_signal_with_moving_returns_is_still_blocked():
    rng = np.random.default_rng(5)
    summary = rv.research_validity_summary(rng.normal(0, 0.01, len(DATES)), signal=_flat_panel())
    assert summary["status"] == rv.STATUS_INSUFFICIENT
    assert "cannot rank the universe" in summary["conclusion"]


def test_healthy_sample_supports_a_claim_without_overclaiming():
    rng = np.random.default_rng(9)
    book = pd.DataFrame(rng.normal(0, 0.3, (len(DATES), 3)), index=DATES, columns=["AAA", "BBB", "CCC"])
    summary = rv.research_validity_summary(
        rng.normal(0.001, 0.01, len(DATES)), positions=book, signal=_varying_panel())
    assert summary["status"] == rv.STATUS_OK
    assert summary["supports_performance_claim"] is True
    assert "does not by itself make them credible" in summary["conclusion"]


def test_undefined_statistics_are_collected_from_report_sections():
    rng = np.random.default_rng(4)
    sources = [
        {"statistic_diagnostics": [{"statistic": "correlation", "status": "undefined", "reason": "x"}]},
        {"statistic_diagnostics": [{"statistic": "minimum_track_record_length_95", "reason": "y"}]},
        None,
    ]
    summary = rv.research_validity_summary(
        rng.normal(0.001, 0.01, len(DATES)), diagnostic_sources=sources)
    assert summary["has_undefined_statistics"] is True
    assert summary["undefined_statistics"] == ["correlation", "minimum_track_record_length_95"]


def test_summary_never_adjusts_a_statistic():
    summary = rv.research_validity_summary(np.zeros(10))
    assert "not a performance measure" in summary["note"]
    assert "does not adjust any statistic" in summary["note"]


# ------------------------------------------------ strategy-report wiring


def test_strategy_report_includes_research_validity_and_is_json_safe():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, len(DATES)), index=DATES)
    report = build_strategy_report(returns, benchmark_returns=returns * 0.5, signal=_varying_panel())
    validity = report["research_validity"]
    assert validity["status"] == rv.STATUS_OK
    json.dumps(report)


def test_strategy_report_flags_a_flat_factor_as_uncomputable():
    """The observed gross_profitability case, reduced to its inputs."""
    flat_returns = pd.Series(np.zeros(len(DATES)), index=DATES)
    benchmark = pd.Series(np.linspace(0.0, 0.004, len(DATES)), index=DATES)
    empty_book = pd.DataFrame(0.0, index=DATES, columns=["AAA", "BBB", "CCC"])
    report = build_strategy_report(flat_returns, positions=empty_book,
                                   benchmark_returns=benchmark, signal=_flat_panel())
    validity = report["research_validity"]
    assert validity["supports_performance_claim"] is False
    assert validity["signal_dispersion"]["zero_dispersion_share"] == pytest.approx(1.0)
    assert validity["positions"]["average_active_names"] == 0.0
    assert validity["returns"]["non_flat_returns"] == 0
    # The undefined statistics from the other sections surface in one place.
    assert "correlation" in validity["undefined_statistics"]
    assert "error" not in report


# ------------------------------------- API / UI / export surfaces


def _event_study(dates, sentiments=None):
    """Run a stubbed event study over model-time eligible dates."""
    from api.service import run_filing_event_study

    index = pd.bdate_range("2020-01-02", periods=2000)
    close = pd.DataFrame({"AAA": 100.0 + np.arange(len(index)) * 0.05}, index=index)
    market = pd.Series(200.0 + 0.01 * np.arange(len(index)), index=index, name="SPY")
    features = [{
        "symbol": "AAA", "available_at": date, "document_id": f"doc-{i}",
        "sentiment": (sentiments[i] if sentiments else round(-0.8 + i * 0.1, 3)),
        "model": "ProsusAI/finbert",
    } for i, date in enumerate(dates)]
    return run_filing_event_study(
        {"features": features, "horizon": 3, "benchmark": "SPY"},
        price_loader=lambda *a: (close, market))


def _eligible_dates(n=12):
    return [str(d.date()) for d in pd.bdate_range("2021-01-05", periods=n, freq="20B")]


def test_event_study_payload_carries_research_validity():
    result = _event_study(_eligible_dates())
    validity = result["research_validity"]
    assert validity["status"] == rv.STATUS_OK
    assert validity["returns"]["effective_observations"] == 12
    assert validity["signal_dispersion"]["available"] is True


def test_event_study_flags_a_cohort_whose_tone_never_varies():
    dates = _eligible_dates()
    result = _event_study(dates, sentiments=[0.25] * len(dates))
    validity = result["research_validity"]
    assert validity["supports_performance_claim"] is False
    assert validity["signal_dispersion"]["is_degenerate"] is True


def test_event_study_validity_is_json_serializable_for_export():
    payload = json.dumps(_event_study(_eligible_dates()))
    assert "research_validity" in payload
    assert "sufficient_variation" in payload


def test_backtest_export_includes_the_data_health_block():
    """The audit trail must retain the diagnostic, not just the UI."""
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.01, len(DATES)), index=DATES)
    report = build_strategy_report(returns, benchmark_returns=returns * 0.4, signal=_varying_panel())
    payload = json.loads(json.dumps(report))
    assert payload["research_validity"]["conclusion"]
    assert payload["research_validity"]["returns"]["effective_observations"] == len(DATES)


def test_strategy_lab_shows_a_data_health_panel(monkeypatch):
    import pathlib

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    home = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "Home.py")
    app = AppTest.from_file(home, default_timeout=180)
    app.run()
    button = next(b for b in app.button if "Run backtest" in b.label)
    button.click()
    app.run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Effective observations" in labels
    assert "Non-flat returns" in labels
    assert "Signal dispersion" in labels
    assert "Avg active names" in labels
