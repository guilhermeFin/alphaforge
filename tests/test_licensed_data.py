"""Contract tests for customer-supplied, vendor-neutral licensed data bundles."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from api.service import WorkflowError, run_backtest_workflow
from research import licensed_data
from tests.research_fixtures import SYMBOLS, write_degenerate_bundle, write_research_bundle


def test_price_panel_applies_historical_universe_membership(tmp_path):
    write_research_bundle(tmp_path, ineligible_on_index=10)
    dropped = SYMBOLS[-1]
    close, volume = licensed_data.load_price_panel(list(SYMBOLS), tmp_path)
    # The ineligible name is blanked on that date; the rest of the panel stands.
    assert close.loc[close.index[10], dropped] != close.loc[close.index[10], dropped]
    assert close.loc[close.index[10], "AAA"] > 0
    assert volume.loc[volume.index[10], dropped] != volume.loc[volume.index[10], dropped]


def test_status_does_not_claim_readiness_for_missing_or_restated_inputs(tmp_path):
    missing = licensed_data.bundle_status(tmp_path)
    assert missing["configured"] is True
    assert missing["ready_for_price_research"] is False
    write_research_bundle(tmp_path, point_in_time_fundamentals=False, include_universe=False)
    status = licensed_data.bundle_status(tmp_path)
    assert status["ready_for_price_research"] is True
    assert status["ready_for_fundamental_research"] is False
    assert any("point-in-time" in issue.lower() for issue in status["issues"])


def test_price_and_fundamental_workflows_accept_complete_licensed_bundle(monkeypatch, tmp_path):
    write_research_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    base = {
        "provider": "licensed_bundle", "symbols": list(SYMBOLS),
        "periods": 520, "lookback": 126, "skip": 21, "n_trials": 5,
        "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    }
    price = run_backtest_workflow({**base, "factor": "momentum"})
    assert price["meta"]["provider"] == "licensed_bundle"
    assert price["meta"]["n_symbols"] == len(SYMBOLS)

    fundamental = run_backtest_workflow({**base, "factor": "gross_profitability"})
    assert fundamental["meta"]["provider"] == "licensed_bundle"
    assert fundamental["attribution"]["fundamentals_used"] is True
    # The realistic fixture must let the analysis actually run. This asserts the
    # sample could support a claim, never that the strategy made money.
    assert fundamental["strategy_report"]["research_validity"]["supports_performance_claim"] is True


def test_fundamental_study_refuses_bundle_without_pit_attestation(monkeypatch, tmp_path):
    write_research_bundle(tmp_path, point_in_time_fundamentals=False)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    with pytest.raises(WorkflowError, match="point-in-time fundamentals"):
        run_backtest_workflow({
            "provider": "licensed_bundle", "symbols": list(SYMBOLS),
            "factor": "gross_profitability", "periods": 520, "lookback": 126,
            "skip": 21, "n_trials": 5, "start": "2020-01-02",
        })


def test_bundle_rejects_duplicate_price_observations(tmp_path):
    write_research_bundle(tmp_path)
    path = tmp_path / "prices.csv"
    rows = pd.read_csv(path)
    pd.concat([rows, rows.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(licensed_data.LicensedDataError, match="duplicate"):
        licensed_data.load_price_panel(list(SYMBOLS), tmp_path)


def test_degenerate_bundle_is_routed_explicitly_and_trips_the_guard(monkeypatch, tmp_path):
    """The flat fixture exists only to prove the insufficient-variation guard fires.

    Every metric there is one per-symbol constant times a shared base, so the
    constant cancels in any ratio and gross profitability is identical across
    names. Nothing can be ranked, so nothing is ever held.
    """
    write_degenerate_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    result = run_backtest_workflow({
        "provider": "licensed_bundle", "symbols": ["AAA", "BBB", "CCC"],
        "factor": "gross_profitability", "periods": 520, "lookback": 126,
        "skip": 21, "n_trials": 5, "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    })
    validity = result["strategy_report"]["research_validity"]
    assert validity["supports_performance_claim"] is False
    assert validity["signal_dispersion"]["zero_dispersion_share"] == pytest.approx(1.0)
    assert validity["positions"]["average_active_names"] == 0.0
    assert "insufficient variation for a performance claim" in validity["conclusion"]
