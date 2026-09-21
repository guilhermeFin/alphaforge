import numpy as np
import pandas as pd

from api.service import WorkflowError, run_filing_event_study
from research.event_study import FilingEvent, build_event_returns, evaluate_sentiment


def _prices():
    index = pd.bdate_range("2024-01-02", periods=100)
    close = pd.DataFrame({"AAA": 100.0 + np.arange(len(index)), "BBB": 120.0 + 0.8 * np.arange(len(index))}, index=index)
    market = pd.Series(200.0 + 0.3 * np.arange(len(index)), index=index, name="SPY")
    return close, market


def test_event_return_starts_after_date_only_filing():
    close, market = _prices()
    event = FilingEvent("AAA", pd.Timestamp("2024-01-02"), 0.5, "a-1")
    observed, warnings = build_event_returns([event], close, market, horizon=3)
    assert not warnings
    assert observed.loc[0, "entry_date"] == pd.Timestamp("2024-01-03")
    assert observed.loc[0, "exit_date"] == pd.Timestamp("2024-01-08")


def test_oos_evaluation_purges_boundary_labels_and_keeps_baseline_separate():
    index = pd.bdate_range("2024-01-02", periods=100)
    events = pd.DataFrame({
        "entry_date": index[::5][:16],
        "exit_date": index[::5][:16] + pd.offsets.BDay(2),
        "sentiment": np.linspace(-1, 1, 16),
        "abnormal_return": np.linspace(-0.04, 0.04, 16),
    })
    result = evaluate_sentiment(events, horizon=2)
    assert result["status"] == "complete"
    assert result["n_train"] >= 5 and result["n_oos"] >= 4
    assert "baseline_mse" in result and "text_mse" in result
    assert "not a trading" not in result["verdict"].lower()


def test_service_uses_bounded_feature_payload_and_fake_prices():
    close, market = _prices()

    def loader(symbols, start, end, benchmark):
        assert symbols == ["AAA", "BBB"] and benchmark == "SPY"
        return close, market

    features = []
    for i, date in enumerate(pd.bdate_range("2024-01-02", periods=15, freq="5B")):
        # 2024 filings post-date FinBERT's availability, so the whole batch is
        # model-time eligible and the headline cohort is the full sample.
        features.append({"symbol": "AAA" if i % 2 else "BBB", "available_at": str(date.date()),
                         "document_id": f"doc-{i}", "sentiment": -0.8 + i * 0.1, "model": "ProsusAI/finbert"})
    result = run_filing_event_study({"features": features, "horizon": 3, "benchmark": "SPY"}, price_loader=loader)
    assert result["status"] == "complete"
    assert result["requested_documents"] == 15
    assert result["benchmark"] == "SPY"
    assert "next trading session" in result["note"]
    assert result["cohort_coverage"]["excluded_events"] == 0


def test_service_rejects_unbounded_or_invalid_event_input():
    with np.testing.assert_raises_regex(WorkflowError, "50"):
        run_filing_event_study({"features": [{"symbol": "AAA", "available_at": "2024-01-01", "document_id": "x", "sentiment": 0.0}] * 51})
    with np.testing.assert_raises_regex(WorkflowError, "horizon"):
        run_filing_event_study({"features": [{"symbol": "AAA", "available_at": "2024-01-01", "document_id": "x", "sentiment": 0.0}], "horizon": 40})
