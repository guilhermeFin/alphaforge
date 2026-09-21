from research.local_history import LocalResearchHistory


def test_history_persists_compact_filing_and_backtest_summaries(tmp_path):
    history = LocalResearchHistory(tmp_path / "history.db")
    history.record("filing_batch", "five companies", {"features": [
        {"content_kind": "earnings_release", "sentiment": 0.3},
        {"content_kind": "filing_excerpt", "sentiment": -0.1},
    ]})
    history.record("backtest", "quality", {
        "meta": {"factor": "quality"}, "scorecard": {"cagr": 0.1, "ann_sharpe": 1.2},
        "verdict": "NOT CREDIBLE", "temporal_stability": {"status": "weakened"},
    })
    rows = history.list()
    assert len(rows) == 2
    filing = next(row for row in rows if row["kind"] == "filing_batch")
    assert filing["summary"] == {"documents": 2, "earnings_releases": 1, "average_tone": 0.1}
    assert filing["payload"]["features"][0]["sentiment"] == 0.3
    backtest = next(row for row in rows if row["kind"] == "backtest")
    assert backtest["summary"]["temporal_status"] == "weakened"


def test_history_records_event_study_summary(tmp_path):
    history = LocalResearchHistory(tmp_path / "history.db")
    history.record("event_study", "15 filings", {"n_events": 15, "n_oos": 5, "mse_improvement": 0.001, "oos_pearson": 0.2, "evidence_established": False})
    row = history.list()[0]
    assert row["summary"]["n_events"] == 15
    assert row["summary"]["evidence_established"] is False


def test_history_records_portfolio_research_summary(tmp_path):
    history = LocalResearchHistory(tmp_path / "history.db")
    history.record("portfolio_research", "monthly momentum", {
        "meta": {"factor": "momentum"},
        "scorecard": {"cagr": 0.08, "ann_sharpe": 0.9},
        "portfolio": {"max_name_weight": 0.1},
        "verdict": "RESEARCH-ONLY: portfolio simulation, not a trading recommendation.",
    })
    row = history.list()[0]
    assert row["summary"] == {
        "factor": "momentum",
        "annual_return": 0.08,
        "sharpe": 0.9,
        "verdict": "RESEARCH-ONLY: portfolio simulation, not a trading recommendation.",
        "max_name_weight": 0.1,
    }
