from api import service
from research import benchmark_suite


def test_yahoo_suite_explicitly_skips_fundamental_references():
    runs, skipped = benchmark_suite.requests_for_suite({"provider": "yfinance", "n_trials": 1})
    assert [definition["factor"] for definition, _request in runs] == ["momentum", "lowvol"]
    assert [row["factor"] for row in skipped] == ["value", "quality"]


def test_suite_is_fixed_and_preserves_component_audit(monkeypatch):
    calls = []

    def fake_run(request, ledger=None):
        calls.append(request)
        factor = request["factor"]
        return {
            "verdict": f"{factor} result",
            "scorecard": {"cagr": 0.1, "ann_sharpe": 0.5, "deflated_sr": 0.8, "max_drawdown": -0.2},
            "evidence_card": {"status": "exploratory"},
            "data_audit": {"status": "exploratory"},
            "manifest": {"research_fingerprint": f"fingerprint-{factor}"},
        }

    monkeypatch.setattr(service, "run_backtest_workflow", fake_run)
    result = service.run_benchmark_suite({"provider": "synthetic", "factor": "blend", "n_trials": 1})
    assert [row["factor"] for row in result["benchmarks"]] == ["value", "quality", "momentum", "lowvol"]
    assert [call["factor"] for call in calls] == ["value", "quality", "momentum", "lowvol"]
    assert all(call["n_trials"] == 4 for call in calls)
    assert result["manifest"]["research_fingerprint"]
