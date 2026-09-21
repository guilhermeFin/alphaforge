import pytest

from api.service import run_backtest_workflow
from research.protocol_runner import ProtocolRunner
from research.research_protocol import ProtocolError, ResearchProtocolStore


BOUNDARIES = {
    "research_end": "2017-12-29",
    "validation_end": "2019-12-31",
    "final_holdout_end": "2020-10-19",
}


def _request():
    return {
        "provider": "synthetic", "factor": "quality", "start": "2015-01-02",
        "periods": 120, "seed": 42, "lookback": 252, "skip": 21,
        "cost_bps": 5.0, "gross": 1.0, "n_trials": 4,
    }


def _fake_engine(request):
    return {
        "manifest": {"research_fingerprint": f"fp-{request['evaluation_start']}"},
        "meta": {"start_date": request["evaluation_start"], "end_date": request["end"]},
        "data_audit": {"status": "supported"},
        "evidence_card": {"status": "research_supported"},
        "scorecard": {"ann_sharpe": 0.5},
        "verdict": "NOT CREDIBLE: test fixture",
    }


def test_protocol_runner_builds_disjoint_windows_and_consumes_final_holdout(tmp_path):
    store = ResearchProtocolStore(tmp_path / "protocol.db")
    study = store.create("Quality survives fixed costs", BOUNDARIES)
    runner = ProtocolRunner(store)

    validation = runner.run(study["id"], "validation", _request(), _fake_engine)
    assert validation["protocol"]["window"] == {
        "source_start": "2015-01-02",
        "evaluation_start": "2017-12-30",
        "evaluation_end": "2019-12-31",
        "evaluation_business_days": 522,
    }
    assert validation["meta"]["protocol_stage"] == "validation"

    validation_request, _ = runner.prepare(study["id"], "validation", _request())
    final_request, _ = runner.prepare(study["id"], "final_holdout", _request())
    assert validation_request["periods"] != final_request["periods"]
    assert validation_request["_trial_identity"] == final_request["_trial_identity"]

    final = runner.run(study["id"], "final_holdout", _request(), _fake_engine)
    assert final["protocol"]["final_holdout_consumed"] is True
    with pytest.raises(ProtocolError, match="already been consumed"):
        runner.run(study["id"], "final_holdout", _request(), _fake_engine)


def test_protocol_stage_reports_only_its_evaluation_dates():
    result = run_backtest_workflow({
        **_request(), "periods": 1512, "end": "2020-10-19", "evaluation_start": "2020-01-01",
    })
    assert result["meta"]["source_start_date"] == "2015-01-02"
    assert result["meta"]["start_date"] >= "2020-01-01"
    assert result["meta"]["end_date"] == "2020-10-19"
    assert result["meta"]["evaluation_window_applied"] is True
    assert result["scorecard"]["n_periods"] == result["meta"]["n_days"]
    assert result["manifest"]["data_snapshot"]["start_date"] == result["meta"]["start_date"]
    assert "fixed validation or final-holdout stage" in result["pbo"]["error"]


def test_protocol_runner_rejects_requests_that_start_after_research_phase(tmp_path):
    store = ResearchProtocolStore(tmp_path / "protocol.db")
    study = store.create("Test", BOUNDARIES)
    with pytest.raises(ProtocolError, match="on or before the protocol research end"):
        ProtocolRunner(store).prepare(study["id"], "validation", {**_request(), "start": "2018-01-02"})


def test_protocol_runner_locks_the_first_protected_specification(tmp_path):
    store = ResearchProtocolStore(tmp_path / "protocol.db")
    study = store.create("Test", BOUNDARIES)
    runner = ProtocolRunner(store)
    runner.run(study["id"], "exploration", _request(), _fake_engine)
    with pytest.raises(ProtocolError, match="locked to its first protected strategy specification"):
        runner.prepare(study["id"], "validation", {**_request(), "cost_bps": 15.0})


def test_protocol_runner_rejects_a_thin_actual_provider_window(tmp_path):
    store = ResearchProtocolStore(tmp_path / "protocol.db")
    study = store.create("Test", BOUNDARIES)

    def thin_engine(request):
        return {**_fake_engine(request), "meta": {"n_days": 10}}

    with pytest.raises(ProtocolError, match="only 10 usable sessions"):
        ProtocolRunner(store).run(study["id"], "validation", _request(), thin_engine)
    assert store.summary(study["id"])["runs"] == []
