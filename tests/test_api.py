import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.service import WorkflowError, run_backtest_workflow, _downsample_curve

# NOTE: this module-level client keeps ONE cookie jar, so every test using it shares
# ONE server-side trial ledger (distinct_count accumulates across calls in order).
# That's fine for tests pinning ledger-INDEPENDENT values (ann_sharpe, verdict, key
# presence). Any test pinning deflated_sr / meta['n_trials'] MUST use a fresh
# TestClient(app) so it starts from a clean ledger.
client = TestClient(app)


def _small_req(**over):
    base = {"provider": "synthetic", "factor": "momentum", "periods": 400,
            "lookback": 126, "skip": 21, "n_trials": 10, "seed": 1}
    base.update(over)
    return base


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "not investment advice" in r.json()["disclaimer"]


def test_data_connections_endpoint_hides_credentials_and_raw_paths(monkeypatch):
    monkeypatch.setenv("ALPHAFORGE_LICENSED_DATA_PATH", "")
    body = TestClient(app).get("/data-connections").json()
    assert body["licensed_bundle"]["configured"] is False
    rendered = str(body)
    assert "C:\\" not in rendered and "/Users/" not in rendered
    assert "SIMFIN_API_KEY" not in rendered and "NASDAQ_DATA_LINK_API_KEY" not in rendered


def test_protocol_api_executes_a_bounded_stage_and_prevents_url_mismatch(tmp_path, monkeypatch):
    from research.research_protocol import ResearchProtocolStore

    monkeypatch.setattr("api.main._PROTOCOL_STORE", ResearchProtocolStore(tmp_path / "protocol.db"))

    def fake_run(request, ledger=None):
        return {
            "manifest": {"research_fingerprint": "protocol-fingerprint"},
            "meta": {"start_date": request["evaluation_start"], "end_date": request["end"]},
            "data_audit": {"status": "supported"},
            "evidence_card": {"status": "research_supported"},
            "scorecard": {"ann_sharpe": 0.4},
            "verdict": "NOT CREDIBLE: fixture",
        }

    monkeypatch.setattr("api.main.run_backtest_workflow", fake_run)
    api = TestClient(app)
    created = api.post("/protocols", json={
        "hypothesis": "Quality survives costs in a fixed study.",
        "research_end": "2017-12-29", "validation_end": "2019-12-31", "final_holdout_end": "2020-10-19",
    })
    assert created.status_code == 200, created.text
    study_id = created.json()["id"]
    payload = {**_small_req(factor="quality"), "study_id": study_id, "stage": "validation"}
    response = api.post(f"/protocols/{study_id}/run", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["protocol"]["stage"] == "validation"
    mismatch = api.post(f"/protocols/{study_id}/run", json={**payload, "study_id": "a" * 32})
    assert mismatch.status_code == 400


def test_fixed_benchmark_api_has_no_caller_selected_factor_surface(monkeypatch):
    monkeypatch.setattr("api.main.run_benchmark_suite", lambda req, ledger=None: {
        "kind": "fixed_benchmark_suite", "benchmarks": [{"factor": "momentum"}], "manifest": {"research_fingerprint": "fixed"},
    })
    response = TestClient(app).post("/benchmark-suite", json=_small_req(factor="blend"))
    assert response.status_code == 200
    assert response.json()["kind"] == "fixed_benchmark_suite"


def test_event_study_api_validates_bounded_document_payload(monkeypatch):
    monkeypatch.setattr("api.main.run_filing_event_study", lambda req: {"status": "insufficient_data", "n_events": len(req["features"])})
    payload = {"features": [{"symbol": "AAPL", "available_at": "2024-05-01", "document_id": "doc-1", "sentiment": 0.2}]}
    response = TestClient(app).post("/filing-event-study", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_data"
    assert TestClient(app).post("/filing-event-study", json={**payload, "unknown": True}).status_code == 422


def test_portfolio_research_endpoint_runs_constrained_execution_study():
    payload = _small_req(periods=400, n_trials=10)
    payload.update({"rebalance_frequency": "monthly", "max_name_weight": 0.10,
                    "target_annual_vol": 0.15, "capital": 1_000_000})
    response = TestClient(app).post("/portfolio-research", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert {"portfolio", "execution", "robustness", "manifest", "paper_plan", "data_readiness"} <= body.keys()
    assert body["portfolio"]["max_name_weight"] <= 0.10 + 1e-12
    assert body["manifest"]["research_fingerprint"]
    assert "NaN" not in response.text


def test_backtest_endpoint_full_contract():
    r = client.post("/backtest", json=_small_req())
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("meta", "verdict", "scorecard", "out_of_sample",
                "walk_forward", "fat_tails", "equity_curve", "drawdown_curve"):
        assert key in body, key
    assert body["meta"]["disclaimer"]
    assert body["scorecard"]["n_periods"] == 400
    assert {"deflated_sr", "psr_vs_0", "ann_sharpe", "max_drawdown"} <= body["scorecard"].keys()


def test_backtest_response_includes_strategy_report_risk_and_rigor():
    response = client.post("/backtest", json=_small_req())
    assert response.status_code == 200, response.text
    body = response.json()
    report = body["strategy_report"]
    assert {"volatility_downside_risk", "benchmark_relative", "tail_risk_distribution", "statistical_rigor", "chart_specs"} <= report.keys()
    assert {"sortino_ratio", "ulcer_index", "omega_ratio", "k_ratio"} <= report["volatility_downside_risk"].keys()
    rigor = report["statistical_rigor"]
    assert {"probabilistic_sharpe_ratio", "hac_sharpe_tstat", "minimum_track_record_length_95"} <= rigor.keys()
    assert rigor["cpcv_pbo"]["purge_periods"] == 1
    assert report["benchmark_relative"]["n_observations"] > 10
    assert report["tail_risk_distribution"]["historical_var_99"] is not None
    assert {"overfit_warning", "oos_significant"} <= body["out_of_sample"].keys()
    assert len(body["equity_curve"]) > 10
    # JSON-safety: no NaN should ever be serialised (they become null)
    assert "NaN" not in response.text


def test_backtest_deterministic_same_seed():
    a = client.post("/backtest", json=_small_req()).json()
    b = client.post("/backtest", json=_small_req()).json()
    assert a["scorecard"]["ann_sharpe"] == b["scorecard"]["ann_sharpe"]
    assert a["equity_curve"][-1] == b["equity_curve"][-1]


def test_validation_errors_are_400():
    assert client.post("/backtest", json=_small_req(provider="bloomberg")).status_code in (400, 422)
    assert client.post("/backtest", json=_small_req(factor="astrology")).status_code in (400, 422)
    assert client.post("/backtest", json=_small_req(lookback=126, skip=200)).status_code in (400, 422)
    # yfinance without symbols must be rejected before any network call
    assert client.post("/backtest", json=_small_req(provider="yfinance", symbols=[])).status_code in (400, 422)
    # lookback longer than the data window
    assert client.post("/backtest", json=_small_req(periods=150, lookback=140)).status_code in (400, 422)


def test_service_matches_api_numbers():
    """The Streamlit direct-import fallback must produce identical numbers — including
    the ledger-touched ones. Use a FRESH client so the API call starts a clean ledger
    (distinct_count=1); with declared n_trials=10 the enforced floor is 10 on BOTH the
    API and the stateless direct path, so even deflated_sr / meta n_trials must match."""
    req = _small_req()
    c = TestClient(app)
    via_api = c.post("/backtest", json=req).json()
    direct = run_backtest_workflow(req)
    assert via_api["scorecard"]["ann_sharpe"] == direct["scorecard"]["ann_sharpe"]
    assert via_api["scorecard"]["deflated_sr"] == direct["scorecard"]["deflated_sr"]
    assert via_api["meta"]["n_trials"] == direct["meta"]["n_trials"] == 10
    assert via_api["meta"]["declared_n_trials"] == 10
    assert via_api["verdict"] == direct["verdict"]


def test_service_rejects_garbage():
    for bad in (
        {"provider": "synthetic", "periods": 50},                # too short
        {"provider": "synthetic", "n_trials": 0},                # zero trials
        {"provider": "yfinance", "symbols": ["DROP TABLE;"]},    # junk ticker
        {"provider": "synthetic", "cost_bps": -1},               # negative costs
        {"provider": "synthetic", "seed": -1},                   # negative seed -> rng error
        {"provider": "synthetic", "start": "not-a-date"},        # unparseable date
        {"provider": "yfinance", "symbols": ["AAPL"]},           # single ticker (need >= 2)
    ):
        try:
            run_backtest_workflow(bad)
        except WorkflowError:
            continue
        raise AssertionError(f"accepted bad request: {bad}")


def test_downsample_curve_preserves_extrema():
    """The drawdown chart must never look shallower than the max_drawdown beside
    it: downsampling has to keep the true trough and peak."""
    idx = pd.bdate_range("2020-01-01", periods=2000)
    vals = np.zeros(2000)
    vals[1234] = -0.5   # the true trough, off the stride grid
    vals[777] = 0.9     # the true peak
    curve = _downsample_curve(pd.Series(vals, index=idx), max_points=200)
    ys = [p["value"] for p in curve]
    assert min(ys) == -0.5
    assert max(ys) == 0.9
    assert len(ys) <= 210  # still downsampled, not the full 2000


def test_verdict_is_out_of_sample_based():
    body = client.post("/backtest", json=_small_req()).json()
    assert "walk-forward" in body["verdict_basis"]
    assert body["verdict"].startswith(("CREDIBLE", "NOT CREDIBLE"))
    # synthetic momentum is noise -> must not be sold as credible
    assert body["verdict"].startswith("NOT CREDIBLE")


def test_meta_reports_effective_lookback():
    body = client.post("/backtest", json=_small_req(factor="reversal", lookback=252)).json()
    assert body["meta"]["effective_lookback"] == 63   # reversal caps the lookback
    body2 = client.post("/backtest", json=_small_req(factor="momentum", lookback=126)).json()
    assert body2["meta"]["effective_lookback"] == 126  # momentum uses it as-is


def test_fundamental_factors_run_on_synthetic():
    for f in ("value", "quality", "value_quality"):
        body = client.post("/backtest", json=_small_req(factor=f, periods=800)).json()
        assert "scorecard" in body
        assert body["meta"]["effective_lookback"] is None          # lookback n/a for fundamentals
        assert body["verdict"].startswith(("CREDIBLE", "NOT CREDIBLE"))


def test_quality_factor_is_certified_on_planted_synthetic():
    # the synthetic world plants a real point-in-time quality signal -> the engine
    # must CERTIFY it (proves the tool also passes genuine signals, not just rejects).
    body = client.post("/backtest", json=_small_req(factor="quality", periods=1512, n_trials=20)).json()
    assert body["verdict"].startswith("CREDIBLE"), body["verdict"]


def test_fundamental_factor_rejects_yfinance():
    r = client.post("/backtest", json=_small_req(factor="quality", provider="yfinance",
                                                 symbols=["AAPL", "MSFT"]))
    assert r.status_code in (400, 422)


def test_response_includes_pbo_and_cvar():
    body = client.post("/backtest", json=_small_req(factor="momentum", periods=800)).json()
    assert "pbo" in body and "trial_audit" in body
    # CVaR/ES flow through summarize() into the scorecard
    assert {"var_95", "cvar_95", "cvar_99"} <= body["scorecard"].keys()
    # technical factor -> PBO is computed (not the fundamental n/a path)
    pbo = body["pbo"]
    assert "error" not in pbo, pbo
    assert 0.0 <= pbo["pbo"] <= 1.0
    assert "histogram" in pbo
    assert "NaN" not in client.post("/backtest", json=_small_req(periods=800)).text


def test_response_includes_attribution():
    # fundamental factor on synthetic -> full FF5 attribution (RMW present)
    body = client.post("/backtest", json=_small_req(factor="gross_profitability",
                                                    periods=1512, n_trials=10)).json()
    attr = body["attribution"]
    assert "error" not in attr, attr
    assert attr["model"] == "ff5"
    assert "RMW" in attr.get("betas", {})
    assert 0.0 <= (attr.get("r_squared") or 0.0) <= 1.0
    # price factor -> price-only fallback (carhart4: MKT + UMD, fundamentals absent)
    a2 = client.post("/backtest", json=_small_req(factor="momentum", periods=1512, n_trials=10)).json()["attribution"]
    assert "error" not in a2
    assert a2["model"] == "carhart4"
    assert a2.get("fundamentals_used") is False


def test_ml_compare_rejects_yfinance():
    # provider guard fires BEFORE any ML import, so this needs no ML deps installed
    r = client.post("/ml-compare", json=_small_req(provider="yfinance", symbols=["AAPL", "MSFT"]))
    assert r.status_code in (400, 422)


def test_model_comparison_runs_reduced():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    from api.service import run_model_comparison
    out = run_model_comparison(_small_req(periods=600), model_names=["elastic_net", "lightgbm"],
                               n_splits=4)
    assert out["leaderboard"] and {"model", "oos_rank_ic"} <= set(out["leaderboard"][0].keys())
    assert "complexity_beats_linear" in out and out["verdict"]
    assert out["n_splits"] == 4


def test_factorlib_factors_are_selectable_and_run():
    from api.service import FACTOR_CATALOG, FACTORS
    assert len(FACTORS) == len(FACTOR_CATALOG)  # catalog and validator agree
    # a quality-family factor_lib factor runs and, on the quality-planted synthetic
    # world, earns CREDIBLE (proves the new wiring passes genuine signals)
    body = client.post("/backtest", json=_small_req(factor="gross_profitability",
                                                    periods=1512, n_trials=10)).json()
    assert body["verdict"].startswith("CREDIBLE"), body["verdict"]
    assert body["meta"]["effective_lookback"] is None      # fundamental -> no price lookback
    assert "error" in body["pbo"]                           # PBO grid is price-only -> n/a
    # value + "lower is better" sign factors all run without error (status 200)
    for f in ("earnings_yield", "book_to_price", "low_leverage", "earnings_quality",
              "conservative_investment", "low_issuance"):
        r = client.post("/backtest", json=_small_req(factor=f, periods=1200, n_trials=5))
        assert r.status_code == 200, (f, r.text)
        assert r.json()["verdict"].startswith(("CREDIBLE", "NOT CREDIBLE"))


def test_pbo_grid_distinct_for_capped_factors():
    # reversal caps lookback at 63 and lowvol at 252 inside _score; the grid is built
    # in EFFECTIVE space so it must still yield >= 2 DISTINCT configs (not identical
    # clones that would manufacture a bogus 'overfit' verdict).
    for f in ("reversal", "lowvol"):
        body = client.post("/backtest", json=_small_req(factor=f, periods=900, lookback=252)).json()
        pbo = body["pbo"]
        assert "error" not in pbo, (f, pbo)
        assert 0.0 <= pbo["pbo"] <= 1.0


def test_pbo_is_na_for_fundamental_factors():
    body = client.post("/backtest", json=_small_req(factor="quality", periods=900)).json()
    assert "error" in body["pbo"]  # grid not wired for fundamentals yet (honest n/a)


def test_walk_forward_reports_purge():
    # long enough that purging doesn't eat a fold -> real walk-forward with audit
    body = client.post("/backtest", json=_small_req(factor="momentum", periods=1512,
                                                    lookback=126, skip=21)).json()
    wf = body["walk_forward"]
    if "error" not in wf:
        assert wf["purge_bars"] == 147  # lookback(126) + skip(21)
        assert wf["embargo_bars"] == 1
        assert wf["total_purged_bars"] == sum(wf["purged_bars_per_fold"]) > 0


def test_trial_ledger_floor_raises_haircut():
    # fresh client => isolated cookie => isolated server-side ledger
    c = TestClient(app)
    for lb in (80, 100, 120):
        c.post("/backtest", json=_small_req(factor="momentum", lookback=lb, n_trials=1))
    ta = c.post("/backtest", json=_small_req(factor="momentum", lookback=140, n_trials=1)).json()["trial_audit"]
    assert ta["declared_n_trials"] == 1
    assert ta["distinct_count"] >= 4
    assert ta["effective_n_trials"] == ta["distinct_count"]   # floor enforced
    assert ta["haircut_was_raised"] is True


def test_rerun_same_config_does_not_inflate_distinct():
    c = TestClient(app)
    a = c.post("/backtest", json=_small_req(lookback=111, n_trials=1)).json()
    b = c.post("/backtest", json=_small_req(lookback=111, n_trials=1)).json()
    assert a["trial_audit"]["distinct_count"] == 1
    assert b["trial_audit"]["distinct_count"] == 1   # identical config is not a new trial
    assert b["trial_audit"]["total_runs"] == 2


def test_session_reset_zeros_ledger():
    c = TestClient(app)
    c.post("/backtest", json=_small_req(lookback=90, n_trials=1))
    c.post("/backtest", json=_small_req(lookback=110, n_trials=1))
    entry = c.post("/session/reset", params={"reason": "test"}).json()
    assert entry["discarded_distinct"] == 2
    after = c.post("/backtest", json=_small_req(lookback=130, n_trials=1)).json()
    assert after["trial_audit"]["distinct_count"] == 1   # counting restarted


def test_response_includes_signal_quality():
    body = client.post("/backtest", json=_small_req(factor="quality", periods=1000)).json()
    sq = body["signal_quality"]
    for k in ("mean_ic", "ic_tstat", "ic_ir", "ic_decay", "coverage", "rank_autocorr", "significant"):
        assert k in sq
    # the synthetic world plants a real predictive quality signal -> IC must be significant
    assert sq["significant"] is True
    temporal = body["temporal_stability"]
    assert temporal["status"] in {"stable", "weakened", "unstable", "insufficient_evidence"}
    assert temporal["cohorts"]
    assert sq["mean_ic"] > 0
