import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api.service import WorkflowError, run_backtest_workflow, _downsample_curve

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
    assert {"overfit_warning", "oos_significant"} <= body["out_of_sample"].keys()
    assert len(body["equity_curve"]) > 10
    # JSON-safety: no NaN should ever be serialised (they become null)
    assert "NaN" not in r.text


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
    """The Streamlit direct-import fallback must produce identical numbers."""
    req = _small_req()
    via_api = client.post("/backtest", json=req).json()
    direct = run_backtest_workflow(req)
    assert via_api["scorecard"]["ann_sharpe"] == direct["scorecard"]["ann_sharpe"]
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


def test_response_includes_signal_quality():
    body = client.post("/backtest", json=_small_req(factor="quality", periods=1000)).json()
    sq = body["signal_quality"]
    for k in ("mean_ic", "ic_tstat", "ic_ir", "ic_decay", "coverage", "rank_autocorr", "significant"):
        assert k in sq
    # the synthetic world plants a real predictive quality signal -> IC must be significant
    assert sq["significant"] is True
    assert sq["mean_ic"] > 0
