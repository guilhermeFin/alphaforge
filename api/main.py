"""AlphaForge API — a thin FastAPI layer over the research engine.

Run from the repo root:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health     liveness + engine version
    POST /backtest   the one end-to-end workflow: universe -> factor ->
                     honest backtest -> scorecard + overfit/fat-tail verdicts

The API adds no statistics of its own; everything comes from api/service.py,
which both this app and the Streamlit UI share.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root

import logging  # noqa: E402
from uuid import uuid4  # noqa: E402

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from api.service import (  # noqa: E402
    DISCLAIMER, FACTORS, PROVIDERS, WorkflowError, run_backtest_workflow,
    run_filing_event_study, run_model_comparison, run_public_data_pilot, run_real_document_batch,
    run_portfolio_research, run_benchmark_suite, data_connections_status,
    run_microstructure_lab,
)
from research.protocol_runner import ProtocolRunner  # noqa: E402
from research.research_protocol import ProtocolError, ResearchProtocolStore  # noqa: E402
from research.trial_ledger import TrialLedger  # noqa: E402
from api.security import install_security, secure_cookies  # noqa: E402
from api.accounts_api import (  # noqa: E402
    current_principal, get_service, install_accounts,
)

_log = logging.getLogger("alphaforge.api")

app = FastAPI(
    title="AlphaForge API",
    version="0.0.1",
    description="The research OS for emerging quant managers — "
                "the backtest that won't let you lie to yourself. " + DISCLAIMER,
)
install_security(app)  # CORS lockdown + security headers + body cap + optional rate limit
install_accounts(app)  # optional accounts/auth surface (off unless ALPHAFORGE_ACCOUNTS set)


class BacktestRequest(BaseModel):
    # reject unknown fields outright (mass-assignment / unknown-field injection guard)
    model_config = ConfigDict(extra="forbid")

    provider: str = Field("synthetic", description=f"one of {PROVIDERS}")
    symbols: list[str] = Field(default_factory=list, max_length=64,
                               description="tickers (required for yfinance; optional for synthetic)")
    factor: str = Field("momentum", description=f"one of {FACTORS}")
    lookback: int = Field(252, ge=5, le=756)
    skip: int = Field(21, ge=0)
    cost_bps: float = Field(5.0, ge=0.0, le=200.0)
    gross: float = Field(1.0, ge=0.1, le=3.0)
    n_trials: int = Field(1, ge=1, le=100_000,
                          description="how many strategy variants you have tried — "
                                      "be honest; it drives the Deflated Sharpe haircut")
    periods: int = Field(1512, ge=120, le=6000, description="synthetic only")
    seed: int = Field(42, description="synthetic only")
    start: str = Field("2015-01-02")


class ProtocolCreateRequest(BaseModel):
    """A hypothesis and fixed chronological boundaries for a protected study."""

    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(..., min_length=8, max_length=1_000)
    research_end: str = Field(..., min_length=10, max_length=10)
    validation_end: str = Field(..., min_length=10, max_length=10)
    final_holdout_end: str = Field(..., min_length=10, max_length=10)


class ProtocolRunRequest(BacktestRequest):
    """A normal strategy request plus the named protected stage to execute."""

    study_id: str = Field(..., min_length=16, max_length=64)
    stage: str = Field(..., pattern="^(exploration|validation|final_holdout)$")


class MLCompareRequest(BacktestRequest):
    """A backtest request plus the optional knobs that only the ML ladder uses."""
    ml_factors: list[str] | None = Field(
        default=None, max_length=12,
        description="factor_lib features for the model panel (default: a built-in set)")
    ml_models: list[str] | None = Field(
        default=None, max_length=16,
        description="which models in the ladder to run (default: the full ladder)")
    horizon: int = Field(21, ge=1, le=63, description="forward-return label horizon (trading days)")


class PortfolioResearchRequest(BacktestRequest):
    """Free, historical portfolio controls.  No field can place a broker order."""
    rebalance_frequency: str = Field("monthly")
    allocation_method: str = Field(
        "score_weighted",
        description="score_weighted, equal_weight, inverse_volatility, or hierarchical_risk_parity",
    )
    max_name_weight: float = Field(0.10, ge=0.01, le=3.0)
    max_turnover: float | None = Field(None, ge=0.0, le=3.0)
    target_annual_vol: float | None = Field(0.15, ge=0.0, le=1.0)
    spread_bps: float = Field(2.0, ge=0.0, le=200.0)
    impact_bps: float = Field(12.0, ge=0.0, le=500.0)
    short_borrow_bps: float = Field(50.0, ge=0.0, le=5_000.0)
    capital: float = Field(1_000_000.0, gt=0.0, le=1_000_000_000.0)
    max_participation: float = Field(0.05, gt=0.0, le=1.0)
    adv_lookback: int = Field(20, ge=5, le=126)


class PilotTextDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=10)
    available_at: str = Field(..., min_length=10, max_length=64)
    source: str = Field(..., min_length=1, max_length=80)
    document_id: str = Field(..., min_length=1, max_length=160)
    text: str = Field(..., min_length=1, max_length=12_000)


class PublicDataPilotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "JPM", "XOM"], max_length=5)
    macro_series: list[str] = Field(default_factory=lambda: ["CPIAUCSL", "UNRATE", "DGS10"], max_length=5)
    macro_start: str = Field("2015-01-01", min_length=10, max_length=10)
    as_of: str = Field(..., min_length=10, max_length=10)
    text_document: PilotTextDocument | None = None


class TimeIntegrityPolicyRequest(BaseModel):
    """Optional overrides for the model-time integrity screen.

    Omitting this object keeps the conservative defaults: observations scored by
    a model that was not yet available, and observations with no model metadata,
    stay out of the primary evidence cohort.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    exclude_historically_unavailable: bool = True
    exclude_missing_metadata: bool = True
    exclude_training_overlap: bool = False


class RealDocumentBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "JPM", "XOM"], max_length=25)
    forms: list[str] = Field(default_factory=lambda: ["8-K"], max_length=3)
    as_of: str = Field(..., min_length=10, max_length=10)
    per_symbol: int = Field(1, ge=1, le=2)
    time_integrity: TimeIntegrityPolicyRequest | None = None


class FilingEventFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    symbol: str = Field(..., min_length=1, max_length=10)
    available_at: str = Field(..., min_length=10, max_length=64)
    document_id: str = Field(..., min_length=1, max_length=160)
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    # Optional so saved results from before the model-time gate still post. A
    # missing model is reported as missing metadata, never assumed to be valid.
    model: str | None = Field(None, max_length=160)


class FilingEventStudyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[FilingEventFeature] = Field(..., min_length=1, max_length=50)
    horizon: int = Field(5, ge=1, le=21)
    benchmark: str = Field("SPY", min_length=1, max_length=10)
    time_integrity: TimeIntegrityPolicyRequest | None = None


class MicrostructureTradeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(..., min_length=10, max_length=64)
    price: float = Field(..., gt=0.0)
    quantity: float = Field(..., gt=0.0)
    is_buyer_maker: bool


class MicrostructureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field("synthetic", pattern="^(synthetic|binance_events)$")
    events: list[MicrostructureTradeEvent] | None = Field(None, max_length=20_000)
    seed: int = Field(42, ge=0, le=2_147_483_647)
    window_events: int = Field(32, ge=8, le=500)
    horizon_events: int = Field(30, ge=5, le=500)
    maker_fee_bps: float = Field(1.0, ge=0.0, le=100.0)
    latency_events: int = Field(2, ge=0, le=100)
    max_inventory: int = Field(8, ge=1, le=100)


# Per-workspace trial ledgers. A single global ledger would conflate users (one
# person's sweep would haircut a stranger's first run), so we scope each ledger to a
# workspace: an AUTHENTICATED caller's ledger is keyed by their workspace id (shared
# across their keys/sessions); an anonymous caller's is keyed by a session cookie.
# Bounded so an unauthenticated caller minting endless cookies can't grow it without
# limit (memory-DoS guard); evicts the oldest when full.
MAX_SESSIONS = 10_000
_LEDGERS: dict[str, TrialLedger] = {}
_PROTOCOL_STORE = ResearchProtocolStore()


def _ledger_for_key(key: str) -> TrialLedger:
    if key not in _LEDGERS:
        if len(_LEDGERS) >= MAX_SESSIONS:
            _LEDGERS.pop(next(iter(_LEDGERS)))  # evict oldest (insertion-ordered)
        _LEDGERS[key] = TrialLedger()
    return _LEDGERS[key]


def _get_ledger(request: Request, response: Response, principal=None) -> TrialLedger:
    if principal is not None:
        # authenticated: one honest counter per workspace, identity carried by the key
        return _ledger_for_key(f"ws:{principal.workspace_id}")
    sid = request.cookies.get("af_session")
    if not sid or sid not in _LEDGERS:
        sid = uuid4().hex
        ledger = _ledger_for_key(sid)
        response.set_cookie("af_session", sid, max_age=8 * 3600,
                            httponly=True, samesite="lax", secure=secure_cookies())
        return ledger
    return _LEDGERS[sid]


def _maybe_persist_run(request: Request, kind: str, req_dict: dict, result: dict) -> None:
    """Best-effort: if the caller is authenticated, persist the run to their workspace
    history (encrypted) + audit it. A persistence failure NEVER fails the actual
    backtest response — but it is logged, never silently swallowed as success."""
    principal = current_principal(request)
    svc = get_service()
    if principal is None or svc is None:
        return
    try:
        verdict = result.get("verdict") if isinstance(result, dict) else None
        if kind == "backtest":
            summary = {"meta": result.get("meta"), "scorecard": result.get("scorecard"),
                       "verdict": verdict}
        elif kind == "portfolio_research":
            summary = {
                "meta": result.get("meta"),
                "scorecard": result.get("scorecard"),
                "portfolio": result.get("portfolio"),
                "manifest": result.get("manifest"),
                "verdict": verdict,
            }
        else:
            summary = {"verdict": verdict, "leaderboard": result.get("leaderboard"),
                       "complexity_beats_linear": result.get("complexity_beats_linear"),
                       "feature_importance": result.get("feature_importance")}
        svc.record_run(principal, kind, req_dict, verdict, summary,
                       request_id=request.headers.get("x-request-id"),
                       ip=request.client.host if request.client else None)
    except Exception as e:  # noqa: BLE001
        _log.warning("run persistence failed (%s: %s) — backtest result still returned",
                     type(e).__name__, e)


@app.get("/")
def root() -> dict:
    """Friendly root so the bare URL doesn't look like a 404. The UI lives on the
    Streamlit port (8501); this is the JSON API."""
    return {"service": "AlphaForge API", "ui": "the Streamlit app (port 8501)",
            "interactive_docs": "/docs", "health": "/health",
            "backtest": "POST /backtest", "disclaimer": DISCLAIMER}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine_version": "0.0.1", "disclaimer": DISCLAIMER}


@app.get("/data-connections")
def data_connections() -> dict:
    """Connection readiness only. It never returns paths, keys, or raw vendor data."""
    return data_connections_status()


@app.post("/protocols")
def create_protocol(req: ProtocolCreateRequest) -> dict:
    """Create an append-only study protocol before inspecting a protected stage."""
    try:
        return _PROTOCOL_STORE.create(req.hypothesis, {
            "research_end": req.research_end,
            "validation_end": req.validation_end,
            "final_holdout_end": req.final_holdout_end,
        })
    except ProtocolError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/protocols/{study_id}")
def get_protocol(study_id: str) -> dict:
    try:
        return _PROTOCOL_STORE.summary(study_id)
    except ProtocolError as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.post("/protocols/{study_id}/run")
def run_protocol_stage(study_id: str, req: ProtocolRunRequest, request: Request, response: Response) -> dict:
    """Run one stage through the protocol-enforced chronological window."""
    if study_id != req.study_id:
        raise HTTPException(status_code=400, detail="Protocol URL and request study_id must match.")
    ledger = _get_ledger(request, response, principal=current_principal(request))
    payload = req.model_dump(exclude={"study_id", "stage"})
    try:
        result = ProtocolRunner(_PROTOCOL_STORE).run(
            study_id, req.stage, payload,
            lambda prepared: run_backtest_workflow(prepared, ledger=ledger),
        )
    except (ProtocolError, WorkflowError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"protocol stage failed: {type(error).__name__}: {error}")
    _maybe_persist_run(request, "backtest", {**payload, "protocol": {"study_id": study_id, "stage": req.stage}}, result)
    return result


@app.post("/backtest")
def run_backtest(req: BacktestRequest, request: Request, response: Response) -> dict:
    principal = current_principal(request)
    ledger = _get_ledger(request, response, principal=principal)
    try:
        result = run_backtest_workflow(req.model_dump(), ledger=ledger)
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # surface unexpected errors without a stack trace
        raise HTTPException(status_code=500, detail=f"backtest failed: {type(e).__name__}: {e}")
    _maybe_persist_run(request, "backtest", req.model_dump(), result)
    return result


@app.post("/benchmark-suite")
def benchmark_suite(req: BacktestRequest, request: Request, response: Response) -> dict:
    """Run the non-configurable reference-factor suite under one cost model."""
    ledger = _get_ledger(request, response, principal=current_principal(request))
    try:
        return run_benchmark_suite(req.model_dump(), ledger=ledger)
    except WorkflowError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"benchmark suite failed: {type(error).__name__}: {error}")


@app.post("/session/reset")
def reset_session(request: Request, response: Response, reason: str = "") -> dict:
    """Honest, logged reset of this workspace's trial counter (never silent)."""
    return _get_ledger(request, response, principal=current_principal(request)).reset(reason=reason)


@app.post("/ml-compare")
def ml_compare(req: MLCompareRequest, request: Request) -> dict:
    """Opt-in, slow (~1-2 min): leak-aware ML model-ladder comparison. Does NOT
    consume a trial — it evaluates models, not a single strategy hypothesis. Optional
    ``ml_factors`` / ``ml_models`` / ``horizon`` let the caller compose the feature
    panel and ladder; defaults run a built-in feature set over the full ladder."""
    try:
        result = run_model_comparison(
            req.model_dump(exclude={"ml_factors", "ml_models", "horizon"}),
            factor_names=req.ml_factors, model_names=req.ml_models, horizon=req.horizon)
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model comparison failed: {type(e).__name__}: {e}")
    _maybe_persist_run(request, "ml_compare", req.model_dump(), result)
    return result


@app.post("/portfolio-research")
def portfolio_research(req: PortfolioResearchRequest, request: Request, response: Response) -> dict:
    """Historical portfolio construction, execution stress, and paper-plan output."""
    ledger = _get_ledger(request, response, principal=current_principal(request))
    try:
        result = run_portfolio_research(req.model_dump(), ledger=ledger)
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"portfolio research failed: {type(e).__name__}: {e}")
    _maybe_persist_run(request, "portfolio_research", req.model_dump(), result)
    return result


@app.post("/public-data-pilot")
def public_data_pilot(req: PublicDataPilotRequest) -> dict:
    try:
        return run_public_data_pilot(req.model_dump())
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"public-data pilot failed: {type(e).__name__}: {e}")


@app.post("/real-document-batch")
def real_document_batch(req: RealDocumentBatchRequest) -> dict:
    try:
        return run_real_document_batch(req.model_dump())
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"real-document batch failed: {type(e).__name__}: {e}")


@app.post("/filing-event-study")
def filing_event_study(req: FilingEventStudyRequest) -> dict:
    try:
        return run_filing_event_study(req.model_dump())
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"filing event study failed: {type(e).__name__}: {e}")


@app.post("/microstructure-study")
def microstructure_study(req: MicrostructureRequest) -> dict:
    """Trade-flow study and assumption-visible market-making simulation."""
    try:
        return run_microstructure_lab(req.model_dump())
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"microstructure study failed: {type(e).__name__}: {e}")
