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
    run_model_comparison,
)
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


class MLCompareRequest(BacktestRequest):
    """A backtest request plus the optional knobs that only the ML ladder uses."""
    ml_factors: list[str] | None = Field(
        default=None, max_length=12,
        description="factor_lib features for the model panel (default: a built-in set)")
    ml_models: list[str] | None = Field(
        default=None, max_length=16,
        description="which models in the ladder to run (default: the full ladder)")
    horizon: int = Field(21, ge=1, le=63, description="forward-return label horizon (trading days)")


# Per-workspace trial ledgers. A single global ledger would conflate users (one
# person's sweep would haircut a stranger's first run), so we scope each ledger to a
# workspace: an AUTHENTICATED caller's ledger is keyed by their workspace id (shared
# across their keys/sessions); an anonymous caller's is keyed by a session cookie.
# Bounded so an unauthenticated caller minting endless cookies can't grow it without
# limit (memory-DoS guard); evicts the oldest when full.
MAX_SESSIONS = 10_000
_LEDGERS: dict[str, TrialLedger] = {}


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
