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

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from api.service import (  # noqa: E402
    DISCLAIMER, FACTORS, PROVIDERS, WorkflowError, run_backtest_workflow,
)

app = FastAPI(
    title="AlphaForge API",
    version="0.0.1",
    description="The research OS for emerging quant managers — "
                "the backtest that won't let you lie to yourself. " + DISCLAIMER,
)


class BacktestRequest(BaseModel):
    provider: str = Field("synthetic", description=f"one of {PROVIDERS}")
    symbols: list[str] = Field(default_factory=list,
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine_version": "0.0.1", "disclaimer": DISCLAIMER}


@app.post("/backtest")
def run_backtest(req: BacktestRequest) -> dict:
    try:
        return run_backtest_workflow(req.model_dump())
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # surface unexpected errors without a stack trace
        raise HTTPException(status_code=500, detail=f"backtest failed: {type(e).__name__}: {e}")
