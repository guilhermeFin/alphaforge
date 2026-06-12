"""Backtest workflow service — ONE source of truth for the browser workflow.

Both the FastAPI app (api/main.py) and the Streamlit page (app/Home.py, in
direct-import fallback mode) call ``run_backtest_workflow``. Keeping the
orchestration here means the API and the UI can never drift apart, and the
honesty guards are applied identically no matter how the engine is reached.

This layer adds NOTHING statistical — it only validates inputs, calls research/,
and serialises results. All honesty guarantees live in research/ and its tests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research import data, factors, fundamentals, stats_guards
from research.backtest import backtest
from research.walkforward import split_backtest, walk_forward

DISCLAIMER = (
    "AlphaForge is research software, not investment advice. Backtests are "
    "hypothetical, ignore some real-world frictions, and past performance does "
    "not predict future results. Synthetic-data results validate the engine, "
    "never an edge."
)

FACTORS = ("momentum", "reversal", "lowvol", "blend", "value", "quality", "value_quality")
FUNDAMENTAL_FACTORS = ("value", "quality", "value_quality")
PROVIDERS = ("synthetic", "yfinance")

# hard input bounds — the service refuses absurd requests instead of hanging
MAX_SYMBOLS = 50
MIN_PERIODS, MAX_PERIODS = 120, 6000
MAX_TICKER_LEN = 10


class WorkflowError(ValueError):
    """User-facing input/validation error (maps to HTTP 400)."""


def _validate(req: dict) -> dict:
    provider = str(req.get("provider", "synthetic")).lower()
    if provider not in PROVIDERS:
        raise WorkflowError(f"provider must be one of {PROVIDERS}")

    factor = str(req.get("factor", "momentum")).lower()
    if factor not in FACTORS:
        raise WorkflowError(f"factor must be one of {FACTORS}")

    symbols = req.get("symbols") or []
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if provider == "yfinance":
        if len(symbols) < 2:
            raise WorkflowError("yfinance provider needs at least 2 ticker symbols "
                                "(this is a cross-sectional long/short strategy)")
        bad = [s for s in symbols if len(s) > MAX_TICKER_LEN or not s.replace(".", "").replace("-", "").isalnum()]
        if bad:
            raise WorkflowError(f"invalid ticker(s): {bad}")
    if len(symbols) > MAX_SYMBOLS:
        raise WorkflowError(f"too many symbols (max {MAX_SYMBOLS})")

    periods = int(req.get("periods", 1512))
    if not (MIN_PERIODS <= periods <= MAX_PERIODS):
        raise WorkflowError(f"periods must be in [{MIN_PERIODS}, {MAX_PERIODS}]")

    lookback = int(req.get("lookback", 252))
    skip = int(req.get("skip", 21))
    if not (5 <= lookback <= 756):
        raise WorkflowError("lookback must be in [5, 756]")
    if not (0 <= skip < lookback):
        raise WorkflowError("skip must satisfy 0 <= skip < lookback")

    cost_bps = float(req.get("cost_bps", 5.0))
    if not (0.0 <= cost_bps <= 200.0):
        raise WorkflowError("cost_bps must be in [0, 200]")

    n_trials = int(req.get("n_trials", 1))
    if not (1 <= n_trials <= 100_000):
        raise WorkflowError("n_trials must be in [1, 100000]")

    gross = float(req.get("gross", 1.0))
    if not (0.1 <= gross <= 3.0):
        raise WorkflowError("gross must be in [0.1, 3.0]")

    seed = int(req.get("seed", 42))
    if seed < 0:
        raise WorkflowError("seed must be >= 0")

    start = str(req.get("start", "2015-01-02"))
    try:
        pd.Timestamp(start)
    except Exception:
        raise WorkflowError("start must be a valid date, e.g. 2015-01-02")

    return {
        "provider": provider, "factor": factor, "symbols": symbols,
        "periods": periods, "seed": seed, "start": start,
        "lookback": lookback, "skip": skip,
        "cost_bps": cost_bps, "n_trials": n_trials, "gross": gross,
    }


def _effective_lookback(factor: str, lookback: int) -> int:
    """The lookback actually used — some factors cap it (reported honestly in meta)."""
    if factor == "reversal":
        return min(lookback, 63)
    if factor == "lowvol":
        return min(lookback, 252)
    return lookback  # momentum (and the momentum leg of blend) uses it as-is


def _score(close: pd.DataFrame, factor: str, lookback: int, skip: int) -> pd.DataFrame:
    z = factors.cross_sectional_zscore
    if factor == "momentum":
        return z(factors.momentum(close, lookback=lookback, skip=skip))
    if factor == "reversal":
        return z(factors.short_term_reversal(close, lookback=min(lookback, 63)))
    if factor == "lowvol":
        return z(-factors.trailing_volatility(close, lookback=min(lookback, 252)))
    # blend: equal-weight momentum + low-vol + reversal
    return factors.blend(
        z(factors.momentum(close, lookback=lookback, skip=skip)),
        z(-factors.trailing_volatility(close, lookback=min(lookback, 252))),
        z(factors.short_term_reversal(close, lookback=21)),
    )


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain JSON-safe Python."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def _downsample_curve(s: pd.Series, max_points: int = 500) -> list[dict]:
    """Downsample a curve for charting WITHOUT hiding extrema.

    Plain stride-slicing can skip the true peak/trough, so a drawdown chart could
    show a shallower trough than the max-drawdown number printed beside it. We
    always keep the global min and max (and the first/last point), so the chart
    can never understate the worst drawdown — which for an honesty tool matters.
    """
    if s.empty:
        return []
    stride = max(1, len(s) // max_points)
    keep = set(range(0, len(s), stride))
    keep.update({0, len(s) - 1, int(np.argmin(s.values)), int(np.argmax(s.values))})
    sub = s.iloc[sorted(keep)]
    return [{"date": str(getattr(idx, "date", lambda: idx)()), "value": round(float(v), 6)}
            for idx, v in sub.items()]


def run_backtest_workflow(req: dict) -> dict:
    """Validate -> load data -> factor -> honest backtest -> full verdict dict."""
    p = _validate(req)

    if p["factor"] in FUNDAMENTAL_FACTORS:
        # Fundamental factors need point-in-time fundamentals. Free vendors serve
        # RESTATED numbers (look-ahead), so the only honest path today is the
        # synthetic world; real PIT data plugs into the same interface later.
        if p["provider"] != "synthetic":
            raise WorkflowError(
                "value/quality factors require the 'synthetic' provider in this MVP — "
                "free point-in-time fundamentals are not wired yet, and using restated "
                "free data would reintroduce the look-ahead this tool refuses to allow")
        syms = p["symbols"] or [f"S{i:02d}" for i in range(20)]
        world = data.make_synthetic_world(syms, start=p["start"], periods=p["periods"],
                                          seed=p["seed"], quality_to_drift=0.0016)
        close = world.close
        if len(close) < 400:
            raise WorkflowError("fundamental factors need >= 400 days (several quarters + OOS)")
        obs = data.make_synthetic_fundamentals(world, seed=p["seed"])
        fund = fundamentals.build_fundamentals(obs, close.index, list(close.columns))
        if p["factor"] == "value":
            score = fundamentals.value_score(fund, close)
        elif p["factor"] == "quality":
            score = fundamentals.quality_score(fund, close)
        else:
            score = fundamentals.value_quality_score(fund, close)
    else:
        if p["provider"] == "synthetic":
            panel = data.get_panel(
                p["symbols"] or [f"S{i:02d}" for i in range(15)],
                provider="synthetic", start=p["start"],
                periods=p["periods"], seed=p["seed"],
            )
        else:
            panel = data.get_panel(p["symbols"], start=p["start"], provider="yfinance")
        close = panel.close.dropna(how="all", axis=1)
        if close.shape[1] < 2:
            raise WorkflowError("need at least 2 symbols with data for a cross-sectional strategy")
        if len(close) < MIN_PERIODS:
            raise WorkflowError(f"only {len(close)} usable days of data (need >= {MIN_PERIODS})")
        if len(close) <= p["lookback"] + p["skip"] + 21:
            raise WorkflowError("lookback too long for the data window")
        score = _score(close, p["factor"], p["lookback"], p["skip"])

    weights = factors.long_short_weights(score, gross=p["gross"])
    res = backtest(close, weights, cost_bps=p["cost_bps"])

    scorecard = res.summary(n_trials=p["n_trials"])
    oos = split_backtest(close, weights, split=0.7, cost_bps=p["cost_bps"])
    try:
        wf = walk_forward(close, weights, n_splits=5, cost_bps=p["cost_bps"])
    except ValueError as e:
        wf = {"error": str(e)}
    tails = stats_guards.fat_tail_report(res.returns)

    # Headline verdict is OUT-OF-SAMPLE first. A strategy that only passes the
    # full-sample Deflated Sharpe but fails walk-forward / decays out-of-sample is
    # NOT credible — anything else would let the tool flatter a data-mined fit.
    def _ok(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    full_dsr = scorecard.get("deflated_sr")
    if "error" not in wf:
        credible = bool(wf.get("passes")) and not oos.get("overfit_warning", False)
        basis = "out-of-sample walk-forward Deflated Sharpe"
    else:  # series too short to walk-forward — fall back to full-sample, and say so
        credible = (_ok(full_dsr) and full_dsr > 0.95) and not oos.get("overfit_warning", False)
        basis = "full-sample Deflated Sharpe (series too short for walk-forward)"
    verdict = (
        f"CREDIBLE: survives the {basis} bar"
        if credible
        else f"NOT CREDIBLE: does not survive the {basis} bar"
    )

    dd = (res.equity / res.equity.cummax() - 1.0)
    return _jsonable({
        "meta": {
            **p,
            "effective_lookback": (None if p["factor"] in FUNDAMENTAL_FACTORS
                                   else _effective_lookback(p["factor"], p["lookback"])),
            "n_symbols": int(close.shape[1]),
            "n_days": int(len(close)),
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "engine_version": "0.0.1",
            "disclaimer": DISCLAIMER,
        },
        "verdict": verdict,
        "verdict_basis": basis,
        "scorecard": scorecard,
        "out_of_sample": oos,
        "walk_forward": wf,
        "fat_tails": tails,
        "equity_curve": _downsample_curve(res.equity),
        "drawdown_curve": _downsample_curve(dd),
    })
