"""Free robustness checks that expose resampling and regime fragility."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def moving_block_bootstrap(returns: pd.Series, *, block_size: int = 21, samples: int = 400, seed: int = 7, periods_per_year: int = 252) -> dict:
    """Resample contiguous blocks; unlike iid bootstraps, serial dependence survives."""
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if r.size < max(40, block_size * 2):
        return {"available": False, "reason": "need at least two bootstrap blocks of clean returns"}
    if not (2 <= block_size <= r.size):
        raise ValueError("bootstrap block size must be between 2 and the return count")
    if not (100 <= samples <= 2_000):
        raise ValueError("bootstrap samples must be in [100, 2000]")
    rng, stat = np.random.default_rng(seed), np.empty((samples, 3), dtype=float)
    starts_max = r.size - block_size + 1
    for sample in range(samples):
        blocks, remaining = [], r.size
        while remaining > 0:
            block = r[int(rng.integers(0, starts_max)):][:min(block_size, remaining)]
            blocks.append(block)
            remaining -= block.size
        draw = np.concatenate(blocks)
        stat[sample] = (metrics.annualised_return(draw, periods_per_year), metrics.annualised_sharpe(draw, periods_per_year), metrics.max_drawdown(metrics.equity_curve(draw)))
    cagr, sharpe, drawdown = stat.T
    return {
        "available": True, "method": "moving-block bootstrap", "samples": int(samples), "block_size": int(block_size),
        "annual_return_p05": float(np.quantile(cagr, 0.05)), "annual_return_median": float(np.quantile(cagr, 0.50)), "annual_return_p95": float(np.quantile(cagr, 0.95)),
        "sharpe_p05": float(np.quantile(sharpe, 0.05)), "sharpe_median": float(np.quantile(sharpe, 0.50)),
        "probability_positive_annual_return": float((cagr > 0).mean()), "probability_positive_sharpe": float((sharpe > 0).mean()),
        "drawdown_p95": float(np.quantile(drawdown, 0.05)),
    }


def chronological_subperiods(returns: pd.Series, *, segments: int = 4, periods_per_year: int = 252) -> list[dict]:
    """Score equal chronological slices so one market regime cannot hide the rest."""
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.size < segments * 20:
        return []
    rows = []
    # ``np.array_split`` turns a Series into bare ndarrays, which would discard
    # the audit dates. Split positions, then retain the original time index.
    for i, positions in enumerate(np.array_split(np.arange(r.size), segments), start=1):
        chunk = r.iloc[positions]
        rows.append({
            "segment": i, "start": str(chunk.index[0].date()) if hasattr(chunk.index[0], "date") else str(chunk.index[0]),
            "end": str(chunk.index[-1].date()) if hasattr(chunk.index[-1], "date") else str(chunk.index[-1]),
            "annual_return": metrics.annualised_return(chunk, periods_per_year), "sharpe": metrics.annualised_sharpe(chunk, periods_per_year),
            "max_drawdown": metrics.max_drawdown(metrics.equity_curve(chunk)),
        })
    return rows


def robustness_report(returns: pd.Series, *, seed: int = 7) -> dict:
    bootstrap, subperiods = moving_block_bootstrap(returns, seed=seed), chronological_subperiods(returns)
    positive = sum(row["sharpe"] > 0 for row in subperiods)
    if not subperiods:
        verdict = "insufficient history for subperiod stability"
    elif positive == len(subperiods):
        verdict = "all chronological segments had positive Sharpe; still exploratory"
    elif not positive:
        verdict = "no chronological segment had positive Sharpe"
    else:
        verdict = f"mixed stability: {positive} of {len(subperiods)} chronological segments had positive Sharpe"
    return {"bootstrap": bootstrap, "subperiods": subperiods, "verdict": verdict}
