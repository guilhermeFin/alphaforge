"""Small, repeatable stress scenarios for research robustness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest


def cost_delay_matrix(close: pd.DataFrame, weights: pd.DataFrame, *, base_cost_bps: float) -> dict:
    """Replay the same pre-known signal under ordinary cost and fill-delay stress."""
    scenarios = []
    for multiplier in (1.0, 2.0, 3.0):
        for delay in (1, 2, 3):
            result = backtest(close, weights, cost_bps=base_cost_bps * multiplier, execution_lag=delay)
            summary = result.summary()
            scenarios.append({
                "cost_bps": float(base_cost_bps * multiplier), "execution_lag_days": delay,
                "annual_return": summary.get("cagr"), "sharpe": summary.get("ann_sharpe"),
                "positive_sharpe": bool((summary.get("ann_sharpe") or 0.0) > 0.0),
            })
    pass_rate = float(np.mean([s["positive_sharpe"] for s in scenarios])) if scenarios else 0.0
    status = "stable" if pass_rate >= 0.70 else "fragile"
    return {
        "available": True, "scenarios": scenarios, "positive_sharpe_rate": pass_rate,
        "status": status,
        "summary": f"{sum(s['positive_sharpe'] for s in scenarios)} of {len(scenarios)} cost/delay scenarios retained a positive Sharpe.",
    }
