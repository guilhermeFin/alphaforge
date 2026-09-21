"""Factor-health diagnostics that make cross-sectional signals inspectable."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import signal_quality


def _quantile_turnover(signal: pd.DataFrame, quantiles: int) -> dict:
    top_changes, bottom_changes = [], []
    prior_top: set[str] | None = None
    prior_bottom: set[str] | None = None
    for _, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < quantiles * 2:
            continue
        n = max(1, len(valid) // quantiles)
        bottom = set(valid.nsmallest(n).index.astype(str))
        top = set(valid.nlargest(n).index.astype(str))
        if prior_top is not None:
            top_changes.append(1.0 - len(top & prior_top) / max(1, len(top | prior_top)))
            bottom_changes.append(1.0 - len(bottom & prior_bottom) / max(1, len(bottom | prior_bottom)))
        prior_top, prior_bottom = top, bottom
    return {"top_quantile_turnover": float(np.mean(top_changes)) if top_changes else None,
            "bottom_quantile_turnover": float(np.mean(bottom_changes)) if bottom_changes else None}


def factor_health(signal: pd.DataFrame, close: pd.DataFrame, *, horizon: int = 1,
                  quantiles: int = 5) -> dict:
    """IC stability, quantile monotonicity, coverage, and implementability proxies."""
    aligned = signal.reindex_like(close)
    ic = signal_quality.ic_series(aligned, close, horizon=horizon)
    monthly = (ic.resample("ME").mean() if isinstance(ic.index, pd.DatetimeIndex) else pd.Series(dtype=float))
    quantile = signal_quality.quantile_forward_returns(aligned, close, n_quantiles=quantiles, horizon=horizon)
    coverage = aligned.notna().mean(axis=1)
    return {
        "available": bool(len(ic)), "horizon": horizon,
        "ic_summary": signal_quality.ic_summary(aligned, close, horizon=horizon),
        "monthly_ic": [{"month": str(idx.date()), "ic": float(value)} for idx, value in monthly.dropna().items()],
        "quantile_returns": quantile,
        "coverage": {"average": float(coverage.mean()), "minimum": float(coverage.min()),
                     "days_below_80pct": int((coverage < 0.80).sum())},
        "turnover": _quantile_turnover(aligned, quantiles),
        "rank_autocorrelation": signal_quality.rank_autocorrelation(aligned),
        "group_analysis": {"available": False, "reason": "No point-in-time sector classification was supplied."},
    }
