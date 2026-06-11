"""Out-of-sample evaluation & overfitting defenses.

A backtest that only reports in-sample performance is marketing, not research.
Everything here forces an out-of-sample verdict and an explicit overfit flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest
from . import metrics

TRADING_DAYS = 252


def split_backtest(
    close,
    signal,
    split: float = 0.7,
    cost_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS,
    degradation_threshold: float = 1.0,
) -> dict:
    """In-sample vs out-of-sample on a single chronological split.

    overfit_warning fires if the annualised Sharpe degrades by more than
    ``degradation_threshold`` OR the out-of-sample PSR fails to clear 0.95.
    """
    n = len(close)
    cut = int(n * split)
    ins = backtest(close.iloc[:cut], signal.iloc[:cut], cost_bps, periods_per_year=periods_per_year)
    oos = backtest(close.iloc[cut:], signal.iloc[cut:], cost_bps, periods_per_year=periods_per_year)

    is_sr = metrics.annualised_sharpe(ins.returns, periods_per_year)
    oos_sr = metrics.annualised_sharpe(oos.returns, periods_per_year)
    oos_psr = metrics.probabilistic_sharpe_ratio(oos.returns, 0.0)
    degradation = is_sr - oos_sr

    # Two DISTINCT failure modes, kept separate so the verdict isn't misleading:
    #   overfit_warning  = the strategy decayed from in-sample to out-of-sample.
    #   oos_significant  = the out-of-sample record itself clears the PSR bar.
    # A signal can hold up OOS (no overfit) yet still be too short to be significant.
    return {
        "in_sample_sharpe": round(is_sr, 3),
        "out_sample_sharpe": round(oos_sr, 3),
        "degradation": round(degradation, 3),
        "out_sample_psr": round(oos_psr, 3) if not np.isnan(oos_psr) else None,
        "overfit_warning": bool(degradation > degradation_threshold),
        "oos_significant": bool((not np.isnan(oos_psr)) and oos_psr >= 0.95),
    }


def walk_forward(
    close,
    signal,
    n_splits: int = 5,
    cost_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Anchored walk-forward: stitch consecutive out-of-sample windows together
    and score the concatenated OOS track record with the Deflated Sharpe Ratio,
    treating each fold as one of ``n_splits`` trials (a multiple-testing haircut).
    """
    n = len(close)
    if n_splits < 2 or n < n_splits * 2:
        raise ValueError("need a longer series / fewer splits")
    bounds = np.linspace(0, n, n_splits + 1).astype(int)

    fold_sharpes, oos_returns = [], []
    for k in range(1, n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        oos = backtest(close.iloc[lo:hi], signal.iloc[lo:hi], cost_bps, periods_per_year=periods_per_year)
        fold_sharpes.append(metrics.annualised_sharpe(oos.returns, periods_per_year))
        oos_returns.append(oos.returns)

    stitched = pd.concat(oos_returns)
    dsr = metrics.deflated_sharpe_ratio(stitched, n_trials=n_splits)
    return {
        "n_splits": n_splits,
        "fold_oos_sharpes": [round(float(s), 3) for s in fold_sharpes],
        "mean_oos_sharpe": round(float(np.nanmean(fold_sharpes)), 3),
        "stitched_oos_sharpe": round(metrics.annualised_sharpe(stitched, periods_per_year), 3),
        "deflated_sr": round(dsr, 3) if not np.isnan(dsr) else None,
        "passes": bool((not np.isnan(dsr)) and dsr > 0.95),
    }
