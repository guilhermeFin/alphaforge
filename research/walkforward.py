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


def _flatten_lead(signal_slice, lead: int):
    """Return a copy of ``signal_slice`` with its first ``lead`` rows set to 0.0.

    Flattening (not truncating) is deliberate: the backtest still sees the full
    [lo, hi) price path, so no fabricated return discontinuity appears at the
    fold start. The flattened rows simply hold a flat (zero) position, and the
    caller excludes them from scoring so the structural zeros never enter Sharpe.
    """
    if lead <= 0:
        return signal_slice
    out = signal_slice.copy()
    if isinstance(out, pd.DataFrame):
        out.iloc[:lead, :] = 0.0
    else:
        out.iloc[:lead] = 0.0
    return out


def walk_forward(
    close,
    signal,
    n_splits: int = 5,
    cost_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> dict:
    """Anchored walk-forward: stitch consecutive out-of-sample windows together
    and score the concatenated OOS track record with the Deflated Sharpe Ratio,
    treating each fold as one of ``n_splits`` trials (a multiple-testing haircut).

    Purging + embargo (``purge_bars``, ``embargo_bars``)
    ----------------------------------------------------
    Weights are computed ONCE over the full panel, so a factor with an L-bar
    lookback derives its first ~L out-of-sample values from PRIOR-fold prices —
    boundary contamination that inflates the stitched OOS Sharpe and the
    credibility verdict. For each OOS fold we therefore FLATTEN the leading
    ``purge_bars + embargo_bars`` signal rows to a zero position, run the
    backtest on the full [lo, hi) slice (no truncation → no fabricated return
    jump at the boundary), then SCORE ONLY the returns *after* that flattened
    lead. Purging drops the contaminated lookback lead; embargo additionally
    absorbs forward-return serial correlation across the fold boundary.

    With the defaults ``purge_bars=0, embargo_bars=0`` the lead is empty and the
    function reproduces the legacy (un-purged) behaviour exactly.
    """
    n = len(close)
    if n_splits < 2 or n < n_splits * 2:
        raise ValueError("need a longer series / fewer splits")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars must be non-negative")
    bounds = np.linspace(0, n, n_splits + 1).astype(int)

    lead = purge_bars + embargo_bars
    fold_lengths = [bounds[k + 1] - bounds[k] for k in range(1, n_splits)]
    smallest_fold = min(fold_lengths)
    if lead >= smallest_fold:
        raise ValueError(
            f"purge_bars + embargo_bars ({lead}) must be smaller than the "
            f"smallest fold length ({smallest_fold})"
        )

    fold_sharpes, oos_returns, purged_per_fold = [], [], []
    for k in range(1, n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        sig_slice = _flatten_lead(signal.iloc[lo:hi], lead)
        oos = backtest(close.iloc[lo:hi], sig_slice, cost_bps, periods_per_year=periods_per_year)
        # Exclude the flattened lead from scoring so structural zeros don't bias Sharpe.
        kept = oos.returns.iloc[lead:]
        fold_sharpes.append(metrics.annualised_sharpe(kept, periods_per_year))
        oos_returns.append(kept)
        purged_per_fold.append(int(lead))

    stitched = pd.concat(oos_returns)
    dsr = metrics.deflated_sharpe_ratio(stitched, n_trials=n_splits)
    return {
        "n_splits": n_splits,
        # the first window is the warm-up (signals need history), so there are
        # n_splits-1 genuinely out-of-sample folds — reported honestly here.
        "n_oos_folds": len(fold_sharpes),
        "fold_oos_sharpes": [round(float(s), 3) for s in fold_sharpes],
        "mean_oos_sharpe": round(float(np.nanmean(fold_sharpes)), 3),
        "stitched_oos_sharpe": round(metrics.annualised_sharpe(stitched, periods_per_year), 3),
        "deflated_sr": round(dsr, 3) if not np.isnan(dsr) else None,
        "passes": bool((not np.isnan(dsr)) and dsr > 0.95),
        # Purge/embargo audit trail.
        "purge_bars": int(purge_bars),
        "embargo_bars": int(embargo_bars),
        "purged_bars_per_fold": purged_per_fold,
        "total_purged_bars": int(sum(purged_per_fold)),
    }
