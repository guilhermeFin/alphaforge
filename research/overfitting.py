"""Probability of Backtest Overfitting (PBO) via CSCV.

Combinatorially-Symmetric Cross-Validation (Bailey, Borwein, López de Prado &
Zhu, 2015). Given the per-period NET returns of every configuration you tried
(one column per config), CSCV estimates the probability that the configuration
which looked best in-sample (IS) then UNDER-performs the median out-of-sample
(OOS). A PBO near 0.5 means your selection process is no better than a coin flip
— the headline edge is almost certainly an artefact of the search itself.

The honesty story: a backtester can tell you a strategy's Sharpe. It cannot tell
you whether you only found that Sharpe because you tried five hundred knobs. PBO
is the number that catches exactly that lie.

Implementation notes
--------------------
* Rows are split into S CONTIGUOUS, equal-length time blocks (any ragged tail is
  dropped) so every IS/OOS partition is perfectly balanced. Rows are NEVER
  shuffled — that would destroy the autocorrelation structure of returns and is
  not what CSCV does.
* Per-block ``sum``, ``sumsq`` and ``n`` are precomputed ONCE. The Sharpe of a
  strategy over any union of blocks is then assembled from those pooled moments
  in O(S·M) per combination instead of O(T·M) — this is what makes the
  C(S, S/2) loop tractable.
* The pooled per-period Sharpe matches metrics.sharpe_per_period exactly:
  mean = sum/n, var = (sumsq - n·mean²)/(n-1) (ddof=1), annualised by
  ·sqrt(periods_per_year). Verified to 1e-9 in tests/test_overfitting.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EPS = 1e-9
MAX_COMBINATIONS = 13_000
_SHARPE_FLOOR = 1e-12  # matches metrics.sharpe_per_period std floor


@dataclass
class CSCVResult:
    pbo: float
    n_strategies: int
    n_splits: int
    n_combinations: int
    oos_sharpe_selected: list
    logits: list
    median_oos_sharpe_selected: float
    prob_oos_loss: float
    performance_degradation_slope: float
    histogram: dict = field(default_factory=dict)
    insufficient: bool = False
    note: str = ""


def _nan_safe_sharpe_from_moments(
    block_sum: np.ndarray,
    block_sumsq: np.ndarray,
    block_n: np.ndarray,
    mask: np.ndarray,
    periods_per_year: int,
) -> np.ndarray:
    """Annualised per-period Sharpe per strategy over the blocks in ``mask``.

    block_sum / block_sumsq are (S x M); block_n is (S,). ``mask`` is a boolean
    (S,) selecting which blocks to pool. Returns an (M,) array; entries with a
    degenerate (zero / non-finite) std are 0.0, matching
    metrics.sharpe_per_period.
    """
    n = float(block_n[mask].sum())
    if n < 2:
        return np.zeros(block_sum.shape[1], dtype=float)
    s = block_sum[mask].sum(axis=0)         # (M,)
    sq = block_sumsq[mask].sum(axis=0)      # (M,)
    mean = s / n
    var = (sq - n * mean * mean) / (n - 1.0)
    # Floating-point noise can push a (near-)constant series slightly negative.
    var = np.where(var < 0.0, 0.0, var)
    sd = np.sqrt(var)
    out = np.zeros_like(mean)
    ok = np.isfinite(sd) & (sd >= _SHARPE_FLOOR) & np.isfinite(mean)
    out[ok] = (mean[ok] / sd[ok]) * np.sqrt(periods_per_year)
    return out


def cscv_pbo(
    returns_matrix,
    n_splits: int = 8,
    periods_per_year: int = TRADING_DAYS,
    thorough: bool = False,
    n_bins: int = 20,
) -> CSCVResult:
    """Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    returns_matrix : (T x M) array / DataFrame of per-period NET returns; one
        column per configuration tried.
    n_splits : number of contiguous time blocks S. Must be even and >= 4.
        ``thorough=True`` (with the default n_splits) bumps S to 16.
    periods_per_year : annualisation factor for the Sharpe (matches metrics).
    n_bins : number of bins for the returned logit histogram.
    """
    df = returns_matrix if isinstance(returns_matrix, pd.DataFrame) else pd.DataFrame(returns_matrix)

    if thorough and n_splits == 8:
        n_splits = 16
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError("n_splits must be even and >= 4")

    half = n_splits // 2
    n_combinations = math.comb(n_splits, half)
    if n_combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"n_splits={n_splits} -> C({n_splits},{half})={n_combinations} "
            f"exceeds MAX_COMBINATIONS={MAX_COMBINATIONS}"
        )

    values = df.to_numpy(dtype=float)
    T, M = values.shape

    # --- insufficiency guards (no crash) ---
    if M < 2 or T < 2 * n_splits:
        return CSCVResult(
            pbo=float("nan"),
            n_strategies=int(M),
            n_splits=int(n_splits),
            n_combinations=int(n_combinations),
            oos_sharpe_selected=[],
            logits=[],
            median_oos_sharpe_selected=float("nan"),
            prob_oos_loss=float("nan"),
            performance_degradation_slope=float("nan"),
            histogram={"edges": [], "counts": []},
            insufficient=True,
            note=(
                f"insufficient data for CSCV: need M>=2 and T>=2*S "
                f"(M={M}, T={T}, S={n_splits})"
            ),
        )

    # --- contiguous equal-length blocks; drop ragged tail ---
    block_len = T // n_splits
    used = block_len * n_splits
    trimmed = values[:used]
    # reshape into (S, block_len, M) and reduce over the block axis
    blocks = trimmed.reshape(n_splits, block_len, M)
    block_sum = np.nansum(blocks, axis=1)                    # (S, M)
    block_sumsq = np.nansum(blocks * blocks, axis=1)         # (S, M)
    finite = np.isfinite(blocks).all(axis=2)                 # (S, block_len)
    block_n = finite.sum(axis=1).astype(float)               # (S,) per-block usable rows
    # Fallback: if columns disagree on NaN positions, count per row where the
    # row is fully finite (keeps pooled n consistent across strategies).
    if not np.isfinite(block_n).all() or (block_n == 0).any():
        block_n = np.full(n_splits, float(block_len))

    oos_selected = []
    is_selected = []
    logits = []

    for is_blocks in combinations(range(n_splits), half):
        is_mask = np.zeros(n_splits, dtype=bool)
        is_mask[list(is_blocks)] = True
        oos_mask = ~is_mask

        is_sharpe = _nan_safe_sharpe_from_moments(
            block_sum, block_sumsq, block_n, is_mask, periods_per_year
        )
        oos_sharpe = _nan_safe_sharpe_from_moments(
            block_sum, block_sumsq, block_n, oos_mask, periods_per_year
        )

        n_star = int(np.argmax(is_sharpe))           # best IS configuration
        oos_star = float(oos_sharpe[n_star])

        # relative rank of the selected config's OOS Sharpe among all M
        ranks = stats.rankdata(oos_sharpe, method="average")
        rel = ranks[n_star] / (M + 1.0)
        rel = min(max(rel, EPS), 1.0 - EPS)
        logit = math.log(rel / (1.0 - rel))

        logits.append(logit)
        oos_selected.append(oos_star)
        is_selected.append(float(is_sharpe[n_star]))

    logits_arr = np.asarray(logits, dtype=float)
    oos_arr = np.asarray(oos_selected, dtype=float)
    is_arr = np.asarray(is_selected, dtype=float)

    pbo = float(np.mean(logits_arr <= 0.0))
    median_oos = float(np.median(oos_arr))
    prob_oos_loss = float(np.mean(oos_arr < 0.0))

    # performance degradation: OLS slope of OOS Sharpe on IS Sharpe of the
    # selected config. A slope <= 0 means picking a better IS config does not
    # buy you better OOS performance — the hallmark of overfitting.
    if is_arr.size >= 2 and np.ptp(is_arr) > EPS:
        slope = float(np.polyfit(is_arr, oos_arr, 1)[0])
    else:
        slope = float("nan")

    counts, edges = np.histogram(logits_arr, bins=n_bins)
    histogram = {"edges": edges.tolist(), "counts": counts.tolist()}

    note = (
        f"CSCV PBO over {n_combinations} symmetric splits of S={n_splits} blocks; "
        f"M={M} configurations."
    )

    return CSCVResult(
        pbo=pbo,
        n_strategies=int(M),
        n_splits=int(n_splits),
        n_combinations=int(n_combinations),
        oos_sharpe_selected=oos_arr.tolist(),
        logits=logits_arr.tolist(),
        median_oos_sharpe_selected=median_oos,
        prob_oos_loss=prob_oos_loss,
        performance_degradation_slope=slope,
        histogram=histogram,
        insufficient=False,
        note=note,
    )


def _verdict(pbo: float, insufficient: bool) -> str:
    if insufficient or not math.isfinite(pbo):
        return "insufficient data — PBO not estimable"
    if pbo > 0.5:
        return "OVERFIT: the best in-sample config is more likely than not to be below-median out-of-sample"
    if pbo > 0.25:
        return "elevated overfitting risk — treat the selected config with caution"
    return "low overfitting risk for this search"


def compact_pbo(
    returns_matrix,
    n_splits: int = 8,
    periods_per_year: int = TRADING_DAYS,
    thorough: bool = False,
    n_bins: int = 20,
) -> dict:
    """JSON-able summary of cscv_pbo for readouts / reports."""
    res = cscv_pbo(
        returns_matrix,
        n_splits=n_splits,
        periods_per_year=periods_per_year,
        thorough=thorough,
        n_bins=n_bins,
    )
    overfit = bool(math.isfinite(res.pbo) and res.pbo > 0.5)
    return {
        "pbo": res.pbo,
        "n_strategies": res.n_strategies,
        "n_combinations": res.n_combinations,
        "median_oos_sharpe_selected": res.median_oos_sharpe_selected,
        "prob_oos_loss": res.prob_oos_loss,
        "performance_degradation_slope": res.performance_degradation_slope,
        "histogram": res.histogram,
        "insufficient": res.insufficient,
        "overfit": overfit,
        "verdict": _verdict(res.pbo, res.insufficient),
        "note": res.note,
    }


def candidate_returns_from_factor_grid(
    close,
    weight_fn,
    param_grid,
    cost_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS,
    n_splits: int = 8,
) -> pd.DataFrame:
    """Return one point-in-time-safe net-return column per parameter configuration.

    Parameters
    ----------
    close : prices (DataFrame panel or Series) passed to research.backtest.backtest.
    weight_fn : callable(close, **params) -> signal/target weights, same shape as
        close. Must be point-in-time-safe; the backtester adds the execution lag.
    param_grid : iterable of param dicts; one column of NET returns per dict.
    cost_bps, periods_per_year, n_splits : forwarded as documented.

    The matrix is reusable by CSCV, purged CPCV, White's Reality Check, and other
    multiple-testing diagnostics without rerunning the expensive backtests.
    """
    from .backtest import backtest  # local import: avoid import cycle

    columns: dict[str, pd.Series] = {}
    for i, params in enumerate(param_grid):
        signal = weight_fn(close, **params)
        res = backtest(
            close,
            signal,
            cost_bps=cost_bps,
            periods_per_year=periods_per_year,
        )
        label = "|".join(f"{k}={v}" for k, v in sorted(params.items())) or f"cfg{i}"
        if label in columns:
            label = f"{label}#{i}"
        columns[label] = pd.Series(res.returns)

    return pd.DataFrame(columns)


def pbo_from_factor_grid(
    close,
    weight_fn,
    param_grid,
    cost_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS,
    n_splits: int = 8,
) -> dict:
    """Run a parameter grid through the honest backtester, then CSCV-PBO it."""
    matrix = candidate_returns_from_factor_grid(
        close, weight_fn, param_grid, cost_bps=cost_bps, periods_per_year=periods_per_year,
    )
    return compact_pbo(
        matrix,
        n_splits=n_splits,
        periods_per_year=periods_per_year,
    )
