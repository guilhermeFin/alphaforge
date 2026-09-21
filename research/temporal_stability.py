"""Chronological stability diagnostics for a fixed, point-in-time signal.

The module does not discover parameters or certify an alpha. It asks a narrower
question: after a signal is fixed, does its observed association with future
returns persist across contiguous historical cohorts? Cohorts are separated by
an explicit embargo so a horizon label cannot straddle two reported eras.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_COHORTS = 5
DEFAULT_MIN_OBSERVATIONS = 20
DEFAULT_EMBARGO_BARS = 1
DEFAULT_BLOCK_SIZE = 5
DEFAULT_RESAMPLES = 500


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _finite_series(values: Iterable[float] | pd.Series) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series[np.isfinite(series.to_numpy(dtype=float))]


def _cohort_slices(n: int, n_cohorts: int, embargo_bars: int) -> list[slice]:
    usable = n - (n_cohorts - 1) * embargo_bars
    lengths = [usable // n_cohorts + (index < usable % n_cohorts) for index in range(n_cohorts)]
    start, windows = 0, []
    for length in lengths:
        windows.append(slice(start, start + length))
        start += length + embargo_bars
    return windows


def _block_bootstrap_ci(values: pd.Series, *, block_size: int, n_resamples: int, seed: int) -> list[float | None]:
    values = _finite_series(values)
    if values.size < 2:
        return [None, None]
    array = values.to_numpy(dtype=float)
    block = max(1, min(int(block_size), array.size))
    blocks_needed = int(np.ceil(array.size / block))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        starts = rng.integers(0, array.size, size=blocks_needed)
        sample = np.concatenate([array[(start + np.arange(block)) % array.size] for start in starts])[:array.size]
        means[index] = sample.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return [_round(low), _round(high)]


def _block_bootstrap_difference_ci(
    earliest: pd.Series, latest: pd.Series, *, block_size: int, n_resamples: int, seed: int,
) -> list[float | None]:
    """Block-bootstrap latest minus earliest cohort means, preserving chronology."""
    early, late = _finite_series(earliest).to_numpy(dtype=float), _finite_series(latest).to_numpy(dtype=float)
    if early.size < 2 or late.size < 2:
        return [None, None]
    rng = np.random.default_rng(seed)

    def draw_mean(values: np.ndarray) -> float:
        block = max(1, min(int(block_size), values.size))
        starts = rng.integers(0, values.size, size=int(np.ceil(values.size / block)))
        sample = np.concatenate([values[(start + np.arange(block)) % values.size] for start in starts])[:values.size]
        return float(sample.mean())

    differences = np.asarray([draw_mean(late) - draw_mean(early) for _ in range(n_resamples)])
    low, high = np.quantile(differences, [0.025, 0.975])
    return [_round(low), _round(high)]


def _cohort_summary(index: int, values: pd.Series, strategy_returns: pd.Series | None, *, block_size: int, n_resamples: int) -> dict:
    values = _finite_series(values)
    mean = float(values.mean()) if values.size else None
    std = float(values.std(ddof=1)) if values.size > 1 else None
    strategy_slice = _finite_series(strategy_returns.reindex(values.index)) if strategy_returns is not None else pd.Series(dtype=float)
    cumulative = float((1.0 + strategy_slice).prod() - 1.0) if strategy_slice.size else None
    return {
        "cohort": index + 1,
        "start": str(values.index.min()) if values.size else None,
        "end": str(values.index.max()) if values.size else None,
        "ic_observations": int(values.size),
        "mean_ic": _round(mean),
        "ic_ir": _round(mean / std if std and std > 0 else None, 3),
        "ic_hit_rate": _round(float((values > 0).mean()) if values.size else None, 3),
        "mean_ic_ci_95": _block_bootstrap_ci(values, block_size=block_size, n_resamples=n_resamples, seed=20_260_917 + index),
        "strategy_return_observations": int(strategy_slice.size),
        "strategy_cumulative_return": _round(cumulative),
    }


def chronological_stability_summary(
    ic_values: Iterable[float] | pd.Series,
    strategy_returns: Iterable[float] | pd.Series | None = None,
    *,
    n_cohorts: int = DEFAULT_COHORTS,
    min_observations_per_cohort: int = DEFAULT_MIN_OBSERVATIONS,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict:
    """Assess temporal persistence without selecting a favorable historical era.

    A result describes observed chronology, not a deployable claim. Small samples
    and inconsistent signs deliberately return ``insufficient_evidence`` or
    ``unstable`` instead of producing a pooled, flattering statistic.
    """
    if n_cohorts < 2 or min_observations_per_cohort < 2:
        raise ValueError("n_cohorts and min_observations_per_cohort must both be at least 2")
    if embargo_bars < 0 or block_size < 1 or n_resamples < 50:
        raise ValueError("embargo_bars must be non-negative; block_size >= 1 and n_resamples >= 50")

    values = _finite_series(ic_values).sort_index()
    returns = _finite_series(strategy_returns).sort_index() if strategy_returns is not None else None
    required = n_cohorts * min_observations_per_cohort + (n_cohorts - 1) * embargo_bars
    base = {
        "method": "contiguous chronological IC cohorts with block-bootstrap mean intervals",
        "n_cohorts": int(n_cohorts),
        "min_observations_per_cohort": int(min_observations_per_cohort),
        "embargo_bars": int(embargo_bars),
        "block_size": int(block_size),
        "n_resamples": int(n_resamples),
        "available_ic_observations": int(values.size),
        "required_ic_observations": int(required),
        "cohorts": [],
        "supports_current_evidence": False,
    }
    if values.size < required:
        return {
            **base,
            "status": "insufficient_evidence",
            "conclusion": (
                f"Insufficient chronological evidence: {values.size} valid IC observations are available; "
                f"{required} are required for {n_cohorts} pre-declared cohorts with embargoes."
            ),
        }

    windows = _cohort_slices(values.size, n_cohorts, embargo_bars)
    cohorts = [
        _cohort_summary(index, values.iloc[window], returns, block_size=block_size, n_resamples=n_resamples)
        for index, window in enumerate(windows)
    ]
    means = np.asarray([cohort["mean_ic"] for cohort in cohorts], dtype=float)
    overall = float(values.mean())
    direction = int(np.sign(overall)) or int(np.sign(np.median(means)))
    consistent = np.sign(means) == direction if direction else np.zeros_like(means, dtype=bool)
    consistency = float(consistent.mean())
    earliest, latest = float(means[0]), float(means[-1])
    delta = latest - earliest
    delta_ci = _block_bootstrap_difference_ci(
        values.iloc[windows[0]], values.iloc[windows[-1]],
        block_size=block_size, n_resamples=n_resamples, seed=20_260_990,
    )

    if direction == 0 or consistency < 0.8:
        status = "unstable"
        conclusion = (
            "Chronological relationship is unstable: the cohort IC direction changes across eras. "
            "The pooled result is not evidence of a persistent current relationship."
        )
    elif abs(latest) < abs(earliest) * 0.7:
        status = "weakened"
        conclusion = (
            "Chronological relationship weakened in the most recent cohort. Earlier performance must not be "
            "treated as evidence that the current relationship persists."
        )
    else:
        status = "stable"
        conclusion = (
            "The observed IC direction is consistent across the pre-declared cohorts. This is a temporal "
            "stability screen, not proof of an investment edge, causality, or future performance."
        )

    return {
        **base,
        "status": status,
        "cohorts": cohorts,
        "overall_mean_ic": _round(overall),
        "direction": "positive" if direction > 0 else "negative" if direction < 0 else "flat",
        "sign_consistency": _round(consistency, 3),
        "earliest_mean_ic": _round(earliest),
        "latest_mean_ic": _round(latest),
        "latest_minus_earliest_ic": _round(delta),
        "latest_minus_earliest_ic_ci_95": delta_ci,
        "supports_current_evidence": status == "stable",
        "conclusion": conclusion,
    }
