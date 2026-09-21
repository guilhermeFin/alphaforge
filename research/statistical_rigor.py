"""Autocorrelation- and multiple-testing-aware statistical rigor diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import TRADING_DAYS, probabilistic_sharpe_ratio, sharpe_per_period


def _clean(values) -> np.ndarray:
    arr = np.asarray(pd.Series(values), dtype=float)
    return arr[np.isfinite(arr)]


def default_hac_lag(n_obs: int) -> int:
    """Newey-West automatic bandwidth: ``floor(4*(T/100)^(2/9))``."""
    if n_obs < 2:
        return 0
    return min(n_obs - 1, int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def newey_west_long_run_variance(values, max_lag: int | None = None) -> float:
    """Bartlett-kernel Newey-West long-run variance of a series.

    ``LRV = gamma_0 + 2 sum_{k=1}^L (1-k/(L+1))*gamma_k`` where gamma uses a
    finite-sample ``1/(T-1)`` normalisation. This estimates serially correlated
    mean uncertainty while making the lag-zero result exactly the usual sample
    standard-error t-statistic.
    """
    x = _clean(values)
    if x.size < 2:
        return float("nan")
    lag = default_hac_lag(x.size) if max_lag is None else int(max_lag)
    if lag < 0:
        raise ValueError("max_lag must be >= 0")
    lag = min(lag, x.size - 1)
    centered = x - x.mean()
    normalizer = x.size - 1
    lrv = float(np.dot(centered, centered) / normalizer)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / normalizer)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return max(lrv, 0.0)


def newey_west_mean_tstat(values, max_lag: int | None = None) -> float:
    """HAC t-statistic for the null mean return (or IC) of zero.

    ``t_HAC = mean(x) / sqrt(LRV(x)/T)``.  With ``max_lag=0`` this is the
    conventional one-sample t-statistic; larger lags correct autocorrelation.
    """
    x = _clean(values)
    if x.size < 2:
        return float("nan")
    lrv = newey_west_long_run_variance(x, max_lag=max_lag)
    if not np.isfinite(lrv) or lrv <= 1e-18:
        return float("nan")
    return float(x.mean() / math.sqrt(lrv / x.size))


def hac_sharpe_tstat(returns, max_lag: int | None = None) -> float:
    """HAC-corrected t-statistic for a zero-Sharpe strategy.

    Testing zero Sharpe is equivalent to testing a zero expected excess return;
    this applies Newey-West to the strategy's excess-return series.
    """
    return newey_west_mean_tstat(returns, max_lag=max_lag)


def hac_ic_tstat(ic_values, max_lag: int | None = None) -> float:
    """HAC-corrected t-statistic for a zero mean Information Coefficient."""
    return newey_west_mean_tstat(ic_values, max_lag=max_lag)


def minimum_track_record_length(
    observed_sharpe_pp: float,
    sharpe_benchmark_pp: float = 0.0,
    confidence: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> int | None:
    """Minimum observations needed for a Sharpe to clear a benchmark at confidence.

    Bailey & Lopez de Prado's formula is ``MinTRL = 1 + D*(z/(SR-SR*))^2``, where
    ``D = 1 - skew*SR + ((kurtosis-1)/4)*SR^2`` and ``z=Phi^-1(confidence)``.
    Sharpe inputs are per-period; return ``None`` when the observed Sharpe does not
    exceed the benchmark or the inputs are invalid.
    """
    sr = float(observed_sharpe_pp)
    hurdle = float(sharpe_benchmark_pp)
    if not (0.5 < confidence < 1.0) or not all(np.isfinite(v) for v in (sr, hurdle, skew, kurtosis)):
        raise ValueError("confidence must be in (0.5, 1) and inputs must be finite")
    if sr <= hurdle:
        return None
    denominator = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2
    if denominator <= 0:
        return None
    required = 1.0 + denominator * (stats.norm.ppf(confidence) / (sr - hurdle)) ** 2
    return int(math.ceil(required))


@dataclass(frozen=True)
class CPCVResult:
    """Result of a return-matrix CPCV PBO calculation."""

    pbo: float
    n_strategies: int
    n_splits: int
    n_test_splits: int
    n_combinations: int
    purge_periods: int
    embargo_periods: int
    logits: list[float]
    median_oos_sharpe_selected: float
    insufficient: bool
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def purged_cpcv_pbo(
    candidate_returns,
    n_splits: int = 8,
    n_test_splits: int | None = None,
    purge_periods: int = 0,
    embargo_periods: int = 0,
    periods_per_year: int = TRADING_DAYS,
    max_combinations: int = 13_000,
) -> CPCVResult:
    """Estimate PBO with contiguous Combinatorial Purged Cross-Validation.

    Candidate net returns are arranged ``T x M``. For every choice of test blocks,
    select the best candidate on the remaining *purged and embargoed* observations,
    rank it across candidates on held-out blocks, and set ``PBO=P(logit(rank)<=0)``.
    A zero purge/embargo and half the blocks held out reduces to symmetric CSCV.

    This return-matrix implementation evaluates already-defined candidate strategies;
    callers that refit models must generate each candidate's returns using the same
    point-in-time purge before passing the matrix here.
    """
    frame = pd.DataFrame(candidate_returns).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    T, M = frame.shape
    test_splits = n_splits // 2 if n_test_splits is None else int(n_test_splits)
    if n_splits < 4 or not (1 <= test_splits < n_splits):
        raise ValueError("n_splits must be >= 4 and n_test_splits must be in [1, n_splits)")
    if purge_periods < 0 or embargo_periods < 0:
        raise ValueError("purge_periods and embargo_periods must be >= 0")
    n_combinations = math.comb(n_splits, test_splits)
    if n_combinations > max_combinations:
        raise ValueError(f"C({n_splits},{test_splits})={n_combinations} exceeds max_combinations={max_combinations}")
    if M < 2 or T < n_splits * 2:
        return CPCVResult(float("nan"), int(M), n_splits, test_splits, n_combinations,
                          int(purge_periods), int(embargo_periods), [], float("nan"), True,
                          f"insufficient data for CPCV: need M>=2 and T>=2*S (M={M}, T={T}, S={n_splits})")

    values = frame.to_numpy(dtype=float)
    groups = np.array_split(np.arange(T), n_splits)
    logits: list[float] = []
    oos_selected: list[float] = []
    for test_groups in combinations(range(n_splits), test_splits):
        test_mask = np.zeros(T, dtype=bool)
        for group in test_groups:
            test_mask[groups[group]] = True
        train_mask = ~test_mask
        for group in test_groups:
            start, end = int(groups[group][0]), int(groups[group][-1]) + 1
            train_mask[max(0, start - purge_periods):start] = False
            train_mask[end:min(T, end + embargo_periods)] = False
        if train_mask.sum() < 2 or test_mask.sum() < 2:
            continue
        is_sharpes = np.asarray([
            sharpe_per_period(values[train_mask, column]) * math.sqrt(periods_per_year)
            for column in range(M)
        ])
        oos_sharpes = np.asarray([
            sharpe_per_period(values[test_mask, column]) * math.sqrt(periods_per_year)
            for column in range(M)
        ])
        selected = int(np.argmax(is_sharpes))
        rank = float(stats.rankdata(oos_sharpes, method="average")[selected]) / (M + 1.0)
        rank = min(max(rank, 1e-9), 1.0 - 1e-9)
        logits.append(float(math.log(rank / (1.0 - rank))))
        oos_selected.append(float(oos_sharpes[selected]))

    if not logits:
        return CPCVResult(float("nan"), int(M), n_splits, test_splits, n_combinations,
                          int(purge_periods), int(embargo_periods), [], float("nan"), True,
                          "purging and embargo leave no valid CPCV train/test partitions")
    return CPCVResult(
        pbo=float(np.mean(np.asarray(logits) <= 0.0)), n_strategies=int(M),
        n_splits=n_splits, n_test_splits=test_splits, n_combinations=n_combinations,
        purge_periods=int(purge_periods), embargo_periods=int(embargo_periods), logits=logits,
        median_oos_sharpe_selected=float(np.median(oos_selected)), insufficient=False,
        note="CPCV PBO uses contiguous blocks with an explicit pre-test purge and post-test embargo.",
    )


def _stationary_bootstrap_indices(n_obs: int, expected_block_length: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices, preserving short-run dependence."""
    if n_obs < 1 or expected_block_length <= 1.0:
        raise ValueError("n_obs must be positive and expected_block_length must exceed 1")
    restart_probability = 1.0 / expected_block_length
    out = np.empty(n_obs, dtype=int)
    out[0] = int(rng.integers(n_obs))
    for i in range(1, n_obs):
        out[i] = int(rng.integers(n_obs)) if rng.random() < restart_probability else (out[i - 1] + 1) % n_obs
    return out


def white_reality_check(
    candidate_returns,
    benchmark_returns=None,
    n_bootstrap: int = 1_000,
    expected_block_length: float = 10.0,
    seed: int = 0,
) -> dict:
    """White's Reality Check using a stationary bootstrap over candidate excess returns.

    Let ``d_t,m = r_t,m - r_t,benchmark``. The observed statistic is
    ``sqrt(T) * max_m(mean(d_m))``. Under the null all candidate means are zero, so
    bootstrap samples are drawn from ``d - mean(d)`` with stationary blocks. The
    p-value is the fraction of bootstrap maxima at least as large as observed.
    """
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    frame = pd.DataFrame(candidate_returns).apply(pd.to_numeric, errors="coerce")
    if benchmark_returns is not None:
        benchmark = pd.Series(benchmark_returns, dtype=float).reindex(frame.index)
        frame = frame.sub(benchmark, axis=0)
    frame = frame.dropna(how="any")
    T, M = frame.shape
    if T < 10 or M < 1:
        return {"p_value": None, "observed_statistic": None, "n_observations": int(T),
                "n_candidates": int(M), "insufficient": True, "computed": True,
                "note": "need at least 10 aligned observations and one candidate for White's Reality Check"}
    values = frame.to_numpy(dtype=float)
    observed = float(math.sqrt(T) * np.max(values.mean(axis=0)))
    centred = values - values.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    bootstrap_stats = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(T, expected_block_length, rng)
        bootstrap_stats[draw] = math.sqrt(T) * np.max(centred[idx].mean(axis=0))
    p_value = float((1 + np.sum(bootstrap_stats >= observed)) / (n_bootstrap + 1))
    return {
        "p_value": p_value, "observed_statistic": observed, "n_observations": int(T),
        "n_candidates": int(M), "n_bootstrap": int(n_bootstrap),
        "expected_block_length": float(expected_block_length), "insufficient": False, "computed": True,
        "note": "A low p-value rejects the hypothesis that the best candidate's mean excess return is data-mining noise.",
    }


def statistical_rigor_summary(
    returns,
    ic_values=None,
    candidate_returns=None,
    periods_per_year: int = TRADING_DAYS,
    hac_lag: int | None = None,
    white_n_bootstrap: int = 250,
    include_white_reality_check: bool = False,
) -> dict:
    """Compose Module 4 diagnostics; optional inputs unlock IC and search tests.

    CPCV is light enough for an interactive tearsheet. White's stationary bootstrap
    is intentionally opt-in because it is a heavier research-grade resampling test;
    call with ``include_white_reality_check=True`` or use ``white_reality_check``
    directly with its 1,000-draw default for an offline report.
    """
    r = _clean(returns)
    sr_pp = sharpe_per_period(r)
    skew = float(stats.skew(r, bias=False)) if r.size > 2 else 0.0
    kurtosis = float(stats.kurtosis(r, fisher=False, bias=False)) if r.size > 3 else 3.0
    # A constant return series makes the Sharpe (and its moments) 0/0. MinTRL
    # rightly rejects non-finite input, but that must surface as a named
    # diagnostic here rather than raising and discarding the whole report.
    degenerate = (
        "The return series is constant across the sample, so its Sharpe ratio and "
        "higher moments are undefined."
        if r.size > 1 and float(np.var(r, ddof=1)) <= 0.0
        else None
    )
    mintrl_inputs_valid = all(np.isfinite(v) for v in (sr_pp, skew, kurtosis))
    out = {
        "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(r, 0.0),
        "sharpe_per_period": sr_pp,
        "hac_lag": default_hac_lag(r.size) if hac_lag is None else int(hac_lag),
        "hac_sharpe_tstat": hac_sharpe_tstat(r, max_lag=hac_lag),
        "minimum_track_record_length_95": (
            minimum_track_record_length(sr_pp, confidence=0.95, skew=skew, kurtosis=kurtosis)
            if mintrl_inputs_valid else None
        ),
        "ic_hac_tstat": hac_ic_tstat(ic_values, max_lag=hac_lag) if ic_values is not None else None,
    }
    diagnostics: list[dict] = []
    if not mintrl_inputs_valid:
        diagnostics.append({
            "statistic": "minimum_track_record_length_95",
            "status": "undefined",
            "reason": degenerate or (
                "The Sharpe ratio or its higher moments are not finite for this sample."
            ),
        })
    out["statistic_diagnostics"] = diagnostics
    out["has_undefined_statistics"] = bool(diagnostics)
    if degenerate:
        out["degenerate_returns_note"] = (
            degenerate + " Statistics derived from it are undefined, not zero, and this sample "
            "cannot support a performance claim."
        )
    if candidate_returns is not None:
        cpcv = purged_cpcv_pbo(candidate_returns, periods_per_year=periods_per_year)
        out["cpcv_pbo"] = cpcv.as_dict()
        out["white_reality_check"] = (
            white_reality_check(candidate_returns, n_bootstrap=white_n_bootstrap)
            if include_white_reality_check else {
                "computed": False,
                "note": "White's Reality Check is available for the candidate matrix; run it in the deep report to avoid delaying an interactive backtest.",
            }
        )
    else:
        out["cpcv_pbo"] = None
        out["white_reality_check"] = None
    return out


def chart_specs() -> dict:
    """Library-agnostic Module 4 chart specifications with exact transforms."""
    return {
        "pbo_logit_histogram": {
            "type": "histogram",
            "transform": "for each CPCV split, logit(rank of IS-selected candidate in OOS); bin equal-width logits",
            "x": "logit of out-of-sample rank", "y": "number of CPCV splits",
            "annotations": ["vertical zero line: below-median OOS", "PBO percentage"],
        },
        "hac_comparison": {
            "type": "grouped_bar",
            "transform": "compute naive mean t-stat and Bartlett-kernel HAC mean t-stat on identical return or IC series",
            "x": "metric (Sharpe proxy, IC)", "y": "t-statistic",
            "annotations": ["HAC lag", "two-sided 5% reference lines at -1.96 and +1.96"],
        },
    }
