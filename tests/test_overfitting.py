"""Tests for research.overfitting — CSCV Probability of Backtest Overfitting."""
import math

import numpy as np
import pandas as pd
import pytest

from research import overfitting, metrics
from research.data import make_synthetic_panel
from research import factors


def _noise_matrix(M=30, T=2016, seed=0):
    """M iid-noise return columns, no edge."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(0.01 * rng.standard_normal((T, M)))


def test_pure_noise_pbo_high():
    """(1) M=30 columns of iid noise -> PBO is high (~0.5): selection is a coin flip.

    Any single draw is noisy (CSCV PBO for noise is itself a random variable
    centred near 0.5), so we assert the *mean* over several seeds clears 0.4 —
    the honest theoretical property, robust to an unlucky single draw.
    """
    pbos = []
    for seed in range(8):
        mat = _noise_matrix(M=30, T=2016, seed=seed)
        res = overfitting.cscv_pbo(mat, n_splits=8)
        assert not res.insufficient
        assert res.n_strategies == 30
        pbos.append(res.pbo)
    assert np.mean(pbos) > 0.4


def test_genuine_edge_lowers_pbo():
    """(2) Injecting one real-edge column makes PBO strictly lower than all-noise."""
    seed = 7
    T, M = 2016, 30
    rng = np.random.default_rng(seed)
    noise = 0.01 * rng.standard_normal((T, M))
    all_noise = pd.DataFrame(noise.copy())

    with_edge = noise.copy()
    # one column with a clear positive drift -> dominates IS, persists OOS
    with_edge[:, 0] = 0.01 * rng.standard_normal(T) + 0.004
    with_edge = pd.DataFrame(with_edge)

    pbo_noise = overfitting.cscv_pbo(all_noise, n_splits=8).pbo
    pbo_edge = overfitting.cscv_pbo(with_edge, n_splits=8).pbo
    assert pbo_edge < pbo_noise


def test_pooled_moment_sharpe_matches_metrics():
    """(3) CRITICAL: pooled-moment annualised Sharpe over a random block mask
    equals metrics.annualised_sharpe on the explicitly pooled rows (1e-9)."""
    rng = np.random.default_rng(123)
    n_splits = 8
    block_len = 200
    T = block_len * n_splits
    M = 5
    values = 0.01 * rng.standard_normal((T, M)) + 0.0003
    periods_per_year = 252

    blocks = values.reshape(n_splits, block_len, M)
    block_sum = blocks.sum(axis=1)
    block_sumsq = (blocks * blocks).sum(axis=1)
    block_n = np.full(n_splits, float(block_len))

    # random mask of S/2 blocks
    mask = np.zeros(n_splits, dtype=bool)
    chosen = rng.choice(n_splits, size=n_splits // 2, replace=False)
    mask[chosen] = True

    pooled_sharpe = overfitting._nan_safe_sharpe_from_moments(
        block_sum, block_sumsq, block_n, mask, periods_per_year
    )

    # explicitly pool the rows belonging to the selected blocks
    sel_rows = np.concatenate(
        [np.arange(b * block_len, (b + 1) * block_len) for b in np.where(mask)[0]]
    )
    pooled_rows = values[sel_rows]
    for j in range(M):
        ref = metrics.annualised_sharpe(pooled_rows[:, j], periods_per_year=periods_per_year)
        assert abs(pooled_sharpe[j] - ref) < 1e-9


def test_n_combinations_and_thorough():
    """(4) C(8,4)=70; thorough -> C(16,8)=12870."""
    mat = _noise_matrix(M=10, T=2016, seed=2)
    res = overfitting.cscv_pbo(mat, n_splits=8)
    assert res.n_combinations == math.comb(8, 4) == 70

    res_thorough = overfitting.cscv_pbo(mat, n_splits=8, thorough=True)
    assert res_thorough.n_splits == 16
    assert res_thorough.n_combinations == math.comb(16, 8) == 12870


def test_invalid_n_splits_raise():
    """(5) odd or <4 n_splits raise; n_splits=20 exceeds MAX_COMBINATIONS."""
    mat = _noise_matrix(M=10, T=2016, seed=3)
    with pytest.raises(ValueError):
        overfitting.cscv_pbo(mat, n_splits=7)   # odd
    with pytest.raises(ValueError):
        overfitting.cscv_pbo(mat, n_splits=2)   # < 4
    with pytest.raises(ValueError):
        overfitting.cscv_pbo(mat, n_splits=20)  # C(20,10)=184756 > 13000


def test_single_strategy_insufficient():
    """(6) M=1 -> insufficient True, no exception."""
    rng = np.random.default_rng(4)
    mat = pd.DataFrame(0.01 * rng.standard_normal((2016, 1)))
    res = overfitting.cscv_pbo(mat, n_splits=8)
    assert res.insufficient is True
    assert math.isnan(res.pbo)
    assert res.n_strategies == 1


def test_short_panel_insufficient():
    """T < 2*S -> insufficient True, no crash."""
    rng = np.random.default_rng(5)
    mat = pd.DataFrame(0.01 * rng.standard_normal((10, 5)))  # T=10 < 2*8
    res = overfitting.cscv_pbo(mat, n_splits=8)
    assert res.insufficient is True


def test_deterministic_on_fixed_seed():
    """(7) deterministic: identical input -> identical PBO and logits."""
    mat = _noise_matrix(M=20, T=2016, seed=42)
    a = overfitting.cscv_pbo(mat, n_splits=8)
    b = overfitting.cscv_pbo(mat, n_splits=8)
    assert a.pbo == b.pbo
    assert a.logits == b.logits
    assert a.oos_sharpe_selected == b.oos_sharpe_selected


def test_compact_pbo_jsonable_keys():
    import json

    mat = _noise_matrix(M=15, T=2016, seed=8)
    out = overfitting.compact_pbo(mat, n_splits=8)
    for key in (
        "pbo", "n_strategies", "n_combinations", "median_oos_sharpe_selected",
        "prob_oos_loss", "performance_degradation_slope", "histogram",
        "insufficient", "overfit", "verdict", "note",
    ):
        assert key in out
    assert out["overfit"] == (out["pbo"] > 0.5)
    json.dumps(out)  # must be JSON-serialisable


def test_compact_pbo_insufficient_no_crash():
    rng = np.random.default_rng(9)
    mat = pd.DataFrame(0.01 * rng.standard_normal((2016, 1)))
    out = overfitting.compact_pbo(mat, n_splits=8)
    assert out["insufficient"] is True
    assert "insufficient" in out["verdict"].lower()


def test_pbo_from_factor_grid():
    """End-to-end: parameter grid -> honest backtester -> compact PBO dict."""
    symbols = [f"S{i}" for i in range(12)]
    panel = make_synthetic_panel(symbols, periods=1260, seed=11)
    close = panel.close

    def weight_fn(close, lookback, skip):
        score = factors.cross_sectional_zscore(factors.momentum(close, lookback=lookback, skip=skip))
        return factors.long_short_weights(score)

    grid = [
        {"lookback": lb, "skip": sk}
        for lb in (63, 126, 189, 252)
        for sk in (5, 21)
    ]
    out = overfitting.pbo_from_factor_grid(close, weight_fn, grid, n_splits=8)
    assert out["n_strategies"] == len(grid)
    assert out["n_combinations"] == math.comb(8, 4)
    assert 0.0 <= out["pbo"] <= 1.0
