import numpy as np
import pandas as pd
import pytest

from research import data, factors
from research.walkforward import split_backtest, walk_forward


def test_overfit_signal_is_flagged():
    """A signal that is hand-fitted in-sample (perfect next-day foresight) but
    useless out-of-sample must trip the overfit warning."""
    panel = data.make_synthetic_panel(["A"], periods=600, seed=4)
    close = panel.close["A"]
    rets = close.pct_change(fill_method=None)
    cut = int(len(close) * 0.7)

    signal = pd.Series(0.0, index=close.index)
    # in-sample: cheat with tomorrow's return sign (look-ahead) -> huge IS sharpe
    signal.iloc[:cut] = np.sign(rets.shift(-1)).iloc[:cut].fillna(0.0)
    # out-of-sample: no edge
    signal.iloc[cut:] = 0.0

    res = split_backtest(close, signal, split=0.7)
    assert res["in_sample_sharpe"] > res["out_sample_sharpe"]
    assert res["overfit_warning"] is True


def test_walk_forward_runs_and_reports():
    panel = data.make_synthetic_panel([f"S{i}" for i in range(10)], periods=800, seed=5)
    score = factors.cross_sectional_zscore(factors.momentum(panel.close))
    w = factors.long_short_weights(score)
    wf = walk_forward(panel.close, w, n_splits=5)
    assert wf["n_splits"] == 5
    assert len(wf["fold_oos_sharpes"]) == 4
    assert "deflated_sr" in wf and "passes" in wf


def _lookahead_signal(close):
    """A signal whose ONLY edge is a 1-bar look-ahead: position = sign of the
    NEXT bar's return. This is contamination that lives entirely on the bar
    immediately before each realised return, so flattening the leading rows of
    a fold removes the cheating that would otherwise leak across the boundary."""
    rets = close.pct_change(fill_method=None)
    # next-day return sign, as the held weight (long if tomorrow is up)
    return np.sign(rets.shift(-1)).fillna(0.0)


def test_purging_reduces_lookahead_sharpe():
    """Purge+embargo must strip contaminated leading bars, lowering the stitched
    OOS Sharpe of a pure look-ahead signal, and the audit total must be exact."""
    panel = data.make_synthetic_panel(["A"], periods=600, seed=7)
    close = panel.close["A"]
    sig = _lookahead_signal(close)

    base = walk_forward(close, sig, n_splits=5)
    purged = walk_forward(close, sig, n_splits=5, purge_bars=2, embargo_bars=1)

    assert purged["stitched_oos_sharpe"] < base["stitched_oos_sharpe"]
    assert purged["total_purged_bars"] == purged["n_oos_folds"] * 3


def test_default_call_reproduces_legacy_output():
    """purge_bars=0, embargo_bars=0 (the defaults) must reproduce the legacy
    un-purged metrics byte-for-byte."""
    panel = data.make_synthetic_panel([f"S{i}" for i in range(10)], periods=800, seed=5)
    score = factors.cross_sectional_zscore(factors.momentum(panel.close))
    w = factors.long_short_weights(score)

    explicit_zero = walk_forward(panel.close, w, n_splits=5, purge_bars=0, embargo_bars=0)
    legacy = walk_forward(panel.close, w, n_splits=5)

    assert explicit_zero["fold_oos_sharpes"] == legacy["fold_oos_sharpes"]
    assert explicit_zero["stitched_oos_sharpe"] == legacy["stitched_oos_sharpe"]
    assert explicit_zero["deflated_sr"] == legacy["deflated_sr"]
    assert explicit_zero["passes"] == legacy["passes"]
    # And the new audit fields confirm nothing was purged.
    assert legacy["purge_bars"] == 0
    assert legacy["embargo_bars"] == 0
    assert legacy["total_purged_bars"] == 0


def test_purge_too_large_raises():
    """A lead >= the smallest fold length is a configuration error."""
    panel = data.make_synthetic_panel(["A"], periods=120, seed=3)
    close = panel.close["A"]
    sig = _lookahead_signal(close)
    # 5 splits over 120 bars => folds of ~24 bars; a lead of 30 overflows.
    with pytest.raises(ValueError, match="smallest fold"):
        walk_forward(close, sig, n_splits=5, purge_bars=20, embargo_bars=10)


def test_audit_fields_present_and_consistent():
    """All purge/embargo audit keys present; total == sum(per_fold); legacy keys kept."""
    panel = data.make_synthetic_panel([f"S{i}" for i in range(8)], periods=500, seed=11)
    score = factors.cross_sectional_zscore(factors.momentum(panel.close))
    w = factors.long_short_weights(score)

    wf = walk_forward(panel.close, w, n_splits=5, purge_bars=3, embargo_bars=2)

    for key in ("purge_bars", "embargo_bars", "purged_bars_per_fold", "total_purged_bars"):
        assert key in wf
    # legacy keys still present
    for key in ("n_splits", "n_oos_folds", "fold_oos_sharpes", "mean_oos_sharpe",
                "stitched_oos_sharpe", "deflated_sr", "passes"):
        assert key in wf

    assert wf["purge_bars"] == 3
    assert wf["embargo_bars"] == 2
    assert len(wf["purged_bars_per_fold"]) == wf["n_oos_folds"]
    assert all(b == 5 for b in wf["purged_bars_per_fold"])
    assert wf["total_purged_bars"] == sum(wf["purged_bars_per_fold"])
