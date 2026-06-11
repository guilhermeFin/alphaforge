import numpy as np
import pandas as pd

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
