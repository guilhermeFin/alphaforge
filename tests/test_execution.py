import numpy as np
import pandas as pd

from research.backtest import backtest
from research.execution import ExecutionModel, simulate_execution


def _panel():
    index = pd.bdate_range("2024-01-02", periods=30)
    close = pd.DataFrame({"A": np.linspace(100, 110, len(index)), "B": np.linspace(100, 95, len(index))}, index=index)
    volume = pd.DataFrame({"A": 1_000.0, "B": 1_000.0}, index=index)
    return close, volume


def test_execution_partially_fills_when_trade_exceeds_prior_adv():
    close, volume = _panel()
    target = pd.DataFrame({"A": 0.8, "B": -0.8}, index=close.index)
    result = simulate_execution(target, close, volume, ExecutionModel(capital=10_000_000, max_participation=0.01, adv_lookback=5))
    assert result.liquidity_breaches > 0
    assert result.unfilled_turnover.sum() > 0
    assert abs(result.positions.iloc[-1, 0]) < 0.8
    assert result.costs.sum() > 0


def test_unknown_volume_is_disclosed_not_assumed_liquid():
    close, _volume = _panel()
    target = pd.DataFrame({"A": 0.2, "B": -0.2}, index=close.index)
    result = simulate_execution(target, close, None, ExecutionModel())
    assert result.unknown_liquidity_trades > 0
    assert result.audit()["max_participation"] is None


def test_backtest_with_execution_keeps_one_bar_signal_lag():
    close, volume = _panel()
    signal = pd.DataFrame({"A": 0.4, "B": -0.4}, index=close.index)
    result = backtest(close, signal, volume=volume, execution_model=ExecutionModel(adv_lookback=5))
    assert (result.positions.iloc[0] == 0.0).all()
    assert result.execution_audit is not None
