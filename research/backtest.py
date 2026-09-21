"""The honest backtester — THE MOAT.

Three non-negotiable honesty guards, enforced here and verified in tests/:
  1. NO LOOK-AHEAD: the signal is shifted by ``execution_lag`` before it trades.
     The position held over the (t-1 → t) return was decided using only data up
     to t-1. tests/test_backtest.py proves point-in-time consistency: truncating
     the data does not change any earlier point of the equity curve.
  2. COSTS ON TURNOVER: every change in position pays ``cost_bps`` of frictions.
  3. HONEST SCORING: results are summarised with fat-tail-/multiple-testing-aware
     statistics (see metrics.summarize), never a naked Sharpe.

Works for a single asset (Series signal/close) or a cross-sectional portfolio
(DataFrame of target weights / close), in which case per-name P&L is summed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import metrics
from .execution import ExecutionModel, simulate_execution

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series           # net per-period strategy returns
    gross_returns: pd.Series
    positions: pd.DataFrame | pd.Series
    cost_bps: float
    periods_per_year: int
    costs: pd.Series | None = None
    turnover: pd.Series | None = None
    execution_audit: dict | None = None

    def summary(self, n_trials: int = 1) -> dict:
        return metrics.summarize(
            self.returns,
            positions=self.positions,
            periods_per_year=self.periods_per_year,
            n_trials=n_trials,
        )


def backtest(
    close: pd.DataFrame | pd.Series,
    signal: pd.DataFrame | pd.Series,
    cost_bps: float = 5.0,
    execution_lag: int = 1,
    periods_per_year: int = TRADING_DAYS,
    volume: pd.DataFrame | None = None,
    execution_model: ExecutionModel | None = None,
) -> BacktestResult:
    """Run a point-in-time-safe backtest.

    Parameters
    ----------
    close : prices, DateTime-indexed. Series (one asset) or DataFrame (panel).
    signal : target position / weight, same shape & index as ``close``.
             For a panel, pass dollar-neutral weights (see factors.long_short_weights).
    cost_bps : transaction cost in basis points charged on turnover.
    execution_lag : bars between signal and execution (>=1 ⇒ no look-ahead).
    """
    if execution_lag < 1:
        raise ValueError("execution_lag must be >= 1 to avoid look-ahead bias")

    signal = signal.reindex_like(close) if hasattr(signal, "reindex_like") else signal
    # Position actually held: the signal known `execution_lag` bars ago.
    target_pos = signal.shift(execution_lag).fillna(0.0)
    rets = close.pct_change(fill_method=None).fillna(0.0)

    if execution_model is not None:
        if not isinstance(close, pd.DataFrame) or not isinstance(target_pos, pd.DataFrame):
            raise ValueError("liquidity execution simulation requires a multi-asset price panel")
        executed = simulate_execution(target_pos, close, volume, execution_model)
        pos, costs, turnover, execution_audit = executed.positions, executed.costs, executed.turnover, executed.audit()
        gross = (pos * rets).sum(axis=1)
    elif isinstance(close, pd.DataFrame):
        pos = target_pos
        gross = (pos * rets).sum(axis=1)
        turnover = pos.diff().abs().sum(axis=1).fillna(0.0)
        costs = turnover * (cost_bps / 1e4)
        execution_audit = None
    else:
        pos = target_pos
        gross = pos * rets
        turnover = pos.diff().abs().fillna(0.0)
        costs = turnover * (cost_bps / 1e4)
        execution_audit = None
    net = gross - costs
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        equity=equity,
        returns=net,
        gross_returns=gross,
        positions=pos,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        costs=costs,
        turnover=turnover,
        execution_audit=execution_audit,
    )
