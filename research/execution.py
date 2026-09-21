"""Conservative daily execution stress model using only prior-day liquidity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionModel:
    base_cost_bps: float = 5.0
    spread_bps: float = 2.0
    impact_bps: float = 12.0
    short_borrow_bps: float = 50.0
    capital: float = 1_000_000.0
    max_participation: float = 0.05
    adv_lookback: int = 20

    def validate(self) -> None:
        if any(v < 0 for v in (self.base_cost_bps, self.spread_bps, self.impact_bps, self.short_borrow_bps)):
            raise ValueError("execution cost assumptions cannot be negative")
        if self.capital <= 0:
            raise ValueError("portfolio capital must be positive")
        if not (0 < self.max_participation <= 1):
            raise ValueError("maximum ADV participation must be in (0, 1]")
        if not (5 <= self.adv_lookback <= 126):
            raise ValueError("ADV lookback must be in [5, 126]")


@dataclass
class ExecutionResult:
    positions: pd.DataFrame
    costs: pd.Series
    turnover: pd.Series
    participation: pd.DataFrame
    unfilled_turnover: pd.Series
    liquidity_breaches: int
    unknown_liquidity_trades: int

    def audit(self) -> dict:
        known = self.participation.stack().dropna()
        return {
            "liquidity_model": "prior-day rolling ADV", "avg_turnover": float(self.turnover.mean()),
            "unfilled_turnover": float(self.unfilled_turnover.sum()), "liquidity_breaches": int(self.liquidity_breaches),
            "unknown_liquidity_trades": int(self.unknown_liquidity_trades),
            "max_participation": float(known.max()) if not known.empty else None,
            "median_participation": float(known.median()) if not known.empty else None,
            "total_cost": float(self.costs.sum()),
        }


def simulate_execution(target_positions: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame | None, model: ExecutionModel) -> ExecutionResult:
    """Partially fill a target book where orders exceed prior rolling ADV capacity.

    An order with no prior ADV is deliberately left unfilled.  Missing data is
    not evidence that a stock can absorb an arbitrary order.
    """
    model.validate()
    target = target_positions.reindex_like(close).fillna(0.0)
    if volume is None:
        adv = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    else:
        dollar_volume = close * volume.reindex_like(close)
        adv = dollar_volume.shift(1).rolling(model.adv_lookback, min_periods=min(5, model.adv_lookback)).mean()
    positions = pd.DataFrame(0.0, index=target.index, columns=target.columns)
    participation = pd.DataFrame(np.nan, index=target.index, columns=target.columns)
    costs, turnover, unfilled = [], [], []
    previous = np.zeros(target.shape[1], dtype=float)
    breaches = unknown = 0
    for i, (date, desired_row) in enumerate(target.iterrows()):
        delta = desired_row.to_numpy(dtype=float) - previous
        trade_abs = np.abs(delta)
        adv_values = adv.loc[date].to_numpy(dtype=float)
        valid_adv = np.isfinite(adv_values) & (adv_values > 0)
        part = np.full_like(trade_abs, np.nan, dtype=float)
        part[valid_adv] = trade_abs[valid_adv] * model.capital / adv_values[valid_adv]
        participation.iloc[i] = part
        fill_ratio = np.ones_like(trade_abs)
        fill_ratio[(trade_abs > 1e-12) & ~valid_adv] = 0.0
        over = valid_adv & (part > model.max_participation)
        fill_ratio[over] = model.max_participation / part[over]
        breaches += int(over.sum())
        unknown += int(((trade_abs > 1e-12) & ~valid_adv).sum())
        executed_delta = delta * fill_ratio
        actual = previous + executed_delta
        impact_rate = model.impact_bps / 1e4 * np.sqrt(np.nan_to_num(part, nan=0.0, posinf=0.0))
        trade_rate = (model.base_cost_bps + model.spread_bps) / 1e4 + impact_rate
        trade_cost = float((np.abs(executed_delta) * trade_rate).sum())
        borrow_cost = float(np.abs(np.minimum(actual, 0.0)).sum() * model.short_borrow_bps / 1e4 / 252.0)
        positions.iloc[i] = actual
        costs.append(trade_cost + borrow_cost)
        turnover.append(float(np.abs(executed_delta).sum()))
        unfilled.append(float(np.abs(delta - executed_delta).sum()))
        previous = actual
    return ExecutionResult(
        positions=positions, costs=pd.Series(costs, index=target.index, name="execution_cost"),
        turnover=pd.Series(turnover, index=target.index, name="turnover"), participation=participation,
        unfilled_turnover=pd.Series(unfilled, index=target.index, name="unfilled_turnover"),
        liquidity_breaches=breaches, unknown_liquidity_trades=unknown,
    )
