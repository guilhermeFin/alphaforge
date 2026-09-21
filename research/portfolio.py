"""Constrained, auditable portfolio construction for research runs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import allocation


REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")
ALLOCATION_METHODS = ("score_weighted", "equal_weight", "inverse_volatility", "hierarchical_risk_parity")


def _bounded_proportional(values: np.ndarray, budget: float, cap: float) -> np.ndarray:
    """Allocate a side budget proportionally without exceeding a name cap."""
    out = np.zeros(len(values), dtype=float)
    active = np.isfinite(values) & (values > 0)
    remaining = float(min(budget, cap * int(active.sum())))
    while remaining > 1e-12 and active.any():
        total = float(values[active].sum())
        if total <= 0:
            break
        room = cap - out[active]
        add = np.minimum(values[active] / total * remaining, room)
        out[active] += add
        used = float(add.sum())
        indices = np.flatnonzero(active)
        active[indices[room - add <= 1e-12]] = False
        remaining -= used
        if used <= 1e-12:
            break
    return out


def constrained_long_short_weights(
    score: pd.DataFrame, *, gross: float = 1.0, max_name_weight: float = 0.10,
) -> pd.DataFrame:
    """Convert scores into a capped, dollar-neutral book.

    Where too few names exist to honour the cap, the book remains underinvested;
    it never defeats its own concentration control to spend all gross exposure.
    """
    if gross <= 0:
        raise ValueError("gross exposure must be positive")
    if not (0 < max_name_weight <= gross):
        raise ValueError("max_name_weight must be in (0, gross]")
    raw = score.astype(float).replace([np.inf, -np.inf], np.nan)
    result = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for date, row in raw.iterrows():
        values = row.to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue
        centered = np.zeros_like(values)
        centered[valid] = values[valid] - float(values[valid].mean())
        positives, negatives = np.maximum(centered, 0.0), np.maximum(-centered, 0.0)
        n_long, n_short = int((positives > 0).sum()), int((negatives > 0).sum())
        if not n_long or not n_short:
            continue
        side_budget = min(gross / 2.0, max_name_weight * n_long, max_name_weight * n_short)
        result.loc[date] = _bounded_proportional(positives, side_budget, max_name_weight) - _bounded_proportional(
            negatives, side_budget, max_name_weight
        )
    return result


def _hrp_long_short_weights(
    score: pd.DataFrame,
    close: pd.DataFrame,
    *,
    gross: float,
    max_name_weight: float,
    volatility_lookback: int,
) -> pd.DataFrame:
    """Risk-cluster each side using only prior return history, then cap it."""
    raw = score.astype(float).replace([np.inf, -np.inf], np.nan)
    prior_returns = close.pct_change(fill_method=None).shift(1)
    result = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    min_history = max(20, min(volatility_lookback, 63))
    for position, (date, row) in enumerate(raw.iterrows()):
        values = row.to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue
        centered = np.zeros_like(values)
        centered[valid] = values[valid] - float(values[valid].mean())
        long_positions = np.flatnonzero(centered > 0)
        short_positions = np.flatnonzero(centered < 0)
        if not len(long_positions) or not len(short_positions):
            continue
        side_budget = min(
            gross / 2.0,
            max_name_weight * len(long_positions),
            max_name_weight * len(short_positions),
        )
        history = prior_returns.iloc[max(0, position - volatility_lookback) : position]

        def side_values(items: np.ndarray) -> np.ndarray:
            names = raw.columns[items]
            history_slice = history.loc[:, names].dropna(how="any")
            if len(history_slice) < min_history or len(names) < 2:
                return np.abs(centered[items])
            try:
                weights = allocation.hierarchical_risk_parity(
                    history_slice, min_observations=min_history
                )["weights"].reindex(names)
                return weights.to_numpy(dtype=float)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                return np.abs(centered[items])

        long_values = side_values(long_positions)
        short_values = side_values(short_positions)
        output = np.zeros_like(values)
        output[long_positions] = _bounded_proportional(long_values, side_budget, max_name_weight)
        output[short_positions] = -_bounded_proportional(short_values, side_budget, max_name_weight)
        result.loc[date] = output
    return result


def construct_long_short_weights(
    score: pd.DataFrame, close: pd.DataFrame, *, method: str = "score_weighted", gross: float = 1.0,
    max_name_weight: float = 0.10, volatility_lookback: int = 63,
) -> pd.DataFrame:
    """Choose a transparent portfolio construction rule without changing the signal.

    ``inverse_volatility`` and ``hierarchical_risk_parity`` only use volatility
    known through the prior bar. The latter clusters each side of the book using
    a shrinkage covariance estimate, then retains the same gross, neutrality, and
    single-name limits as every other construction rule.
    """
    if method not in ALLOCATION_METHODS:
        raise ValueError(f"allocation method must be one of {ALLOCATION_METHODS}")
    adjusted = score.reindex_like(close).astype(float)
    if method == "equal_weight":
        ranks = adjusted.rank(axis=1, pct=True)
        adjusted = ranks.where(ranks >= 0.75, 0.0) - (1.0 - ranks).where(ranks <= 0.25, 0.0).fillna(0.0)
    elif method == "inverse_volatility":
        vol = close.pct_change(fill_method=None).shift(1).rolling(volatility_lookback, min_periods=20).std()
        inverse_vol = (1.0 / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        adjusted = adjusted * inverse_vol
    elif method == "hierarchical_risk_parity":
        return _hrp_long_short_weights(
            adjusted,
            close,
            gross=gross,
            max_name_weight=max_name_weight,
            volatility_lookback=volatility_lookback,
        )
    return constrained_long_short_weights(adjusted, gross=gross, max_name_weight=max_name_weight)


def rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """Return pre-known calendar rebalance dates; never infer them from future rows."""
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"rebalance frequency must be one of {REBALANCE_FREQUENCIES}")
    dates = pd.DatetimeIndex(index)
    if frequency == "daily":
        flags = np.ones(len(dates), dtype=bool)
    elif frequency == "weekly":
        flags = dates.dayofweek == 4
    elif frequency == "monthly":
        flags = dates.is_month_end
    else:
        flags = dates.is_quarter_end
    if len(flags):
        flags[0] = True
    return pd.Series(flags, index=index)


def apply_rebalance_schedule(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Hold target weights between scheduled rebalances."""
    return weights.where(rebalance_mask(pd.DatetimeIndex(weights.index), frequency), np.nan).ffill().fillna(0.0)


def limit_turnover(target_weights: pd.DataFrame, max_turnover: float | None) -> pd.DataFrame:
    """Move toward a desired book no faster than an L1 turnover budget per bar."""
    if max_turnover is None:
        return target_weights.copy()
    if max_turnover <= 0:
        raise ValueError("max_turnover must be positive when supplied")
    actual = pd.DataFrame(0.0, index=target_weights.index, columns=target_weights.columns)
    previous = np.zeros(target_weights.shape[1], dtype=float)
    for i, (_, desired) in enumerate(target_weights.iterrows()):
        delta = desired.to_numpy(dtype=float) - previous
        requested = float(np.abs(delta).sum())
        if requested > max_turnover:
            delta *= max_turnover / requested
        previous += delta
        actual.iloc[i] = previous
    return actual


def apply_volatility_ceiling(
    target_weights: pd.DataFrame, close: pd.DataFrame, *, target_annual_vol: float | None, lookback: int = 63,
) -> pd.DataFrame:
    """De-risk using volatility known through the prior bar, without leverage."""
    if target_annual_vol is None:
        return target_weights.copy()
    if not (0 < target_annual_vol <= 1.0):
        raise ValueError("target annual volatility must be in (0, 1]")
    if lookback < 20:
        raise ValueError("volatility lookback must be at least 20 trading days")
    returns = close.pct_change(fill_method=None).fillna(0.0)
    base_returns = (target_weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    realized = base_returns.shift(1).rolling(lookback, min_periods=20).std() * np.sqrt(252.0)
    scale = (target_annual_vol / realized).replace([np.inf, -np.inf], np.nan).clip(upper=1.0).fillna(1.0)
    return target_weights.mul(scale, axis=0)


def portfolio_diagnostics(positions: pd.DataFrame, close: pd.DataFrame | None = None) -> dict:
    """Report realised exposure, concentration and final historical holdings."""
    if positions.empty:
        return {"avg_gross": 0.0, "max_gross": 0.0, "avg_net": 0.0, "avg_turnover": 0.0,
                "max_name_weight": 0.0, "avg_active_names": 0.0, "latest_holdings": []}
    gross, net = positions.abs().sum(axis=1), positions.sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    last = positions.iloc[-1]
    latest = [
        {"symbol": str(symbol), "weight": float(weight)}
        for symbol, weight in last.loc[last.abs().sort_values(ascending=False).index].items()
        if abs(float(weight)) > 1e-10
    ]
    output = {
        "avg_gross": float(gross.mean()), "max_gross": float(gross.max()),
        "avg_net": float(net.mean()), "max_abs_net": float(net.abs().max()),
        "avg_turnover": float(turnover.mean()), "max_name_weight": float(positions.abs().max().max()),
        "avg_active_names": float((positions.abs() > 1e-10).sum(axis=1).mean()),
        "latest_holdings": latest,
    }
    if close is not None and len(close) >= 21:
        returns = close.reindex_like(positions).pct_change(fill_method=None).iloc[-63:].dropna(how="all")
        latest_weights = last.reindex(returns.columns).fillna(0.0).to_numpy(dtype=float)
        covariance = returns.cov().to_numpy(dtype=float) * 252.0
        marginal = covariance @ latest_weights
        contribution = latest_weights * marginal
        total = float(contribution.sum())
        output["latest_risk_contribution"] = [
            {"symbol": str(symbol), "risk_contribution": float(value / total) if abs(total) > 1e-12 else None}
            for symbol, value in zip(returns.columns, contribution) if abs(value) > 1e-12
        ]
    return output
