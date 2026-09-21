"""Volatility and downside-risk analytics for a strategy tearsheet.

All functions consume one-period arithmetic returns.  ``mar`` and ``threshold``
are therefore per-period rates; callers using daily returns should pass daily
values.  Annualisation is applied only where the formula explicitly calls for it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import TRADING_DAYS, annualised_return, equity_curve, max_drawdown


def _clean(returns) -> np.ndarray:
    values = np.asarray(pd.Series(returns), dtype=float)
    return values[np.isfinite(values)]


def annualized_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized volatility: ``std(r, ddof=1) * sqrt(periods_per_year)``."""
    r = _clean(returns)
    return float(r.std(ddof=1) * math.sqrt(periods_per_year)) if r.size > 1 else 0.0


def downside_deviation(
    returns,
    mar: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    annualize: bool = False,
) -> float:
    """Target semi-deviation below a minimum acceptable return (MAR).

    ``DD = sqrt(mean(min(0, r_t - MAR)^2))``.  Zero-return / above-MAR periods
    remain in the denominator, which is the standard target-downside convention.
    Set ``annualize=True`` to multiply the per-period result by ``sqrt(P)``.
    """
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    shortfall = np.minimum(r - float(mar), 0.0)
    value = float(np.sqrt(np.mean(shortfall ** 2)))
    return float(value * math.sqrt(periods_per_year)) if annualize else value


def sortino_ratio(
    returns,
    mar: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sortino ratio: ``mean(r_t - MAR) / DD(MAR) * sqrt(P)``."""
    r = _clean(returns)
    dd = downside_deviation(r, mar=mar)
    if r.size < 2 or not np.isfinite(dd) or dd <= 0:
        return float("nan")
    return float(np.mean(r - float(mar)) / dd * math.sqrt(periods_per_year))


def underwater_curve(returns) -> pd.DataFrame:
    """Return equity, high-water mark, drawdown, and time-under-water per bar.

    Transform: ``equity=cumprod(1+r)``, ``drawdown=equity/cummax(equity)-1``;
    time-under-water is the consecutive count of strictly negative drawdowns.
    """
    r = pd.Series(returns, dtype=float).fillna(0.0)
    equity = equity_curve(r)
    high_water = equity.cummax()
    drawdown = equity / high_water - 1.0
    duration, current = [], 0
    for value in drawdown.to_numpy(dtype=float):
        current = current + 1 if value < 0.0 else 0
        duration.append(current)
    return pd.DataFrame(
        {"equity": equity, "high_water_mark": high_water, "drawdown": drawdown,
         "time_under_water_periods": duration},
        index=r.index,
    )


def ulcer_index(returns) -> float:
    """Ulcer Index: ``sqrt(mean(drawdown_t^2))`` using the compounded equity path."""
    dd = underwater_curve(returns)["drawdown"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(dd ** 2))) if dd.size else float("nan")


def pain_index(returns) -> float:
    """Pain Index: ``mean(abs(drawdown_t))`` using the compounded equity path."""
    dd = underwater_curve(returns)["drawdown"].to_numpy(dtype=float)
    return float(np.mean(np.abs(dd))) if dd.size else float("nan")


def calmar_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Calmar ratio: ``CAGR / abs(max_drawdown)``; undefined without a drawdown."""
    cagr = annualised_return(returns, periods_per_year)
    drawdown = abs(max_drawdown(equity_curve(returns)))
    return float(cagr / drawdown) if drawdown > 0 else float("nan")


def mar_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """MAR ratio, an explicit alias for the Calmar definition used by AlphaForge."""
    return calmar_ratio(returns, periods_per_year)


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega ratio at ``threshold``: ``sum(max(r-theta, 0)) / sum(max(theta-r, 0))``.

    Returns ``inf`` when there are gains above the threshold but no losses below it;
    this is mathematically meaningful, not silently replaced with a large number.
    """
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    gains = float(np.maximum(r - threshold, 0.0).sum())
    losses = float(np.maximum(threshold - r, 0.0).sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return gains / losses


def annual_max_drawdowns(returns, periods_per_year: int = TRADING_DAYS) -> np.ndarray:
    """Maximum drawdown in each contiguous annual-sized block (last block retained)."""
    r = _clean(returns)
    if r.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray([
        abs(max_drawdown(equity_curve(r[start:start + periods_per_year])))
        for start in range(0, r.size, periods_per_year)
    ], dtype=float)


def sterling_ratio(
    returns,
    periods_per_year: int = TRADING_DAYS,
    annual_drawdown_adjustment: float = 0.10,
) -> float:
    """Modified Sterling ratio with an explicit, reproducible convention.

    ``Sterling = CAGR / mean(max(annual_max_drawdown - adjustment, 0))``.
    AlphaForge defaults to the traditional 10% annual drawdown adjustment.  Industry
    sources use several incompatible Sterling variants; exposing this convention and
    parameter prevents a label from hiding a different denominator.
    """
    if annual_drawdown_adjustment < 0:
        raise ValueError("annual_drawdown_adjustment must be >= 0")
    annual_dd = annual_max_drawdowns(returns, periods_per_year)
    if annual_dd.size == 0:
        return float("nan")
    denominator = float(np.mean(np.maximum(annual_dd - annual_drawdown_adjustment, 0.0)))
    return float(annualised_return(returns, periods_per_year) / denominator) if denominator > 0 else float("nan")


def k_ratio(returns) -> float:
    """K-ratio: OLS slope of log equity divided by the slope's standard error.

    Regress ``log(equity_t) = a + b*t + e_t`` and return ``b / SE(b)``.  A higher
    positive value means a steadier compounding path, not simply a higher endpoint.
    """
    equity = equity_curve(returns)
    if equity.size < 3 or (equity <= 0).any():
        return float("nan")
    fit = stats.linregress(np.arange(equity.size, dtype=float), np.log(equity.to_numpy(dtype=float)))
    if not np.isfinite(fit.stderr) or fit.stderr <= 1e-12:
        return float("nan")
    return float(fit.slope / fit.stderr)


def downside_risk_summary(returns, periods_per_year: int = TRADING_DAYS, mar: float = 0.0) -> dict:
    """Compose Module 1 metrics into a JSON-friendly tearsheet section."""
    underwater = underwater_curve(returns)
    return {
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "downside_deviation": downside_deviation(returns, mar=mar),
        "annualized_downside_deviation": downside_deviation(
            returns, mar=mar, periods_per_year=periods_per_year, annualize=True),
        "mar_per_period": float(mar),
        "sortino_ratio": sortino_ratio(returns, mar=mar, periods_per_year=periods_per_year),
        "ulcer_index": ulcer_index(returns),
        "pain_index": pain_index(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "mar_ratio": mar_ratio(returns, periods_per_year),
        "omega_ratio": omega_ratio(returns, threshold=mar),
        "sterling_ratio": sterling_ratio(returns, periods_per_year),
        "k_ratio": k_ratio(returns),
        "max_time_under_water_periods": int(underwater["time_under_water_periods"].max()) if not underwater.empty else 0,
    }


def chart_specs() -> dict:
    """Library-agnostic Module 1 chart specifications with exact transforms."""
    return {
        "underwater": {
            "type": "area_time_series",
            "transform": "equity=cumprod(1+r); drawdown=equity/cummax(equity)-1; duration=consecutive(drawdown<0)",
            "x": "return timestamp", "y": "drawdown as a percentage",
            "annotations": ["zero line", "deepest drawdown", "maximum time under water"],
        },
        "drawdown_distribution": {
            "type": "histogram",
            "transform": "local drawdown episode depths from the underwater series",
            "x": "episode maximum drawdown as a percentage", "y": "number of episodes",
            "annotations": ["mean episode depth", "worst episode"],
        },
    }
