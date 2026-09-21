"""Tail-risk, distribution, and drawdown-duration analytics for strategy reports."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import equity_curve
from .risk import underwater_curve


def _clean(returns) -> np.ndarray:
    values = np.asarray(pd.Series(returns), dtype=float)
    return values[np.isfinite(values)]


def historical_var(returns, confidence: float = 0.95) -> float:
    """Historical VaR loss: ``-quantile(r, 1-confidence)`` using linear quantiles."""
    r = _clean(returns)
    if r.size == 0 or not 0.0 < confidence < 1.0:
        return float("nan")
    return float(-np.quantile(r, 1.0 - confidence, method="linear"))


def parametric_var(returns, confidence: float = 0.95) -> float:
    """Gaussian VaR loss: ``-(mu + sigma*Phi^-1(1-confidence))``."""
    r = _clean(returns)
    if r.size < 2 or not 0.0 < confidence < 1.0:
        return float("nan")
    return float(-(r.mean() + r.std(ddof=1) * stats.norm.ppf(1.0 - confidence)))


def historical_expected_shortfall(returns, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall: negative mean return at/below the historical VaR threshold."""
    r = _clean(returns)
    if r.size == 0 or not 0.0 < confidence < 1.0:
        return float("nan")
    threshold = np.quantile(r, 1.0 - confidence, method="linear")
    tail = r[r <= threshold]
    return float(-tail.mean()) if tail.size else float("nan")


def parametric_expected_shortfall(returns, confidence: float = 0.95) -> float:
    """Gaussian ES loss: ``-(mu-sigma*phi(z)/(1-confidence))``, ``z=Phi^-1(1-confidence)``."""
    r = _clean(returns)
    if r.size < 2 or not 0.0 < confidence < 1.0:
        return float("nan")
    z = stats.norm.ppf(1.0 - confidence)
    return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1.0 - confidence)))


def tail_ratio(returns) -> float:
    """Tail ratio: 95th percentile gain divided by absolute 5th percentile loss."""
    r = _clean(returns)
    if r.size == 0:
        return float("nan")
    downside = abs(float(np.quantile(r, 0.05, method="linear")))
    upside = float(np.quantile(r, 0.95, method="linear"))
    return upside / downside if downside > 0 else float("nan")


def worst_periods(returns) -> dict:
    """Worst daily, compounded weekly, and compounded monthly return observations."""
    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return {"worst_day": None, "worst_week": None, "worst_month": None}
    def item(values):
        if values.empty:
            return None
        index = values.idxmin()
        return {"date": str(getattr(index, "date", lambda: index)()), "return": float(values.loc[index])}
    output = {"worst_day": item(series)}
    if isinstance(series.index, pd.DatetimeIndex):
        weekly = (1.0 + series).resample("W-FRI").prod() - 1.0
        monthly = (1.0 + series).resample("ME").prod() - 1.0
        output.update({"worst_week": item(weekly), "worst_month": item(monthly)})
    else:
        output.update({"worst_week": None, "worst_month": None})
    return output


def drawdown_duration_distribution(returns) -> list[dict]:
    """Local drawdown episodes with start/end, duration, and deepest depth."""
    curve = underwater_curve(returns)
    if curve.empty:
        return []
    episodes, start, depths = [], None, []
    for index, row in curve.iterrows():
        if row.drawdown < 0:
            if start is None:
                start, depths = index, []
            depths.append(float(row.drawdown))
        elif start is not None:
            episodes.append({"start": str(getattr(start, "date", lambda: start)()),
                             "end": str(getattr(index, "date", lambda: index)()),
                             "duration_periods": len(depths), "max_drawdown": min(depths)})
            start, depths = None, []
    if start is not None:
        last = curve.index[-1]
        episodes.append({"start": str(getattr(start, "date", lambda: start)()),
                         "end": str(getattr(last, "date", lambda: last)()),
                         "duration_periods": len(depths), "max_drawdown": min(depths)})
    return episodes


def distribution_chart_data(returns, bins: int = 30, max_qq_points: int = 250) -> dict:
    """Histogram bins and normal QQ coordinates used by the Module 3 plots."""
    r = _clean(returns)
    if r.size < 2:
        return {"histogram": [], "qq": []}
    counts, edges = np.histogram(r, bins=bins)
    histogram = [{"left": float(edges[i]), "right": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]
    observed = np.sort(r)
    if observed.size > max_qq_points:
        observed = observed[np.linspace(0, observed.size - 1, max_qq_points, dtype=int)]
    probabilities = (np.arange(1, observed.size + 1) - 0.5) / observed.size
    normal = r.mean() + r.std(ddof=1) * stats.norm.ppf(probabilities)
    return {"histogram": histogram,
            "qq": [{"normal_quantile": float(x), "observed_return": float(y)} for x, y in zip(normal, observed)]}


def tail_risk_summary(returns) -> dict:
    """Compose Module 3 risk, distribution, and chart data into one report section."""
    r = _clean(returns)
    episodes = drawdown_duration_distribution(returns)
    return {
        "historical_var_95": historical_var(r, 0.95), "historical_var_99": historical_var(r, 0.99),
        "parametric_var_95": parametric_var(r, 0.95), "parametric_var_99": parametric_var(r, 0.99),
        "historical_expected_shortfall_95": historical_expected_shortfall(r, 0.95),
        "historical_expected_shortfall_99": historical_expected_shortfall(r, 0.99),
        "parametric_expected_shortfall_95": parametric_expected_shortfall(r, 0.95),
        "parametric_expected_shortfall_99": parametric_expected_shortfall(r, 0.99),
        "skewness": float(stats.skew(r, bias=False)) if r.size > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)) if r.size > 3 else float("nan"),
        "tail_ratio": tail_ratio(r), "worst_periods": worst_periods(returns),
        "drawdown_episodes": episodes,
        "max_time_under_water_periods": max((episode["duration_periods"] for episode in episodes), default=0),
        "distribution_chart_data": distribution_chart_data(r),
    }


def chart_specs() -> dict:
    """Library-agnostic Module 3 plotting specifications."""
    return {
        "return_histogram": {
            "type": "histogram_with_normal_density", "transform": "bin daily returns; overlay Normal(mean(r), std(r)) density scaled to bin width and observation count",
            "x": "one-period return", "y": "frequency", "annotations": ["historical VaR 95/99", "mean", "Normal fit"],
        },
        "normal_qq": {
            "type": "scatter", "transform": "sort returns; pair with Normal quantiles at (i-.5)/n using sample mean and standard deviation",
            "x": "fitted Normal quantile", "y": "observed return", "annotations": ["45-degree reference line"],
        },
        "underwater_duration": {
            "type": "area_time_series", "transform": "equity=cumprod(1+r); drawdown=equity/cummax(equity)-1; count consecutive negative drawdowns",
            "x": "return timestamp", "y": "drawdown percentage", "annotations": ["maximum duration", "deepest drawdown"],
        },
    }
