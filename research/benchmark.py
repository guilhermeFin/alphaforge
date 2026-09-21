"""Benchmark-relative performance and rolling analytics for strategy reports."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import TRADING_DAYS, annualised_return, annualised_sharpe


def align_returns(strategy_returns, benchmark_returns) -> tuple[pd.Series, pd.Series]:
    """Align finite strategy and benchmark return observations on their common index."""
    strategy = pd.Series(strategy_returns, dtype=float)
    benchmark = pd.Series(benchmark_returns, dtype=float).reindex(strategy.index)
    frame = pd.DataFrame({"strategy": strategy, "benchmark": benchmark}).replace([np.inf, -np.inf], np.nan).dropna()
    return frame["strategy"], frame["benchmark"]


def tracking_error(strategy_returns, benchmark_returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized tracking error: ``std(r_strategy-r_benchmark, ddof=1)*sqrt(P)``."""
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    active = strategy - benchmark
    return float(active.std(ddof=1) * math.sqrt(periods_per_year)) if active.size > 1 else float("nan")


def information_ratio(strategy_returns, benchmark_returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Information Ratio: ``mean(active)/std(active)*sqrt(P)``."""
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    active = strategy - benchmark
    sd = float(active.std(ddof=1)) if active.size > 1 else float("nan")
    return float(active.mean() / sd * math.sqrt(periods_per_year)) if np.isfinite(sd) and sd > 0 else float("nan")


def beta(strategy_returns, benchmark_returns) -> float:
    """OLS beta: ``cov(r_strategy, r_benchmark) / var(r_benchmark)``."""
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    if strategy.size < 2:
        return float("nan")
    variance = float(benchmark.var(ddof=1))
    return float(strategy.cov(benchmark) / variance) if variance > 0 else float("nan")


def jensens_alpha(strategy_returns, benchmark_returns, periods_per_year: int = TRADING_DAYS) -> dict:
    """Jensen alpha from ``r_p = alpha + beta*r_b + epsilon``.

    Returns per-period intercept, arithmetic annualized intercept (``alpha*P``),
    OLS t-statistic, p-value, beta, and R-squared. Inputs should be excess returns
    when a nonzero risk-free rate matters.
    """
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    if strategy.size < 3 or benchmark.var(ddof=1) <= 0:
        return {"alpha_per_period": float("nan"), "alpha_annualized": float("nan"),
                "alpha_tstat": float("nan"), "alpha_pvalue": float("nan"),
                "beta": float("nan"), "r_squared": float("nan"), "n_observations": int(strategy.size)}
    fit = stats.linregress(benchmark.to_numpy(dtype=float), strategy.to_numpy(dtype=float))
    alpha_tstat = float(fit.intercept / fit.intercept_stderr) if fit.intercept_stderr > 0 else float("nan")
    alpha_pvalue = (
        float(2.0 * stats.t.sf(abs(alpha_tstat), df=strategy.size - 2))
        if np.isfinite(alpha_tstat)
        else float("nan")
    )
    return {
        "alpha_per_period": float(fit.intercept),
        "alpha_annualized": float(fit.intercept * periods_per_year),
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "beta": float(fit.slope), "r_squared": float(fit.rvalue ** 2),
        "n_observations": int(strategy.size),
    }


def capture_ratios(strategy_returns, benchmark_returns) -> dict:
    """Up/down capture using arithmetic mean returns on benchmark up/down periods.

    ``up_capture=mean(r_p | r_b>0)/mean(r_b | r_b>0)`` and the analogous ratio
    on ``r_b<0``. A down-capture below 1 means the strategy lost less than its
    benchmark on down periods.
    """
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    up = benchmark > 0
    down = benchmark < 0
    def ratio(mask):
        if int(mask.sum()) == 0:
            return float("nan")
        base = float(benchmark[mask].mean())
        return float(strategy[mask].mean() / base) if base != 0 else float("nan")
    return {"up_capture": ratio(up), "down_capture": ratio(down),
            "n_up_periods": int(up.sum()), "n_down_periods": int(down.sum())}


def rolling_benchmark_metrics(strategy_returns, benchmark_returns, window: int = 63,
                              periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Rolling annualized Sharpe and OLS beta with approximate 95% IID bands.

    Sharpe bands use ``SE(SR_ann)=sqrt(P*(1+0.5*SR_pp^2)/window)``. Beta bands use
    ``beta +/- 1.96*SE(beta)`` from each rolling OLS fit. They are explicitly IID
    approximation bands; the report's headline HAC tests handle autocorrelation.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    output = pd.DataFrame(index=strategy.index, columns=[
        "rolling_sharpe", "sharpe_lower", "sharpe_upper", "rolling_beta", "beta_lower", "beta_upper"], dtype=float)
    for end in range(window, len(strategy) + 1):
        idx = strategy.index[end - 1]
        p, b = strategy.iloc[end - window:end], benchmark.iloc[end - window:end]
        sr = annualised_sharpe(p, periods_per_year)
        sr_pp = sr / math.sqrt(periods_per_year)
        sr_se = math.sqrt(periods_per_year * (1.0 + 0.5 * sr_pp ** 2) / window)
        regression = jensens_alpha(p, b, periods_per_year)
        beta_value = regression["beta"]
        if np.isfinite(beta_value) and b.var(ddof=1) > 0:
            residuals = p.to_numpy() - (regression["alpha_per_period"] + beta_value * b.to_numpy())
            beta_se = math.sqrt(np.sum(residuals ** 2) / (window - 2) / np.sum((b - b.mean()) ** 2))
        else:
            beta_se = float("nan")
        output.loc[idx] = [sr, sr - 1.96 * sr_se, sr + 1.96 * sr_se,
                           beta_value, beta_value - 1.96 * beta_se, beta_value + 1.96 * beta_se]
    return output


def rolling_ic_metrics(ic_values, window: int = 63) -> pd.DataFrame:
    """Rolling mean IC with a 95% IID confidence band: ``mean +/- 1.96*sd/sqrt(n)``."""
    if window < 3:
        raise ValueError("window must be >= 3")
    ic = pd.Series(ic_values, dtype=float)
    mean = ic.rolling(window, min_periods=window).mean()
    se = ic.rolling(window, min_periods=window).std(ddof=1) / math.sqrt(window)
    return pd.DataFrame({"rolling_ic": mean, "ic_lower": mean - 1.96 * se, "ic_upper": mean + 1.96 * se})


def _records(frame: pd.DataFrame, max_points: int = 500) -> list[dict]:
    frame = frame.dropna(how="all")
    if len(frame) > max_points:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, max_points, dtype=int)]
    records = []
    for index, row in frame.iterrows():
        record = {"date": str(getattr(index, "date", lambda: index)())}
        record.update({key: (float(value) if np.isfinite(value) else None) for key, value in row.items()})
        records.append(record)
    return records


def _undefined_reason(series: pd.Series, label: str) -> str | None:
    """Explain why a series cannot support a correlation, or None when it can."""
    if series.size < 2:
        return f"{label} has fewer than two aligned observations."
    variance = float(series.var(ddof=1))
    if not np.isfinite(variance):
        return f"{label} variance is not finite across the aligned sample."
    if variance <= 0.0:
        return (f"{label} is constant across all {series.size} aligned observations, "
                "so a correlation is undefined rather than zero.")
    return None


def return_correlation(strategy_returns, benchmark_returns) -> tuple[float, str | None]:
    """Pearson correlation, or NaN with a stated reason when it is undefined.

    ``pandas.Series.corr`` divides by each series' standard deviation, so a
    constant input produces 0/0 and a bare NaN. A flat return series is a real
    research outcome worth naming -- an undefined correlation is not the same
    claim as an estimated correlation of zero.
    """
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    reason = (_undefined_reason(strategy, "The strategy return series")
              or _undefined_reason(benchmark, "The benchmark return series"))
    if reason is not None:
        return float("nan"), reason
    return float(strategy.corr(benchmark)), None


def _statistic_diagnostics(summary: dict, correlation_reason: str | None,
                           strategy: pd.Series, benchmark: pd.Series) -> list[dict]:
    """Name every headline statistic that could not be computed, and why.

    Without this, a degenerate sample reaches the report as a field of NaNs that
    reads like a weak result instead of an uncomputable one.
    """
    findings: list[dict] = []

    def note(statistic: str, reason: str) -> None:
        findings.append({"statistic": statistic, "status": "undefined", "reason": reason})

    if correlation_reason is not None:
        note("correlation", correlation_reason)
    active = strategy - benchmark
    if not np.isfinite(summary["tracking_error"]):
        note("tracking_error", "Fewer than two aligned observations.")
    if not np.isfinite(summary["information_ratio"]):
        note("information_ratio", (
            "Active returns are constant, so their standard deviation is zero."
            if active.size > 1 and float(active.var(ddof=1)) <= 0.0
            else "Fewer than two aligned active-return observations."))
    if not np.isfinite(summary["jensens_alpha"]["beta"]):
        note("jensens_alpha", (
            "The benchmark return series is constant, so the regression slope is undefined."
            if benchmark.size > 1 and float(benchmark.var(ddof=1)) <= 0.0
            else "Fewer than three aligned observations for the regression."))
    captures = summary["capture_ratios"]
    for side in ("up", "down"):
        if not np.isfinite(captures[f"{side}_capture"]):
            note(f"capture_ratios.{side}_capture", (
                f"No benchmark {side} periods in the aligned sample."
                if captures[f"n_{side}_periods"] == 0
                else f"Mean benchmark return on {side} periods is zero."))
    return findings


def benchmark_relative_summary(strategy_returns, benchmark_returns, ic_values=None,
                               periods_per_year: int = TRADING_DAYS, rolling_window: int = 63) -> dict:
    """Compose Module 2 metrics and compact rolling-chart data into one report section."""
    strategy, benchmark = align_returns(strategy_returns, benchmark_returns)
    active = strategy - benchmark
    alpha = jensens_alpha(strategy, benchmark, periods_per_year)
    rolling = rolling_benchmark_metrics(strategy, benchmark, rolling_window, periods_per_year)
    correlation, correlation_reason = return_correlation(strategy, benchmark)
    summary = {
        "n_observations": int(strategy.size),
        "annualized_active_return": float(active.mean() * periods_per_year),
        "active_cagr_spread": float(annualised_return(strategy, periods_per_year) - annualised_return(benchmark, periods_per_year)),
        "tracking_error": tracking_error(strategy, benchmark, periods_per_year),
        "information_ratio": information_ratio(strategy, benchmark, periods_per_year),
        "correlation": correlation,
        "jensens_alpha": alpha,
        "capture_ratios": capture_ratios(strategy, benchmark),
        "rolling_window": int(rolling_window),
        "rolling_chart_data": _records(rolling.join(rolling_ic_metrics(ic_values, rolling_window), how="left") if ic_values is not None else rolling),
    }
    diagnostics = _statistic_diagnostics(summary, correlation_reason, strategy, benchmark)
    summary["statistic_diagnostics"] = diagnostics
    summary["has_undefined_statistics"] = bool(diagnostics)
    if diagnostics:
        summary["diagnostics_note"] = (
            "One or more benchmark-relative statistics are undefined for this sample. An undefined "
            "statistic is not a weak result; it means the input could not support the calculation."
        )
    return summary


def chart_specs() -> dict:
    """Library-agnostic Module 2 plotting specifications."""
    return {
        "rolling_sharpe_beta_ic": {
            "type": "three_time_series_panels",
            "transform": "63-day rolling annualized Sharpe, rolling OLS beta, and rolling mean IC; calculate their stated 95% confidence bands",
            "x": "return timestamp", "y": "statistic value",
            "annotations": ["zero line", "window length", "confidence-band methodology"],
        },
    }
