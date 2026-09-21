"""Optional, reproducible econometric diagnostics for research reports."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd


def _load_arch() -> tuple[Any, Any, Any] | None:
    """Import optional arch components only when an analysis requests them."""
    try:
        from arch import arch_model
        from arch.bootstrap import MCS, SPA
    except ImportError:
        return None
    return arch_model, MCS, SPA


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _column_names(values: Any, columns: pd.Index) -> list[str]:
    names: list[str] = []
    for value in list(values):
        if value in columns:
            names.append(str(value))
        elif isinstance(value, (int, np.integer)) and 0 <= int(value) < len(columns):
            names.append(str(columns[int(value)]))
        else:
            names.append(str(value))
    return names


def conditional_volatility_forecast(
    returns: pd.Series,
    *,
    horizon: int = 21,
    min_observations: int = 100,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Fit a GARCH(1,1) model using only the supplied return history."""
    loaded = _load_arch()
    series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if loaded is None:
        return {
            "available": False,
            "reason": "Install the optional econometrics extra to enable GARCH diagnostics.",
        }
    if len(series) < min_observations:
        return {
            "available": False,
            "reason": f"At least {min_observations} return observations are required for GARCH diagnostics.",
            "n_observations": int(len(series)),
        }

    arch_model, _, _ = loaded
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = arch_model(
                series.to_numpy(dtype=float) * 100.0,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="t",
                rescale=False,
            ).fit(disp="off", show_warning=False)
        if getattr(fitted, "convergence_flag", 0) != 0:
            return {
                "available": False,
                "reason": "GARCH optimization did not converge for this return series.",
                "n_observations": int(len(series)),
            }
        variance = np.asarray(
            fitted.forecast(horizon=horizon, reindex=False).variance.iloc[-1], dtype=float
        )
        daily_volatility = np.sqrt(np.clip(variance, 0.0, None)) / 100.0
        params = {str(key): _float(value) for key, value in fitted.params.items()}
        return {
            "available": True,
            "model": "GARCH(1,1) with Student-t innovations",
            "n_observations": int(len(series)),
            "horizon_days": int(horizon),
            "forecast_daily_volatility": [float(value) for value in daily_volatility],
            "first_day_annualized_volatility": float(daily_volatility[0] * np.sqrt(periods_per_year)),
            "average_horizon_annualized_volatility": float(
                np.mean(daily_volatility) * np.sqrt(periods_per_year)
            ),
            "parameters": params,
            "note": "A conditional-risk estimate, not a return forecast or trading recommendation.",
        }
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            "available": False,
            "reason": f"GARCH fit was not stable for this return series ({type(exc).__name__}).",
            "n_observations": int(len(series)),
        }


def candidate_model_tests(
    candidate_returns: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,
    alpha: float = 0.10,
    reps: int = 500,
    seed: int = 7,
) -> dict[str, Any]:
    """Run SPA and Model Confidence Set tests on candidate return histories.

    The tests operate on negative returns as losses. A supplied benchmark is the
    SPA benchmark; without one, the benchmark is a zero excess-return series.
    """
    loaded = _load_arch()
    if loaded is None:
        return {
            "available": False,
            "reason": "Install the optional econometrics extra to enable SPA and MCS tests.",
        }

    candidates = pd.DataFrame(candidate_returns).apply(pd.to_numeric, errors="coerce")
    candidates = candidates.dropna(axis=1, how="all")
    if candidates.shape[1] < 2:
        return {
            "available": False,
            "reason": "At least two candidate return series are required for model comparison.",
            "n_candidates": int(candidates.shape[1]),
        }

    if benchmark_returns is None:
        benchmark = pd.Series(0.0, index=candidates.index, name="zero_excess_return")
        benchmark_name = "zero excess return"
    else:
        benchmark = pd.to_numeric(pd.Series(benchmark_returns), errors="coerce").rename("benchmark")
        benchmark_name = str(benchmark.name or "benchmark")
    aligned = pd.concat([benchmark, candidates], axis=1).dropna(how="any")
    if len(aligned) < 30:
        return {
            "available": False,
            "reason": "At least 30 aligned observations are required for bootstrap model comparison.",
            "n_observations": int(len(aligned)),
            "n_candidates": int(candidates.shape[1]),
        }

    _, mcs_class, spa_class = loaded
    model_returns = aligned[candidates.columns]
    benchmark_loss = -aligned.iloc[:, 0].to_numpy(dtype=float)
    model_losses = -model_returns
    try:
        spa = spa_class(
            benchmark_loss,
            model_losses,
            reps=reps,
            bootstrap="stationary",
            seed=seed,
        )
        spa.compute()
        mcs = mcs_class(
            model_losses,
            size=alpha,
            reps=reps,
            bootstrap="stationary",
            seed=seed,
        )
        mcs.compute()

        spa_pvalues = {str(key): _float(value) for key, value in spa.pvalues.items()}
        mcs_pvalues = [
            {"model": str(index), "p_value": _float(row.iloc[0])}
            for index, row in mcs.pvalues.iterrows()
        ]
        included = _column_names(mcs.included, model_returns.columns)
        excluded = _column_names(mcs.excluded, model_returns.columns)
        superior = _column_names(
            spa.better_models(pvalue=alpha, pvalue_type="consistent"), model_returns.columns
        )
        return {
            "available": True,
            "method": "stationary-bootstrap SPA and Model Confidence Set",
            "n_observations": int(len(aligned)),
            "n_candidates": int(model_returns.shape[1]),
            "benchmark": benchmark_name,
            "alpha": float(alpha),
            "bootstrap_repetitions": int(reps),
            "spa": {
                "p_values": spa_pvalues,
                "superior_models": superior,
                "interpretation": "Models listed here cleared the SPA comparison at the stated alpha against the benchmark.",
            },
            "model_confidence_set": {
                "included_models": included,
                "excluded_models": excluded,
                "p_values": mcs_pvalues,
                "interpretation": "Included models could not be rejected as inferior within this candidate set at the stated alpha.",
            },
            "note": "These comparisons account for a candidate set; they do not establish deployable performance.",
        }
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            "available": False,
            "reason": f"Bootstrap model comparison could not be completed ({type(exc).__name__}).",
            "n_observations": int(len(aligned)),
            "n_candidates": int(model_returns.shape[1]),
        }


def advanced_econometrics_summary(
    returns: pd.Series,
    *,
    candidate_returns: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Return optional advanced diagnostics without making a report fail closed."""
    summary: dict[str, Any] = {
        "conditional_volatility": conditional_volatility_forecast(
            returns, periods_per_year=periods_per_year
        )
    }
    if candidate_returns is not None:
        summary["candidate_model_tests"] = candidate_model_tests(
            candidate_returns, benchmark_returns=benchmark_returns
        )
    return summary
