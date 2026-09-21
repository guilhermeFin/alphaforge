"""Historical-only structural-break diagnostics for return series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _load_ruptures() -> Any | None:
    try:
        import ruptures as rpt
    except ImportError:
        return None
    return rpt


def _timestamp(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def point_in_time_regime_analysis(
    returns: pd.Series,
    *,
    lookback: int = 252,
    min_history: int = 126,
    refit_every: int = 21,
    min_segment_size: int = 21,
    penalty: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Detect return-distribution breaks with expanding point-in-time refits.

    At each refit date the detector sees only a trailing window ending on that
    date. Consequently, a regime label never incorporates subsequent returns.
    """
    rpt = _load_ruptures()
    series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if rpt is None:
        return {
            "available": False,
            "reason": "Install the optional econometrics extra to enable structural-break diagnostics.",
        }
    if len(series) < min_history:
        return {
            "available": False,
            "reason": f"At least {min_history} return observations are required for regime analysis.",
            "n_observations": int(len(series)),
        }
    if lookback < min_history or refit_every < 1 or min_segment_size < 2:
        raise ValueError("Invalid regime-analysis configuration.")

    index = pd.Index(series.index)
    values = series.to_numpy(dtype=float)
    labels: list[Any | None] = [None] * len(series)
    break_dates: list[Any] = []
    active_start: Any | None = None
    refits = 0

    for end_position in range(min_history - 1, len(series), refit_every):
        window_start = max(0, end_position - lookback + 1)
        window = values[window_start : end_position + 1]
        window_index = index[window_start : end_position + 1]
        active_start = active_start or window_index[0]
        effective_penalty = float(penalty) if penalty is not None else float(3.0 * np.log(len(window)))
        try:
            detector = rpt.Pelt(
                model="rbf", min_size=min_segment_size, jump=1
            ).fit(window.reshape(-1, 1))
            breakpoints = detector.predict(pen=effective_penalty)[:-1]
            if breakpoints:
                candidate_start = window_index[int(breakpoints[-1])]
                if candidate_start > active_start:
                    active_start = candidate_start
                    if not break_dates or candidate_start > break_dates[-1]:
                        break_dates.append(candidate_start)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            # A non-convergent refit leaves the last point-in-time state intact.
            pass

        next_refit = min(len(series), end_position + refit_every)
        for position in range(end_position, next_refit):
            labels[position] = active_start
        refits += 1

    labelled = pd.DataFrame({"return": series, "regime_start": labels}).dropna(subset=["regime_start"])
    summaries: list[dict[str, Any]] = []
    for regime_start, group in labelled.groupby("regime_start", sort=True):
        volatility = float(group["return"].std(ddof=1) * np.sqrt(periods_per_year))
        mean_return = float(group["return"].mean())
        summaries.append(
            {
                "regime_start": _timestamp(regime_start),
                "observations": int(len(group)),
                "annualized_mean_return": float(mean_return * periods_per_year),
                "annualized_volatility": volatility,
                "annualized_sharpe": (
                    float(mean_return / group["return"].std(ddof=1) * np.sqrt(periods_per_year))
                    if np.isfinite(volatility) and volatility > 0
                    else None
                ),
            }
        )

    current_start = labels[-1]
    current_age = int(sum(label == current_start for label in labels if label is not None))
    chart_positions = np.linspace(0, len(labelled) - 1, min(300, len(labelled)), dtype=int)
    chart_frame = labelled.iloc[np.unique(chart_positions)]
    return {
        "available": True,
        "method": "trailing PELT structural-break detector (RBF cost)",
        "n_observations": int(len(series)),
        "lookback_observations": int(lookback),
        "refit_every_observations": int(refit_every),
        "n_refits": int(refits),
        "n_detected_breaks": int(len(break_dates)),
        "break_dates": [_timestamp(value) for value in break_dates],
        "latest_break_date": _timestamp(break_dates[-1]) if break_dates else None,
        "current_regime_age_observations": current_age,
        "regime_summary": summaries,
        "chart_data": [
            {"date": _timestamp(date), "regime_start": _timestamp(regime_start)}
            for date, regime_start in zip(chart_frame.index, chart_frame["regime_start"], strict=False)
        ],
        "note": "Every state was fit only with return observations available on or before its refit date. This is descriptive, not a trading signal.",
    }
