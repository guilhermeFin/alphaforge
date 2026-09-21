"""Time-aware filing event studies and deliberately small OOS text checks.

This module measures what happened after a dated document became usable.  It is
not a trading simulator: dates are treated as date-only, so the first eligible
price is always the *next* trading session.  That conservative convention avoids
pretending a filing's intraday publication time is known when it is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


MIN_TRAIN_EVENTS = 5
MIN_OOS_EVENTS = 4


@dataclass(frozen=True)
class FilingEvent:
    symbol: str
    available_at: pd.Timestamp
    sentiment: float
    document_id: str


def _next_session(index: pd.DatetimeIndex, available_at: pd.Timestamp) -> int | None:
    """Return the first price session strictly after a date-only filing timestamp."""
    normalized = pd.Timestamp(available_at).normalize()
    position = int(index.searchsorted(normalized, side="right"))
    return position if position < len(index) else None


def build_event_returns(
    events: list[FilingEvent],
    close: pd.DataFrame,
    benchmark: pd.Series,
    *,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Create event returns from adjusted closes without using a same-day price.

    Each row's abnormal return is company return minus benchmark return over the
    same holding window.  Missing prices and events with insufficient future data
    become explicit warnings rather than silently becoming zero-return observations.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one trading day")
    index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    close = close.copy()
    close.index = index
    benchmark = benchmark.copy()
    benchmark.index = pd.DatetimeIndex(pd.to_datetime(benchmark.index)).normalize()
    benchmark = benchmark.reindex(index).ffill()
    rows: list[dict] = []
    warnings: list[str] = []
    for event in events:
        if event.symbol not in close:
            warnings.append(f"{event.symbol}: no market-price series returned.")
            continue
        entry = _next_session(index, event.available_at)
        if entry is None or entry + horizon >= len(index):
            warnings.append(f"{event.symbol}: insufficient future prices after {event.available_at.date()}.")
            continue
        stock = close[event.symbol]
        start_price, end_price = stock.iloc[entry], stock.iloc[entry + horizon]
        market_start, market_end = benchmark.iloc[entry], benchmark.iloc[entry + horizon]
        if not all(np.isfinite([start_price, end_price, market_start, market_end])) or min(start_price, market_start) <= 0:
            warnings.append(f"{event.symbol}: incomplete adjusted-close window after {event.available_at.date()}.")
            continue
        company_return = float(end_price / start_price - 1.0)
        market_return = float(market_end / market_start - 1.0)
        rows.append({
            "symbol": event.symbol,
            "document_id": event.document_id,
            "available_at": pd.Timestamp(event.available_at),
            "entry_date": index[entry],
            "exit_date": index[entry + horizon],
            "sentiment": float(event.sentiment),
            "company_return": company_return,
            "benchmark_return": market_return,
            "abnormal_return": company_return - market_return,
        })
    return pd.DataFrame(rows), warnings


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a small linear explanatory model; a flat predictor is valid input."""
    if np.ptp(x) == 0:
        return float(np.mean(y)), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def evaluate_sentiment(events: pd.DataFrame, *, horizon: int) -> dict:
    """Chronological, purged OOS comparison against a no-text mean baseline.

    The baseline predicts the training-period average abnormal return.  The text
    model uses only training sentiment to estimate a linear relationship.  A small
    sample is reported as exploratory even if its point estimates look attractive.
    """
    required = {"entry_date", "exit_date", "sentiment", "abnormal_return"}
    if not required.issubset(events.columns):
        raise ValueError("event rows are missing required fields")
    frame = events.sort_values("entry_date").reset_index(drop=True).copy()
    if len(frame) < MIN_TRAIN_EVENTS + MIN_OOS_EVENTS:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_TRAIN_EVENTS + MIN_OOS_EVENTS} usable events; found {len(frame)}.",
            "n_events": int(len(frame)),
            "horizon": horizon,
        }
    initial_cut = max(MIN_TRAIN_EVENTS, int(len(frame) * 0.7))
    test_start = pd.Timestamp(frame.loc[initial_cut, "entry_date"])
    train = frame.iloc[:initial_cut].copy()
    # A training label that runs into the OOS period would overlap the holdout.
    purged = train[train["exit_date"] >= test_start]
    train = train[train["exit_date"] < test_start]
    test = frame.iloc[initial_cut:].copy()
    if len(train) < MIN_TRAIN_EVENTS or len(test) < MIN_OOS_EVENTS:
        return {
            "status": "insufficient_data",
            "message": "Too few independent events after the chronological boundary purge.",
            "n_events": int(len(frame)), "n_train": int(len(train)), "n_oos": int(len(test)),
            "purged_events": int(len(purged)), "horizon": horizon,
        }
    x_train, y_train = train["sentiment"].to_numpy(float), train["abnormal_return"].to_numpy(float)
    x_oos, y_oos = test["sentiment"].to_numpy(float), test["abnormal_return"].to_numpy(float)
    intercept, slope = _fit_linear(x_train, y_train)
    baseline = np.full_like(y_oos, np.mean(y_train), dtype=float)
    text_prediction = intercept + slope * x_oos
    baseline_mse = float(np.mean((y_oos - baseline) ** 2))
    text_mse = float(np.mean((y_oos - text_prediction) ** 2))
    mae = float(np.mean(np.abs(y_oos - text_prediction)))
    if np.ptp(x_oos) == 0 or np.ptp(y_oos) == 0:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_r, spearman_p = 0.0, 1.0
    else:
        pearson_r, pearson_p = stats.pearsonr(x_oos, y_oos)
        spearman_r, spearman_p = stats.spearmanr(x_oos, y_oos)
    same_sign = float(np.mean(np.sign(text_prediction) == np.sign(y_oos)))
    improves = text_mse < baseline_mse
    established = bool(len(test) >= 12 and improves and pearson_p < 0.05)
    return {
        "status": "complete",
        "n_events": int(len(frame)), "n_train": int(len(train)), "n_oos": int(len(test)),
        "purged_events": int(len(purged)), "horizon": horizon,
        "baseline_mse": baseline_mse, "text_mse": text_mse, "text_mae": mae,
        "mse_improvement": baseline_mse - text_mse,
        "oos_pearson": float(pearson_r), "oos_pearson_p_value": float(pearson_p),
        "oos_spearman": float(spearman_r), "oos_spearman_p_value": float(spearman_p),
        "sign_accuracy": same_sign, "intercept": intercept, "sentiment_slope": slope,
        "text_improves_baseline": improves,
        "evidence_established": established,
        "verdict": (
            "Out-of-sample explanatory value is established in this sample."
            if established else
            "Out-of-sample explanatory value is not established; treat this as exploratory research."
        ),
        "events": frame.to_dict(orient="records"),
    }
