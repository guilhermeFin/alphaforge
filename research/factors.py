"""Factor library — combine many weak, *past-only* signals into target weights.

Every factor here is a function of information available strictly up to and
including each date's close. The backtester applies an additional execution lag,
so a factor must never peek across its own row into the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def momentum(close: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """Price change over ``lookback`` days, skipping the most recent ``skip`` days.

    Skipping the last month avoids the well-known short-term reversal effect. Uses
    only shifted (past) prices, so it is point-in-time safe by construction.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def trailing_volatility(close: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Annualised trailing volatility of daily returns (low-vol factor = -this)."""
    rets = close.pct_change(fill_method=None)
    return rets.rolling(lookback).std() * np.sqrt(252)


def short_term_reversal(close: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Negative of the last ~month's return (mean-reversion signal)."""
    return -(close / close.shift(lookback) - 1.0)


def cross_sectional_zscore(panel: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Row-wise (cross-sectional) z-score, clipped. Each date is standardised
    across symbols, so a factor's raw units don't matter."""
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1)
    z = panel.sub(mu, axis=0).div(sd + EPS, axis=0)
    return z.clip(-clip, clip)


def blend(*factors: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight blend of several (already standardised) factors."""
    if not factors:
        raise ValueError("blend() needs at least one factor")
    stacked = sum(f.fillna(0.0) for f in factors)
    return stacked / len(factors)


def long_short_weights(score: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Convert a cross-sectional score into dollar-neutral target weights.

    Per date: demean (so longs ≈ shorts) then scale so gross exposure (sum of
    absolute weights) equals ``gross``. Rows with no dispersion → all zeros.
    """
    s = score.sub(score.mean(axis=1), axis=0)
    denom = s.abs().sum(axis=1)
    w = s.div(denom.replace(0.0, np.nan), axis=0).mul(gross)
    return w.fillna(0.0)
