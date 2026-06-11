"""Data ingestion with a stable interface.

The MVP needs to run offline, deterministically, in tests and demos — so the
default provider is a *synthetic* market generator with two properties that make
it a fair stress-test of an honest backtester:

  * a 2-state (bull/bear) Markov regime drives the market factor — so strategies
    must survive regime change, not just a single trend (Blitzstein Ch.11);
  * innovations are Student-t (fat-tailed) by default — so the engine's tail and
    normality diagnostics have something real to flag (Blitzstein Ch.6/Ch.10).

A free, keyless yfinance provider is included for real tickers. We deliberately
serve *derived* results downstream, never redistribute a raw vendor feed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Panel:
    """A point-in-time-aligned panel: dates x symbols."""
    close: pd.DataFrame
    volume: pd.DataFrame
    regime: pd.Series | None = None  # 1=bull, 0=bear (synthetic only)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def _innovations(rng: np.random.Generator, size, fat_tails: bool, t_df: int) -> np.ndarray:
    """Unit-variance shocks. Student-t (standardised) if fat_tails else Normal."""
    if fat_tails:
        if t_df <= 2:
            raise ValueError("t_df must be > 2 for finite variance")
        x = rng.standard_t(t_df, size=size)
        return x / np.sqrt(t_df / (t_df - 2.0))  # standardise to unit variance
    return rng.standard_normal(size)


def make_synthetic_panel(
    symbols: list[str],
    periods: int = 1260,
    start: str = "2015-01-02",
    seed: int = 0,
    regime: bool = True,
    fat_tails: bool = True,
    t_df: int = 4,
) -> Panel:
    """Generate a deterministic synthetic OHLCV panel.

    Returns daily prices for ``symbols`` over ``periods`` business days. Each name
    has a market beta plus a small persistent alpha (so cross-sectional momentum
    has something faint and noisy to find — honest, not planted to look easy) and
    idiosyncratic fat-tailed noise. A shared Markov regime flips drift/vol.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    n = len(symbols)
    T = periods

    # --- shared market regime (2-state Markov chain) ---
    if regime:
        # rows = current state, cols = next state; state 1 = bull, 0 = bear
        P = np.array([[0.95, 0.05],   # from bear
                      [0.02, 0.98]])  # from bull (bulls persist longer)
        states = np.empty(T, dtype=int)
        states[0] = 1
        u = rng.random(T)
        for t in range(1, T):
            states[t] = 1 if u[t] < P[states[t - 1], 1] else 0
    else:
        states = np.ones(T, dtype=int)

    mkt_mu = np.where(states == 1, 0.0005, -0.0009)   # daily market drift by regime
    mkt_sd = np.where(states == 1, 0.008, 0.020)      # bear vol > bull vol
    market = mkt_mu + mkt_sd * _innovations(rng, T, fat_tails, t_df)

    betas = rng.uniform(0.6, 1.4, n)
    alphas = rng.uniform(-0.00025, 0.00045, n)        # persistent winners/losers
    idio_sd = rng.uniform(0.15, 0.45, n) / np.sqrt(TRADING_DAYS) * 0.7

    rets = np.empty((T, n))
    for j in range(n):
        rets[:, j] = alphas[j] + betas[j] * market + idio_sd[j] * _innovations(rng, T, fat_tails, t_df)

    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=symbols)
    # volume: lognormal, mildly higher on big-move days (crude but plausible)
    base = rng.uniform(5e5, 5e6, n)
    vmult = np.exp(0.5 * np.abs(rets) / idio_sd - 0.125)
    volume = pd.DataFrame(base * vmult * np.exp(rng.normal(0, 0.2, (T, n))),
                          index=dates, columns=symbols).round()

    return Panel(close=close, volume=volume,
                 regime=pd.Series(states, index=dates, name="regime") if regime else None)


def _yf_close_panel(symbols: list[str], start: str, end: str | None) -> Panel:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("yfinance not installed. `pip install alphaforge[data]`") from e
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    if isinstance(close, pd.Series):  # single symbol
        close = close.to_frame(symbols[0])
        volume = volume.to_frame(symbols[0])
    close = close.ffill().dropna(how="all")
    volume = volume.reindex(close.index)
    return Panel(close=close, volume=volume)


def get_panel(
    symbols: list[str],
    start: str = "2015-01-02",
    end: str | None = None,
    provider: str = "synthetic",
    **kwargs,
) -> Panel:
    """Stable entry point. provider ∈ {"synthetic", "yfinance"}."""
    if provider == "synthetic":
        return make_synthetic_panel(symbols, start=start, **kwargs)
    if provider in ("yfinance", "yahoo"):
        return _yf_close_panel(symbols, start, end)
    raise ValueError(f"unknown provider: {provider!r}")
