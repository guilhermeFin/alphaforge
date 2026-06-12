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


@dataclass
class World:
    """A synthetic world for testing quantamental pipelines end-to-end.

    Prices are driven by a *latent* per-name fundamental state ``quality`` that is
    known at each date and drives the NEXT day's drift. ``events`` are points in
    time where that state would be 'disclosed' (e.g. an earnings call), carrying
    the ground-truth latent sentiment. Text is generated from those events
    elsewhere (research/quantamental.py); the backtester only ever sees a signal
    *extracted from text*, never the latent state — so a strategy that works is a
    genuine, point-in-time-correct predictive signal, not a leak.
    """
    close: pd.DataFrame
    volume: pd.DataFrame
    quality: pd.DataFrame   # latent fundamental state (ground truth), dates x symbols
    events: pd.DataFrame    # columns: symbol, date, latent_sentiment
    regime: pd.Series | None = None

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def make_synthetic_world(
    symbols: list[str],
    periods: int = 1512,
    start: str = "2015-01-02",
    seed: int = 0,
    n_events_per_symbol: int = 12,
    quality_persistence: float = 0.985,
    quality_to_drift: float = 0.0006,
    fat_tails: bool = True,
    t_df: int = 4,
) -> World:
    """Coupled price + latent-fundamental generator (see World)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    n, T = len(symbols), periods

    # latent fundamental quality: stationary AR(1) per name (unit variance)
    rho = quality_persistence
    q = np.empty((T, n))
    q[0] = rng.standard_normal(n)
    shock = rng.standard_normal((T, n))
    for t in range(1, T):
        q[t] = rho * q[t - 1] + np.sqrt(1.0 - rho ** 2) * shock[t]

    # market factor with a 2-state regime
    P = np.array([[0.95, 0.05], [0.02, 0.98]])
    states = np.empty(T, dtype=int)
    states[0] = 1
    u = rng.random(T)
    for t in range(1, T):
        states[t] = 1 if u[t] < P[states[t - 1], 1] else 0
    market = np.where(states == 1, 0.0005, -0.0009) + \
        np.where(states == 1, 0.008, 0.020) * _innovations(rng, T, fat_tails, t_df)

    betas = rng.uniform(0.6, 1.4, n)
    idio_sd = rng.uniform(0.15, 0.45, n) / np.sqrt(TRADING_DAYS) * 0.7
    q_lag = np.vstack([q[0:1], q[:-1]])  # yesterday's quality drives today's drift -> no same-day leak

    rets = np.empty((T, n))
    for j in range(n):
        rets[:, j] = betas[j] * market + quality_to_drift * q_lag[:, j] + idio_sd[j] * _innovations(rng, T, fat_tails, t_df)

    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=symbols)
    volume = pd.DataFrame(rng.uniform(5e5, 5e6, n) * np.exp(rng.normal(0, 0.2, (T, n))),
                          index=dates, columns=symbols).round()
    quality = pd.DataFrame(q, index=dates, columns=symbols)

    warm = 10
    rows = []
    cand = np.arange(warm, T - 5)
    for j, sym in enumerate(symbols):
        picks = np.sort(rng.choice(cand, size=min(n_events_per_symbol, cand.size), replace=False))
        for d in picks:
            rows.append((sym, dates[d], float(np.tanh(q[d, j]))))  # disclosed sentiment ~ current quality
    events = pd.DataFrame(rows, columns=["symbol", "date", "latent_sentiment"]).sort_values("date").reset_index(drop=True)

    return World(close=close, volume=volume, quality=quality, events=events,
                 regime=pd.Series(states, index=dates, name="regime"))


def make_synthetic_fundamentals(
    world: World,
    reporting_lag_days: int = 75,
    period_days: int = 63,
    seed: int = 0,
) -> pd.DataFrame:
    """Quarterly fundamentals for a synthetic World, with a realistic reporting lag.

    Returns a long table [symbol, period_end, available_date, metric, value]. The
    KEY honesty property: ``available_date`` = fiscal period end + ``reporting_lag_days``
    (a quarter's numbers are not knowable until the filing lands ~75 days later).
    Quality-type metrics (roe, gross_profitability, low leverage) track the latent
    ``world.quality`` so the QUALITY factor has a genuine, point-in-time-correct
    signal to find; the price-ratio inputs (eps/bvps/sps) are made ~scale-free, so
    the VALUE factor is intentionally NOT planted (on synthetic data it should look
    like noise — and the engine should be honestly unconvinced by it).
    """
    rng = np.random.default_rng(seed)
    idx = world.close.index
    T = len(idx)
    rows = []
    period_ends = idx[period_days - 1::period_days]
    for pe in period_ends:
        pos = idx.searchsorted(pe + pd.Timedelta(days=reporting_lag_days))
        if pos >= T:
            continue
        avail = idx[pos]
        for sym in world.symbols:
            ql = float(world.quality.loc[pe, sym])
            px = float(world.close.loc[pe, sym])
            metrics = {
                "roe": 0.12 + 0.06 * ql + rng.normal(0, 0.02),
                "gross_profitability": 0.30 + 0.10 * ql + rng.normal(0, 0.03),
                "leverage": max(0.05, 0.80 - 0.25 * ql + rng.normal(0, 0.10)),
                "eps_ttm": px * (0.05 + rng.normal(0, 0.010)),   # ~constant earnings yield
                "bvps": px * (0.50 + rng.normal(0, 0.100)),       # ~constant book-to-price
                "sps": px * (0.80 + rng.normal(0, 0.150)),        # ~constant sales-to-price
            }
            for metric, value in metrics.items():
                rows.append((sym, pe, avail, metric, float(value)))
    return pd.DataFrame(rows, columns=["symbol", "period_end", "available_date", "metric", "value"])
