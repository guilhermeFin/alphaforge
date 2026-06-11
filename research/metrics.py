"""Performance & risk metrics — fat-tail- and multiple-testing-aware.

The headline number is NOT the Sharpe ratio. It is the Probabilistic Sharpe Ratio
(PSR) and, when the user has tried several strategies, the Deflated Sharpe Ratio
(DSR) — Bailey & López de Prado. Both use the return skewness and kurtosis, so a
juicy Sharpe built on fat-tailed, negatively-skewed returns is correctly
discounted. This is the statistical backbone of "won't let you lie to yourself".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
_EULER = 0.5772156649015329


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def sharpe_per_period(returns) -> float:
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    # Floor guards against floating-point noise on a (near-)constant series
    # producing an absurd Sharpe (e.g. std ~1e-18). Real strategy vols are far above this.
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd)


def annualised_sharpe(returns, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _clean(returns) - rf / periods_per_year
    return sharpe_per_period(r) * np.sqrt(periods_per_year)


def annualised_return(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    growth = np.prod(1.0 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / r.size) - 1.0)


def annualised_vol(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0


def sortino(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    downside = r[r < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else np.nan


def equity_curve(returns) -> pd.Series:
    r = pd.Series(returns).fillna(0.0)
    return (1.0 + r).cumprod()


def max_drawdown(equity) -> float:
    eq = pd.Series(equity).dropna()
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1.0).min())


def calmar(returns, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity_curve(returns)))
    return float(annualised_return(returns, periods_per_year) / mdd) if mdd > 0 else np.nan


def hit_rate(returns) -> float:
    r = _clean(returns)
    nz = r[r != 0]
    return float((nz > 0).mean()) if nz.size else np.nan


def skewness(returns) -> float:
    r = _clean(returns)
    return float(stats.skew(r, bias=False)) if r.size > 2 else np.nan


def excess_kurtosis(returns) -> float:
    r = _clean(returns)
    return float(stats.kurtosis(r, fisher=True, bias=False)) if r.size > 3 else np.nan


def jarque_bera_pvalue(returns) -> float:
    """p-value of the Jarque-Bera normality test. Low p ⇒ returns are NOT Normal
    (the expected, honest verdict for real financial returns)."""
    r = _clean(returns)
    if r.size < 8:
        return np.nan
    return float(stats.jarque_bera(r).pvalue)


def probabilistic_sharpe_ratio(returns, sr_benchmark_pp: float = 0.0) -> float:
    """PSR: probability the *true* per-period Sharpe exceeds ``sr_benchmark_pp``.

    Accounts for track-record length, skewness and kurtosis (Bailey & López de
    Prado, 2012). Returns a probability in [0, 1]; > 0.95 is the usual bar.
    ``sr_benchmark_pp`` is a per-period (not annualised) Sharpe.
    """
    r = _clean(returns)
    T = r.size
    if T < 3:
        return np.nan
    sr = sharpe_per_period(r)
    g3 = stats.skew(r, bias=False)
    g4 = stats.kurtosis(r, fisher=False, bias=False)  # full (Normal = 3)
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return np.nan
    z = (sr - sr_benchmark_pp) * np.sqrt(T - 1.0) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe_pp(n_trials: int, sr_trials_std_pp: float) -> float:
    """Expected maximum per-period Sharpe from ``n_trials`` independent strategies
    whose true Sharpe is 0 but whose *estimates* have dispersion ``sr_trials_std_pp``.
    The multiple-testing hurdle a real edge must clear (Bailey & López de Prado)."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_trials_std_pp * ((1.0 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe_ratio(returns, n_trials: int, sr_trials_std_pp: float | None = None) -> float:
    """DSR: PSR measured against the expected-max-Sharpe hurdle from ``n_trials``.

    If ``sr_trials_std_pp`` (the std of Sharpe estimates across the strategies you
    tried) is unknown, it is approximated by this strategy's own Sharpe estimation
    error — a conservative-ish default. Strongly discounts data-mined results.
    """
    r = _clean(returns)
    T = r.size
    if T < 3 or n_trials < 1:
        return np.nan
    if n_trials == 1:
        return probabilistic_sharpe_ratio(r, 0.0)
    if sr_trials_std_pp is None:
        sr = sharpe_per_period(r)
        sr_trials_std_pp = np.sqrt((1.0 + 0.5 * sr ** 2) / (T - 1.0))
    hurdle = expected_max_sharpe_pp(n_trials, sr_trials_std_pp)
    return probabilistic_sharpe_ratio(r, hurdle)


def summarize(
    returns,
    positions=None,
    periods_per_year: int = TRADING_DAYS,
    n_trials: int = 1,
) -> dict:
    """One call → the full honesty-aware scorecard."""
    r = pd.Series(returns).dropna()
    eq = equity_curve(r)
    out = {
        "n_periods": int(r.size),
        "total_return": float(eq.iloc[-1] - 1.0) if not eq.empty else 0.0,
        "cagr": annualised_return(r, periods_per_year),
        "ann_vol": annualised_vol(r, periods_per_year),
        "ann_sharpe": annualised_sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(r, periods_per_year),
        "hit_rate": hit_rate(r),
        "skew": skewness(r),
        "excess_kurtosis": excess_kurtosis(r),
        "jarque_bera_p": jarque_bera_pvalue(r),
        "psr_vs_0": probabilistic_sharpe_ratio(r, 0.0),
        "deflated_sr": deflated_sharpe_ratio(r, n_trials),
        "n_trials": n_trials,
    }
    if positions is not None:
        pos = positions
        if isinstance(pos, pd.DataFrame):
            turn = pos.diff().abs().sum(axis=1)
        else:
            turn = pd.Series(pos).diff().abs()
        out["avg_turnover"] = float(turn.mean())
    out["returns_are_normal"] = bool(out["jarque_bera_p"] >= 0.05) if not np.isnan(out["jarque_bera_p"]) else None
    return out
