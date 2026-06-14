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


# ---------------------------------------------------------------------------
# Tail risk — HISTORICAL / EMPIRICAL CVaR (Expected Shortfall).
#
# We deliberately AVOID a parametric-Normal CVaR. The engine already flags fat
# tails (Jarque-Bera, excess kurtosis); a Gaussian ES would systematically
# understate the real loss and contradict that warning. We instead read the
# loss straight off the empirical return distribution. The Normal-implied ES is
# computed ONLY as a baseline, to *expose* how much fatter the real tail is.
# All numbers are per-period, NOT annualised.
# ---------------------------------------------------------------------------

_MIN_TAIL_OBS = 20


def value_at_risk(returns, alpha: float = 0.95) -> float:
    """Historical Value-at-Risk as a POSITIVE loss number.

    VaR_alpha = -quantile(r, 1-alpha). With alpha=0.95 this is the negative of
    the 5th-percentile return: the loss you expect to exceed (1-alpha) of the
    time. Uses the empirical distribution (method="lower"), never a Normal fit.
    Returns NaN if fewer than 20 observations.
    """
    r = _clean(returns)
    if r.size < _MIN_TAIL_OBS:
        return np.nan
    q = np.quantile(r, 1.0 - alpha, method="lower")
    return float(-q)


def conditional_var(returns, alpha: float = 0.95) -> float:
    """Historical Conditional VaR / Expected Shortfall, as a POSITIVE loss.

    CVaR_alpha = -mean of all returns at or below the (1-alpha) empirical
    quantile — the *average* loss in the bad tail, not just its edge. This is
    coherent (sub-additive) where VaR is not, and being empirical it captures
    fat tails. Returns NaN if fewer than 20 observations or the tail is empty.
    """
    r = _clean(returns)
    if r.size < _MIN_TAIL_OBS:
        return np.nan
    threshold = np.quantile(r, 1.0 - alpha, method="lower")
    tail = r[r <= threshold]
    if tail.size == 0:
        return np.nan
    return float(-tail.mean())


def _normal_implied_es(vol_pp: float, alpha: float = 0.99) -> float:
    """Expected Shortfall a Normal distribution WOULD predict at ``alpha`` for a
    zero-mean series of per-period vol ``vol_pp``. ES_Normal = sigma * phi(z) / (1-alpha)
    where z = Phi^{-1}(alpha). Baseline only — used to show how much the real
    empirical tail exceeds the Gaussian fantasy."""
    if not np.isfinite(vol_pp) or vol_pp <= 0:
        return np.nan
    tail_p = 1.0 - alpha
    z = stats.norm.ppf(alpha)
    return float(vol_pp * stats.norm.pdf(z) / tail_p)


def _cvar_verdict(cvar95: float, cvar99: float, vol_pp: float) -> str:
    """Plain-language read on the tail. Compares the empirical 99% ES against
    the Normal-implied 99% ES; a large ratio means the real tail is far fatter
    than a Gaussian model would lead you to believe."""
    if any(not np.isfinite(x) for x in (cvar95, cvar99, vol_pp)) or vol_pp <= 0:
        return "insufficient data for a tail-risk read"
    normal_es99 = _normal_implied_es(vol_pp, 0.99)
    if not np.isfinite(normal_es99) or normal_es99 <= 0:
        return "insufficient data for a tail-risk read"
    ratio = cvar99 / normal_es99
    cv95 = cvar95 * 100.0
    cv99 = cvar99 * 100.0
    head = (
        f"empirical ES: {cv95:.2f}% (95%) / {cv99:.2f}% (99%) per period; "
        f"the real 99% tail is {ratio:.2f}x the Normal-implied {normal_es99 * 100.0:.2f}%"
    )
    if ratio >= 1.5:
        return head + " — FAT TAIL: a Gaussian VaR badly understates your worst losses"
    if ratio >= 1.15:
        return head + " — moderately fat tail; Normal risk models are optimistic here"
    return head + " — tail roughly in line with a Normal model"


# ---------------------------------------------------------------------------
# AFML Ch.14 backtest statistics — concentration & drawdown structure.
# ---------------------------------------------------------------------------


def hhi(x) -> float:
    """Herfindahl-Hirschman concentration index, normalised to [0, 1].

    HHI = (sum(w**2) - 1/n) / (1 - 1/n) with weights w = x / sum(x), computed
    over the NON-EMPTY, SAME-SIGN subset of ``x``. 0 = perfectly uniform spread,
    1 = a single element carries everything. Returns NaN if fewer than 2
    usable (finite, non-zero, same-sign) observations.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    a = a[a != 0.0]
    if a.size < 2:
        return np.nan
    s = a.sum()
    if s == 0.0 or not np.isfinite(s):
        return np.nan
    w = a / s
    n = w.size
    raw = float(np.sum(w ** 2))
    denom = 1.0 - 1.0 / n
    if denom <= 0:
        return np.nan
    return float((raw - 1.0 / n) / denom)


def returns_concentration(returns) -> dict:
    """Are gains / losses concentrated in just a few periods? (AFML 14.5)

    positive_hhi: HHI over the positive returns.
    negative_hhi: HHI over the magnitudes of the negative returns.
    High values warn that the track record hangs on a handful of lucky (or
    unlucky) periods rather than a broad, repeatable edge.
    """
    r = _clean(returns)
    pos = r[r > 0]
    neg = r[r < 0]
    positive_hhi = hhi(pos)
    negative_hhi = hhi(-neg)  # magnitudes, same-sign positive subset
    if np.isnan(positive_hhi) and np.isnan(negative_hhi):
        verdict = "too few non-zero returns to judge concentration"
    else:
        worst = np.nanmax([
            positive_hhi if np.isfinite(positive_hhi) else np.nan,
            negative_hhi if np.isfinite(negative_hhi) else np.nan,
        ])
        if worst >= 0.5:
            verdict = "HIGHLY concentrated: P&L driven by a few periods — fragile, likely not repeatable"
        elif worst >= 0.2:
            verdict = "moderately concentrated returns"
        else:
            verdict = "well-spread returns across periods"
    return {
        "positive_hhi": positive_hhi,
        "negative_hhi": negative_hhi,
        "hhi_verdict": verdict,
    }


def _to_equity(equity_or_returns) -> pd.Series:
    """Heuristic: a series that is clearly a price/equity path (all positive,
    spans away from 0) is used as-is; otherwise it is treated as returns and
    compounded. An explicit equity curve almost always starts near 1 and stays
    positive; returns straddle zero."""
    s = pd.Series(equity_or_returns).dropna()
    if s.empty:
        return s
    vals = s.to_numpy(dtype=float)
    looks_like_equity = bool(np.all(vals > 0)) and float(np.min(vals)) > 0.0
    # Returns series typically contain values <= 0; an equity path does not.
    if looks_like_equity and (vals < 0).sum() == 0:
        return s.reset_index(drop=True)
    return equity_curve(s).reset_index(drop=True)


def drawdown_stats(equity_or_returns) -> dict:
    """Walk the high-water mark to characterise drawdowns.

    max_drawdown: deepest peak-to-trough decline (negative fraction).
    max_time_under_water_periods: longest run of periods spent below a prior
        peak before a new high is set (or before the series ends).
    n_drawdown_episodes: count of distinct underwater stretches.

    Accepts either an equity/price path or a returns series.
    """
    eq = _to_equity(equity_or_returns)
    if eq.empty:
        return {
            "max_drawdown": 0.0,
            "max_time_under_water_periods": 0,
            "n_drawdown_episodes": 0,
        }
    v = eq.to_numpy(dtype=float)
    peak = v[0]
    max_dd = 0.0
    cur_tuw = 0
    max_tuw = 0
    n_episodes = 0
    in_dd = False
    for x in v:
        if x >= peak:
            peak = x
            # a new high closes any open episode
            if in_dd:
                in_dd = False
            cur_tuw = 0
        else:
            if not in_dd:
                in_dd = True
                n_episodes += 1
            cur_tuw += 1
            if cur_tuw > max_tuw:
                max_tuw = cur_tuw
            dd = x / peak - 1.0
            if dd < max_dd:
                max_dd = dd
    return {
        "max_drawdown": float(max_dd),
        "max_time_under_water_periods": int(max_tuw),
        "n_drawdown_episodes": int(n_episodes),
    }


def _avg_holding_period(positions, periods_per_year: int = TRADING_DAYS) -> float:
    """Average holding period in periods, AFML-style: average book size divided
    by average (one-sided) turnover. NaN if there is no turnover."""
    if positions is None:
        return np.nan
    if isinstance(positions, pd.DataFrame):
        book = positions.abs().sum(axis=1)
        turn = positions.diff().abs().sum(axis=1) / 2.0
    else:
        s = pd.Series(positions, dtype=float)
        book = s.abs()
        turn = s.diff().abs() / 2.0
    avg_turn = float(turn.mean())
    avg_book = float(book.mean())
    if not np.isfinite(avg_turn) or avg_turn <= 0 or not np.isfinite(avg_book):
        return np.nan
    return float(avg_book / avg_turn)


def backtest_stats(returns, positions=None, periods_per_year: int = TRADING_DAYS) -> dict:
    """AFML Ch.14 bundle: return concentration + drawdown structure, plus the
    average holding period when ``positions`` are supplied. Kept SEPARATE from
    ``summarize`` so that headline call stays lean."""
    out = {}
    out.update(returns_concentration(returns))
    out.update(drawdown_stats(returns))
    if positions is not None:
        out["avg_holding_period_periods"] = _avg_holding_period(positions, periods_per_year)
    return out


def summarize(
    returns,
    positions=None,
    periods_per_year: int = TRADING_DAYS,
    n_trials: int = 1,
) -> dict:
    """One call → the full honesty-aware scorecard."""
    r = pd.Series(returns).dropna()
    eq = equity_curve(r)
    # Per-period vol (not annualised) — feeds the empirical-vs-Normal tail read.
    vol_pp = float(r.std(ddof=1)) if r.size > 1 else np.nan
    cvar_95 = conditional_var(r, 0.95)
    cvar_99 = conditional_var(r, 0.99)
    conc = returns_concentration(r)
    out = {
        "n_periods": int(r.size),
        "total_return": float(eq.iloc[-1] - 1.0) if not eq.empty else 0.0,
        "cagr": annualised_return(r, periods_per_year),
        "ann_vol": annualised_vol(r, periods_per_year),
        "ann_sharpe": annualised_sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(r, periods_per_year),
        # Tail risk — historical/empirical, per-period (ADDED; never parametric-Normal).
        "var_95": value_at_risk(r, 0.95),
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "cvar_verdict": _cvar_verdict(cvar_95, cvar_99, vol_pp),
        # Return-concentration HHIs (ADDED; full bundle lives in backtest_stats).
        "positive_hhi": conc["positive_hhi"],
        "negative_hhi": conc["negative_hhi"],
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
