"""Factor-model attribution — how much of a strategy's return is SYSTEMATIC factor
exposure you could buy cheaply, versus genuine idiosyncratic ALPHA.

The honesty claim, in one sentence: a juicy Sharpe that regresses almost entirely
onto Fama-French/Carhart factor-portfolio returns is NOT an edge — it is factor
beta dressed up as skill. This module regresses a strategy's daily net returns on
factor-portfolio returns and reports alpha (the intercept), the factor betas, and
the share of variance that is systematic vs idiosyncratic. A high R-squared with a
near-zero, statistically-insignificant alpha is the engine telling you, plainly,
that your "edge" is mostly cheap beta.

Two honesty guards, inherited from the rest of the engine:
  1. NO LOOK-AHEAD in the factor portfolios. Every leg is formed from a ranking
     signal that is SHIFTED by one day before it trades — exactly like the
     backtester (research/backtest.py). The return earned over (t-1 -> t) is decided
     using only information through t-1. tests/test_attribution.py proves it with the
     same truncation-invariance test used for prices, fundamentals and text signals.
  2. HONEST DEGENERACY GUARDS. Too few observations -> we refuse to report a
     misleading regression (an "insufficient" flag). A factor column that isn't
     available is dropped and named, never silently zero-filled.

Factor portfolios built here (all EQUAL-WEIGHT, daily, point-in-time tercile spreads):
  * MKT  : cross-sectional mean daily return of the universe (equal-weight market
           proxy). Always available (price-only).
  * UMD  : momentum, long top-tercile / short bottom-tercile by 12-1 momentum
           (factor_lib.momentum_12_1). Always available (price-only).
  * SMB  : small-minus-big by market_cap (factor_lib.market_cap). Needs fundamentals.
  * HML  : high-minus-low book_to_price.                          Needs fundamentals.
  * RMW  : robust-minus-weak gross_profitability.                 Needs fundamentals.
  * CMA  : conservative-minus-aggressive = low-minus-high asset_growth. Needs funds.

Pure numpy/pandas/scipy: OLS via numpy.linalg.lstsq, t-stats from the classic OLS
standard errors, p-values via scipy.stats.t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import factor_lib
from .fundamentals import Fundamentals

TRADING_DAYS = 252

# Which factor columns each named model wants, in a stable order. We select the
# AVAILABLE subset at attribution time and name any that were dropped.
MODEL_FACTORS: dict[str, list[str]] = {
    "capm": ["MKT"],
    "ff3": ["MKT", "SMB", "HML"],
    "carhart4": ["MKT", "SMB", "HML", "UMD"],
    "ff5": ["MKT", "SMB", "HML", "RMW", "CMA"],
    "ff6": ["MKT", "SMB", "HML", "RMW", "CMA", "UMD"],
}

_MIN_OBS = 60  # below this an OLS attribution is too noisy to be honest


# ----------------------------- factor-portfolio construction -----------------------------
def _daily_simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Per-name daily simple returns (NaN where price is missing; never filled)."""
    return close.pct_change(fill_method=None)


def _tercile_long_short(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Daily return of a POINT-IN-TIME, equal-weight, top-minus-bottom tercile spread.

    ``signal`` is the ranking signal known at each date's close; ``fwd_ret`` is the
    per-name simple return earned over (t-1 -> t). To avoid look-ahead we form the
    portfolio from YESTERDAY's signal (``signal.shift(1)``) and earn today's return —
    the same one-bar execution lag the backtester enforces. Each day:

      * rank names into terciles by the (shifted) signal across the names that HAVE
        both a signal and a return that day;
      * go equal-weight long the top tercile, equal-weight short the bottom tercile;
      * the leg return is mean(top returns) - mean(bottom returns).

    Days with fewer than 3 jointly-valid names yield NaN (no honest tercile split).
    Returns a daily Series aligned to ``fwd_ret.index``.
    """
    sig = signal.shift(1)  # decide today's book from data known through yesterday
    out = pd.Series(np.nan, index=fwd_ret.index, dtype=float)
    sig_arr = sig.to_numpy()
    ret_arr = fwd_ret.to_numpy()
    for i in range(sig_arr.shape[0]):
        s = sig_arr[i]
        r = ret_arr[i]
        valid = np.isfinite(s) & np.isfinite(r)
        m = int(valid.sum())
        if m < 3:
            continue
        sv = s[valid]
        rv = r[valid]
        k = m // 3  # tercile size; at least 1 when m >= 3
        if k < 1:
            continue
        order = np.argsort(sv, kind="stable")
        bottom = rv[order[:k]]
        top = rv[order[-k:]]
        out.iloc[i] = float(top.mean() - bottom.mean())
    return out


def build_factor_returns(
    close: pd.DataFrame,
    fund: Fundamentals | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Build the DAILY factor-portfolio return panel, point-in-time safe.

    Columns depend on what the inputs can support:
      * MKT, UMD are always built (price-only).
      * If ``fund`` is provided, SMB, HML, RMW, CMA are added as point-in-time tercile
        long-short spreads from market_cap, book_to_price, gross_profitability and
        (negative) asset_growth respectively.

    Every leg is a cross-sectional tercile spread whose ranking signal is shifted by
    one day before it trades, so no factor return on or before date T uses any price
    or filing dated after T. Returns a dates x factors DataFrame aligned to ``close``'s
    index, dtype float, with NaN on the warmup / thin cross-sections (never inf).

    ``periods_per_year`` is accepted for interface symmetry; the factor RETURNS here
    are per-period (daily) and need no annualisation — it is used downstream by
    :func:`attribution` for the alpha annualisation only.
    """
    fwd = _daily_simple_returns(close)

    cols: dict[str, pd.Series] = {}
    # MKT — equal-weight market proxy: the cross-sectional mean daily return. This is
    # already a (t-1 -> t) realised return using only past prices, so it is PIT-safe
    # without an extra shift (no ranking signal is involved).
    cols["MKT"] = fwd.mean(axis=1, skipna=True)

    # UMD — 12-1 momentum tercile spread. momentum_12_1 is past-only by construction;
    # _tercile_long_short shifts it one more bar before trading (belt and braces, and
    # consistent with the FF legs).
    cols["UMD"] = _tercile_long_short(factor_lib.momentum_12_1(close), fwd)

    if fund is not None:
        # SMB = small-minus-big: SMALL caps are the LONG leg, so rank on NEGATIVE
        # market cap (top tercile of -mktcap = smallest names).
        cols["SMB"] = _tercile_long_short(-factor_lib.market_cap(fund, close), fwd)
        # HML = high-minus-low book-to-price (value minus growth).
        cols["HML"] = _tercile_long_short(factor_lib.book_to_price(fund, close), fwd)
        # RMW = robust-minus-weak profitability (high gross profitability long).
        cols["RMW"] = _tercile_long_short(factor_lib.gross_profitability(fund, close), fwd)
        # CMA = conservative-minus-aggressive investment: LOW asset growth is the long
        # leg, so rank on NEGATIVE asset growth.
        cols["CMA"] = _tercile_long_short(-factor_lib.asset_growth(fund, close), fwd)

    factors = pd.DataFrame(cols, index=close.index)
    # Guarantee finiteness: any inf that slipped through becomes NaN (honest "absent").
    return factors.replace([np.inf, -np.inf], np.nan)


# ----------------------------- OLS attribution -----------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Ordinary least squares with an intercept already in ``X`` (first column).

    Returns coefficients, their standard errors and t-stats, the residuals and the
    R-squared. Uses numpy.linalg.lstsq for the fit and the classic
    sigma^2 (X'X)^{-1} covariance for the standard errors (scipy.stats.t for p-values).
    """
    n, p = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - p
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Coefficient covariance: sigma^2 (X'X)^{-1}. Use pinv for numerical safety on a
    # (near-)degenerate design; dof<=0 means no residual freedom -> SEs undefined.
    if dof > 0 and ss_res > 0:
        sigma2 = ss_res / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0, beta / se, np.nan)
        pval = 2.0 * stats.t.sf(np.abs(tstat), dof)
    else:
        se = np.full(p, np.nan)
        tstat = np.full(p, np.nan)
        pval = np.full(p, np.nan)

    return {
        "beta": beta,
        "se": se,
        "tstat": tstat,
        "pval": pval,
        "resid": resid,
        "r_squared": r_squared,
        "dof": dof,
    }


def attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    model: str = "ff5",
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Regress strategy daily returns on factor-portfolio returns; report alpha & betas.

    Parameters
    ----------
    strategy_returns : daily (per-period) net returns of the strategy (a Series).
    factor_returns   : daily factor-portfolio returns (from build_factor_returns).
    model            : one of {"capm","ff3","carhart4","ff5","ff6"}. The available
                       subset of that model's factor columns is used; any requested
                       column absent from ``factor_returns`` is dropped and named in
                       ``note``.
    periods_per_year : for annualising the (per-period) alpha.

    Returns a dict (see module docstring / coordinator wiring) with keys:
      model, alpha_daily, alpha_annualized, alpha_tstat, alpha_pvalue,
      betas {factor: beta}, beta_tstats {factor: t}, beta_pvalues {factor: p},
      r_squared, systematic_var_share, idiosyncratic_var_share,
      n_obs, factors_used, factors_missing, note, insufficient (bool), verdict.

    On too few aligned observations (< ~60) returns the same dict shape with
    ``insufficient=True`` and NaN statistics, rather than a misleading regression.
    """
    model = (model or "ff5").lower()
    if model not in MODEL_FACTORS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODEL_FACTORS)}")

    wanted = MODEL_FACTORS[model]
    available = [c for c in wanted if c in factor_returns.columns]
    missing = [c for c in wanted if c not in factor_returns.columns]

    notes: list[str] = []
    if missing:
        notes.append(
            f"requested factors {missing} not available (need fundamentals) — dropped; "
            f"ran with {available or 'no factors'}"
        )

    def _insufficient(n_obs: int, reason: str) -> dict:
        return {
            "model": model,
            "alpha_daily": None,
            "alpha_annualized": None,
            "alpha_tstat": None,
            "alpha_pvalue": None,
            "betas": {},
            "beta_tstats": {},
            "beta_pvalues": {},
            "r_squared": None,
            "systematic_var_share": None,
            "idiosyncratic_var_share": None,
            "n_obs": int(n_obs),
            "factors_used": list(available),
            "factors_missing": list(missing),
            "insufficient": True,
            "note": "; ".join(notes + [reason]),
            "verdict": "insufficient data for an honest attribution",
        }

    if not available:
        return _insufficient(0, "no factor columns available to regress on")

    # Align strategy and the chosen factor columns on common dates, drop any row with
    # a NaN anywhere (warmup, thin cross-sections). This is the honest common support.
    strat = pd.Series(strategy_returns, dtype=float).rename("strategy")
    aligned = pd.concat([strat, factor_returns[available]], axis=1, join="inner").dropna()
    n_obs = int(aligned.shape[0])
    if n_obs < _MIN_OBS:
        return _insufficient(n_obs, f"only {n_obs} aligned observations (< {_MIN_OBS})")

    y = aligned["strategy"].to_numpy(dtype=float)
    F = aligned[available].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n_obs), F])  # intercept first

    fit = _ols(y, X)
    beta = fit["beta"]
    tstat = fit["tstat"]
    pval = fit["pval"]

    alpha_daily = float(beta[0])
    alpha_t = float(tstat[0]) if np.isfinite(tstat[0]) else None
    alpha_p = float(pval[0]) if np.isfinite(pval[0]) else None
    # Annualise the per-period alpha by compounding: (1 + a)^ppy - 1.
    alpha_ann = float((1.0 + alpha_daily) ** periods_per_year - 1.0)

    betas = {f: float(b) for f, b in zip(available, beta[1:])}
    beta_ts = {f: (float(t) if np.isfinite(t) else None) for f, t in zip(available, tstat[1:])}
    beta_ps = {f: (float(p) if np.isfinite(p) else None) for f, p in zip(available, pval[1:])}

    # Variance shares: systematic = 1 - Var(resid)/Var(strategy), computed honestly off
    # the realised residuals (equals the regression R^2 here, but we compute it from the
    # variances so the relationship is explicit and robust if the model changes).
    var_strat = float(np.var(y, ddof=1)) if n_obs > 1 else np.nan
    var_resid = float(np.var(fit["resid"], ddof=1)) if n_obs > 1 else np.nan
    if np.isfinite(var_strat) and var_strat > 0 and np.isfinite(var_resid):
        systematic = 1.0 - var_resid / var_strat
        idiosyncratic = var_resid / var_strat
    else:
        systematic = np.nan
        idiosyncratic = np.nan

    r2 = fit["r_squared"]

    verdict = _verdict(
        r_squared=r2,
        systematic=systematic,
        alpha_ann=alpha_ann,
        alpha_t=alpha_t,
    )

    return {
        "model": model,
        "alpha_daily": alpha_daily,
        "alpha_annualized": alpha_ann,
        "alpha_tstat": alpha_t,
        "alpha_pvalue": alpha_p,
        "betas": betas,
        "beta_tstats": beta_ts,
        "beta_pvalues": beta_ps,
        "r_squared": (float(r2) if np.isfinite(r2) else None),
        "systematic_var_share": (float(systematic) if np.isfinite(systematic) else None),
        "idiosyncratic_var_share": (float(idiosyncratic) if np.isfinite(idiosyncratic) else None),
        "n_obs": n_obs,
        "factors_used": list(available),
        "factors_missing": list(missing),
        "insufficient": False,
        "note": "; ".join(notes) if notes else "ok",
        "verdict": verdict,
    }


def _verdict(r_squared, systematic, alpha_ann, alpha_t) -> str:
    """Plain-language honesty read on the attribution.

    The key message the brand insists on: a high R-squared with a near-zero,
    statistically-insignificant alpha means the "edge" is mostly factor beta you could
    replicate cheaply. A significant alpha that survives a high R-squared is the rarer,
    genuine idiosyncratic edge.
    """
    if any(x is None or (isinstance(x, float) and not np.isfinite(x))
           for x in (r_squared, systematic, alpha_ann)):
        return "attribution computed, but the variance read is degenerate"
    sig = (alpha_t is not None) and np.isfinite(alpha_t) and abs(alpha_t) >= 2.0
    high_r2 = systematic >= 0.80
    pct_sys = 100.0 * systematic
    head = (
        f"{pct_sys:.0f}% of return variance is systematic factor exposure; "
        f"annualised alpha {alpha_ann * 100:.2f}%"
        + (f" (t={alpha_t:.2f})" if alpha_t is not None and np.isfinite(alpha_t) else "")
    )
    if high_r2 and not sig:
        return (head + " — your 'edge' is mostly factor beta you could buy cheaply; "
                "alpha is not statistically distinguishable from zero")
    if sig and alpha_ann > 0:
        return (head + " — genuine idiosyncratic alpha survives the factor controls "
                "(statistically significant and positive)")
    if not sig:
        return head + " — alpha is not statistically significant; treat any 'edge' with skepticism"
    return head + " — significant alpha, but check sign and robustness"


# ----------------------------- convenience: one-call bundle -----------------------------
def _jsonable(obj):
    """Recursively coerce numpy scalars to Python scalars and non-finite floats to None
    so the result is plain-JSON serialisable (no numpy types; no NaN/inf — emit null)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def compact_attribution(
    strategy_returns: pd.Series,
    close: pd.DataFrame,
    fund: Fundamentals | None = None,
    model: str = "ff5",
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """One call: build the factor returns, run the attribution, return a jsonable dict.

    Robust to the price-only (no-fund) case: if ``fund`` is None the FF legs cannot be
    built, so models that need them (ff3/ff5/ff6/the SMB/HML legs of carhart4) fall
    back to the AVAILABLE subset — for a price-only world that is MKT (and UMD where the
    model includes it). The fallback and any dropped factors are recorded in ``note``.

    The returned dict has NO numpy scalars and NO NaN/inf (non-finite -> null), so it is
    directly JSON-serialisable for the API response / UI panel.
    """
    factor_returns = build_factor_returns(close, fund=fund, periods_per_year=periods_per_year)
    result = attribution(strategy_returns, factor_returns, model=model, periods_per_year=periods_per_year)

    # Surface what was actually available so the coordinator/UI can explain the model.
    result["available_factors"] = list(factor_returns.columns)
    result["fundamentals_used"] = bool(fund is not None)
    if fund is None:
        price_only_note = (
            "price-only world: no fundamentals supplied, so SMB/HML/RMW/CMA were not "
            "built; attribution falls back to the price-only factors (MKT"
            + (", UMD" if "UMD" in MODEL_FACTORS.get(model.lower(), []) else "")
            + ") available for the requested model"
        )
        result["note"] = (
            price_only_note if result.get("note") in (None, "", "ok")
            else f"{result['note']}; {price_only_note}"
        )

    return _jsonable(result)
