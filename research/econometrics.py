"""Econometric research primitives for AlphaForge.

This module adds the inference layer that complements AlphaForge's existing
backtesting and machine-learning stack.  It is deliberately small, deterministic,
and dependency-light: NumPy, pandas, and SciPy only.

The core distinction is important:

* predictive models ask whether a feature improves out-of-sample forecasts;
* econometrics asks what relationship the data supports, with uncertainty made
  explicit.

The functions below therefore return coefficients, standard errors, t statistics,
p-values, confidence intervals, and sample metadata rather than a single score.
They do not place trades and do not bypass AlphaForge's existing walk-forward,
transaction-cost, trial-ledger, or multiple-testing guards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RegressionResult:
    """Immutable result from a linear regression.

    ``covariance`` is retained so downstream research can form joint tests without
    re-estimating the model.  All labelled outputs use the same coefficient index.
    """

    params: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    covariance: pd.DataFrame
    residuals: pd.Series
    fitted: pd.Series
    nobs: int
    df_resid: int
    r_squared: float
    adjusted_r_squared: float
    covariance_type: str

    def summary_frame(self) -> pd.DataFrame:
        """Return the coefficient table in a notebook/UI-friendly shape."""
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_error": self.std_errors,
                "t_stat": self.t_stats,
                "p_value": self.p_values,
                "ci_low": self.conf_int["low"],
                "ci_high": self.conf_int["high"],
            }
        )


@dataclass(frozen=True)
class FamaMacBethResult:
    """Second-pass Fama-MacBeth coefficient averages and inference."""

    params: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    period_params: pd.DataFrame
    n_periods: int
    average_cross_section_n: float
    covariance_type: str

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_error": self.std_errors,
                "t_stat": self.t_stats,
                "p_value": self.p_values,
                "ci_low": self.conf_int["low"],
                "ci_high": self.conf_int["high"],
            }
        )


def _as_series(y, name: str = "y") -> pd.Series:
    if isinstance(y, pd.Series):
        out = y.astype(float).copy()
        if out.name is None:
            out.name = name
        return out
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    return pd.Series(arr, name=name)


def _as_frame(X, index: pd.Index) -> pd.DataFrame:
    if isinstance(X, pd.Series):
        name = X.name or "x1"
        out = X.astype(float).to_frame(name)
    elif isinstance(X, pd.DataFrame):
        out = X.astype(float).copy()
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")
        out = pd.DataFrame(arr, columns=[f"x{i + 1}" for i in range(arr.shape[1])])

    if len(out) != len(index):
        raise ValueError("X and y must have the same number of rows")
    # Positional alignment for array-like inputs; label alignment for pandas is
    # handled by assigning y's index only when X did not already carry that index.
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        out.index = index
    return out


def _prepare_xy(y, X, add_constant: bool) -> tuple[pd.Series, pd.DataFrame]:
    ys = _as_series(y)
    xf = _as_frame(X, ys.index)

    if isinstance(X, (pd.Series, pd.DataFrame)):
        joined = pd.concat([ys.rename("__y__"), xf], axis=1, join="inner")
    else:
        joined = pd.concat([ys.rename("__y__"), xf], axis=1)
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise ValueError("no finite observations remain after aligning X and y")

    ys = joined.pop("__y__")
    xf = joined
    if xf.shape[1] == 0:
        raise ValueError("X must contain at least one regressor")

    if add_constant:
        if "const" in xf.columns:
            raise ValueError("X already contains a column named 'const'")
        xf.insert(0, "const", 1.0)

    if len(ys) <= xf.shape[1]:
        raise ValueError("regression requires more observations than coefficients")
    return ys, xf


def _newey_west_covariance(
    X: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    max_lags: int,
) -> np.ndarray:
    """Newey-West HAC covariance with Bartlett kernel and finite-sample scaling."""
    n, k = X.shape
    if max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    if max_lags >= n:
        raise ValueError("max_lags must be smaller than the number of observations")

    xu = X * residuals[:, None]
    meat = xu.T @ xu
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        gamma = xu[lag:].T @ xu[:-lag]
        meat += weight * (gamma + gamma.T)

    # HC1-style small-sample scaling.  Keeping the correction explicit avoids
    # silently understating uncertainty in the relatively short samples common in
    # exploratory quant research.
    scale = n / (n - k)
    return scale * (xtx_inv @ meat @ xtx_inv)


def ols(
    y,
    X,
    *,
    add_constant: bool = True,
    covariance: str = "classic",
    max_lags: int | None = None,
    alpha: float = 0.05,
) -> RegressionResult:
    """Estimate ordinary least squares with classical or Newey-West inference.

    Parameters
    ----------
    y : array-like or Series
        Dependent variable.
    X : array-like, Series, or DataFrame
        Regressors.  Pandas inputs are aligned on their index before missing rows
        are dropped.
    add_constant : bool
        Add an intercept named ``const``.
    covariance : {"classic", "newey_west", "hac"}
        Standard-error estimator.  Coefficients are OLS in all cases.
    max_lags : int, optional
        HAC lag length.  If omitted for Newey-West, use the common automatic rule
        floor(4 * (n / 100) ** (2 / 9)).
    alpha : float
        Two-sided confidence-interval size; 0.05 gives a 95% interval.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    ys, xf = _prepare_xy(y, X, add_constant)
    yv = ys.to_numpy(dtype=float)
    xv = xf.to_numpy(dtype=float)
    n, k = xv.shape

    xtx_inv = np.linalg.pinv(xv.T @ xv)
    beta = xtx_inv @ xv.T @ yv
    fitted = xv @ beta
    resid = yv - fitted
    df_resid = n - k

    cov_key = covariance.lower().replace("-", "_")
    if cov_key == "classic":
        sigma2 = float(resid @ resid) / df_resid
        cov = sigma2 * xtx_inv
        cov_name = "classic"
    elif cov_key in {"newey_west", "hac"}:
        if max_lags is None:
            max_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        cov = _newey_west_covariance(xv, resid, xtx_inv, int(max_lags))
        cov_name = f"newey_west({int(max_lags)})"
    else:
        raise ValueError("covariance must be 'classic', 'newey_west', or 'hac'")

    variances = np.clip(np.diag(cov), a_min=0.0, a_max=None)
    se = np.sqrt(variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid)
    critical = stats.t.ppf(1.0 - alpha / 2.0, df=df_resid)

    y_centered = yv - yv.mean()
    ss_tot = float(y_centered @ y_centered)
    ss_res = float(resid @ resid)
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    adj_r2 = (
        np.nan
        if np.isnan(r2)
        else 1.0 - (1.0 - r2) * (n - 1) / df_resid
    )

    names = xf.columns
    params = pd.Series(beta, index=names, name="coef")
    std_errors = pd.Series(se, index=names, name="std_error")
    t_series = pd.Series(t_stats, index=names, name="t_stat")
    p_series = pd.Series(p_values, index=names, name="p_value")
    conf = pd.DataFrame(
        {
            "low": beta - critical * se,
            "high": beta + critical * se,
        },
        index=names,
    )

    return RegressionResult(
        params=params,
        std_errors=std_errors,
        t_stats=t_series,
        p_values=p_series,
        conf_int=conf,
        covariance=pd.DataFrame(cov, index=names, columns=names),
        residuals=pd.Series(resid, index=ys.index, name="residual"),
        fitted=pd.Series(fitted, index=ys.index, name="fitted"),
        nobs=n,
        df_resid=df_resid,
        r_squared=float(r2),
        adjusted_r_squared=float(adj_r2),
        covariance_type=cov_name,
    )


def rolling_ols(
    y,
    X,
    window: int,
    *,
    min_periods: int | None = None,
    add_constant: bool = True,
    covariance: str = "classic",
    max_lags: int | None = None,
) -> pd.DataFrame:
    """Run trailing-window regressions without peeking beyond each timestamp.

    The output is indexed by each window END date.  This makes the result suitable
    for regime analysis and for constructing point-in-time coefficient features;
    callers must still respect AlphaForge's normal signal shift before trading.
    """
    if window < 3:
        raise ValueError("window must be at least 3 observations")
    if min_periods is None:
        min_periods = window
    if min_periods < 3 or min_periods > window:
        raise ValueError("min_periods must be between 3 and window")

    ys = _as_series(y)
    xf = _as_frame(X, ys.index)
    if isinstance(X, (pd.Series, pd.DataFrame)):
        joined = pd.concat([ys.rename("__y__"), xf], axis=1, join="inner")
    else:
        joined = pd.concat([ys.rename("__y__"), xf], axis=1)

    rows: list[pd.Series] = []
    row_index: list[object] = []
    for end in range(min_periods, len(joined) + 1):
        start = max(0, end - window)
        chunk = joined.iloc[start:end]
        clean = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < min_periods:
            continue
        res = ols(
            clean["__y__"],
            clean.drop(columns="__y__"),
            add_constant=add_constant,
            covariance=covariance,
            max_lags=max_lags,
        )
        row = res.params.copy()
        row["r_squared"] = res.r_squared
        row["nobs"] = float(res.nobs)
        rows.append(row)
        row_index.append(chunk.index[-1])

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, index=pd.Index(row_index, name=joined.index.name))
    return out


def fama_macbeth(
    y: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    *,
    add_constant: bool = True,
    min_cross_section: int | None = None,
    covariance: str = "newey_west",
    max_lags: int | None = None,
    alpha: float = 0.05,
) -> FamaMacBethResult:
    """Two-pass Fama-MacBeth cross-sectional regression.

    ``y`` and every feature are wide date x asset panels.  For each date, the
    function regresses the cross-section of ``y`` on the contemporaneous feature
    values, then averages those period coefficients.  Inference is performed on
    the coefficient time series; Newey-West is the default because factor premia
    can be serially correlated.

    AlphaForge intentionally does not shift features here.  The research question
    determines whether ``y`` is same-period, next-period, or residual return.  A
    caller testing prediction should construct a forward-return label explicitly
    and then pass it in, making the timing choice auditable.
    """
    if not isinstance(y, pd.DataFrame) or y.empty:
        raise ValueError("y must be a non-empty date x asset DataFrame")
    if not features:
        raise ValueError("features must contain at least one panel")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    common_dates = y.index
    common_assets = y.columns
    aligned: dict[str, pd.DataFrame] = {}
    for name, panel in features.items():
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"feature {name!r} must be a DataFrame")
        common_dates = common_dates.intersection(panel.index)
        common_assets = common_assets.intersection(panel.columns)
    if len(common_dates) == 0 or len(common_assets) == 0:
        raise ValueError("y and features have no overlapping dates/assets")

    yy = y.loc[common_dates, common_assets].astype(float)
    for name, panel in features.items():
        aligned[name] = panel.loc[common_dates, common_assets].astype(float)

    k = len(features) + int(add_constant)
    if min_cross_section is None:
        min_cross_section = max(k + 2, 5)
    if min_cross_section <= k:
        raise ValueError("min_cross_section must exceed the number of coefficients")

    period_rows: list[pd.Series] = []
    period_index: list[object] = []
    cross_n: list[int] = []
    feature_names = list(features.keys())

    for date in common_dates:
        frame = pd.DataFrame({"__y__": yy.loc[date]})
        for name in feature_names:
            frame[name] = aligned[name].loc[date]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < min_cross_section:
            continue
        cross = ols(
            frame["__y__"],
            frame[feature_names],
            add_constant=add_constant,
            covariance="classic",
        )
        period_rows.append(cross.params)
        period_index.append(date)
        cross_n.append(cross.nobs)

    if len(period_rows) < 3:
        raise ValueError("need at least 3 valid cross-sectional periods")

    period_params = pd.DataFrame(period_rows, index=pd.Index(period_index, name=y.index.name))
    means = period_params.mean(axis=0)
    t_count = len(period_params)

    cov_key = covariance.lower().replace("-", "_")
    if cov_key == "classic":
        cov_mean = period_params.cov(ddof=1) / t_count
        cov_name = "classic"
    elif cov_key in {"newey_west", "hac"}:
        if max_lags is None:
            max_lags = int(np.floor(4 * (t_count / 100.0) ** (2.0 / 9.0)))
        demeaned = period_params - means
        z = demeaned.to_numpy(dtype=float)
        q = z.shape[1]
        meat = z.T @ z
        for lag in range(1, int(max_lags) + 1):
            weight = 1.0 - lag / (int(max_lags) + 1.0)
            gamma = z[lag:].T @ z[:-lag]
            meat += weight * (gamma + gamma.T)
        # Covariance of the SAMPLE MEAN: long-run covariance divided by T^2.
        cov_mean = pd.DataFrame(meat / (t_count ** 2), index=means.index, columns=means.index)
        cov_name = f"newey_west({int(max_lags)})"
    else:
        raise ValueError("covariance must be 'classic', 'newey_west', or 'hac'")

    if isinstance(cov_mean, pd.DataFrame):
        cov_values = cov_mean.to_numpy(dtype=float)
    else:
        cov_values = np.asarray(cov_mean, dtype=float)
    se = np.sqrt(np.clip(np.diag(cov_values), a_min=0.0, a_max=None))
    t_stats = means.to_numpy(dtype=float) / se
    df = t_count - 1
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df)
    critical = stats.t.ppf(1.0 - alpha / 2.0, df=df)

    std_errors = pd.Series(se, index=means.index, name="std_error")
    t_series = pd.Series(t_stats, index=means.index, name="t_stat")
    p_series = pd.Series(p_values, index=means.index, name="p_value")
    conf = pd.DataFrame(
        {
            "low": means.to_numpy(dtype=float) - critical * se,
            "high": means.to_numpy(dtype=float) + critical * se,
        },
        index=means.index,
    )

    return FamaMacBethResult(
        params=means.rename("coef"),
        std_errors=std_errors,
        t_stats=t_series,
        p_values=p_series,
        conf_int=conf,
        period_params=period_params,
        n_periods=t_count,
        average_cross_section_n=float(np.mean(cross_n)),
        covariance_type=cov_name,
    )


def forward_returns(close: pd.DataFrame | pd.Series, horizon: int = 1) -> pd.DataFrame | pd.Series:
    """Point-in-time explicit forward-return label for econometric/ML research.

    ``horizon=1`` at date t is close[t+1] / close[t] - 1.  Keeping label creation
    explicit prevents an easy class of accidental same-period/forward-period mixups.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return close.shift(-horizon) / close - 1.0
