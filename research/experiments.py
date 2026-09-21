"""Flagship research experiments built on AlphaForge's existing honesty rails.

This module deliberately composes the existing engine instead of creating a
parallel research stack:

* ``model_eval.build_feature_panel`` creates the point-in-time feature/label panel.
* ``model_eval.purged_kfold_split`` supplies the shared OOS timing regime.
* ``econometrics`` supplies pooled HAC and Fama-MacBeth inference.
* ``models`` / ``model_eval.compare_ladder`` supply the ML ladder, anchored on
  ElasticNet by default.
* ``backtest``, ``walkforward``, ``metrics``, ``overfitting`` and the optional
  ``TrialLedger`` score the resulting signals with the same cost, DSR, PBO and
  audit conventions as the rest of AlphaForge.

Synthetic demonstrations validate this harness. They are not alpha claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import math
import warnings

import numpy as np
import pandas as pd

from . import econometrics, model_eval, overfitting
from .backtest import backtest
from .factors import long_short_weights
from .metrics import annualised_sharpe, TRADING_DAYS
from .models import MODEL_NAMES, make_model
from .trial_ledger import TrialLedger
from .walkforward import walk_forward


DEFAULT_ECONOMETRIC_SPECS = ("pooled_ols_hac", "fama_macbeth")
DEFAULT_ML_BASELINE = "elastic_net"


@dataclass(frozen=True)
class OOSPrediction:
    """Out-of-sample predictions aligned to the feature panel's MultiIndex."""

    name: str
    kind: str
    predictions: pd.Series
    n_folds: int
    n_oos_predictions: int
    skipped_folds: int = 0


def _summary_rows(result) -> list[dict]:
    frame = result.summary_frame().reset_index(names="term")
    rows: list[dict] = []
    for row in frame.to_dict("records"):
        rows.append({k: _json_scalar(v) for k, v in row.items()})
    return rows


def _json_scalar(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(v) for v in obj.to_dict("records")]
    return _json_scalar(obj)


def _wide_from_long(series: pd.Series) -> pd.DataFrame:
    if not isinstance(series.index, pd.MultiIndex) or series.index.names != ["date", "asset"]:
        raise ValueError("expected a Series indexed by ['date', 'asset']")
    return series.unstack("asset").sort_index()


def _add_constant_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out.insert(0, "const", 1.0)
    return out


def _predict_from_params(X: pd.DataFrame, params: pd.Series) -> np.ndarray:
    x = _add_constant_frame(X) if "const" in params.index else X
    x = x.reindex(columns=params.index, fill_value=0.0)
    return x.to_numpy(dtype=float) @ params.to_numpy(dtype=float)


def _fit_predict_pooled_ols(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> OOSPrediction:
    parts: list[pd.Series] = []
    skipped = 0
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            skipped += 1
            continue
        try:
            fit = econometrics.ols(y.iloc[train_idx], X.iloc[train_idx], covariance="classic")
            pred = _predict_from_params(X.iloc[test_idx], fit.params)
        except ValueError:
            skipped += 1
            continue
        parts.append(pd.Series(pred, index=X.index[test_idx], name="pooled_ols_hac"))

    predictions = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float, name="pooled_ols_hac")
    return OOSPrediction(
        name="pooled_ols_hac",
        kind="econometric",
        predictions=predictions,
        n_folds=len(parts),
        n_oos_predictions=int(predictions.size),
        skipped_folds=skipped,
    )


def _fit_predict_fama_macbeth(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    min_cross_section: int | None,
    max_lags: int | None,
) -> OOSPrediction:
    parts: list[pd.Series] = []
    skipped = 0
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            skipped += 1
            continue
        train_y = y.iloc[train_idx]
        train_X = X.iloc[train_idx]
        try:
            fit = econometrics.fama_macbeth(
                _wide_from_long(train_y),
                {col: _wide_from_long(train_X[col]) for col in train_X.columns},
                min_cross_section=min_cross_section,
                covariance="newey_west",
                max_lags=max_lags,
            )
            pred = _predict_from_params(X.iloc[test_idx], fit.params)
        except ValueError:
            skipped += 1
            continue
        parts.append(pd.Series(pred, index=X.index[test_idx], name="fama_macbeth"))

    predictions = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float, name="fama_macbeth")
    return OOSPrediction(
        name="fama_macbeth",
        kind="econometric",
        predictions=predictions,
        n_folds=len(parts),
        n_oos_predictions=int(predictions.size),
        skipped_folds=skipped,
    )


def _fit_predict_ml_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> OOSPrediction:
    from sklearn.base import clone

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    parts: list[pd.Series] = []
    skipped = 0
    base = make_model(model_name)
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            skipped += 1
            continue
        model = clone(base)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            model.fit(Xv[train_idx], yv[train_idx])
            pred = np.asarray(model.predict(Xv[test_idx]), dtype=float)
        parts.append(pd.Series(pred, index=X.index[test_idx], name=model_name))

    predictions = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float, name=model_name)
    return OOSPrediction(
        name=model_name,
        kind="ml",
        predictions=predictions,
        n_folds=len(parts),
        n_oos_predictions=int(predictions.size),
        skipped_folds=skipped,
    )


def _predictive_metrics(pred: pd.Series, y: pd.Series) -> dict:
    aligned = pd.concat(
        [pred.rename("pred"), y.rename("actual")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return {
            "oos_rank_ic": None,
            "oos_rank_ic_tstat": None,
            "oos_long_short_sharpe": None,
            "n_ic_dates": 0,
        }
    dates = aligned.index.get_level_values("date")
    mean_ic, ic_t, n_ic_dates, _ = model_eval._rank_ic(
        aligned["pred"], aligned["actual"], dates
    )
    ls = model_eval._long_short_returns(aligned["pred"], aligned["actual"], dates)
    ls_sharpe = annualised_sharpe(ls) if ls.size >= 2 else float("nan")
    return {
        "oos_rank_ic": round(float(mean_ic), 4) if np.isfinite(mean_ic) else None,
        "oos_rank_ic_tstat": round(float(ic_t), 2) if np.isfinite(ic_t) else None,
        "oos_long_short_sharpe": round(float(ls_sharpe), 3) if np.isfinite(ls_sharpe) else None,
        "n_ic_dates": int(n_ic_dates),
    }


def _score_panel(pred: pd.Series, close: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    panel = pred.unstack("asset")
    return panel.reindex(index=close.index, columns=close.columns).fillna(0.0)


def _score_candidate(
    candidate: OOSPrediction,
    close: pd.DataFrame,
    y: pd.Series,
    *,
    gross: float,
    cost_bps: float,
    n_trials: int,
    n_splits: int,
    horizon: int,
    embargo_pct: float,
) -> tuple[dict, pd.Series]:
    metrics = _predictive_metrics(candidate.predictions, y)
    score = _score_panel(candidate.predictions, close)
    weights = long_short_weights(score, gross=gross)
    bt = backtest(close, weights, cost_bps=cost_bps, periods_per_year=TRADING_DAYS)
    embargo_bars = max(1, int(math.ceil(horizon * embargo_pct)))
    try:
        wf = walk_forward(
            close,
            weights,
            n_splits=n_splits,
            cost_bps=cost_bps,
            periods_per_year=TRADING_DAYS,
            purge_bars=int(horizon),
            embargo_bars=embargo_bars,
        )
    except ValueError as exc:
        wf = {"error": str(exc)}
    row = {
        "method": candidate.name,
        "kind": candidate.kind,
        "n_folds": candidate.n_folds,
        "n_oos_predictions": candidate.n_oos_predictions,
        "skipped_folds": candidate.skipped_folds,
        **metrics,
        "costed_scorecard": bt.summary(n_trials=n_trials),
        "walk_forward": wf,
    }
    return row, bt.returns.rename(candidate.name)


def _record_trials(
    ledger: TrialLedger | None,
    *,
    candidates: Iterable[str],
    close: pd.DataFrame,
    factor_names: Iterable[str],
    horizon: int,
    cost_bps: float,
    gross: float,
    declared_n_trials: int,
    request_context: Mapping | None,
) -> dict:
    candidate_names = list(candidates)
    base = dict(request_context or {})
    base.setdefault("provider", "synthetic")
    base.setdefault("symbols", list(close.columns))
    base.setdefault("periods", len(close))
    base.setdefault("start", str(close.index[0].date()))
    base.setdefault("seed", 0)
    base["lookback"] = int(horizon)
    base["skip"] = 0
    base["cost_bps"] = float(cost_bps)
    base["gross"] = float(gross)

    records = []
    if ledger is not None:
        for name in candidate_names:
            req = {
                **base,
                "factor": f"econometrics_ml:{name}:{','.join(factor_names)}",
            }
            records.append(ledger.record(req))
        effective = ledger.effective_n_trials(max(declared_n_trials, len(candidate_names)))
    else:
        effective = max(int(declared_n_trials), len(candidate_names), 1)

    out = {
        "declared_n_trials": int(declared_n_trials),
        "effective_n_trials": int(effective),
        "candidate_trials": len(candidate_names),
        "haircut_was_raised": bool(effective > declared_n_trials),
    }
    if ledger is not None:
        out["records"] = records
        out.update(ledger.snapshot())
    return out


def econometric_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: int,
    min_cross_section: int | None = None,
) -> dict:
    """Run full-sample inference diagnostics on the feature panel.

    These are inference readouts, not OOS performance claims. OOS predictive
    scoring is handled separately by the purged-fold candidate predictions.
    """
    pooled = econometrics.ols(
        y,
        X,
        covariance="newey_west",
        max_lags=max(1, int(horizon)),
    )
    y_wide = _wide_from_long(y)
    features_wide = {col: _wide_from_long(X[col]) for col in X.columns}
    fmb = econometrics.fama_macbeth(
        y_wide,
        features_wide,
        min_cross_section=min_cross_section,
        covariance="newey_west",
        max_lags=max(1, int(horizon)),
    )
    return {
        "pooled_ols_hac": {
            "covariance_type": pooled.covariance_type,
            "nobs": pooled.nobs,
            "r_squared": pooled.r_squared,
            "adjusted_r_squared": pooled.adjusted_r_squared,
            "coefficients": _summary_rows(pooled),
            "note": "Full-sample pooled OLS with HAC standard errors; inference only, not an OOS score.",
        },
        "fama_macbeth": {
            "covariance_type": fmb.covariance_type,
            "n_periods": fmb.n_periods,
            "average_cross_section_n": fmb.average_cross_section_n,
            "coefficients": _summary_rows(fmb),
            "note": "Full-sample Fama-MacBeth coefficient averages with HAC inference; inference only, not an OOS score.",
        },
    }


def run_econometrics_ml_experiment(
    close: pd.DataFrame,
    fund,
    factor_names: Iterable[str],
    *,
    horizon: int = 21,
    model_names: Iterable[str] | None = None,
    econometric_specs: Iterable[str] = DEFAULT_ECONOMETRIC_SPECS,
    n_splits: int = 6,
    embargo_pct: float = 0.01,
    cost_bps: float = 5.0,
    gross: float = 1.0,
    baseline: str = DEFAULT_ML_BASELINE,
    declared_n_trials: int = 1,
    ledger: TrialLedger | None = None,
    request_context: Mapping | None = None,
    min_cross_section: int | None = None,
    pbo_splits: int = 8,
) -> dict:
    """Compare econometric inference with the ML ladder under shared OOS rails.

    Parameters
    ----------
    close, fund, factor_names
        Inputs accepted by ``model_eval.build_feature_panel``.
    horizon
        Forward-return label horizon in trading days.
    model_names
        ML models from ``research.models.MODEL_NAMES``. ``elastic_net`` is added
        if absent so the default linear ML baseline is always present.
    econometric_specs
        Supported values: ``pooled_ols_hac`` and ``fama_macbeth``.
    ledger
        Optional ``TrialLedger``. When supplied, every econometric spec and every
        ML model candidate is recorded as a distinct trial.

    Returns
    -------
    JSON-friendly dict with separated inference diagnostics, predictive OOS
    metrics, costed backtest scorecards, PBO, ML ladder verdict, and trial audit.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    factor_names = tuple(factor_names)
    if not factor_names:
        raise ValueError("need at least one factor")

    if model_names is None:
        model_names = tuple(MODEL_NAMES)
    else:
        model_names = tuple(dict.fromkeys(str(m).strip() for m in model_names if str(m).strip()))
    unknown = [m for m in model_names if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {tuple(MODEL_NAMES)}")
    if baseline not in model_names:
        model_names = (baseline, *model_names)

    econometric_specs = tuple(econometric_specs)
    bad_specs = [s for s in econometric_specs if s not in DEFAULT_ECONOMETRIC_SPECS]
    if bad_specs:
        raise ValueError(f"unknown econometric spec(s) {bad_specs}; choose from {DEFAULT_ECONOMETRIC_SPECS}")

    X, y, meta = model_eval.build_feature_panel(close, fund, factor_names, horizon=horizon)
    splits = list(
        model_eval.purged_kfold_split(
            meta["label_start"],
            meta["label_end"],
            n_splits=n_splits,
            embargo_pct=embargo_pct,
        )
    )

    diagnostics = econometric_diagnostics(
        X,
        y,
        horizon=horizon,
        min_cross_section=min_cross_section,
    )

    ml_ladder = model_eval.compare_ladder(
        X,
        y,
        meta["label_start"],
        meta["label_end"],
        model_names=model_names,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        baseline=baseline,
    )

    candidates: list[OOSPrediction] = []
    if "pooled_ols_hac" in econometric_specs:
        candidates.append(_fit_predict_pooled_ols(X, y, splits))
    if "fama_macbeth" in econometric_specs:
        candidates.append(
            _fit_predict_fama_macbeth(
                X,
                y,
                splits,
                min_cross_section=min_cross_section,
                max_lags=max(1, int(horizon)),
            )
        )
    for name in model_names:
        candidates.append(_fit_predict_ml_model(name, X, y, splits))

    trial_audit = _record_trials(
        ledger,
        candidates=[c.name for c in candidates],
        close=close,
        factor_names=factor_names,
        horizon=horizon,
        cost_bps=cost_bps,
        gross=gross,
        declared_n_trials=declared_n_trials,
        request_context=request_context,
    )
    effective_trials = trial_audit["effective_n_trials"]

    rows = []
    returns_cols = []
    for candidate in candidates:
        row, returns = _score_candidate(
            candidate,
            close,
            y,
            gross=gross,
            cost_bps=cost_bps,
            n_trials=effective_trials,
            n_splits=n_splits,
            horizon=horizon,
            embargo_pct=embargo_pct,
        )
        rows.append(row)
        returns_cols.append(returns)

    def _finite_or_floor(value) -> float:
        if value is None:
            return float("-inf")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        return value if np.isfinite(value) else float("-inf")

    leaderboard = sorted(
        rows,
        key=lambda r: (
            _finite_or_floor(r["oos_rank_ic"]),
            _finite_or_floor(r["costed_scorecard"]["deflated_sr"]),
        ),
        reverse=True,
    )

    try:
        returns_matrix = pd.concat(returns_cols, axis=1).dropna(how="all")
        pbo = overfitting.compact_pbo(
            returns_matrix,
            n_splits=pbo_splits,
            periods_per_year=TRADING_DAYS,
        )
    except Exception as exc:  # noqa: BLE001 - PBO is a read-out, not a run blocker
        pbo = {"error": f"{type(exc).__name__}: {exc}"}

    return _jsonable({
        "meta": {
            "factor_names": list(factor_names),
            "horizon": int(horizon),
            "n_splits": int(n_splits),
            "embargo_pct": float(embargo_pct),
            "cost_bps": float(cost_bps),
            "gross": float(gross),
            "n_symbols": int(close.shape[1]),
            "n_days": int(len(close)),
            "n_samples": int(len(X)),
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "baseline": baseline,
            "synthetic_data_note": "Synthetic results validate the experiment harness; they are not alpha claims.",
        },
        "econometric_inference": diagnostics,
        "ml_ladder": ml_ladder,
        "predictive_oos": {
            "leaderboard": leaderboard,
            "verdict": (
                "Use the highest OOS rank-IC only if its costed DSR/PBO and walk-forward checks also survive; "
                "ElasticNet remains the default ML baseline unless complexity earns its keep."
            ),
        },
        "pbo": pbo,
        "trial_audit": trial_audit,
    })
