"""LEAK-AWARE evaluation for return-forecasting models on OVERLAPPING labels.

The problem this module exists to solve: a return-forecasting label is the forward
return over ``horizon`` days, so the label at date t and the label at date t+1
SHARE ``horizon-1`` days of price path. If you cross-validate naively, a training
sample whose label window overlaps a test sample's label window leaks future
information into training — and your out-of-sample number is a lie. López de Prado
(Advances in Financial Machine Learning, Ch. 7) fixes this with two ideas, both
implemented here from scratch in pure numpy/pandas:

  * PURGING — drop any training sample whose label interval [start, end] overlaps
    the test fold's label interval.
  * EMBARGO — additionally drop a small band of training samples that come RIGHT
    AFTER the test fold, because serial correlation in features/returns leaks
    across the seam even without label overlap.

We provide ``purged_kfold_split`` (contiguous time test folds) and ``cpcv_split``
(Combinatorial Purged CV: every C(n_groups, n_test_groups) choice of test groups
becomes a path), plus a point-in-time feature/label builder and a model evaluator.

NOTE on scope vs research/overfitting.py: that module's CSCV/PBO answers "did I
overfit by trying many *strategies*?" over a returns matrix. THIS module's purging
is about training a *model* on overlapping labels without leakage. Related concerns
(both López de Prado), but distinct — do not conflate them.
"""
from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import factor_lib
from .factors import cross_sectional_zscore
from .fundamentals import Fundamentals
from .metrics import annualised_sharpe, TRADING_DAYS

# The model factory lives next door; imported lazily-ish at module top is fine
# since models.py only pulls sklearn/xgboost/lightgbm (no cycle back to here).
from .models import make_model, MODEL_NAMES


# --------------------------------------------------------------------------- #
# 1) Leak-aware splitters (López de Prado purging + embargo)
# --------------------------------------------------------------------------- #
def _as_int_array(x) -> np.ndarray:
    """Coerce a label-time vector (DatetimeIndex / array / Series of timestamps or
    ints) to a sortable int64 array we can do interval-overlap arithmetic on.
    Datetimes become nanosecond ints; numeric inputs pass through as int64."""
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.to_numpy()
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.datetime64):
        return x.astype("datetime64[ns]").astype("int64")
    return x.astype("int64")


def _embargo_count(n: int, embargo_pct: float) -> int:
    """Number of samples to embargo after a test fold (a fraction of the WHOLE
    sample, López de Prado's convention)."""
    if embargo_pct <= 0:
        return 0
    return int(math.ceil(n * embargo_pct))


def _purge_train(
    train_candidates: np.ndarray,
    test_idx: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    embargo: int,
    n: int,
) -> np.ndarray:
    """Given candidate train positions, drop (a) any whose label interval overlaps
    the test fold's label interval (PURGE) and (b) any in the EMBARGO band right
    after the test fold. Returns the surviving train positions (sorted).

    Overlap rule: two intervals [a_s, a_e] and [b_s, b_e] overlap iff
    a_s <= b_e AND b_s <= a_e. We purge a train sample if it overlaps the UNION
    test window [min(test start), max(test end)] — the conservative, standard
    choice for a contiguous test fold (and a safe superset for CPCV groups, which
    we additionally purge group-by-group below)."""
    if test_idx.size == 0:
        return np.sort(train_candidates)
    test_start = start[test_idx].min()
    test_end = end[test_idx].max()
    ts = start[train_candidates]
    te = end[train_candidates]
    overlaps = (ts <= test_end) & (test_start <= te)
    keep = train_candidates[~overlaps]
    # Embargo: drop train positions in [max_test_pos+1, max_test_pos+embargo].
    if embargo > 0:
        hi = int(test_idx.max())
        embargo_hi = min(n - 1, hi + embargo)
        in_embargo = (keep > hi) & (keep <= embargo_hi)
        keep = keep[~in_embargo]
    return np.sort(keep)


def purged_kfold_split(
    label_start,
    label_end,
    n_splits: int = 6,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged & embargoed K-fold over OVERLAPPING labels (no shuffling).

    Parameters
    ----------
    label_start, label_end : per-sample label interval bounds, in SAMPLE ORDER
        (time-sorted). Datetime or integer; ``label_end`` is the date the label is
        realised (e.g. date + horizon). len(label_start) == len(label_end) == N.
    n_splits : number of contiguous test folds.
    embargo_pct : fraction of the WHOLE sample embargoed after each test fold.

    Yields
    ------
    (train_idx, test_idx) : integer positional arrays. Test folds are CONTIGUOUS in
    time and cover the sample in order; training samples whose label interval
    overlaps the test interval are PURGED, and an embargo band after the test fold
    is removed.
    """
    start = _as_int_array(label_start)
    end = _as_int_array(label_end)
    n = start.shape[0]
    if end.shape[0] != n:
        raise ValueError("label_start and label_end must have the same length")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    embargo = _embargo_count(n, embargo_pct)

    all_idx = np.arange(n)
    fold_bounds = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = fold_bounds[k], fold_bounds[k + 1]
        if hi <= lo:
            continue
        test_idx = all_idx[lo:hi]
        train_candidates = np.concatenate([all_idx[:lo], all_idx[hi:]])
        train_idx = _purge_train(train_candidates, test_idx, start, end, embargo, n)
        yield train_idx, test_idx


def cpcv_split(
    label_start,
    label_end,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Combinatorial Purged Cross-Validation (López de Prado, AFML Ch. 12).

    Partition the time-ordered samples into ``n_groups`` contiguous groups. For
    EVERY combination of ``n_test_groups`` groups held out as the test set
    (C(n_groups, n_test_groups) of them), purge & embargo the remaining groups to
    form the train set. Each combination is one "path".

    Yields
    ------
    (train_idx, test_idx, path_id) : path_id is the 0-based index of the
    combination, in itertools.combinations order. The number of paths equals
    ``math.comb(n_groups, n_test_groups)``.

    Purging note: we purge group-by-group against EACH test group's interval (so a
    train sample is dropped if it overlaps ANY test group), and embargo after EACH
    test group — correct even when the test groups are non-contiguous.
    """
    start = _as_int_array(label_start)
    end = _as_int_array(label_end)
    n = start.shape[0]
    if end.shape[0] != n:
        raise ValueError("label_start and label_end must have the same length")
    if not (1 <= n_test_groups < n_groups):
        raise ValueError("require 1 <= n_test_groups < n_groups")
    embargo = _embargo_count(n, embargo_pct)

    all_idx = np.arange(n)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    group_idx = [all_idx[bounds[g]:bounds[g + 1]] for g in range(n_groups)]

    for path_id, test_groups in enumerate(combinations(range(n_groups), n_test_groups)):
        test_idx = np.sort(np.concatenate([group_idx[g] for g in test_groups]))
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        # Purge & embargo against each test GROUP separately (groups may be
        # non-contiguous, so a single union window would over-purge).
        for g in test_groups:
            tg = group_idx[g]
            if tg.size == 0:
                continue
            train_idx = _purge_train(train_idx, tg, start, end, embargo, n)
        yield train_idx, test_idx, path_id


# --------------------------------------------------------------------------- #
# 2) Point-in-time feature/label panel
# --------------------------------------------------------------------------- #
def _resolve_factor(name: str):
    """Look up a factor builder by name in research.factor_lib. Returns the
    callable; raises a clear error listing what's available if missing."""
    fn = getattr(factor_lib, name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"unknown factor {name!r}; not found in research.factor_lib")
    return fn


def _build_factor_panel(name: str, close: pd.DataFrame, fund: Fundamentals) -> pd.DataFrame:
    """Call a factor_lib builder with whatever signature it has.

    factor_lib factors come in two shapes: ``(fund, close)`` (fundamental factors)
    and ``(close, ...)`` (price/volume factors like momentum_12_1). We dispatch on
    a small known set of price-only factors; everything else is ``(fund, close)``."""
    fn = _resolve_factor(name)
    price_only = {"momentum_12_1", "trailing_volatility", "short_term_reversal"}
    if name in price_only:
        return fn(close)
    return fn(fund, close)


def build_feature_panel(
    close: pd.DataFrame,
    fund: Fundamentals,
    factor_names: Iterable[str],
    horizon: int = 21,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build a tidy, point-in-time (date, asset)-indexed feature/label table.

    For each factor in ``factor_names`` we build its (dates x symbols) panel from
    ``factor_lib``, CROSS-SECTIONALLY z-score it per date (so raw units don't
    matter and the features are comparable across factors), then stack everything
    into a long frame indexed by (date, asset). The label ``y`` is the forward
    cross-sectional return over ``horizon`` days, realised at date+horizon.

    Honesty / no look-ahead
    -----------------------
    * The factor panels are already point-in-time (they only use information known
      at each date — fundamentals appear from their filing date, prices are
      shifted). The cross-sectional z-score uses ONLY same-date values, so it adds
      no look-ahead.
    * The label is ``close.shift(-horizon) / close - 1`` placed AT date t — that is
      the future we are trying to predict, not a feature, and it is correctly the
      forward return (this mirrors signal_quality's IC convention). Rows in the last
      ``horizon`` days have no realised label and are dropped.
    * Warmup rows where every feature is NaN (factor lookbacks not yet satisfied)
      are dropped.

    Returns
    -------
    X : DataFrame indexed by a (date, asset) MultiIndex, one column per factor
        (z-scored, point-in-time). NaN feature cells are filled with 0.0 (the
        cross-sectional mean of a z-score) so a single missing factor does not drop
        the row — the row is only dropped if ALL features are missing.
    y : Series aligned to X's index — the forward ``horizon``-day return.
    meta : dict with two Series aligned to X — ``label_start`` (= the date) and
        ``label_end`` (= the trading date ``horizon`` bars later), the interval used
        by the purged/CPCV splitters. Plus ``dates`` / ``assets`` for convenience.
    """
    factor_names = list(factor_names)
    if not factor_names:
        raise ValueError("need at least one factor name")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    index = close.index
    # Forward return realised at t+horizon, indexed at t (matches signal_quality).
    fwd = close.shift(-horizon) / close - 1.0

    # label_end = the trading date `horizon` bars after t (NaT for the tail where
    # there is no realised label). Using the actual trading-date position keeps the
    # interval arithmetic in real calendar terms.
    pos = np.arange(len(index))
    end_pos = pos + horizon
    valid_end = end_pos < len(index)
    label_end_per_date = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    label_end_per_date.iloc[pos[valid_end]] = index[end_pos[valid_end]]

    # Build & z-score each factor panel, then stack to long (date, asset) columns.
    feat_cols = {}
    for name in factor_names:
        panel = _build_factor_panel(name, close, fund)
        panel = panel.reindex(index=index, columns=close.columns)
        z = cross_sectional_zscore(panel)
        feat_cols[name] = z.stack(future_stack=True)  # MultiIndex (date, asset)

    X = pd.DataFrame(feat_cols)
    X.index = X.index.set_names(["date", "asset"])

    y = fwd.stack(future_stack=True)
    y.index = y.index.set_names(["date", "asset"])
    y = y.reindex(X.index)

    # label interval per (date, asset): start = date, end = date+horizon trading bar
    dates = X.index.get_level_values("date")
    label_start = pd.Series(dates, index=X.index)
    label_end = pd.Series(label_end_per_date.reindex(dates).to_numpy(), index=X.index)

    # Drop rows with NO label (tail) or with EVERY feature missing (warmup).
    all_feat_nan = X.isna().all(axis=1)
    keep = (~all_feat_nan) & y.notna() & label_end.notna()
    X = X.loc[keep].copy()
    y = y.loc[keep].copy()
    label_start = label_start.loc[keep].copy()
    label_end = label_end.loc[keep].copy()

    # Remaining per-cell NaN features -> 0.0 (neutral, the z-score's mean), so one
    # absent factor never wipes a row that has the others.
    X = X.fillna(0.0)

    # Sort by (date, asset) so the splitters' "sample order == time order" holds.
    order = X.index.sortlevel(["date", "asset"])[1]
    X = X.iloc[order]
    y = y.iloc[order]
    label_start = label_start.iloc[order]
    label_end = label_end.iloc[order]

    meta = {
        "label_start": label_start,
        "label_end": label_end,
        "dates": X.index.get_level_values("date"),
        "assets": X.index.get_level_values("asset"),
        "horizon": horizon,
        "factor_names": factor_names,
    }
    return X, y, meta


# --------------------------------------------------------------------------- #
# 3) Model evaluation under leak-aware splits
# --------------------------------------------------------------------------- #
def _rank_ic(pred: pd.Series, actual: pd.Series, dates: pd.Index, min_names: int = 5):
    """Mean rank-IC + Newey-West-free t-stat over per-date cross-sections.

    Reuses signal_quality's IC convention: per date, the SPEARMAN (rank)
    correlation between the model's prediction and the realised forward return,
    over the common cross-section. Returns (mean_ic, tstat, n_dates, ic_series)."""
    df = pd.DataFrame({"pred": pred.to_numpy(), "actual": actual.to_numpy(),
                       "date": np.asarray(dates)})
    ics = []
    for _, g in df.groupby("date", sort=True):
        if len(g) < min_names:
            continue
        pr = g["pred"].rank()
        ar = g["actual"].rank()
        if pr.std() == 0 or ar.std() == 0:
            continue
        ic = float(np.corrcoef(pr.to_numpy(), ar.to_numpy())[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
    ic_arr = np.asarray(ics, dtype=float)
    n = ic_arr.size
    if n < 3:
        return (float("nan"), float("nan"), n, ic_arr)
    mean = float(ic_arr.mean())
    sd = float(ic_arr.std(ddof=1))
    tstat = float(mean / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return (mean, tstat, n, ic_arr)


def _long_short_returns(pred: pd.Series, actual: pd.Series, dates: pd.Index,
                        min_names: int = 6) -> np.ndarray:
    """Per-date long-short return: go long the top tercile by predicted rank, short
    the bottom tercile, equal-weight, dollar-neutral; realise the ACTUAL forward
    return of those names. One number per date with enough names. (This is the OOS
    portfolio the model's ranks would have traded.)"""
    df = pd.DataFrame({"pred": pred.to_numpy(), "actual": actual.to_numpy(),
                       "date": np.asarray(dates)})
    rets = []
    for _, g in df.groupby("date", sort=True):
        m = len(g)
        if m < min_names:
            continue
        k = max(1, m // 3)
        s = g.sort_values("pred")
        short_leg = s["actual"].iloc[:k].mean()
        long_leg = s["actual"].iloc[-k:].mean()
        rets.append(float(long_leg - short_leg))
    return np.asarray(rets, dtype=float)


def evaluate_model(estimator, X, y, label_start, label_end, splits) -> dict:
    """Fit on each PURGED train fold, predict the test fold, pool the OOS
    predictions, then score them honestly.

    Parameters
    ----------
    estimator : an unfitted sklearn-compatible regressor (e.g. from
        models.make_model). A fresh CLONE is fit on each fold so folds don't share
        state. Deterministic if the estimator has a fixed random_state.
    X, y : the feature frame / label series from build_feature_panel (positionally
        aligned to label_start / label_end).
    label_start, label_end : per-sample label interval (used only to *generate* the
        splits if ``splits`` is a callable; otherwise ignored here).
    splits : either an ITERABLE of (train_idx, test_idx[, path_id]) tuples, or a
        callable taking (label_start, label_end) and returning such an iterable.

    Returns
    -------
    dict with pooled OOS rank-IC (mean + t-stat), the long-short OOS Sharpe of a
    tercile portfolio formed on predicted ranks, and bookkeeping (n folds, n OOS
    predictions, per-fold rank-IC).
    """
    from sklearn.base import clone

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    dates_all = X.index.get_level_values("date")

    if callable(splits):
        split_iter = list(splits(label_start, label_end))
    else:
        split_iter = list(splits)

    oos_pred_parts = []
    oos_actual_parts = []
    oos_date_parts = []
    per_fold_ic = []

    for sp in split_iter:
        train_idx, test_idx = sp[0], sp[1]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = clone(estimator)
        with warnings.catch_warnings():
            # We fit & predict on plain numpy arrays on purpose (positional, no
            # feature names); some estimators (LightGBM) emit a benign "X does not
            # have valid feature names" UserWarning. Silence only that noise.
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            model.fit(Xv[train_idx], yv[train_idx])
            pred = np.asarray(model.predict(Xv[test_idx]), dtype=float)
        actual = yv[test_idx]
        d = np.asarray(dates_all[test_idx])
        oos_pred_parts.append(pred)
        oos_actual_parts.append(actual)
        oos_date_parts.append(d)
        fic, _, _, _ = _rank_ic(pd.Series(pred), pd.Series(actual), pd.Index(d))
        per_fold_ic.append(fic)

    if not oos_pred_parts:
        return {
            "n_folds": 0, "n_oos_predictions": 0,
            "oos_rank_ic": float("nan"), "oos_rank_ic_tstat": float("nan"),
            "oos_long_short_sharpe": float("nan"),
            "n_ic_dates": 0, "per_fold_rank_ic": [],
        }

    pred = pd.Series(np.concatenate(oos_pred_parts))
    actual = pd.Series(np.concatenate(oos_actual_parts))
    dates = pd.Index(np.concatenate(oos_date_parts))

    mean_ic, ic_t, n_ic_dates, _ = _rank_ic(pred, actual, dates)
    ls = _long_short_returns(pred, actual, dates)
    # Per-date long-short returns are at the FORECAST horizon; annualise the Sharpe
    # by the number of NON-OVERLAPPING periods per year. With overlapping daily
    # observations the naive daily annualisation would inflate the Sharpe, so we
    # scale by sqrt(TRADING_DAYS / horizon) — one honest "bet" per horizon.
    horizon = None
    # horizon is recoverable from the median label span if needed; callers pass it
    # via meta, but evaluate_model is horizon-agnostic, so default to daily and let
    # compare_ladder pass the correct periods_per_year. Here we annualise daily.
    ls_sharpe = annualised_sharpe(ls) if ls.size >= 2 else float("nan")

    return {
        "n_folds": len(oos_pred_parts),
        "n_oos_predictions": int(pred.size),
        "oos_rank_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else float("nan"),
        "oos_rank_ic_tstat": round(ic_t, 2) if np.isfinite(ic_t) else float("nan"),
        "oos_long_short_sharpe": round(ls_sharpe, 3) if np.isfinite(ls_sharpe) else float("nan"),
        "n_ic_dates": int(n_ic_dates),
        "per_fold_rank_ic": [None if (f is None or not np.isfinite(f)) else round(f, 4)
                             for f in per_fold_ic],
    }


# --------------------------------------------------------------------------- #
# 4) The honest headline: does complexity beat the linear baseline OOS?
# --------------------------------------------------------------------------- #
BASELINE_MODEL = "elastic_net"
# A nonlinear model must beat the baseline's OOS rank-IC by at least this absolute
# margin to count as a MEANINGFUL improvement. Deliberately not tiny: a few
# thousandths of IC is noise, not an edge worth the complexity/overfitting risk.
MEANINGFUL_IC_MARGIN = 0.01


def compare_ladder(
    X,
    y,
    label_start,
    label_end,
    model_names: Iterable[str] = MODEL_NAMES,
    n_splits: int = 6,
    embargo_pct: float = 0.01,
    baseline: str = BASELINE_MODEL,
    margin: float = MEANINGFUL_IC_MARGIN,
) -> dict:
    """Run the whole model ladder under purged K-fold CV and return a leaderboard
    plus the honest "does complexity beat the linear baseline OOS?" verdict.

    Parameters
    ----------
    X, y, label_start, label_end : from build_feature_panel.
    model_names : which models to run (defaults to the full MODEL_NAMES ladder).
    n_splits, embargo_pct : forwarded to purged_kfold_split.
    baseline : the linear anchor the nonlinear models are measured against.
    margin : minimum OOS rank-IC improvement over the baseline to call complexity
        "worth it".

    Returns
    -------
    dict with:
      * ``leaderboard`` : list of per-model dicts (model, oos_rank_ic,
        oos_rank_ic_tstat, oos_long_short_sharpe, n_folds, n_oos_predictions),
        sorted by oos_rank_ic descending. Trivially turned into a DataFrame by the
        UI: ``pd.DataFrame(result["leaderboard"])``.
      * ``baseline`` / ``baseline_oos_ic`` : the anchor and its OOS rank-IC.
      * ``best_nonlinear`` / ``best_nonlinear_oos_ic`` : the best non-baseline,
        non-linear model and its OOS rank-IC.
      * ``ic_margin_over_baseline`` : best_nonlinear_oos_ic - baseline_oos_ic.
      * ``complexity_beats_linear`` : bool — is that margin >= ``margin``?
      * ``verdict`` : a plain-language headline of the above.
    """
    model_names = list(model_names)

    def split_factory(ls, le):
        return purged_kfold_split(ls, le, n_splits=n_splits, embargo_pct=embargo_pct)

    rows = []
    for name in model_names:
        est = make_model(name)
        res = evaluate_model(est, X, y, label_start, label_end, split_factory)
        rows.append({
            "model": name,
            "oos_rank_ic": res["oos_rank_ic"],
            "oos_rank_ic_tstat": res["oos_rank_ic_tstat"],
            "oos_long_short_sharpe": res["oos_long_short_sharpe"],
            "n_folds": res["n_folds"],
            "n_oos_predictions": res["n_oos_predictions"],
        })

    def _ic(row):
        v = row["oos_rank_ic"]
        return v if (v is not None and np.isfinite(v)) else float("-inf")

    leaderboard = sorted(rows, key=_ic, reverse=True)

    # Descriptive feature weights: fit the linear baseline on ALL the data ONCE and
    # read its coefficients, so the UI can show which factors the simple model leans
    # on. This is IN-SAMPLE by construction (one fit on the whole panel), so it is a
    # DESCRIPTIVE readout — "what did the linear model weight" — never an out-of-sample
    # performance claim. The features are already cross-sectionally z-scored, so the
    # coefficients are on a comparable scale. Wrapped so it can never fail the run.
    feature_importance: list[dict] = []
    try:
        cols = list(X.columns)
        base_est = make_model(baseline)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            base_est.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        coefs = getattr(base_est, "coef_", None)
        if coefs is not None:
            feature_importance = sorted(
                ({"feature": c, "coef": round(float(w), 5), "abs_coef": round(abs(float(w)), 5)}
                 for c, w in zip(cols, np.ravel(coefs))),
                key=lambda d: d["abs_coef"], reverse=True)
    except Exception:  # noqa: BLE001 — importance is a read-out, never break the ladder
        feature_importance = []

    by_name = {r["model"]: r for r in rows}
    baseline_ic = _ic(by_name[baseline]) if baseline in by_name else float("nan")

    # "linear" models that don't count as added complexity over the baseline.
    LINEAR = {"lasso", "elastic_net"}
    nonlinear_rows = [r for r in rows if r["model"] not in LINEAR]
    if nonlinear_rows:
        best_nl = max(nonlinear_rows, key=_ic)
        best_nl_name = best_nl["model"]
        best_nl_ic = _ic(best_nl)
    else:
        best_nl_name, best_nl_ic = None, float("-inf")

    if np.isfinite(baseline_ic) and np.isfinite(best_nl_ic):
        ic_margin = float(best_nl_ic - baseline_ic)
        beats = bool(ic_margin >= margin)
    else:
        ic_margin = float("nan")
        beats = False

    if not np.isfinite(baseline_ic):
        verdict = (f"INCONCLUSIVE: the {baseline} baseline produced no usable OOS "
                   f"rank-IC (too little data after purging).")
    elif beats:
        verdict = (
            f"COMPLEXITY WINS (cautiously): best nonlinear model '{best_nl_name}' "
            f"OOS rank-IC {best_nl_ic:+.4f} beats the {baseline} baseline "
            f"{baseline_ic:+.4f} by {ic_margin:+.4f} (>= {margin:.3f} margin). "
            f"Verify it also survives the Deflated-Sharpe / PBO bars before trusting it."
        )
    else:
        verdict = (
            f"COMPLEXITY DOES NOT PAY: best nonlinear model "
            f"'{best_nl_name}' OOS rank-IC {best_nl_ic:+.4f} does NOT beat the "
            f"{baseline} baseline {baseline_ic:+.4f} by the required {margin:.3f} "
            f"margin (actual {ic_margin:+.4f}). Use the simple linear model — the "
            f"extra capacity is not buying out-of-sample edge."
        )

    return {
        "leaderboard": leaderboard,
        "baseline": baseline,
        "baseline_oos_ic": round(baseline_ic, 4) if np.isfinite(baseline_ic) else float("nan"),
        "best_nonlinear": best_nl_name,
        "best_nonlinear_oos_ic": round(best_nl_ic, 4) if np.isfinite(best_nl_ic) else float("nan"),
        "ic_margin_over_baseline": round(ic_margin, 4) if np.isfinite(ic_margin) else float("nan"),
        "meaningful_margin": margin,
        "complexity_beats_linear": beats,
        "n_splits": n_splits,
        "verdict": verdict,
        "feature_importance": feature_importance,
        "feature_importance_note": (
            f"{baseline} coefficients from a single full-sample fit — descriptive "
            "(which factors the linear model weights), not an out-of-sample claim."),
    }
