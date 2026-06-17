"""Tests for the leak-aware model-evaluation layer.

The headline guarantees, validated by ASSERTION (never by eye):
  * purged_kfold_split produces ZERO train/test label-interval overlap;
  * the embargo removes exactly the right post-test region;
  * cpcv_split yields math.comb(n_groups, n_test_groups) paths;
  * build_feature_panel is point-in-time (truncation invariance);
  * evaluate_model is deterministic.
"""
import math

import numpy as np
import pandas as pd
import pytest

# The ML layer is an optional extra (`pip install .[ml]`); skip cleanly if absent.
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")

from research import data
from research.fundamentals import build_fundamentals
from research import model_eval as me
from research.models import make_model


SYMBOLS = [f"S{i:02d}" for i in range(12)]
PERIODS = 700
SEED = 5
HORIZON = 21


def _build_panel(periods=PERIODS, horizon=HORIZON, factors=None, close=None):
    """Synthetic world -> raw PIT fundamentals -> feature/label panel."""
    factors = factors or ["gross_profitability", "roe", "book_to_price", "momentum_12_1"]
    world = data.make_synthetic_world(SYMBOLS, periods=periods, seed=SEED)
    obs = data.make_synthetic_raw_fundamentals(world, seed=SEED)
    c = world.close if close is None else close
    fund = build_fundamentals(obs, c.index, list(c.columns))
    X, y, meta = me.build_feature_panel(c, fund, factors, horizon=horizon)
    return world, obs, X, y, meta


# --------------------------------------------------------------------------- #
# Purging: zero train/test label-interval overlap
# --------------------------------------------------------------------------- #
def _intervals_overlap(a_s, a_e, b_s, b_e):
    """[a_s,a_e] overlaps [b_s,b_e] iff a_s <= b_e and b_s <= a_e."""
    return (a_s <= b_e) & (b_s <= a_e)


def test_purged_kfold_has_zero_label_overlap():
    """No TRAINING sample's label interval may overlap ANY test sample's label
    interval within the same fold. Asserted exhaustively per fold via a vectorised
    interval-overlap check."""
    _, _, X, _, meta = _build_panel()
    ls = me._as_int_array(meta["label_start"])
    le = me._as_int_array(meta["label_end"])

    n_folds = 0
    for train_idx, test_idx in me.purged_kfold_split(meta["label_start"], meta["label_end"],
                                                     n_splits=6, embargo_pct=0.01):
        n_folds += 1
        assert train_idx.size > 0 and test_idx.size > 0
        test_lo = int(le[test_idx].min())  # min test label-end
        test_hi = int(le[test_idx].max())  # max test label-end
        test_start_min = int(ls[test_idx].min())
        test_start_max = int(ls[test_idx].max())
        # The conservative purge uses the UNION test window [min start, max end].
        union_start, union_end = test_start_min, test_hi
        # Every surviving train sample must NOT overlap the union test window.
        ov = _intervals_overlap(ls[train_idx], le[train_idx], union_start, union_end)
        assert not ov.any(), (
            f"fold had {int(ov.sum())} training samples overlapping the test interval"
        )
    assert n_folds == 6


def test_purged_kfold_test_folds_partition_in_time_order():
    """Test folds are contiguous and, concatenated in fold order, cover the sample
    in time order without shuffling."""
    _, _, X, _, meta = _build_panel()
    n = len(X)
    seen = []
    for _, test_idx in me.purged_kfold_split(meta["label_start"], meta["label_end"], n_splits=5):
        # contiguous
        assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1))
        seen.append(test_idx)
    concat = np.concatenate(seen)
    assert np.array_equal(concat, np.arange(n))  # full, ordered coverage


def test_embargo_removes_exactly_the_post_test_region():
    """With integer labels (horizon=0 so there is NO purge from overlap), the only
    samples removed from the train set after a test fold are exactly the embargo
    band [test_hi+1, test_hi+embargo]."""
    n = 1000
    # Degenerate labels: start == end == position (no interval overlap possible),
    # so purging removes nothing and we isolate the EMBARGO behaviour.
    start = np.arange(n)
    end = np.arange(n)
    n_splits = 5
    embargo_pct = 0.02
    embargo = math.ceil(n * embargo_pct)  # = 20

    folds = list(me.purged_kfold_split(start, end, n_splits=n_splits, embargo_pct=embargo_pct))
    for k, (train_idx, test_idx) in enumerate(folds):
        hi = int(test_idx.max())
        lo = int(test_idx.min())
        # the post-test embargo band that must be ABSENT from train
        band = set(range(hi + 1, min(n, hi + embargo + 1)))
        train_set = set(train_idx.tolist())
        assert band.isdisjoint(train_set), f"fold {k}: embargo band leaked into train"
        # everything OUTSIDE the test fold and OUTSIDE the embargo band must be present
        expected_train = (set(range(0, lo)) | set(range(hi + 1, n))) - band
        assert train_set == expected_train, f"fold {k}: train set != expected"


def test_zero_embargo_keeps_all_non_overlapping_train():
    """With embargo_pct=0 and degenerate (non-overlapping) labels, train = all rows
    outside the test fold (purge removes nothing, embargo removes nothing)."""
    n = 300
    start = np.arange(n)
    end = np.arange(n)
    for train_idx, test_idx in me.purged_kfold_split(start, end, n_splits=6, embargo_pct=0.0):
        expected = np.setdiff1d(np.arange(n), test_idx)
        assert np.array_equal(np.sort(train_idx), expected)


# --------------------------------------------------------------------------- #
# CPCV: combination count + purging
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_groups,n_test", [(6, 2), (5, 2), (8, 3), (6, 1)])
def test_cpcv_yields_n_choose_k_paths(n_groups, n_test):
    n = 400
    start = np.arange(n)
    end = np.arange(n) + 10  # 10-wide overlapping labels
    paths = list(me.cpcv_split(start, end, n_groups=n_groups,
                               n_test_groups=n_test, embargo_pct=0.0))
    assert len(paths) == math.comb(n_groups, n_test)
    # path ids are 0..C-1, unique and in order
    ids = [p[2] for p in paths]
    assert ids == list(range(math.comb(n_groups, n_test)))


def test_cpcv_has_zero_label_overlap_per_test_group():
    """CPCV train samples must not overlap ANY held-out test group's interval."""
    _, _, X, _, meta = _build_panel()
    ls = me._as_int_array(meta["label_start"])
    le = me._as_int_array(meta["label_end"])
    n_groups, n_test = 6, 2
    n = len(X)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(bounds[g], bounds[g + 1]) for g in range(n_groups)]

    seen_paths = 0
    for train_idx, test_idx, path_id in me.cpcv_split(meta["label_start"], meta["label_end"],
                                                      n_groups=n_groups, n_test_groups=n_test,
                                                      embargo_pct=0.01):
        seen_paths += 1
        # which groups are the test groups for this path?
        test_set = set(test_idx.tolist())
        test_groups = [g for g in range(n_groups) if set(groups[g].tolist()) <= test_set]
        for g in test_groups:
            tg = groups[g]
            g_start = int(ls[tg].min())
            g_end = int(le[tg].max())
            ov = _intervals_overlap(ls[train_idx], le[train_idx], g_start, g_end)
            assert not ov.any(), f"path {path_id}: train overlaps test group {g}"
    assert seen_paths == math.comb(n_groups, n_test)


# --------------------------------------------------------------------------- #
# build_feature_panel: shape, labels, point-in-time
# --------------------------------------------------------------------------- #
def test_feature_panel_shape_and_labels():
    _, _, X, y, meta = _build_panel()
    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == ["gross_profitability", "roe", "book_to_price", "momentum_12_1"]
    assert X.index.names == ["date", "asset"]
    assert len(X) == len(y) == len(meta["label_start"]) == len(meta["label_end"])
    # no residual NaN in features (filled to 0) or labels (dropped)
    assert not X.isna().any().any()
    assert not y.isna().any()
    # label_end is strictly after label_start by ~horizon trading days
    spans = (meta["label_end"].to_numpy().astype("datetime64[ns]")
             - meta["label_start"].to_numpy().astype("datetime64[ns]"))
    assert (spans > np.timedelta64(0)).all()


def test_feature_panel_is_point_in_time_via_truncation():
    """Truncation invariance: features built on a truncated price history must equal
    the prefix of the features built on the full history. A future bar cannot change
    a past feature value (no look-ahead)."""
    world, obs, X_full, _, _ = _build_panel()
    close = world.close
    factors = ["gross_profitability", "roe", "book_to_price", "momentum_12_1"]

    for k in (400, 550):
        c = close.iloc[:k]
        fund = build_fundamentals(obs, c.index, list(c.columns))
        Xk, _, _ = me.build_feature_panel(c, fund, factors, horizon=HORIZON)
        # compare the overlap: rows whose date is in the truncated index AND that
        # survived (had a realised label) in BOTH runs.
        common = Xk.index.intersection(X_full.index)
        # Every truncated row that exists must match the full-run value exactly.
        a = Xk.loc[common].to_numpy()
        b = X_full.loc[common].to_numpy()
        assert common.size > 0
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_feature_panel_label_matches_forward_return():
    """The label y at (date, asset) equals the realised horizon-forward return of
    that asset — computed independently here."""
    world, obs, X, y, meta = _build_panel()
    close = world.close
    fwd = close.shift(-HORIZON) / close - 1.0
    fwd_stacked = fwd.stack(future_stack=True)
    fwd_stacked.index = fwd_stacked.index.set_names(["date", "asset"])
    # sample a handful of rows
    sample = X.index[:: max(1, len(X) // 50)]
    for idx in sample:
        assert np.isclose(y.loc[idx], fwd_stacked.loc[idx], atol=1e-12)


# --------------------------------------------------------------------------- #
# evaluate_model: determinism + sane structure
# --------------------------------------------------------------------------- #
def test_evaluate_model_is_deterministic():
    _, _, X, y, meta = _build_panel()
    splits = lambda ls, le: me.purged_kfold_split(ls, le, n_splits=5, embargo_pct=0.01)
    est = make_model("random_forest")
    r1 = me.evaluate_model(est, X, y, meta["label_start"], meta["label_end"], splits)
    r2 = me.evaluate_model(make_model("random_forest"), X, y,
                           meta["label_start"], meta["label_end"], splits)
    assert r1["oos_rank_ic"] == r2["oos_rank_ic"]
    assert r1["oos_long_short_sharpe"] == r2["oos_long_short_sharpe"]
    assert r1["per_fold_rank_ic"] == r2["per_fold_rank_ic"]
    assert r1["n_oos_predictions"] == r2["n_oos_predictions"] == len(X)


def test_evaluate_model_accepts_precomputed_splits_list():
    """evaluate_model should accept a plain list of (train, test) tuples, not only a
    split-factory callable."""
    _, _, X, y, meta = _build_panel()
    splits = list(me.purged_kfold_split(meta["label_start"], meta["label_end"], n_splits=4))
    res = me.evaluate_model(make_model("elastic_net"), X, y,
                            meta["label_start"], meta["label_end"], splits)
    assert res["n_folds"] == 4
    assert np.isfinite(res["oos_rank_ic"])


# --------------------------------------------------------------------------- #
# compare_ladder: shape + the honest verdict on synthetic data
# --------------------------------------------------------------------------- #
def test_compare_ladder_shape_and_verdict_keys():
    _, _, X, y, meta = _build_panel()
    res = me.compare_ladder(X, y, meta["label_start"], meta["label_end"],
                            model_names=["elastic_net", "lightgbm", "random_forest"],
                            n_splits=5)
    # leaderboard structure (so the UI can build a table)
    lb = res["leaderboard"]
    assert isinstance(lb, list) and len(lb) == 3
    cols = {"model", "oos_rank_ic", "oos_rank_ic_tstat",
            "oos_long_short_sharpe", "n_folds", "n_oos_predictions"}
    for row in lb:
        assert cols <= set(row.keys())
    # sorted by oos_rank_ic descending
    ics = [r["oos_rank_ic"] for r in lb]
    assert ics == sorted(ics, reverse=True)
    # the honest headline keys
    for key in ("baseline", "baseline_oos_ic", "best_nonlinear",
                "best_nonlinear_oos_ic", "ic_margin_over_baseline",
                "complexity_beats_linear", "verdict"):
        assert key in res
    assert isinstance(res["complexity_beats_linear"], bool)


def _build_wide_panel(periods=1512, n_symbols=25, horizon=21):
    """A demo-scale world (wide cross-section, long history) where the OOS rank-IC
    is stable enough to make a CONFIDENT verdict — unlike the thin 12-name panel,
    where the OOS IC is so noisy the verdict flips with sample length (a finding the
    layer must not pretend away). See test_compare_ladder_verdict_is_unstable_on_thin_universe."""
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    world = data.make_synthetic_world(symbols, periods=periods, seed=7,
                                      quality_to_drift=0.0012)
    obs = data.make_synthetic_raw_fundamentals(world, seed=7)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    X, y, meta = me.build_feature_panel(
        world.close, fund,
        ["gross_profitability", "roe", "operating_margin",
         "book_to_price", "asset_growth", "momentum_12_1"],
        horizon=horizon,
    )
    return X, y, meta


def test_compare_ladder_reports_feature_importance():
    """compare_ladder exposes the baseline's full-sample coefficients (a DESCRIPTIVE
    read-out for the UI) — one entry per feature, sorted by |coef| descending, plus an
    honesty note that it is in-sample, not an OOS performance claim."""
    _, _, X, y, meta = _build_panel()
    res = me.compare_ladder(X, y, meta["label_start"], meta["label_end"],
                            model_names=["elastic_net", "random_forest"], n_splits=5)
    fi = res["feature_importance"]
    assert isinstance(fi, list) and len(fi) == X.shape[1]
    assert {d["feature"] for d in fi} == set(X.columns)
    abscoefs = [d["abs_coef"] for d in fi]
    assert abscoefs == sorted(abscoefs, reverse=True)  # sorted by importance
    assert "not an out-of-sample claim" in res["feature_importance_note"]


def test_compare_ladder_complexity_does_not_beat_linear_on_synthetic():
    """The brand claim, asserted on the STABLE demo-scale world: the planted edge is
    ~linear in quality, so the boosted/tree/net/stack models do NOT meaningfully beat
    the elastic-net baseline OOS. We assert the honest, complexity-skeptical verdict."""
    X, y, meta = _build_wide_panel()
    res = me.compare_ladder(X, y, meta["label_start"], meta["label_end"], n_splits=6)
    # baseline carries a real positive OOS signal (quality IS planted)
    assert res["baseline_oos_ic"] > 0
    # the linear models sit at/near the top of the leaderboard
    top_two = {row["model"] for row in res["leaderboard"][:2]}
    assert "elastic_net" in top_two
    # complexity does not clear the margin -> use the simple model
    assert res["complexity_beats_linear"] is False
    assert "DOES NOT PAY" in res["verdict"]


def test_compare_ladder_verdict_is_unstable_on_thin_universe():
    """Honesty guard: on a THIN 12-name cross-section the OOS rank-IC is so noisy
    that the complexity_beats_linear verdict is not reproducible across sample
    lengths. We do NOT assert a particular verdict here — we assert only that the
    machinery runs and returns a well-formed verdict. (Documenting, not hiding, that
    a thin universe cannot support a confident model-complexity conclusion.)"""
    _, _, X, y, meta = _build_panel(periods=900)
    res = me.compare_ladder(X, y, meta["label_start"], meta["label_end"],
                            model_names=["elastic_net", "xgboost", "lightgbm", "random_forest"],
                            n_splits=6)
    assert isinstance(res["complexity_beats_linear"], bool)
    assert isinstance(res["verdict"], str) and len(res["verdict"]) > 0
    assert np.isfinite(res["baseline_oos_ic"])
