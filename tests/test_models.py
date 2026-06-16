"""Tests for the model FACTORY.

Every model in MODEL_NAMES must:
  * build from make_model,
  * fit a tiny random Xy and predict the right shape,
  * carry the documented regularized defaults,
and stacking must train without base->meta leakage (smoke).
"""
import warnings

import numpy as np
import pytest

# The ML layer is an optional extra (`pip install .[ml]`); skip cleanly if absent.
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")

from sklearn.base import clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor

from research.models import make_model, MODEL_NAMES, RANDOM_STATE


def _tiny_xy(n=200, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # a mostly-linear target so even the linear baselines have something to fit
    beta = np.array([1.0, -0.5, 0.25, 0.0, 0.0])[:p]
    y = X @ beta + 0.1 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_builds_fits_predicts(name):
    X, y = _tiny_xy()
    model = make_model(name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*valid feature names.*")
        model.fit(X, y)
        pred = model.predict(X)
    pred = np.asarray(pred)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred).all()


def test_model_names_is_the_full_ladder():
    assert MODEL_NAMES == [
        "lasso", "elastic_net", "random_forest",
        "xgboost", "lightgbm", "mlp", "stacking",
    ]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_model("transformer")


def test_overrides_are_forwarded():
    m = make_model("random_forest", max_depth=7, n_estimators=11)
    params = m.get_params()
    assert params["max_depth"] == 7
    assert params["n_estimators"] == 11


def test_xgboost_eta_alias_maps_to_learning_rate():
    m = make_model("xgboost", eta=0.5)
    assert m.get_params()["learning_rate"] == 0.5
    # explicit learning_rate wins over eta if both given
    m2 = make_model("xgboost", eta=0.5, learning_rate=0.1)
    assert m2.get_params()["learning_rate"] == 0.1


# --------------------------------------------------------------------------- #
# Regularized defaults present (the discipline the layer is built on)
# --------------------------------------------------------------------------- #
def test_linear_baselines_have_penalties():
    lasso = make_model("lasso")
    en = make_model("elastic_net")
    assert lasso.get_params()["alpha"] > 0
    assert en.get_params()["alpha"] > 0
    assert 0 < en.get_params()["l1_ratio"] < 1


def test_random_forest_defaults_are_regularized():
    p = make_model("random_forest").get_params()
    assert p["max_depth"] is not None and p["max_depth"] <= 5     # shallow
    assert p["min_samples_leaf"] >= 10                             # large leaves
    assert p["max_features"] in ("sqrt", "log2") or isinstance(p["max_features"], (int, float))
    assert p["random_state"] == RANDOM_STATE


def test_xgboost_defaults_are_regularized():
    p = make_model("xgboost").get_params()
    assert p["learning_rate"] <= 0.1          # small steps
    assert p["max_depth"] <= 4                 # shallow
    assert p["subsample"] < 1.0                # row subsampling
    assert p["colsample_bytree"] < 1.0         # column subsampling
    assert p["reg_lambda"] > 0 and p["reg_alpha"] > 0


def test_lightgbm_defaults_are_regularized():
    p = make_model("lightgbm").get_params()
    assert p["num_leaves"] <= 15               # shallow
    assert p["learning_rate"] <= 0.1
    assert p["subsample"] < 1.0
    assert p["colsample_bytree"] < 1.0
    assert p["min_child_samples"] >= 10


def test_mlp_defaults_have_l2_and_early_stopping():
    p = make_model("mlp").get_params()
    assert p["alpha"] > 0                       # L2 weight decay
    assert p["early_stopping"] is True          # cannot train into memorisation
    assert max(p["hidden_layer_sizes"]) <= 32   # small net


# --------------------------------------------------------------------------- #
# Stacking: structure + no base->meta leakage (smoke)
# --------------------------------------------------------------------------- #
def test_stacking_structure_and_shuffled_inner_cv():
    m = make_model("stacking")
    assert isinstance(m, StackingRegressor)
    names = [n for n, _ in m.estimators]
    assert names == ["elastic_net", "lightgbm"]
    assert isinstance(m.estimators[0][1], ElasticNet)
    assert isinstance(m.estimators[1][1], LGBMRegressor)
    # inner cv must be a SHUFFLED, seeded KFold (documented anti-leak / anti-
    # regime-flip choice), not a bare contiguous integer fold.
    cv = m.cv
    assert getattr(cv, "shuffle", False) is True
    assert getattr(cv, "random_state", None) == RANDOM_STATE


def test_stacking_trains_without_leakage_smoke():
    """Smoke test that stacking fits and predicts, and that the meta-model was
    trained on cross-validated OOF base predictions (sklearn's contract). We assert
    a clean fit + finite predictions and that the meta-model saw exactly the two
    base columns (passthrough=False)."""
    X, y = _tiny_xy(n=300, p=5, seed=1)
    m = make_model("stacking")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*valid feature names.*")
        m.fit(X, y)
        pred = m.predict(X)
    assert np.asarray(pred).shape == (X.shape[0],)
    assert np.isfinite(pred).all()
    # passthrough=False => meta-model has exactly len(base) input features
    assert m.final_estimator_.coef_.shape[0] == 2


def test_models_are_deterministic():
    """Two fits of the same model on the same data must give identical predictions
    (fixed random_state across the ladder)."""
    X, y = _tiny_xy(seed=3)
    for name in MODEL_NAMES:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            a = clone(make_model(name)).fit(X, y).predict(X)
            b = clone(make_model(name)).fit(X, y).predict(X)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-10,
                                   err_msg=name)
