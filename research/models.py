"""Model FACTORY — a thin, documented wrapper over sklearn / xgboost / lightgbm
return-forecasting regressors with DISCIPLINED, regularized defaults.

This is the "anchor" layer of AlphaForge's honest model ladder. Every model here
is configured to *resist* overfitting out of the box: shallow trees, small
learning rates, sub-sampling of rows and columns, explicit L1/L2 penalties, early
stopping for the neural net. The point of the layer is to find out — under the
leak-aware evaluation in research/model_eval.py — whether ADDING COMPLEXITY (trees,
boosting, a net, a stack) actually beats a sparse linear baseline out-of-sample.
If it does not, the honest answer is "use the linear model"; this factory is built
so that a complex model has to *earn* its keep, never to flatter it.

The two sparse linear models (lasso, elastic_net) are the BASELINES the rest of
the ladder is measured against. `MODEL_NAMES` lists the ladder in roughly
increasing complexity.

Design rules
------------
* Defaults are conservative/regularized. A caller who wants to loosen them passes
  ``**overrides`` (forwarded straight to the underlying estimator constructor).
* Every model that has a random seed gets a fixed ``random_state`` so the whole
  ladder is deterministic (the brand: a result you cannot reproduce is a result
  you cannot trust).
* The wrapper is intentionally thin — it returns the genuine sklearn-compatible
  estimator (``fit`` / ``predict`` / ``get_params``), so it drops straight into the
  purged-CV machinery and into sklearn pipelines.
"""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# A single shared seed so the entire ladder is reproducible.
RANDOM_STATE = 0

# The ladder, in roughly increasing complexity. elastic_net is the baseline the
# nonlinear models are judged against in model_eval.compare_ladder.
MODEL_NAMES = [
    "lasso",
    "elastic_net",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
    "stacking",
]


# --------------------------------------------------------------------------- #
# Per-model regularized defaults. Each builder returns a fresh estimator so the
# factory never hands back a shared, already-fitted object.
# --------------------------------------------------------------------------- #
def _lasso(**overrides: Any) -> Lasso:
    """Sparse linear baseline (L1). A small alpha keeps it from zeroing every
    coefficient on a weak-signal panel, but it is still a genuine sparsity prior."""
    params = dict(alpha=0.001, max_iter=20_000, random_state=RANDOM_STATE)
    params.update(overrides)
    return Lasso(**params)


def _elastic_net(**overrides: Any) -> ElasticNet:
    """Sparse linear baseline (L1 + L2). This is THE anchor: the nonlinear models
    in the ladder must beat *this* out-of-sample to justify their complexity."""
    params = dict(alpha=0.001, l1_ratio=0.5, max_iter=20_000, random_state=RANDOM_STATE)
    params.update(overrides)
    return ElasticNet(**params)


def _random_forest(**overrides: Any) -> RandomForestRegressor:
    """Bagged shallow trees. Shallow depth + a high min_samples_leaf + column
    subsampling (max_features) keep each tree from memorising noise."""
    params = dict(
        n_estimators=300,
        max_depth=3,            # shallow on purpose
        max_features="sqrt",    # column subsampling
        min_samples_leaf=50,    # large leaves => smooth, low-variance fits
        n_jobs=1,               # deterministic, no thread-order noise
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return RandomForestRegressor(**params)


def _xgboost(**overrides: Any) -> XGBRegressor:
    """Gradient-boosted trees with heavy regularization: small learning rate,
    shallow trees, row/column subsampling, and explicit L1/L2 leaf penalties.

    Note: ``eta`` is accepted as an alias for ``learning_rate`` (xgboost's native
    name) so callers can use either; if both are given, ``learning_rate`` wins."""
    if "eta" in overrides and "learning_rate" not in overrides:
        overrides = dict(overrides)
        overrides["learning_rate"] = overrides.pop("eta")
    params = dict(
        n_estimators=200,
        learning_rate=0.03,     # small steps
        max_depth=3,            # shallow
        gamma=1.0,              # min loss reduction to split (pruning)
        subsample=0.7,          # row subsampling
        colsample_bytree=0.7,   # column subsampling
        reg_lambda=2.0,         # L2 on leaf weights
        reg_alpha=0.5,          # L1 on leaf weights
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    params.update(overrides)
    return XGBRegressor(**params)


def _lightgbm(**overrides: Any) -> LGBMRegressor:
    """Leaf-wise gradient boosting, regularized: few leaves, small learning rate,
    row bagging + feature sub-sampling, and a large min_child_samples so a leaf
    must be backed by real data."""
    params = dict(
        n_estimators=200,
        num_leaves=7,           # shallow (a depth-3 tree has <= 8 leaves)
        learning_rate=0.03,     # small steps
        subsample=0.7,          # row bagging (a.k.a. bagging_fraction)
        subsample_freq=1,       # ...actually applied every iteration
        colsample_bytree=0.7,   # feature_fraction
        min_child_samples=50,   # large leaves
        reg_lambda=2.0,
        reg_alpha=0.5,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    params.update(overrides)
    return LGBMRegressor(**params)


def _mlp(**overrides: Any) -> MLPRegressor:
    """Small multilayer perceptron with L2 weight decay and EARLY STOPPING on a
    held-out validation slice — the net cannot train itself into memorising the
    panel. Kept deliberately small (two thin hidden layers)."""
    params = dict(
        hidden_layer_sizes=(16, 8),
        alpha=1e-2,             # L2 weight decay (relatively strong)
        early_stopping=True,    # stop when validation score stops improving
        n_iter_no_change=10,
        validation_fraction=0.2,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return MLPRegressor(**params)


def _stacking(**overrides: Any) -> StackingRegressor:
    """Stacking ensemble: base = [elastic_net, lightgbm], meta = Ridge.

    Leakage note (important): sklearn's StackingRegressor trains the META-model on
    CROSS-VALIDATED out-of-fold predictions of the base models (controlled by
    ``cv``), NOT on their in-sample fits. So the meta-learner never sees a base
    prediction that was made on a row the base model was trained on — no leakage
    from base to meta. The OUTER purged/embargoed CV in model_eval is what enforces
    *temporal* leak-awareness across the real train/test boundary; this inner ``cv``
    only governs how the meta-features are generated WITHIN an (already in-sample)
    training set, where every row is mutually in-sample by definition.

    We pass a SHUFFLED, SEEDED ``KFold`` here on purpose. A contiguous (unshuffled)
    inner fold would train each base learner on one stretch of time and produce its
    meta-features on a *different* stretch; on a regime-switching, persistent-quality
    world the cross-sectional relationship drifts between those stretches, so the
    base out-of-fold predictions can come out ANTI-correlated with the label and the
    Ridge meta-model "learns" a negative, non-generalising weight. Shuffling the
    inner folds (rows are all in-sample, so this introduces no train→test leak) gives
    the meta-model honest, representative base predictions. The seed keeps it
    deterministic.

    ``passthrough=False`` => the meta-model sees ONLY the base predictions, keeping
    it a clean, low-dimensional combiner (a Ridge over two columns)."""
    params = dict(cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                  n_jobs=1, passthrough=False)
    params.update(overrides)
    estimators = [
        ("elastic_net", _elastic_net()),
        ("lightgbm", _lightgbm()),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
        **params,
    )


_BUILDERS = {
    "lasso": _lasso,
    "elastic_net": _elastic_net,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "mlp": _mlp,
    "stacking": _stacking,
}


def make_model(name: str, **overrides: Any):
    """Return a fresh, sklearn-compatible regressor with regularized defaults.

    Parameters
    ----------
    name : one of ``MODEL_NAMES``.
    **overrides : forwarded to the underlying estimator constructor, overriding the
        regularized defaults documented per model above.

    Returns
    -------
    A scikit-learn-compatible estimator (``fit`` / ``predict`` / ``get_params``).
    """
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return _BUILDERS[key](**overrides)
