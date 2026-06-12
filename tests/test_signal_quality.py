import numpy as np
import pandas as pd

from research import data, fundamentals
from research import signal_quality as sq


def _world_quality(seed=5, periods=1260):
    world = data.make_synthetic_world([f"S{i}" for i in range(15)], periods=periods,
                                      seed=seed, quality_to_drift=0.0016)
    obs = data.make_synthetic_fundamentals(world, seed=seed)
    fund = fundamentals.build_fundamentals(obs, world.close.index, world.symbols)
    return world.close, fundamentals.quality_score(fund)


def test_ic_of_perfect_signal_is_one():
    close, _ = _world_quality()
    fwd = sq._forward_return(close, 1)           # signal == the thing it predicts
    ic = sq.ic_series(fwd, close, horizon=1)
    assert ic.mean() > 0.99                       # IC machinery sanity check


def test_predictive_signal_is_scored_predictive():
    close, q = _world_quality()
    card = sq.scorecard(q, close)
    assert card["headline_ic"]["mean_ic"] > 0
    assert card["headline_ic"]["ic_tstat"] > 2     # statistically significant
    assert card["predictive"] is True
    assert card["quantiles"]["top_minus_bottom"] > 0   # higher signal -> higher return


def test_random_signal_is_not_predictive():
    close, _ = _world_quality()
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.standard_normal(close.shape), index=close.index, columns=close.columns)
    card = sq.scorecard(noise, close, with_quantiles=False)
    assert abs(card["headline_ic"]["mean_ic"]) < 0.05
    assert card["predictive"] is False


def test_ic_decay_structure():
    close, q = _world_quality()
    d = sq.ic_decay(q, close, (1, 5, 21))
    assert set(d.keys()) == {1, 5, 21}
    assert all(v is not None for v in d.values())


def test_extraction_stability_detects_noise():
    deterministic = [[0.5, 0.5, 0.5], [-0.2, -0.2, -0.2]]
    assert sq.extraction_stability(deterministic)["deterministic"] is True
    noisy = [[0.5, 0.7, 0.3], [-0.2, 0.1, -0.5]]
    rep = sq.extraction_stability(noisy)
    assert rep["deterministic"] is False
    assert rep["mean_within_item_std"] > 0


def test_compact_scorecard_keys():
    close, q = _world_quality()
    c = sq.compact_scorecard(q, close)
    for k in ("mean_ic", "ic_tstat", "ic_ir", "ic_decay", "coverage", "significant"):
        assert k in c
