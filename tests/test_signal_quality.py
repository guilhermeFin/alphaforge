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


def test_ic_sign_stability_stable_signal():
    # A genuinely predictive signal (quality factor) should keep its IC sign across
    # essentially every era -> high consistency, stable True.
    close, q = _world_quality()
    res = sq.ic_sign_stability(q, close, n_windows=6, min_obs_per_window=10)
    assert res["stable"] is True
    assert res["sign_consistency"] >= 0.80
    assert res["dominant_sign"] in (1, -1)
    assert len(res["era_mean_ics"]) == 6


def test_ic_sign_stability_flips_halfway():
    # Construct a signal whose edge reverses at the midpoint: in the first half it
    # equals the forward return (IC ~ +1), in the second half its negation (IC ~ -1).
    # The full-sample IC averages near zero and the per-era signs disagree -> unstable.
    close, _ = _world_quality()
    fwd = sq._forward_return(close, 1)
    flip = fwd.copy()
    half = len(close) // 2
    flip.iloc[half:] = -fwd.iloc[half:]
    res = sq.ic_sign_stability(flip, close, n_windows=6, min_obs_per_window=10)
    assert res["stable"] is False
    assert res["sign_consistency"] < 0.80
    assert "regime flip" in res["verdict"]


def test_ic_sign_stability_short_series_is_none():
    # A panel far too short to produce n_windows * min_obs_per_window IC obs must
    # return stable=None (never a confident False) and an "insufficient" verdict.
    panel = data.make_synthetic_panel([f"S{i}" for i in range(8)], periods=25, seed=3)
    close = panel.close
    rng = np.random.default_rng(3)
    sig = pd.DataFrame(rng.standard_normal(close.shape), index=close.index, columns=close.columns)
    res = sq.ic_sign_stability(sig, close, n_windows=6, min_obs_per_window=10)
    assert res["stable"] is None
    assert res["sign_consistency"] is None
    assert "INSUFFICIENT" in res["verdict"].upper()


def test_compact_scorecard_exposes_sign_stability_keys():
    close, q = _world_quality()
    c = sq.compact_scorecard(q, close)
    for k in ("ic_sign_consistency", "ic_sign_stable", "ic_sign_verdict"):
        assert k in c
    # On the genuinely predictive quality factor the flat stable flag is True.
    assert c["ic_sign_stable"] is True
    assert isinstance(c["ic_sign_verdict"], str)
