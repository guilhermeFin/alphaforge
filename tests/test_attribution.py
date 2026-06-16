"""Attribution tests — prove the honesty guards: no look-ahead in the factor
portfolios, a pure-market strategy attributes ~all variance to MKT with beta~1 and
~zero alpha, a factor-leg strategy loads on its factor, the insufficient-data guard
fires, and compact_attribution is plain-JSON serialisable.
"""
import json
import math

import numpy as np
import pandas as pd

from research import data
from research.fundamentals import build_fundamentals
from research import attribution as attr
from research import factor_lib as fl


SYMBOLS = [f"S{i}" for i in range(10)]
PERIODS = 900
SEED = 5


def _build():
    """Small synthetic world -> raw fundamentals -> PIT bundle, reused by tests."""
    world = data.make_synthetic_world(SYMBOLS, periods=PERIODS, seed=SEED)
    obs = data.make_synthetic_raw_fundamentals(world, seed=SEED)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    return world, obs, fund


# ----------------------------- shape / availability -----------------------------
def test_factor_returns_columns_price_only_vs_with_fundamentals():
    world, _, fund = _build()
    # price-only: MKT + UMD
    fr_px = attr.build_factor_returns(world.close)
    assert list(fr_px.columns) == ["MKT", "UMD"]
    assert list(fr_px.index) == list(world.close.index)
    # with fundamentals: the full FF5 + UMD set
    fr_full = attr.build_factor_returns(world.close, fund=fund)
    assert set(fr_full.columns) == {"MKT", "UMD", "SMB", "HML", "RMW", "CMA"}
    # no inf leaked; late cross-section has finite factor returns
    for fr in (fr_px, fr_full):
        late = fr.iloc[-50:].to_numpy()
        finite = late[~np.isnan(late)]
        assert finite.size > 0
        assert np.isfinite(finite).all()


# ----------------------------- the no-look-ahead proof -----------------------------
def test_factor_returns_are_point_in_time_via_truncation():
    """Truncation invariance: factor returns up to date T must be UNCHANGED if you
    truncate the input after T. If any leg peeked into the future, the prefixes
    would diverge. Tests both the price-only and the fundamentals path."""
    world, obs, _ = _build()
    close = world.close

    def build(c, with_fund):
        fund = build_fundamentals(obs, c.index, list(c.columns)) if with_fund else None
        return attr.build_factor_returns(c, fund=fund)

    for with_fund in (False, True):
        full = build(close, with_fund)
        for k in (400, 650):
            trunc = build(close.iloc[:k], with_fund)
            assert list(trunc.columns) == list(full.columns)
            np.testing.assert_allclose(
                full.iloc[:k].to_numpy(), trunc.to_numpy(),
                rtol=0, atol=1e-12, equal_nan=True,
            )


# ----------------------------- pure-market strategy -> MKT beta ~1, ~zero alpha -----------------------------
def test_pure_market_strategy_attributes_to_mkt():
    """A strategy that simply holds the equal-weight market (its return == MKT) must
    attribute ~all variance to MKT with beta ~ 1 and ~zero alpha."""
    world, _, _ = _build()
    fr = attr.build_factor_returns(world.close)
    market = fr["MKT"].dropna()
    # strategy IS the market portfolio's daily return
    res = attr.attribution(market, fr, model="capm")
    assert res["insufficient"] is False
    assert res["factors_used"] == ["MKT"]
    assert abs(res["betas"]["MKT"] - 1.0) < 1e-6
    assert abs(res["alpha_daily"]) < 1e-8
    assert res["r_squared"] > 0.999
    assert res["systematic_var_share"] > 0.999
    assert res["idiosyncratic_var_share"] < 1e-3


def test_pure_market_strategy_under_carhart_still_loads_mkt():
    """Under a richer model the market portfolio still loads ~1 on MKT and ~0 on UMD,
    with ~zero alpha and near-unity systematic share."""
    world, _, _ = _build()
    fr = attr.build_factor_returns(world.close)
    market = fr["MKT"].dropna()
    res = attr.attribution(market, fr, model="carhart4")  # MKT,SMB,HML,UMD; only MKT,UMD avail
    assert res["insufficient"] is False
    assert abs(res["betas"]["MKT"] - 1.0) < 1e-3
    assert abs(res["betas"].get("UMD", 0.0)) < 0.05
    assert abs(res["alpha_annualized"]) < 0.02
    assert res["systematic_var_share"] > 0.98
    # SMB / HML were requested but unavailable price-only -> named as missing
    assert set(res["factors_missing"]) == {"SMB", "HML"}


# ----------------------------- a factor-leg strategy loads on that factor -----------------------------
def test_strategy_built_from_a_factor_leg_loads_on_it():
    """A strategy whose daily return IS the RMW (profitability) leg must load strongly
    positively on RMW under FF5, and far more on RMW than on, say, HML."""
    world, _, fund = _build()
    fr = attr.build_factor_returns(world.close, fund=fund)
    rmw = fr["RMW"].dropna()
    res = attr.attribution(rmw, fr, model="ff5")
    assert res["insufficient"] is False
    assert set(res["factors_used"]) == {"MKT", "SMB", "HML", "RMW", "CMA"}
    # loads ~1 on its own leg, with a strongly significant t-stat
    assert res["betas"]["RMW"] > 0.9
    assert res["beta_tstats"]["RMW"] > 5.0
    # and dominates the other legs
    assert res["betas"]["RMW"] > res["betas"]["HML"]
    assert res["betas"]["RMW"] > res["betas"]["CMA"]
    # near-zero idiosyncratic alpha: the return IS the factor, not skill on top of it
    assert abs(res["alpha_daily"]) < 1e-6
    assert res["systematic_var_share"] > 0.99


def test_momentum_strategy_loads_on_umd():
    """A long-short momentum strategy attributes substantially onto the UMD leg."""
    world, _, fund = _build()
    fr = attr.build_factor_returns(world.close, fund=fund)
    umd = fr["UMD"].dropna()
    res = attr.attribution(umd, fr, model="ff6")
    assert res["insufficient"] is False
    assert res["betas"]["UMD"] > 0.9
    assert res["beta_tstats"]["UMD"] > 5.0


# ----------------------------- insufficient-data guard -----------------------------
def test_insufficient_data_guard_fires_on_short_series():
    world, _, _ = _build()
    fr = attr.build_factor_returns(world.close)
    short = fr["MKT"].dropna().iloc[:30]  # < _MIN_OBS
    res = attr.attribution(short, fr, model="capm")
    assert res["insufficient"] is True
    assert res["alpha_daily"] is None
    assert res["r_squared"] is None
    assert res["n_obs"] < attr._MIN_OBS
    assert "insufficient" in res["verdict"].lower() or "aligned observations" in res["note"].lower()


def test_unknown_model_raises():
    world, _, _ = _build()
    fr = attr.build_factor_returns(world.close)
    try:
        attr.attribution(fr["MKT"], fr, model="nope")
    except ValueError:
        return
    raise AssertionError("unknown model should raise ValueError")


# ----------------------------- compact_attribution: jsonable -----------------------------
def _assert_jsonable_no_nan(obj):
    """Round-trips through json (no numpy scalars survive that) and contains no NaN/inf
    (those must have been emitted as null)."""
    text = json.dumps(obj)  # raises TypeError on a numpy scalar
    reparsed = json.loads(text)

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, float):
            assert math.isfinite(o), f"non-finite float leaked: {o}"
    walk(reparsed)
    return reparsed


def test_compact_attribution_with_fundamentals_is_jsonable():
    world, _, fund = _build()
    bt_signal = fl.gross_profitability(fund, world.close)  # any per-name signal
    from research import factors
    w = factors.long_short_weights(factors.cross_sectional_zscore(bt_signal))
    from research.backtest import backtest
    strat = backtest(world.close, w, cost_bps=5.0).returns

    out = attr.compact_attribution(strat, world.close, fund=fund, model="ff5")
    _assert_jsonable_no_nan(out)
    assert out["fundamentals_used"] is True
    assert set(out["available_factors"]) == {"MKT", "SMB", "HML", "RMW", "CMA", "UMD"}
    assert out["insufficient"] is False
    # every numeric is a plain python type, never numpy
    assert isinstance(out["alpha_daily"], float)
    assert all(isinstance(b, float) for b in out["betas"].values())
    assert isinstance(out["n_obs"], int)


def test_compact_attribution_price_only_falls_back_and_says_so():
    """Price-only (no fund): ff5 cannot build SMB/HML/RMW/CMA, so it falls back to the
    available subset (just MKT for ff5) and explains the fallback in note. Still jsonable."""
    world, _, _ = _build()
    from research import factors
    from research.backtest import backtest
    w = factors.long_short_weights(factors.cross_sectional_zscore(factors.momentum(world.close)))
    strat = backtest(world.close, w, cost_bps=5.0).returns

    out = attr.compact_attribution(strat, world.close, fund=None, model="ff5")
    reparsed = _assert_jsonable_no_nan(out)
    assert out["fundamentals_used"] is False
    assert out["available_factors"] == ["MKT", "UMD"]
    # ff5 wants MKT,SMB,HML,RMW,CMA; only MKT is available price-only
    assert out["factors_used"] == ["MKT"]
    assert set(out["factors_missing"]) == {"SMB", "HML", "RMW", "CMA"}
    assert "price-only" in out["note"].lower()
    assert out["insufficient"] is False  # MKT alone is enough to run (>= 60 obs)


def test_compact_attribution_carhart_price_only_keeps_umd():
    """carhart4 price-only keeps MKT+UMD (UMD is a price-only factor)."""
    world, _, _ = _build()
    from research import factors
    from research.backtest import backtest
    w = factors.long_short_weights(factors.cross_sectional_zscore(factors.momentum(world.close)))
    strat = backtest(world.close, w, cost_bps=5.0).returns

    out = attr.compact_attribution(strat, world.close, fund=None, model="carhart4")
    _assert_jsonable_no_nan(out)
    assert set(out["factors_used"]) == {"MKT", "UMD"}
    assert set(out["factors_missing"]) == {"SMB", "HML"}
