import numpy as np
import pandas as pd

from research import data
from research.fundamentals import build_fundamentals
from research.signal_quality import ic_summary
from research import factor_lib as fl


SYMBOLS = [f"S{i}" for i in range(8)]
PERIODS = 900
SEED = 8


def _pooled_ic(signal, close, horizons=(1, 5, 21)):
    """Mean of the per-date mean-IC across a few short horizons.

    A single-date cross-sectional IC over only 8 names is very noisy, and the
    synthetic World plants a *persistent* drift signal (quality drives the next
    day's drift, so the edge accumulates over days). Pooling the mean IC across a
    handful of short horizons is therefore the honest, lower-variance way to read
    whether a factor carries the planted signal — without ever using future data
    (each horizon's IC is the signal at t vs the return that FOLLOWS t)."""
    vals = [ic_summary(signal, close, h)["mean_ic"] for h in horizons]
    return float(np.mean([v for v in vals if v is not None]))


def _build():
    """Small synthetic world -> raw fundamentals -> PIT bundle, reused by tests."""
    world = data.make_synthetic_world(SYMBOLS, periods=PERIODS, seed=SEED)
    obs = data.make_synthetic_raw_fundamentals(world, seed=SEED)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    return world, obs, fund


# ----------------------------- helpers -----------------------------
def test_safe_div_zero_is_nan():
    # scalar
    assert np.isnan(fl.safe_div(1.0, 0.0))
    assert np.isnan(fl.safe_div(5.0, 1e-15))   # |den| <= eps treated invalid
    assert fl.safe_div(6.0, 2.0) == 3.0
    # DataFrame path: a zero denominator cell -> NaN, never inf
    num = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    den = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 0.0]})
    out = fl.safe_div(num, den)
    assert np.isnan(out.loc[0, "a"])
    assert np.isnan(out.loc[1, "b"])
    assert out.loc[1, "a"] == 1.0
    assert np.isfinite(out.to_numpy()[~np.isnan(out.to_numpy())]).all()


def test_delta_and_binary():
    a = pd.DataFrame({"x": [3.0, np.nan]})
    b = pd.DataFrame({"x": [1.0, 2.0]})
    d = fl.delta(a, b)
    assert d.loc[0, "x"] == 2.0
    assert np.isnan(d.loc[1, "x"])
    cond = pd.DataFrame({"x": [True, False]})
    assert list(fl.binary(cond)["x"]) == [1.0, 0.0]


# ----------------------------- shape / finiteness -----------------------------
def test_panels_are_dates_x_symbols_and_finite_after_warmup():
    world, _, fund = _build()
    close = world.close
    funcs = [
        fl.gross_profitability, fl.operating_profitability, fl.roa, fl.roe,
        fl.gross_margin, fl.operating_margin, fl.asset_turnover,
        fl.earnings_yield, fl.book_to_price, fl.sales_to_price, fl.fcf_yield,
        fl.leverage, fl.current_ratio, fl.sloan_accruals,
        fl.quality_score, fl.value_score,
    ]
    for fn in funcs:
        panel = fn(fund, close)
        assert panel.shape == close.shape, fn.__name__
        assert list(panel.index) == list(close.index)
        assert list(panel.columns) == list(close.columns)
        # after a generous warmup, the late cross-section has finite values
        late = panel.iloc[-50:]
        vals = late.to_numpy()
        finite = vals[~np.isnan(vals)]
        assert finite.size > 0, fn.__name__
        assert np.isfinite(finite).all(), fn.__name__  # no inf/huge leaked through


def test_growth_and_issuance_panels_align_and_warm_up():
    world, _, fund = _build()
    close = world.close
    for fn in (fl.asset_growth, fl.net_equity_issuance):
        panel = fn(fund, close)
        assert panel.shape == close.shape, fn.__name__
        # first ~252 bars are NaN (need a full year shift)
        assert panel.iloc[:200].isna().all().all(), fn.__name__
        vals = panel.iloc[-50:].to_numpy()
        finite = vals[~np.isnan(vals)]
        assert np.isfinite(finite).all()


def test_price_volume_factors():
    world, _, _ = _build()
    mom = fl.momentum_12_1(world.close)
    illiq = fl.amihud_illiquidity(world.close, world.volume, window=21)
    assert mom.shape == world.close.shape
    assert illiq.shape == world.close.shape
    for panel in (mom, illiq):
        vals = panel.iloc[-50:].to_numpy()
        finite = vals[~np.isnan(vals)]
        assert finite.size > 0
        assert np.isfinite(finite).all()
    # illiquidity is non-negative (|ret| / dollar-vol)
    iv = illiq.to_numpy()
    assert np.all(iv[~np.isnan(iv)] >= 0.0)


# ----------------------------- the honesty claim: quality beats value -----------------------------
def test_gross_profitability_ic_materially_beats_value_composite():
    world, _, fund = _build()
    close = world.close
    gp_ic = _pooled_ic(fl.gross_profitability(fund, close), close)
    val_ic = _pooled_ic(fl.value_score(fund, close), close)
    # quality is PLANTED in the synthetic world (gross margin loads on latent quality,
    # which drives next-day drift); value is built to be ~scale-free noise (the noisy
    # below-EBIT line swamps the quality content of net_income / FCF). So the
    # profitability factor must carry a positive signal materially above value.
    assert gp_ic > 0.0                      # genuine positive signal
    assert gp_ic > val_ic + 0.01            # materially above value (loose threshold)
    # at the raw 1-day horizon the ordering already holds (no future-data needed)
    assert ic_summary(fl.gross_profitability(fund, close), close, 1)["mean_ic"] > \
        ic_summary(fl.value_score(fund, close), close, 1)["mean_ic"]


def test_quality_composite_outranks_value_composite_ic():
    world, _, fund = _build()
    close = world.close
    q_ic = _pooled_ic(fl.quality_score(fund, close), close)
    v_ic = _pooled_ic(fl.value_score(fund, close), close)
    assert q_ic > v_ic


# ----------------------------- point-in-time (inherited from build_fundamentals) -----------------------------
def test_factors_are_point_in_time_via_truncation():
    world, obs, _ = _build()
    close = world.close

    def gp_panel(c):
        fund = build_fundamentals(obs, c.index, list(c.columns))
        return fl.gross_profitability(fund, c)

    full = gp_panel(close)
    for k in (400, 650):
        trunc = gp_panel(close.iloc[:k])
        # the factor on a truncated history must equal the prefix of the full run —
        # i.e. no future filing can change a past value.
        np.testing.assert_allclose(
            full.iloc[:k].to_numpy(), trunc.to_numpy(),
            rtol=0, atol=1e-12, equal_nan=True,
        )


# ----------------------------- distress screens: run, NaN-tolerant -----------------------------
def test_distress_screens_run_nan_tolerant():
    world, _, fund = _build()
    close = world.close
    for fn in (fl.altman_z, fl.ohlson_o, fl.beneish_m):
        panel = fn(fund, close)
        assert panel.shape == close.shape, fn.__name__
        vals = panel.to_numpy()
        finite = vals[~np.isnan(vals)]
        # must not blow up to inf/huge even on partial inputs
        assert np.isfinite(finite).all() if finite.size else True, fn.__name__
