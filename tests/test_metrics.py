import numpy as np
import pandas as pd

from research import metrics


def test_max_drawdown_known():
    eq = pd.Series([1.0, 2.0, 1.5, 3.0])
    assert abs(metrics.max_drawdown(eq) - (-0.25)) < 1e-12


def test_sharpe_signs():
    pos = pd.Series([0.01] * 100)            # constant positive, zero vol-> handled as 0
    assert metrics.annualised_sharpe(pos) == 0.0  # zero std -> 0 by convention
    rng = np.random.default_rng(0)
    good = pd.Series(0.001 + 0.005 * rng.standard_normal(1000))
    assert metrics.annualised_sharpe(good) > 0
    zero = pd.Series(0.005 * rng.standard_normal(1000))
    assert abs(metrics.annualised_sharpe(zero)) < 1.0


def test_psr_bounds_and_behaviour():
    rng = np.random.default_rng(1)
    strong = pd.Series(0.001 + 0.003 * rng.standard_normal(2000))
    psr = metrics.probabilistic_sharpe_ratio(strong, 0.0)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.95  # a genuine, long, positive-Sharpe record clears the bar


def test_deflated_sr_discounts_multiple_trials():
    rng = np.random.default_rng(2)
    r = pd.Series(0.0006 + 0.004 * rng.standard_normal(1500))
    psr = metrics.probabilistic_sharpe_ratio(r, 0.0)
    dsr_many = metrics.deflated_sharpe_ratio(r, n_trials=100)
    assert dsr_many <= psr  # trying 100 strategies must lower confidence


def test_expected_max_sharpe_grows_with_trials():
    a = metrics.expected_max_sharpe_pp(10, 0.05)
    b = metrics.expected_max_sharpe_pp(1000, 0.05)
    assert b > a > 0


def test_summarize_keys():
    rng = np.random.default_rng(3)
    r = pd.Series(0.0005 + 0.01 * rng.standard_normal(500))
    s = metrics.summarize(r, n_trials=5)
    for key in ("ann_sharpe", "max_drawdown", "psr_vs_0", "deflated_sr",
                "skew", "excess_kurtosis", "returns_are_normal"):
        assert key in s


# --------------------------------------------------------------------------
# Task 1 — CVaR / Expected Shortfall (historical, never parametric-Normal).
# --------------------------------------------------------------------------

def test_var_cvar_ordering():
    rng = np.random.default_rng(10)
    r = pd.Series(rng.standard_normal(2000) * 0.01)
    var95 = metrics.value_at_risk(r, 0.95)
    cvar95 = metrics.conditional_var(r, 0.95)
    cvar99 = metrics.conditional_var(r, 0.99)
    # cvar_99 >= cvar_95 >= var_95 >= 0
    assert cvar99 >= cvar95 >= var95 >= 0.0


def test_var_cvar_nan_below_min_obs():
    short = pd.Series(np.linspace(-0.01, 0.01, 19))  # n < 20
    assert np.isnan(metrics.value_at_risk(short))
    assert np.isnan(metrics.conditional_var(short))


def test_cvar99_exceeds_normal_implied_on_fat_tail():
    rng = np.random.default_rng(11)
    # Student-t with low dof => genuinely fat tails.
    raw = rng.standard_t(df=3, size=5000)
    r = pd.Series(raw / raw.std() * 0.01)  # scale to ~1% per-period vol
    vol_pp = float(r.std(ddof=1))
    cvar99 = metrics.conditional_var(r, 0.99)
    normal_es99 = metrics._normal_implied_es(vol_pp, 0.99)
    assert cvar99 > normal_es99  # real tail is fatter than the Gaussian fantasy


def test_cvar_verdict_is_string():
    rng = np.random.default_rng(12)
    r = pd.Series(rng.standard_normal(500) * 0.01)
    v = metrics._cvar_verdict(
        metrics.conditional_var(r, 0.95),
        metrics.conditional_var(r, 0.99),
        float(r.std(ddof=1)),
    )
    assert isinstance(v, str) and len(v) > 0


# --------------------------------------------------------------------------
# Task 2 — AFML Ch.14 concentration & drawdown statistics.
# --------------------------------------------------------------------------

def test_hhi_uniform_near_zero():
    uniform = np.ones(100)
    assert abs(metrics.hhi(uniform) - 0.0) < 1e-9


def test_hhi_one_spike_near_one():
    spike = np.zeros(100)
    spike[0] = 1.0
    # one non-zero element => only 1 usable obs => NaN by design; use a dominant spike instead
    spike = np.full(100, 1e-9)
    spike[0] = 1.0
    h = metrics.hhi(spike)
    assert 0.95 < h <= 1.0


def test_hhi_bounds():
    rng = np.random.default_rng(13)
    for _ in range(20):
        x = np.abs(rng.standard_normal(50)) + 1e-6
        h = metrics.hhi(x)
        assert 0.0 <= h <= 1.0


def test_hhi_too_few_obs_nan():
    assert np.isnan(metrics.hhi([5.0]))
    assert np.isnan(metrics.hhi([0.0, 0.0]))


def test_returns_concentration_keys_and_bounds():
    rng = np.random.default_rng(14)
    r = pd.Series(rng.standard_normal(500) * 0.01)
    c = metrics.returns_concentration(r)
    for k in ("positive_hhi", "negative_hhi", "hhi_verdict"):
        assert k in c
    assert 0.0 <= c["positive_hhi"] <= 1.0
    assert 0.0 <= c["negative_hhi"] <= 1.0
    assert isinstance(c["hhi_verdict"], str)


def test_drawdown_stats_handbuilt():
    # Equity path (peaks underlined):  1.0  0.9  0.8  1.0  1.2  1.1  0.9
    #   idx0=1.0 peak; idx1=0.9, idx2=0.8 below peak 1.0 -> dd reaches 0.8/1.0-1 = -0.20
    #   idx3=1.0 ties the high (>= peak) -> closes episode 1, TUW = 2 periods (idx1,2)
    #   idx4=1.2 new peak
    #   idx5=1.1, idx6=0.9 below peak 1.2 -> dd reaches 0.9/1.2-1 = -0.25 (the WORST)
    # So max drawdown = -0.25; two underwater episodes, each 2 periods long.
    eq = pd.Series([1.0, 0.9, 0.8, 1.0, 1.2, 1.1, 0.9])
    d = metrics.drawdown_stats(eq)
    assert abs(d["max_drawdown"] - (-0.25)) < 1e-12
    assert d["max_time_under_water_periods"] == 2
    assert d["n_drawdown_episodes"] == 2


def test_drawdown_stats_matches_max_drawdown_helper():
    rng = np.random.default_rng(15)
    r = pd.Series(0.0003 + 0.01 * rng.standard_normal(800))
    eq = metrics.equity_curve(r)
    d = metrics.drawdown_stats(eq)
    assert abs(d["max_drawdown"] - metrics.max_drawdown(eq)) < 1e-9


def test_backtest_stats_bundle():
    rng = np.random.default_rng(16)
    r = pd.Series(0.0003 + 0.01 * rng.standard_normal(400))
    pos = pd.Series(np.sign(rng.standard_normal(400)))
    b = metrics.backtest_stats(r, positions=pos)
    for k in ("positive_hhi", "negative_hhi", "hhi_verdict",
              "max_drawdown", "max_time_under_water_periods",
              "n_drawdown_episodes", "avg_holding_period_periods"):
        assert k in b


# --------------------------------------------------------------------------
# Task 3 — summarize still returns all original keys PLUS the new ones.
# --------------------------------------------------------------------------

def test_summarize_keeps_original_and_adds_new_keys():
    rng = np.random.default_rng(17)
    r = pd.Series(0.0005 + 0.01 * rng.standard_normal(500))
    s = metrics.summarize(r, n_trials=5)
    original = (
        "n_periods", "total_return", "cagr", "ann_vol", "ann_sharpe",
        "sortino", "max_drawdown", "calmar", "hit_rate", "skew",
        "excess_kurtosis", "jarque_bera_p", "psr_vs_0", "deflated_sr",
        "n_trials", "returns_are_normal",
    )
    for k in original:
        assert k in s, f"original key {k} missing"
    new_keys = ("var_95", "cvar_95", "cvar_99", "cvar_verdict",
                "positive_hhi", "negative_hhi")
    for k in new_keys:
        assert k in s, f"new key {k} missing"
    assert s["cvar_99"] >= s["cvar_95"] >= s["var_95"] >= 0.0
    assert isinstance(s["cvar_verdict"], str)
