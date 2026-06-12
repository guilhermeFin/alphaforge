import numpy as np
import pandas as pd

from research import data, factors, stats_guards
from research.backtest import backtest
from research.fundamentals import (
    point_in_time_panel, build_fundamentals, quality_score, value_score, earnings_yield,
)


def test_pit_panel_placement_lag_and_staleness():
    idx = pd.bdate_range("2020-01-01", periods=60)
    obs = pd.DataFrame(
        [("A", idx[10], idx[30], "roe", 0.15)],
        columns=["symbol", "period_end", "available_date", "metric", "value"],
    )
    panel = point_in_time_panel(obs, idx, ["A", "B"], "roe", max_staleness=10)
    assert panel["A"].iloc[:30].isna().all()    # nothing before the FILING date (not the period end)
    assert panel["A"].iloc[30] == 0.15          # appears on available_date
    assert panel["A"].iloc[40] == 0.15          # persists within staleness window
    assert np.isnan(panel["A"].iloc[41])        # goes stale after max_staleness
    assert panel["B"].isna().all()              # symbol with no filings stays NaN


def test_reporting_lag_delays_availability():
    world = data.make_synthetic_world([f"S{i}" for i in range(6)], periods=500, seed=2)
    fast = data.make_synthetic_fundamentals(world, reporting_lag_days=30, seed=2)
    slow = data.make_synthetic_fundamentals(world, reporting_lag_days=150, seed=2)
    pf = build_fundamentals(fast, world.close.index, world.symbols)["roe"]
    ps = build_fundamentals(slow, world.close.index, world.symbols)["roe"]
    first_fast = pf.notna().any(axis=1).idxmax()
    first_slow = ps.notna().any(axis=1).idxmax()
    assert first_slow > first_fast   # a longer reporting lag delays first availability


def test_earnings_yield_uses_filed_eps_over_price():
    world = data.make_synthetic_world(["A", "B", "C"], periods=400, seed=1)
    obs = data.make_synthetic_fundamentals(world, seed=1)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    ep = earnings_yield(fund, world.close)
    assert ep.shape == world.close.shape
    # where defined, E/P is positive (eps and price are positive) and finite
    vals = ep.to_numpy()
    assert np.all((vals[~np.isnan(vals)] > 0) & np.isfinite(vals[~np.isnan(vals)]))


def test_quality_pipeline_point_in_time_consistency():
    world = data.make_synthetic_world([f"S{i}" for i in range(8)], periods=600, seed=5)
    obs = data.make_synthetic_fundamentals(world, seed=5)

    def pipe(close):
        fund = build_fundamentals(obs, close.index, list(close.columns))
        w = factors.long_short_weights(quality_score(fund))
        return backtest(close, w).equity

    full = pipe(world.close)
    for k in (300, 450):
        trunc = pipe(world.close.iloc[:k])
        np.testing.assert_allclose(full.iloc[:k].values, trunc.values, atol=1e-12)


def test_quality_signal_not_a_contemporaneous_leak():
    world = data.make_synthetic_world([f"S{i}" for i in range(10)], periods=700, seed=9)
    obs = data.make_synthetic_fundamentals(world, seed=9)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    score = quality_score(fund).fillna(0.0)
    name = world.symbols[0]
    look = stats_guards.lookahead_warning(score[name], world.close[name].pct_change(fill_method=None))
    assert look["suspected_lookahead"] is False
