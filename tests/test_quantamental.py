import numpy as np
import pandas as pd

from research import data, factors, stats_guards
from research.backtest import backtest
from research.quantamental import (
    keyword_extractor, build_signal_panel, make_transcripts,
)
from research.signals_llm import validate_signal


def test_keyword_extractor_directional_and_bounded():
    pos = keyword_extractor("Results were strong; record beat and raised guidance with momentum.")
    neg = keyword_extractor("Results were weak; miss, withdrawing guidance amid headwinds.")
    flat = keyword_extractor("Results were broadly in line; outlook reaffirmed.")
    assert pos["tone"] > 0 and neg["tone"] < 0 and flat["tone"] == 0
    for r in (pos, neg, flat):
        validate_signal(r)  # always within schema bounds
        assert -1.0 <= r["tone"] <= 1.0


def test_signal_panel_is_point_in_time():
    idx = pd.bdate_range("2020-01-01", periods=50)
    docs = pd.DataFrame({"symbol": ["A"], "date": [idx[20]],
                         "text": ["strong record beat raised momentum"]})
    panel = build_signal_panel(docs, idx, ["A", "B"], keyword_extractor, horizon=10)
    assert (panel["A"].iloc[:20] == 0).all()   # nothing before the document date
    assert panel["A"].iloc[20] > 0             # signal appears on the document date
    assert panel["A"].iloc[30] > 0             # persists within the horizon
    assert panel["A"].iloc[31] == 0            # decays to 0 after the horizon
    assert (panel["B"] == 0).all()             # untouched symbol stays flat


def test_pipeline_no_lookahead_truncation_consistency():
    world = data.make_synthetic_world([f"S{i}" for i in range(8)], periods=400,
                                      seed=3, n_events_per_symbol=8)
    docs = make_transcripts(world.events, seed=3)

    def pipe(close, idx):
        raw = build_signal_panel(docs, idx, list(close.columns), keyword_extractor, horizon=40)
        w = factors.long_short_weights(factors.cross_sectional_zscore(raw))
        return backtest(close, w).equity

    full = pipe(world.close, world.close.index)
    for k in (200, 300):
        trunc = pipe(world.close.iloc[:k], world.close.index[:k])
        np.testing.assert_allclose(full.iloc[:k].values, trunc.values, atol=1e-12)


def test_extracted_sentiment_is_not_contemporaneous_leak():
    world = data.make_synthetic_world([f"S{i}" for i in range(10)], periods=600,
                                      seed=9, n_events_per_symbol=14)
    docs = make_transcripts(world.events, seed=9)
    raw = build_signal_panel(docs, world.close.index, world.symbols, keyword_extractor, horizon=63)
    score = factors.cross_sectional_zscore(raw)
    name = world.symbols[0]
    look = stats_guards.lookahead_warning(score[name], world.close[name].pct_change(fill_method=None))
    assert look["suspected_lookahead"] is False
