import numpy as np
import pandas as pd

from research import regimes


def _returns(rows: int = 260) -> pd.Series:
    rng = np.random.default_rng(4)
    values = np.r_[
        rng.normal(-0.003, 0.002, rows // 2),
        rng.normal(0.004, 0.002, rows - rows // 2),
    ]
    return pd.Series(values, index=pd.bdate_range("2023-01-03", periods=rows))


def test_regime_analysis_detects_and_summarizes_historical_breaks():
    result = regimes.point_in_time_regime_analysis(_returns(), min_history=80, lookback=160, refit_every=10)
    assert result["available"]
    assert result["n_refits"] > 0
    assert result["n_detected_breaks"] >= 1
    assert result["current_regime_age_observations"] > 0
    assert result["regime_summary"]


def test_regime_history_is_unchanged_when_future_data_is_appended():
    full = _returns()
    prefix = full.iloc[:180]
    prefix_result = regimes.point_in_time_regime_analysis(prefix, min_history=80, lookback=160, refit_every=10)
    full_result = regimes.point_in_time_regime_analysis(full, min_history=80, lookback=160, refit_every=10)
    full_prefix = [row for row in full_result["chart_data"] if row["date"] <= prefix.index[-1].isoformat()]
    assert prefix_result["chart_data"] == full_prefix


def test_regime_analysis_explains_insufficient_history():
    result = regimes.point_in_time_regime_analysis(_returns(40), min_history=80)
    assert not result["available"]
    assert "80" in result["reason"]


def test_optional_break_detector_absence_is_an_explained_degradation(monkeypatch):
    monkeypatch.setattr(regimes, "_load_ruptures", lambda: None)
    result = regimes.point_in_time_regime_analysis(_returns())
    assert not result["available"]
    assert "optional econometrics" in result["reason"]
