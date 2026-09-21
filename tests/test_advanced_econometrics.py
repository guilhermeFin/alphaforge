import numpy as np
import pandas as pd

from research import advanced_econometrics


def _returns(rows: int = 180, seed: int = 9) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.012, rows), index=pd.bdate_range("2023-01-03", periods=rows))


def test_garch_diagnostic_is_json_safe_and_has_no_return_claim():
    result = advanced_econometrics.conditional_volatility_forecast(_returns())
    assert result["available"]
    assert result["first_day_annualized_volatility"] > 0
    assert len(result["forecast_daily_volatility"]) == 21
    assert "not a return forecast" in result["note"]


def test_candidate_model_tests_reports_spa_and_mcs_for_aligned_candidates():
    returns = _returns(90)
    rng = np.random.default_rng(11)
    candidates = pd.DataFrame({
        "linear": returns + rng.normal(0.0, 0.002, len(returns)),
        "tree": returns + rng.normal(0.0, 0.003, len(returns)),
        "boosted": returns + rng.normal(0.0, 0.004, len(returns)),
    })
    result = advanced_econometrics.candidate_model_tests(candidates, reps=100, seed=5)
    assert result["available"]
    assert set(result["spa"]["p_values"]) == {"lower", "consistent", "upper"}
    assert set(result["model_confidence_set"]["included_models"]).issubset(set(candidates.columns))


def test_candidate_model_tests_requires_a_real_candidate_set():
    result = advanced_econometrics.candidate_model_tests(pd.DataFrame({"only": _returns(50)}))
    assert not result["available"]
    assert "two candidate" in result["reason"]


def test_optional_econometrics_absence_is_an_explained_degradation(monkeypatch):
    monkeypatch.setattr(advanced_econometrics, "_load_arch", lambda: None)
    result = advanced_econometrics.conditional_volatility_forecast(_returns())
    assert not result["available"]
    assert "optional econometrics" in result["reason"]
