import numpy as np
import pandas as pd

from research.temporal_stability import chronological_stability_summary


def _series(parts):
    values = np.concatenate(parts)
    return pd.Series(values, index=pd.bdate_range("2020-01-02", periods=len(values)))


def _summary(parts):
    values = _series(parts)
    returns = pd.Series(0.001, index=values.index)
    return chronological_stability_summary(
        values, returns, n_cohorts=4, min_observations_per_cohort=10,
        embargo_bars=1, n_resamples=50,
    )


def test_stable_chronological_relationship_keeps_direction_and_disjoint_cohorts():
    result = _summary([np.full(12, value) for value in (0.04, 0.05, 0.045, 0.042)])
    assert result["status"] == "stable"
    assert result["supports_current_evidence"] is True
    assert result["sign_consistency"] == 1.0
    ranges = [(row["start"], row["end"]) for row in result["cohorts"]]
    assert len(set(ranges)) == 4
    assert result["cohorts"][0]["end"] < result["cohorts"][1]["start"]


def test_weakened_relationship_is_not_reported_as_stable():
    result = _summary([np.full(12, value) for value in (0.10, 0.09, 0.08, 0.03)])
    assert result["status"] == "weakened"
    assert result["supports_current_evidence"] is False
    assert result["latest_mean_ic"] < result["earliest_mean_ic"]


def test_sign_flip_is_unstable_even_when_a_pooled_number_exists():
    result = _summary([np.full(12, value) for value in (0.08, 0.08, -0.06, -0.06)])
    assert result["status"] == "unstable"
    assert result["sign_consistency"] < 0.8
    assert "pooled" in result["conclusion"]


def test_insufficient_history_refuses_a_temporal_conclusion():
    values = _series([np.full(10, 0.05)])
    result = chronological_stability_summary(values, n_cohorts=4, min_observations_per_cohort=10, n_resamples=50)
    assert result["status"] == "insufficient_evidence"
    assert result["cohorts"] == []
    assert result["supports_current_evidence"] is False


def test_bootstrap_difference_is_centered_on_latest_minus_earliest():
    result = _summary([np.full(12, value) for value in (0.10, 0.08, 0.06, 0.02)])
    low, high = result["latest_minus_earliest_ic_ci_95"]
    assert low <= result["latest_minus_earliest_ic"] <= high
