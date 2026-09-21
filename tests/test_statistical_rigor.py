"""Hand-checkable reference tests for Module 4 statistical-rigor analytics."""
import math

import numpy as np
import pandas as pd
from scipy import stats

from research import metrics, overfitting, statistical_rigor as rigor


def test_hac_lag_zero_is_the_usual_sample_mean_tstat():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    expected = values.mean() / (values.std(ddof=1) / math.sqrt(values.size))
    assert rigor.newey_west_mean_tstat(values, max_lag=0) == expected
    assert rigor.hac_sharpe_tstat(values, max_lag=0) == expected
    assert rigor.hac_ic_tstat(values, max_lag=0) == expected


def test_default_hac_lag_has_documented_formula():
    assert rigor.default_hac_lag(100) == 4


def test_psr_delegates_to_existing_bailey_lopez_de_prado_implementation():
    returns = pd.Series([-0.01, 0.01, 0.02, -0.005, 0.015, 0.01])
    got = rigor.statistical_rigor_summary(returns)["probabilistic_sharpe_ratio"]
    assert got == metrics.probabilistic_sharpe_ratio(returns)


def test_minimum_track_record_length_hand_check_for_normal_returns():
    # With SR=.1, SR*=0, skew=0, kurtosis=3, confidence=95%:
    # ceil(1 + (1 + .5*.1^2) * (Phi^-1(.95)/.1)^2).
    expected = math.ceil(1 + (1 + 0.5 * 0.1 ** 2) * (stats.norm.ppf(0.95) / 0.1) ** 2)
    assert rigor.minimum_track_record_length(0.1, confidence=0.95, skew=0.0, kurtosis=3.0) == expected
    assert rigor.minimum_track_record_length(0.0) is None


def test_purged_cpcv_reduces_to_cscv_without_purge_or_embargo():
    rng = np.random.default_rng(4)
    matrix = pd.DataFrame(rng.normal(0.0002, 0.01, size=(160, 4)))
    old = overfitting.cscv_pbo(matrix, n_splits=8).pbo
    new = rigor.purged_cpcv_pbo(matrix, n_splits=8, purge_periods=0, embargo_periods=0)
    assert new.n_combinations == math.comb(8, 4)
    assert new.pbo == old


def test_purged_cpcv_records_purge_and_embargo():
    rng = np.random.default_rng(5)
    matrix = pd.DataFrame(rng.normal(0.0, 0.01, size=(160, 3)))
    result = rigor.purged_cpcv_pbo(matrix, n_splits=8, purge_periods=2, embargo_periods=3)
    assert not result.insufficient
    assert result.purge_periods == 2 and result.embargo_periods == 3
    assert 0.0 <= result.pbo <= 1.0


def test_white_reality_check_all_zero_returns_has_pvalue_one():
    result = rigor.white_reality_check(pd.DataFrame({"a": [0.0] * 20, "b": [0.0] * 20}), n_bootstrap=100)
    assert result["observed_statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_white_reality_check_detects_obvious_best_candidate():
    rng = np.random.default_rng(6)
    matrix = pd.DataFrame({
        "edge": 0.01 + rng.normal(0.0, 0.001, 80),
        "noise": rng.normal(0.0, 0.001, 80),
    })
    result = rigor.white_reality_check(matrix, n_bootstrap=200, seed=7)
    assert result["p_value"] < 0.05


def test_summary_has_search_diagnostics_when_matrix_is_supplied():
    rng = np.random.default_rng(8)
    matrix = pd.DataFrame(rng.normal(0.0, 0.01, size=(80, 3)))
    out = rigor.statistical_rigor_summary(
        matrix.iloc[:, 0], candidate_returns=matrix, include_white_reality_check=True)
    assert out["cpcv_pbo"]["n_strategies"] == 3
    assert out["white_reality_check"]["n_candidates"] == 3


def test_constant_returns_name_the_undefined_minimum_track_record_length():
    """A flat series previously raised and discarded the whole strategy report."""
    out = rigor.statistical_rigor_summary(np.zeros(60))
    assert out["minimum_track_record_length_95"] is None
    assert out["has_undefined_statistics"] is True
    named = {item["statistic"] for item in out["statistic_diagnostics"]}
    assert "minimum_track_record_length_95" in named
    assert "constant" in out["degenerate_returns_note"]
    assert "undefined, not zero" in out["degenerate_returns_note"]


def test_minimum_track_record_length_still_rejects_invalid_input_directly():
    """The primitive keeps its strict contract; only the summary layer softens it."""
    with np.testing.assert_raises_regex(ValueError, "finite"):
        rigor.minimum_track_record_length(float("nan"), confidence=0.95)


def test_healthy_returns_report_no_undefined_statistics():
    rng = np.random.default_rng(11)
    out = rigor.statistical_rigor_summary(rng.normal(0.001, 0.01, 200))
    assert out["has_undefined_statistics"] is False
    assert out["statistic_diagnostics"] == []
    assert "degenerate_returns_note" not in out
