import numpy as np
import pandas as pd

from research.validation import moving_block_bootstrap, robustness_report


def test_block_bootstrap_is_seed_reproducible_and_has_bounds():
    returns = pd.Series(0.0003 + 0.01 * np.random.default_rng(3).standard_normal(300))
    a = moving_block_bootstrap(returns, samples=100, seed=9)
    b = moving_block_bootstrap(returns, samples=100, seed=9)
    assert a == b
    assert a["available"] is True
    assert 0 <= a["probability_positive_annual_return"] <= 1


def test_robustness_reports_chronological_segments():
    index = pd.bdate_range("2023-01-02", periods=160)
    report = robustness_report(pd.Series(0.0002 + 0.01 * np.random.default_rng(4).standard_normal(160), index=index))
    assert len(report["subperiods"]) == 4
    assert isinstance(report["verdict"], str)
