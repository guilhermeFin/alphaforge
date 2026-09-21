import numpy as np
import pandas as pd

from research.robustness_matrix import cost_delay_matrix


def test_cost_delay_matrix_runs_fixed_nonoptimized_scenarios():
    index = pd.bdate_range("2024-01-02", periods=60)
    close = pd.DataFrame({"A": np.linspace(100, 130, len(index)), "B": np.linspace(100, 85, len(index))}, index=index)
    weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=index)
    report = cost_delay_matrix(close, weights, base_cost_bps=5)
    assert len(report["scenarios"]) == 9
    assert {row["execution_lag_days"] for row in report["scenarios"]} == {1, 2, 3}
    assert 0.0 <= report["positive_sharpe_rate"] <= 1.0
