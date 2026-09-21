import numpy as np
import pandas as pd

from research.factor_diagnostics import factor_health


def test_factor_health_reports_coverage_turnover_and_quantiles():
    index = pd.bdate_range("2024-01-02", periods=45)
    rng = np.random.default_rng(4)
    returns = 0.01 * rng.standard_normal((45, 10))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=list("ABCDEFGHIJ"))
    signal = pd.DataFrame(np.tile(np.arange(10), (45, 1)), index=index, columns=close.columns)
    report = factor_health(signal, close)
    assert report["available"]
    assert report["coverage"]["average"] == 1.0
    assert report["turnover"]["top_quantile_turnover"] == 0.0
    assert len(report["quantile_returns"]["quantile_mean_fwd_returns"]) == 5
