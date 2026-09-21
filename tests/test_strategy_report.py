import json

import numpy as np
import pandas as pd

from research.strategy_report import StrategyReport, build_strategy_report


def test_strategy_report_composes_standard_inputs_and_new_modules():
    rng = np.random.default_rng(12)
    returns = pd.Series(rng.normal(0.0005, 0.01, 80))
    positions = pd.DataFrame({"A": np.linspace(0.0, 0.5, 80), "B": np.linspace(0.0, -0.5, 80)})
    candidates = pd.DataFrame(rng.normal(0.0002, 0.01, size=(80, 3)))
    report = build_strategy_report(
        returns, positions=positions, trades=pd.DataFrame({"trade_id": [1]}),
        benchmark_returns=pd.Series(rng.normal(0.0, 0.01, 80)), candidate_returns=candidates,
        ic_values=pd.Series(rng.normal(0.01, 0.05, 80)), include_white_reality_check=True,
        metadata={"strategy": "test"},
    )
    assert report["inputs"]["has_positions"]
    assert report["inputs"]["has_trade_log"]
    assert report["inputs"]["has_benchmark"]
    assert report["volatility_downside_risk"]["ulcer_index"] is not None
    assert report["benchmark_relative"]["information_ratio"] is not None
    assert report["tail_risk_distribution"]["historical_var_95"] is not None
    assert report["statistical_rigor"]["cpcv_pbo"]["n_strategies"] == 3
    assert "conditional_volatility" in report["advanced_econometrics"]
    assert "regime_analysis" in report
    assert report["temporal_stability"]["status"] == "insufficient_evidence"
    assert "underwater" in report["chart_specs"]["volatility_downside_risk"]
    json.dumps(report)


def test_strategy_report_object_keeps_optional_inputs_optional():
    report = StrategyReport(pd.Series([0.01, -0.01, 0.02])).as_dict()
    assert report["inputs"]["has_positions"] is False
    assert report["benchmark_relative"] is None
    assert report["statistical_rigor"]["cpcv_pbo"] is None
    assert report["temporal_stability"]["status"] == "insufficient_evidence"
