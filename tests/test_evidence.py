import numpy as np
import pandas as pd

from research.evidence import data_integrity_audit, evidence_card


def _inputs(provider="licensed_bundle"):
    close = pd.DataFrame({"A": [100, 101, 102], "B": [100, 99, 101]}, index=pd.bdate_range("2024-01-02", periods=3))
    audit = data_integrity_audit(provider, close, ["A", "B"], {
        "point_in_time_fundamentals": True, "survivorship_free_universe": True,
        "includes_delisted_securities": True, "prices_adjusted_for_corporate_actions": True,
    })
    return audit


def test_synthetic_data_never_receives_research_supported_status():
    audit = _inputs("synthetic")
    card = evidence_card(data_audit=audit, scorecard={"deflated_sr": 0.99}, signal_quality={"ic_tstat": 3.0},
                         out_of_sample={"overfit_warning": False}, walk_forward={"passes": True},
                         pbo={"pbo": 0.1}, robustness={"status": "stable", "summary": "ok"})
    assert card["status"] == "exploratory"


def test_attested_data_and_passing_gates_is_research_supported():
    card = evidence_card(data_audit=_inputs(), scorecard={"deflated_sr": 0.99}, signal_quality={"ic_tstat": 2.5},
                         out_of_sample={"overfit_warning": False}, walk_forward={"passes": True},
                         pbo={"pbo": 0.1}, robustness={"status": "stable", "summary": "ok"})
    assert card["status"] == "research_supported"


def test_missing_oos_gate_is_inconclusive_on_attested_data():
    card = evidence_card(data_audit=_inputs(), scorecard={}, signal_quality={"ic_tstat": 3.0},
                         out_of_sample={"overfit_warning": True}, walk_forward={"passes": False},
                         pbo={"pbo": 0.1}, robustness={"status": "stable", "summary": "ok"})
    assert card["status"] == "inconclusive"
