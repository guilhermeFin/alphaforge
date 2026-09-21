"""UI verification via Streamlit's official AppTest harness — runs app/Home.py
for real (headless, in-process), clicks the Run button, and asserts the page
renders the verdict AND that its numbers are identical to the service layer.
"""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

from api.service import run_backtest_workflow

HOME = str(pathlib.Path(__file__).resolve().parent.parent / "app" / "Home.py")


@pytest.fixture()
def at(monkeypatch):
    # Force direct (in-process) mode: point the app at a dead API port so the
    # test never depends on a running server.
    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    t = AppTest.from_file(HOME, default_timeout=180)
    t.run()
    assert not t.exception
    return t


def _run_button(t: AppTest):
    btns = [b for b in t.button if "Run backtest" in b.label]
    assert btns, "Run button not found"
    return btns[0]


def test_initial_render(at):
    assert "Strategy lab" in at.title[0].value
    assert any("in-process" in str(i.value) for i in at.markdown)


def test_click_runs_and_renders_verdict(at):
    _run_button(at).click()
    at.run()
    assert not at.exception

    # the verdict banner is st.success (credible) or st.error (not credible)
    banners = [e.value for e in at.warning] + [s.value for s in at.success]
    assert any("statistical check" in b for b in banners), banners
    assert any("Synthetic-data demonstration" in b for b in banners)

    # headline metrics rendered
    labels = [m.label for m in at.metric]
    for expect in ("Annual return", "Sharpe ratio", "Largest drawdown", "OOS confidence"):
        assert expect in labels
    for expect in ("Information ratio", "Tracking error", "Jensen alpha", "Historical VaR (95%)"):
        assert expect in labels
    assert "Chronology" in labels

    # the UI's numbers must equal the service layer's for the same payload
    ui = at.session_state["result"]
    assert ui["evidence_card"]["status"] == "exploratory"
    assert ui["data_audit"]["status"] == "exploratory"
    assert ui["robustness_matrix"]["scenarios"]
    assert ui["factor_health"]["available"]
    assert any("Exploratory evidence" in str(item.value) for item in at.info)
    assert any(ui["pbo"]["verdict"] == item.value for item in at.markdown)
    assert all(item.value != "robust" for item in at.markdown)
    direct = run_backtest_workflow({
        "provider": "synthetic", "symbols": [], "factor": "momentum",
        "lookback": 252, "skip": 21, "cost_bps": 5.0, "gross": 1.0,
        "n_trials": 50, "periods": 1512, "seed": 42, "start": "2015-01-02",
    })
    assert ui["scorecard"]["ann_sharpe"] == direct["scorecard"]["ann_sharpe"]
    assert ui["scorecard"]["deflated_sr"] == direct["scorecard"]["deflated_sr"]
    assert ui["verdict"] == direct["verdict"]

    # the disclaimer must be on the page
    assert any("not investment advice" in str(c.value) for c in at.caption)
    at.switch_page("pages/2_Real_Document_Batch.py").run()
    assert not at.exception
    at.switch_page("Home.py").run()
    assert not at.exception
    assert at.session_state["result"]["scorecard"] == ui["scorecard"]


def test_ml_tab_exposes_feature_and_model_pickers(at):
    """The ML tab lets the user compose the feature panel + ladder (Part B wiring).
    A momentum run uses the synthetic provider, so the pickers render (not the
    'needs synthetic' notice)."""
    _run_button(at).click()
    at.run()
    assert not at.exception
    labels = [m.label for m in at.multiselect]
    assert any("Features" in lbl for lbl in labels), labels
    assert any("Models in the ladder" in lbl for lbl in labels), labels


def test_portfolio_lab_runs_free_constrained_workflow(at):
    # Start from Home so Streamlit registers the multipage app before the page
    # renders its shared sidebar links.
    lab = at.switch_page("pages/4_Portfolio_Lab.py").run()
    assert not lab.exception
    button = next(button for button in lab.button if "Run portfolio research" in button.label)
    button.click()
    lab.run()
    assert not lab.exception
    labels = [metric.label for metric in lab.metric]
    for expected in ("Annual return", "Sharpe ratio", "Largest drawdown", "Expected shortfall (99%)"):
        assert expected in labels
    assert any("Research-only portfolio simulation" in str(element.value) for element in lab.info)


def test_data_connections_page_renders_without_credentials(at):
    connections = at.switch_page("pages/5_Data_Connections.py").run()
    assert not connections.exception
    assert "Data connections" in connections.title[0].value
    assert any("licensed bundle" in str(item.value).lower() for item in connections.info)


def test_market_microstructure_lab_runs_synthetic_engine_check(at):
    lab = at.switch_page("pages/6_Market_Microstructure_Lab.py").run()
    assert not lab.exception
    button = next(button for button in lab.button if "Run market-quality study" in button.label)
    button.click()
    lab.run()
    assert not lab.exception
    labels = [metric.label for metric in lab.metric]
    assert {"Trade events", "Final-holdout events", "Evidence status"} <= set(labels)
    assert any("Synthetic engine check" in str(item.value) for item in lab.warning)
