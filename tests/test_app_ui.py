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
    btns = [b for b in t.button if "Run honest backtest" in b.label]
    assert btns, "Run button not found"
    return btns[0]


def test_initial_render(at):
    assert "AlphaForge" in at.title[0].value
    assert any("in-process" in str(i.value) for i in at.info)  # fallback-mode banner


def test_click_runs_and_renders_verdict(at):
    _run_button(at).click()
    at.run()
    assert not at.exception

    # the verdict banner is st.success (credible) or st.error (not credible)
    banners = [e.value for e in at.error] + [s.value for s in at.success]
    assert any("Deflated Sharpe" in b for b in banners), banners

    # headline metrics rendered
    labels = [m.label for m in at.metric]
    for expect in ("CAGR", "Ann. Sharpe", "Max drawdown", "Deflated SR (OOS)"):
        assert expect in labels

    # the UI's numbers must equal the service layer's for the same payload
    ui = at.session_state["result"]
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
