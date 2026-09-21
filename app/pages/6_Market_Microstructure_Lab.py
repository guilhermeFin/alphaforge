"""Research workspace for trade-flow evidence and market-making simulation."""
from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.service import DISCLAIMER, WorkflowError, run_microstructure_lab
from app.ui import empty_state, footer, page, plot
from research.local_history import LocalResearchHistory
from research.microstructure import MicrostructureError, load_binance_trades, run_microstructure_study


API_URL = os.environ.get("ALPHAFORGE_API", "http://127.0.0.1:8000")
page("Market microstructure lab", "Trade flow / Quote simulation / Options diagnostics", section="Market microstructure")


def _request(payload: dict) -> dict:
    """Use the shared service when available; local files never leave this process."""
    if payload["source"] == "binance_events":
        raise RuntimeError("Local event uploads are handled directly to preserve the raw file locally.")
    try:
        response = httpx.post(f"{API_URL}/microstructure-study", json=payload, timeout=60.0)
    except httpx.ConnectError:
        return run_microstructure_lab(payload)
    except httpx.RequestError as error:
        raise WorkflowError("The microstructure service did not finish. No prior result is being shown.") from error
    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", f"API error {response.status_code}")
    except ValueError:
        detail = f"Microstructure service unavailable (HTTP {response.status_code})."
    raise WorkflowError(str(detail))


def _format(value, spec: str = ".3f", fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return fallback


if "microstructure_result" not in st.session_state:
    st.session_state["microstructure_result"] = None
if "microstructure_history" not in st.session_state:
    st.session_state["microstructure_history"] = LocalResearchHistory()

with st.expander("Study setup", expanded=st.session_state["microstructure_result"] is None):
    source_col, feature_col, simulation_col = st.columns(3)
    with source_col:
        st.subheader("Event source")
        source_label = st.radio("Source", ["Synthetic engine check", "Binance trade CSV"], horizontal=True)
        uploaded = None
        if source_label == "Synthetic engine check":
            seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
            st.caption("Synthetic events validate calculations and controls. They are never market evidence.")
        else:
            seed = 42
            uploaded = st.file_uploader("Binance trade export", type=["csv"],
                                        help="Required columns: time or timestamp, price, qty or quantity, and isBuyerMaker.")
            st.caption("The raw upload stays local. The lab uses trade events, not order-book or quote history.")
    with feature_col:
        st.subheader("Pre-declared test")
        window = st.slider("Flow window (events)", 8, 200, 32, step=4)
        horizon = st.slider("Markout horizon (later trades)", 5, min(300, window * 9), min(30, window * 2), step=5)
        st.caption("Candidate: signed flow imbalance. Baseline: most recent trade return. Splits are chronological 60 / 20 / 20.")
    with simulation_col:
        st.subheader("Simulation stress")
        maker_fee = st.slider("Maker fee (bps)", 0.0, 20.0, 1.0, step=0.5)
        latency = st.slider("Cancellation delay (events)", 0, 25, 2)
        inventory = st.slider("Maximum simulated inventory", 1, 25, 8)
        st.caption("Quote fills are modeled assumptions, not reconstructed exchange fills or queue position.")
    run = st.button("Run market-quality study", type="primary", icon=":material/play_arrow:",
                    disabled=source_label == "Binance trade CSV" and uploaded is None)

if run:
    st.session_state["microstructure_result"] = None
    try:
        with st.spinner("Validating event lineage, reserving the final holdout, and comparing pre-declared baselines..."):
            if source_label == "Binance trade CSV":
                events = load_binance_trades(uploaded)
                st.session_state["microstructure_result"] = run_microstructure_study(
                    events, source="binance_events", seed=int(seed), window=int(window), horizon_events=int(horizon),
                    maker_fee_bps=float(maker_fee), latency_events=int(latency), max_inventory=int(inventory),
                )
            else:
                payload = {"source": "synthetic", "seed": int(seed), "window_events": int(window),
                           "horizon_events": int(horizon), "maker_fee_bps": float(maker_fee),
                           "latency_events": int(latency), "max_inventory": int(inventory)}
                st.session_state["microstructure_result"] = _request(payload)
        completed = st.session_state["microstructure_result"]
        st.session_state["microstructure_history"].record(
            "microstructure",
            f"Trade-flow study / {completed['data_audit']['source']} / {completed['data_audit']['n_events']} events",
            completed,
        )
        st.rerun()
    except (WorkflowError, MicrostructureError) as error:
        st.error(f"Study setup needs attention: {error}", icon=":material/error:")
    except Exception as error:  # pragma: no cover - unexpected runtime protection for the UI
        st.error(f"The study could not finish: {type(error).__name__}. No result was saved.", icon=":material/error:")

result = st.session_state["microstructure_result"]
if not result:
    empty_state("No market-quality study yet", "Run a fixed trade-flow hypothesis to inspect data quality, out-of-sample evidence, and the simulation assumptions.")
    footer(DISCLAIMER)
    st.stop()

audit, flow, simulation = result["data_audit"], result["trade_flow"], result["simulation"]
if audit["source"] == "synthetic":
    st.warning("Synthetic engine check. It verifies the workflow only and cannot establish a market effect.", icon=":material/science:")
else:
    st.warning("Exploratory trade-only study. It does not contain BBO, full depth, queue position, or realised quote fills.", icon=":material/warning:")

st.info(result["research_question"], icon=":material/science:")
top1, top2, top3, top4 = st.columns(4)
top1.metric("Trade events", f"{audit['n_events']:,}")
top2.metric("Largest timestamp gap", _format(audit["largest_timestamp_gap_seconds"], ".1f") + " sec")
top3.metric("Final-holdout events", f"{flow['splits']['final_holdout_events']:,}")
top4.metric("Evidence status", "Exploratory" if audit["status"] != "engine_validation" else "Engine check")

evidence_tab, simulator_tab, options_tab, audit_tab = st.tabs(["Trade-flow evidence", "Quote simulation", "Options diagnostics", "Audit"])
with evidence_tab:
    st.subheader("Pre-declared final-holdout comparison")
    st.caption(flow["definition"])
    final_rows = pd.DataFrame(flow["results"])
    final_rows = final_rows.loc[final_rows["stage"] == "final_holdout"].copy()
    final_rows["Model"] = final_rows["model"].str.replace("_", " ").str.title()
    final_rows = final_rows.rename(columns={"spearman_ic": "Spearman IC", "directional_accuracy": "Direction accuracy", "n_events": "Events"})
    st.dataframe(final_rows.loc[:, ["Model", "Spearman IC", "Direction accuracy", "Events"]], hide_index=True,
                 use_container_width=True,
                 column_config={"Spearman IC": st.column_config.NumberColumn(format="%.4f"), "Direction accuracy": st.column_config.NumberColumn(format="%.1%")})
    if flow["survives_baseline"]:
        st.success(flow["verdict"], icon=":material/check_circle:")
    else:
        st.warning(flow["verdict"], icon=":material/warning:")
    calibration = pd.DataFrame(flow["calibration"])
    if not calibration.empty:
        figure = go.Figure(go.Bar(x=calibration["decile"], y=calibration["mean_markout"], marker_color="#087f70"))
        figure.update_layout(title="Flow-imbalance calibration on final holdout", yaxis_tickformat=".3%", xaxis_title="Flow-imbalance decile", yaxis_title="Mean future trade-price markout")
        plot(figure, height=320)
    preview = pd.DataFrame(result["panel_preview"])
    figure = go.Figure(go.Scatter(x=preview["timestamp"], y=preview["price"], name="Trade price", line={"color": "#4064c6"}))
    figure.update_layout(title="Sampled trade-price path", yaxis_title="Price")
    plot(figure, height=300)

with simulator_tab:
    st.subheader("Assumption-visible quote-policy comparison")
    st.warning(simulation["label"], icon=":material/warning:")
    outcomes = pd.DataFrame(simulation["outcomes"])
    st.dataframe(outcomes, hide_index=True, use_container_width=True,
                 column_config={"simulated_pnl": st.column_config.NumberColumn("Simulated P&L", format="%.3f"),
                                "maker_fees": st.column_config.NumberColumn("Maker fees", format="%.3f"),
                                "gross_spread_capture": st.column_config.NumberColumn("Gross spread capture", format="%.3f"),
                                "adverse_selection_proxy": st.column_config.NumberColumn("Adverse-selection proxy", format="%.3f"),
                                "max_drawdown": st.column_config.NumberColumn("Maximum drawdown", format="%.3f")})
    curve_figure = go.Figure()
    for policy, points in simulation["curves"].items():
        curve = pd.DataFrame(points)
        curve_figure.add_trace(go.Scatter(x=curve["timestamp"], y=curve["equity"], name=policy))
    curve_figure.update_layout(title="Simulated marked-to-market equity by policy", yaxis_title="Simulation currency units")
    plot(curve_figure, height=340)
    st.caption("Simulation inputs: " + ", ".join(f"{key.replace('_', ' ')} = {value}" for key, value in simulation["assumptions"].items()))

with options_tab:
    st.subheader("Options mathematics diagnostic")
    st.caption(result["options_diagnostics"]["note"])
    contracts = pd.DataFrame(result["options_diagnostics"]["contracts"])
    st.dataframe(contracts, hide_index=True, use_container_width=True,
                 column_config={"implied_volatility": st.column_config.NumberColumn("Implied volatility", format="%.2%"), "delta": st.column_config.NumberColumn("Delta", format="%.3f"), "model_price": st.column_config.NumberColumn("Black-Scholes price", format="%.3f"), "market_price": st.column_config.NumberColumn("Observed price", format="%.3f")})
    st.info("This validates calculations and data consistency. It does not test an options strategy or imply that a contract is mispriced.", icon=":material/info:")

with audit_tab:
    st.subheader("Evidence boundary")
    st.write("The first hypothesis is frozen before the run. It compares signed trade-flow imbalance with a last-return baseline, using chronological train, validation, and final-holdout blocks.")
    st.json(result["protocol"], expanded=True)
    st.subheader("Data limitations")
    for limitation in audit["limitations"]:
        st.warning(limitation, icon=":material/warning:")
    st.caption(f"Research fingerprint: {result['manifest']['research_fingerprint']}")
    st.download_button("Export research audit", data=json.dumps(result, indent=2), file_name="alphaforge-microstructure-audit.json", mime="application/json", icon=":material/download:")

footer(result["disclaimer"])
