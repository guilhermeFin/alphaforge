"""Saved local AlphaForge research runs with lightweight comparison."""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from api.service import DISCLAIMER
from app.ui import empty_state, footer, page
from research.local_history import LocalResearchHistory


page("Research history", "Saved local runs / Audit summaries / Side-by-side comparison", section="Research history")
if "local_history" not in st.session_state:
    st.session_state["local_history"] = LocalResearchHistory()
history = st.session_state["local_history"]
records = history.list()


def _label(record: dict) -> str:
    return f"{record['created_at'][:19].replace('T', ' ')} | {record['kind'].replace('_', ' ')} | {record['label']}"


def _comparison_row(record: dict) -> dict:
    summary = record["summary"]
    row = {"Saved": record["created_at"][:19].replace("T", " "), "Type": record["kind"].replace("_", " "), "Run": record["label"]}
    if record["kind"] == "filing_batch":
        row.update({"Documents": summary.get("documents"), "Earnings releases": summary.get("earnings_releases"), "Average tone": summary.get("average_tone")})
    elif record["kind"] == "event_study":
        row.update({"Usable events": summary.get("n_events"), "Held-out events": summary.get("n_oos"), "MSE improvement": summary.get("mse_improvement"), "OOS correlation": summary.get("oos_pearson"), "Evidence established": summary.get("evidence_established")})
    elif record["kind"] in {"backtest", "portfolio_research"}:
        row.update({"Factor": summary.get("factor"), "Annual return": summary.get("annual_return"), "Sharpe": summary.get("sharpe"), "Verdict": summary.get("verdict")})
        if record["kind"] == "backtest":
            row["Temporal stability"] = summary.get("temporal_status") or "not recorded"
        if record["kind"] == "portfolio_research":
            row["Largest company"] = summary.get("max_name_weight")
    elif record["kind"] == "coverage":
        row.update({"Companies": summary.get("companies"), "SEC observations": summary.get("sec_observations")})
    elif record["kind"] == "benchmark_suite":
        row.update({"Benchmarks": summary.get("benchmarks"), "Provider": summary.get("provider")})
    return row


def _overview_row(record: dict) -> dict:
    summary = record["summary"]
    row = {
        "Saved": record["created_at"][:19].replace("T", " "),
        "Type": record["kind"].replace("_", " "),
        "Run": record["label"],
    }
    if record["kind"] == "filing_batch":
        tone = summary.get("average_tone")
        tone_label = f"{float(tone):+.3f}" if tone is not None else "not available"
        row["Summary"] = f"{summary.get('documents', 0)} documents | {tone_label} average tone"
    elif record["kind"] == "event_study":
        correlation = summary.get("oos_pearson")
        correlation_label = f"{float(correlation):+.3f}" if correlation is not None else "not available"
        row["Summary"] = f"{summary.get('n_events', 0)} events | OOS correlation {correlation_label}"
    elif record["kind"] in {"backtest", "portfolio_research"}:
        sharpe = summary.get("sharpe")
        sharpe_label = f"{float(sharpe):+.2f}" if sharpe is not None else "not available"
        detail = "portfolio study" if record["kind"] == "portfolio_research" else "strategy"
        stability = summary.get("temporal_status")
        stability_label = f" | temporal {stability.replace('_', ' ')}" if stability else ""
        row["Summary"] = f"{summary.get('factor', detail)} | Sharpe {sharpe_label}{stability_label}"
    elif record["kind"] == "coverage":
        row["Summary"] = f"{summary.get('companies', 0)} companies | {summary.get('sec_observations', 0):,} SEC observations"
    elif record["kind"] == "benchmark_suite":
        row["Summary"] = f"{summary.get('benchmarks', 0)} fixed reference factors | {summary.get('provider', 'unknown provider')}"
    return row


def _metric_rows(record: dict) -> pd.DataFrame:
    row = _comparison_row(record)
    return pd.DataFrame([
        {"Measure": key, "Value": value}
        for key, value in row.items()
        if key not in {"Saved", "Type", "Run"} and value is not None
    ])


if not records:
    empty_state("No saved research yet", "Completed backtests, portfolio studies, coverage checks, filing batches, and event studies will appear here on this computer.")
else:
    st.info("Saved locally on this computer. Results remain research records, not investment recommendations.", icon=":material/info:")
    overview, compare = st.tabs(["Saved runs", "Compare runs"])
    with overview:
        st.subheader("Saved runs")
        for index, record in enumerate(records):
            row = _overview_row(record)
            st.markdown(f"**{row['Type'].title()}** | {row['Run']}")
            st.caption(f"{row['Saved']} | {row['Summary']}")
            if index < len(records) - 1:
                st.divider()
        st.download_button("Export saved research", json.dumps(records, indent=2), "alphaforge-research-history.json", "application/json", icon=":material/download:")
        selected = st.selectbox("Inspect saved run", range(len(records)), format_func=lambda index: _label(records[index]))
        st.caption("Saved audit record")
        st.json(records[selected]["payload"], expanded=False)
    with compare:
        st.subheader("Compare two runs")
        left, right = st.columns(2)
        with left:
            first = st.selectbox("First run", range(len(records)), format_func=lambda index: _label(records[index]), key="history_first")
        with right:
            second_default = 1 if len(records) > 1 else 0
            second = st.selectbox("Second run", range(len(records)), index=second_default, format_func=lambda index: _label(records[index]), key="history_second")
        if records[first]["kind"] == records[second]["kind"]:
            st.dataframe(pd.DataFrame([_comparison_row(records[first]), _comparison_row(records[second])]), use_container_width=True, hide_index=True)
        else:
            st.caption("Different research types intentionally keep different metrics; no cross-type score is invented.")
            first_detail, second_detail = st.columns(2)
            with first_detail:
                st.caption(_label(records[first]))
                st.dataframe(_metric_rows(records[first]), use_container_width=True, hide_index=True)
            with second_detail:
                st.caption(_label(records[second]))
                st.dataframe(_metric_rows(records[second]), use_container_width=True, hide_index=True)

footer(DISCLAIMER)
