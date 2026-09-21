"""Public-data coverage and optional dated text classification."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
import streamlit as st

from api.service import DISCLAIMER, WorkflowError, run_public_data_pilot
from app.ui import empty_state, footer, page, sentiment_metrics
from research.local_history import LocalResearchHistory

API_URL = os.environ.get("ALPHAFORGE_API", "http://127.0.0.1:8000")
page("Data quality", "SEC fundamentals / ALFRED economic vintages / Availability checks", section="Data quality")

if "local_history" not in st.session_state:
    st.session_state["local_history"] = LocalResearchHistory()


def _run(payload: dict) -> dict:
    try:
        response = httpx.post(f"{API_URL}/public-data-pilot", json=payload, timeout=300.0)
    except httpx.ConnectError:
        return run_public_data_pilot(payload)
    except httpx.RequestError as error:
        raise WorkflowError("The data request did not finish. Retry when the connection is ready.") from error
    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", f"API error {response.status_code}")
    except ValueError:
        detail = f"Data service unavailable (HTTP {response.status_code})."
    raise WorkflowError(str(detail))


with st.expander("Coverage setup", expanded="public_data_pilot" not in st.session_state):
    first, second = st.columns(2)
    with first:
        tickers_text = st.text_input("Companies", "AAPL, MSFT, NVDA, JPM, XOM", help="Up to five comma-separated tickers.")
        macro_text = st.text_input("Economic series", "CPIAUCSL, UNRATE, DGS10",
                                  help="CPIAUCSL: consumer prices. UNRATE: unemployment. DGS10: 10-year Treasury yield.")
    with second:
        macro_start = st.date_input("History starts", value=dt.date(2015, 1, 1))
        as_of = st.date_input("Available through", value=dt.date.today() - dt.timedelta(days=1))
    include_text = st.toggle("Include a dated text check")
    text_document = None
    if include_text:
        doc_left, doc_right = st.columns(2)
        with doc_left:
            document_symbol = st.text_input("Document ticker", "AAPL")
            source = st.selectbox("Document source", ["earnings release", "filing excerpt", "manual note"])
            document_id = st.text_input("Document ID", "aapl-pilot-001")
        with doc_right:
            available_date = st.date_input("Document available date", value=as_of)
            available_time = st.time_input("Document available time", value=dt.time(16, 30))
        document_text = st.text_area("Document text", height=160, max_chars=12_000)
        text_document = {
            "symbol": document_symbol, "available_at": dt.datetime.combine(available_date, available_time).isoformat(),
            "source": source, "document_id": document_id, "text": document_text,
        }
    run = st.button("Check data coverage", type="primary", icon=":material/fact_check:")

if run:
    st.session_state.pop("public_data_pilot", None)
    payload = {
        "symbols": [v.strip().upper() for v in tickers_text.split(",") if v.strip()],
        "macro_series": [v.strip().upper() for v in macro_text.split(",") if v.strip()],
        "macro_start": macro_start.isoformat(), "as_of": as_of.isoformat(), "text_document": text_document,
    }
    try:
        with st.spinner("Checking filing history and economic vintages..."):
            st.session_state["public_data_pilot"] = _run(payload)
        st.session_state["local_history"].record(
            "coverage", f"{len(payload['symbols'])} companies through {payload['as_of']}",
            st.session_state["public_data_pilot"],
        )
        st.rerun()
    except WorkflowError as error:
        st.error(f"Coverage needs attention: {error}", icon=":material/error:")
    except Exception as error:
        st.error(f"Coverage unavailable: {type(error).__name__}. Check provider access and retry.", icon=":material/error:")

result = st.session_state.get("public_data_pilot")
if not result:
    empty_state("No coverage report yet", "No source history checked for this session.")
else:
    for warning in result.get("warnings", []):
        st.warning(warning, icon=":material/warning:")
    sec, macro = pd.DataFrame(result["sec"]), pd.DataFrame(result["macro"])
    left, middle, right = st.columns(3)
    left.metric("Companies checked", len(result["symbols"]))
    middle.metric("SEC observations", f"{int(sec['observations'].sum()):,}", help="Reported company facts with filing-date availability.")
    right.metric("Economic vintages", f"{int(macro['vintages'].sum()):,}", help="Historical versions of economic observations, including revisions.")
    st.info("Availability checked, not investment performance. Coverage counts do not guarantee complete history.", icon=":material/info:")
    sec_tab, macro_tab, text_tab = st.tabs(["Company filings", "Economic data", "Text check"])
    with sec_tab:
        st.subheader("Company filing coverage")
        sec_view = sec.rename(columns={"symbol": "Company", "observations": "Observations", "metrics": "Metrics",
                                      "first_available": "First available", "latest_available": "Latest available"})
        for name in ("First available", "Latest available"):
            sec_view[name] = pd.to_datetime(sec_view[name]).dt.date
        st.dataframe(sec_view, use_container_width=True, hide_index=True)
        st.download_button("Export company coverage", sec_view.to_csv(index=False), "alphaforge-company-coverage.csv", "text/csv", icon=":material/download:")
    with macro_tab:
        st.subheader("Economic data availability")
        macro_view = macro.rename(columns={"series_id": "Series", "vintages": "Vintages", "latest_value": "Latest value",
                                          "observation_date": "Observation date", "available_date": "Available date"})
        for name in ("Observation date", "Available date"):
            macro_view[name] = pd.to_datetime(macro_view[name]).dt.date
        st.dataframe(macro_view, use_container_width=True, hide_index=True)
        st.caption("Observation date: the period measured. Available date: when that vintage became public.")
        st.download_button("Export economic coverage", macro_view.to_csv(index=False), "alphaforge-economic-coverage.csv", "text/csv", icon=":material/download:")
    with text_tab:
        feature = result.get("text_feature")
        if feature:
            st.subheader("Document tone")
            sentiment_metrics(feature)
            st.caption(f"{feature['symbol']} | {feature['source']} | Available {feature['available_at']} | {feature['model']}")
            integrity = feature.get("model_time_integrity")
            if integrity:
                metadata = integrity.get("model_metadata", {})
                available = (metadata.get("model_available_at") or {})
                cutoff = (metadata.get("training_data_cutoff") or {})
                st.caption(
                    f"Model provenance: available {available.get('value') or 'not established'} "
                    f"({available.get('status', 'unknown')}); training-data cutoff "
                    f"{cutoff.get('value') or 'not established'} ({cutoff.get('status', 'unknown')})."
                )
                if integrity.get("status") != "eligible" or integrity.get("training_overlap_risk"):
                    st.info(integrity.get("reason", ""), icon=":material/schedule:")
                st.caption(integrity.get("note", ""))
            # The nested provenance block renders on its own above; keep the
            # table to the flat score fields.
            st.dataframe(
                pd.DataFrame([{k: v for k, v in feature.items() if k != "model_time_integrity"}]),
                use_container_width=True, hide_index=True)
        else:
            empty_state("No text check in this run", "The coverage report contains company filings and economic data only.")
    with st.expander("Source audit"):
        st.json(result, expanded=False)
        st.download_button("Export audit", json.dumps(result, indent=2), "alphaforge-coverage-audit.json", "application/json", icon=":material/download:")

footer(DISCLAIMER)
