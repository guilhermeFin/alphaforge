"""Public-data provenance pilot for SEC, ALFRED, and optional FinBERT features."""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
import streamlit as st

from api.service import DISCLAIMER, WorkflowError, run_public_data_pilot

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Public Data Pilot - AlphaForge", page_icon="📊", layout="wide")
st.title("Public Data Pilot")
st.caption("SEC filings, ALFRED vintages, and an optional FinBERT document check.")


def _run(payload: dict) -> dict:
    try:
        response = httpx.post(f"{API_URL}/public-data-pilot", json=payload, timeout=300.0)
        if response.status_code == 200:
            return response.json()
        detail = response.json().get("detail", f"API error {response.status_code}")
        raise WorkflowError(detail)
    except httpx.RequestError:
        return run_public_data_pilot(payload)


with st.sidebar:
    st.header("Pilot setup")
    tickers_text = st.text_input("Tickers", "AAPL, MSFT, NVDA, JPM, XOM")
    macro_text = st.text_input("Macro series", "CPIAUCSL, UNRATE, DGS10")
    macro_start = st.date_input("Macro history from", value=dt.date(2015, 1, 1))
    as_of = st.date_input("Available through", value=dt.date.today() - dt.timedelta(days=1))
    include_text = st.checkbox("Classify one dated document")

    text_document = None
    if include_text:
        document_symbol = st.text_input("Document ticker", "AAPL")
        source = st.selectbox("Document source", ["earnings release", "filing excerpt", "manual note"])
        document_id = st.text_input("Document ID", "aapl-pilot-001")
        available_date = st.date_input("Document available date", value=as_of)
        available_time = st.time_input("Document available time", value=dt.time(16, 30))
        document_text = st.text_area("Document text", height=180, max_chars=12_000)
        text_document = {
            "symbol": document_symbol,
            "available_at": dt.datetime.combine(available_date, available_time).isoformat(),
            "source": source,
            "document_id": document_id,
            "text": document_text,
        }

    run = st.button("Run public-data pilot", type="primary", use_container_width=True)

if run:
    symbols = [value.strip().upper() for value in tickers_text.split(",") if value.strip()]
    macro_series = [value.strip().upper() for value in macro_text.split(",") if value.strip()]
    payload = {
        "symbols": symbols,
        "macro_series": macro_series,
        "macro_start": macro_start.isoformat(),
        "as_of": as_of.isoformat(),
        "text_document": text_document,
    }
    try:
        with st.spinner("Checking public-data provenance..."):
            st.session_state["public_data_pilot"] = _run(payload)
    except WorkflowError as error:
        st.error(f"Pilot needs attention: {error}")
    except Exception as error:
        st.error(f"Pilot failed: {type(error).__name__}: {error}")

result = st.session_state.get("public_data_pilot")
if result:
    st.success(result["note"])
    for warning in result.get("warnings", []):
        st.warning(warning)
    sec, macro = pd.DataFrame(result["sec"]), pd.DataFrame(result["macro"])
    left, middle, right = st.columns(3)
    left.metric("SEC tickers checked", len(result["symbols"]))
    middle.metric("SEC observations", int(sec["observations"].sum()))
    right.metric("ALFRED vintages", int(macro["vintages"].sum()))

    st.subheader("SEC filing availability")
    st.dataframe(sec, use_container_width=True, hide_index=True)
    st.subheader("ALFRED macro availability")
    st.dataframe(macro, use_container_width=True, hide_index=True)

    feature = result.get("text_feature")
    if feature:
        st.subheader("FinBERT document feature")
        metrics = st.columns(4)
        metrics[0].metric("Sentiment", f"{feature['sentiment']:.3f}")
        metrics[1].metric("Positive", f"{feature['positive_probability']:.1%}")
        metrics[2].metric("Negative", f"{feature['negative_probability']:.1%}")
        metrics[3].metric("Neutral", f"{feature['neutral_probability']:.1%}")
        st.dataframe(pd.DataFrame([feature]), use_container_width=True, hide_index=True)

st.divider()
st.caption(DISCLAIMER)
