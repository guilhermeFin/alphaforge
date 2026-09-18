"""Batch classify real SEC filing excerpts with filing-date provenance."""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
import streamlit as st

from api.service import DISCLAIMER, WorkflowError, run_real_document_batch

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Real Document Batch - AlphaForge", page_icon="📄", layout="wide")
st.title("Real Document Batch")
st.caption("Dated SEC filing excerpts, classified in a small research batch.")


def _run(payload: dict) -> dict:
    try:
        response = httpx.post(f"{API_URL}/real-document-batch", json=payload, timeout=300.0)
        if response.status_code == 200:
            return response.json()
        raise WorkflowError(response.json().get("detail", f"API error {response.status_code}"))
    except httpx.RequestError:
        return run_real_document_batch(payload)


with st.sidebar:
    st.header("Batch setup")
    symbols_text = st.text_input("Tickers", "AAPL, MSFT, NVDA, JPM, XOM")
    forms = st.multiselect("SEC forms", ["8-K", "10-Q", "10-K"], default=["8-K"])
    as_of = st.date_input("Filed through", value=dt.date.today() - dt.timedelta(days=1))
    per_symbol = st.select_slider("Filings per ticker", options=[1, 2], value=1)
    run = st.button("Classify real filing batch", type="primary", use_container_width=True)

if run:
    payload = {
        "symbols": [value.strip().upper() for value in symbols_text.split(",") if value.strip()],
        "forms": forms,
        "as_of": as_of.isoformat(),
        "per_symbol": per_symbol,
    }
    try:
        with st.spinner("Retrieving and classifying SEC filing excerpts..."):
            st.session_state["real_document_batch"] = _run(payload)
    except WorkflowError as error:
        st.error(f"Batch needs attention: {error}")
    except Exception as error:
        st.error(f"Batch failed: {type(error).__name__}: {error}")

result = st.session_state.get("real_document_batch")
if result:
    st.success(result["note"])
    for warning in result.get("warnings", []):
        st.warning(warning)
    features = pd.DataFrame(result["features"])
    left, middle, right = st.columns(3)
    left.metric("Filing excerpts", len(features))
    middle.metric("Average sentiment", f"{features['sentiment'].mean():.3f}")
    right.metric("Most negative", f"{features['sentiment'].min():.3f}")
    st.subheader("Timestamped filing features")
    st.dataframe(features, use_container_width=True, hide_index=True, column_config={
        "source_url": st.column_config.LinkColumn("SEC source"),
    })

st.divider()
st.caption(DISCLAIMER)
