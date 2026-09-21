"""Shared presentation primitives for the three research workspaces."""
from __future__ import annotations

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).parent
COLORS = {"positive": "#087f70", "negative": "#c64b60", "neutral": "#668099", "blue": "#4064c6"}


def page(title: str, description: str, *, section: str, sidebar: str = "auto") -> None:
    st.set_page_config(page_title=f"{title} | AlphaForge", page_icon=":material/query_stats:",
                       layout="wide", initial_sidebar_state=sidebar)
    st.html((APP / "styles.css").read_text(encoding="utf-8"))
    with st.sidebar:
        st.markdown('<div class="af-brand"><span class="af-mark">AF</span><span>AlphaForge</span></div>', unsafe_allow_html=True)
        st.caption("RESEARCH WORKSPACE")
        for path, label, icon in [
            ("Home.py", "Strategy lab", "query_stats"),
            ("pages/1_Public_Data_Pilot.py", "Data quality", "database"),
            ("pages/2_Real_Document_Batch.py", "Filing research", "article"),
            ("pages/4_Portfolio_Lab.py", "Portfolio lab", "account_balance"),
            ("pages/5_Data_Connections.py", "Data connections", "hub"),
            ("pages/6_Market_Microstructure_Lab.py", "Market microstructure", "candlestick_chart"),
            ("pages/3_Research_History.py", "Research history", "history"),
        ]:
            st.page_link(path, label=label, icon=f":material/{icon}:", use_container_width=True)
        st.divider()
        st.caption("LOCAL WORKSPACE")
        st.markdown("Research only")
    st.markdown(f'<div class="af-eyebrow">WORKSPACE / {escape(section.upper())}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def empty_state(title: str, detail: str) -> None:
    st.markdown(f'<div class="af-empty"><strong>{escape(title)}</strong><p>{escape(detail)}</p></div>', unsafe_allow_html=True)


def footer(disclaimer: str) -> None:
    st.divider()
    st.caption(disclaimer)


def plot(fig: go.Figure, *, height: int = 340) -> None:
    fig.update_layout(
        template="plotly_white", height=height, colorway=list(COLORS.values()),
        font=dict(family="Arial, sans-serif", color="#202a30", size=12),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=8, r=8, t=42, b=24), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        title=dict(font=dict(size=15)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e9edef", zerolinecolor="#d9e0e4")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})


def sentiment_metrics(feature: dict, *, aggregate: bool = False) -> None:
    score, pos, neg, neutral = st.columns(4)
    score.metric("Tone score", f"{feature['sentiment']:+.3f}",
                 help="Positive score minus negative score, from -1 to +1. Text tone, not expected return.")
    help_text = "Average model score across selected paragraphs." if aggregate else "Model classification score for this passage, not a probability of a price move."
    for column, label in [(pos, "Positive"), (neg, "Negative"), (neutral, "Neutral")]:
        column.metric(label, f"{feature[label.lower() + '_probability']:.1%}", help=help_text)


def source_label(record: dict) -> str:
    if record.get("content_kind") == "earnings_release":
        return "Earnings release"
    if record.get("content_kind") == "mda_excerpt":
        return "MD&A excerpt"
    return "Filing excerpt"
