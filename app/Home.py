"""AlphaForge — Streamlit MVP. One workflow: universe -> factor -> honest backtest.

Run from the repo root (API optional but preferred):
    uvicorn api.main:app --port 8000          # terminal 1 (optional)
    streamlit run app/Home.py                 # terminal 2

The page talks to the FastAPI backend when it's up (dogfooding the API); if the
API isn't running it falls back to importing the same service function directly,
so the demo always works. Either way the numbers are identical by construction.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.service import DISCLAIMER, WorkflowError, run_backtest_workflow

API_URL = os.environ.get("ALPHAFORGE_API", "http://127.0.0.1:8000")


def fmt(v, spec: str = "", na: str = "n/a") -> str:
    """Format a value that may legitimately be None/NaN (the engine emits null for
    undefined stats like Sharpe on a flat series) without ever crashing the page."""
    if v is None:
        return na
    try:
        return format(v, spec)
    except (ValueError, TypeError):
        return str(v)

st.set_page_config(page_title="AlphaForge", page_icon="🛠️", layout="wide")
st.title("AlphaForge")
st.caption("The research OS for emerging quant managers — **the backtest that won't let you lie to yourself.**")


def api_is_up(retries: int = 3) -> bool:
    """Probe the API a few times so a just-launched server (started moments before
    the UI) is detected instead of silently falling back to in-process mode."""
    for i in range(retries):
        try:
            if httpx.get(f"{API_URL}/health", timeout=1.5).status_code == 200:
                return True
        except Exception:
            pass
        if i < retries - 1:
            time.sleep(0.6)
    return False


def run_request(payload: dict) -> dict:
    if st.session_state.get("api_mode"):
        r = httpx.post(f"{API_URL}/backtest", json=payload, timeout=180.0)
        if r.status_code != 200:
            raise WorkflowError(r.json().get("detail", f"API error {r.status_code}"))
        return r.json()
    return run_backtest_workflow(payload)


if "api_mode" not in st.session_state:
    st.session_state["api_mode"] = api_is_up()

mode_label = (f"connected to API at {API_URL}" if st.session_state["api_mode"]
              else "API not detected — running the engine in-process (same code path)")
st.info(f"Mode: {mode_label}", icon="🔌")

# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.header("Workflow")
    provider = st.radio("Data source", ["synthetic", "yfinance"],
                        help="Synthetic = offline, deterministic engine demo. "
                             "Yahoo Finance = real (survivorship-biased!) free data.")
    if provider == "yfinance":
        tickers = st.text_input("Tickers (comma-separated)",
                                "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,XOM,JNJ,PG,KO,WMT,HD,BAC,DIS")
        symbols = [t.strip() for t in tickers.split(",") if t.strip()]
        periods, seed = 1512, 42
        st.caption("⚠️ Today's tickers backtested into the past = survivorship bias. "
                   "Treat results as a demo, not validation.")
    else:
        symbols = []
        # min days kept comfortably above the longest default lookback window so a
        # default-adjacent selection never triggers a "lookback too long" 400.
        periods = st.slider("Days (synthetic)", 400, 3000, 1512, step=50)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

    factor = st.selectbox("Factor", ["momentum", "reversal", "lowvol", "blend",
                                     "value", "quality", "value_quality"])
    if factor in ("value", "quality", "value_quality"):
        st.caption("📒 Fundamental factors use point-in-time fundamentals (filing-date lagged) — "
                   "synthetic provider only in this MVP, since free data isn't point-in-time.")
    lookback = st.slider("Lookback (days)", 21, 504, 252, step=21)
    # skip must stay below lookback (the service rejects skip >= lookback); cap the
    # widget so the UI can't produce a request the engine will refuse.
    skip = st.slider("Skip recent (days)", 0, max(7, lookback - 7),
                     min(21, lookback - 7), step=7,
                     help="Momentum convention: skip the last month (short-term reversal).")
    cost_bps = st.slider("Cost (bps per unit turnover)", 0.0, 50.0, 5.0, step=0.5)
    n_trials = st.slider("Variants tried (honesty input)", 1, 500, 50,
                         help="How many strategy variations you have explored, including "
                              "everything you tried and discarded. Drives the Deflated "
                              "Sharpe haircut. Be honest — that's the whole point.")
    run = st.button("Run honest backtest", type="primary", use_container_width=True)

# ----------------------------- run -----------------------------
if run:
    payload = {"provider": provider, "symbols": symbols, "factor": factor,
               "lookback": lookback, "skip": skip, "cost_bps": cost_bps,
               "gross": 1.0, "n_trials": n_trials, "periods": periods,
               "seed": int(seed), "start": "2015-01-02"}
    try:
        with st.spinner("Running point-in-time backtest…"):
            st.session_state["result"] = run_request(payload)
    except WorkflowError as e:
        st.error(f"Input problem: {e}")
    except Exception as e:
        st.error(f"Run failed: {e}")

res = st.session_state.get("result")
if not res:
    st.markdown("⬅️ Configure a universe and factor, then **Run honest backtest**. "
                "The synthetic source works offline and is deterministic.")
    st.stop()

meta, card = res["meta"], res["scorecard"]
oos, wf, tails = res["out_of_sample"], res["walk_forward"], res["fat_tails"]

# ----------------------------- verdict -----------------------------
credible = res["verdict"].startswith("CREDIBLE")
wf_dsr = wf.get("deflated_sr") if "error" not in wf else None
(st.success if credible else st.error)(f"**{res['verdict']}**  \n"
    f"(Deflated Sharpe: full-sample {fmt(card['deflated_sr'], '.3f')} vs. "
    f"out-of-sample walk-forward {fmt(wf_dsr, '.3f')}, at {meta['n_trials']} declared trials; "
    f"naive in-sample PSR = {fmt(card['psr_vs_0'], '.3f')} — the gaps are the honesty haircuts)")

st.caption(f"{meta['n_symbols']} symbols × {meta['n_days']} days "
           f"({meta['start_date']} → {meta['end_date']}) · factor: {meta['factor']} "
           f"(effective lookback {meta.get('effective_lookback') or 'n/a'}) · "
           f"costs: {meta['cost_bps']} bps · provider: {meta['provider']}")

# ----------------------------- headline metrics -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CAGR", fmt(card['cagr'], '.2%'))
c2.metric("Ann. Sharpe", fmt(card['ann_sharpe'], '.2f'))
c3.metric("Max drawdown", fmt(card['max_drawdown'], '.2%'))
c4.metric("Deflated SR (OOS)", fmt(wf_dsr, '.3f'))
c5.metric("Hit rate", fmt(card['hit_rate'], '.1%'))

# ----------------------------- charts -----------------------------
eq = pd.DataFrame(res["equity_curve"])
dd = pd.DataFrame(res["drawdown_curve"])
left, right = st.columns([3, 2])
with left:
    fig = go.Figure(go.Scatter(x=eq["date"], y=eq["value"], mode="lines", name="equity"))
    fig.update_layout(title="Equity curve (net of costs)", height=380,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig2 = go.Figure(go.Scatter(x=dd["date"], y=dd["value"], mode="lines",
                                fill="tozeroy", name="drawdown"))
    fig2.update_layout(title="Drawdown", height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- honesty panels -----------------------------
a, b, c = st.columns(3)
with a:
    st.subheader("Out-of-sample (70/30)")
    st.write(f"In-sample Sharpe: **{fmt(oos['in_sample_sharpe'], '+.3f')}**")
    st.write(f"Out-of-sample Sharpe: **{fmt(oos['out_sample_sharpe'], '+.3f')}**")
    st.write(f"Degradation: **{fmt(oos['degradation'], '+.3f')}**")
    st.write(("🔴 **Overfit warning** — the edge decays out-of-sample"
              if oos["overfit_warning"] else "🟢 No IS→OOS decay"))
    st.write(("🟢 OOS record is statistically significant"
              if oos["oos_significant"] else "🟡 OOS record not yet significant (PSR < 0.95)"))
with b:
    folds = wf.get("n_oos_folds", "?") if "error" not in wf else 0
    st.subheader(f"Walk-forward ({folds} OOS folds)")
    if "error" in wf:
        st.write(f"n/a: {wf['error']}")
    else:
        st.write(f"Fold OOS Sharpes: {wf['fold_oos_sharpes']}")
        st.write(f"Stitched OOS Sharpe: **{fmt(wf['stitched_oos_sharpe'], '+.3f')}**")
        st.write(f"Deflated SR (folds-as-trials): **{fmt(wf['deflated_sr'], '.3f')}**")
        st.write("🟢 passes" if wf["passes"] else "🔴 does not pass")
with c:
    st.subheader("Tail risk")
    st.write(f"Skew: **{fmt(tails.get('skew'), '+.2f')}** · "
             f"Excess kurtosis: **{fmt(tails.get('excess_kurtosis'), '.1f')}**")
    st.write(f"Jarque-Bera p: {fmt(tails.get('jarque_bera_p'), '.2g')}")
    st.write(("🟢 returns ≈ Normal" if tails.get("returns_are_normal")
              else "🔴 FAT TAILS — Gaussian VaR understates risk"))
    st.caption(tails.get("verdict", ""))

with st.expander("Full scorecard"):
    st.dataframe(pd.DataFrame([card]).T.rename(columns={0: "value"}), use_container_width=True)

st.divider()
st.caption(f"⚖️ {DISCLAIMER}")
