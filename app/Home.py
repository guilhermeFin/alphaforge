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
from research.trial_ledger import TrialLedger

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

# --- screenshot mode (?shot=1) ----------------------------------------------
# Auto-runs a backtest on load and hides dev chrome so marketing captures show a
# finished run. Params: shot=1, factor, periods, trials, sb=0 (collapse sidebar).
# Used by scripts/capture_screens.py; harmless in normal use.
_qp = st.query_params
SHOT_MODE = _qp.get("shot") == "1"
_sb = "collapsed" if _qp.get("sb") == "0" else "expanded"

st.set_page_config(page_title="AlphaForge", page_icon="🛠️", layout="wide",
                   initial_sidebar_state=_sb)
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
        # carry the af_session cookie so the SERVER-side trial ledger accumulates
        # per workspace across runs (the haircut is enforced server-side).
        r = httpx.post(f"{API_URL}/backtest", json=payload, timeout=180.0,
                       cookies=st.session_state.get("af_cookies", {}))
        # The server sends Set-Cookie ONLY when it mints a session; later responses
        # carry none. Merge non-empty cookies and never clobber a good one with {},
        # or the per-workspace ledger silently resets after the first run.
        _new = dict(r.cookies)
        if _new:
            st.session_state["af_cookies"] = {**st.session_state.get("af_cookies", {}), **_new}
        if r.status_code != 200:
            raise WorkflowError(r.json().get("detail", f"API error {r.status_code}"))
        return r.json()
    # in-process mode: the UI owns the ledger and enforcement happens here.
    return run_backtest_workflow(payload, ledger=st.session_state["ledger"])


if "ledger" not in st.session_state:
    st.session_state["ledger"] = TrialLedger()
if "api_mode" not in st.session_state:
    st.session_state["api_mode"] = False if SHOT_MODE else api_is_up()

if not SHOT_MODE:
    mode_label = (f"connected to API at {API_URL}" if st.session_state["api_mode"]
                  else "API not detected — running the engine in-process (same code path)")
    st.info(f"Mode: {mode_label}", icon="🔌")

if SHOT_MODE and "result" not in st.session_state:
    _payload = {
        "provider": "synthetic", "symbols": [],
        "factor": _qp.get("factor", "quality"),
        "lookback": int(_qp.get("lookback", 252)), "skip": int(_qp.get("skip", 21)),
        "cost_bps": float(_qp.get("cost", 5.0)), "gross": 1.0,
        "n_trials": int(_qp.get("trials", 20)),
        "periods": int(_qp.get("periods", 1512)), "seed": int(_qp.get("seed", 42)),
        "start": "2015-01-02",
    }
    st.session_state["result"] = run_backtest_workflow(_payload)

# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.header("Workflow")
    _led = st.session_state["ledger"]
    _ta_now = (st.session_state.get("result") or {}).get("trial_audit") or {}
    _distinct = _ta_now.get("distinct_count")
    if _distinct is None:
        _distinct = _led.distinct_count
    st.metric("Distinct strategies run", _distinct,
              help="The engine deflates the Sharpe by AT LEAST this many trials, no matter "
                   "what you type below. Honest counting is the whole point.")
    if st.button("Reset counter", use_container_width=True):
        if st.session_state.get("api_mode"):
            try:
                httpx.post(f"{API_URL}/session/reset",
                           cookies=st.session_state.get("af_cookies", {}), timeout=10.0)
            except Exception:
                pass
        _led.reset()
        st.session_state.pop("result", None)
        st.rerun()
    st.divider()
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

    _factors = ["momentum", "reversal", "lowvol", "blend", "value", "quality", "value_quality"]
    _fidx = _factors.index(_qp.get("factor")) if (SHOT_MODE and _qp.get("factor") in _factors) else 0
    factor = st.selectbox("Factor", _factors, index=_fidx)
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
_folds = wf.get("n_splits") if "error" not in wf else None
_ta = res.get("trial_audit") or {}
# the full-sample DSR is deflated by the trial-ledger count; the OOS walk-forward
# DSR is deflated by folds-as-trials — two DISTINCT haircuts, kept visibly separate.
_trial_word = "ledger-enforced" if _ta.get("distinct_count") is not None else "declared"
(st.success if credible else st.error)(f"**{res['verdict']}**  \n"
    f"(full-sample Deflated Sharpe {fmt(card['deflated_sr'], '.3f')} — deflated by "
    f"{meta['n_trials']} {_trial_word} trial(s); out-of-sample walk-forward "
    f"{fmt(wf_dsr, '.3f')} — deflated by {fmt(_folds, '.0f') if _folds else 'n/a'} folds; "
    f"naive in-sample PSR {fmt(card['psr_vs_0'], '.3f')}. The gaps are the honesty haircuts.)")
if _ta.get("haircut_was_raised"):
    st.warning(f"You declared {_ta['declared_n_trials']} trial(s), but this workspace has run "
               f"{_ta['effective_n_trials']} distinct strategies — the Deflated Sharpe uses the "
               f"larger, honest number. Reset the counter (sidebar) to start a fresh search.")

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
        st.caption(f"Purged at fold boundaries: {wf.get('total_purged_bars', 0)} bars "
                   f"(purge {wf.get('purge_bars', 0)} + embargo {wf.get('embargo_bars', 0)}) — "
                   f"no lookback leak into out-of-sample.")
with c:
    st.subheader("Tail risk")
    st.write(f"Skew: **{fmt(tails.get('skew'), '+.2f')}** · "
             f"Excess kurtosis: **{fmt(tails.get('excess_kurtosis'), '.1f')}**")
    st.write(f"Jarque-Bera p: {fmt(tails.get('jarque_bera_p'), '.2g')}")
    st.write(("🟢 returns ≈ Normal" if tails.get("returns_are_normal")
              else "🔴 FAT TAILS — Gaussian VaR understates risk"))
    st.caption(tails.get("verdict", ""))
    st.write(f"CVaR 95%: **{fmt(card.get('cvar_95'), '.2%')}** · "
             f"CVaR 99%: **{fmt(card.get('cvar_99'), '.2%')}** _(historical, per-day loss)_")
    if card.get("cvar_verdict"):
        st.caption(card["cvar_verdict"])

sigq = res.get("signal_quality") or {}
if sigq:
    st.subheader("Is the signal itself predictive? (Information Coefficient)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean IC", fmt(sigq.get("mean_ic"), ".4f"))
    s2.metric("IC t-stat", fmt(sigq.get("ic_tstat"), ".2f"))
    s3.metric("IC hit-rate", fmt(sigq.get("ic_hit_rate"), ".0%"))
    s4.metric("Signal autocorr", fmt(sigq.get("rank_autocorr"), ".2f"))
    st.write(("🟢 The signal carries statistically significant predictive information (|t| > 2)"
              if sigq.get("significant") else
              "🔴 The signal is NOT statistically predictive — a good backtest here would be luck"))
    decay = sigq.get("ic_decay") or {}
    if decay:
        st.write("IC by horizon: " + " · ".join(f"{k}d = {fmt(v, '+.3f')}" for k, v in decay.items()))
    st.caption("IC = cross-sectional rank correlation between the signal today and the return that "
               "follows it. A credible backtest should rest on a signal with a real, significant IC.")

pbo = res.get("pbo") or {}
if pbo and "error" not in pbo and not pbo.get("insufficient"):
    st.subheader("Probability of Backtest Overfitting (PBO)")
    p1, p2 = st.columns([1, 2])
    with p1:
        st.metric("PBO", fmt(pbo.get("pbo"), ".0%"))
        st.write("🟢 robust" if not pbo.get("overfit") else "🔴 overfit risk")
        st.caption(pbo.get("verdict", ""))
    with p2:
        hist = pbo.get("histogram") or {}
        edges, counts = hist.get("edges") or [], hist.get("counts") or []
        if counts and len(edges) == len(counts) + 1:
            centers = [round((edges[i] + edges[i + 1]) / 2, 3) for i in range(len(counts))]
            figp = go.Figure(go.Bar(x=centers, y=counts))
            figp.add_vline(x=0.0, line_dash="dash", line_color="#888")
            figp.update_layout(
                title="OOS rank-logit of the in-sample-best config across CSCV splits "
                      "(mass left of 0 = below-median out-of-sample = overfit)",
                xaxis_title="logit(out-of-sample rank)",
                height=300, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(figp, use_container_width=True)
    st.caption("PBO = the fraction of combinatorial splits where the best in-sample config lands in "
               "the LOSING half out-of-sample (logit ≤ 0). > 50% means the search is overfitting itself.")
elif pbo.get("insufficient"):
    st.caption(f"PBO: n/a — {pbo.get('note') or 'insufficient data for CSCV'}")
elif pbo.get("error"):
    st.caption(f"PBO: n/a — {pbo['error']}")

with st.expander("Full scorecard"):
    # cast to str: the scorecard mixes floats with text (e.g. cvar_verdict), which
    # a single Arrow column can't hold — stringifying keeps the debug table honest.
    _scoredf = pd.DataFrame([card]).T.rename(columns={0: "value"})
    st.dataframe(_scoredf.astype(str), use_container_width=True)

st.divider()
st.caption(f"⚖️ {DISCLAIMER}")
