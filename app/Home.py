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
import json
import pathlib
import sys
import time
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.service import (DISCLAIMER, FACTOR_CATALOG, ML_FEATURE_CHOICES,
                         ML_FEATURE_FACTORS, WorkflowError,
                         data_connections_status, run_backtest_workflow, run_model_comparison)
from research.local_history import LocalResearchHistory
from research.research_protocol import ProtocolError, ResearchProtocolStore
from research.trial_ledger import TrialLedger
from app.ui import empty_state, footer, page, plot

# The model ladder, in increasing complexity. Mirrors research.models.MODEL_NAMES,
# hardcoded here so launching the UI never requires the ML extras (sklearn/xgboost/
# lightgbm) to be installed — they're only needed when a comparison is actually run.
ML_MODEL_CHOICES = ("lasso", "elastic_net", "random_forest", "xgboost",
                    "lightgbm", "mlp", "stacking")

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

page("Strategy lab", "Backtests / Validation / Model comparison", section="Strategy lab", sidebar=_sb)


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


def run_ml_request(payload: dict) -> dict:
    """Opt-in ML model comparison — slow (~1-2 min). Same API/in-process split as run_request.

    ``payload`` may carry ml_factors / ml_models / horizon (the feature panel + ladder
    selection). In API mode they ride the JSON body; in-process we pull them out and
    pass them as keyword args so both paths honor the selection identically."""
    if st.session_state.get("api_mode"):
        r = httpx.post(f"{API_URL}/ml-compare", json=payload, timeout=600.0,
                       cookies=st.session_state.get("af_cookies", {}))
        if r.status_code != 200:
            raise WorkflowError(r.json().get("detail", f"API error {r.status_code}"))
        return r.json()
    body = {k: v for k, v in payload.items() if k not in ("ml_factors", "ml_models", "horizon")}
    return run_model_comparison(body, factor_names=payload.get("ml_factors"),
                                model_names=payload.get("ml_models"),
                                horizon=int(payload.get("horizon", 21)))


if "ledger" not in st.session_state:
    st.session_state["ledger"] = TrialLedger()
if "api_mode" not in st.session_state:
    st.session_state["api_mode"] = False if SHOT_MODE else api_is_up()
if "local_history" not in st.session_state:
    st.session_state["local_history"] = LocalResearchHistory()
if "protocol_store" not in st.session_state:
    st.session_state["protocol_store"] = ResearchProtocolStore()

with st.sidebar:
    st.caption("ENGINE")
    st.markdown("API connected" if st.session_state["api_mode"] else "Local engine (in-process)")
    if st.session_state["api_mode"]:
        st.link_button("API reference", f"{API_URL}/docs", icon=":material/open_in_new:")

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
    st.session_state["last_payload"] = _payload

# Research history remains independent of the editable strategy controls.
with st.sidebar:
    st.divider()
    _led = st.session_state["ledger"]
    _ta_now = (st.session_state.get("result") or {}).get("trial_audit") or {}
    _distinct = _ta_now.get("distinct_count", _led.distinct_count)
    st.metric("Strategies tested", _distinct, help="Distinct strategies in the current search. This count constrains the statistical correction.")
    with st.expander("Search history"):
        reset_confirmed = st.checkbox("Start a new search")
        if st.button("Reset search", disabled=not reset_confirmed, icon=":material/restart_alt:"):
            reset_ok = True
            if st.session_state.get("api_mode"):
                try:
                    response = httpx.post(f"{API_URL}/session/reset", cookies=st.session_state.get("af_cookies", {}), timeout=10.0)
                    reset_ok = response.status_code == 200
                except httpx.RequestError:
                    reset_ok = False
            if reset_ok:
                _led.reset()
                for key in ("result", "ml_result", "last_payload"):
                    st.session_state.pop(key, None)
                st.rerun()
            else:
                st.error("The server could not reset this search. History has been preserved.")

with st.expander("Research setup", expanded="result" not in st.session_state):
    universe, strategy, assumptions = st.columns(3)
    with universe:
        st.subheader("Universe")
        _bundle = data_connections_status()["licensed_bundle"]
        provider = st.radio(
            "Data source", ["synthetic", "yfinance", "licensed_bundle"], horizontal=True,
            format_func=lambda x: {
                "synthetic": "Synthetic demo", "yfinance": "Yahoo Finance",
                "licensed_bundle": "Licensed data bundle",
            }[x],
        )
        if provider in {"yfinance", "licensed_bundle"}:
            tickers = st.text_input("Companies", "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,XOM,JNJ,PG,KO,WMT,HD,BAC,DIS")
            symbols = [t.strip() for t in tickers.split(",") if t.strip()]
            periods, seed = 1512, 42
        else:
            symbols = []
            periods = st.slider("History (trading days)", 400, 3000, 1512, step=50)
            seed = st.number_input("Random seed", min_value=0, value=42, step=1, help="The same seed reproduces the same synthetic dataset.")
    with strategy:
        st.subheader("Strategy")
        _cats = list(dict.fromkeys(f["category"] for f in FACTOR_CATALOG))
        _want = _qp.get("factor", "quality") if SHOT_MODE else None
        _want_entry = next((f for f in FACTOR_CATALOG if f["name"] == _want), None)
        _cat_idx = _cats.index(_want_entry["category"]) if _want_entry else 0
        category = st.selectbox("Factor family", _cats, index=_cat_idx)
        _in_cat = [f for f in FACTOR_CATALOG if f["category"] == category]
        _names = [f["name"] for f in _in_cat]
        _labels = [f["label"] for f in _in_cat]
        _f_idx = _names.index(_want) if (_want_entry and _want in _names) else 0
        _label = st.selectbox("Factor", _labels, index=_f_idx)
        _entry = _in_cat[_labels.index(_label)]
        factor = _entry["name"]
        if _entry["kind"] == "price":
            lookback = st.slider("Lookback (days)", 21, 504, 252, step=21)
            skip = st.slider("Skip recent (days)", 0, max(7, lookback - 7), min(21, lookback - 7), step=7,
                             help="Recent observations excluded from the signal calculation.")
        else:
            lookback, skip = 252, 21
            st.caption("Fundamental factors need synthetic data or a filing-dated licensed bundle.")
    with assumptions:
        st.subheader("Assumptions")
        cost_bps = st.slider("Trading cost (basis points)", 0.0, 50.0, 5.0, step=0.5,
                             help="One basis point is 0.01%. Costs apply to each unit of turnover.")
        n_trials = st.number_input("Strategy variants tried", min_value=1, max_value=500, value=50, step=1,
                                   help="Include discarded variants. The engine also enforces the recorded search count.")
        st.caption("Gross exposure: 100% | Start: January 2015")
    _bundle_missing = provider == "licensed_bundle" and not _bundle["ready_for_price_research"]
    _fundamentals_missing = (
        provider == "licensed_bundle" and _entry["kind"] != "price"
        and not _bundle["ready_for_fundamental_research"]
    )
    _unsupported = _bundle_missing or _fundamentals_missing or (
        provider == "yfinance" and _entry["kind"] != "price"
    )
    if _bundle_missing:
        st.warning("No usable licensed data bundle is configured. Review Data connections before running this study.")
    elif _fundamentals_missing:
        st.warning("This bundle does not yet attest to filing-dated fundamentals, so fundamental factors remain blocked.")
    elif provider == "yfinance" and _entry["kind"] != "price":
        st.warning("Yahoo Finance does not provide the point-in-time fundamentals required for this factor.")
    elif provider == "licensed_bundle":
        st.caption(f"Connected licensed source: {_bundle['provider_name'] or 'local bundle'}")
    run = st.button("Run backtest", type="primary", icon=":material/play_arrow:", disabled=_unsupported)

# ----------------------------- run -----------------------------
if run:
    payload = {"provider": provider, "symbols": symbols, "factor": factor,
               "lookback": lookback, "skip": skip, "cost_bps": cost_bps,
               "gross": 1.0, "n_trials": n_trials, "periods": periods,
               "seed": int(seed), "start": "2015-01-02"}
    st.session_state.pop("result", None)
    st.session_state["last_payload"] = payload   # reused by the opt-in ML comparison
    st.session_state.pop("ml_result", None)       # stale once the universe changes
    try:
        with st.spinner("Running point-in-time backtest…"):
            st.session_state["result"] = run_request(payload)
        st.session_state["local_history"].record(
            "backtest", f"{factor} / {provider}", st.session_state["result"]
        )
        st.rerun()
    except WorkflowError as e:
        st.error(f"Input problem: {e}")
    except Exception as e:
        st.error(f"Run failed: {e}")

res = st.session_state.get("result")
if not res:
    empty_state("No strategy run yet", "No performance or validation results in this search.")
    footer(DISCLAIMER)
    st.stop()

meta, card = res["meta"], res["scorecard"]
oos, wf, tails = res["out_of_sample"], res["walk_forward"], res["fat_tails"]
evidence_card = res.get("evidence_card") or {}

# ----------------------------- verdict -----------------------------
credible = res["verdict"].startswith("CREDIBLE")
wf_dsr = wf.get("deflated_sr") if "error" not in wf else None
_folds = wf.get("n_splits") if "error" not in wf else None
_ta = res.get("trial_audit") or {}
# the full-sample DSR is deflated by the trial-ledger count; the OOS walk-forward
# DSR is deflated by folds-as-trials — two DISTINCT haircuts, kept visibly separate.
_trial_word = "ledger-enforced" if _ta.get("distinct_count") is not None else "declared"
if meta["provider"] == "synthetic":
    st.warning("Synthetic-data demonstration. Passing a statistical check here does not establish a real investment edge.", icon=":material/science:")
elif meta["provider"] == "yfinance":
    st.warning("Current-ticker universe: survivorship bias is not removed. This is exploratory research.", icon=":material/warning:")
elif meta["provider"] == "licensed_bundle":
    _readiness = data_connections_status()["licensed_bundle"]
    if _readiness["survivorship_free_universe"] and _readiness["includes_delisted_securities"]:
        st.info("Licensed bundle declares historical universe membership and delisted-security coverage. Keep the exported audit alongside the vendor evidence.", icon=":material/fact_check:")
    else:
        st.warning("Licensed bundle coverage is limited by its manifest. Review Data connections before interpreting this study.", icon=":material/warning:")
# Data health comes BEFORE the verdict: if the sample never varied, the verdict
# below is describing nothing and the reader needs to know that first.
_validity = (res.get("strategy_report") or {}).get("research_validity") or {}
if _validity and not _validity.get("supports_performance_claim", True):
    st.error(
        f"**Insufficient variation for a performance claim.** {_validity.get('conclusion', '')}",
        icon=":material/error:",
    )
elif _validity.get("status") == "limited_variation":
    st.warning(_validity.get("conclusion", ""), icon=":material/info:")
(st.success if credible else st.warning)(
    "**Passes the walk-forward statistical check**" if credible else "**Does not pass the walk-forward statistical check**",
    icon=":material/check_circle:" if credible else ":material/info:",
)
_evidence_status = evidence_card.get("status", "needs_review")
_evidence_headline = evidence_card.get("headline", "Evidence needs review")
_evidence_message = {
    "research_supported": st.success,
    "exploratory": st.info,
    "needs_review": st.warning,
    "inconclusive": st.warning,
}.get(_evidence_status, st.info)
_evidence_message(f"**{_evidence_headline}** — {evidence_card.get('note', '')}", icon=":material/fact_check:")
st.caption(f"Deflated Sharpe: full sample {fmt(card['deflated_sr'], '.3f')} "
           f"({meta['n_trials']} {_trial_word} trials) | Walk-forward {fmt(wf_dsr, '.3f')} "
           f"({_folds or 'n/a'} folds) | Unadjusted in-sample PSR {fmt(card['psr_vs_0'], '.3f')}")
if _ta.get("haircut_was_raised"):
    st.warning(f"You declared {_ta['declared_n_trials']} trial(s), but this workspace has run "
               f"{_ta['effective_n_trials']} distinct strategies — the Deflated Sharpe uses the "
               f"larger, honest number. The recorded search count takes priority.")

st.caption(f"{meta['n_symbols']} symbols × {meta['n_days']} days "
           f"({meta['start_date']} → {meta['end_date']}) · factor: {meta['factor']} "
           f"(effective lookback {meta.get('effective_lookback') or 'n/a'}) · "
           f"costs: {meta['cost_bps']} bps · provider: {meta['provider']}")

# ----------------------------- headline metrics -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annual return", fmt(card['cagr'], '.2%'), help="Compound annual growth rate, net of modeled trading costs.")
c2.metric("Sharpe ratio", fmt(card['ann_sharpe'], '.2f'), help="Annualized return relative to return variability. Higher is not automatically more credible.")
c3.metric("Largest drawdown", fmt(card['max_drawdown'], '.2%'), help="Largest peak-to-trough loss in the backtest.")
c4.metric("OOS confidence", fmt(wf_dsr, '.3f'), help="Walk-forward Deflated Sharpe statistic (0-1), corrected by folds. Not the probability of making money.")
c5.metric("Winning days", fmt(card['hit_rate'], '.1%'), help="Share of measured daily returns above zero.")

# ----------------------------- detail tabs -----------------------------
# The dense detail is organized into tabs so the page reads top-down: verdict and
# headline numbers above, then drill into Performance / Honesty / Signal / Overfitting.
t_evidence, t_perf, t_honest, t_risk, t_relative, t_signal, t_pbo, t_attr, t_ml, t_details = st.tabs(
    ["Evidence", "Performance", "Validation", "Risk & rigor", "Relative & tail risk", "Signal quality", "Overfitting",
     "Return drivers", "Model comparison", "Run details"])

with t_evidence:
    st.caption("A conservative decision record. Every gate must be inspected; a green status is not a trading instruction.")
    _audit = res.get("data_audit") or {}
    _stress = res.get("robustness_matrix") or {}
    left, right = st.columns(2)
    with left:
        st.markdown("**Data audit**")
        st.metric("Data status", str(_audit.get("status", "n/a")).replace("_", " ").title())
        st.write(f"Panel coverage: **{fmt(_audit.get('panel_coverage'), '.1%')}**")
        st.write(f"Historical universe: **{'attested' if _audit.get('historical_universe_membership') else 'not attested'}**")
        st.write(f"Delisted securities: **{'attested' if _audit.get('delisted_securities') else 'not attested'}**")
        for _reason in _audit.get("reasons", []):
            st.caption(_reason)
    with right:
        st.markdown("**Cost and delay stress**")
        st.metric("Positive-Sharpe scenarios", fmt(_stress.get("positive_sharpe_rate"), ".0%"))
        st.write(_stress.get("summary", "No stress result available."))
        _stress_rows = _stress.get("scenarios") or []
        if _stress_rows:
            st.dataframe(pd.DataFrame(_stress_rows), use_container_width=True, hide_index=True)
    st.markdown("**Evidence gates**")
    for _gate in evidence_card.get("gates", []):
        _render = {"pass": st.success, "warning": st.warning, "fail": st.error}.get(_gate.get("status"), st.info)
        _render(f"**{_gate.get('label')}** — {_gate.get('detail')}", icon={"pass": ":material/check_circle:", "warning": ":material/warning:", "fail": ":material/cancel:"}.get(_gate.get("status"), ":material/info:"))
    with st.expander("Research protocol", expanded=False):
        st.caption("Pre-register a hypothesis and protect the final holdout from repeated inspection. This is stored only in this local workspace.")
        _store = st.session_state["protocol_store"]
        _study_id = st.session_state.get("active_protocol_id")
        if not _study_id:
            _hypothesis = st.text_input("Hypothesis", key="protocol_hypothesis", placeholder="Example: quality remains positive after conservative costs.")
            _p1, _p2, _p3 = st.columns(3)
            with _p1:
                _research_end = st.date_input("Research phase ends", value=date(2021, 12, 31), key="protocol_research_end")
            with _p2:
                _validation_end = st.date_input("Validation phase ends", value=date(2023, 12, 31), key="protocol_validation_end")
            with _p3:
                _holdout_end = st.date_input("Final holdout ends", value=date(2025, 12, 31), key="protocol_holdout_end")
            if st.button("Create protocol", icon=":material/lock:"):
                try:
                    _study = _store.create(_hypothesis, {"research_end": _research_end.isoformat(), "validation_end": _validation_end.isoformat(), "final_holdout_end": _holdout_end.isoformat()})
                    st.session_state["active_protocol_id"] = _study["id"]
                    st.rerun()
                except ProtocolError as _error:
                    st.error(str(_error))
        else:
            _study = _store.summary(_study_id)
            st.write(f"**Hypothesis:** {_study['hypothesis']}")
            st.caption("Boundaries: research through " + _study["boundaries"]["research_end"] + " · validation through " + _study["boundaries"]["validation_end"] + " · final holdout through " + _study["boundaries"]["final_holdout_end"] + ".")
            _stages = ["exploration", "validation"] + (["final_holdout"] if _study["final_holdout_available"] else [])
            _stage = st.selectbox("Record this run as", _stages, key="protocol_stage")
            if st.button("Record this evidence", icon=":material/lock:"):
                try:
                    _store.record(_study_id, _stage, (res.get("manifest") or {})["research_fingerprint"], {"evidence_status": evidence_card.get("status"), "run_end": meta.get("end_date")})
                    st.success("Research record locked locally.")
                    st.rerun()
                except (ProtocolError, KeyError) as _error:
                    st.error(str(_error))
            if _study["runs"]:
                st.dataframe(pd.DataFrame(_study["runs"])[["created_at", "stage", "fingerprint"]], use_container_width=True, hide_index=True)
            if not _study["final_holdout_available"]:
                st.warning("Final holdout is consumed. Fork this study before another final test.")

with t_perf:
    eq = pd.DataFrame(res["equity_curve"])
    dd = pd.DataFrame(res["drawdown_curve"])
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Scatter(x=eq["date"], y=eq["value"], mode="lines", name="equity"))
        fig.update_layout(title="Equity curve (net of costs)", height=380,
                          margin=dict(l=10, r=10, t=40, b=10))
        plot(fig)
    with right:
        fig2 = go.Figure(go.Scatter(x=dd["date"], y=dd["value"], mode="lines",
                                    fill="tozeroy", name="drawdown", line=dict(color="#c64b60"), fillcolor="rgba(198,75,96,0.12)"))
        fig2.update_layout(title="Drawdown", height=380, margin=dict(l=10, r=10, t=40, b=10))
        plot(fig2)

with t_honest:
    st.caption("Out-of-sample checks and tail-risk diagnostics.")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Out-of-sample (70/30)**")
        st.write(f"In-sample Sharpe: **{fmt(oos['in_sample_sharpe'], '+.3f')}**")
        st.write(f"Out-of-sample Sharpe: **{fmt(oos['out_sample_sharpe'], '+.3f')}**")
        st.write(f"Degradation: **{fmt(oos['degradation'], '+.3f')}**")
        st.write(("**Overfit warning** — the edge decays out-of-sample"
                  if oos["overfit_warning"] else "No IS→OOS decay"))
        st.write(("OOS record is statistically significant"
                  if oos["oos_significant"] else "OOS record not yet significant (PSR < 0.95)"))
    with b:
        folds = wf.get("n_oos_folds", "?") if "error" not in wf else 0
        st.markdown(f"**Walk-forward ({folds} OOS folds)**")
        if "error" in wf:
            st.write(f"n/a: {wf['error']}")
        else:
            st.write(f"Fold OOS Sharpes: {wf['fold_oos_sharpes']}")
            st.write(f"Stitched OOS Sharpe: **{fmt(wf['stitched_oos_sharpe'], '+.3f')}**")
            st.write(f"Deflated SR (folds-as-trials): **{fmt(wf['deflated_sr'], '.3f')}**")
            st.write("passes" if wf["passes"] else "does not pass")
            st.caption(f"Purged at fold boundaries: {wf.get('total_purged_bars', 0)} bars "
                       f"(purge {wf.get('purge_bars', 0)} + embargo {wf.get('embargo_bars', 0)}) — "
                       f"no lookback leak into out-of-sample.")
    with c:
        st.markdown("**Tail risk**")
        st.write(f"Skew: **{fmt(tails.get('skew'), '+.2f')}** · "
                 f"Excess kurtosis: **{fmt(tails.get('excess_kurtosis'), '.1f')}**")
        st.write(f"Jarque-Bera p: {fmt(tails.get('jarque_bera_p'), '.2g')}")
        st.write(("No strong departure from normality detected" if tails.get("returns_are_normal")
                  else "FAT TAILS — Gaussian VaR understates risk"))
        st.caption(tails.get("verdict", ""))
        st.write(f"CVaR 95%: **{fmt(card.get('cvar_95'), '.2%')}** · "
                 f"CVaR 99%: **{fmt(card.get('cvar_99'), '.2%')}** _(historical, per-day loss)_")
        if card.get("cvar_verdict"):
            st.caption(card["cvar_verdict"])

with t_risk:
    report = res.get("strategy_report") or {}
    if report.get("error"):
        st.caption(f"Expanded tearsheet: n/a — {report['error']}")
    else:
        downside = report.get("volatility_downside_risk") or {}
        rigor = report.get("statistical_rigor") or {}
        validity = report.get("research_validity") or {}
        if validity:
            st.markdown("**Data health**")
            v_returns = validity.get("returns") or {}
            v_signal = validity.get("signal_dispersion") or {}
            v_book = validity.get("positions") or {}
            (st.error if not validity.get("supports_performance_claim", True)
             else st.info)(validity.get("conclusion", ""),
                           icon=":material/error:" if not validity.get("supports_performance_claim", True)
                           else ":material/info:")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Effective observations", v_returns.get("effective_observations", 0),
                      help="Finite return observations available to every statistic below.")
            h2.metric("Non-flat returns", v_returns.get("non_flat_returns", 0),
                      help="Observations where the return actually moved. Zero means nothing was traded.")
            h3.metric("Signal dispersion",
                      fmt(v_signal.get("mean_dispersion"), ".4f") if v_signal.get("available") else "n/a",
                      help="Average spread of the signal across names on a date. Zero means it cannot rank the universe.")
            h4.metric("Avg active names",
                      fmt(v_book.get("average_active_names"), ".2f") if v_book.get("available") else "n/a",
                      help="Average number of held positions per date.")
            if validity.get("has_undefined_statistics"):
                st.caption("Undefined statistics in this run: "
                           + ", ".join(validity.get("undefined_statistics", []))
                           + ". Undefined means the sample could not support the calculation, not that the result was weak.")
            if v_book.get("available") and v_book.get("concentration_herfindahl") is not None:
                st.caption(f"Average position concentration (Herfindahl): "
                           f"{fmt(v_book['concentration_herfindahl'], '.3f')} — 1.00 is a single-name book.")
            st.caption(validity.get("note", ""))
            st.divider()
        st.caption("Downside-risk and inference checks. These describe the backtest; they do not create an investment recommendation.")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Annual volatility", fmt(downside.get("annualized_volatility"), ".2%"),
                  help="Sample return volatility annualized using 252 trading days.")
        r2.metric("Sortino ratio", fmt(downside.get("sortino_ratio"), ".2f"),
                  help="Return relative to downside deviation below a 0% daily minimum acceptable return.")
        r3.metric("Calmar / MAR", fmt(downside.get("calmar_ratio"), ".2f"),
                  help="CAGR divided by the largest drawdown. MAR is the same convention here.")
        r4.metric("Ulcer Index", fmt(downside.get("ulcer_index"), ".2%"),
                  help="Root-mean-square drawdown: depth and persistence of pain both matter.")
        r5.metric("Omega (0%)", fmt(downside.get("omega_ratio"), ".2f"),
                  help="Gains above the threshold divided by losses below it.")

        st.divider()
        st.markdown("**Statistical rigor**")
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Probabilistic Sharpe", fmt(rigor.get("probabilistic_sharpe_ratio"), ".3f"),
                  help="Probability the true per-period Sharpe exceeds zero, accounting for skewness and kurtosis.")
        q2.metric("HAC Sharpe t-stat", fmt(rigor.get("hac_sharpe_tstat"), ".2f"),
                  help="Newey-West corrected t-statistic. It discounts serial correlation in the return record.")
        q3.metric("HAC IC t-stat", fmt(rigor.get("ic_hac_tstat"), ".2f"),
                  help="Newey-West corrected t-statistic for the signal's Information Coefficient.")
        _mtrl = rigor.get("minimum_track_record_length_95")
        q4.metric("Min. record length", f"{_mtrl} days" if _mtrl is not None else "n/a",
                  help="Observations required for the current per-period Sharpe to clear zero at 95% confidence.")
        _cpcv = rigor.get("cpcv_pbo") or {}
        q5.metric("Purged CPCV PBO", fmt(_cpcv.get("pbo"), ".0%"),
                  help="Fraction of combinatorial purged splits where the in-sample winner ranks below the OOS median.")
        _white = rigor.get("white_reality_check") or {}
        if _white.get("computed"):
            st.caption("White's Reality Check p-value: " + fmt(_white.get("p_value"), ".3f") +
                       " — lower values mean the best parameter variant is less consistent with pure search noise.")
        elif _white:
            st.caption(_white.get("note", "White's Reality Check is available in the deep report."))
        _k = downside.get("k_ratio")
        st.caption("Pain Index " + fmt(downside.get("pain_index"), ".2%") +
                   " · Sterling ratio " + fmt(downside.get("sterling_ratio"), ".2f") +
                   " · K-ratio " + fmt(_k, ".2f") +
                   ". Formula definitions and chart transforms are in the run export and the analytics guide.")

        advanced = report.get("advanced_econometrics") or {}
        regime = report.get("regime_analysis") or {}
        with st.expander("Advanced evidence checks", expanded=False):
            garch = advanced.get("conditional_volatility") or {}
            model_tests = advanced.get("candidate_model_tests") or {}
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("GARCH 1-day vol", fmt(garch.get("first_day_annualized_volatility"), ".2%"),
                      help="Conditional annualized volatility estimated from the return history, not a return forecast.")
            e2.metric("Detected breaks", str(regime.get("n_detected_breaks", "n/a")),
                      help="Historical structural breaks found by trailing, refit-only analysis.")
            e3.metric("Current regime age", f"{regime.get('current_regime_age_observations')} days" if regime.get("current_regime_age_observations") is not None else "n/a",
                      help="Observations since the latest detected historical break.")
            mcs = model_tests.get("model_confidence_set") or {}
            e4.metric("MCS survivors", str(len(mcs.get("included_models", []))) if model_tests.get("available") else "n/a",
                      help="Candidate models not rejected as inferior within the tested model set.")
            if not garch.get("available"):
                st.caption(garch.get("reason", "GARCH diagnostics unavailable for this run."))
            elif garch.get("average_horizon_annualized_volatility") is not None:
                st.caption("GARCH average 21-day conditional annualized volatility: " + fmt(garch.get("average_horizon_annualized_volatility"), ".2%") + ".")
            if model_tests.get("available"):
                spa = model_tests.get("spa") or {}
                st.caption("SPA consistent p-value: " + fmt((spa.get("p_values") or {}).get("consistent"), ".3f") +
                           " · superior candidates: " + (", ".join(spa.get("superior_models") or []) or "none") + ".")
            if regime.get("available"):
                st.caption("Latest detected break: " + (regime.get("latest_break_date") or "none") +
                           ". Regime labels are fit using only data available at each historical refit date.")
            elif regime.get("reason"):
                st.caption(regime["reason"])

with t_relative:
    report = res.get("strategy_report") or {}
    relative = report.get("benchmark_relative") or {}
    tail_report = report.get("tail_risk_distribution") or {}
    if not relative or not tail_report:
        st.caption("Benchmark-relative and tail-risk analytics are unavailable for this run.")
    else:
        benchmark_label = (report.get("metadata") or {}).get("benchmark_label", "Selected benchmark")
        st.caption(f"Comparison benchmark: {benchmark_label}. It is a research comparison, not a trading recommendation.")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Information ratio", fmt(relative.get("information_ratio"), ".2f"),
                  help="Annualized active return divided by tracking error.")
        b2.metric("Tracking error", fmt(relative.get("tracking_error"), ".2%"),
                  help="Annualized variability of the strategy's return relative to the comparison benchmark.")
        alpha = relative.get("jensens_alpha") or {}
        b3.metric("Jensen alpha", fmt(alpha.get("alpha_annualized"), ".2%"),
                  help="Regression intercept annualized from strategy returns against the benchmark.")
        b4.metric("Beta", fmt(alpha.get("beta"), ".2f"), help="Sensitivity of strategy returns to benchmark returns.")
        b5.metric("Correlation", fmt(relative.get("correlation"), ".2f"), help="Linear co-movement with the benchmark.")
        captures = relative.get("capture_ratios") or {}
        st.caption("Up capture " + fmt(captures.get("up_capture"), ".2f") +
                   " · Down capture " + fmt(captures.get("down_capture"), ".2f") +
                   " · active CAGR spread " + fmt(relative.get("active_cagr_spread"), ".2%") + ".")

        rolling = pd.DataFrame(relative.get("rolling_chart_data") or [])
        if not rolling.empty:
            rolling_left, rolling_right = st.columns(2)
            with rolling_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["sharpe_lower"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["sharpe_upper"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(12,137,123,0.14)", name="95% IID band"))
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["rolling_sharpe"], mode="lines", name="rolling Sharpe", line=dict(color="#0c897b")))
                fig.add_hline(y=0, line_color="#87929d", line_width=1)
                fig.update_layout(title=f"{relative.get('rolling_window', 63)}-day rolling Sharpe", height=320, margin=dict(l=10, r=10, t=40, b=10))
                plot(fig)
            with rolling_right:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["beta_lower"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["beta_upper"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(69,115,165,0.14)", name="95% OLS band"))
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["rolling_beta"], mode="lines", name="rolling beta", line=dict(color="#4573a5")))
                fig.add_hline(y=1, line_color="#87929d", line_width=1)
                fig.update_layout(title=f"{relative.get('rolling_window', 63)}-day rolling beta", height=320, margin=dict(l=10, r=10, t=40, b=10))
                plot(fig)
            if "rolling_ic" in rolling:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["ic_lower"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["ic_upper"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(161,102,37,0.14)", name="95% IID band"))
                fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["rolling_ic"], mode="lines", name="rolling IC", line=dict(color="#a16625")))
                fig.add_hline(y=0, line_color="#87929d", line_width=1)
                fig.update_layout(title=f"{relative.get('rolling_window', 63)}-day rolling information coefficient", height=280, margin=dict(l=10, r=10, t=40, b=10))
                plot(fig)

        st.divider()
        st.markdown("**Tail risk and return shape**")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Historical VaR (95%)", fmt(tail_report.get("historical_var_95"), ".2%"), help="Historical one-period loss threshold at 95% confidence.")
        t2.metric("Historical ES (95%)", fmt(tail_report.get("historical_expected_shortfall_95"), ".2%"), help="Average loss beyond historical VaR at 95% confidence.")
        t3.metric("Skewness", fmt(tail_report.get("skewness"), ".2f"))
        t4.metric("Excess kurtosis", fmt(tail_report.get("excess_kurtosis"), ".2f"))
        t5.metric("Max time underwater", f"{tail_report.get('max_time_under_water_periods', 0)} days")
        st.caption("Parametric VaR 95% " + fmt(tail_report.get("parametric_var_95"), ".2%") +
                   " · Historical VaR 99% " + fmt(tail_report.get("historical_var_99"), ".2%") +
                   " · Tail ratio " + fmt(tail_report.get("tail_ratio"), ".2f") + ".")
        dist = tail_report.get("distribution_chart_data") or {}
        hist = pd.DataFrame(dist.get("histogram") or [])
        qq = pd.DataFrame(dist.get("qq") or [])
        if not hist.empty or not qq.empty:
            hist_col, qq_col = st.columns(2)
            with hist_col:
                if not hist.empty:
                    centers = (hist["left"] + hist["right"]) / 2
                    fig = go.Figure(go.Bar(x=centers, y=hist["count"], width=hist["right"] - hist["left"], marker_color="#0c897b"))
                    fig.update_layout(title="Daily-return distribution", height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="return", yaxis_title="observations")
                    plot(fig)
            with qq_col:
                if not qq.empty:
                    lower = min(qq["normal_quantile"].min(), qq["observed_return"].min())
                    upper = max(qq["normal_quantile"].max(), qq["observed_return"].max())
                    fig = go.Figure(go.Scatter(x=qq["normal_quantile"], y=qq["observed_return"], mode="markers", marker=dict(color="#4573a5", size=6), name="returns"))
                    fig.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", line=dict(color="#87929d", dash="dash"), name="Normal reference"))
                    fig.update_layout(title="Normal QQ plot", height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="fitted Normal quantile", yaxis_title="observed return")
                    plot(fig)

with t_signal:
    sigq = res.get("signal_quality") or {}
    health = res.get("factor_health") or {}
    if sigq:
        st.markdown("**Is the signal itself predictive? (Information Coefficient)**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean IC", fmt(sigq.get("mean_ic"), ".4f"))
        s2.metric("IC t-stat", fmt(sigq.get("ic_tstat"), ".2f"))
        s3.metric("IC hit-rate", fmt(sigq.get("ic_hit_rate"), ".0%"))
        s4.metric("Signal autocorr", fmt(sigq.get("rank_autocorr"), ".2f"))
        st.write(("The signal carries statistically significant predictive information (|t| > 2)"
                  if sigq.get("significant") else
                  "Predictive information is not statistically established in this sample"))
        decay = sigq.get("ic_decay") or {}
        if decay:
            st.write("IC by horizon: " + " · ".join(f"{k}d = {fmt(v, '+.3f')}" for k, v in decay.items()))
        st.caption("IC = cross-sectional rank correlation between the signal today and the return that "
                   "follows it. A credible backtest should rest on a signal with a real, significant IC.")
    else:
        st.caption("No signal-quality scorecard available for this run.")
    temporal = res.get("temporal_stability") or {}
    if temporal:
        st.divider()
        st.markdown("**Chronological stability**")
        status = temporal.get("status", "insufficient_evidence")
        conclusion = temporal.get("conclusion", "")
        if status == "stable":
            st.success(conclusion, icon=":material/timeline:")
        elif status == "weakened":
            st.warning(conclusion, icon=":material/trending_down:")
        elif status == "unstable":
            st.error(conclusion, icon=":material/compare_arrows:")
        else:
            st.info(conclusion, icon=":material/info:")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Chronology", status.replace("_", " ").title())
        t2.metric("Valid IC observations", temporal.get("available_ic_observations", 0))
        t3.metric("Earliest to latest IC", fmt(temporal.get("latest_minus_earliest_ic"), "+.4f"))
        t4.metric("Direction consistency", fmt(temporal.get("sign_consistency"), ".0%"))
        cohorts = pd.DataFrame(temporal.get("cohorts") or [])
        if not cohorts.empty:
            display = cohorts.rename(columns={
                "cohort": "Cohort", "start": "Start", "end": "End", "ic_observations": "IC observations",
                "mean_ic": "Mean IC", "ic_ir": "IC information ratio", "ic_hit_rate": "IC hit rate",
                "mean_ic_ci_95": "95% mean IC interval", "strategy_cumulative_return": "Strategy return (descriptive)",
            })
            display["95% mean IC interval"] = display["95% mean IC interval"].map(
                lambda interval: "n/a" if not isinstance(interval, list) or None in interval
                else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
            )
            st.dataframe(display[[
                "Cohort", "Start", "End", "IC observations", "Mean IC", "IC information ratio",
                "IC hit rate", "95% mean IC interval", "Strategy return (descriptive)",
            ]], use_container_width=True, hide_index=True,
                column_config={
                    "Mean IC": st.column_config.NumberColumn(format="%.4f"),
                    "IC information ratio": st.column_config.NumberColumn(format="%.3f"),
                    "IC hit rate": st.column_config.NumberColumn(format="%.0f%%"),
                    "Strategy return (descriptive)": st.column_config.NumberColumn(format="%.2f%%"),
                })
            figure = go.Figure(go.Scatter(
                x=display["Cohort"], y=display["Mean IC"], mode="lines+markers",
                marker=dict(color="#0c897b", size=8), line=dict(color="#0c897b"), name="Cohort mean IC",
            ))
            figure.add_hline(y=0, line_color="#87929d", line_width=1)
            figure.update_layout(
                title="Information coefficient by chronological cohort", height=280,
                margin=dict(l=10, r=10, t=40, b=10), xaxis_title="chronological cohort", yaxis_title="mean IC",
            )
            plot(figure, height=280)
        st.caption("Cohorts are fixed chronological eras separated by an embargo. This checks whether an observed relationship persists over time; it does not prove an edge, causality, or future performance.")
    if health.get("available"):
        st.divider()
        st.markdown("**Factor health**")
        _coverage, _turnover, _quantiles = health.get("coverage") or {}, health.get("turnover") or {}, health.get("quantile_returns") or {}
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Signal coverage", fmt(_coverage.get("average"), ".1%"))
        h2.metric("Lowest daily coverage", fmt(_coverage.get("minimum"), ".1%"))
        h3.metric("Top-quantile turnover", fmt(_turnover.get("top_quantile_turnover"), ".1%"))
        h4.metric("Top-minus-bottom return", fmt(_quantiles.get("top_minus_bottom"), ".3%"))
        _monthly_ic = pd.DataFrame(health.get("monthly_ic") or [])
        if not _monthly_ic.empty:
            fig = go.Figure(go.Bar(x=_monthly_ic["month"], y=_monthly_ic["ic"], marker_color="#0c897b"))
            fig.add_hline(y=0, line_color="#87929d", line_width=1)
            fig.update_layout(title="Monthly information coefficient", height=280, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="month", yaxis_title="IC")
            plot(fig)
        st.caption("Coverage and turnover expose when a signal disappears or changes too quickly to implement. Quantile spreads show whether high-ranked names actually separate from low-ranked names.")

with t_pbo:
    pbo = res.get("pbo") or {}
    if pbo and "error" not in pbo and not pbo.get("insufficient"):
        st.markdown("**Probability of Backtest Overfitting (PBO)**")
        p1, p2 = st.columns([1, 2])
        with p1:
            st.metric("PBO", fmt(pbo.get("pbo"), ".0%"))
            st.write(pbo.get("verdict") or "No risk assessment available.")
        with p2:
            hist = pbo.get("histogram") or {}
            edges, counts = hist.get("edges") or [], hist.get("counts") or []
            if counts and len(edges) == len(counts) + 1:
                centers = [round((edges[i] + edges[i + 1]) / 2, 3) for i in range(len(counts))]
                figp = go.Figure(go.Bar(x=centers, y=counts))
                figp.add_vline(x=0.0, line_dash="dash", line_color="#888")
                figp.update_layout(
                    title="Out-of-sample ranking of selected strategies",
                    xaxis_title="logit(out-of-sample rank)",
                    height=300, margin=dict(l=10, r=10, t=55, b=10))
                plot(figp, height=300)
        st.caption("PBO = the fraction of combinatorial splits where the best in-sample config lands in "
                   "the LOSING half out-of-sample (logit ≤ 0). > 50% means the search is overfitting itself.")
    elif pbo.get("insufficient"):
        st.caption(f"PBO: n/a — {pbo.get('note') or 'insufficient data for CSCV'}")
    elif pbo.get("error"):
        st.caption(f"PBO: n/a — {pbo['error']}")
    else:
        st.caption("PBO not computed for this run.")

with t_attr:
    attr = res.get("attribution") or {}
    if attr.get("error"):
        st.caption(f"Attribution: n/a — {attr['error']}")
    elif attr.get("insufficient"):
        st.caption(f"Attribution: n/a — {attr.get('note') or 'insufficient data'}")
    else:
        st.markdown(f"**Factor attribution ({attr.get('model', '?').upper()})** — how much of the "
                    "return is just factor beta vs. genuine alpha")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alpha (annualized)", fmt(attr.get("alpha_annualized"), "+.2%"))
        m2.metric("Alpha t-stat", fmt(attr.get("alpha_tstat"), ".2f"))
        m3.metric("R²", fmt(attr.get("r_squared"), ".0%"))
        m4.metric("Systematic var", fmt(attr.get("systematic_var_share"), ".0%"))
        _at = attr.get("alpha_tstat")
        st.write(("Significant alpha beyond the factors (|t| > 2)"
                  if (_at is not None and abs(_at) > 2) else
                  "No significant alpha — the evidence for return beyond factor exposure is inconclusive"))
        if attr.get("verdict"):
            st.caption(attr["verdict"])
        betas, bts = attr.get("betas") or {}, attr.get("beta_tstats") or {}
        if betas:
            st.dataframe(pd.DataFrame({
                "factor": list(betas.keys()),
                "beta": [None if betas[k] is None else round(betas[k], 3) for k in betas],
                "t-stat": [None if bts.get(k) is None else round(bts[k], 1) for k in betas],
            }), use_container_width=True, hide_index=True)
        if attr.get("factors_missing"):
            st.caption(f"Factors not available for this run: {', '.join(attr['factors_missing'])}.")
        st.caption("Regression against Fama-French / Carhart factor returns. R² measures explained variation; "
                   "an insignificant alpha is not evidence of an independent return advantage.")

with t_ml:
    st.caption("Purged cross-validation | Benchmark: ElasticNet | Evaluation: out-of-sample")
    _lp = st.session_state.get("last_payload") or {}
    if _lp.get("provider") != "synthetic":
        st.info("ML comparison needs the **synthetic** provider — it builds a point-in-time "
                "feature panel from fundamentals, which free data can't supply yet.")
    else:
        # --- build the feature panel + ladder from the factor library --------
        mc1, mc2 = st.columns(2)
        with mc1:
            sel_factors = st.multiselect(
                "Features (from the factor library)", list(ML_FEATURE_CHOICES),
                default=list(ML_FEATURE_FACTORS),
                help="The cross-sectional signals fed to every model as inputs. "
                     "Each is z-scored per day and lagged point-in-time — no look-ahead.")
        with mc2:
            sel_models = st.multiselect(
                "Models in the ladder", list(ML_MODEL_CHOICES), default=list(ML_MODEL_CHOICES),
                help="ElasticNet is always the baseline the rest are judged against — "
                     "it's added automatically if you leave it out.")
        horizon = st.slider("Forward-return horizon (days)", 5, 63, 21, step=1,
                            help="The label each model predicts: the cross-sectional return "
                                 "over the next N trading days.")
        st.caption("Purged cross-validation | Estimated runtime: 1-2 minutes | Results limited to this synthetic dataset")
        _disabled = len(sel_factors) == 0 or len(sel_models) == 0
        if st.button("Run model comparison", type="primary", disabled=_disabled, icon=":material/play_arrow:"):
            st.session_state.pop("ml_result", None)
            ml_payload = {**_lp, "ml_factors": sel_factors, "ml_models": sel_models,
                          "horizon": int(horizon)}
            try:
                with st.spinner("Fitting the model ladder under purged CV…"):
                    st.session_state["ml_result"] = run_ml_request(ml_payload)
            except WorkflowError as e:
                st.error(f"ML comparison problem: {e}")
            except Exception as e:
                st.error(f"ML comparison failed: {e}")
        if _disabled:
            st.caption("Pick at least one feature and one model to run.")
    mlr = st.session_state.get("ml_result")
    if mlr and "leaderboard" in mlr:
        _beats = mlr.get("complexity_beats_linear")
        (st.info if _beats else st.success)(
            "A nonlinear model beats the linear baseline out-of-sample here — treat with "
            "the usual skepticism (small samples can flatter complexity)." if _beats else
            "Linear is enough — added complexity does NOT beat the ElasticNet baseline "
            "out-of-sample in this dataset. This is not a deployment recommendation.")
        st.caption(mlr.get("verdict", ""))
        st.dataframe(pd.DataFrame(mlr["leaderboard"]).rename(columns={
            "model": "Model", "oos_rank_ic": "OOS rank correlation", "oos_rank_ic_tstat": "Correlation t-stat",
            "oos_long_short_sharpe": "OOS long/short Sharpe", "n_folds": "Folds", "n_oos_predictions": "OOS predictions",
        }), use_container_width=True, hide_index=True)
        st.caption(f"{mlr.get('n_symbols', '?')} names × {mlr.get('n_days', '?')} days · "
                   f"{len(mlr.get('factors_used', []))} features · "
                   f"purged {mlr.get('n_splits', '?')}-fold CV · {mlr.get('horizon', '?')}-day "
                   "forward-return labels. OOS = out-of-sample (the model never saw it).")

        # --- which factors did the linear baseline actually lean on? ----------
        fi = mlr.get("feature_importance") or []
        if fi:
            st.markdown("**What the linear baseline weights** (descriptive, not OOS)")
            fidf = pd.DataFrame(fi).sort_values("coef")
            figi = go.Figure(go.Bar(x=fidf["coef"], y=fidf["feature"], orientation="h"))
            figi.update_layout(height=max(220, 34 * len(fidf)),
                               margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="ElasticNet coefficient (z-scored features)")
            plot(figi)
            st.caption(mlr.get("feature_importance_note", ""))

with t_details:
    st.subheader("Run record")
    st.download_button("Export research result", json.dumps(res, indent=2), "alphaforge-backtest.json", "application/json", icon=":material/download:")
    # cast to str: the scorecard mixes floats with text (e.g. cvar_verdict), which
    # a single Arrow column can't hold — stringifying keeps the debug table honest.
    _scoredf = pd.DataFrame([card]).T.rename(columns={0: "value"})
    st.dataframe(_scoredf.astype(str), use_container_width=True)

footer(DISCLAIMER)
