"""Free constrained-portfolio research workspace; deliberately broker-free."""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.service import (DISCLAIMER, FACTOR_CATALOG, WorkflowError, data_connections_status,
                         run_portfolio_research)
from app.ui import empty_state, footer, page, plot
from research.local_history import LocalResearchHistory
from research.paper import LocalPaperBook
from research.trial_ledger import TrialLedger


API_URL = os.environ.get("ALPHAFORGE_API", "http://127.0.0.1:8000")
page("Portfolio lab", "Constrained weights / Execution stress / Research audit", section="Portfolio lab")


def fmt(value, spec: str = "", fallback: str = "n/a") -> str:
    if value is None:
        return fallback
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def run_request(payload: dict) -> dict:
    try:
        response = httpx.post(f"{API_URL}/portfolio-research", json=payload, timeout=180.0)
    except httpx.ConnectError:
        return run_portfolio_research(payload, ledger=st.session_state["portfolio_ledger"])
    except httpx.RequestError as error:
        raise WorkflowError("The portfolio service did not finish. No prior result is being shown; retry when it is ready.") from error
    if response.status_code == 200:
        return response.json()
    # A running development API may predate this page during a hot UI reload.
    # The local engine shares the same service function, so continue honestly
    # rather than make the new workspace unusable until a server restart.
    if response.status_code in (404, 405):
        return run_portfolio_research(payload, ledger=st.session_state["portfolio_ledger"])
    try:
        detail = response.json().get("detail", f"API error {response.status_code}")
    except ValueError:
        detail = f"Portfolio service unavailable (HTTP {response.status_code})."
    raise WorkflowError(str(detail))


if "portfolio_ledger" not in st.session_state:
    st.session_state["portfolio_ledger"] = TrialLedger()
if "local_history" not in st.session_state:
    st.session_state["local_history"] = LocalResearchHistory()
if "paper_book" not in st.session_state:
    st.session_state["paper_book"] = LocalPaperBook()

with st.expander("Portfolio setup", expanded="portfolio_result" not in st.session_state):
    source, signal, construction = st.columns(3)
    with source:
        st.subheader("Universe")
        bundle = data_connections_status()["licensed_bundle"]
        provider = st.radio(
            "Data source", ["synthetic", "yfinance", "licensed_bundle"], horizontal=True,
            format_func=lambda value: {
                "synthetic": "Synthetic demo", "yfinance": "Yahoo Finance",
                "licensed_bundle": "Licensed data bundle",
            }[value],
        )
        if provider in {"yfinance", "licensed_bundle"}:
            symbols = [item.strip().upper() for item in st.text_input(
                "Companies", "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,XOM,JNJ,PG,KO,WMT,HD,BAC,DIS"
            ).split(",") if item.strip()]
            periods, seed = 1512, 42
        else:
            symbols = []
            periods = st.slider("History (trading days)", 400, 3000, 1512, step=50, key="portfolio_periods")
            seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="portfolio_seed")
        capital = st.number_input("Portfolio capital", min_value=10_000, max_value=1_000_000_000,
                                  value=1_000_000, step=100_000, help="Used only to estimate market participation and hypothetical paper notionals.")
    with signal:
        st.subheader("Signal")
        categories = list(dict.fromkeys(row["category"] for row in FACTOR_CATALOG))
        category = st.selectbox("Factor family", categories, key="portfolio_category")
        entries = [row for row in FACTOR_CATALOG if row["category"] == category]
        labels = [row["label"] for row in entries]
        label = st.selectbox("Factor", labels, key="portfolio_factor")
        entry = entries[labels.index(label)]
        factor = entry["name"]
        if entry["kind"] == "price":
            lookback = st.slider("Lookback (days)", 21, 504, 252, step=21, key="portfolio_lookback")
            skip = st.slider("Skip recent (days)", 0, max(7, lookback - 7), min(21, lookback - 7), step=7, key="portfolio_skip")
        else:
            lookback, skip = 252, 21
            st.caption("Fundamental factors need synthetic data or a filing-dated licensed bundle.")
        if provider == "yfinance" and entry["kind"] != "price":
            st.warning("This factor requires point-in-time fundamentals, which Yahoo Finance does not provide.")
    with construction:
        st.subheader("Portfolio controls")
        rebalance = st.selectbox("Rebalance", ["daily", "weekly", "monthly", "quarterly"], index=2)
        allocation_method = st.selectbox(
            "Weighting rule", ["score_weighted", "equal_weight", "inverse_volatility", "hierarchical_risk_parity"],
            format_func=lambda value: {
                "score_weighted": "Score-weighted",
                "equal_weight": "Equal-weighted sides",
                "inverse_volatility": "Inverse-volatility adjusted",
                "hierarchical_risk_parity": "Hierarchical risk parity",
            }[value],
            help="A construction choice, not a new signal. Risk-aware methods use only prior return history; hierarchical risk parity also accounts for correlation clusters.",
        )
        name_cap = st.slider("Largest company weight", 0.02, 0.25, 0.10, step=0.01,
                             help="Absolute long or short weight in any one company. The book stays underinvested if this cap cannot be met.")
        use_turnover_cap = st.checkbox("Limit turnover", value=False)
        turnover_cap = st.slider("Maximum turnover per trading day", 0.05, 3.0, 0.50, step=0.05) if use_turnover_cap else None
        use_vol_ceiling = st.checkbox("Use volatility ceiling", value=True)
        target_vol = st.slider("Annual volatility ceiling", 0.05, 0.40, 0.15, step=0.01) if use_vol_ceiling else None
        gross = st.select_slider("Gross exposure", options=[0.5, 1.0, 1.5, 2.0], value=1.0)

    with st.expander("Execution stress assumptions", expanded=False):
        exec1, exec2, exec3, exec4 = st.columns(4)
        with exec1:
            cost_bps = st.slider("Base trading cost", 0.0, 50.0, 5.0, step=0.5)
            spread_bps = st.slider("Estimated spread", 0.0, 50.0, 2.0, step=0.5)
        with exec2:
            impact_bps = st.slider("Market impact at 100% ADV", 0.0, 100.0, 12.0, step=1.0)
            max_participation = st.slider("Maximum ADV participation", 0.01, 0.20, 0.05, step=0.01)
        with exec3:
            borrow_bps = st.slider("Annual short borrow cost", 0.0, 1_000.0, 50.0, step=10.0)
            adv_lookback = st.select_slider("ADV lookback", options=[5, 10, 20, 40, 63], value=20)
        with exec4:
            n_trials = st.number_input("Strategy variants tried", min_value=1, max_value=500, value=50, step=1,
                                        help="Include discarded variants. Portfolio-control changes are also recorded in the trial ledger.")
            st.caption("ADV is calculated from prior days only. A trade without volume history stays unfilled rather than being assumed liquid.")
    bundle_missing = provider == "licensed_bundle" and not bundle["ready_for_price_research"]
    fundamentals_missing = (
        provider == "licensed_bundle" and entry["kind"] != "price"
        and not bundle["ready_for_fundamental_research"]
    )
    unsupported = bundle_missing or fundamentals_missing or (provider == "yfinance" and entry["kind"] != "price")
    if bundle_missing:
        st.warning("No usable licensed data bundle is configured. Review Data connections before running this study.")
    elif fundamentals_missing:
        st.warning("This bundle does not yet attest to filing-dated fundamentals, so fundamental factors remain blocked.")
    elif provider == "licensed_bundle":
        st.caption(f"Connected licensed source: {bundle['provider_name'] or 'local bundle'}")
    run = st.button("Run portfolio research", type="primary", icon=":material/play_arrow:", disabled=unsupported)

if run:
    payload = {
        "provider": provider, "symbols": symbols, "factor": factor, "lookback": lookback, "skip": skip,
        "cost_bps": cost_bps, "gross": gross, "n_trials": int(n_trials), "periods": periods, "seed": int(seed),
        "start": "2015-01-02", "rebalance_frequency": rebalance, "allocation_method": allocation_method, "max_name_weight": name_cap,
        "max_turnover": turnover_cap, "target_annual_vol": target_vol, "spread_bps": spread_bps,
        "impact_bps": impact_bps, "short_borrow_bps": borrow_bps, "capital": float(capital),
        "max_participation": max_participation, "adv_lookback": adv_lookback,
    }
    st.session_state.pop("portfolio_result", None)
    try:
        with st.spinner("Constructing the constrained portfolio and replaying its historical execution..."):
            st.session_state["portfolio_result"] = run_request(payload)
        st.session_state["local_history"].record("portfolio_research", f"{factor} / {provider} / {rebalance}", st.session_state["portfolio_result"])
        st.rerun()
    except WorkflowError as error:
        st.error(f"Portfolio setup needs attention: {error}", icon=":material/error:")
    except Exception as error:
        st.error(f"Portfolio research could not finish: {type(error).__name__}. Retry after checking the data source.", icon=":material/error:")

result = st.session_state.get("portfolio_result")
if not result:
    empty_state("No portfolio study yet", "Run a constrained historical study to review risk, execution pressure, and audit evidence.")
    footer(DISCLAIMER)
    st.stop()

meta, scorecard = result["meta"], result["scorecard"]
readiness = result["data_readiness"]
if meta["provider"] == "synthetic":
    st.warning("Synthetic-data demonstration. It validates the portfolio engine, not a real investment thesis.", icon=":material/science:")
else:
    st.warning(readiness["note"], icon=":material/warning:")
st.info(result["verdict"], icon=":material/info:")
st.caption(f"{meta['n_symbols']} companies / {meta['n_days']} trading days / {meta['portfolio_controls']['rebalance_frequency']} rebalance / historical window {meta['start_date']} to {meta['end_date']}")

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Annual return", fmt(scorecard.get("cagr"), ".2%"))
top2.metric("Sharpe ratio", fmt(scorecard.get("ann_sharpe"), ".2f"))
top3.metric("Largest drawdown", fmt(scorecard.get("max_drawdown"), ".2%"))
top4.metric("Expected shortfall (99%)", fmt(scorecard.get("cvar_99"), ".2%"), help="Historical average loss in the worst 1% of periods.")
top5.metric("OOS confidence", fmt(result["out_of_sample"].get("out_sample_deflated_sr"), ".3f"), help="Deflated Sharpe on the final chronological 30%, not a probability of profit.")

portfolio_tab, execution_tab, validation_tab, audit_tab = st.tabs(["Portfolio", "Execution", "Validation", "Audit & paper"])
with portfolio_tab:
    diag = result["portfolio"]
    one, two, three, four = st.columns(4)
    one.metric("Average gross", fmt(diag.get("avg_gross"), ".1%"))
    two.metric("Largest company", fmt(diag.get("max_name_weight"), ".1%"))
    three.metric("Average active companies", fmt(diag.get("avg_active_names"), ".1f"))
    four.metric("Average net", fmt(diag.get("avg_net"), ".2%"))
    st.caption(f"Construction: {meta['portfolio_controls'].get('allocation_method', 'score_weighted').replace('_', ' ')}. This changes sizing, not the factor signal.")
    if meta["portfolio_controls"].get("allocation_method") == "hierarchical_risk_parity":
        st.info("Historical covariance is shrinkage-stabilized and clustered within each side of the book before the normal gross, neutrality, and name-cap controls are applied.", icon=":material/account_tree:")
    chart_left, chart_right = st.columns(2)
    equity = pd.DataFrame(result["equity_curve"])
    drawdown = pd.DataFrame(result["drawdown_curve"])
    with chart_left:
        figure = go.Figure(go.Scatter(x=equity.get("date"), y=equity.get("value"), name="Net equity", line={"color": "#087f70"}))
        figure.update_layout(title="Equity after modeled execution costs")
        plot(figure, height=300)
    with chart_right:
        figure = go.Figure(go.Scatter(x=drawdown.get("date"), y=drawdown.get("value"), name="Drawdown", fill="tozeroy", line={"color": "#c64b60"}))
        figure.update_layout(title="Drawdown")
        plot(figure, height=300)
    holdings = pd.DataFrame(diag.get("latest_holdings", [])).rename(columns={"symbol": "Company", "weight": "Historical final weight"})
    st.subheader("Final historical holdings")
    if holdings.empty:
        st.info("No positions were formed under the selected constraints.")
    else:
        st.dataframe(holdings, use_container_width=True, hide_index=True,
                     column_config={"Historical final weight": st.column_config.NumberColumn(format="%.2%")})
        st.caption("These are historical end-of-study target weights, not current recommendations.")
    risk_contribution = pd.DataFrame(diag.get("latest_risk_contribution", []))
    if not risk_contribution.empty:
        st.subheader("Latest historical risk contribution")
        st.dataframe(risk_contribution, use_container_width=True, hide_index=True,
                     column_config={"risk_contribution": st.column_config.NumberColumn("Risk contribution", format="%.1%")})

with execution_tab:
    execution = result.get("execution") or {}
    one, two, three, four = st.columns(4)
    one.metric("Total modeled cost", fmt(execution.get("total_cost"), ".2%"), help="Fraction of starting capital across the historical study.")
    two.metric("Liquidity constraints", execution.get("liquidity_breaches", 0))
    three.metric("Unfilled turnover", fmt(execution.get("unfilled_turnover"), ".2f"))
    four.metric("Largest ADV participation", fmt(execution.get("max_participation"), ".2%"))
    if execution.get("unknown_liquidity_trades"):
        st.warning(f"{execution['unknown_liquidity_trades']} trades had unavailable volume history. They were not treated as evidence of liquidity.", icon=":material/warning:")
    st.caption("Costs include base trading cost, estimated spread, square-root impact, short borrow, and partial fills above the configured participation limit. This remains a model, not a fill guarantee.")

with validation_tab:
    robustness = result["robustness"]
    (st.success if "all chronological" in robustness["verdict"] else st.warning)(robustness["verdict"], icon=":material/info:")
    bootstrap = robustness["bootstrap"]
    if bootstrap.get("available"):
        one, two, three, four = st.columns(4)
        one.metric("Bootstrap 5th percentile return", fmt(bootstrap["annual_return_p05"], ".2%"))
        two.metric("Bootstrap median return", fmt(bootstrap["annual_return_median"], ".2%"))
        three.metric("Positive-return resamples", fmt(bootstrap["probability_positive_annual_return"], ".1%"))
        four.metric("Bootstrap 5th percentile Sharpe", fmt(bootstrap["sharpe_p05"], ".2f"))
        st.caption(f"{bootstrap['samples']} moving-block resamples of {bootstrap['block_size']} trading days preserve short-run return dependence. They do not prove future performance.")
    else:
        st.info(bootstrap["reason"])
    periods = pd.DataFrame(robustness["subperiods"])
    if not periods.empty:
        st.subheader("Chronological stability")
        st.dataframe(periods, use_container_width=True, hide_index=True, column_config={
            "annual_return": st.column_config.NumberColumn("Annual return", format="%.2%"),
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("Largest drawdown", format="%.2%"),
        })
    oos = result["out_of_sample"]
    st.caption(f"Holdout: in-sample Sharpe {oos['in_sample_sharpe']:.2f}; final chronological 30% Sharpe {oos['out_sample_sharpe']:.2f}. {oos['method']}.")

with audit_tab:
    manifest = result["manifest"]
    st.subheader("Reproducible research record")
    st.code(manifest["research_fingerprint"], language=None)
    st.caption("This fingerprint binds the request, full input panel, realised positions, scorecard, and engine version. It is stable when the same research run is repeated.")
    down1, down2, save = st.columns([1, 1, 2])
    down1.download_button("Export portfolio audit", json.dumps(result, indent=2), "alphaforge-portfolio-audit.json", "application/json", icon=":material/download:")
    down2.download_button("Export paper plan", json.dumps(result["paper_plan"], indent=2), "alphaforge-paper-rebalance.json", "application/json", icon=":material/download:")
    if save.button("Record local paper rebalance", icon=":material/bookmark_add:"):
        entry = st.session_state["paper_book"].record(result["paper_plan"])
        st.success(f"Recorded a local historical paper plan for {entry['as_of']}. No broker was contacted.")
    plans = st.session_state["paper_book"].list(limit=5)
    if plans:
        st.subheader("Recent local paper records")
        st.dataframe(pd.DataFrame([{
            "Recorded": entry["recorded_at"], "Historical as of": entry["as_of"], "Capital": entry["capital"],
            "Gross exposure": entry["gross_exposure"], "Companies": len(entry["orders"]),
        } for entry in plans]), use_container_width=True, hide_index=True,
                     column_config={"Capital": st.column_config.NumberColumn(format="$%,.0f"), "Gross exposure": st.column_config.NumberColumn(format="%.1%")})

footer(DISCLAIMER)
