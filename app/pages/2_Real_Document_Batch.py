"""Source-traceable earnings releases and filing research."""
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

from api.service import DISCLAIMER, WorkflowError, run_filing_event_study, run_real_document_batch
from app.ui import empty_state, footer, page, sentiment_metrics, source_label
from research.local_history import LocalResearchHistory

API_URL = os.environ.get("ALPHAFORGE_API", "http://127.0.0.1:8000")
page("Filing research", "SEC documents / FinBERT / Filing-date provenance", section="Filing research")


def _run(payload: dict) -> dict:
    try:
        response = httpx.post(f"{API_URL}/real-document-batch", json=payload, timeout=300.0)
    except httpx.ConnectError:
        return run_real_document_batch(payload)
    except httpx.RequestError as error:
        raise WorkflowError("The request did not finish. No previous result is being shown; retry when the connection is ready.") from error
    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", f"API error {response.status_code}")
    except ValueError:
        detail = f"Service unavailable (HTTP {response.status_code}). Try again shortly."
    raise WorkflowError(str(detail))


def _run_event_study(payload: dict) -> dict:
    try:
        response = httpx.post(f"{API_URL}/filing-event-study", json=payload, timeout=180.0)
    except httpx.ConnectError:
        return run_filing_event_study(payload)
    except httpx.RequestError as error:
        raise WorkflowError("The market-data request did not finish. Retry when the connection is ready.") from error
    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", f"API error {response.status_code}")
    except ValueError:
        detail = f"Market-data service unavailable (HTTP {response.status_code})."
    raise WorkflowError(str(detail))


if "local_history" not in st.session_state:
    st.session_state["local_history"] = LocalResearchHistory()


with st.expander("Batch setup", expanded="real_document_batch" not in st.session_state):
    with st.form("filing_batch_setup", border=False):
        left, middle, right = st.columns([2, 1, 1])
        with left:
            symbols_text = st.text_input("Companies", "AAPL, MSFT, NVDA, JPM, XOM", help="Up to 25 SEC-listed tickers, separated by commas. Use a larger sample for source validation.")
            forms = st.multiselect("Filing types", ["8-K", "10-Q", "10-K"], default=["8-K"],
                                   help="8-K: company events, including earnings attachments. 10-Q: quarterly report. 10-K: annual report.")
        with middle:
            as_of = st.date_input("Filed through", value=dt.date.today() - dt.timedelta(days=1))
        with right:
            per_symbol = st.select_slider("Filings per company", options=[1, 2], value=1)
        run = st.form_submit_button("Analyze filings", type="primary", icon=":material/play_arrow:")

if run:
    st.session_state.pop("real_document_batch", None)
    st.session_state.pop("filing_event_study", None)
    payload = {"symbols": [s.strip().upper() for s in symbols_text.split(",") if s.strip()],
               "forms": forms, "as_of": as_of.isoformat(), "per_symbol": per_symbol}
    try:
        with st.spinner("Reading SEC filings and eligible earnings attachments..."):
            st.session_state["real_document_batch"] = _run(payload)
        st.session_state["local_history"].record(
            "filing_batch", f"{len(payload['symbols'])} companies through {payload['as_of']}",
            st.session_state["real_document_batch"],
        )
        st.rerun()
    except WorkflowError as error:
        st.error(f"Batch needs attention: {error}", icon=":material/error:")
    except Exception as error:
        st.error(f"Batch unavailable: {type(error).__name__}. Check provider access and retry.", icon=":material/error:")

result = st.session_state.get("real_document_batch")
if not result:
    empty_state("No filing batch yet", "No documents analyzed in this session.")
else:
    records = result["features"]
    st.caption(f"FILED THROUGH {str(result.get('as_of', ''))[:10]}  /  RESEARCH SAMPLE")
    warnings = result.get("warnings", [])
    if warnings:
        st.warning(
            f"{len(warnings)} requested filings could not be included. No substitute scores were used; "
            "review the source-quality details before interpreting this sample.",
            icon=":material/warning:",
        )
        with st.expander("Review unavailable filings"):
            st.dataframe(pd.DataFrame({"Reason": warnings}), use_container_width=True, hide_index=True)
    releases = sum(r.get("content_kind") == "earnings_release" for r in records)
    earnings_requested = "8-K" in result.get("forms", [])
    one, two, three, four = st.columns(4)
    one.metric("Documents", len(records))
    two.metric("Earnings releases", releases if earnings_requested else "Not requested",
               help="Eligible earnings attachments are searched only inside selected 8-K filings.")
    mda_excerpts = sum(r.get("content_kind") == "mda_excerpt" for r in records)
    three.metric("MD&A excerpts", mda_excerpts,
                 help="10-Q and 10-K MD&A sections that passed deterministic narrative-quality checks.")
    four.metric("Average tone", f"{sum(r['sentiment'] for r in records) / len(records):+.3f}",
                help="Equal-weight average across documents. Describes text tone, not expected performance.")
    validation = result.get("source_validation", {})
    if validation:
        st.caption(
            f"Source validation: {validation.get('classified_documents', len(records))} of "
            f"{validation.get('requested_documents', len(records))} requested filings were classified "
            f"({validation.get('coverage_rate', 1.0):.0%} coverage). "
            f"{validation.get('earnings_releases', releases)} earnings releases; "
            f"{validation.get('mda_excerpts', mda_excerpts)} MD&A excerpts; "
            f"{validation.get('filing_excerpts', len(records) - releases - mda_excerpts)} explicit filing fallbacks."
        )
    st.info("Text analysis only. These scores are not trading signals or evidence of investment performance.", icon=":material/info:")

    integrity = result.get("time_integrity")
    if not integrity:
        # A result saved before the model-time gate existed. Say so rather than
        # letting its silence read as a pass.
        st.subheader("Model-time integrity")
        st.warning(
            "This batch was produced before AlphaForge recorded model-time integrity, so it carries no "
            "record of whether the scoring model was available when each document was filed. It is not "
            "evidence of a historically deployable result. Re-run the batch to obtain the audit.",
            icon=":material/history:",
        )
    if integrity:
        st.subheader("Model-time integrity")
        st.caption(
            "Document provenance says when the filing was available. This check says whether the scoring "
            "model itself was available then. Scores produced by a model that did not yet exist cannot "
            "support a historically deployable study."
        )
        counts = integrity.get("counts", {})
        total_obs = counts.get("total_observations", 0) or 0
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Eligible", counts.get("eligible_observations", 0),
                  help="Filed on or after the scoring model became usable in a live workflow.")
        t2.metric("Unavailable model", counts.get("historically_unavailable_model_observations", 0),
                  help="Filed before the scoring model existed. Excluded from the primary evidence cohort by default.")
        t3.metric("Overlap risk", counts.get("training_overlap_risk_observations", 0),
                  help="Filed on or before the model's training-data cutoff. A risk flag, not proof of memorization.")
        t4.metric("Missing metadata", counts.get("missing_model_metadata_observations", 0),
                  help="No registered model-time metadata. Flagged rather than treated as valid.")
        eligible_share = (integrity.get("coverage", {}) or {}).get("eligible_share", 0.0)
        if total_obs and counts.get("eligible_observations", 0) < total_obs:
            st.warning(
                f"{total_obs - counts.get('eligible_observations', 0)} of {total_obs} documents are not "
                f"model-time eligible ({eligible_share:.0%} eligible). The event study below reports the "
                "eligible cohort as its headline result and keeps the rest in a separate descriptive view.",
                icon=":material/warning:",
            )
        if integrity.get("has_assumptions"):
            st.warning(
                "This audit relies on assumed model metadata, not documented values: "
                + " ".join(integrity.get("assumptions", [])),
                icon=":material/help:",
            )
        metadata = integrity.get("model_metadata", {})
        with st.expander("Model metadata and sources"):
            for field, label in (("model_available_at", "Model available at"),
                                 ("training_data_cutoff", "Training-data cutoff")):
                fact = metadata.get(field) or {}
                badge = {"documented": "Documented", "inferred": "Inferred", "assumed": "ASSUMPTION"}.get(
                    fact.get("status"), str(fact.get("status", "")).title())
                st.markdown(f"**{label}:** {fact.get('value') or 'not established'} — _{badge}_")
                st.caption(fact.get("rationale", ""))
                if fact.get("source"):
                    st.caption(f"Source: {fact['source']}")
            st.caption(f"Metadata source: {metadata.get('metadata_source', 'unknown')}")
            st.caption((integrity.get("policy", {}) or {}).get("description", ""))
        observations = integrity.get("observations", [])
        if observations:
            with st.expander("Per-document standing"):
                st.dataframe(pd.DataFrame([{
                    "Company": row.get("symbol"),
                    "Filed": row.get("available_at"),
                    "Standing": row.get("status_label"),
                    "Overlap risk": bool(row.get("training_overlap_risk")),
                    "In primary cohort": bool(row.get("included_in_primary")),
                    "Reason": row.get("reason"),
                } for row in observations]), use_container_width=True, hide_index=True)
        st.caption(integrity.get("note", ""))

    st.subheader("Document results")
    table = pd.DataFrame([{
        "Company": r["symbol"], "Filed": pd.Timestamp(r["available_at"]).date(),
        "Source": source_label(r), "Tone": r["sentiment"],
        "Positive": r["positive_probability"], "Negative": r["negative_probability"],
        "Neutral": r["neutral_probability"], "SEC source": r["source_url"],
    } for r in records])
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={
        "Tone": st.column_config.NumberColumn(format="%.3f"),
        **{name: st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent") for name in ("Positive", "Negative", "Neutral")},
        "SEC source": st.column_config.LinkColumn(display_text="Open source"),
    })
    export1, export2, _ = st.columns([1, 1, 3])
    export1.download_button("Export scores", table.to_csv(index=False), "alphaforge-filing-scores.csv", "text/csv", icon=":material/download:")
    export2.download_button("Export audit", json.dumps(result, indent=2), "alphaforge-filing-audit.json", "application/json", icon=":material/download:")

    st.subheader("Post-filing event study")
    st.caption("Compare dated document tone with stock returns after the next trading session, versus SPY. This is an explanatory study, not a trading strategy.")
    study_left, study_middle, _ = st.columns([1, 1, 3])
    with study_left:
        horizon = st.select_slider("Return window", options=[1, 3, 5, 10, 21], value=5,
                                   format_func=lambda value: f"{value} trading days")
    with study_middle:
        run_study = st.button("Run event study", icon=":material/analytics:")
    if run_study:
        study_payload = {
            # The model travels with the event so the study can tell whether the
            # scorer existed when each document did.
            "features": [
                {key: record.get(key) for key in ("symbol", "available_at", "document_id", "sentiment", "model")}
                for record in records
            ],
            "horizon": horizon, "benchmark": "SPY",
        }
        try:
            with st.spinner("Loading post-filing market prices and holding out the latest events..."):
                st.session_state["filing_event_study"] = _run_event_study(study_payload)
            st.session_state["local_history"].record(
                "event_study", f"{len(records)} filings / {horizon}-day window", st.session_state["filing_event_study"]
            )
            st.rerun()
        except WorkflowError as error:
            st.error(f"Event study needs attention: {error}", icon=":material/error:")
    study = st.session_state.get("filing_event_study")
    if study:
        for warning in study.get("warnings", []):
            st.warning(warning, icon=":material/warning:")
        coverage = study.get("cohort_coverage") or {}
        if coverage.get("excluded_events"):
            st.warning(
                f"Model-time screen: {coverage['primary_cohort_events']} of {coverage['events_with_prices']} "
                f"priced observations are in the primary cohort "
                f"({coverage.get('primary_cohort_share', 0.0):.0%}). "
                f"{coverage['excluded_events']} were held out because the scoring model was not yet "
                "available or its metadata is missing. A smaller cohort means a weaker test, not a worse result.",
                icon=":material/filter_alt:",
            )
        validity = study.get("research_validity") or {}
        if validity and not validity.get("supports_performance_claim", True):
            st.error(validity.get("conclusion", ""), icon=":material/error:")
        elif validity.get("status") == "limited_variation":
            st.warning(validity.get("conclusion", ""), icon=":material/info:")
        if study.get("status") == "insufficient_data":
            st.info(study["message"] + " Run a larger filing batch; the source-validation limit is 50 documents.", icon=":material/info:")
        else:
            (st.success if study.get("evidence_established") else st.warning)(study["verdict"], icon=":material/verified:" if study.get("evidence_established") else ":material/info:")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Usable events", study["usable_events"])
            e2.metric("Held-out events", study["n_oos"], help="Latest chronological events, after removing boundary overlap.")
            e3.metric("Text vs baseline MSE", f"{study['mse_improvement']:+.5f}", help="Positive means the text model had lower held-out squared error than a no-text average-return baseline.")
            e4.metric("OOS correlation", f"{study['oos_pearson']:+.3f}", help="Correlation between held-out tone and benchmark-adjusted return. Not a probability of profit.")
            event_rows = pd.DataFrame(study.get("events", []))
            if not event_rows.empty:
                event_rows = event_rows.rename(columns={"symbol": "Company", "available_at": "Filed", "entry_date": "Entry", "exit_date": "Exit", "sentiment": "Tone", "abnormal_return": "Benchmark-adjusted return"})
                st.dataframe(event_rows[["Company", "Filed", "Entry", "Exit", "Tone", "Benchmark-adjusted return"]], use_container_width=True, hide_index=True,
                             column_config={"Tone": st.column_config.NumberColumn(format="%.3f"), "Benchmark-adjusted return": st.column_config.NumberColumn(format="%.2f%%")})
            st.caption("The no-text baseline uses the training-period average benchmark-adjusted return. Training events that overlap the held-out period are purged.")
            st.caption(
                "Headline cohort: model-time eligible observations only. Documents whose scoring model "
                "was not yet available are never blended into this conclusion."
            )
            v_returns = validity.get("returns") or {}
            v_signal = validity.get("signal_dispersion") or {}
            if v_returns:
                d1, d2, d3 = st.columns(3)
                d1.metric("Effective observations", v_returns.get("effective_observations", 0),
                          help="Priced, model-time eligible events behind every number above.")
                d2.metric("Non-flat returns", v_returns.get("non_flat_returns", 0),
                          help="Events whose benchmark-adjusted return was not exactly zero.")
                d3.metric("Tone dispersion",
                          f"{v_signal['mean_dispersion']:.3f}" if v_signal.get("available") else "n/a",
                          help="Spread of document tone across the cohort. Near zero means tone cannot separate events.")
                st.caption(validity.get("note", ""))
        all_view = study.get("all_observations_view")
        if all_view and coverage.get("excluded_events"):
            with st.expander("All observations (descriptive only, not evidence)"):
                st.caption(all_view.get("note", ""))
                if all_view.get("status") == "insufficient_data":
                    st.info(all_view.get("message", ""), icon=":material/info:")
                else:
                    d1, d2, d3 = st.columns(3)
                    d1.metric("All observations", all_view.get("usable_events", 0))
                    d2.metric("Text vs baseline MSE", f"{all_view.get('mse_improvement', 0.0):+.5f}")
                    d3.metric("OOS correlation", f"{all_view.get('oos_pearson', 0.0):+.3f}")
                    st.caption(
                        "These numbers describe the full sample including observations the model-time "
                        "screen excluded. They carry no deployability claim and no verdict."
                    )

    st.divider()
    st.subheader("Inspect excerpt")
    selected = st.selectbox("Filing", options=range(len(records)), format_func=lambda i: (
        f"{records[i]['symbol']} | {source_label(records[i])} | {str(records[i]['available_at'])[:10]}"
    ))
    record = records[selected]
    st.caption(f"Filed {pd.Timestamp(record['available_at']).date()} | Filing date only; publication time unavailable.")
    links1, links2, _ = st.columns([1, 1, 2])
    links1.link_button("Open source", record["source_url"], icon=":material/open_in_new:")
    links2.link_button("Parent filing", record.get("filing_url", record["source_url"]), icon=":material/open_in_new:")
    if record.get("content_kind") == "mda_excerpt":
        st.success(record["selection_note"], icon=":material/verified:")
        if record.get("selection_quality_note"):
            score = record.get("selection_quality_score")
            suffix = f" Quality score {score:.1f}." if score is not None else ""
            st.caption(record["selection_quality_note"] + suffix)
    elif record.get("content_kind") != "earnings_release":
        st.warning(record.get("selection_note", "Primary filing excerpt; no earnings attachment selected."), icon=":material/article:")
    else:
        st.caption(record["selection_note"])
    sentiment_metrics(record, aggregate=record.get("passage_count", 1) > 1)
    st.caption(f"Document {record['document_id']} | {record['model']} | {record.get('extraction_version', 'Legacy extraction')}")

    review = record.get("review")
    if not review:
        st.info("Exact excerpt unavailable for this saved result. A new batch is required to capture it.")
    else:
        passages = review.get("passages") or []
        passage = None
        if len(passages) > 1:
            pi = st.radio("Passage", range(len(passages)), horizontal=True,
                          format_func=lambda i: f"{i + 1}. {passages[i]['section']}", key=f"passage_{record['document_id']}")
            passage = passages[pi]
            st.caption(f"Passage tone {passage['sentiment']:+.3f} | Document scores above average all {len(passages)} selected paragraphs equally.")
        analyzed = passage["analyzed_text"] if passage else review["analyzed_text"]
        original = passage["retrieved_text"] if passage else review["retrieved_excerpt"]
        shortened = len(analyzed) < len(original)
        status = "Shortened for FinBERT" if shortened else "No further shortening for FinBERT"
        st.caption(f"{status} | {len(analyzed):,} of {len(original):,} passage characters analyzed.")
        source_chars, start = review.get("source_text_chars"), review.get("excerpt_start_char")
        section_end = review.get("section_end_char")
        selection = review.get("selection") or {}
        if source_chars is not None and start is not None and section_end is not None:
            st.caption(
                f"{selection.get('label', 'Selected section')} spans characters {start + 1:,}-{section_end:,} "
                f"of {source_chars:,} in the filing's extracted text."
            )
        if source_chars is not None and start is not None:
            end = start + review["input_char_count"]
            st.caption(f"Reviewed excerpt | Characters {start + 1:,}-{end:,} of {source_chars:,} in the filing's extracted text.")
        elif source_chars is not None:
            st.caption(f"Selected narrative sample from {source_chars:,} extracted characters. Not a full-document classification.")
        st.caption("Exact text sent to FinBERT")
        st.code(analyzed, language=None, wrap_lines=True, height=280)
        if shortened:
            with st.expander("Retrieved passage before shortening"):
                st.code(original, language=None, wrap_lines=True, height=280)

footer(DISCLAIMER)
