"""Backtest workflow service — ONE source of truth for the browser workflow.

Both the FastAPI app (api/main.py) and the Streamlit page (app/Home.py, in
direct-import fallback mode) call ``run_backtest_workflow``. Keeping the
orchestration here means the API and the UI can never drift apart, and the
honesty guards are applied identically no matter how the engine is reached.

This layer adds NOTHING statistical — it only validates inputs, calls research/,
and serialises results. All honesty guarantees live in research/ and its tests.
"""
from __future__ import annotations

import os
from typing import Any
from dataclasses import replace

import numpy as np
import pandas as pd

from research import (data, event_study, factors, factor_lib, fundamentals, signal_quality,
                      stats_guards, overfitting, metrics, portfolio, validation, reproducibility, paper,
                      statistical_rigor, evidence, robustness_matrix, factor_diagnostics, benchmark_suite)
from research import licensed_data
from research import model_time_integrity, research_validity
from research.macro import FredAlfredProvider
from research.providers import SecEdgarProvider
from research.sec_documents import SUPPORTED_FORMS, SecDocumentProvider
from research.text_features import FinBertExtractor, TextDocument
from research.backtest import backtest
from research.execution import ExecutionModel
from research.walkforward import split_backtest, walk_forward
from research.trial_ledger import TrialLedger
from research.attribution import compact_attribution
from research.strategy_report import build_strategy_report
from research.microstructure import MicrostructureError, run_microstructure_study

DISCLAIMER = (
    "AlphaForge is research software, not investment advice. Backtests are "
    "hypothetical, ignore some real-world frictions, and past performance does "
    "not predict future results. Synthetic-data results validate the engine, "
    "never an edge."
)

# --- Factor catalog: the selectable strategies, grouped for the Strategy Lab UI.
# kind="price" uses the lookback/skip price window; kind="fundamental" uses
# point-in-time filings (lookback/skip ignored). Each entry carries a plain-English
# blurb so the UI can explain it. This is the single source of truth shared by the
# API validator and the UI dropdowns.
FACTOR_CATALOG = [
    # Technical / price
    {"name": "momentum", "label": "Momentum (12-1)", "category": "Technical (price)", "kind": "price",
     "blurb": "Buy recent winners, short recent losers — skipping the last month."},
    {"name": "reversal", "label": "Short-term reversal", "category": "Technical (price)", "kind": "price",
     "blurb": "Bet that recent short-term moves reverse."},
    {"name": "lowvol", "label": "Low volatility", "category": "Technical (price)", "kind": "price",
     "blurb": "Prefer calm, stable stocks over jumpy ones."},
    {"name": "blend", "label": "Blend (momentum + low-vol + reversal)", "category": "Technical (price)", "kind": "price",
     "blurb": "Equal-weight mix of the three price factors."},
    # Value
    {"name": "value", "label": "Value (composite)", "category": "Value", "kind": "fundamental",
     "blurb": "Cheapness composite across earnings, book and cash-flow yields."},
    {"name": "earnings_yield", "label": "Earnings yield (E/P)", "category": "Value", "kind": "fundamental",
     "blurb": "Earnings ÷ price — higher means cheaper."},
    {"name": "book_to_price", "label": "Book-to-price (B/P)", "category": "Value", "kind": "fundamental",
     "blurb": "Book equity ÷ price — the classic value characteristic."},
    {"name": "fcf_yield", "label": "Free-cash-flow yield", "category": "Value", "kind": "fundamental",
     "blurb": "Free cash flow ÷ price — cash-based cheapness."},
    # Quality
    {"name": "quality", "label": "Quality (composite)", "category": "Quality", "kind": "fundamental",
     "blurb": "Profitability/health composite (the planted edge in the synthetic demo)."},
    {"name": "gross_profitability", "label": "Gross profitability", "category": "Quality", "kind": "fundamental",
     "blurb": "Gross profit ÷ assets — Novy-Marx's robust quality signal."},
    {"name": "operating_profitability", "label": "Operating profitability", "category": "Quality", "kind": "fundamental",
     "blurb": "Fama-French operating profitability over book equity."},
    {"name": "roe", "label": "Return on equity (ROE)", "category": "Quality", "kind": "fundamental",
     "blurb": "Net income ÷ shareholder equity."},
    {"name": "roa", "label": "Return on assets (ROA)", "category": "Quality", "kind": "fundamental",
     "blurb": "Net income ÷ total assets."},
    {"name": "earnings_quality", "label": "Earnings quality (low accruals)", "category": "Quality", "kind": "fundamental",
     "blurb": "Low Sloan accruals — earnings backed by cash, not estimates."},
    {"name": "low_leverage", "label": "Low leverage", "category": "Quality", "kind": "fundamental",
     "blurb": "Less debt ÷ assets — lower balance-sheet risk."},
    # Investment
    {"name": "conservative_investment", "label": "Conservative investment", "category": "Investment", "kind": "fundamental",
     "blurb": "Low asset growth — aggressive expanders tend to underperform."},
    {"name": "low_issuance", "label": "Low share issuance", "category": "Investment", "kind": "fundamental",
     "blurb": "Less new stock issued — persistent diluters underperform."},
    # Composite alpha (multi-signal recipes that combine orthogonal mechanisms)
    {"name": "value_with_fraud_guardrail", "label": "Value (fraud-guarded)", "category": "Composite alpha", "kind": "fundamental",
     "blurb": "Cheap on B/P and EBIT/EV, penalized by Beneish-M manipulation and Ohlson-O distress."},
    {"name": "profitable_value", "label": "Profitable value", "category": "Composite alpha", "kind": "fundamental",
     "blurb": "Scores high only when a name is BOTH very cheap and very profitable (a z×z AND-gate)."},
    {"name": "conservative_compounder", "label": "Conservative compounder", "category": "Composite alpha", "kind": "fundamental",
     "blurb": "High gross profitability, low asset growth, low realized volatility — quality with discipline."},
    # Multi-factor
    {"name": "value_quality", "label": "Value + Quality", "category": "Multi-factor", "kind": "fundamental",
     "blurb": "Cheap AND healthy — value and quality combined."},
]

FACTORS = tuple(f["name"] for f in FACTOR_CATALOG)
PRICE_FACTORS = tuple(f["name"] for f in FACTOR_CATALOG if f["kind"] == "price")
LEGACY_FUNDAMENTAL = ("value", "quality", "value_quality")  # composites via fundamentals.py
# factor_lib single-factor fundamentals: name -> (factor_lib function, sign).
# sign=-1 flips "lower is better" signals (leverage/accruals/asset-growth/issuance)
# so the cross-sectional rank points the economically correct way.
FACTORLIB_SPECS = {
    "earnings_yield": ("earnings_yield", +1),
    "book_to_price": ("book_to_price", +1),
    "fcf_yield": ("fcf_yield", +1),
    "gross_profitability": ("gross_profitability", +1),
    "operating_profitability": ("operating_profitability", +1),
    "roe": ("roe", +1),
    "roa": ("roa", +1),
    "earnings_quality": ("sloan_accruals", -1),
    "low_leverage": ("leverage", -1),
    "conservative_investment": ("asset_growth", -1),
    "low_issuance": ("net_equity_issuance", -1),
    # composites already return a combined score panel; +1 (higher is better by construction)
    "value_with_fraud_guardrail": ("value_with_fraud_guardrail", +1),
    "profitable_value": ("profitable_value", +1),
    "conservative_compounder": ("conservative_compounder", +1),
}
FUNDAMENTAL_FACTORS = LEGACY_FUNDAMENTAL + tuple(FACTORLIB_SPECS.keys())
PROVIDERS = ("synthetic", "yfinance", "licensed_bundle")

# hard input bounds — the service refuses absurd requests instead of hanging
MAX_SYMBOLS = 50
MIN_PERIODS, MAX_PERIODS = 120, 6000
MAX_TICKER_LEN = 10


class WorkflowError(ValueError):
    """User-facing input/validation error (maps to HTTP 400)."""


def run_microstructure_lab(req: dict) -> dict:
    """Run the fixed first market-microstructure hypothesis.

    This accepts a bounded, normalized event payload rather than a file path so
    the API cannot read arbitrary local files.  Large raw archives stay local and
    are processed through the Streamlit upload workflow.
    """
    source = str(req.get("source", "synthetic")).strip().lower()
    if source not in {"synthetic", "binance_events"}:
        raise WorkflowError("source must be synthetic or binance_events")
    try:
        seed = int(req.get("seed", 42))
        window = int(req.get("window_events", 32))
        horizon = int(req.get("horizon_events", 30))
        fee = float(req.get("maker_fee_bps", 1.0))
        latency = int(req.get("latency_events", 2))
        inventory = int(req.get("max_inventory", 8))
    except (TypeError, ValueError) as error:
        raise WorkflowError("microstructure settings must be numeric") from error
    if not (0 <= seed <= 2_147_483_647):
        raise WorkflowError("seed must be a non-negative 32-bit integer")
    if not (8 <= window <= 500 and 5 <= horizon <= 500 and horizon < window * 10):
        raise WorkflowError("window_events must be 8-500 and horizon_events must be 5-500")
    if not (0.0 <= fee <= 100.0 and 0 <= latency <= 100 and 1 <= inventory <= 100):
        raise WorkflowError("fee, latency, or inventory setting is outside the supported research bounds")
    records = req.get("events")
    if source == "binance_events":
        if not isinstance(records, list) or not (120 <= len(records) <= 20_000):
            raise WorkflowError("binance_events needs between 120 and 20,000 normalized trade events")
        events = pd.DataFrame(records)
    elif records:
        raise WorkflowError("events are accepted only with source=binance_events")
    else:
        events = None
    try:
        return run_microstructure_study(
            events, source=source, seed=seed, window=window, horizon_events=horizon,
            maker_fee_bps=fee, latency_events=latency, max_inventory=inventory,
        )
    except MicrostructureError as error:
        raise WorkflowError(str(error)) from error


def _validate(req: dict) -> dict:
    provider = str(req.get("provider", "synthetic")).lower()
    if provider not in PROVIDERS:
        raise WorkflowError(f"provider must be one of {PROVIDERS}")

    factor = str(req.get("factor", "momentum")).lower()
    if factor not in FACTORS:
        raise WorkflowError(f"factor must be one of {FACTORS}")

    symbols = req.get("symbols") or []
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if provider in {"yfinance", "licensed_bundle"}:
        if len(symbols) < 2:
            raise WorkflowError(f"{provider} provider needs at least 2 ticker symbols "
                                "(this is a cross-sectional long/short strategy)")
        bad = [s for s in symbols if len(s) > MAX_TICKER_LEN or not s.replace(".", "").replace("-", "").isalnum()]
        if bad:
            raise WorkflowError(f"invalid ticker(s): {bad}")
    if len(symbols) > MAX_SYMBOLS:
        raise WorkflowError(f"too many symbols (max {MAX_SYMBOLS})")

    periods = int(req.get("periods", 1512))
    if not (MIN_PERIODS <= periods <= MAX_PERIODS):
        raise WorkflowError(f"periods must be in [{MIN_PERIODS}, {MAX_PERIODS}]")

    lookback = int(req.get("lookback", 252))
    skip = int(req.get("skip", 21))
    if not (5 <= lookback <= 756):
        raise WorkflowError("lookback must be in [5, 756]")
    if not (0 <= skip < lookback):
        raise WorkflowError("skip must satisfy 0 <= skip < lookback")

    cost_bps = float(req.get("cost_bps", 5.0))
    if not (0.0 <= cost_bps <= 200.0):
        raise WorkflowError("cost_bps must be in [0, 200]")

    n_trials = int(req.get("n_trials", 1))
    if not (1 <= n_trials <= 100_000):
        raise WorkflowError("n_trials must be in [1, 100000]")

    gross = float(req.get("gross", 1.0))
    if not (0.1 <= gross <= 3.0):
        raise WorkflowError("gross must be in [0.1, 3.0]")

    seed = int(req.get("seed", 42))
    if seed < 0:
        raise WorkflowError("seed must be >= 0")

    start = str(req.get("start", "2015-01-02"))
    try:
        start_at = pd.Timestamp(start).normalize()
    except Exception:
        raise WorkflowError("start must be a valid date, e.g. 2015-01-02")

    end_value = req.get("end")
    if end_value in (None, ""):
        end = None
    else:
        try:
            end_at = pd.Timestamp(end_value).normalize()
        except Exception as error:
            raise WorkflowError("end must be a valid date, e.g. 2024-12-31") from error
        if end_at <= start_at:
            raise WorkflowError("end must be after start")
        end = end_at.date().isoformat()

    evaluation_value = req.get("evaluation_start")
    if evaluation_value in (None, ""):
        evaluation_start = None
    else:
        try:
            evaluation_at = pd.Timestamp(evaluation_value).normalize()
        except Exception as error:
            raise WorkflowError("evaluation_start must be a valid date, e.g. 2024-01-01") from error
        if evaluation_at < start_at or (end is not None and evaluation_at > pd.Timestamp(end)):
            raise WorkflowError("evaluation_start must fall within the loaded date window")
        evaluation_start = evaluation_at.date().isoformat()

    return {
        "provider": provider, "factor": factor, "symbols": symbols,
        "periods": periods, "seed": seed, "start": start_at.date().isoformat(), "end": end,
        "evaluation_start": evaluation_start,
        "lookback": lookback, "skip": skip,
        "cost_bps": cost_bps, "n_trials": n_trials, "gross": gross,
    }


def _effective_lookback(factor: str, lookback: int) -> int:
    """The lookback actually used — some factors cap it (reported honestly in meta)."""
    if factor == "reversal":
        return min(lookback, 63)
    if factor == "lowvol":
        return min(lookback, 252)
    return lookback  # momentum (and the momentum leg of blend) uses it as-is


def _score(close: pd.DataFrame, factor: str, lookback: int, skip: int) -> pd.DataFrame:
    z = factors.cross_sectional_zscore
    if factor == "momentum":
        return z(factors.momentum(close, lookback=lookback, skip=skip))
    if factor == "reversal":
        return z(factors.short_term_reversal(close, lookback=min(lookback, 63)))
    if factor == "lowvol":
        return z(-factors.trailing_volatility(close, lookback=min(lookback, 252)))
    # blend: equal-weight momentum + low-vol + reversal
    return factors.blend(
        z(factors.momentum(close, lookback=lookback, skip=skip)),
        z(-factors.trailing_volatility(close, lookback=min(lookback, 252))),
        z(factors.short_term_reversal(close, lookback=21)),
    )


def _compute_pbo(close: pd.DataFrame, p: dict) -> tuple[dict, pd.DataFrame | None]:
    """Probability of Backtest Overfitting via CSCV over a small lookback grid —
    the configs a user would realistically sweep. Technical factors only in this
    MVP (the fundamental scoring path differs). Always returns a dict; on any
    failure (short series, degenerate grid) it reports an honest n/a, never raises.
    """
    if p["factor"] in FUNDAMENTAL_FACTORS:
        return {"error": "PBO grid not wired for fundamental factors yet"}, None
    lb, sk, factor, gross = p["lookback"], p["skip"], p["factor"], p["gross"]

    # Build the grid in EFFECTIVE-lookback space. _score caps the lookback inside
    # the factor (reversal -> 63, lowvol -> 252), so a raw-lookback grid would map
    # several points to the SAME effective window -> identical return columns ->
    # a degenerate CSCV that manufactures a false "overfit" verdict. Centre the
    # sweep on the effective lookback and respect the factor's cap so every config
    # is genuinely distinct.
    base = _effective_lookback(factor, lb)
    cap = {"reversal": 63, "lowvol": 252}.get(factor, 756)
    grid = sorted({min(cap, max(5, int(round(base * f))))
                   for f in (0.4, 0.6, 0.8, 1.0, 1.3)})
    param_grid = [{"lookback": g} for g in grid]
    if len(param_grid) < 2:
        return {"error": "PBO grid degenerate for this factor (fewer than 2 distinct configs)"}, None

    def weight_fn(c, lookback):
        return factors.long_short_weights(
            _score(c, factor, lookback, min(sk, lookback - 1)), gross=gross)

    try:
        matrix = overfitting.candidate_returns_from_factor_grid(
            close, weight_fn, param_grid, cost_bps=p["cost_bps"])
        cpcv = statistical_rigor.purged_cpcv_pbo(
            matrix, n_splits=8, purge_periods=1, embargo_periods=1)
        counts, edges = np.histogram(cpcv.logits, bins=20) if cpcv.logits else (np.array([]), np.array([]))
        pbo = {
            "pbo": cpcv.pbo,
            "n_strategies": cpcv.n_strategies,
            "n_combinations": cpcv.n_combinations,
            "median_oos_sharpe_selected": cpcv.median_oos_sharpe_selected,
            "histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
            "insufficient": cpcv.insufficient,
            "overfit": bool(np.isfinite(cpcv.pbo) and cpcv.pbo > 0.5),
            "verdict": (
                "insufficient data — purged CPCV PBO not estimable" if cpcv.insufficient else
                "OVERFIT: the in-sample winner is more likely than not to be below-median out-of-sample" if cpcv.pbo > 0.5 else
                "elevated overfitting risk — treat the selected configuration with caution" if cpcv.pbo > 0.25 else
                "low overfitting risk for this parameter search"
            ),
            "note": cpcv.note,
            "purge_periods": cpcv.purge_periods,
            "embargo_periods": cpcv.embargo_periods,
        }
        return pbo, matrix
    except Exception as e:  # noqa: BLE001 - honest n/a beats a crashed workflow
        return {"error": f"{type(e).__name__}: {e}"}, None


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain JSON-safe Python."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    return obj


PILOT_DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")
PILOT_DEFAULT_MACRO_SERIES = ("CPIAUCSL", "UNRATE", "DGS10")
PILOT_MAX_SYMBOLS = 5
PILOT_MAX_MACRO_SERIES = 5
PILOT_MAX_DOCUMENT_CHARS = 12_000
DOCUMENT_BATCH_MAX_PER_SYMBOL = 2
DOCUMENT_BATCH_MAX_SYMBOLS = 25
DOCUMENT_BATCH_MAX_DOCUMENTS = 50


def _pilot_symbols(values: list[str] | None) -> list[str]:
    symbols = [str(value).strip().upper() for value in (values or PILOT_DEFAULT_SYMBOLS) if str(value).strip()]
    if not (1 <= len(symbols) <= PILOT_MAX_SYMBOLS):
        raise WorkflowError(f"public-data pilot accepts 1 to {PILOT_MAX_SYMBOLS} tickers")
    if len(set(symbols)) != len(symbols):
        raise WorkflowError("public-data pilot tickers must be unique")
    bad = [symbol for symbol in symbols if len(symbol) > MAX_TICKER_LEN or not symbol.replace(".", "").replace("-", "").isalnum()]
    if bad:
        raise WorkflowError(f"invalid ticker(s): {bad}")
    return symbols


def _document_symbols(values: list[str] | None) -> list[str]:
    """Validate the broader, bounded validation sample separately from the pilot."""
    symbols = [str(value).strip().upper() for value in (values or PILOT_DEFAULT_SYMBOLS) if str(value).strip()]
    if not (1 <= len(symbols) <= DOCUMENT_BATCH_MAX_SYMBOLS):
        raise WorkflowError(f"filing research accepts 1 to {DOCUMENT_BATCH_MAX_SYMBOLS} tickers")
    if len(set(symbols)) != len(symbols):
        raise WorkflowError("filing research tickers must be unique")
    bad = [symbol for symbol in symbols if len(symbol) > MAX_TICKER_LEN or not symbol.replace(".", "").replace("-", "").isalnum()]
    if bad:
        raise WorkflowError(f"invalid ticker(s): {bad}")
    return symbols


def _latest_known_macro(observations: pd.DataFrame, as_of: pd.Timestamp) -> dict | None:
    known = observations[pd.to_datetime(observations["available_date"]) <= as_of].copy()
    if known.empty:
        return None
    known["observation_date"] = pd.to_datetime(known["observation_date"])
    latest_period = known["observation_date"].max()
    latest = known[known["observation_date"] == latest_period].sort_values("available_date")
    row = latest.iloc[-1]
    return {
        "value": float(row["value"]),
        "observation_date": pd.Timestamp(row["observation_date"]),
        "available_date": pd.Timestamp(row["available_date"]),
    }


def run_real_document_batch(
    req: dict,
    *,
    document_provider: Any | None = None,
    text_extractor: Any | None = None,
) -> dict:
    """Classify a small, timestamped batch of real SEC filing excerpts.

    This is a provenance step only. It retrieves documents filed by a declared
    cutoff, preserves each filing date, and does not create a trading result.
    """
    symbols = _document_symbols(req.get("symbols"))
    forms = {str(value).strip().upper() for value in req.get("forms", ["8-K"]) if str(value).strip()}
    if not forms or not forms.issubset(SUPPORTED_FORMS):
        raise WorkflowError(f"document batch forms must be selected from {sorted(SUPPORTED_FORMS)}")
    try:
        as_of = pd.Timestamp(req.get("as_of", pd.Timestamp.now().date()))
    except Exception as error:
        raise WorkflowError("available-through date must be a valid ISO date") from error
    per_symbol = int(req.get("per_symbol", 1))
    if not 1 <= per_symbol <= DOCUMENT_BATCH_MAX_PER_SYMBOL:
        raise WorkflowError(f"document batch accepts 1 to {DOCUMENT_BATCH_MAX_PER_SYMBOL} filings per ticker")
    if len(symbols) * per_symbol > DOCUMENT_BATCH_MAX_DOCUMENTS:
        raise WorkflowError(f"filing research is limited to {DOCUMENT_BATCH_MAX_DOCUMENTS} documents per run")

    provider = document_provider or SecDocumentProvider()
    documents, warnings = provider.documents(symbols, forms, as_of, per_symbol)
    if not documents:
        raise WorkflowError("SEC returned no usable filing excerpts for the selected cutoff and forms")

    extractor = text_extractor or FinBertExtractor()
    features = []
    for document in documents:
        text_document = document.as_text_document()
        selected = document.passages
        try:
            scored = [extractor.extract(replace(text_document, text=passage.text)) for passage in selected] if selected else [extractor.extract(text_document)]
        except Exception as error:
            warnings.append(f"{document.symbol}: classification unavailable for {document.accession_number} ({type(error).__name__}); no score substituted.")
            continue
        feature = scored[0]
        # Use the captured model input, never reconstruct it from today's clipping rule.
        analyzed_text = "\n\n".join(item.analyzed_text for item in scored)
        input_chars = sum(item.input_char_count or 0 for item in scored)
        review = None
        if analyzed_text and feature.input_char_count is not None:
            review = {
                "analyzed_text": analyzed_text,
                "retrieved_excerpt": document.text,
                "input_char_count": input_chars,
                "analyzed_char_count": sum(len(item.analyzed_text) for item in scored),
                "model_input_shortened": any(len(item.analyzed_text) < (item.input_char_count or 0) for item in scored),
                "source_text_chars": document.source_text_chars,
                "excerpt_start_char": document.excerpt_start_char,
                "section_end_char": document.section_end_char,
                "selection": {
                    "method": document.selection_method,
                    "label": document.selection_label,
                    "quality_score": document.selection_quality_score,
                    "quality_note": document.selection_quality_note,
                },
                "passages": [{
                    "section": selected[index].section if selected else document.selection_label,
                    "analyzed_text": item.analyzed_text,
                    "retrieved_text": selected[index].text if selected else document.text,
                    "input_char_count": item.input_char_count,
                    "sentiment": item.sentiment,
                    "positive_probability": item.positive_probability,
                    "negative_probability": item.negative_probability,
                    "neutral_probability": item.neutral_probability,
                } for index, item in enumerate(scored)],
            }
        features.append({
            "symbol": feature.symbol,
            "available_at": feature.available_at,
            "form": document.form,
            "document_id": feature.document_id,
            "source_url": document.url,
            "filing_url": document.filing_url or document.url,
            "content_kind": document.content_kind,
            "selection_note": document.selection_note,
            "selection_method": document.selection_method,
            "selection_label": document.selection_label,
            "selection_quality_score": document.selection_quality_score,
            "selection_quality_note": document.selection_quality_note,
            "extraction_version": (
                document.selection_method if document.content_kind == "mda_excerpt"
                else document.extraction_version
            ),
            "passage_count": len(scored),
            "model": feature.model,
            "sentiment": float(np.mean([item.sentiment for item in scored])),
            "positive_probability": float(np.mean([item.positive_probability for item in scored])),
            "negative_probability": float(np.mean([item.negative_probability for item in scored])),
            "neutral_probability": float(np.mean([item.neutral_probability for item in scored])),
            "review": review,
        })
    if not features:
        raise WorkflowError("No documents could be classified. Check Hugging Face access and retry. " + " ".join(warnings))
    # Was the scoring model itself usable when each document appeared?  Document
    # provenance alone does not answer that question.
    policy = model_time_integrity.TimeIntegrityPolicy.from_request(req.get("time_integrity"))
    time_integrity = model_time_integrity.time_integrity_audit(features, policy=policy)
    by_document = {row["document_id"]: row for row in time_integrity["observations"]}
    for item in features:
        verdict = by_document.get(item["document_id"])
        if verdict:
            item["time_integrity"] = {
                key: verdict[key] for key in
                ("status", "status_label", "training_overlap_risk", "reason", "included_in_primary")
            }
    source_validation = {
        "requested_tickers": len(symbols),
        "requested_documents": len(symbols) * per_symbol,
        "classified_documents": len(features),
        "earnings_releases": sum(item["content_kind"] == "earnings_release" for item in features),
        "mda_excerpts": sum(item["content_kind"] == "mda_excerpt" for item in features),
        "filing_excerpts": sum(item["content_kind"] == "filing_excerpt" for item in features),
        "warnings": len(warnings),
        "coverage_rate": len(features) / (len(symbols) * per_symbol),
    }
    return _jsonable({
        "symbols": symbols,
        "as_of": as_of,
        "forms": sorted(forms),
        "filings_requested": per_symbol,
        "features": features,
        "warnings": warnings,
        "source_validation": source_validation,
        "time_integrity": time_integrity,
        "note": "SEC source documents classified with filing-date provenance; not a trading result.",
    })


def _event_price_panel(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, benchmark: str) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch adjusted closes only for the bounded filing-study window."""
    wanted = list(dict.fromkeys([*symbols, benchmark]))
    try:
        panel = data.get_panel(wanted, start=str(start.date()), end=str(end.date()), provider="yfinance")
    except ImportError as error:
        raise WorkflowError("Event studies need market-price support. Install it with `pip install alphaforge[data]`.") from error
    close = panel.close
    if benchmark not in close:
        raise WorkflowError(f"Benchmark {benchmark} did not return adjusted close prices.")
    return close.drop(columns=[benchmark], errors="ignore"), close[benchmark]


def run_filing_event_study(req: dict, *, price_loader: Any | None = None) -> dict:
    """Evaluate dated text features against post-filing abnormal returns.

    The result is an explanatory event study with a chronological holdout, not a
    portfolio backtest. Filing dates have no publication time, so every event enters
    on the next available trading session.
    """
    rows = req.get("features") or []
    if not (1 <= len(rows) <= DOCUMENT_BATCH_MAX_DOCUMENTS):
        raise WorkflowError(f"event study accepts 1 to {DOCUMENT_BATCH_MAX_DOCUMENTS} scored documents")
    try:
        horizon = int(req.get("horizon", 5))
    except (TypeError, ValueError) as error:
        raise WorkflowError("event-study horizon must be a whole number of trading days") from error
    if not 1 <= horizon <= 21:
        raise WorkflowError("event-study horizon must be in [1, 21] trading days")
    benchmark = str(req.get("benchmark", "SPY")).upper().strip()
    if not benchmark or len(benchmark) > MAX_TICKER_LEN:
        raise WorkflowError("event-study benchmark must be a valid ticker")
    events: list[event_study.FilingEvent] = []
    audit_rows: list[dict] = []
    for row in rows:
        try:
            symbol = str(row["symbol"]).upper().strip()
            available_at = pd.Timestamp(row["available_at"])
            sentiment = float(row["sentiment"])
            document_id = str(row["document_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise WorkflowError("each event-study document needs symbol, available_at, sentiment, and document_id") from error
        if not symbol or not document_id or not np.isfinite(sentiment) or pd.isna(available_at):
            raise WorkflowError("event-study documents contain invalid provenance or sentiment")
        events.append(event_study.FilingEvent(symbol, available_at, sentiment, document_id))
        # A legacy record with no model field resolves to missing metadata; it is
        # never silently treated as if the model had been available.
        audit_rows.append({
            "symbol": symbol, "available_at": available_at, "document_id": document_id,
            "model": str(row.get("model") or "").strip(),
        })
    policy = model_time_integrity.TimeIntegrityPolicy.from_request(req.get("time_integrity"))
    time_integrity = model_time_integrity.time_integrity_audit(audit_rows, policy=policy)
    primary_ids = {row["document_id"] for row in time_integrity["observations"] if row["included_in_primary"]}

    first = min(event.available_at for event in events).normalize() - pd.Timedelta(days=10)
    last = max(event.available_at for event in events).normalize() + pd.Timedelta(days=horizon + 35)
    loader = price_loader or _event_price_panel
    close, market = loader(sorted({event.symbol for event in events}), first, last, benchmark)
    observed, warnings = event_study.build_event_returns(events, close, market, horizon=horizon)
    eligible_observed = (
        observed[observed["document_id"].isin(primary_ids)].reset_index(drop=True)
        if not observed.empty else observed
    )

    if eligible_observed.empty and not observed.empty:
        counts = time_integrity["counts"]
        causes = [
            (counts["historically_unavailable_model_observations"],
             "predate the scoring model's availability date"),
            (counts["missing_model_metadata_observations"], "carry no model metadata"),
            (counts["training_overlap_risk_observations"] if policy.exclude_training_overlap else 0,
             "carry training-data overlap risk under the selected policy"),
        ]
        reasons = [f"{count} {label}" for count, label in causes if count]
        outcome = {
            "status": "insufficient_data",
            "message": (
                "No observation survived the model-time integrity screen, so there is no primary "
                "evidence cohort: " + "; ".join(reasons) + "."
            ),
            "n_events": 0,
            "horizon": horizon,
        }
    else:
        outcome = event_study.evaluate_sentiment(eligible_observed, horizon=horizon)

    # A descriptive all-observations view, kept strictly separate so the headline
    # verdict is never a blend of deployable and non-deployable observations.
    all_view = event_study.evaluate_sentiment(observed, horizon=horizon) if not observed.empty else {
        "status": "insufficient_data", "message": "No usable events.", "n_events": 0, "horizon": horizon,
    }
    all_view.pop("verdict", None)
    all_view.pop("evidence_established", None)
    all_view.update({
        "usable_events": int(len(observed)),
        "label": "All observations (descriptive only)",
        "note": (
            "Descriptive view over every scored document, including observations the model-time "
            "screen excluded. It is not evidence of a historically deployable result and must not be "
            "reported as the study's conclusion."
        ),
    })

    # Data health for the cohort that produces the headline, so a degenerate
    # sample cannot read as a normal completed study.
    validity = research_validity.research_validity_summary(
        eligible_observed["abnormal_return"] if not eligible_observed.empty else [],
        signal=eligible_observed["sentiment"] if not eligible_observed.empty else None,
    )

    excluded = int(len(observed) - len(eligible_observed))
    outcome.update({
        "research_validity": validity,
        "benchmark": benchmark,
        "requested_documents": len(events),
        "usable_events": int(len(eligible_observed)),
        "cohort": "model-time eligible",
        "all_observations_view": all_view,
        "time_integrity": time_integrity,
        "cohort_coverage": {
            "events_with_prices": int(len(observed)),
            "primary_cohort_events": int(len(eligible_observed)),
            "excluded_events": excluded,
            "primary_cohort_share": (len(eligible_observed) / len(observed)) if len(observed) else 0.0,
        },
        "warnings": warnings,
        "note": (
            "Date-only filing availability enters on the next trading session. This is an explanatory "
            "event study, not a trading result. The headline result covers only observations whose "
            f"scoring model was already available; {excluded} of {len(observed)} priced observations "
            "were held out of it."
        ),
    })
    return _jsonable(outcome)


def run_public_data_pilot(
    req: dict,
    *,
    sec_provider: Any | None = None,
    macro_provider: Any | None = None,
    text_extractor: Any | None = None,
) -> dict:
    """Run a small, fixed-scope data-provenance check.

    This intentionally does not build a signal, optimize parameters, or submit a
    backtest.  It proves that SEC filing dates, ALFRED vintages, and optional text
    classifications carry enough metadata for a later honest experiment.
    """
    symbols = _pilot_symbols(req.get("symbols"))
    macro_series = [str(value).strip().upper() for value in req.get("macro_series", PILOT_DEFAULT_MACRO_SERIES) if str(value).strip()]
    if not (1 <= len(macro_series) <= PILOT_MAX_MACRO_SERIES):
        raise WorkflowError(f"public-data pilot accepts 1 to {PILOT_MAX_MACRO_SERIES} macro series")
    if len(set(macro_series)) != len(macro_series):
        raise WorkflowError("public-data pilot macro series must be unique")
    try:
        macro_start = pd.Timestamp(req.get("macro_start", "2015-01-01"))
        as_of = pd.Timestamp(req.get("as_of", pd.Timestamp.now().date()))
    except Exception as error:
        raise WorkflowError("macro dates must be valid ISO dates") from error
    if macro_start > as_of:
        raise WorkflowError("macro start must be on or before the as-of date")

    text_doc = None
    document = req.get("text_document")
    if document:
        text = str(document.get("text", ""))
        if not text.strip():
            raise WorkflowError("text document cannot be empty")
        if len(text) > PILOT_MAX_DOCUMENT_CHARS:
            raise WorkflowError(f"text document exceeds {PILOT_MAX_DOCUMENT_CHARS:,} characters")
        try:
            text_doc = TextDocument(
                symbol=str(document.get("symbol", "")).strip().upper(),
                available_at=pd.Timestamp(document.get("available_at")),
                source=str(document.get("source", "")).strip(),
                document_id=str(document.get("document_id", "")).strip(),
                text=text,
            )
        except (TypeError, ValueError) as error:
            raise WorkflowError(f"invalid text document metadata: {error}") from error
        if pd.Timestamp(text_doc.available_at).date() > as_of.date():
            raise WorkflowError(
                "document available date must be on or before the pilot's available-through date"
            )

    sec = sec_provider or SecEdgarProvider()
    macro = macro_provider or FredAlfredProvider()
    sec_obs = sec.fundamentals(symbols)
    sec_summary = []
    warnings = []
    for symbol in symbols:
        rows = sec_obs[sec_obs["symbol"] == symbol]
        first_available = pd.Timestamp(rows["available_date"].min()) if not rows.empty else None
        latest_available = pd.Timestamp(rows["available_date"].max()) if not rows.empty else None
        if rows.empty:
            warnings.append(
                f"{symbol}: SEC returned no supported 10-K/10-Q company facts. "
                "Do not use this ticker in a fundamental experiment until its coverage is resolved."
            )
        elif first_available > macro_start:
            warnings.append(
                f"{symbol}: SEC coverage begins on {first_available.date()}, after the requested "
                f"history start of {macro_start.date()}. Treat earlier history as unavailable."
            )
        sec_summary.append({
            "symbol": symbol,
            "observations": int(len(rows)),
            "metrics": int(rows["metric"].nunique()) if not rows.empty else 0,
            "first_available": first_available,
            "latest_available": latest_available,
        })

    macro_summary = []
    for series_id in macro_series:
        try:
            observations = macro.observations(series_id, start=str(macro_start.date()), end=str(as_of.date()))
        except RuntimeError as error:
            raise WorkflowError(str(error)) from error
        if observations.empty:
            raise WorkflowError(
                f"{series_id}: FRED returned no point-in-time observations for the requested date range."
            )
        latest = _latest_known_macro(observations, as_of)
        if latest is None:
            raise WorkflowError(
                f"{series_id}: FRED returned no value known by the requested as-of date of {as_of.date()}."
            )
        macro_summary.append({
            "series_id": series_id,
            "vintages": int(len(observations)),
            "latest_value": None if latest is None else latest["value"],
            "observation_date": None if latest is None else latest["observation_date"],
            "available_date": None if latest is None else latest["available_date"],
        })

    text_result = None
    if text_doc:
        feature = (text_extractor or FinBertExtractor()).extract(text_doc)
        # The pilot certifies nothing, but a score still has to carry the
        # provenance of the model that produced it.
        profile = model_time_integrity.resolve_profile(feature.model)
        verdict = model_time_integrity.classify_observation(feature.available_at, profile)
        text_result = {
            "symbol": feature.symbol,
            "available_at": feature.available_at,
            "source": feature.source,
            "document_id": feature.document_id,
            "model": feature.model,
            "sentiment": feature.sentiment,
            "positive_probability": feature.positive_probability,
            "negative_probability": feature.negative_probability,
            "neutral_probability": feature.neutral_probability,
            "model_time_integrity": {
                "status": verdict.status,
                "status_label": model_time_integrity.STATUS_LABELS.get(verdict.status, verdict.status),
                "training_overlap_risk": verdict.training_overlap_risk,
                "reason": verdict.reason,
                "model_metadata": profile.to_dict(),
                "note": (
                    "Recorded for provenance only. The pilot validates metadata and makes no "
                    "deployability claim either way, so no observation is admitted or excluded here."
                ),
            },
        }

    return _jsonable({
        "symbols": symbols,
        "as_of": as_of,
        "sec": sec_summary,
        "macro": macro_summary,
        "warnings": warnings,
        "text_feature": text_result,
        "note": "Pilot only: provenance and availability validation, not a trading result.",
    })


def _downsample_curve(s: pd.Series, max_points: int = 500) -> list[dict]:
    """Downsample a curve for charting WITHOUT hiding extrema.

    Plain stride-slicing can skip the true peak/trough, so a drawdown chart could
    show a shallower trough than the max-drawdown number printed beside it. We
    always keep the global min and max (and the first/last point), so the chart
    can never understate the worst drawdown — which for an honesty tool matters.
    """
    if s.empty:
        return []
    stride = max(1, len(s) // max_points)
    keep = set(range(0, len(s), stride))
    keep.update({0, len(s) - 1, int(np.argmin(s.values)), int(np.argmax(s.values))})
    sub = s.iloc[sorted(keep)]
    return [{"date": str(getattr(idx, "date", lambda: idx)()), "value": round(float(v), 6)}
            for idx, v in sub.items()]


def _prepare_strategy_inputs(p: dict) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Any]:
    """Load one point-in-time-safe panel and its factor score.

    Both Strategy Lab and Portfolio Lab use this exact preparation path.  That
    prevents a portfolio feature from quietly using different data or factor
    maths than the backtest it is meant to extend.
    """
    fund_for_attr = None
    if p["factor"] in FUNDAMENTAL_FACTORS:
        if p["provider"] == "synthetic":
            syms = p["symbols"] or [f"S{i:02d}" for i in range(20)]
            world = data.make_synthetic_world(syms, start=p["start"], periods=p["periods"],
                                              seed=p["seed"], quality_to_drift=0.0016)
            close, volume = world.close, world.volume
            if p["end"] is not None:
                cutoff = pd.Timestamp(p["end"])
                close, volume = close.loc[close.index <= cutoff], volume.loc[volume.index <= cutoff]
            if len(close) < 400:
                raise WorkflowError("fundamental factors need >= 400 days (several quarters + OOS)")
            if p["factor"] in LEGACY_FUNDAMENTAL:
                obs = data.make_synthetic_fundamentals(world, seed=p["seed"])
                fund = fundamentals.build_fundamentals(obs, close.index, list(close.columns))
                if p["factor"] == "value":
                    score = fundamentals.value_score(fund, close)
                elif p["factor"] == "quality":
                    score = fundamentals.quality_score(fund, close)
                else:
                    score = fundamentals.value_quality_score(fund, close)
                fund_for_attr = fundamentals.build_fundamentals(
                    data.make_synthetic_raw_fundamentals(world, seed=p["seed"]),
                    close.index, list(close.columns))
            else:
                obs = data.make_synthetic_raw_fundamentals(world, seed=p["seed"])
                fund = fundamentals.build_fundamentals(obs, close.index, list(close.columns))
                fn_name, sign = FACTORLIB_SPECS[p["factor"]]
                raw = getattr(factor_lib, fn_name)(fund, close)
                score = factors.cross_sectional_zscore(raw if sign > 0 else -raw)
                fund_for_attr = fund
        elif p["provider"] == "licensed_bundle":
            try:
                panel = data.get_panel(p["symbols"], start=p["start"], end=p["end"], provider="licensed_bundle")
                close = panel.close.dropna(how="all", axis=1)
                volume = panel.volume.reindex_like(close)
                if close.shape[1] < 2:
                    raise WorkflowError("need at least 2 eligible symbols with licensed price history")
                if len(close) < 400:
                    raise WorkflowError("fundamental factors need >= 400 days (several quarters + OOS)")
                obs = licensed_data.load_fundamentals(list(close.columns), start=p["start"], end=p["end"])
            except licensed_data.LicensedDataError as error:
                raise WorkflowError(str(error)) from error
            fund = fundamentals.build_fundamentals(obs, close.index, list(close.columns))
            if p["factor"] == "value":
                raw = factor_lib.value_score(fund, close)
            elif p["factor"] == "quality":
                raw = factor_lib.quality_score(fund, close)
            elif p["factor"] == "value_quality":
                raw = factors.blend(factor_lib.value_score(fund, close), factor_lib.quality_score(fund, close))
            else:
                fn_name, sign = FACTORLIB_SPECS[p["factor"]]
                raw = getattr(factor_lib, fn_name)(fund, close)
                raw = raw if sign > 0 else -raw
            score = factors.cross_sectional_zscore(raw)
            if not score.notna().any().any():
                raise WorkflowError("Licensed fundamentals do not cover the fields required by this factor.")
            fund_for_attr = fund
        else:
            raise WorkflowError(
                "Fundamental factors require either synthetic data or a licensed bundle with filing-dated fundamentals. "
                "Yahoo Finance does not provide the needed point-in-time history."
            )
    else:
        if p["provider"] == "synthetic":
            panel = data.get_panel(
                p["symbols"] or [f"S{i:02d}" for i in range(15)], provider="synthetic",
                start=p["start"], end=p["end"], periods=p["periods"], seed=p["seed"],
            )
        else:
            try:
                panel = data.get_panel(p["symbols"], start=p["start"], end=p["end"], provider=p["provider"])
            except licensed_data.LicensedDataError as error:
                raise WorkflowError(str(error)) from error
        close = panel.close.dropna(how="all", axis=1)
        volume = panel.volume.reindex_like(close)
        if close.shape[1] < 2:
            raise WorkflowError("need at least 2 symbols with data for a cross-sectional strategy")
        if len(close) < MIN_PERIODS:
            raise WorkflowError(f"only {len(close)} usable days of data (need >= {MIN_PERIODS})")
        if len(close) <= p["lookback"] + p["skip"] + 21:
            raise WorkflowError("lookback too long for the data window")
        score = _score(close, p["factor"], p["lookback"], p["skip"])
    return close, volume, score, fund_for_attr


def run_backtest_workflow(req: dict, ledger: "TrialLedger | None" = None) -> dict:
    """Validate -> load data -> factor -> honest backtest -> full verdict dict.

    ``ledger`` (optional) is a per-workspace TrialLedger: when supplied, the engine
    counts DISTINCT strategy configs and enforces the multiple-testing haircut
    FLOOR on the full-sample Deflated Sharpe (declared n_trials can be raised, never
    used to under-deflate). No ledger => stateless, declared n_trials used as-is.
    """
    p = _validate(req)
    close, _volume, score, fund_for_attr = _prepare_strategy_inputs(p)

    weights = factors.long_short_weights(score, gross=p["gross"])
    full_res = backtest(close, weights, cost_bps=p["cost_bps"])
    evaluation_start = pd.Timestamp(p["evaluation_start"]) if p["evaluation_start"] else close.index[0]
    evaluation_mask = close.index >= evaluation_start
    evaluation_close = close.loc[evaluation_mask]
    evaluation_score = score.loc[evaluation_mask]
    evaluation_weights = weights.loc[evaluation_mask]
    if evaluation_close.empty:
        raise WorkflowError("No usable trading sessions fall inside the requested evaluation window.")

    # Build positions across the complete source window so stage one opens with
    # legitimate signal history.  Every reported statistic is then calculated
    # only from the requested evaluation range, normalized to a fresh equity base.
    stage_returns = full_res.returns.loc[evaluation_mask]
    res = replace(
        full_res,
        equity=(1.0 + stage_returns).cumprod(),
        returns=stage_returns,
        gross_returns=full_res.gross_returns.loc[evaluation_mask],
        positions=full_res.positions.loc[evaluation_mask],
        costs=None if full_res.costs is None else full_res.costs.loc[evaluation_mask],
        turnover=None if full_res.turnover is None else full_res.turnover.loc[evaluation_mask],
    )

    # Trial ledger: count DISTINCT configs this workspace has run and ENFORCE the
    # multiple-testing haircut FLOOR on the full-sample Deflated Sharpe. The declared
    # n_trials can only be raised by the engine, never used to under-deflate.
    declared_trials = p["n_trials"]
    if ledger is not None:
        trial_identity = req.get("_trial_identity")
        rec = ledger.record(trial_identity if isinstance(trial_identity, dict) else req)
        eff_trials = ledger.effective_n_trials(declared_trials)
    else:
        rec, eff_trials = None, declared_trials
    trial_audit = {
        "declared_n_trials": declared_trials,
        "effective_n_trials": eff_trials,
        "haircut_was_raised": bool(eff_trials > declared_trials),
    }
    if ledger is not None:
        trial_audit.update(rec or {})
        trial_audit.update(ledger.snapshot())

    sigq = signal_quality.compact_scorecard(evaluation_score, evaluation_close)  # is the SIGNAL itself predictive?
    scorecard = res.summary(n_trials=eff_trials)
    oos = split_backtest(evaluation_close, evaluation_weights, split=0.7, cost_bps=p["cost_bps"])

    # Walk-forward with purge + embargo at fold boundaries: a factor's lookback can
    # otherwise leak prior-fold prices into the first OOS bars of each fold. Purge =
    # the bars of history the signal needs (lookback + skip); fundamental factors
    # have no rolling price lookback (purge 0, small embargo).
    purge = (0 if p["factor"] in FUNDAMENTAL_FACTORS
             else _effective_lookback(p["factor"], p["lookback"]) + p["skip"])
    try:
        wf = walk_forward(evaluation_close, evaluation_weights, n_splits=5, cost_bps=p["cost_bps"],
                          purge_bars=int(purge), embargo_bars=1)
    except ValueError as e:
        wf = {"error": str(e)}

    if p["evaluation_start"] and pd.Timestamp(p["evaluation_start"]) > pd.Timestamp(p["start"]):
        pbo, candidate_returns = ({
            "error": "PBO is a search diagnostic and is not recomputed inside a fixed validation or final-holdout stage."
        }, None)
    else:
        pbo, candidate_returns = _compute_pbo(close, p)  # CSCV + CPCV search diagnostics
    tails = stats_guards.fat_tail_report(res.returns)
    # Stress the unchanged, point-in-time signal. These are deliberately ordinary
    # frictions (worse costs and slower fills), not another optimized search.
    stress = robustness_matrix.cost_delay_matrix(evaluation_close, evaluation_weights, base_cost_bps=p["cost_bps"])
    factor_health = factor_diagnostics.factor_health(evaluation_score, evaluation_close)
    bundle_status = licensed_data.bundle_status() if p["provider"] == "licensed_bundle" else None
    data_audit = evidence.data_integrity_audit(p["provider"], evaluation_close, p["symbols"], bundle_status)

    try:
        ic_values = signal_quality.ic_series(evaluation_score, evaluation_close)
        benchmark_returns = evaluation_close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
        strategy_report = build_strategy_report(
            res.returns, positions=res.positions, benchmark_returns=benchmark_returns,
            candidate_returns=candidate_returns, ic_values=ic_values, signal=evaluation_score, metadata={
                "factor": p["factor"], "provider": p["provider"],
                "benchmark_label": "Equal-weight selected universe (benchmark proxy, not SPY)",
            },
        )
        if candidate_returns is not None:
            strategy_report["inputs"]["has_candidate_matrix"] = True
            strategy_report["statistical_rigor"]["cpcv_pbo"] = pbo
            strategy_report["statistical_rigor"]["white_reality_check"] = {
                "computed": False,
                "note": "White's Reality Check is available for this parameter grid in the deep report.",
            }
    except Exception as e:  # report diagnostics must not block a valid backtest
        strategy_report = {"error": f"{type(e).__name__}: {e}"}

    # Factor-model attribution: how much of the strategy's return is just FF/Carhart
    # factor beta vs. genuine idiosyncratic alpha. Full FF5 when synthetic fundamentals
    # are available; price-only (MKT + UMD) otherwise.
    attr_model = "ff5" if fund_for_attr is not None else "carhart4"
    try:
        attribution = compact_attribution(res.returns, evaluation_close, fund=fund_for_attr, model=attr_model)
    except Exception as e:  # noqa: BLE001 - attribution is a read-out, never fail the run
        attribution = {"error": f"{type(e).__name__}: {e}"}

    # Headline verdict is OUT-OF-SAMPLE first. A strategy that only passes the
    # full-sample Deflated Sharpe but fails walk-forward / decays out-of-sample is
    # NOT credible — anything else would let the tool flatter a data-mined fit.
    def _ok(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    full_dsr = scorecard.get("deflated_sr")
    if "error" not in wf:
        credible = bool(wf.get("passes")) and not oos.get("overfit_warning", False)
        basis = "out-of-sample walk-forward Deflated Sharpe"
    else:  # series too short to walk-forward — fall back to full-sample, and say so
        credible = (_ok(full_dsr) and full_dsr > 0.95) and not oos.get("overfit_warning", False)
        basis = "full-sample Deflated Sharpe (series too short for walk-forward)"
    verdict = (
        f"CREDIBLE: survives the {basis} bar"
        if credible
        else f"NOT CREDIBLE: does not survive the {basis} bar"
    )
    decision_evidence = evidence.evidence_card(
        data_audit=data_audit, scorecard=scorecard, signal_quality=sigq,
        out_of_sample=oos, walk_forward=wf, pbo=pbo, robustness=stress,
    )

    dd = (res.equity / res.equity.cummax() - 1.0)
    meta = {
        **p,
        "n_trials": eff_trials,
        "declared_n_trials": declared_trials,
        "effective_lookback": (None if p["factor"] in FUNDAMENTAL_FACTORS else _effective_lookback(p["factor"], p["lookback"])),
        "n_symbols": int(evaluation_close.shape[1]), "n_days": int(len(evaluation_close)),
        "start_date": str(evaluation_close.index[0].date()), "end_date": str(evaluation_close.index[-1].date()),
        "source_start_date": str(close.index[0].date()),
        "evaluation_window_applied": bool(p["evaluation_start"]),
        "engine_version": "0.0.1", "disclaimer": DISCLAIMER,
    }
    manifest = reproducibility.build_manifest(
        request=req, meta=meta, close=evaluation_close, positions=res.positions, scorecard=scorecard,
        engine_version=meta["engine_version"],
    )
    return _jsonable({
        "meta": meta,
        "verdict": verdict,
        "verdict_basis": basis,
        "scorecard": scorecard,
        "signal_quality": sigq,
        "factor_health": factor_health,
        "data_audit": data_audit,
        "robustness_matrix": stress,
        "evidence_card": decision_evidence,
        "manifest": manifest,
        "out_of_sample": oos,
        "walk_forward": wf,
        "pbo": pbo,
        "strategy_report": strategy_report,
        "temporal_stability": strategy_report.get("temporal_stability") if isinstance(strategy_report, dict) else None,
        "trial_audit": trial_audit,
        "attribution": attribution,
        "fat_tails": tails,
        "equity_curve": _downsample_curve(res.equity),
        "drawdown_curve": _downsample_curve(dd),
    })


def run_benchmark_suite(req: dict, ledger: "TrialLedger | None" = None) -> dict:
    """Evaluate fixed reference factors under the exact same study assumptions.

    The factor list is intentionally not caller-configurable.  Reference models
    are useful only when they remain reference models, rather than a quiet second
    search surface.  Unsupported inputs are listed as skips, never replaced with
    synthetic scores or a different data source.
    """
    runs, skipped = benchmark_suite.requests_for_suite(req)
    rows = []
    fingerprints = []
    for definition, run_request in runs:
        result = run_backtest_workflow(run_request, ledger=ledger)
        scorecard = result.get("scorecard") or {}
        manifest = result.get("manifest") or {}
        fingerprints.append(manifest.get("research_fingerprint"))
        rows.append({
            "factor": definition["factor"],
            "label": definition["label"],
            "definition": definition["definition"],
            "verdict": result.get("verdict"),
            "evidence_status": (result.get("evidence_card") or {}).get("status"),
            "data_status": (result.get("data_audit") or {}).get("status"),
            "annual_return": scorecard.get("cagr"),
            "annual_sharpe": scorecard.get("ann_sharpe"),
            "deflated_sharpe": scorecard.get("deflated_sr"),
            "max_drawdown": scorecard.get("max_drawdown"),
            "research_fingerprint": manifest.get("research_fingerprint"),
        })
    if not rows:
        raise WorkflowError("No fixed benchmarks are compatible with this data source.")
    base = _validate(req)
    suite_fingerprint = reproducibility.fingerprint({
        "kind": "fixed_benchmark_suite_v1",
        "request": req,
        "components": fingerprints,
    })
    return _jsonable({
        "kind": "fixed_benchmark_suite",
        "definition": "Predefined value, quality, momentum, and low-volatility references. No benchmark parameters are optimized in this workflow.",
        "meta": {
            "provider": base["provider"],
            "start": base["start"],
            "end": base["end"],
            "evaluation_start": base["evaluation_start"],
            "cost_bps": base["cost_bps"],
            "declared_n_trials": max(base["n_trials"], len(benchmark_suite.BENCHMARKS)),
        },
        "benchmarks": rows,
        "skipped": skipped,
        "manifest": {
            "research_fingerprint": suite_fingerprint,
            "component_fingerprints": fingerprints,
            "engine_version": "0.0.1",
        },
    })


def _portfolio_options(req: dict, p: dict) -> dict:
    """Validate the free Portfolio Lab controls without mutating the base request."""
    frequency = str(req.get("rebalance_frequency", "monthly")).lower()
    if frequency not in portfolio.REBALANCE_FREQUENCIES:
        raise WorkflowError(f"rebalance_frequency must be one of {portfolio.REBALANCE_FREQUENCIES}")
    allocation_method = str(req.get("allocation_method", "score_weighted")).lower()
    if allocation_method not in portfolio.ALLOCATION_METHODS:
        raise WorkflowError(f"allocation_method must be one of {portfolio.ALLOCATION_METHODS}")
    max_name_weight = float(req.get("max_name_weight", min(0.10, p["gross"])))
    if not (0.01 <= max_name_weight <= p["gross"]):
        raise WorkflowError("max_name_weight must be between 1% and gross exposure")
    supplied_turnover = req.get("max_turnover")
    max_turnover = None if supplied_turnover in (None, 0, 0.0) else float(supplied_turnover)
    if max_turnover is not None and not (0.01 <= max_turnover <= 3.0):
        raise WorkflowError("max_turnover must be in [0.01, 3.0] when supplied")
    supplied_vol = req.get("target_annual_vol")
    target_vol = None if supplied_vol in (None, 0, 0.0) else float(supplied_vol)
    if target_vol is not None and not (0.05 <= target_vol <= 1.0):
        raise WorkflowError("target_annual_vol must be in [0.05, 1.0] when supplied")
    options = {
        "rebalance_frequency": frequency, "allocation_method": allocation_method, "max_name_weight": max_name_weight,
        "max_turnover": max_turnover, "target_annual_vol": target_vol,
        "spread_bps": float(req.get("spread_bps", 2.0)), "impact_bps": float(req.get("impact_bps", 12.0)),
        "short_borrow_bps": float(req.get("short_borrow_bps", 50.0)),
        "capital": float(req.get("capital", 1_000_000.0)),
        "max_participation": float(req.get("max_participation", 0.05)),
        "adv_lookback": int(req.get("adv_lookback", 20)),
    }
    try:
        ExecutionModel(base_cost_bps=p["cost_bps"], **{key: options[key] for key in (
            "spread_bps", "impact_bps", "short_borrow_bps", "capital", "max_participation", "adv_lookback"
        )}).validate()
    except ValueError as error:
        raise WorkflowError(str(error)) from error
    return options


def _portfolio_oos(returns: pd.Series, *, n_trials: int) -> dict:
    """Chronological holdout on the realised costed portfolio return stream."""
    cut = int(len(returns) * 0.7)
    insample, oos = returns.iloc[:cut], returns.iloc[cut:]
    is_sr = metrics.annualised_sharpe(insample)
    oos_sr = metrics.annualised_sharpe(oos)
    dsr = metrics.deflated_sharpe_ratio(oos, n_trials=max(1, n_trials))
    return {
        "method": "chronological 70/30 holdout on realised constrained returns",
        "in_sample_sharpe": float(is_sr), "out_sample_sharpe": float(oos_sr),
        "out_sample_deflated_sr": None if np.isnan(dsr) else float(dsr),
        "overfit_warning": bool(is_sr - oos_sr > 1.0),
        "oos_significant": bool(not np.isnan(dsr) and dsr >= 0.95),
    }


def data_connections_status() -> dict:
    """Secret-free connection facts for the UI and future account settings."""
    sec_user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    return {
        "licensed_bundle": licensed_data.bundle_status(),
        "fundamental_adapters": [
            {
                "id": "sec_edgar",
                "label": "SEC EDGAR company facts",
                "configured": bool(sec_user_agent) and "example.com" not in sec_user_agent,
                "point_in_time_fundamentals": True,
                "note": "Public filing-date data; it does not supply a survivorship-free historical universe.",
            },
            {
                "id": "simfin",
                "label": "SimFin",
                "configured": bool(os.environ.get("SIMFIN_API_KEY", "").strip()),
                "point_in_time_fundamentals": False,
                "note": "Budget adapter; latest-restated values remain blocked for point-in-time factor research.",
            },
            {
                "id": "sharadar",
                "label": "Sharadar SF1 / Nasdaq Data Link",
                "configured": bool(os.environ.get("NASDAQ_DATA_LINK_API_KEY", "").strip()),
                "point_in_time_fundamentals": True,
                "note": "As-reported fundamentals adapter. Pair it with price and universe history in a licensed bundle for full studies.",
            },
        ],
    }


def _data_readiness(provider: str) -> dict:
    if provider == "synthetic":
        return {
            "source": "deterministic synthetic market", "point_in_time_fundamentals": True,
            "survivorship_free_universe": False, "live_investible": False,
            "note": "Useful for engine validation only; it cannot establish a tradeable edge.",
        }
    if provider == "yfinance":
        return {
            "source": "Yahoo Finance adjusted historical prices and volume", "point_in_time_fundamentals": False,
            "survivorship_free_universe": False, "live_investible": False,
            "note": "Adjusted prices help with splits/dividends, but current-ticker history can exclude delisted names. Use a point-in-time licensed universe before investment use.",
        }
    status = licensed_data.bundle_status()
    provider_name = status.get("provider_name") or "licensed local bundle"
    has_full_attestation = all((
        status.get("point_in_time_fundamentals"),
        status.get("survivorship_free_universe"),
        status.get("includes_delisted_securities"),
        status.get("prices_adjusted_for_corporate_actions"),
    ))
    return {
        "source": provider_name,
        "point_in_time_fundamentals": bool(status.get("point_in_time_fundamentals")),
        "survivorship_free_universe": bool(status.get("survivorship_free_universe")),
        "live_investible": False,
        "note": (
            "The local bundle declares point-in-time fundamentals, historical membership, delisted coverage, and corporate-action-adjusted prices. Keep the vendor manifest and license evidence with the research audit."
            if has_full_attestation
            else "The local bundle is usable only within the coverage attested in its manifest. Review Data connections before interpreting the study."
        ),
    }


def run_portfolio_research(req: dict, ledger: "TrialLedger | None" = None) -> dict:
    """Run the free, constrained portfolio-research workflow.

    This route intentionally stops before broker integration.  It tests historical
    portfolio mechanics, liquidity stress, and validation, then can save a clearly
    labelled historical paper-rebalance plan for review.
    """
    p = _validate(req)
    options = _portfolio_options(req, p)
    close, volume, score, _fund = _prepare_strategy_inputs(p)
    weights = portfolio.construct_long_short_weights(
        score, close, method=options["allocation_method"], gross=p["gross"],
        max_name_weight=options["max_name_weight"]
    )
    weights = portfolio.apply_rebalance_schedule(weights, options["rebalance_frequency"])
    weights = portfolio.limit_turnover(weights, options["max_turnover"])
    weights = portfolio.apply_volatility_ceiling(
        weights, close, target_annual_vol=options["target_annual_vol"], lookback=63
    )
    model = ExecutionModel(
        base_cost_bps=p["cost_bps"], spread_bps=options["spread_bps"], impact_bps=options["impact_bps"],
        short_borrow_bps=options["short_borrow_bps"], capital=options["capital"],
        max_participation=options["max_participation"], adv_lookback=options["adv_lookback"],
    )
    result = backtest(close, weights, cost_bps=p["cost_bps"], volume=volume, execution_model=model)
    if ledger is not None:
        rec = ledger.record(req)
        effective_trials = ledger.effective_n_trials(p["n_trials"])
    else:
        rec, effective_trials = None, p["n_trials"]
    scorecard = result.summary(n_trials=effective_trials)
    meta = {
        **p, "n_trials": effective_trials, "declared_n_trials": p["n_trials"],
        "n_symbols": int(close.shape[1]), "n_days": int(len(close)),
        "start_date": str(close.index[0].date()), "end_date": str(close.index[-1].date()),
        "engine_version": "0.0.1", "disclaimer": DISCLAIMER, "portfolio_controls": options,
    }
    manifest = reproducibility.build_manifest(
        request=req, meta=meta, close=close, positions=result.positions, scorecard=scorecard,
        engine_version=meta["engine_version"],
    )
    paper_plan = paper.rebalance_plan(result.positions, capital=options["capital"], research_fingerprint=manifest["research_fingerprint"])
    diagnostics = portfolio.portfolio_diagnostics(result.positions, close)
    trial_audit = {
        "declared_n_trials": p["n_trials"], "effective_n_trials": effective_trials,
        "haircut_was_raised": bool(effective_trials > p["n_trials"]),
    }
    if ledger is not None:
        trial_audit.update(rec or {})
        trial_audit.update(ledger.snapshot())
    drawdown = result.equity / result.equity.cummax() - 1.0
    return _jsonable({
        "meta": meta,
        "verdict": "Research-only portfolio simulation. It is not a live allocation or a prediction of investment performance.",
        "scorecard": scorecard, "portfolio": diagnostics, "execution": result.execution_audit,
        "out_of_sample": _portfolio_oos(result.returns, n_trials=effective_trials),
        "robustness": validation.robustness_report(result.returns, seed=p["seed"]),
        "data_readiness": _data_readiness(p["provider"]), "trial_audit": trial_audit,
        "manifest": manifest, "paper_plan": paper_plan,
        "equity_curve": _downsample_curve(result.equity), "drawdown_curve": _downsample_curve(drawdown),
    })


# factors used as ML features (resolved by model_eval against factor_lib)
ML_FEATURE_FACTORS = ("gross_profitability", "roe", "operating_margin",
                      "book_to_price", "asset_growth", "momentum_12_1")

# The full allow-list of factor_lib builders selectable as ML features in the UI/API.
# Every name here must resolve in research.factor_lib (verified by the test suite).
# Restricting to an allow-list means a user can compose the feature panel from the
# factor library without being able to inject an arbitrary attribute name.
ML_FEATURE_CHOICES = (
    "gross_profitability", "operating_profitability", "roe", "roa", "operating_margin",
    "book_to_price", "earnings_yield", "fcf_yield",
    "asset_growth", "net_equity_issuance", "sloan_accruals", "leverage",
    "momentum_12_1", "trailing_volatility",
)
MAX_ML_FEATURES = 12  # bound the feature panel (compute + overfitting discipline)


def run_model_comparison(req: dict, factor_names: tuple | None = None,
                         model_names=None, n_splits: int = 6, horizon: int = 21) -> dict:
    """Opt-in, heavyweight: compare the ML model ladder under leak-aware purged CV.

    Returns the leaderboard + the honest 'does complexity beat the linear baseline
    out-of-sample?' verdict. Synthetic provider only (needs point-in-time fundamentals),
    and the ML extras must be installed (`pip install alphaforge[ml]`). This is NOT run
    on a normal backtest — it is slow by design (the full ladder is ~1-2 min).
    """
    p = _validate(req)
    if p["provider"] != "synthetic":
        raise WorkflowError(
            "model comparison needs the synthetic provider — point-in-time fundamentals "
            "for the feature panel aren't wired for free data yet")
    try:
        from research.model_eval import build_feature_panel, compare_ladder
        from research.models import MODEL_NAMES
    except ImportError:
        raise WorkflowError("ML extras not installed — run `pip install alphaforge[ml]` "
                            "(scikit-learn, xgboost, lightgbm)")

    # Validate user-chosen features/models against allow-lists (reject unknowns up
    # front with a clear message rather than failing deep in the model code).
    if factor_names is not None:
        feats_req = [str(f).strip() for f in factor_names if str(f).strip()]
        bad = [f for f in feats_req if f not in ML_FEATURE_CHOICES]
        if bad:
            raise WorkflowError(f"unknown ML feature(s) {bad}; choose from {ML_FEATURE_CHOICES}")
        if not (1 <= len(feats_req) <= MAX_ML_FEATURES):
            raise WorkflowError(f"choose between 1 and {MAX_ML_FEATURES} ML features")
        factor_names = tuple(feats_req)
    if model_names is not None:
        models_req = [str(m).strip() for m in model_names if str(m).strip()]
        bad = [m for m in models_req if m not in MODEL_NAMES]
        if bad:
            raise WorkflowError(f"unknown model(s) {bad}; choose from {tuple(MODEL_NAMES)}")
        # the baseline must be present — the whole verdict is measured against it
        if "elastic_net" not in models_req:
            models_req = ["elastic_net"] + models_req
        model_names = models_req
    if not (1 <= int(horizon) <= 63):
        raise WorkflowError("horizon must be in [1, 63] trading days")

    syms = p["symbols"] or [f"S{i:02d}" for i in range(20)]
    world = data.make_synthetic_world(syms, start=p["start"], periods=p["periods"],
                                      seed=p["seed"], quality_to_drift=0.0016)
    close = world.close
    if len(close) < 400:
        raise WorkflowError("model comparison needs >= 400 days")
    fund = fundamentals.build_fundamentals(
        data.make_synthetic_raw_fundamentals(world, seed=p["seed"]),
        close.index, list(close.columns))
    feats = list(factor_names or ML_FEATURE_FACTORS)
    X, y, fmeta = build_feature_panel(close, fund, feats, horizon=horizon)
    kw = {"n_splits": n_splits}
    if model_names is not None:
        kw["model_names"] = model_names
    result = compare_ladder(X, y, fmeta["label_start"], fmeta["label_end"], **kw)
    return _jsonable({
        "factors_used": feats,
        "n_symbols": int(close.shape[1]),
        "n_days": int(len(close)),
        "horizon": horizon,
        "n_splits": n_splits,
        "disclaimer": DISCLAIMER,
        **result,
    })
