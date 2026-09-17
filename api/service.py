"""Backtest workflow service — ONE source of truth for the browser workflow.

Both the FastAPI app (api/main.py) and the Streamlit page (app/Home.py, in
direct-import fallback mode) call ``run_backtest_workflow``. Keeping the
orchestration here means the API and the UI can never drift apart, and the
honesty guards are applied identically no matter how the engine is reached.

This layer adds NOTHING statistical — it only validates inputs, calls research/,
and serialises results. All honesty guarantees live in research/ and its tests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research import data, factors, factor_lib, fundamentals, signal_quality, stats_guards, overfitting
from research.macro import FredAlfredProvider
from research.providers import SecEdgarProvider
from research.text_features import FinBertExtractor, TextDocument
from research.backtest import backtest
from research.walkforward import split_backtest, walk_forward
from research.trial_ledger import TrialLedger
from research.attribution import compact_attribution

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
PROVIDERS = ("synthetic", "yfinance")

# hard input bounds — the service refuses absurd requests instead of hanging
MAX_SYMBOLS = 50
MIN_PERIODS, MAX_PERIODS = 120, 6000
MAX_TICKER_LEN = 10


class WorkflowError(ValueError):
    """User-facing input/validation error (maps to HTTP 400)."""


def _validate(req: dict) -> dict:
    provider = str(req.get("provider", "synthetic")).lower()
    if provider not in PROVIDERS:
        raise WorkflowError(f"provider must be one of {PROVIDERS}")

    factor = str(req.get("factor", "momentum")).lower()
    if factor not in FACTORS:
        raise WorkflowError(f"factor must be one of {FACTORS}")

    symbols = req.get("symbols") or []
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if provider == "yfinance":
        if len(symbols) < 2:
            raise WorkflowError("yfinance provider needs at least 2 ticker symbols "
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
        pd.Timestamp(start)
    except Exception:
        raise WorkflowError("start must be a valid date, e.g. 2015-01-02")

    return {
        "provider": provider, "factor": factor, "symbols": symbols,
        "periods": periods, "seed": seed, "start": start,
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


def _compute_pbo(close: pd.DataFrame, p: dict) -> dict:
    """Probability of Backtest Overfitting via CSCV over a small lookback grid —
    the configs a user would realistically sweep. Technical factors only in this
    MVP (the fundamental scoring path differs). Always returns a dict; on any
    failure (short series, degenerate grid) it reports an honest n/a, never raises.
    """
    if p["factor"] in FUNDAMENTAL_FACTORS:
        return {"error": "PBO grid not wired for fundamental factors yet"}
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
        return {"error": "PBO grid degenerate for this factor (fewer than 2 distinct configs)"}

    def weight_fn(c, lookback):
        return factors.long_short_weights(
            _score(c, factor, lookback, min(sk, lookback - 1)), gross=gross)

    try:
        return overfitting.pbo_from_factor_grid(
            close, weight_fn, param_grid, cost_bps=p["cost_bps"], n_splits=8)
    except Exception as e:  # noqa: BLE001 - honest n/a beats a crashed workflow
        return {"error": f"{type(e).__name__}: {e}"}


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

    sec = sec_provider or SecEdgarProvider()
    macro = macro_provider or FredAlfredProvider()
    sec_obs = sec.fundamentals(symbols)
    sec_summary = []
    for symbol in symbols:
        rows = sec_obs[sec_obs["symbol"] == symbol]
        sec_summary.append({
            "symbol": symbol,
            "observations": int(len(rows)),
            "metrics": int(rows["metric"].nunique()) if not rows.empty else 0,
            "first_available": rows["available_date"].min() if not rows.empty else None,
            "latest_available": rows["available_date"].max() if not rows.empty else None,
        })

    macro_summary = []
    for series_id in macro_series:
        observations = macro.observations(series_id, start=str(macro_start.date()), end=str(as_of.date()))
        latest = _latest_known_macro(observations, as_of)
        macro_summary.append({
            "series_id": series_id,
            "vintages": int(len(observations)),
            "latest_value": None if latest is None else latest["value"],
            "observation_date": None if latest is None else latest["observation_date"],
            "available_date": None if latest is None else latest["available_date"],
        })

    text_result = None
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
        feature = (text_extractor or FinBertExtractor()).extract(text_doc)
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
        }

    return _jsonable({
        "symbols": symbols,
        "as_of": as_of,
        "sec": sec_summary,
        "macro": macro_summary,
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


def run_backtest_workflow(req: dict, ledger: "TrialLedger | None" = None) -> dict:
    """Validate -> load data -> factor -> honest backtest -> full verdict dict.

    ``ledger`` (optional) is a per-workspace TrialLedger: when supplied, the engine
    counts DISTINCT strategy configs and enforces the multiple-testing haircut
    FLOOR on the full-sample Deflated Sharpe (declared n_trials can be raised, never
    used to under-deflate). No ledger => stateless, declared n_trials used as-is.
    """
    p = _validate(req)
    fund_for_attr = None  # raw-field bundle for factor attribution (synthetic fundamentals only)

    if p["factor"] in FUNDAMENTAL_FACTORS:
        # Fundamental factors need point-in-time fundamentals. Free vendors serve
        # RESTATED numbers (look-ahead), so the only honest path today is the
        # synthetic world; real PIT data plugs into the same interface later.
        if p["provider"] != "synthetic":
            raise WorkflowError(
                "value/quality factors require the 'synthetic' provider in this MVP — "
                "free point-in-time fundamentals are not wired yet, and using restated "
                "free data would reintroduce the look-ahead this tool refuses to allow")
        syms = p["symbols"] or [f"S{i:02d}" for i in range(20)]
        world = data.make_synthetic_world(syms, start=p["start"], periods=p["periods"],
                                          seed=p["seed"], quality_to_drift=0.0016)
        close = world.close
        if len(close) < 400:
            raise WorkflowError("fundamental factors need >= 400 days (several quarters + OOS)")
        if p["factor"] in LEGACY_FUNDAMENTAL:
            # Composite value/quality scores from the ratio-metric generator.
            obs = data.make_synthetic_fundamentals(world, seed=p["seed"])
            fund = fundamentals.build_fundamentals(obs, close.index, list(close.columns))
            if p["factor"] == "value":
                score = fundamentals.value_score(fund, close)
            elif p["factor"] == "quality":
                score = fundamentals.quality_score(fund, close)
            else:
                score = fundamentals.value_quality_score(fund, close)
            # attribution needs RAW-field factor portfolios; build them from the same world
            fund_for_attr = fundamentals.build_fundamentals(
                data.make_synthetic_raw_fundamentals(world, seed=p["seed"]),
                close.index, list(close.columns))
        else:
            # Single-factor library factors run on the RAW canonical fields; the
            # sign flips "lower is better" signals so the rank points the right way.
            obs = data.make_synthetic_raw_fundamentals(world, seed=p["seed"])
            fund = fundamentals.build_fundamentals(obs, close.index, list(close.columns))
            fn_name, sign = FACTORLIB_SPECS[p["factor"]]
            raw = getattr(factor_lib, fn_name)(fund, close)
            score = factors.cross_sectional_zscore(raw if sign > 0 else -raw)
            fund_for_attr = fund  # already the raw-field bundle
    else:
        if p["provider"] == "synthetic":
            panel = data.get_panel(
                p["symbols"] or [f"S{i:02d}" for i in range(15)],
                provider="synthetic", start=p["start"],
                periods=p["periods"], seed=p["seed"],
            )
        else:
            panel = data.get_panel(p["symbols"], start=p["start"], provider="yfinance")
        close = panel.close.dropna(how="all", axis=1)
        if close.shape[1] < 2:
            raise WorkflowError("need at least 2 symbols with data for a cross-sectional strategy")
        if len(close) < MIN_PERIODS:
            raise WorkflowError(f"only {len(close)} usable days of data (need >= {MIN_PERIODS})")
        if len(close) <= p["lookback"] + p["skip"] + 21:
            raise WorkflowError("lookback too long for the data window")
        score = _score(close, p["factor"], p["lookback"], p["skip"])

    weights = factors.long_short_weights(score, gross=p["gross"])
    res = backtest(close, weights, cost_bps=p["cost_bps"])

    # Trial ledger: count DISTINCT configs this workspace has run and ENFORCE the
    # multiple-testing haircut FLOOR on the full-sample Deflated Sharpe. The declared
    # n_trials can only be raised by the engine, never used to under-deflate.
    declared_trials = p["n_trials"]
    if ledger is not None:
        rec = ledger.record(req)
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

    sigq = signal_quality.compact_scorecard(score, close)  # is the SIGNAL itself predictive?
    scorecard = res.summary(n_trials=eff_trials)
    oos = split_backtest(close, weights, split=0.7, cost_bps=p["cost_bps"])

    # Walk-forward with purge + embargo at fold boundaries: a factor's lookback can
    # otherwise leak prior-fold prices into the first OOS bars of each fold. Purge =
    # the bars of history the signal needs (lookback + skip); fundamental factors
    # have no rolling price lookback (purge 0, small embargo).
    purge = (0 if p["factor"] in FUNDAMENTAL_FACTORS
             else _effective_lookback(p["factor"], p["lookback"]) + p["skip"])
    try:
        wf = walk_forward(close, weights, n_splits=5, cost_bps=p["cost_bps"],
                          purge_bars=int(purge), embargo_bars=1)
    except ValueError as e:
        wf = {"error": str(e)}

    pbo = _compute_pbo(close, p)  # CSCV Probability of Backtest Overfitting
    tails = stats_guards.fat_tail_report(res.returns)

    # Factor-model attribution: how much of the strategy's return is just FF/Carhart
    # factor beta vs. genuine idiosyncratic alpha. Full FF5 when synthetic fundamentals
    # are available; price-only (MKT + UMD) otherwise.
    attr_model = "ff5" if fund_for_attr is not None else "carhart4"
    try:
        attribution = compact_attribution(res.returns, close, fund=fund_for_attr, model=attr_model)
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

    dd = (res.equity / res.equity.cummax() - 1.0)
    return _jsonable({
        "meta": {
            **p,
            "n_trials": eff_trials,  # ENFORCED count (>= declared) so the verdict caption is honest
            "declared_n_trials": declared_trials,  # what the user submitted (always present)
            "effective_lookback": (None if p["factor"] in FUNDAMENTAL_FACTORS
                                   else _effective_lookback(p["factor"], p["lookback"])),
            "n_symbols": int(close.shape[1]),
            "n_days": int(len(close)),
            "start_date": str(close.index[0].date()),
            "end_date": str(close.index[-1].date()),
            "engine_version": "0.0.1",
            "disclaimer": DISCLAIMER,
        },
        "verdict": verdict,
        "verdict_basis": basis,
        "scorecard": scorecard,
        "signal_quality": sigq,
        "out_of_sample": oos,
        "walk_forward": wf,
        "pbo": pbo,
        "trial_audit": trial_audit,
        "attribution": attribution,
        "fat_tails": tails,
        "equity_curve": _downsample_curve(res.equity),
        "drawdown_curve": _downsample_curve(dd),
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
