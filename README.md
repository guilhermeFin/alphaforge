# AlphaForge

> The research OS for emerging quant managers — *the tool that won't let you lie to yourself.*

AlphaForge takes you from idea → tested strategy → risk-managed portfolio in one place.
This repo currently contains the **crown jewel**: a pure, tested, point-in-time-safe
research engine (`research/`). Everything else (API, UI, billing) wraps this.

## Why it's different

A single embarrassing false-positive backtest damages the brand more than a missing
feature. So correctness comes first, and three honesty guards are enforced in code and
proven in tests:

1. **No look-ahead.** Signals are shifted before they trade; `tests/test_backtest.py`
   proves *point-in-time consistency* — truncating the data never changes an earlier
   point of the equity curve.
2. **Costs on turnover.** Every change in position pays realistic frictions.
3. **Honest scoring.** The headline isn't the Sharpe ratio — it's the **Probabilistic**
   and **Deflated Sharpe Ratio** (Bailey & López de Prado), which use the return skew and
   kurtosis to discount fat-tailed, data-mined results.

Plus a set of "won't let you lie to yourself" guards drawn straight from probability
theory (Blitzstein & Hwang): base-rate fallacy, Simpson's paradox, look-ahead detection,
and fat-tail / non-normality reporting (`research/stats_guards.py`).

## Quickstart

```bash
pip install -e .[api,app,dev]     # engine + FastAPI + Streamlit
pytest                            # full suite incl. the no-look-ahead proof

# Browser workflow (Phase 1): pick universe -> factor -> honest backtest
uvicorn api.main:app --port 8000          # terminal 1 — API (http://127.0.0.1:8000/docs)
streamlit run app/Home.py                 # terminal 2 — UI  (http://localhost:8501)
# or on Windows: scripts/run_app.ps1 starts both

# CLI demos
python examples/demo_momentum.py            # offline, synthetic, no keys
python examples/demo_momentum.py --provider yfinance   # real tickers (free, needs internet)
python examples/demo_quantamental.py        # text -> LLM-style signal -> backtest
```

The workflow runs a long/short factor strategy end-to-end and shows an honest
scorecard, an out-of-sample / walk-forward overfit verdict, and the risk guards. On
synthetic data the momentum strategy *correctly fails* the Deflated-Sharpe haircut —
which is the whole point. The Streamlit page talks to the API when it's up and falls
back to in-process execution otherwise; both paths share `api/service.py`, so the
numbers are identical by construction (and tested to be).

## Layout

```
research/
  data.py         # provider interface + synthetic regime/fat-tail world + fundamentals + yfinance
  factors.py      # momentum, vol, reversal; cross-sectional z-score; dollar-neutral weights
  fundamentals.py # POINT-IN-TIME fundamentals (filing-date lagged) -> value & quality factors
  backtest.py     # THE MOAT: point-in-time backtester (no look-ahead, costs on turnover)
  metrics.py      # Sharpe/Sortino/Calmar/drawdown + Probabilistic & Deflated Sharpe + JB
  walkforward.py  # in-sample vs out-of-sample, walk-forward folds, overfit flag
  stats_guards.py # base-rate, Simpson, look-ahead, fat-tail guards (Blitzstein lessons)
  signals_llm.py  # Module A: validated, bounded, source-cited LLM signal extraction
  quantamental.py # text -> point-in-time signal panel (offline + cached-Claude extractors)
api/
  service.py      # ONE source of truth for the browser workflow (validation + orchestration)
  main.py         # thin FastAPI app: GET /health, POST /backtest
app/
  Home.py         # Streamlit page (API-backed with in-process fallback)
tests/            # pytest — engine + API contract + Streamlit AppTest UI tests
examples/         # runnable demos
scripts/          # run_app.ps1 — start API + UI on Windows
docs/             # distilled references (Edwards & Magee TA, Blitzstein probability)
```

## Knowledge base

`docs/` holds implementation-grade distillations used to design the engine:
- `edwards_magee_ta_reference.md` — classical technical analysis, every rule made codable.
- `blitzstein_probability_reference.md` — probability foundations + the honesty checks above.

## Status

Phase 1 in progress: research engine **working and tested**; quantamental (text → signal)
pipeline live with a cached Claude extractor; **browser workflow live** (FastAPI + Streamlit).
Not investment advice; research software only. Synthetic results are an engine smoke-test,
never evidence of an edge. Free Yahoo data is survivorship-biased — the UI says so.

## Next

Accounts + billing, point-in-time fundamental data, a broader factor library, saved
reproducible workspaces, and cost-metered LLM signals at scale. See `CLAUDE.md` for the
full roadmap.
