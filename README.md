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
pip install -e .            # numpy, pandas, scipy
python examples/demo_momentum.py            # offline, synthetic, no keys
python examples/demo_momentum.py --provider yfinance   # real tickers (free, needs internet)
pytest                       # 20 tests, incl. the no-look-ahead proof
```

The demo runs a 12-1 long/short momentum strategy end-to-end and prints an honest
scorecard, an out-of-sample / walk-forward overfit verdict, and the risk guards. On
synthetic data the strategy *correctly fails* the Deflated-Sharpe haircut — which is the
whole point.

## Layout

```
research/
  data.py         # provider interface + synthetic regime/fat-tail generator + yfinance
  factors.py      # momentum, vol, reversal; cross-sectional z-score; dollar-neutral weights
  backtest.py     # THE MOAT: point-in-time backtester (no look-ahead, costs on turnover)
  metrics.py      # Sharpe/Sortino/Calmar/drawdown + Probabilistic & Deflated Sharpe + JB
  walkforward.py  # in-sample vs out-of-sample, walk-forward folds, overfit flag
  stats_guards.py # base-rate, Simpson, look-ahead, fat-tail guards (Blitzstein lessons)
  signals_llm.py  # Module A: validated, bounded, source-cited LLM signal extraction
tests/            # pytest — the research engine must be well-tested
examples/         # runnable demos
docs/             # distilled references (Edwards & Magee TA, Blitzstein probability)
```

## Knowledge base

`docs/` holds implementation-grade distillations used to design the engine:
- `edwards_magee_ta_reference.md` — classical technical analysis, every rule made codable.
- `blitzstein_probability_reference.md` — probability foundations + the honesty checks above.

## Status

Phase 0/1 research engine: **working and tested.** Not investment advice; research
software only. Synthetic results are an engine smoke-test, never evidence of an edge.

## Next

API (FastAPI) over `research/`, a Streamlit page in front, point-in-time fundamental data,
a factor library, and cost-tested LLM signals. See `CLAUDE.md` for the full roadmap.
