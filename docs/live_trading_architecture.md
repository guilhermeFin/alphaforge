# AlphaForge → Live Auto-Trader: Architecture & Roadmap

> Design doc. Generated 2026-06-16. Answers the question: *how do I turn AlphaForge (a
> research OS that validates edges) into a machine that trades my own money automatically?*
> Grounded in the existing `research/` engine, the Quant Knowledge base, and the live-ops
> pattern already prototyped in the Prediction Bot.
>
> **Not investment advice. Research and engineering design only.** Every dollar figure and
> return assumption here is an illustrative placeholder, never a forecast.

---

## 0. The honest target (read this before anything else)

The stated goal is "always make more money, success rate > 70%." As the quant on this, I have
to reframe that, because chasing it directly is how disciplined people blow up — and AlphaForge
was literally built to stop that.

**Win rate is the wrong objective.** What determines whether the machine makes money is
*expectancy*, not hit rate:

```
expectancy = win_rate × avg_win − loss_rate × avg_loss
```

You can win 70% of trades and lose money (premium-selling that gets one fat-tail blowup), or
win 40% and compound beautifully (trend following). The strategies that *naturally* produce a
high win rate buy it with a fat left tail — exactly the silent lie AlphaForge's DSR/PBO layer
exists to catch. Your own momentum demo *correctly fails* the Deflated-Sharpe haircut. That is
the system working.

**So the real success criteria for this machine are:**

| Metric | Target | Why |
|---|---|---|
| Deflated Sharpe Ratio (DSR) on real PIT data, OOS-first | > 0, ideally > 0.5 | The honest, multiple-testing-adjusted edge test. The gate. |
| PBO (prob. of backtest overfitting) via CSCV | < 0.5, ideally < 0.2 | "Given I picked the best of N, is it below-median OOS?" |
| Information Ratio of the signal (IC_IR) | floor ≈ 0.5, good ≈ 1.0 | Mean IC alone is noise; IR is the headline. |
| Net-of-cost Sharpe (live) | > 0.7 sustained | Survival-grade risk-adjusted return. |
| Max drawdown | within a pre-set hard limit | Survival > returns. The kill-switch trigger. |

Win rate is *reported*, never *optimized*. If a validated signal happens to win 70% of months,
great — but it earns its place by surviving DSR/PBO, not by hitting a hit-rate number. Ambition
is channeled into **compounding a small, real, survivable edge**, which is what a hedge fund
actually does. Anyone promising a reliable >70% win rate on liquid equities is selling something.

---

## 1. What you already have vs. what's missing

AlphaForge is the **research/validation half** of a trading firm: idea → honest verdict. A live
auto-trader needs the **execution half** bolted on, behind the same honesty gate. The crucial
design rule: **the validated edge is the gate. No execution code runs on a signal that hasn't
passed DSR/PBO on real point-in-time data.**

```
   ┌─────────────────────── EXISTING (research/) — the moat ───────────────────────┐
   │ data/providers → factors / fundamentals / quantamental(LLM) → signal panel     │
   │   → backtest (PIT, costs on turnover) → metrics (PSR/DSR) → walkforward (OOS)   │
   │   → signal_quality (IC/IC-IR) → stats_guards → VERDICT: real or overfit         │
   └───────────────────────────────────────┬───────────────────────────────────────┘
                                            │  (only a CERTIFIED signal passes)
   ┌────────────────────────────────────────▼──────────────────────────────────────┐
   │                          NEW (live/) — the auto-trader                          │
   │  1 PIT data feed (live)  →  2 signal refresh  →  3 allocator (target weights)    │
   │  →  4 risk gate (pre-trade limits)  →  5 execution (broker API)                  │
   │  →  6 reconciliation & monitor (kill-switch)  →  7 scheduler/ops + audit log     │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

### Module-by-module map

| Live need | Status | Where it lives / goes |
|---|---|---|
| Point-in-time data | ⚠️ partial — synthetic + survivorship-biased yfinance only | **Critical path:** wire Sharadar SF1 (SimFin fallback) behind `fundamentals.build_fundamentals` — already your named #1 gate in `strategic_focus.md` |
| Validated signal | ✅ engine exists; ❌ no certified live edge yet | `research/` + `quantamental.build_signal_panel` promoted to first-class |
| Backtest / DSR / IC | ✅ done & tested | `research/backtest.py`, `metrics.py`, `signal_quality.py` |
| PBO / CSCV | ❌ not built | New `research/overfitting.py` (spec already in `product_improvement_research.md` §4.4) |
| Allocator (weights → orders) | ⚠️ dollar-neutral L/S weights exist; no live allocator | `factors.py` weights + **new** `live/allocator.py` (fractional Kelly + HRP from Quant Knowledge) |
| Execution / broker | ❌ none | **New** `live/broker.py` — Alpaca first (paper), IBKR later |
| Live risk gate / kill-switch | ❌ none (but pattern exists) | **New** `live/risk.py` — port the Prediction Bot safety rails |
| Scheduler / ops / audit | ⚠️ Prediction Bot has the pattern | **New** `live/runner.py` + cron, mirroring `main.py --run/--settle/--status/--report` |

---

## 2. The live modules (what to build)

### 2.1 `live/data_feed.py` — point-in-time, but current
Same PIT discipline as research (filing-date lag, no backfill, `searchsorted(available_date) →
place → ffill(limit) → never backfill`), but refreshed to "as of this morning." Reuses the
research provider interface so the live signal is computed *the identical way* it was backtested
— byte-for-byte, the way `api/service.py` already guarantees UI == engine. **This is the #1
correctness rule: the live signal path must be the same code as the backtest signal path.**

### 2.2 `live/allocator.py` — signal → target weights → sized orders
Three stages, all already in your Quant Knowledge notes:

1. **Raw weights** from `factors.py` (cross-sectional z-score → dollar-neutral L/S), with an
   **Amihud illiquidity screen** to drop untradeable names (Microstructure note §9).
2. **Risk-aware sizing**: start with **HRP** (no matrix inversion, robust on small samples) or
   1/N as the honest baseline — *not* raw mean-variance (the error maximizer). Scale gross
   exposure by **fractional Kelly (quarter- to half-Kelly)**, because your true μ estimate is
   mostly noise (IC² is the variance explained — tiny). Optional **beta-neutrality** upgrade
   over plain dollar-neutral.
3. **Order generation**: diff target vs. current positions → list of orders, with a
   **no-trade tolerance band** (only trade if drift > δ, calibrated to cost) so you don't churn.

### 2.3 `live/risk.py` — the pre-trade gate and kill-switch
Direct port of the Prediction Bot's `Safety Rails`, generalized to equities. Every order batch
passes through hard, non-overridable checks **before** it can reach the broker:

- Max position size (% NAV per name) and max gross/net exposure
- Max participation per order (≤ a few % of ADV — your flat-bps cost model is only honest below
  this; above it you'd need the deferred square-root impact model)
- Daily loss limit and **trailing max-drawdown kill-switch** → flatten and halt
- Min-edge filter (don't trade when expected alpha < 2× cost — Rebalancing note §8)
- Circuit breaker on data staleness / broker disconnect / reconciliation mismatch

### 2.4 `live/broker.py` — execution
Thin adapter over a broker API. **Alpaca first** (free paper-trading sandbox, clean REST API,
fractional shares) for the entire validation phase; **Interactive Brokers** later if you outgrow
it. Responsibilities: place orders (start with marketable-limit / VWAP-style child orders for
anything non-trivial), poll fills, return actual fills for reconciliation. **Never** assume a
fill — always read it back.

### 2.5 `live/reconcile.py` + `live/monitor.py` — trust but verify
After every run: compare intended vs. actual positions (implementation shortfall — Microstructure
note §3). Log the gap. If it exceeds a threshold, halt and alert. Monitor tracks live equity,
rolling drawdown, and live IC vs. backtest IC (expect live ≈ ½ backtest as a prior, not a law).

### 2.6 `live/runner.py` — the operator loop
Mirror the Prediction Bot CLI you already run every morning:

```
python -m live.runner --run       # refresh data → signal → allocate → risk-gate → (paper) trade
python -m live.runner --status    # positions, NAV, drawdown, exposure
python -m live.runner --reconcile # intended vs actual, implementation shortfall
python -m live.runner --report    # weekly: live Sharpe, IC, DD, vs benchmark
python -m live.runner --halt      # manual kill-switch → flatten + disable
```

Driven by a scheduler (Task Scheduler / cron). For a **monthly** quantamental strategy this runs
rebalance logic monthly and risk/monitor checks daily — no intraday infra, which is exactly why
the monthly cadence is the right first target (your `strategic_focus.md` reaches the same
conclusion for the product).

---

## 3. Cadence: start monthly, not intraday

Your own notes already make this call, and it's correct for a solo operator:

- **Monthly / quantamental fundamental** = slow-decaying signal, tiny participation per trade,
  flat-bps costs are honest, no intraday data or market-impact modeling, no latency war. Runs on
  cheap monthly vendor snapshots. **This is the build.**
- **Intraday / high-frequency** = fast IC decay, 5–20% ADV participation, square-root impact
  model mandatory, tick data, slippage engineering, and (per your docs) a co-founder. **Deferred.**

The honest math (Microstructure note §7): Information Ratio = IC × √breadth, but IC per bet falls
faster at high turnover than √breadth rises, *and* every bet costs more. High turnover is where
retail strategies go to die. Slow and survivable beats fast and fragile.

---

## 4. Phased build — each phase has a gate that can halt the project

**Phase 0 — Finish the honesty spine (research only, no money).**
Wire one real PIT vendor (Sharadar SF1 / SimFin). Build PBO/CSCV (`overfitting.py`). Add the
trial ledger so DSR can't be gamed. *Gate:* the engine runs a real fundamental/quantamental
signal end-to-end and returns an OOS-first DSR/PBO verdict on **real** data.

**Phase 1 — Find one validated edge.**
Run candidate signals (start with your earnings-call sentiment work + classic value/quality
factors) through the gate. *Gate:* **≥1 signal with DSR > 0 and PBO < 0.5 out-of-sample.** If
nothing passes, the project pauses here — and that's the system protecting you, not failing.
**No execution code is written until this gate is cleared.**

**Phase 2 — Paper trade the validated signal.**
Build `allocator → risk → broker(paper) → reconcile → runner`. Run on Alpaca paper for **at
least 3–6 months**. *Gate:* live paper Sharpe and IC track the backtest within tolerance; max DD
stays inside the limit; reconciliation is clean.

**Phase 3 — Small live capital.**
Deploy an amount you can fully afford to lose. Same code, real fills, real slippage, real
psychology. *Gate:* live net-of-cost performance holds for a meaningful window with no rail
breaches.

**Phase 4 — Scale deliberately.**
Increase size only as live results and capacity (ADV limits) allow. Add HRP/Ledoit-Wolf
refinements, beta-neutrality, regime conditioning. Open the daily lane only if you have both a
proven reason and the infra/co-founder for it.

---

## 5. Kill criteria (decided in advance, in writing)

Pre-committing these is what separates a system from gambling:

- Live drawdown breaches the hard limit → flatten, halt, post-mortem before any restart.
- Live IC goes flat or flips sign for N consecutive rebalances → signal is dead/crowded; retire it.
- Reconciliation gap exceeds threshold → halt until the execution bug is found.
- You can't explain *why* a trade was placed → halt. A black box you don't understand is a
  liability, not an edge.

---

## 6. Open inputs I need from you

1. **Broker** — confirm Alpaca for paper (recommended) or do you want IBKR from day one?
2. **PIT data budget** — SimFin (cheap, validate PMF) vs. Sharadar SF1 (better) for Phase 0?
   This is the literal gate; nothing downstream runs without it.
3. **Capital & risk limits** — rough live capital for Phase 3, and your hard max-drawdown number
   (e.g., 15%? 20%?). This sets the kill-switch.
4. **Universe** — US large/mid-cap liquid names to start? (Keeps Amihud screening simple and
   flat-bps costs honest.)
5. **Signal to validate first** — start with the earnings-call sentiment signal, classic
   value/quality factors, or both combined?

---

## 7. One-paragraph summary

You've already built the hard, rare part: an honest research engine that refuses to lie to you.
The auto-trader is that engine plus a disciplined execution loop — data feed, allocator (HRP +
fractional Kelly), a hard risk gate (ported from your Prediction Bot rails), a broker adapter
(Alpaca paper first), reconciliation, and a scheduled operator loop. The non-negotiable rule is
that **a signal must pass DSR/PBO on real point-in-time data before a single order is ever
generated.** Target the monthly quantamental cadence, paper-trade for months, deploy small,
scale slowly, and measure success by deflated, net-of-cost, drawdown-controlled performance —
never by a win-rate number. That's not a watered-down version of the ambition. That *is* how the
ambition survives contact with the market.

---

*Maps onto: `research/` (existing), `product_improvement_research.md` (Phase 0 specs),
`strategic_focus.md` (cadence decision), Quant Knowledge: [[Portfolio Construction and Risk]],
[[Market Microstructure and Execution]], [[Performance and Risk Metrics]],
[[Overfitting and Validation]], [[Signal Quality and Information Coefficient]].*
