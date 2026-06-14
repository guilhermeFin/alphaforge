# AlphaForge — Strategic Focus: Who We Build For

> Decision doc. Generated 2026-06-13 from a 4-lens + adversarial-synthesis research pass
> (positioning, engine architecture, market/WTP, founder pragmatics). Answers the question:
> *should AlphaForge serve daily-rebalance factor quants AND monthly fundamental/quantamental
> quants, or specialize?*

## The decision

**Specialize the ICP, brand, and monetization on the fundamental / quantamental quant.
Keep ONE shared honest engine.**

This is not a contradiction. The four lenses split 3–1 only on the surface:
- Positioning, Market/WTP, and Founder-pragmatics all said **specialize fundamental/quantamental**.
- Architecture said **serve both, one engine** — but its own words were *"serve both must not imply
  monetize both equally,"* *"the data layer is the ONLY fork,"* *"daily-only technical quants sit on
  the free rung forever and won't fund the roadmap."*

They answer **different questions**: the three "specialize" lenses answer *who you sell to*; the
architecture lens answers *what you build*. The committed answer takes both:
**one engine in `research/`, specialized GTM/pricing on the fundamental buyer.**

## Why (the repo itself forces it)

1. **The paid ladder is already a fundamental-data ladder.** Free (synthetic+yfinance) → SimFin →
   Sharadar → S&P Capital IQ/Compustat. A daily technical trader never needs Compustat, so they never
   climb it. The monetization architecture has already cast the vote — positioning just ratifies it.
   (`service.py` even hard-refuses non-synthetic providers for value/quality today — real PIT data is
   the literal gate on the whole ladder.)

2. **All four moat pieces fire simultaneously only for this buyer:** AI numeric-signal extraction from
   filings (dead weight to a momentum trader), PIT/restatement/filing-lag integrity (a fundamental
   problem), DSR/PBO multiple-testing (most needed where samples are small — monthly rebalance over
   10–15y ≈ ~150 obs), and the honest costed backtest. Against the daily buyer, AlphaForge is one
   bolt-on feature attacking QuantConnect/LEAN on its home turf.

3. **The competitive flank is genuinely open here.** Portfolio123 has gold-standard PIT but zero
   overfitting defenses and no AI numeric layer; AlphaSense/Fintool write prose with no backtester;
   ExtractAlpha sells a feed, not the lab. Nobody fuses PIT fundamentals + AI numeric signals + DSR/PBO
   at emerging-manager price.

4. **The brand only bites at the slow cadence.** "Won't let you lie to yourself" is most credible where
   the lie is *silent* — restatement leakage, filing-date backdating, look-ahead in AI-extracted
   signals. On daily momentum, IC decay is visible within a week; there's no hidden lie to catch.

5. **It's the cheapest path for a solo quant founder.** The fundamental lane runs on cheap monthly
   vendor snapshots (batch, no intraday infra, no market-impact modeling) — exactly the work Guilherme
   can do without the not-yet-looped-in CS co-founder. The daily lane's intraday feeds + slippage
   modeling are his weakest, most COGS-heavy, most co-founder-dependent surface.

## The strongest counterargument (and how the decision bends to survive it)

**Counter:** TAM starvation + concentrated brand blast-radius. The sub-$100M quantamental quant who
rolls their own numeric signals, distrusts black boxes, pays $300–800/mo, AND rebalances monthly may be
a tiny pool — and specializing concentrates the bet so one hallucinated/look-ahead AI signal permanently
loses the one buyer you bet on, while leaning on the two least-proven parts of the stack (unbuilt real-PIT
data + LLM extraction accuracy).

**How it survives — it bends in two ways:**
- **Validate the TAM with B3-level rigor before pouring copy/pricing in.** Structural defense:
  revenue-TAM > headcount-TAM (institutional budgets, fiduciary reason to pay for defensible research).
  A smaller headcount at $300–800/mo out-funds a larger hobbyist crowd churning at $49–99.
- **Make the AI-signal validation gate the product's literal front door.** Because the bet is
  concentrated, every AI numeric signal must pass `signals_llm` pydantic bounds + the `signal_quality`
  IC/decay/OOS-IC/re-run-stability scorecard *before it is allowed to count* — a weak/unstable extraction
  shows as **UNCERTAIN**, never dressed as alpha. That converts the existential risk into the product's proof.
- **Where it does NOT survive:** if the real-PIT data wiring stalls, the whole thesis stalls. So that is
  the one item that cannot be deferred. SimFin is the cheaper PIT fallback to validate PMF before
  committing to Sharadar/Compustat COGS.

## What this means concretely

**Win first — the minimum lovable wedge** (all stayable in `research/` + existing api/app, no co-founder
required): a monthly quantamental quant points at real fundamentals + filings → AlphaForge extracts
bounded numeric signals → runs them through the PIT backtester → returns an OOS-first DSR/PBO verdict +
IC scorecard that says **"real"** or **"overfit"** on REAL data.

**Build sequence (research/):**
1. **Wire one real PIT vendor** — Sharadar SF1 (SimFin as cheaper fallback) behind the existing
   `fundamentals.build_fundamentals` interface, with correct `available_date`/filing-lag + restatement
   vintages. Replaces the synthetic-only gate at `service.py:170-185`. **THE critical path — nothing
   converts until it ships.**
2. **Promote `quantamental.build_signal_panel` to a first-class factor type** in `service.py` + `Home.py`
   (currently stranded in `examples/demo_quantamental.py`) with an upload-or-ticker→filings front door,
   so the AI-signal → PIT-backtest → verdict loop is self-serve.
3. **Implement PBO/CPCV** from the CPCV/HPV guide + López de Prado, **tuned for the low-sample monthly
   regime** (purge/embargo around overlapping monthly labels) — the headline differentiator vs Portfolio123.
4. **Harden the AI-signal validation gate** (`signals_llm.py` + `signal_quality.py`) into a mandatory
   pre-count check surfaced in the verdict.
5. **Default to a wide universe** so monthly-cadence DSR has enough effective cross-sectional samples to
   avoid returning "inconclusive."

**Explicitly defer** (these serve the daily buyer only): the square-root market-impact cost model,
capacity/ADV haircuts, intraday/tick data, fast-IC-decay turnover machinery. The architecture lens is right
that they're cheap and correct *eventually* — they wait for the expansion trigger.

**Expansion trigger to open the daily lane properly:** only after BOTH (a) ≥10 paying
fundamental/quantamental users AND (b) the CS co-founder is fully onboarded to own intraday-data +
market-impact engineering. Until then, daily technical factors (`factors.py`) stay a **free-tier on-ramp**,
never a product — keep them exactly as-is, don't invest in them.

**Never fork** the backtester, metrics, or IC scorecard. The moat is the shared honesty spine; duplicating
it is the named failure mode. (Verified: the PIT spine, metrics, and IC scorecard are already shared via the
identical `searchsorted(available_date) → place → ffill(limit) → never-backfill` pattern across prices,
text, and fundamentals.)

**Marketing alignment:** the landing page currently leans technical (momentum verdicts). Re-point the C/D
image slots to a **quantamental honesty case** — a CERTIFIED real AI signal at high DSR vs a
look-ahead-inflated one correctly discounted — so prospects self-sort into the paying segment.

## Relationship to the improvement plan

This focus decision **re-prioritizes** [product_improvement_research.md](product_improvement_research.md):
the trial ledger, purge/embargo, PBO, and CVaR/rolling-IC all stay high (they serve this buyer directly);
the **market-impact cost model is demoted to the deferred/expansion-trigger bucket** because it serves the
daily buyer. The PIT-vendor wiring and the quantamental-promotion (above) are the new critical-path items
that sit *ahead of* most of the honesty-layer polish, because without real data the wedge has nothing to run on.
