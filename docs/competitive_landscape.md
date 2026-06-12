# AlphaForge — Competitive Landscape & Brazil Thesis

> Produced June 2026 by a 20-agent research workflow (multi-source web research +
> adversarial verification; 15/15 claimed Brazilian players corroborated against
> the open web, with corrections). This is decision-support, not gospel — re-verify
> pricing and feature claims before betting on them.

## The one-line answer
**Yes — study the US idea and bring it to Brazil — but build it GLOBAL-FIRST to win the
"won't let you lie to yourself" brand and outrun copycats, then localize to B3 as a
defensible *beachhead* and a *B2B funnel* into emerging gestoras — keeping it firmly
"research software, not advice."** Brazil is a real white space but a niche/B2B-shaped
one, not a mass-retail land-grab.

---

## 1. US / global landscape — the seam AlphaForge attacks

The field splits into two halves; AlphaForge's wedge is the empty seam between them.

**Half 1 — Backtesting platforms (the "honest engine" axis):**
| Player | What they own | The gap |
|---|---|---|
| QuantConnect / LEAN (+ "Mia" AI agent) | De-facto institutional-grade retail platform | Thousands of parallel optimizations, **no native Deflated Sharpe / PBO gate** |
| Portfolio123 | **Gold standard on point-in-time / survivorship-free fundamentals** | No overfitting defenses, no AI text-signal layer |
| Build Alpha | Closest philosophical rival — sells robustness testing | Heuristic tests (noise/MC), **not** formal DSR/PBO; its genetic generator is itself a multiple-testing minefield |
| vectorbt / QuantRocket / Zipline / Nautilus | Speed / infra | Speed *is* the overfitting accelerant; no enforced verdict |
| Composer, LuxAlgo, Jenova, Surmount, Nexera (2024-26) | NL chat-to-strategy, speed-to-strategy | ~Zero overfitting guardrails — the exact failure mode AlphaForge polices |
| **Quantopian (defunct 2020)** | Not a competitor — the **market-validated wound** | "You can't crowdsource alpha, you crowdsource overfit backtests." Lead the narrative with it. |

**Half 2 — AI-signal tools (the "research copilot" axis):**
- **Research/summarization terminals** (dominant, best-funded): AlphaSense ($7.5B, owns Tegus), Bloomberg AI, Brightwave, Fintool (Microsoft), Hebbia, Rogo, Captide. They write *prose for humans* — not point-in-time numeric features for a backtest, and ship no backtester.
- **Black-box retail scores/bots** in AlphaForge's price band: Danelfin, Tickeron, Kavout. You must *trust* a pre-computed score; can't compose/test your own; published track records are the vendor's own (often the look-ahead/overfit numbers AlphaForge exists to debunk).
- **Quant signal feeds** (only truly backtest-ready): RavenPack/Bigdata.com, ExtractAlpha. Sell the *feed*, not the honest lab around it; premium/institutional, out of reach for sub-$100M.

**The gap:** No single player combines (a) enforced academic-grade overfitting defenses (DSR / PBO-CSCV) + (b) AI extraction of *structured numeric* signals from primary documents + (c) an integrated honest backtester with PIT data + realistic costs + (d) an emerging-manager price point. DSR/PBO exist as *papers and scattered GitHub*, not as a polished "this backtest is probably overfit — here's the truth" gate inside any mainstream product. **That verdict-as-a-feature, fused with signal→strategy→validation ownership, is the defensible seam.** Threat = timing: QuantConnect's Mia or a funded newcomer bolting honesty + AI-doc-signals on first. The two moats must launch together.

## 2. Brazil (B3) landscape — strikingly thin

Only **two genuine competitors**, neither covering the full thesis:

| Player | What it is | Verdict |
|---|---|---|
| **QuantBrasil** (quantbrasil.com.br, Rafael Quintanilha) | Closest thing to a retail quant lab: backtest sim across stocks/ETFs/crypto/futures/BDRs, pre-built strategies, a 0-100 "Backtest Score". ~R$75-100/mo. | **Real competitor on the honest-backtest axis** — but the Score is a *relative-ranking heuristic*: **no** walk-forward, OOS, Monte Carlo, Deflated Sharpe, multiple-testing correction, or PIT data. **No AI signal layer, no custom code/API.** Leaves the AI-signal axis and the rigor axis wide open. |
| **Nelogica Profit/ProfitChart** | Dominant B3 terminal (~80% retail day-trade share, ~20 brokers), NTSL scripting, automation, **parameter optimizer sweeping up to 100,000 combos**. | **Real competitor by audience** — but the **overfitting ANTI-PATTERN**: brute-force curve-fitting, no walk-forward, no OOS, no DSR, no AI. The perfect "this tool helps you fool yourself" foil. |

**Adjacent / NOT real competitors (verified):** SmarttBot (no-code bot-builder + paid marketplace; 6-month-window replay backtest — structurally rewards overfit equity curves), Tryd (execution terminal), MetaTrader 5 on B3 (has genetic optimization + a basic OOS forward-split — more than most local tools — but no DSR/PBO/PIT), **Bridgewise** (Israeli AI stock-scoring embedded *free* in B3's Área do Investidor since Aug 2025 — validates AI-on-equities demand, but it's fundamentals scoring + buy/sell recs, foreign IP, no backtesting).

**Data terminals (potential data sources, not competitors):** Comdinheiro (Nelogica-owned), Economatica (40-yr institutional DB, has an MCP endpoint), Quantum Axis, Mais Retorno (notable live MCP data server), Preço Justo AI (fundamentals/dividend screener + allocation-level backtester). Trackers: Gorila, Kinvo (BTG-owned), Real Valor (Empiricus-owned), TradeMap.

**What does NOT exist in Brazil:** No B3-native product combining point-in-time B3 data + enforced overfitting defenses + AI extraction of numeric signals from *Portuguese-language* filings/transcripts/news. The entire AI-signal-from-documents axis is empty; the academic rigor axis is empty.

## 3. The gap verdict — real but narrow

**Real on three counts:** (1) no B3 player fuses AI signals + honest backtester + overfitting defenses; (2) the two incumbents leave clean room; (3) a genuine **language + B3-microstructure moat** (the serious toolchain is English-only/US-data-centric; real localization — auctions/circuit-breakers/lot-sizes/mini-contracts/FII-BDR quirks, emolumentos + 20% day-trade vs 15% swing IR + JCP, PIT B3 data — is more than translation and not worth a US incumbent's bother).

**But constrained:** The "19.4M B3 investors" headline is inflated by Tesouro/savings; **active monthly equity traders ≈ 1.6M**, and (active ∩ sophisticated ∩ willing-to-pay) is realistically **thousands to low-tens-of-thousands of seats**. WTP is low and hype-anchored (Empiricus trained the market to pay for *promises*, now R$14.90-19.90/mo "cheaper than Netflix"). **High Selic** competes with the "build alpha" pitch. Day-trade culture is double-edged: 72-97% lose money (perfect honesty narrative) but were conditioned to want promises, not discipline.

**Net:** defensible white space, but a **beachhead/credibility/B2B** opportunity (serious retail quants + emerging gestoras + prop/family-office analysts), not a scale engine.

## 4. Copy-to-Brazil thesis — sound, with conditions

**What travels:** Empiricus is the proof the US paid-research model travels and Brazilians pay (R$200M+, ~180k subs) — it was literally an adaptation of Stansberry/Agora. AlphaForge is the deliberate **anti-Empiricus** (rigor vs hype) = real differentiation. India (QuantInsti/Blueshift, Tradetron) proves emerging-market quant-tooling demand.

**What must be adapted:** The robo-advisor graveyard (Magnetis, Vérios, Monetus, early Warren — <R$1bn combined vs $43bn US) warns against US-style automation in a high-Selic, advice-seeking culture. The lesson — **favor a TOOL/copilot the user drives over automated advice** — happens to align perfectly with AlphaForge's design AND keeps it outside CVM regulation. Winners (XP, Nubank) won on localized distribution + a concrete grievance; AlphaForge's grievance is ready-made: *"97% of day-traders lose; the tools actively help you fool yourself — this one won't."*

**Regulatory (decisive enabler, not blocker):** "Research software, not advice" keeps AlphaForge cleanly outside the three CVM buckets — Consultor (RCVM 19, incl. robo source-code inspection), Analista (RCVM 20/598/APIMEC), Gestor (RCVM 21) — analogous to Nelogica Profit / TradingView / Excel. **Two non-negotiables:** (a) NEVER emit personalized or specific buy/sell/price-target output; (b) properly license B3 market data. Emerging gestoras (RCVM 21) are a **channel**, not a constraint.

**COGS caveat:** B3 market-data licensing is a structural cost (fixed distributor + per-user variable). **Lean on end-of-day / point-in-time historical B3 data** (cheaper, and exactly what an honest backtester needs); live quotes = premium add-on.

## 5. Recommendation — global-first, Brazil as fast-following beachhead

1. **Phase 1 (0-12 mo) — ship the global English product** as scale engine + differentiation proof. The whole moat (honest backtester + PIT + costs + DSR/PBO **fused with** AI numeric-signal extraction) launches **together** on US data, because the threat is a US incumbent copying it first. Win the brand where the serious-quant audience already is.
2. **Phase 2 (12-24 mo) — Brazil as a credibility beachhead, not a separate product.** Same engine + a B3 data layer (start EOD/PIT, not live) + Portuguese + BR cost/tax models. Position explicitly as "research software, not advice"; never emit buy/sell calls. Lead the narrative with Quantopian + the "97% lose" study.
3. **Phase 3 — turn Brazil into a B2B channel:** sell into newly-authorized emerging gestoras (RCVM 21), prop desks, family-office analysts — the segment that can pay professional pricing — rather than chasing mass retail. Consider hybrid (tool + community/education) given the Brazilian "want a human too" signal.

**Pricing:** professional/prosumer/B2B tier, anchored *above* QuantBrasil's ~R$75-100/mo, sold on rigor — not a Netflix-priced mass play. Budget B3 data as COGS.

## 6. Key risks (ranked)
1. **Timing/copycat (highest):** QuantConnect's Mia or a funded newcomer adds DSR/PBO + AI-doc-signals onto an existing base first. The dual moat must launch together.
2. **Brazil TAM is a sliver** (~1.6M active traders; serviceable seats in the thousands–low-tens-of-thousands). Don't treat Brazil as a scale engine.
3. **Low, hype-anchored WTP** — the rigor audience is the small premium tail.
4. **High-Selic headwind** shrinks the serious-quant pool (the force that killed BR robo-advisors).
5. **Regulatory reclassification** the moment output personalizes/auto-executes/publishes specific calls → robo-consultor (RCVM 19) or analista (RCVM 20). Police in product design, not just disclaimers.
6. **B3 data licensing as COGS** — underestimating it erodes unit economics.
7. **Dual-moat execution** — out-rigor Portfolio123/Build Alpha AND out-honest the AI-hype newcomers, with a two-founder team.
8. **AI-signal credibility** — garbage signals into an "honest" backtester destroy the whole brand. PIT Portuguese-doc extraction must be accurate.
9. **Day-trade-culture mismatch** — the loudest segment wants promises; the honesty brand may repel the largest-but-wrong audience. Target the narrower serious tier.

## 7. Eight products to personally study
1. **Portfolio123** — the PIT-fundamentals rigor bar to match/exceed (no overfitting defenses, no AI → your opening).
2. **Build Alpha** — closest anti-overfitting rival; proves traders pay for robustness; shows why formal DSR/PBO beats heuristics.
3. **QuantConnect (LEAN) + Mia** — the most likely fast-follower; study Mia as the primary copycat threat.
4. **ExtractAlpha (+ RavenPack/Bigdata.com)** — the exact numeric-signal-from-NLP product Module A must produce, sold as a premium feed (the seam you own = the affordable lab around it).
5. **Numerai Signals** — closest "bring your own structured signal" model; borrow overfitting-discipline credibility, contrast user-ownership vs Numerai's one-way pipe.
6. **QuantBrasil** — the local benchmark to beat (study the Backtest Score, ~R$75-100/mo, the no-AI/no-DSR/no-PIT gaps).
7. **Nelogica Profit** — entrenched B3 competitor AND the perfect "helps you fool yourself" foil (100k-combo optimizer).
8. **Bridgewise on B3 + Empiricus** — Bridgewise validates AI-on-equities demand on B3 rails (leaves your niche open); Empiricus is the copy-to-Brazil research-subscription analog whose hype model you must consciously invert.
