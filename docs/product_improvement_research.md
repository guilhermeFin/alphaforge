# AlphaForge — Product Improvement Research

> Verified research pass on reliability, speed, and trust. Generated 2026-06-13.
> Every code claim below was audited against the actual source; every external claim
> was independently re-verified (adversarially). Corrections from that verification
> are called out explicitly — do not skip the "Corrections" section, three of the
> original research findings were wrong or inverted.

---

## 0. How to read this doc

- **Section 1** — what the engine already does right (don't rebuild it).
- **Section 2** — the six gaps, each CONFIRMED present in the code with file:line evidence.
- **Section 3** — corrections from adversarial verification (load-bearing; one is a marketing landmine).
- **Section 4** — Tier-1 implementation specs (implementation-grade: signatures, tests, integration, pitfalls).
- **Section 5** — Tier-2/3 roadmap.
- **Section 6** — speed.
- **Section 7** — open inputs still needed.
- **Section 8** — priority order.

Scope note: near-term work stays in `research/` (+ the existing `api/service.py` / `app/Home.py`
seams where a spec needs them). No new architecture.

---

## 1. What the engine already does right

The honesty core is genuinely ahead of QuantConnect and Portfolio123 on the dimension AlphaForge
claims to own. Verified present and correct:

- **Point-in-time-safe backtest** — `backtest.py` enforces `execution_lag >= 1`, shifts the signal
  before it trades, charges costs on turnover; `tests/test_backtest.py` proves truncation invariance.
- **Fat-tail-/multiple-testing-aware scoring** — PSR + DSR as the headline (not naked Sharpe),
  skew/kurtosis/Jarque-Bera (`metrics.py`).
- **Walk-forward OOS with separate overfit vs significance flags** (`walkforward.py`).
- **IC scorecard** with decay, IS/OOS split, quantile monotonicity, rank-autocorr, coverage
  (`signal_quality.py`).
- **Probability guards** — base-rate/Simpson/look-ahead/fat-tail (`stats_guards.py`).
- **50 passing tests, green CI.**

The gaps below are what separates "good research tool" from "the tool professionals trust with
their own money."

---

## 2. The six gaps — all CONFIRMED against the code

| # | Gap | Evidence | Severity |
|---|-----|----------|----------|
| 1 | **No automatic trial ledger.** `deflated_sharpe_ratio(n_trials)` is caller-supplied; defaults to 1; `n_trials==1` applies **zero** multiple-testing haircut. UI slider is self-reported "honesty input." A user can run 100 distinct strategies one-by-one at `n_trials=1` and get full DSR credit on each. | `metrics.py:137,148-149,161`; `service.py:79,215`; `Home.py:130-133` | **Critical — it's the brand claim, wide open** |
| 2 | **No purge/embargo at walk-forward boundaries.** `walk_forward` slices precomputed full-history weights with `signal.iloc[lo:hi]`. A factor with an L-bar lookback (momentum L≈147) computes its first ~L OOS values from prior-fold prices → boundary contamination inflates stitched OOS Sharpe & the `passes` verdict. | `walkforward.py:68,72-73`; weights computed once over full panel at `service.py:209-211` | **Critical — un-purged WF is the most embarrassing possible bug for this brand** |
| 3 | **Flat, size-blind cost model only.** `costs = turnover * cost_bps/1e4`. No market impact / participation. A strategy trading 50% of a name's ADV looks identical to a tiny one. Systematically overstates net Sharpe for high-turnover strategies — exactly the ones most likely data-mined. | `backtest.py:76` | **High** |
| 4 | **No CVaR / Expected Shortfall.** Engine *shouts* "FAT TAILS — Gaussian VaR understates risk" then hands the user only `max_drawdown` (one realized path) + Calmar. No distributional tail-loss number. Lie by omission. | `metrics.py` (192 lines, no VaR/CVaR/ES); `stats_guards.py:88-90` | **Medium-high** |
| 5 | **IC robustness judged by a single 70/30 split.** `scorecard()` compares two aggregate means once. A signal +0.04 IC for 3y then −0.04 for 2y can still show positive IS & OOS means and get stamped "PREDICTIVE." Regime flips are laundered. | `signal_quality.py:178-180,187` | **High** |
| 6 | **No PBO / CSCV anywhere.** All defenses answer "is THIS config good OOS?" None answer the question that kills quant books: "given I picked the best of N I tried, how likely is it below-median OOS?" The memory note "honest backtester + DSR/PBO" overstates — only DSR exists, PBO does not. | grep across all of `research/`: no matches for PBO/CSCV/combinatorial/purged | **High — the capstone honesty test** |

---

## 3. Corrections from adversarial verification (READ THIS)

Three claims from the first research pass were wrong. They are corrected here so they don't
propagate into code or marketing.

### 3.1 ❌→ The arXiv "implementation risk" paper does NOT show alpha-sign flips

- **Paper is real:** arXiv:2603.20319, *"Implementation Risk in Portfolio Backtesting: A Previously
  Unquantified Source of Error"* (Dong Yin, Takeshi Miki, Vladislav Lesnichenko, Vasyl Gural; 19 Mar 2026;
  SSRN 6443898). 15 strategies × 5 engines × 30 buckets × cost regimes {0, 18, 36, 60} bps.
- **What's true:** at 0 cost all engines agree exactly (0.000% divergence); under realistic costs they
  diverge, correlated with cost intensity (Spearman ρ=0.93), peaking at 3.71% for high-turnover rotation.
- **What was INVERTED:** the original claim said divergence is "enough to flip sign-of-alpha." The paper
  found the **opposite** — *all engines agree on the sign of every metric* (conclusion stability index = 1).
  Even the worst strategy stays negative across all five engines. Implementation risk introduces
  measurable ambiguity in performance **attribution (magnitudes)**, NOT in the investment **conclusion**.
- **Implication for AlphaForge:** the honest, still-valuable feature is to report an **implementation-
  uncertainty interval** — run the strategy through two internal cost models and show the *range* of plausible
  net Sharpe. Do **not** market it as "other engines flip your alpha; we don't." That would be a falsifiable
  overclaim, fatal for a brand built on honesty.

### 3.2 ❌→ "Almgren-Chriss square-root model" is a misnomer

- The **original Almgren-Chriss (2000)** model is **LINEAR** in trade rate (permanent γ·v, temporary η·v),
  not square-root.
- The **empirical equity calibration** (Almgren, Thum, Hauptmann & Li 2005, Citigroup data) explicitly
  **rejects** exponent 0.5 for temporary impact in favor of **≈0.6**, and it scales with the trade **rate**
  (shares/time vs daily volume), not raw `size/ADV`. Fitted η ≈ 0.142.
- The robust **"square-root law"** (exponent ≈0.5, impact ∝ σ·√(Q/V)) is a **different** model — Bouchaud/CFM,
  about the impact of a full **metaorder** of total size Q, not the AC instantaneous-rate term.
- **Implication:** the cost spec in §4.3 uses the √-participation form (defensible as the universal
  square-root law). **Rename it** — call it `SquareRootImpactCost` / "square-root market-impact law," **not**
  "Almgren-Chriss." Keep the exponent **γ configurable** (default 0.5; expose 0.6 for the AC-2005 calibration)
  and put η, spread, γ in `meta` so the tool never fakes precision on numbers that are venue/asset-dependent
  assumptions.

### 3.3 ⚠️ IC benchmarks need tightening

- Original "typical IC 0.05–0.15" is slightly generous and conflates single factors with composites.
  Verified: **single raw factors ≈ 0.02–0.08; strong/composite signals ≈ 0.05–0.15; >0.15 is rare and
  overfitting-suspect** (suspicion should start ~0.15–0.20 for monthly cross-sectional equity, not wait for 0.30).
- **The mean IC matters far less than the Information Ratio (mean IC / std IC).** Monthly IC std is typically
  3–11× its mean, so report **IR (IC_IR), floor ≈ 0.5, excellent ≈ 1.0** as the headline, not mean IC alone.
  (The engine already computes `ic_ir` — promote it in the UI.)
- "Live IC ≈ half backtest IC" is a reasonable **prior, not a law** — degradation scales with how many
  strategies were tested (→ ties straight back to the trial ledger, §4.1) and with crowding.

### 3.4 ✅ Confirmed correct (use as-is)

- **CPCV / CSCV methodology** — with two precisions: purging is **bilateral** (drops contaminated train obs
  on *both* sides of the test fold); embargo is **post-test only**, sized as a % of T; **PBO = Pr(logit ≤ 0)**
  over the combinatorial splits (the "fraction below median" is its empirical estimator). Matches your
  supplied CPCV/HPV guide.
- **Portfolio123 PIT** — SUPPORTED exactly: effective/filing dates first, then **30-day lag (interim) / 45-day
  (annual)** as last resort; survivorship-bias-free (keeps delisted). Caveat: their FactSet-era data backfills
  later-released figures, so PIT fidelity is imperfect — a gap AlphaForge can beat with a stricter audit log.
- **HRP** — SUPPORTED: hierarchical clustering on a **distance transform** of correlation
  (d=√(0.5(1−ρ))) → quasi-diagonalization → recursive bisection with inverse-variance cluster weights.
  Note it's **inversion-free, not covariance-free** (still uses Σ for cluster variance).

---

## 4. Tier-1 implementation specs

All five are pure `research/` additions (+ the `service.py`/`Home.py` seams). No new dependencies.
Order shipped should be 1 → 2 → 5 → 3 → 6 (rationale in §8).

### 4.1 Trial-count ledger → auto-feeds `n_trials` into Deflated Sharpe

**Why:** closes gap #1 — the single most-gamed honesty knob. The engine itself counts the **distinct**
strategy configs run this workspace and refuses to deflate by less, regardless of what's declared.

**New module `research/trial_ledger.py`** (pure stdlib — `hashlib`/`json`/`time`/`dataclasses`, no numpy):

- `trial_fingerprint(req) -> str` — stable SHA over the *strategy-defining* knobs
  (`provider, factor, lookback, skip, cost_bps, gross, periods, seed, start, symbols`),
  symbol-order-insensitive, **excludes declared `n_trials`** (else it's circular/exploitable),
  **includes `seed`** (sweeping seeds on synthetic data IS multiple testing).
- `TrialLedger` dataclass: `record(req)` increments `total_runs` and a `distinct_count`
  (keyed on fingerprint); `effective_n_trials(declared) = max(declared, distinct_count, 1)`
  — a **floor, not a replacement** (honest over-declaration of paper-trials is respected);
  `reset(reason)` is **append-only logged, never silent** (`reset_log` + `n_resets` in every snapshot).

**Integration:**
- `service.run_backtest_workflow(req, ledger=None)` — ledger defaults `None` so all existing
  stateless callers/tests stay green. After `res = backtest(...)`, call `ledger.record(req)` and
  `eff_trials = ledger.effective_n_trials(p["n_trials"])`.
- Change **only** `res.summary(n_trials=eff_trials)` (`service.py:215`). **Do NOT touch**
  `walk_forward`'s DSR (it uses folds-as-trials — orthogonal axis, double-count if combined).
- Return a `trial_audit` block (`declared`, `effective`, `haircut_was_raised`, `distinct_count`).
- `api/main.py`: per-session **cookie**-keyed ledger (`af_session`, httponly, 8h) in a module dict —
  a single global ledger would conflate users *and* break determinism tests. Add `POST /session/reset`.
- `app/Home.py`: ledger in `st.session_state` (one per browser tab — the natural per-workspace unit you chose);
  sidebar `st.metric("Distinct strategies run this session", …)` above the slider; `st.warning` when the
  engine overrode the declared count.

**Decisive test:** `test_ledger_auto_raises_deflation_across_distinct_runs` — running 9 more distinct configs
in the same ledger must drive the *full-sample* `deflated_sr` of the original config down (`<=`, not `<`,
because a strong planted signal can float-saturate PSR≈1.0). Plus: rerun-doesn't-punish-reproduction,
stateless-path-unchanged, WF-DSR-untouched, cookie-isolates-two-clients, reset-is-logged.

**Pitfalls (subtle, all resolved in the spec):** distinct-count not click-count; floor not replacement;
full-sample-DSR-only; record *after* validation+success (typos don't inflate); seed is a trial knob;
reset visible never silent; route through `_jsonable` (no raw NaN in the API contract).

**Effort:** ~0.5–1 day. ~120 LOC module + ~40 LOC edits + ~14 tests.

---

### 4.2 Purge + embargo at walk-forward boundaries

**Why:** closes gap #2. The leak is **not** inside `backtest()` (its intra-slice `shift` is execution-lag).
It's that `service.py` computes weights once over the full panel and `walk_forward` slices them — so the
first ~L bars of each OOS fold are scored with prior-fold prices.

**Core (`research/walkforward.py`):** add `purge_bars` and `embargo_bars` params (default 0 → byte-identical
legacy behavior). For each fold, **flatten the contaminated leading `purge_bars+embargo_bars` signal rows to
0.0** (don't truncate `close` — that would fabricate a return discontinuity at every fold start), run
`backtest`, then **score only `oos.returns.iloc[n_purged:]`** (leaving the zero-rows in would inject structural
zeros that bias Sharpe toward 0). Report `purge_bars`, `embargo_bars`, `purged_bars_per_fold`,
`total_purged_bars` so the haircut is auditable. **Hard-error** if `drop >= smallest fold` (rather than
silently emitting NaN folds) — `service.py` already catches `ValueError` and degrades to the full-sample
fallback verdict.

**Wire the purge from the real factor lookback:** `service.py` already computes `_effective_lookback(factor,
lookback)` — pass it as `purge_bars` (and `embargo_bars=1`). For fundamental factors (no lookback) default
purge 0 but keep embargo.

**Decisive test:** build a signal whose *only* edge is a 1-bar look-ahead living on the fold boundary; assert
`clean.stitched_oos_sharpe < dirty.stitched_oos_sharpe` and that `total_purged_bars` matches
`folds × (purge+embargo)`. Plus: zero-purge reproduces legacy numbers exactly; purge-too-large raises;
point-in-time invariance preserved (purging is pure left-truncation, never a peek).

**Effort:** ~0.5 day. ~40 LOC + 3 wiring lines + 5 tests. Low regression risk (default path unchanged).

---

### 4.3 Square-root market-impact cost model (NOT "Almgren-Chriss" — see §3.2)

**Why:** closes gap #3. Concave participation-rate impact is the dominant real cost for high-turnover/large-
notional strategies; flat bps hides capacity limits entirely.

**New module `research/costs.py`** — two models, one contract (both return a per-period cost **Series** ≥0
aligned to the return index, so `backtest` just does `net = gross - cost`):
- `flat_bps_cost(turnover, bps)` — legacy, size-blind.
- `square_root_impact_cost(positions, adv_dollar, daily_vol, nav, eta, spread_bps, flat_bps, gamma=0.5)`:
  per name per bar, `u = |Δw|·NAV / ADV$` (participation); `impact_$ = eta·σ_daily·u^γ·dollars_traded`;
  `+ half-spread`; aggregate / NAV. Where ADV is missing/≤0, **fall back to flat bps on that name and FLAG it**
  (`fallback_mask`) — *missing ADV must never mean free trading*. Surface `max_participation` +
  `capacity_warning` (participation > 30% of ADV is not realistically fillable).
- Helpers `adv_dollar_from_panel(close, volume)` (= rolling mean of **shares×price**, `.shift(1)`) and
  `daily_vol_from_close(close)` (rolling std, `.shift(1)`).

**Integration:** extend `backtest()` with **keyword-only** `adv_dollar/daily_vol/nav/impact_eta/spread_bps`
(keyword-only so `walkforward.py`'s positional `backtest(close, signal, cost_bps, periods_per_year=…)` calls
don't break). Add `gross_equity` + `cost_model` + `cost_result` to `BacktestResult`. In `service.py`, capture
`panel.volume` (currently only `.close` is kept), compute ADV$/vol, pass them in; return a second
`gross_equity_curve` and the capacity flags. `Home.py`: overlay gross-vs-net curves; `st.warning` when falling
back to flat bps or when capacity is breached.

**Pitfalls:** **units** — ADV must be in **dollars** (`volume_shares × close`); mixing shares/dollars makes `u`
off by ~price and silently near-zeros impact (synthetic `data.py` volume is in shares). **Look-ahead** — both
`.shift(1)` in the helper *and* `.shift(execution_lag)` in `backtest` (a trade's cost uses ADV/vol known
yesterday; a trade's own volume must not inflate the ADV that prices it). **Don't double-count** spread + flat
bps. **Naming collision** — `gross` already means leverage in `service.py`; call the new curve
`gross_equity_curve`. η/spread/γ are **assumptions** → keep in `meta`, state in UI they're estimates.

**Effort:** ~1 day. ~120 LOC module + ~20 LOC touchy backtest change + wiring + ~10 tests (units, point-in-time,
fallback, concavity are load-bearing).

---

### 4.4 PBO via CSCV — the capstone honesty test

**Why:** closes gap #6 and is the natural consumer of the §4.1 trial ledger. Symmetric (IS/OOS roles swap), so
far more honest than the single 70/30 split.

**New module `research/overfitting.py`:**
- `cscv_pbo(returns_matrix, n_splits=8, thorough=False)` — input is a **(T×M)** matrix of per-period net
  returns (one column per config tried). Split into S contiguous time **blocks** (preserves
  autocorrelation/regime structure — do NOT shuffle rows), form all `C(S,S/2)` IS/OOS partitions (OOS = exact
  complement), select the IS-best, look up its OOS rank, **PBO = fraction with logit(rank) ≤ 0** (IS-best lands
  in losing OOS half). Returns the **OOS-Sharpe distribution + histogram** (the visceral "you cannot lie to
  yourself" artifact) plus a performance-degradation slope (OOS-SR on IS-SR; ~1 healthy, ≤0 overfit).
- **Speed:** precompute per-block sum/sumsq/n once → each combination is **O(S·M)**, not O(T·M). S=8 → 70
  combos (interactive); S=16 → 12,870 (thorough); both sub-second for M up to a few hundred. `MAX_COMBINATIONS`
  guard so a user can't request S=20 and hang the thread.
- `pbo_from_factor_grid(close, weight_fn, param_grid, …)` — PBO is undefined for M=1, so the default single-
  config UI path builds an honest competitor set by running the **same factor across a lookback grid** (the
  configs a user would have data-mined over), reusing the point-in-time `backtest()` so no guard is bypassed.
  Caption must say this measures *parameter*-overfitting, not strategy-zoo overfitting.

**Pitfalls:** **pooled variance** (sum/sumsq over selected blocks), **never** mean-of-block-Sharpes (Sharpe
isn't additive) — property-test it against `metrics.annualised_sharpe` (this is the critical test). **ddof=1**
to match the rest of the engine. **PBO direction:** `Pr(logit ≤ 0)`, relative rank = `rank/(M+1)` (avoids
`logit(1)=+∞`) — assert with noise→high-PBO and real-edge→lower-PBO. NaN warm-up rows poison block stats →
feed already-0-filled `res.returns`. Folding `pbo.overfit` into the headline `credible` flag is a correct,
on-brand one-way ratchet — do it only after the math is validated and update `verdict_basis`.

**Effort:** ~0.5 day. ~250 LOC + ~120 LOC tests; bulk of effort on the pooled-variance + direction proofs.

---

### 4.5 CVaR/ES + rolling-IC sign-stability (the two fast wins)

**Why:** closes gaps #4 and #5. Both are pure-function additions to files already imported/rendered → near-zero
integration risk.

**CVaR (`metrics.py`):** add `value_at_risk(returns, alpha)` and `conditional_var(returns, alpha)` (95% & 99%),
both **historical/empirical** (mean of realized worst-tail returns) and reported as **positive loss
magnitudes, per-period**. Add to `summarize()`. Plain-language `_cvar_verdict` compares the 99% ES to the
**Normal-implied** ES of the same vol to expose how much fatter the real tail is.

> **Critical pitfall:** CVaR here MUST be **empirical**, never parametric-Normal — the engine already prints
> "FAT TAILS — Gaussian VaR understates risk" on the same screen; a Gaussian CVaR would directly contradict it.
> The Normal number appears *only* inside the verdict as the baseline-to-beat. Do **not** annualise (tail risk
> doesn't scale by √T). Guard n<20 → NaN, not a fake number. Use `np.quantile(..., method="lower")` (never
> interpolate the threshold up into less-extreme territory).

**Rolling-IC stability (`signal_quality.py`):** add `ic_sign_stability(signal, close, n_windows=6)` — split the
per-date IC series into contiguous equal eras, take each era's mean IC, report the fraction of eras whose sign
matches the full-sample sign (`stable` if ≥80%). Fold into the `predictive` verdict as
`(stable is True or stable is None)` so a too-short series doesn't *block* an otherwise-passing signal.
**Add the flat keys to `compact_scorecard()` too** — `service.py:214` calls *compact*, not `scorecard`; adding
only to `scorecard` means the API/UI see nothing.

**Pitfalls:** IC series is far shorter than `len(close)` (warm-up + min-cross-section drops) → enforce
`n ≥ n_windows × min_obs_per_window` and return `stable=None` on short series (a false "UNSTABLE" is itself a
lie). Full-sample sign==0 → fall back to mean-of-era-means. CVaR sign convention: assert `cvar95 ≥ var95 ≥ 0`.

**Effort:** ~0.5 day total for both.

---

## 5. Tier-2 / Tier-3 roadmap

| Item | What | Notes |
|------|------|-------|
| **Factor crowding/correlation** (T2) | IC-weighted Spearman correlation matrix across the user's saved signals; flag >0.7 as redundant | High value once users run multiple signals |
| **Implementation-uncertainty interval** (T2) | Run strategy through 2 cost models, report the *range* of net Sharpe | Honest version of the arXiv finding (§3.1) — ambiguity in attribution, NOT alpha-sign flips |
| **HMM regime detection** (T3) | 3-state Gaussian HMM (bull/neutral/bear) on returns+vol; report IC & Sharpe *per regime*; regime-conditional equity curves | `hmmlearn`; answers "does this only work in bull markets?" |
| **Hierarchical Risk Parity** (T3) | Distance-transform clustering → quasi-diagonalization → recursive bisection; inversion-free, stable on small samples | `scipy.cluster.hierarchy`; differentiates on portfolio construction (§3.4) |
| **Barra-style factor exposure** (T3) | Regress portfolio returns on market/SMB/HML/momentum/quality → systematic vs idiosyncratic variance | Buildable from existing `factors.py` |
| **Restatement-vintage PIT audit log** (T3) | "Show me the exact value used on date T for ticker X" inspectable view | The #1 thing pro quants demand; beats Portfolio123's FactSet-backfill weakness (§3.4). Gated on real PIT data — see §7 |

---

## 6. Speed

Current pandas-heavy code is fine for research scale. When the universe hits 500+ names / 20+ years:

1. **Numba JIT the backtest turnover hot path** (`pos.diff().abs().sum(axis=1)`) — 10–50× on large panels.
2. **Cache IC series** keyed on a hash of `(signal, close, horizon)` — currently recomputed every call.
3. **Async LLM extraction** in `signals_llm.py` — `asyncio.gather()` on batched Anthropic calls, ~5× latency
   cut on multi-document runs.
4. **PBO already designed for speed** (O(S·M) per combo, §4.4); if it ever needs more, `joblib.Parallel`
   over combinations is embarrassingly parallel.

---

## 7. Open inputs still needed

1. **Real PIT data vendor** — the data-tier ladder is set (synthetic+yfinance → SimFin → Sharadar →
   S&P Capital IQ/Compustat). The §4.5 audit log / §5 restatement-vintage work becomes a *real* feature only on
   a PIT-correct source; on synthetic/yfinance it's demo-only. **Decision needed:** wire SimFin first as the
   budget PIT tier, or build the audit-log scaffolding now against synthetic and swap the source later?
2. **Daily-factor vs monthly-fundamental users** — ✅ ANSWERED: **specialize on the
   fundamental/quantamental quant, one shared engine.** See [strategic_focus.md](strategic_focus.md).
   This re-prioritizes §8 below (market-impact cost demoted; PIT-vendor wiring + quantamental promotion
   become critical path).
3. **η / spread / γ defaults for the cost model** — defaults (η=0.5, spread=2bps, γ=0.5) are defensible but
   asset/venue-dependent. If you have any execution data or a target asset class, it sharpens these.
4. **AFML book** — your CPCV/HPV guide covers PBO/CSCV; the book's HRP chapter (Ch. 16) would sharpen the
   Tier-3 HRP. Implement from your guide + papers unless you can get the book.

---

## 8. Priority order (reconciled with the focus decision)

Per [strategic_focus.md](strategic_focus.md), the ICP is the **fundamental/quantamental quant**. That
reorders the work: the honesty items that serve *this* buyer stay high; the market-impact cost model
(daily-buyer only) is **deferred to the expansion trigger**; and two *product/data* items that the
improvement research treated as roadmap become **critical path**, because without real data the wedge
has nothing to run on.

**Critical path first (unblocks the paid wedge — see strategic_focus.md §"Build sequence"):**

| # | Item | Effort | Why this slot |
|---|------|--------|---------------|
| C1 | Wire one real PIT vendor (Sharadar SF1; SimFin fallback) behind `fundamentals.build_fundamentals` | 1–2 d | THE gate — `service.py:170-185` refuses real fundamental data today; nothing converts until this ships |
| C2 | Promote `quantamental.build_signal_panel` to a first-class factor type (api + UI front door) | 1 d | The actual moat is stranded in `examples/`; make the AI-signal→PIT-backtest→verdict loop self-serve |

**Then the honesty layer (this is where §4's specs land), in this order:**

| # | Item | Effort | Why this slot |
|---|------|--------|---------------|
| 1 | Trial-count ledger → auto `n_trials` (§4.1) | 0.5–1 d | Brand-defining; closes the biggest loophole; feeds #2 |
| 2 | PBO via CSCV (§4.4), tuned for the low-sample monthly regime | 0.5 d | Headline differentiator vs Portfolio123; the buyer's proof artifact |
| 3 | Purge + embargo at WF boundaries (§4.2) | 0.5 d | Correctness; matters *most* in the low-sample monthly regime (overlapping labels) |
| 4 | CVaR/ES + rolling-IC stability (§4.5) | 0.5 d | Two fast wins, near-zero risk; rolling-IC catches era-flips in slow signals |
| 5 | Harden AI-signal validation gate as a mandatory pre-count check (`signals_llm` + `signal_quality`) | 0.5 d | Concentrated bet → a weak extraction must read UNCERTAIN, never alpha |

**Deferred to the expansion trigger** (≥10 paying users AND co-founder onboarded for intraday/impact):

| Item | Why deferred |
|------|--------------|
| Square-root market-impact cost (§4.3) | Serves the daily buyer; cheap & correct *eventually*, not now. Note: for low-turnover monthly fundamental strategies the existing flat-bps model is adequate — the lie it tells is a *daily/high-turnover* lie. |
| Capacity/ADV haircuts, intraday data, fast-IC-decay tooling | Daily-buyer only |

Net near-term: C1 → C2 give the wedge real data + a self-serve loop; items 1–5 (~2.5 days, all in
`research/` + existing seams) make the verdict trustworthy. Market impact waits.

---

*Verification provenance: 6 code-audit agents (each read the cited source file), 6 web-verification agents
(adversarial, sources cited inline above), 5 spec-deepening agents. Three external claims were corrected
(§3.1–3.3). All six code claims confirmed.*
