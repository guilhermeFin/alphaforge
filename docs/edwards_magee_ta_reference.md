# Edwards, Magee & Bassetti — Technical Analysis of Stock Trends (9th ed.)
## Master Reference for AlphaForge

> Synthesized June 2026 from a full read of all 835 pages. Raw per-section notes (full detail,
> including worked examples and figure-level rules) live in `books/edwards_magee_notes/chunk_00..09.md`
> in the Claude Brain folder. This document is the deduplicated, implementation-grade synthesis:
> every rule stated so it can be coded, plus the AlphaForge application map (§9).
>
> Notation: **QUANT** = directly codable rule. All "3%/2%" thresholds are close-based unless noted.

---

## 0. The book's own epistemology (matches AlphaForge's honesty brand)

- Patterns are recurring crowd psychology; no new patterns since the 5th edition; they recur 1929 → 1962 → 1987 → 2000 because humans don't change.
- Every signal is a **probability statement, never a certainty**: "There is no such thing as a sure-fire method of beating the market... there never will be" (Edwards, 1948). Magee's bean-bag: 700 white/300 black justifies drawing for white even after 10 black draws.
- The authors' own anti-overfit warnings: no mechanical index "always, automatically, without ever failing" works; "all systems work beautifully at least twice in their lives: in research, and in huge monumental Bull Markets"; mechanical-system performance decays as markets counter-adapt (150-day MA crossover on the Dow worked until ~1980s, then stopped).
- Each trade = "an experiment made to confirm a probability. The experiment is ended quickly if the trend does not develop."
- Regime-gating is endorsed by the book itself: MAs work in trending markets and whipsaw in ranges; oscillators work in ranges and mislead in trends. Classify regime first.
- Ch. 39 prescribes the AlphaForge workflow verbatim: record actual AND theoretical trades → re-run rule changes against historical chart records → adopt only after consistent advantage → forward-test. (Their case: a rule revision turned 30 bear-phase trades from −40%/yr to +156%/yr — and they still called it "not conclusive.")

## 1. Core premises and trend taxonomy

Premises: (1) price discounts everything (including insider knowledge); (2) prices move in trends; (3) volume goes with the trend; (4) a trend in force is presumed to continue until definite evidence of reversal — **continuation is the null hypothesis**.

| Trend | Duration | Magnitude |
|---|---|---|
| Primary (Major) | usually ≥ 1 year | > 20% |
| Secondary (Intermediate) | 3 weeks – 3 months (rarely more) | retraces 1/3–2/3 of prior primary swing; mode ≈ 50%; seldom < 1/3 |
| Minor | usually < 6 days, rarely up to 3 weeks | noise; ≥ 3 minor waves per intermediate swing |

- **QUANT Secondary classifier:** counter-primary AND duration ≥ 3 weeks AND retrace ≥ 1/3 (pivot-to-pivot, minor noise ignored).
- Bull phases: accumulation → markup (the trader's harvest) → excess (new-issue surge, junk outperforming quality, "air pockets" inside the uptrend). Bear phases: distribution → panic (vertical, climactic volume, days-to-weeks) → discouraged selling (quality falls last). Phases can be skipped; no fixed durations.
- Asymmetries to encode: bear markets ≈ half the duration of bulls and steeper; bulls start slow and accelerate; bears start fast and taper. Stocks rise ~2/3 of calendar time but fall faster.

## 2. Dow Theory (the original trend-state machine)

The 12 tenets, with the essential subset = {three trends, primary, secondary, minor definitions, **two averages must confirm**, **lines**, **closing prices only**, continuation-until-reversal}:

1. Averages discount everything.
2–5. Trend taxonomy as in §1.
6–7. Bull/bear phase anatomy.
8. **Confirmation gate:** no valid signal from one average alone. Signal date = the LATER close-basis breakout; lags of days to ~2 months are normal. Divergence = negative information only — it can never produce a positive signal; most major reversals occur with the averages in agreement.
9. **Volume goes with the trend** — collateral evidence only, judged over multi-week windows, never 1–few days. Conclusive signals come only from price.
10. **Lines:** sideways range ≥ 2–3 weeks with close-range width ≤ ~5% of mean price. Close outside range = directional signal. Longer + narrower = more significant; follow-through tends to exceed swing-pivot break signals.
11. **Closing prices only**; any penetration counts (even 0.01) — the 1.00-point school lost the 1946 test by ~13 Dow points.
12. Continuation is the null; act on signals immediately, never before. Confidence decays with each successive reaffirmation; reset all levels on each reaffirmation (only the latest secondary extreme is live).

**Signal templates:** Bull = both averages take out the prior intermediate rally top (closes), survive a secondary test, then both exceed the recovery high. Bear = after a reaffirmed bull, both close below the latest secondary's lows.

**QUANT record (book's tables):** $100 in 1897 → ~$345–362K by 2005 long-only (vs. $39,685 buy-and-hold); long+short ≈ $1.9M. ~1 round trip per 2 years; out of the market ~37% of days (structural drawdown reduction). Reference drawdowns avoided: 1929–32 −89%, 1987 −41%, 2000–02 −39%.

**QUANT Colby mechanical proxy** (channel-breakout approximation of Dow Theory — vet before use):
- Enter long: INDU 9-day high AND TRAN 39-day high.
- Exit long / enter short: INDU 22-day low AND TRAN 166-day low.
- Cover short: INDU 36-day high AND TRAN 32-day high.

**Bassetti modernization:** Dow+Transports no longer represent the economy; determine the Broad Market Trend from **DJIA + S&P 500 + NASDAQ jointly** (3-index harmony = safe; disagreement = mixed/hedge). Averages need no 3% breakout margin — any close-through counts on indexes.

## 3. Chart construction

- Data: daily H, L, C, V (open deemed insignificant by Edwards; needed only for candlesticks). Close is privileged.
- **Semilog scale preferred** (equal distance = equal %); pattern geometry and especially trendlines/measuring rules work better in log space. Speculative stocks trend straight on log; heavy investment issues straight on arithmetic — per-issue calibration from history.
- Weekly/monthly charts: for major S/R and long trends; **volume confirmation may be relaxed on weekly/monthly completions**. Bottoms take longer than tops — use weeklies for bases.
- Ex-dividend: shift pattern boundaries, S/R levels, stops down by the dividend amount; ex-div gaps are not technical gaps.
- Splits: rescale; patterns/S-R survive on adjusted basis.

## 4. Pattern library (full spec sheets)

**Universal rules first (apply to every pattern):**
- **QUANT Breakout confirmation = close beyond boundary/neckline by ≈3% of price** (stocks; 2% suffices on the averages; may accumulate over 2–3 days). A 2% close + conspicuous volume + weak close near the extreme of the day also qualifies.
- **QUANT The master volume asymmetry: UP-breakouts require a conspicuous volume burst to be valid; DOWN-breakouts are valid on any volume** ("stocks can fall of their own weight, but it takes buying to put them up"). Heavy volume from the START of a down-break out of a triangle apex is actually a shakeout flag (2–3 day trap, then reversal).
- **QUANT Target cap:** every measured objective ≤ the extent of the prior move being reversed ("a reversal must have something to reverse"). All pattern targets are MINIMUMS, not maximums; live trendline/S-R/volume evidence outranks static targets.
- **QUANT Proportionality:** bigger pattern (price span × duration × turnover) → bigger implied move.
- Pullback (after down-break) / throwback (after up-break): frequent — H&S necklines pulled back in "the great majority" of cases (often twice); rectangles ~40% within 3 days–3 weeks; triangles usually within 2–3 days. More likely when the general market opposes the breakout direction. Premature breakout (re-exits same direction) vs. false breakout (re-exits opposite) are **indistinguishable in real time** → confirmation filters + stops, not prediction.
- **QUANT Failed-signal reversal rule:** return of price to the formation's origin (gap origin, runaway-day low, breakout point) marks the signal false AND is itself a tradable signal in the opposite direction. "Failed signals are often excellent signals for a trade in the other direction."

### 4.1 Head-and-Shoulders Top (most reliable major reversal)
- Elements (all required): LS rally on very heavy volume → reaction; head exceeds LS high, its reaction low undercuts the LS top; RS rally on decidedly less volume fails below head; **close 3% below neckline** (line through the two reaction lows).
- Volume: LS heavy; head heavy (LS>head in ~1/3, equal ~1/3, head>LS ~1/3 — a warning, not a requirement); RS dull = the tip-off. Downside break needs NO volume; the volume burst usually arrives after the pullback.
- Neckline slopes: horizontal tendency; up-sloping valid only if head-reaction low is appreciably below LS top; down-sloping = weaker but part of the move already spent.
- **QUANT Target: head-to-neckline height projected down from the break point; governing minimum = min(target, prior advance).**
- ~20% of forming H&S tops are "saved" before the neckline breaks; false breaks after a decisive penetration are extremely rare. A failed H&S = warning the genuine top is near.
- Tactics: short entry = first of {40% retracement of the breakout move (recomputed to its lowest point), rally to a line across head+RS tops, pullback to neckline}. Decisive breaks: market-out next day beats waiting for stops.

### 4.2 H&S Bottom ("Kilroy")
- Mirror in price, NOT in volume: rally from head must show volume pickup; **upside neckline penetration REQUIRES a conspicuous volume burst + 3% close — both, or don't trust it**.
- Bottoms are longer, flatter, more rounded than tops. Same target formula. Buy = first of {40% reaction of breakout move on decreasing volume, line across head+RS bottoms, throwback to neckline}.

### 4.3 Complex/Multiple H&S
- Shoulders/heads doubled; near-perfect left-right symmetry is the norm; necklines nearly always horizontal; confirmation at the OUTER neckline (3%).
- Same minimum formula but slow starters that "very seldom exceed the bare minimum" quickly — at intermediate turns, take profits at the measured minimum. As reliable as the simple form.

### 4.4 Rounding turns (bowls/saucers)
- Gradual arc; no neckline, **no measuring formula** (cap = prior move). Bottom volume traces a concave bowl reaching extreme dullness at dead center; "they almost never deceive."
- Known trap: premature 1–2 day vertical breakout shortly after dead center → falls back, rounding resumes (don't chase).
- Rounding tops: volume stays high/irregular — rare, mostly in high-grade issues. Dormant-bottom variant in thin stocks = accumulation alert.

### 4.5 Triangles
- **Symmetrical:** ≥4 touches (2 per converging boundary, drawn on intraday extremes); volume shrinks toward apex (high/irregular in-pattern volume = distrust). **QUANT: ~3/4 resolve as consolidation**; direction unknowable pre-break; the worst false-move offender (>2/3 behave; up to 1/3 misbehave).
- **QUANT Apex timing rule: best breakouts occur 1/2–3/4 of the way from base to apex; past 3/4 the pattern is void** (drift to apex → discard; "end runs around the line" after apex stalls).
- Targets, two methods: (a) line from the first-reversal corner parallel to the opposite boundary (in log space); (b) post-break speed ≈ pre-pattern trend speed; rough expectation = move ≈ the prior move. Apex level becomes strong S/R afterward ("cradle"); **stops always go beyond the apex level**.
- **Ascending (flat top + rising lows):** bullish, ~9:1 eventual upside; up-break needs volume. **Descending (flat bottom + falling highs):** bearish; down-break needs none; "you cannot count on a pullback after a support break" — act immediately. Ex-div: shift boundaries down. Triangles on monthly charts: dismiss.

### 4.6 Rectangles
- Horizontal S/R lines, ≥4 reversals; volume shrinks as it lengthens; ~3/4 consolidation; more reliable than symmetrical triangles, less powerful than H&S; more common at bottoms.
- **QUANT Target: pattern height from breakout point (minimum).** Long/narrow ones hesitate at the minimum. Range-trading the edges is viable from the 5th (better 6th) reversal if band ≥ ~8–10% of price.

### 4.7 Double / Triple Tops & Bottoms
- True ones are RARE and only exist on confirmation — never at the moment of the second peak.
- **QUANT Double Top spec:** peaks > 1 month apart (typically 2–3+); valley depth ≥ 15–20% of price (10% explicitly insufficient for primary import; time outweighs depth); peak equality within 3%; second peak on lower volume; **confirm only on close below valley low** → target = peak-to-valley distance projected below valley. Characteristically a PRIMARY reversal phenomenon.
- **Double Bottom:** second bottom dull and rounded; **confirm = close above the intervening rally high on marked volume**; intervening rally ≥ 15%; ≥ 6 weeks between bottoms preferred.
- **Triple Top:** volume monotonically declining peak1>2>3; confirm below the LOWER valley. Confirmed triple bottoms "seldom fail"; unconfirmed ones are treacherous.

### 4.8 Broadening formations
- Diverging swings + HIGH, irregular volume throughout = crowd out of control; habitat = late bull markets. **QUANT: bearish 9 times out of 10; broadening BOTTOMS do not exist** (never implement a bullish mirror).
- **Orthodox Broadening Top:** 5-point reversal (3 rising peaks, 2 declining troughs, points ≤ 2 months apart); **complete on close below the second bottom (point 4)**; on an AVERAGE any close-through counts (1957 + 1929 + 2000 Dow examples). Post-break pullback in ≥ 4/5 of cases, typically ~50% retrace, invalidation ~67% — short the pullback, stop above point 5.
- Right-angled (flat-topped/flat-bottomed): bearish either way; flat-bottom variants "almost always break down"; bullish exception only on a 3% high-volume close through the flat top. "Every H&S begins as a broadening formation."

### 4.9 Diamond
- Broadening first half → converging second half with dying volume; needs active markets (major tops). Don't force H&S into diamonds. **Target = max pattern width from breakout.**

### 4.10 Wedges
- Both boundaries slope the same way, converging; duration **> 3 weeks (else pennant) to ~3 months max**; price penetrates ≥ 2/3 toward apex before breaking.
- **Rising wedge = bear-market-rally hallmark** (frequent rising wedges after a big decline = primary trend still down); breaks fast; **target = full retracement of the wedge**. Falling wedge breaks up but then drifts/saucers — no urgency. Verify multi-month wedges on semilog (arithmetic-scale artifacts fake giant wedges).

### 4.11 Flags & Pennants (most reliable continuation patterns)
- Compact counter-trend parallelogram (flag) or converging mini-triangle (pennant) hanging from a near-vertical "mast" on big volume.
- **QUANT The three reliability tests (all required): (1) preceded by a straight-line move; (2) volume shrinks constantly and markedly throughout; (3) breakout within ≤ 4 weeks (> 3 weeks = suspect).** In-pattern volume spike → put a tight stop under that day's close (failing pattern stops you out; genuine breakouts don't).
- **QUANT Half-mast target: mast length (from its breakaway point — pattern/trendline/S-R break, marked by the volume spurt) projected from the flag's breakout, in log space.** After two masts, stand aside.
- Regime stats: creatures of the LATE dynamic bull phase (their proliferation = advance nearing final weeks); bear-panic flags complete in 3–4 days; ≥4-week "flags" with rising-volume rallies in late bears are traps. Cannot exist on monthly charts, barely on weeklies.
- Scallops (repeated saucers in low-priced actives): gain 10–15%/saucer, dip 20–30%, 5–7 weeks each; first failed saucer = the top signal; buy the saucer bottom, never the late-stage volume peak.

### 4.12 One-day events
- **One-Day Reversal:** volume ≫ any session for months, 2–3% intraday traverse, close ≈ open. Minor-trend significance only; an exit-timing gauge, not an entry pattern.
- **QUANT Key Reversal Day: new high in an upmove + close below the PRIOR day's close** (mirror at lows). Tactics: short the close, stop just above the KRD high. Needs context filters (raw frequency too high in bulls).
- **Runaway day:** range 2–3× normal, open near one extreme, close near other. Return of price to its base = bull trap → reverse.
- **Spike:** classify after the fact; importance ∝ prior move length, close position, prominence.
- **QUANT Exhaustion triad scoring (exit logic): {One-Day Reversal, exhaustion gap, exceptionally heavy volume day late in a move} — any one = watch; any two = strong; all three = act.**
- **Selling Climax:** panic-scale decline → big downside gap open → late-morning dry-up → snap-back. Historically does NOT mark the final bear low (one exception in the entire record: April 1939) — expect the climax low to be retested/broken. Buy only for a quick trade.

### 4.13 Gaps
- Filter first: gap > typical daily change for the issue; exclude habitual gappers and ex-div gaps. **Gap-fill null hypothesis: P(fill) = P(revisiting any traded range) — there is no special "gaps must close" edge.**
- **Common (intra-pattern):** nil significance; many of them inside a pattern → leans consolidation.
- **Breakaway (at breakout):** validates the breakout — **false moves are seldom attended by gaps** (of two simultaneous breakouts, buy the gapped one). No measuring implication. Fill odds: heavy volume on the FAR side → near-term fill remote (throwback stops at the outer gap edge); light far-side volume → ~50–50.
- **Runaway/Measuring (mid-move, in thin "easy" territory):** **QUANT target: distance from move origin to gap, projected beyond the gap (log space; haircut the target; use for exits, not entries).** Stays open until a major/intermediate counter-swing. Two gaps → midpoint between them; each successive gap → more suspicion of exhaustion, especially if wider.
- **Exhaustion (at the end):** test on the day after — huge volume + stalling price; "virtual certainty" if a reversal day closes near the gap edge. Seldom the FIRST gap of a move. **Closes within 2–5 days (the discriminator vs. runaway).** Means "stop," not "reverse."
- **Island reversal:** exhaustion gap + compact high-volume range (1 day–1 week) + breakaway gap at overlapping levels → full retracement of the preceding minor move. Stop just beyond the island.
- One-day gap collapses ("air pockets", −40% days): no prior chart evidence as a rule; exit immediately, rallies are exits, frequently ends in delisting.

## 5. Support & Resistance (E&M's distinctive theory)

- **The core inversion vs. street lore: support/resistance is created by VOLUME TRADED AT A LEVEL, not by swing extremes.** Overhead resistance = prior high-volume BOTTOM zones (trapped buyers waiting to get out even); support = prior high-volume TOP zones (regretful sellers waiting to reinstate). "It is much easier for prices to push up through a former top level than through the resistance set up at a previous volume bottom."
- **QUANT Potency = f(volume at the level [sum across bottoms, discounted], distance traveled beyond it [>10% below for $20–35 stocks; rule breaks down for low-priced issues], time elapsed [1–2 yrs > 4–5 yrs, but old zones survive if never attacked], attack count [3rd attack at a level is odds-on to succeed], no capitalization changes since).** Safer to overestimate resistance than under.
- **QUANT Zone axis = mean of CLOSES of the congestion days** (volume-weighted center of gravity for stop math). Lookback: weekly chart spanning ≥ one full prior bull+bear cycle.
- Role reversal: broken bottoms become tops and vice versa; broken necklines, triangle apexes ("cradle" strongest, decays as time passes the apex), pattern boundaries, prior minor tops in a zigzag — all S/R.
- Regime asymmetry: established-bull reactions hold the TOP edge of support zones; established-bear rallies fail at the BOTTOM edge of resistance zones or short of them. **Panic phases ignore ALL support** — disable support-bounce logic in panic regimes.
- Round numbers (20, 30, 50, 75, 100) act as resistance in new-high ground (no vested interest required); weaker in seasoned large-caps.
- Minor support break (especially with volume after the break — the burst usually comes AFTER penetration) = first step of intermediate reversal; intermediate support break = first sign of major reversal.
- A distant resistance is a maximum possibility, not a target; prefer the stock with the thinner overhead track.

## 6. Trendlines, channels, fans, moving averages

### Trendlines
- Up-lines connect minor-reaction LOWS; down-lines connect minor-rally TOPS. Two confirmed pivots to draw; never bottom-to-top; never through price. Major bull lines start from the FIRST INTERMEDIATE BOTTOM after the absolute low (H&S bottoms: from the right shoulder).
- **QUANT Authority = touch count (3rd touch confirms; weights more than duration) + duration (origin pivots ≥ ~1 week apart) + flatness (flatter = more authoritative; steep lines break trivially).**
- **QUANT Break validity: close beyond by ≈3% of price (2% on averages; cumulative over 2–3 days OK), OR 2% + conspicuous volume + close near the day's extreme. Intraday-only breaks are void. High-volume intraday crash that closes back above the line = shakeout (alert, no action).** Volume expansion is NOT required for downside trendline breaks but usually appears.
- Amendment logic: indecisive 3rd-touch break → redraw on pivots 1+3 (or 2+3); confirmed (3+ touch) lines survive indecisive breaks; 3rd bottom well above the line → run old and new lines in parallel for weeks. Closes often make better lines than intraday extremes.
- Double trendlines (outer shake-line + inner dull-line, ~3% apart): trend ends only on OUTER-line break.
- Pullback to the broken line occurs in the great majority of normal-slope cases and is stopped more exactly than pattern pullbacks → second-chance entry/exit.
- **Channels:** return line parallel through the intervening extreme. **QUANT Failure symmetry: the margin by which a rally fails to reach the return line ≈ the margin by which the ensuing decline penetrates the basic line** (mirror for upside). A pierced return line gives NO support.
- Major DOWNtrends do not yield useful straight trendlines (1929–32 was the unique exception) — use S/R, patterns, basing points instead. **End-of-bear detector:** after ≥1 panic, an orderly multi-touch declining channel whose upside break = candidate major turn.
- **QUANT Fan principle (corrective moves ONLY): three successively flatter lines from the correction's origin; the upside break of fan line 3 = the correction low is in.** Broken fan lines must hold as support. Never apply after a primary reversal has been signaled.
- Index tactical rule (Bassetti): **~3-month index trendline broken ≥2% preceded every historic crash (1929/1987/1998); exit or hedge on the break; expect a retest pullback** (belief dies hard). Magee: 2% index trendline break = liquidate longs. May be vol-scaled to 3%+.

### Moving averages (the "automated trendline")
- SMA preferred ("work just as well and sometimes better"); canonical lengths 50d and 200d (consensus-watched, partially self-fulfilling); 10–20d for sensitivity; commodities use much shorter (3d/6d); "200 is a parameter, not a law" — fit to the market.
- Dual-MA endorsed (9/18 in the futures appendix: buy 9 over 18 both rising; sell mirror). MA-of-highs/MA-of-lows channels endorsed.
- Crossover rules + the reliable variants: buy the successful test of a rising 200d (touch-no-break); sell the failed test of a declining MA; price far below a declining MA = rebound trap, not a buy. Flat MA + broad range = whipsaw regime, stand down. The MA penetration that matters is the one coinciding with a pattern/trendline break.
- Inside congestion patterns the MA runs through the middle = noise. MA = adjunct, subordinate to patterns/trendlines/S-R.

## 7. The Magee mechanical system (basing points, stops, sizing) — AlphaForge's exit/risk module

This is a complete, codable trade-management system; the editor calls Fig. 210.1 "possibly the most important chart in the book."

### 7.1 The stop table (protective stop distance, % beyond the basing point)
**QUANT stop% = 5% × (NormalRangeForPrice(price) ÷ 15.5) × SensitivityIndex**, floor 5%, tabulated:

| Price | Conservative (SI<0.75) | Median (0.75–1.25) | Speculative (>1.25) |
|---|---|---|---|
| >100 | 5% | 5% | 5% |
| 40–100 | 5% | 5% | 6% |
| 20–40 | 5% | 5% | 8% |
| 10–20 | 5% | 6% | 10% |
| 5–10 | 5% | 7% | 12% |
| <5 | 5% | 10% | 15% |

(Modern substitution: implied/realized vol bands for SI: <0.40 / 0.41–0.79 / >0.79.)

### 7.2 The Three-Days-Away rule + Basing Points (codified algorithm)
1. A candidate bottom = the lowest-low day of a reaction (after a new high).
2. **Confirmation: three days (need not be consecutive) whose ENTIRE daily range sits above the candidate day's HIGH, all occurring before any lower low. A lower low voids the candidate.** (Tops: mirror — 3 full ranges below the high day's LOW.)
3. On confirmation: candidate = Basing Point; **stop = BP_low × (1 − stop%)**.
4. **New-high advance: a close ≥3% above the previous minor peak → new BP = the LOW of that new-high day**; advance the stop accordingly. (For parabolic runaways, additionally ratchet on each new percentage high.)
5. **Stops are NEVER lowered** (longs) / never raised (shorts). Sole exception: ex-dividend adjustment.
6. BP quality filters: prior leg ≥ ~15%; retracement ≥ 40% of the leg preferred (or ≥1 week sideways with correct volume); volume must SHRINK during the reaction; a flag can serve as a BP; a 3-week narrow congestion becomes the BP.
7. **Long-term variant: substitute "three BARS away" on weekly charts (~the investor's 4-week basing point)** — ran NASDAQ-100 long 1991→2001, reversed short, re-longed at the 2002 bottom. Stops ~5% under weekly BPs. Avoid stops exactly under the prior close/obvious lows (stop-hunting).

### 7.3 Volume exit overrides + progressive stops
- **The three legitimate heavy-volume days: breakout day, new-ground day, move-completion day. Extra-heavy volume on ANY OTHER day = end-of-move warning** → switch to progressive stops.
- **Progressive stop: next morning, day-only stop 1/8 point (1 tick; decimal offset to be re-optimized — candidates 9–12.5¢) below the prior close; re-place under each successive close until caught.** Exempt day: the breakout through the prior minor top itself; heavy volume on any LATER day → tighten.
- **Climax rule: volume ≫ every prior minor peak's volume = blow-off → exit and stand aside for a full intermediate correction (~40%+ of the whole move).** Climactic volume is mistaken for a continuation signal — that's the common fatal error.

### 7.4 Entries (trendline tactics, Ch. 29/30/34)
- Trade only with the Major Trend; if the averages disagree, only in stocks of the still-trending component; require group (sector) trend agreement AND the individual chart's signal. **All entries on the reaction/rally AFTER the signal, never chasing the breakout** (exception: market-order shorts immediately on a primary bull→bear signal).
- Reaction-entry reconciliation: expected reaction = 40–50% of the prior move (≈45% typical), to the basic trendline, and to the prior minor top — **buy at whichever indicates the SMALLEST reaction; stop beyond the LARGEST.**
- **Three-step rule: after a breakout, at most two reaction-buys; after the third step up, expect the intermediate correction.**
- Get-out list (any of): adverse pattern breakout (H&S/triangle/rectangle/diamond/wedge/flag), new adverse minor extreme, heavy-volume ODR or gap against you, island after a favorable move, support/resistance penetration, basic trendline break with nothing else favorable. A sell signal is NOT automatically a short signal — only completed reversal patterns justify reversing.
- Countertrend trades: targets limited to the correction of the preceding intermediate move; sized as insurance.

### 7.5 Position sizing & portfolio risk
- **QUANT size = (equity × risk%) ÷ (entry − stop); risk% = 2–3%** ("the size of risk per trade is directly correlated to equity volatility"). Stop distance 5–8% off price. Minimum expected move to justify any trade: 15%.
- **Overtrading math: ±40% risk per trade → 10 wins+10 losses in any order leaves <$100 of $1,000; ±8% → $937.** Budget for 10 consecutive stop-outs at major turns (10 months max signal lag observed).
- Caps: mania-class names ≤ 5–10% of capital; 8–10 positions max for an active account; lose 50% of capital → stop trading.
- **Composite Leverage (portfolio heat): CL = Σ(SI × NormalRange(price) × position$) ÷ (15.5 × total capital).** Run highest in the mid-trend, lowest at turns; contrary positions subtract (the formal basis of hedged books). Worked extremes: margined speculative preferred = 693% vs. cash high-grade preferred = 11.8% — 59:1 risk for the same dollars.
- **Normal Range-for-Price curve (volatility ∝ 1/price; from 3,800 charts, % range per ~6 months):** $1.25→38, $2→31.5, $3→27.6, $5→24, $10→19.8, $20→16.5, $25→15.5, $40→13.7, $50→12.7, $75→11.3, $100→10.4, $150→9, $200→8, $250→7. Price level is itself a volatility factor and sizing input.
- Sensitivity Index (Magee's beta, from 52-week ranges normalized by NRfP and the Market Reciprocal): stable per issue across cycles (volatility-class persistence is cross-sectionally testable). Classes: ≤0.5 very conservative; 0.5–1.0 conservative; 1.0–1.5 speculative; >1.5 highly speculative.
- Risk accounting: Operational $Risk = (price − stop) × shares (the technician's true risk); theoretical vol risk is for screening. Portfolio Operational Risk Factor = Σrisk ÷ capital, computed daily; stress at 2σ/4σ/6σ assuming stops DON'T fill. Prefer max-drawdown metrics over Sharpe ("severe deficiencies" — vol ≠ risk; Mandelbrot: Gaussian models discard the 5% of experience containing the storms). **Backtested max drawdown × 3–4 = required capital** for an unproven system.
- Rotation: high-grade leaders top FIRST and decline steeper than they rose; low-priced speculatives skyrocket late and top months AFTER the leaders. Trade quality early-bull, junk late-bull; short quality first in a bear.

### 7.6 The Evaluative Index / Rhythmic Investing (breadth-driven exposure dial)
- **QUANT MEI: chart ~100 stocks; weekly mark each + (bullish major trend) or − (bearish); Index = % bullish.** Scale net exposure continuously to the reading (census 30% bull/30% bear/40% neutral → 30% long/30% short/40% cash, risks balanced via CL). No binary signals; it "automatically withdraws you from a deteriorating market."
- **QUANT MEI extremes: all major bottoms 1965–82 printed ≤5% strong; a bottom at a HIGHER low (8–9%, June 1982) = secular regime change signal.** Exceptionally high readings coincide with broad tops.
- The 1929 breadth lesson (676-stock study): only **27%** of stocks topped with the averages (184 Aug–Oct 1929); 262 were already in downtrends; 44 kept making highs after. Identical dispersion 1999–2000. **Index-level signals are weak proxies for stock-level trend states; breadth divergence is THE constant of major tops.**
- Natural hedge: long the uptrending index, short its downtrending members (or correlated proxies); never short a stock in a confirmed uptrend.

## 8. Secondary material (futures, indicators, Turtle)

- **Futures differences:** contract life ~18 months → long-term S/R meaningless; ~80% of volume is hedging → S/R weaker, seasonality real; no float → volume/OI supplementary; weather shocks. Patterns: H&S/rounding/trendlines work (trendlines BETTER than stocks); triangles/rectangles/flags less reliable; gaps weaker except limit-move gaps. Leverage ≈10× → stops existential. Commodity trendline rules: invalid only after TWO closing breaks; ~45° lines most reliable; >4-week lines persist.
- **Donchian/Turtle (full spec in chunk_08 notes):** S1 = 20-day breakout entry (skip if last breakout won; 55-day failsafe), S2 = 55-day all-taken; N = 20-day Wilder ATR; **unit = 1% equity per N**; pyramid every ½N to 4 units; **stop = 2N (2% risk)**; exits = 10-day (S1) / 20-day (S2) opposite extreme; correlation-tiered unit caps (4/6/10/12); **equity cut 20% per 10% drawdown**. Bassetti: channel systems "self-adjust to market rhythm" but grind capital sideways — add a regime filter.
- **Indicator parameter sheet (Appendix C; subordinate to chart context — "a stochastic signal absent a pattern/trend violation is likely false"):** MACD 12/26/9 (0.15/0.075/0.20; Appel buy-side 8/17; OB/OS ±2.50 S&P scale); RSI 14 (70/30; swing-failure W; 3-day = noise); Stochastics 20-day %K/%D smoothed 3/3 (80/20; length = half the observed cycle); Bollinger 20d/2σ (10d/1.5σ, 50d/2.5σ); ADX > 25 = trending (the regime gate); OBV; price/volume/OI 4-state table; McClellan −150 climax / +100 buying pulse. **Multicollinearity guard: combine one indicator from closes + one from volume + one from ranges — never stack three close-derived oscillators.**
- **Gambler's ruin (risk-of-ruin module): P(ruin) = [(Q/P)^A − (Q/P)^C] / [(Q/P)^A − 1].** With an edge, bet small relative to capital; without an edge, the only "optimal" play is maximum bet (why exchanges set limits).
- Options context: 60/30/10 closed/expire-worthless/exercised; ~90% of retail buyers lose; structural edge to sellers. Covered calls ONLY in confirmed ranges. Vol mean-reverts: sell rich vol, own coiling vol. Bassetti's quant thesis: **quantitative analysis games model-anchored markets (options) successfully; it does not predict directional stock movement** — behavioral markets belong to charts.

## 9. AlphaForge application map

### 9.1 What to build from this book (priority order)
1. **Basing-point trailing-stop engine (§7.2–7.3)** — the book's crown jewel for us: fully mechanical, testable exit overlay applicable to ANY entry signal (factor or pattern). Parameters: stop table %, 3-days-away window, 3% new-high advance, volume overrides. Daily and weekly ("3-bars-away") variants.
2. **Donchian/Turtle + Colby-Dow channel systems (§2, §8)** — already fully specified; perfect honest-backtester validation fixtures (known published rules → reproduce → measure decay across regimes; the book PREDICTS post-1980s decay for MA crossovers — test it).
3. **Breakout signal family** — 20/55-day channels, Dow Lines (≤5% range, ≥15d), rectangle/flag detection with the three reliability tests, the 3%-close rule, **and the volume asymmetry (volume gate on longs only)**. The half-mast and runaway-gap measuring rules give algorithmic profit targets in log space.
4. **Gap classifier** — common/breakaway/runaway/exhaustion via location + relative size + next-day volume/price response + 2–5-day fill behavior; islands. Testable claims: gap-confirmed breakouts > gapless; gap-fill null hypothesis.
5. **Volume-anchored S/R** — E&M's distinctive theory maps directly to **volume-profile** features: S/R potency = f(volume at level, distance, age, attack count); zone axis = volume-weighted close mean; role reversal. Testable: prior high-volume bottoms repel rallies better than swing highs do.
6. **Trend/regime classifier** — trend state machine on swing pivots (higher-highs/higher-lows), secondary-vs-minor filters (3-week/1/3-retrace), ADX>25 gate, MA-slope gates; route signals by regime (the book's own instruction).
7. **Breadth / MEI module** — % of universe in bullish major trends as an exposure dial + the divergence-at-tops detector (the 27% lesson). Maps to AlphaForge portfolio construction.
8. **Pattern detectors (harder, later)** — H&S via swing-pivot sequences + volume signature + neckline math; double tops with the ≥1-month/≥15%-valley/≤3%-peaks spec; triangles with the 1/2–3/4 apex window. Glossary formulas (§4) are the canonical specs.

### 9.2 Honesty constraints when backtesting this material
- **Fix one precise rule-set per concept** — the book admits Dow signal dates vary by interpreter; AlphaForge should ship each rule with its parameterization exposed, never "the" H&S.
- Close-based confirmations, signal lag (signals fire on the close AFTER confirmation → trade next bar), and the 3%/2% filters are integral to the claims — test as specified before optimizing.
- The measuring rules are MINIMUMS with an explicit cap (prior move) — score target-hit rates accordingly, in log space.
- Survivorship: the book's reliability claims ("9:1 ascending triangles", "9/10 broadening tops bearish", "~75% triangles consolidate", "80% H&S confirm") are eyeball-era estimates from curated charts — treat as priors to TEST, not facts. Lo/Mamaysky/Wang (2000) is the academic benchmark for pattern-detection methodology.
- Frictions: Magee assumed odd-lot commissions and 1/8 ticks; re-cost everything; the 1/8 progressive-stop offset must be re-fit for decimal markets (book's own note: candidates 9–12.5¢ or ATR-scaled).
- Regime decay is the book's own forecast (150d MA crossover died ~1980s; trend systems crowded). Walk-forward across decades; report per-regime.
- The book's risk doctrine = ours: max drawdown over Sharpe; vol ≠ risk; fat tails are the data (Mandelbrot endorsed in-text); position sizing is "the single most important aspect."

### 9.3 LLM-signal tie-ins (Module A)
- E&M phase anatomy gives NARRATIVE features an LLM can extract and we can cross-check against price/volume state: new-issue surges, junk-vs-quality leadership, "no bad news left" bottoms, media-swarm contrarian flags ("whenever media swarms a company, be skeptical").
- The book's meta-rule for news: "the chart discounts the news; trust the tape over the ratio" → use LLM signals as CONDITIONING variables on technical state, not standalone directional calls — consistent with our sourced-and-validated signal doctrine.

*End of reference. Full detail per section: books/edwards_magee_notes/chunk_00.md … chunk_09.md.*
