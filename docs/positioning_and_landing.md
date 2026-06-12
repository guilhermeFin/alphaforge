# AlphaForge — Positioning & Landing Copy

> Draft, June 2026. Grounded in the competitive research (docs/competitive_landscape.md)
> and the actual product (honest backtester + Deflated Sharpe + point-in-time data +
> AI signals + signal-quality IC). **"AlphaForge" is a placeholder name — trademark-check
> before adopting.** Voice rule below is the most important thing in this file: an
> honesty tool's marketing must itself be honest. No return promises, ever.

---

## 1. Positioning statement

**For** serious quant researchers and emerging managers (the sub-$100M crowd Bloomberg and FactSet ignore)
**who** keep getting burned when strategies that looked brilliant in a backtest fall apart in live trading,
**AlphaForge is** a research platform with an honest backtester and an AI research copilot
**that** tells you whether your edge is real — point-in-time, after costs, and after accounting for every variation you tried —
**unlike** QuantConnect, Nelogica, or any chat-to-strategy tool, which happily let you optimize your way into a fantasy,
**because** we built the overfitting math (Deflated Sharpe, walk-forward, point-in-time data, information-coefficient checks) into the core as a verdict you can't turn off — and fused it with AI signals you can actually test.

**The one line:** *The backtest that won't let you lie to yourself.*

**Category:** honest backtesting & strategy research (a category we name and own, against the un-named "backtesting tools" pile).

---

## 2. The enemy & the wound (lead with this)

Every honest brand needs an enemy. Ours isn't a competitor — it's **the comfortable lie**: tools that are *designed* to make a bad strategy look good.

- **Quantopian** crowdsourced alpha from 300,000 people and shut down in 2020. The lesson the industry swallowed: *you can't crowdsource alpha — you crowdsource overfit backtests.*
- **~97% of day-traders lose money** (B3 / FGV studies), often while staring at a backtest that "proved" the strategy works.
- The dominant tools make it *worse*: brute-force optimizers that try 100,000 parameter combinations and hand you the luckiest one; "AI" black boxes that give you a score you can't inspect; chat-to-strategy apps optimized for how fast you can deploy, not whether you should.

**The wound, said plainly:** *Your backtest is probably lying to you. You just don't know it yet.*

---

## 3. The promise & proof

**Promise:** We can't promise you alpha. We can promise you'll **know whether you actually have it** — before the market charges you tuition to find out.

**Proof points (every one is a real feature, not a slogan):**
- **Point-in-time, no look-ahead.** Your strategy only ever sees what it could have known on the day. (Proven by a truncation test, not a claim.)
- **Real costs.** Every trade pays. No frictionless fantasy curves.
- **The Deflated Sharpe verdict.** We account for how many variations you tried and tell you the honest probability your result isn't luck — the math (Bailey & López de Prado) that lives in academic papers, finally built into a product as a verdict you can't switch off.
- **Walk-forward & out-of-sample by default.** In-sample brilliance that dies out-of-sample gets flagged, loudly.
- **Signal quality, not just strategy returns.** We measure whether your *signal itself* carries information (its Information Coefficient) — so a pretty equity curve sitting on a meaningless signal gets caught.
- **AI signals you can test.** Our copilot turns filings, transcripts, and news into structured, source-cited numbers that drop straight into the same honest backtest — never a black-box "buy" you have to trust.
- **It tells you when returns aren't Normal.** Fat tails detected; Gaussian risk numbers flagged as the lies they are.

---

## 4. Messaging pillars (everything ladders to one of these)

1. **Rigor you can't turn off.** The honesty checks are the product, not a setting. (Differentiator vs. every "tool with a backtester.")
2. **Signals you can interrogate.** AI that produces testable, sourced numbers — not scores you trust on faith. (Differentiator vs. AlphaSense/Danelfin/black boxes.)
3. **Built for you, priced for you.** Institutional-grade discipline for the researcher and emerging manager the incumbents price out. (Differentiator vs. Bloomberg/FactSet.)

---

## 5. Voice & tone

- **Rigorous, plain, a little contrarian.** We sound like the smartest, most honest person in the quant Discord — not a guru, not a hype-man.
- **We never promise returns, "alpha," "get rich," or anything a regulator would call advice.** We sell a *tool* and a *discipline*. This is both the brand and the legal line (research software, not investment advice).
- **We're allowed to be funny about the hype industry** — never about the user's losses.
- **Show, don't claim.** Prefer "proven by a test" over adjectives. The brand dies the first time we exaggerate.

---

## 6. Landing page copy (section by section)

### Hero
**Headline:** Your backtest is probably lying to you.
**Subhead:** AlphaForge is an honest backtesting and research platform that tells you whether your trading edge is real — point-in-time, after costs, and after every variation you tried. The tool that won't let you lie to yourself.
**Primary CTA:** Test your first strategy — free
**Secondary CTA:** See how the verdict works
**Trust line (small, under buttons):** Research software, not investment advice. We don't tell you what to buy — we tell you whether your idea holds up.

### Section — The problem (agitate the wound)
**Heading:** Most backtests are fiction with good production values.
**Body:** Look-ahead bias. Survivorship bias. Costs that quietly vanish. And the big one: you tried fifty versions and kept the one that looked best — which is just luck wearing a lab coat. Quantopian crowdsourced alpha from 300,000 people and shut down; the industry learned that crowdsourcing alpha mostly crowdsources overfit. The popular tools make it worse, optimizing 100,000 combinations to hand you the prettiest curve. The market eventually corrects the record. It charges for the lesson.

### Section — The shift
**Heading:** What if your tools were on your side instead of your ego's?
**Body:** AlphaForge runs every test the way reality will: only the data you'd have had, costs on every trade, and a verdict that accounts for how hard you searched. When something passes here, you've earned the right to believe it. When it doesn't, you just saved yourself a very expensive education.

### Section — How it works (3 steps)
1. **Bring an idea.** A factor, a rule, or text — a filing, a transcript, news. Our AI copilot turns the messy stuff into clean, sourced, testable numbers.
2. **Test it honestly.** Point-in-time data, real costs, walk-forward and out-of-sample by default. No knobs that let you cheat.
3. **Get the verdict.** A Deflated Sharpe that knows how many things you tried. An out-of-sample reality check. A read on whether the *signal itself* carries information. In plain language: *credible, or probably luck.*

### Section — Why it's different (the honesty guarantees)
- **No look-ahead — proven, not promised.** Truncate the data and the past doesn't move.
- **The Deflated Sharpe verdict.** The multiple-testing math the pros publish and nobody ships — built in, and impossible to switch off.
- **Signal quality, not vibes.** We measure your signal's Information Coefficient, so a lucky curve on a meaningless signal can't sneak through.
- **AI you can interrogate.** Every AI signal is a bounded, source-cited number you can test — never a "trust me" score.
- **It flags fat tails.** When your returns aren't Normal, we say so, and we don't let a Gaussian risk number pretend otherwise.

### Section — Who it's for
**Heading:** For people who'd rather be right than feel right.
**Body:** Serious retail quants. Emerging managers running real money. Analysts at small funds and family offices. If you've ever watched a 3.0-Sharpe backtest become a 0.3-Sharpe live track record, this was built for you. If you want a tool that tells you what you want to hear — there are plenty of those.

### Section — Pricing (teaser)
- **Indie — Free.** Run honest backtests, get the verdict. The funnel and the proof.
- **Pro — $49–99/mo.** Full factor library, AI signals, saved reproducible research.
- **Manager — $300–800/mo.** Multi-asset portfolios, attribution, exports, the emerging-manager stack.
- **Team / API — custom.** Signal API and workspaces for small funds and desks.
*(Indicative; validate before publishing.)*

### Section — Closing CTA
**Heading:** Find out if your edge is real.
**Subhead:** Before the market tells you the expensive way.
**Button:** Test your first strategy — free
**Footer disclaimer:** AlphaForge is research software, not investment advice, and does not provide buy/sell recommendations. Backtests are hypothetical and have inherent limitations; past performance does not guarantee future results.

---

## 7. Headline variants (A/B pool)

1. **Your backtest is probably lying to you.** *(recommended hero — the wound, direct)*
2. The backtest that won't let you lie to yourself.
3. Most backtests are fiction. This one has a fact-checker.
4. Find out if your edge is real — before the market does.
5. You can't crowdsource alpha. You can stop crowdsourcing overfit.
6. Stop optimizing your way into a fantasy.
7. Honest backtesting for people who'd rather be right than feel right.

---

## 8. FAQ (objection-handling)

**"Is this investment advice?"** No. AlphaForge is research software. It never tells you what to buy or sell — it tells you whether your own idea survives honest testing.

**"How is this different from QuantConnect / Nelogica / a backtester I already use?"** Those give you an engine and trust you not to fool yourself. We build the overfitting defenses in as a verdict you can't disable, and we measure whether your signal carries real information — not just whether one curve looked good.

**"Will it find me a winning strategy?"** No — and any tool that promises that is the one to distrust. It will tell you, honestly, whether the strategy you have is likely real.

**"What's the AI actually do?"** Reads filings, transcripts, and news and extracts bounded, source-cited numbers (e.g., a guidance-change score) that drop into the same honest backtest. You can inspect and test every one. No black-box scores.

**"Do you sell my data or my strategies?"** No. Your research is yours.

---

## 9. Taglines (pick one to anchor the brand)
- *The tool that won't let you lie to yourself.* (primary)
- *Know if it's real.*
- *Honest backtesting.*
- *Rigor you can't turn off.*

---

## 10. Notes for whoever builds the page
- The page must **look as disciplined as it sounds** — clean, data-forward, no stock-photo traders-at-sunset, no green up-arrows promising riches. Show a real verdict card (CREDIBLE / NOT CREDIBLE with the Deflated Sharpe) as the hero visual.
- Lead capture: "Test your first strategy free" → the synthetic-data demo (works with no account, no data) is the perfect zero-friction first touch.
- The Quantopian story and the "97% lose" stat are the strongest hooks — use them, cite them, don't overstate them.
- Brazil/Portuguese version later: same copy, localized to B3 and the day-trade-casino grievance — but keep "software, not advice" central (CVM line).
