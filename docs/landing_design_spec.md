# Landing v2 — Design Spec (research-derived)

> Synthesized June 2026 from live-page research across 22 premium SaaS/fintech landing
> pages (Linear, Stripe, Vercel, Notion, Figma, Raycast, Framer, Clerk, Resend, Mercury,
> Ramp, Brex, Carta, AlphaSense, Hebbia, Rogo, Composer, QuantConnect, Kalshi, Public…)
> plus headline-formula literature (Julian Shapiro, Marketing Examples, YC pages).
> Implemented in `marketing/landing_v2.html`. v1 (`marketing/landing.html`) kept for reference.

## The headline decision

**Key research finding: 0 of 22 premium pages lead the H1 with fear/accusation/negative
hooks.** The v1 wound headline ("Your backtest is probably lying to you") is *relocated*
to the diagnosis section as shared diagnosis; the hero asserts what the product IS.

Variants (all sentence case, 3–7 words, no exclamation, never promise returns):

| # | Headline | Subhead | Formula / mirror |
|---|---|---|---|
| **1 (primary)** | The research OS for emerging quant managers | Backtest, stress-test, and validate strategies with deflated Sharpe, overfitting probability, and point-in-time data on every run. | Category claim "The X for Y" — Linear, Resend. Right shape for an unknown product; zero brand drift. |
| 2 | Radically honest backtesting | Every result ships with its deflated Sharpe, probability of overfitting, and the assumptions that produced it. | Modified noun phrase — Mercury "Radically different banking". |
| 3 | For quants who refuse to fool themselves | AlphaForge pressure-tests every strategy with deflated Sharpe, PBO, and walk-forward validation — before any capital is at stake. | Audience definition — Rogo, Public. Converts the wound into self-selection. |
| 4 | Quant research, stress-tested | An honest backtesting engine that runs overfit detection, deflated Sharpe, and regime stress-tests on every run — and shows its work. | Two-word+comma institutional — Carta "Private capital, connected". |
| 5 | Backtesting. Built skeptical. | Deflated Sharpe, PBO, and look-ahead checks run on every backtest, every time. | Period fragments — Composer "Trading. Built better." Skepticism on the PRODUCT, not the reader. |

## Page blueprint (9 sections, dark, Linear model)

S0 nav (sticky 64px) → S1 hero (centered stack, 2 CTAs, disclaimer as visible microcopy)
→ S2 **SLOT A** hero product canvas (16:10, bottom cropped by fold) → S3 proof bar
(quantified rigor stats, grayscale, footnoted) → S4 specimen gallery (Clerk pattern;
H2 = "The backtest that won't let you lie to yourself"; **SLOTS B1/B2/B3**) → S5 diagnosis
(the relocated wound: "Most backtests are optimistic by construction." + 3 columns +
optional **SLOT C** 21:9 naive-vs-deflated exhibit) → S6 three alternating 50/50 feature
rows (**SLOTS D1/D2/D3**, 4:3 crops, one region each) → S7 numbers band (56–64px
tabular-nums stats, every figure asterisked → Methodology) → S8 "Built to be audited"
quiet strip → S9 closing CTA band ("Put your strategy on trial.", the page's one glow)
→ footer with full prose disclaimer (prominent disclosure = premium signal, Mercury model).

## Image slots (drop your pictures/designs here)

| Slot | Where | Ratio | Treatment |
|---|---|---|---|
| A `hero_product_canvas` | under hero CTAs | 16:10 (export 2160×1350) | full-width, 1px hairline, 16px radius, single faint radial glow, flat straight-on, NO browser chrome/tilt |
| B1 `verdict_card_specimen` | specimen gallery | 4:5 | museum-specimen card on #0F1011, 12px radius, caption below |
| B2 `tearsheet_specimen` | specimen gallery | 16:10 | identical frame (consistency = the premium tell) |
| B3 `dsr_pbo_specimen` | specimen gallery | 4:3 | identical frame; ≤1 accent highlight inside |
| C `overfit_diff_band` | diagnosis section | 21:9 | court-exhibit strip, naive vs deflated Sharpe, no glow |
| D1 `walkforward_feature_crop` | feature row 1 | 4:3 | one meaningful region, may bleed past edge |
| D2 `signal_extraction_crop` | feature row 2 | 4:3 | same, opposite side |
| D3 `report_export_crop` | feature row 3 | 4:3 | rendered report with visible methodology footnotes |

## Visual rules (the codable ones)

- Dark: bg `#0A0B0D` (never #000), panels `#0F1011`, hairlines `rgba(255,255,255,.06–.08)`, borders-over-shadows.
- One typeface (Inter-class variable), 3 weights max (450/500/~580). Never 700+ at display size.
- H1 64–72px, lh 1.05, letter-spacing **−0.022em** (the Linear signature); H2 36–40px; eyebrow 13px uppercase +0.08em; body 16px.
- Text hierarchy by COLOR not size: `#F7F8F8` / `#D0D6E0` / `#8A8F98`.
- ONE accent (desaturated indigo `#5E6AD2`) — primary CTA, links, ≤2 highlights. Verdict green/red lives inside product imagery only.
- `tabular-nums` on every numeral. Radius ladder 6/8/12/16. Container 1080px; H1 block 720px; copy 640px.
- Sections 96–120px vertical padding; ≤9 sections; one gradient zone total (hero glow, optionally reused in S9).
- Motion micro only (150ms hover, 300ms fade-up). Disclaimer appears twice (hero microcopy + footer prose), never hidden.

## Anti-patterns (never)

Negative-hook H1 · return promises ("beat the market") · >2 hero CTAs · Title Case ·
tilted/3D mockups or browser chrome · stock photos/illustration-as-product · pure #000 ·
multiple gradients · second typeface · mixed radii · >9 sections · fake/aspirational logos ·
cute headline without proof · burying (or leaking) the disclaimer · scroll-jacking.
