# Chronological Stability

## Purpose

`research.temporal_stability` asks whether a **fixed, point-in-time signal** has
an observed relationship with forward returns across distinct historical eras.
It is deliberately a stability screen, not an optimizer, a causal test, or a
prediction of future returns.

This closes a gap between two existing checks:

- **IC decay** asks how a signal behaves at different forward-return horizons.
- **Walk-forward validation** tests a chronological out-of-sample return stream.
- **Chronological stability** asks whether the signal's daily information
  coefficient keeps a consistent direction across pre-declared eras.

No one result replaces the others.

## Method

The default analysis creates five contiguous chronological IC cohorts. Adjacent
cohorts are separated by one embargo bar, so a forward-return label cannot be
used in two reported eras. Each cohort reports:

- valid IC observations;
- mean IC, IC information ratio, and IC hit rate;
- a deterministic moving-block-bootstrap 95% interval for mean IC; and
- the corresponding realised strategy return, labelled descriptive only.

The output also reports the earliest and latest mean IC, their difference and a
block-bootstrap interval, plus the fraction of cohorts sharing the pooled IC
direction. The bootstrap accounts for short-range serial dependence more
honestly than an independent-observation interval, but it does not turn a small
sample into strong evidence.

## Statuses

| Status | Meaning |
| --- | --- |
| `stable` | The observed IC direction is consistent across cohorts and the latest magnitude is not materially below the earliest. |
| `weakened` | The direction remains consistent, but the latest cohort's IC magnitude is less than 70% of the earliest cohort's. |
| `unstable` | Cohort directions conflict, so a pooled result can hide a regime change. |
| `insufficient_evidence` | There are not enough valid IC dates for every pre-declared cohort and embargo. |

`stable` is a temporal descriptor only. It does **not** prove a signal is
tradeable, statistically significant, causal, capacity-aware, or likely to work
in the future. `weakened`, `unstable`, and `insufficient_evidence` never support
a current-evidence conclusion.

## Boundaries

- The analysis never chooses the most favorable era or tunes the cohort boundary
  after seeing a result.
- The Strategy Lab uses the same point-in-time signal panel used by the backtest.
- Filing studies retain their separate model-time eligibility screen. A document
  score made before its model was available cannot enter a primary event-study
  cohort, regardless of its apparent temporal pattern.
- This is not a replication of Look-Ahead-Bench. Measuring language-model
  training-data bias needs historical model checkpoints or an explicit
  model-vintage benchmark. AlphaForge reports only observed chronological
  stability for the supplied, eligible data.
