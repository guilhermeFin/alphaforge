# Research Validity and Data Health

## Why this exists

A backtest over a universe whose signal never varies still produces an equity
curve, a drawdown series, a full report and a headline verdict. Every number is
arithmetically correct and the whole thing is meaningless: nothing was ever
traded, so there is no performance to evaluate.

`research_validity_summary` answers the prior question — *was there enough
variation in this sample to support any performance claim?* — and states the
answer in one plain sentence. It describes the sample only. It never adjusts a
metric, a verdict, or a statistic.

It is attached to every strategy report (`strategy_report.research_validity`)
and every filing event study (`research_validity` on the study payload), and it
is rendered above the headline verdict in the Strategy Lab, so a reader sees the
data-health finding before the result it qualifies.

## What it reports

| Field | Meaning |
| --- | --- |
| `returns.effective_observations` | Finite return observations behind every downstream statistic |
| `returns.non_flat_returns` | Observations where the return was not exactly zero |
| `returns.is_flat` | True when the return series has zero variance |
| `signal_dispersion.mean_dispersion` | Average cross-sectional spread of the signal per date |
| `signal_dispersion.zero_dispersion_share` | Share of dates on which every name had the same score |
| `positions.average_active_names` | Mean count of held positions per date |
| `positions.concentration_herfindahl` | Mean Herfindahl on gross weights; `1.00` is a single-name book |
| `positions.dates_with_no_positions` | Dates on which the book was empty |
| `undefined_statistics` | Statistics other report sections could not compute |
| `status` / `conclusion` | One of three statuses plus a plain-English sentence |

Statuses are `insufficient_variation` (no performance claim is possible),
`limited_variation` (the sample is usable but partial), and
`sufficient_variation`. `supports_performance_claim` is the single boolean the
UI keys on.

A panel signal is measured **per date across names**, because a signal that is
identical for every name on a date cannot rank anything on that date, however
much it moves through time.

## Diagnosis: the flat `gross_profitability` case

A `gross_profitability` run over the three-symbol licensed test bundle produced a
return series that was constant across all 520 observations. Traced end to end:

```
signal dispersion       0.0 on 100% of dates
average active names    0.0
dates with no positions 520 of 520
non-flat returns        0 of 520
```

**Cause.** The licensed-bundle test fixture (now preserved as
`tests/research_fixtures.py::write_degenerate_bundle`) generates every fundamental metric as `base_value * scale`, where
`scale = 1 + j * 0.2` is one constant per symbol. Gross profitability is
`(revenue - cogs) / total_assets`, so the scale cancels exactly:

| Symbol | scale | (revenue − cogs) / total_assets |
| --- | --- | --- |
| AAA | 1.0 | (3.0bn − 1.70bn) / 5.0bn = **0.26** |
| BBB | 1.2 | (3.6bn − 2.04bn) / 6.0bn = **0.26** |
| CCC | 1.4 | (4.2bn − 2.38bn) / 7.0bn = **0.26** |

All three names score identically, the cross-section has no dispersion, the
long/short construction has nothing to rank, the book stays empty, and returns
are flat at zero.

**Classification: input-data coverage limitation.** Specifically a fixture-
construction artifact, and it generalises — *any* ratio factor built from these
proportionally scaled metrics is constant across the cross-section, so the same
flat result would appear for ROE, operating profitability, and the other ratio
factors in `factor_lib`.

It is explicitly **not**:

- *An expected small-sample limitation.* Three names is small, but three names
  with genuinely different ratios would still rank and still trade. Size is not
  what causes this.
- *A signal-construction issue.* `gross_profitability` implements Novy-Marx
  correctly; `momentum` on the same bundle trades normally
  (`average_active_names` 2.27, 393 of 520 non-flat returns). The factor is
  fine; its input has no variation to work with.

**Resolved.** The licensed-bundle fixture was replaced with a deterministic
generator that separates company size from the drivers that set ratios; see
[research_fixtures.md](research_fixtures.md). The flat design is preserved as
`write_degenerate_bundle`, used only by tests that validate this guard. The
diagnostic still makes the condition impossible to miss wherever it occurs:
*"The return series never changes because no position was ever taken. This
sample has insufficient variation for a performance claim."*

## Limitations

1. **It measures variation, not validity.** `sufficient_variation` means the
   sample can support the statistics; it says nothing about whether they are
   credible, correctly specified, or economically meaningful.
2. **Dispersion thresholds are conventions.** A signal is called degenerate when
   more than 90% of dates show zero cross-sectional spread
   (`MIN_VARYING_DATE_SHARE`). Near-zero-but-nonzero dispersion is not flagged
   even though it may be just as unusable.
3. **Position health needs a position panel.** Workflows that do not supply one
   report `available: false` rather than a concentration figure.
4. **Event studies have no cross-section.** There the signal is a single tone
   series, so dispersion is its spread over events, not across names on a date.
