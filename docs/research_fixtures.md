# Research Fixtures

`tests/research_fixtures.py` provides two licensed-data bundles. They have
opposite jobs, and using the wrong one silently weakens a test.

## The two fixtures

| | `write_research_bundle` | `write_degenerate_bundle` |
| --- | --- | --- |
| **Job** | Test the **normal research path** | Test **error handling and degenerate cases** |
| **Question it answers** | Does this analysis work when the data is usable? | Does the safeguard fire when the data is not? |
| **Cross-section** | Six names with genuinely different ratios that evolve over time | Every fundamental ratio identical across names |
| **Positions** | A factor can rank and hold names | Nothing can be ranked, so nothing is held |
| **Valid to read a result from?** | The analysis is valid; the *result* still proves nothing about real markets | Never |

**Use the realistic bundle by default.** Reach for the degenerate one only when
the test's subject *is* the guard — `research_validity`, undefined statistics,
or a workflow's behaviour on an unusable sample. Name it explicitly at the call
site so the intent is visible:

```python
def test_flat_cross_section_blocks_a_performance_claim(tmp_path):
    write_degenerate_bundle(tmp_path)          # deliberately unusable
```

### Why the distinction matters

A test that runs on an accidentally flat fixture passes for the wrong reason. A
strategy that never trades produces a flat return series, and a flat return
series contradicts nothing — so a broken analysis sails through. That is exactly
what happened before this change: `gross_profitability` on the old bundle held
no position on any of 520 dates and nobody noticed, because the test only
asserted that the workflow returned.

## The defect this replaced

The previous fixture generated every metric as `base_value * scale`, with one
`scale` per symbol. Any ratio of two fundamentals then cancels the scale
exactly:

| Symbol | scale | (revenue − cogs) / total_assets |
| --- | --- | --- |
| AAA | 1.0 | (3.0bn − 1.70bn) / 5.0bn = **0.26** |
| BBB | 1.2 | (3.6bn − 2.04bn) / 6.0bn = **0.26** |
| CCC | 1.4 | (4.2bn − 2.38bn) / 7.0bn = **0.26** |

This was never specific to gross profitability. Eleven of the fifteen registered
`FACTORLIB_SPECS` entries are pure fundamental ratios and all collapsed the same
way; only the four price-involving ratios (`earnings_yield`, `book_to_price`,
`sales_to_price`, `fcf_yield`) varied, and only through price.

The synthetic provider in `research/data.py` never had this problem — it already
draws per-name size independently of per-name ratio drivers. The realistic
bundle follows that established pattern.

## Design rules for the realistic bundle

1. **Scale is separated from ratios.** `ASSET_BASE` and `SHARE_BASE` set company
   size. `TURNOVER` and `MARGIN` set ratios. Gross profitability reduces to
   `turnover * margin`, so it is governed entirely by the ratio drivers and is
   immune to the size term. The two lists are deliberately not ordered alike, so
   the factor carries no size information.
2. **Anything that appears in a difference-over-level needs its own per-name
   rate.** `asset_growth` is `(ta − ta.shift) / ta.shift`, so `ASSET_BASE`
   cancels there too. A single shared growth rate would have reproduced the
   original defect for `conservative_investment` — which is precisely what the
   audit test caught during development. Hence `ASSET_GROWTH` varies per name.
3. **Ratios evolve.** `TURNOVER_DRIFT` and `MARGIN_DRIFT` give each name its own
   per-period drift, so factor values move over time and the cross-sectional
   ranking can change.
4. **Prices are independent of fundamentals.** `PRICE_BASE`, `PRICE_TREND`,
   `PRICE_AMP`, `PRICE_PERIOD` and `PRICE_PHASE` are separate constants; nothing
   links a price path to a fundamental driver. A test asserts the rank
   correlation between total return and `turnover * margin` stays weak.
5. **Point-in-time is preserved.** Every fiscal period carries
   `available_date = period_end + REPORTING_LAG_DAYS` (75 days). Tests assert
   the lag holds for every row and that nothing becomes available after the
   price history ends.
6. **No randomness.** Every value is an explicit closed form over the symbol
   index and the period index. A test hand-checks the first period's gross
   profitability against `TURNOVER[j] * MARGIN[j] / 4`, and a byte-equality test
   pins determinism.
7. **Small enough to reason about.** Six names, 520 business days, seven fiscal
   periods — enough for terciles, active positions and rank changes; few enough
   to read the CSV.

## The no-planted-edge policy

Two separate commitments, which are easy to confuse:

**1. Nothing here is designed to produce a return.** The price constants
(`PRICE_BASE`, `PRICE_TREND`, `PRICE_AMP`, `PRICE_PERIOD`, `PRICE_PHASE`) are
independent of every fundamental driver, and no constant in this file was chosen
by looking at a backtest result. `test_price_generation_never_reads_a_fundamental_driver`
asserts this structurally: perturbing any fundamental driver leaves `prices.csv`
byte-identical, and perturbing any price constant leaves `fundamentals.csv`
byte-identical.

**2. Fixture backtest performance is not a research finding — whatever its sign.**
Six names on one deterministic path is a sample of one. Such a fixture can come
out positive or negative purely by chance, and *neither outcome means anything*.
A positive result here is not evidence of an edge, and it is not evidence of a
broken fixture either. It is simply not information.

The practical rule that follows:

> **Tests must never assert return sign, magnitude, Sharpe, deflated Sharpe, or
> credibility.**

What tests on this fixture legitimately assert:

| Assertable | Not assertable |
| --- | --- |
| Mechanics: the workflow runs and returns a well-formed result | Whether the result is good |
| Point-in-time handling: filing lags, availability windows | — |
| Cross-sectional dispersion: a factor can rank names apart | Whether the ranking predicts anything |
| Active-position formation: the book is not empty | Whether the positions made money |
| Cost behaviour: a higher cost assumption cannot improve the net result | The level of any return |
| Diagnostic behaviour: `research_validity` and undefined-statistic reporting | — |
| That a verdict is *reachable* and honestly labelled | Which verdict it is |

A test that starts asserting performance has stopped testing the software and
started reading tea leaves from a synthetic sample.

## The factor-library audit

`test_every_registered_factor_has_cross_sectional_dispersion` measures mean
per-date cross-sectional dispersion for every entry in `FACTORLIB_SPECS` plus
the `value_score` and `quality_score` composites, and fails naming any factor
that is constant. `test_the_audit_would_catch_a_flat_factor` proves the audit can
fail, by running the same measurement on the degenerate bundle and asserting the
pure ratio factors collapse there while the price-based ones do not.

The audit asserts only that inputs permit ranking. It does not require any
factor to predict returns or earn a profit, and it should never be changed to.

All sixteen audited factors currently show healthy dispersion on the realistic
bundle. None is structurally flat.

## Adding a factor or a metric

When you add a factor, run the audit. If it reports the new factor as constant,
fix the **fixture** — add or vary the driver the factor actually depends on —
rather than adjusting the factor to suit the test data. Production factor
formulas are never changed to accommodate fixture design.
