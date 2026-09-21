# Protocol-Executed Studies

AlphaForge research protocols are executable controls, not labels added after a
backtest. A protocol declares three ordered endpoints before protected results
are inspected:

1. **Exploration** ends at `research_end`.
2. **Validation** runs from the next calendar day through `validation_end`.
3. **Final holdout** runs from the next calendar day through `final_holdout_end`.

The provider selects the first available trading session at each boundary. The
runner loads the full source history through the stage endpoint, so a signal may
use legitimate earlier observations for lookback warm-up. It then discards every
pre-stage return, position statistic, scorecard input, chart value, and data
fingerprint before reporting the stage result. The reported equity curve is
rebased to one at the beginning of the evaluated period.

The final-holdout stage can be recorded only once per study. A failed run still
does not justify retrying it with altered assumptions. Create a fork with a new
hypothesis and preserve the parent identifier if a genuinely new study is needed.
The ledger rejects an unnamed or unchanged-hypothesis fork.
The first protected run also locks the provider, universe, factor, lookback,
skip, costs, gross exposure, trial declaration, seed, and source start date.
Later stages reject a changed specification rather than treating it as the same
hypothesis.

## Running a protocol

Create a study with `POST /protocols`, then use `POST /protocols/{study_id}/run`
with a normal backtest request plus `study_id` and one of `exploration`,
`validation`, or `final_holdout`. The Streamlit Strategy lab exposes the same
control as **Run protected stage** after a local protocol has been created.

Each ledger entry preserves the bounded request, expected and actual evaluation
dates, reproducibility fingerprint, data-readiness status, evidence status, and
headline score. A normal backtest remains an exploratory run; it cannot be
relabelled as a protected holdout after seeing its outcome.
PBO remains a research-phase diagnostic; it is intentionally not recomputed
inside validation or final-holdout data.

## Fixed benchmarks

`POST /benchmark-suite` evaluates the immutable Value, Quality, Momentum (12-1),
and Low-volatility references under one provider, date range, and cost model.
The suite marks a reference unavailable rather than silently changing its data
source. For example, Value and Quality are skipped on Yahoo Finance because it
does not provide filing-dated fundamentals. These are comparison baselines, not
investable recommendations or an optimization menu.

## Real-data completion checklist

Before interpreting a study as real-market research, create a local licensed
bundle with all of the following:

- Corporate-action-adjusted historical prices and volumes.
- Historical membership for every rebalance date.
- Delisted and acquired securities retained through their eligible history.
- As-reported fundamental observations with the earliest public `available_date`.
- A preserved vendor manifest and license evidence alongside the exported audit.

The concrete file contract is in [licensed_data_bundle.md](licensed_data_bundle.md).
The provider interface already rejects incomplete fundamental claims; it does not
invent missing history, membership, or delisting coverage.
