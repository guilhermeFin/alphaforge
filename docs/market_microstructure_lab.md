# Market Microstructure Lab

## First registered question

Does rolling signed trade-flow imbalance improve fixed-horizon trade-price
markouts beyond the most recent trade return?

The research contract is deliberately narrow:

- Source: synthetic smoke-test events or a user-supplied Binance trade CSV.
- Candidate: rolling signed quantity imbalance.
- Baseline: most recent trade return.
- Evaluation: chronological 60% training, 20% validation, 20% final holdout.
- Success rule: the candidate's absolute final-holdout Spearman IC improves by at
  least 0.01 over the baseline.

Trade events do not establish BBO, queue position, cancellation state, or realised
limit-order fills. Consequently, the lab calls its return label a **trade-price
markout** and calls its quote-policy results a **simulation**. Neither is presented
as executable performance evidence.

## Data contracts

The normalized trade schema is:

| field | meaning |
| --- | --- |
| `timestamp` | UTC event timestamp |
| `price` | strictly positive trade price |
| `quantity` | strictly positive trade quantity |
| `is_buyer_maker` | Binance aggressor-side flag |

The importer rejects unordered, duplicate, malformed, or insufficient event streams
instead of sorting or filling them silently. The audit records timestamp gaps and a
content fingerprint for every result.

## Quote simulator

The simulator compares fixed, volatility-aware, and inventory/flow-aware quote
policies. It makes fees, cancellation delay, inventory caps, fill probabilities, and
the random seed visible. It is a stress-testing tool, not a limit-order-book replay.

## Options diagnostics

The options module performs Black-Scholes implied-volatility and delta calculations
for supplied observations. It is a mathematical/data-quality diagnostic. Indicative
or delayed options data must not be represented as executable OPRA evidence.
