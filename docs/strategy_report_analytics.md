# Strategy Report Analytics: Modules 1-4

This document defines the first institutional tearsheet layer. All inputs are
one-period arithmetic returns. A daily strategy therefore uses `P=252` and all
MAR/threshold values are daily rates unless stated otherwise.

## Standard report inputs

`build_strategy_report(returns, positions=None, trades=None, benchmark_returns=None, candidate_returns=None, ic_values=None, periods_per_year=252, metadata=None) -> dict`

`returns` is required. Positions/weights, trade log, and benchmark are retained as
standard report inputs for later exposure, capacity, trade-statistics, and
benchmark-relative modules. `benchmark_returns` unlocks Module 2; when the Strategy
Lab has no supplied market benchmark it uses a clearly-labelled equal-weight selected-
universe proxy, never silently presenting it as SPY. Candidate returns (`T x M`) unlock search-level PBO and
White Reality Check. The IC series unlocks HAC-corrected IC inference.

## Module 1: volatility and downside risk

| Function | Formula / default |
| --- | --- |
| `annualized_volatility(returns, periods_per_year=252)` | `std(r, ddof=1) * sqrt(P)` |
| `downside_deviation(returns, mar=0)` | `sqrt(mean(min(0, r_t - MAR)^2))` across all periods |
| `sortino_ratio(returns, mar=0, periods_per_year=252)` | `mean(r-MAR) / downside_deviation * sqrt(P)` |
| `ulcer_index(returns)` | `sqrt(mean(drawdown_t^2))` |
| `pain_index(returns)` | `mean(abs(drawdown_t))` |
| `calmar_ratio(returns)` / `mar_ratio(returns)` | `CAGR / abs(max_drawdown)` |
| `omega_ratio(returns, threshold=0)` | `sum(max(r-theta,0)) / sum(max(theta-r,0))` |
| `sterling_ratio(returns, annual_drawdown_adjustment=0.10)` | `CAGR / mean(max(annual_max_drawdown - adjustment, 0))` |
| `k_ratio(returns)` | OLS slope of `log(equity)` over time divided by `SE(slope)` |

Sterling has incompatible industry definitions. AlphaForge uses the declared
modified annual-drawdown formula above and exposes its adjustment rather than
presenting a silently ambiguous number.

### Module 1 chart specs

`underwater_curve(returns)` creates `equity=cumprod(1+r)`,
`high_water_mark=cummax(equity)`, `drawdown=equity/high_water_mark-1`, and a
consecutive-bar `time_under_water_periods` column.

- Underwater chart: filled time-series area; x = return timestamp, y = drawdown %.
  Show a zero line; annotate maximum drawdown and maximum time under water.
- Drawdown distribution: histogram of each local episode's maximum drawdown. x =
  episode depth %, y = number of episodes; annotate worst and mean episode depth.

## Module 2: benchmark-relative performance

Inputs are aligned on their shared finite dates before each calculation. Jensen alpha
is a simple OLS regression and should receive excess returns when a risk-free rate is
material.

| Function | Formula / default |
| --- | --- |
| `tracking_error(strategy, benchmark)` | `std(r_strategy-r_benchmark, ddof=1) * sqrt(P)` |
| `information_ratio(strategy, benchmark)` | `mean(active) / std(active, ddof=1) * sqrt(P)` |
| `beta(strategy, benchmark)` | `cov(r_strategy, r_benchmark) / var(r_benchmark)` |
| `jensens_alpha(strategy, benchmark)` | OLS `r_strategy=alpha+beta*r_benchmark+epsilon`; report per-period and arithmetic annualized alpha, OLS t-statistic, two-sided t p-value, beta, and R-squared. |
| `capture_ratios(strategy, benchmark)` | Mean strategy return divided by mean benchmark return separately on benchmark-up and benchmark-down periods. |
| `return_correlation(strategy, benchmark)` | Pearson correlation, returned as `(value, reason)`. A constant input yields `NaN` with a stated reason rather than a bare `NaN` and a NumPy divide warning. |

### Undefined statistics are named, not silently NaN

A degenerate sample — most often a strategy that holds no effective position, so its
return series is constant — makes several of these statistics undefined. Dividing by a
zero standard deviation gives `0/0`, and a report full of `NaN` reads like a weak result
when it is actually an uncomputable one.

`benchmark_relative_summary` therefore returns `statistic_diagnostics`: one entry per
statistic that could not be computed, each with a `statistic`, a `status` of `undefined`,
and a plain-language `reason`. `has_undefined_statistics` is the quick check. An undefined
correlation is explicitly **not** the same claim as an estimated correlation of zero.

### Module 2 chart specs

- Rolling Sharpe: annualized rolling Sharpe with the declared window. The band is an
  IID approximation: `SR_ann +/- 1.96*sqrt(P*(1+0.5*SR_period^2)/window)`.
- Rolling beta: rolling OLS beta with `beta +/- 1.96*SE(beta)`.
- Rolling IC: rolling mean cross-sectional IC with `mean +/- 1.96*sd/sqrt(window)`.
  These rolling bands are descriptive. Module 4's HAC tests are the inference layer
  for autocorrelated series.

## Module 3: tail risk and distribution diagnostics

VaR and expected shortfall are shown as positive one-period loss magnitudes. Historical
metrics use the linear lower-tail quantile; parametric metrics assume a Normal return
distribution using the sample mean and standard deviation.

| Function | Formula / default |
| --- | --- |
| `historical_var(returns, c)` | `-quantile(r, 1-c)` |
| `historical_expected_shortfall(returns, c)` | `-mean(r | r <= quantile(r, 1-c))` |
| `parametric_var(returns, c)` | `-(mu + sigma*Phi^-1(1-c))` |
| `parametric_expected_shortfall(returns, c)` | `-(mu - sigma*phi(z)/(1-c))`, `z=Phi^-1(1-c)` |
| `tail_ratio(returns)` | 95th-percentile gain divided by absolute 5th-percentile loss |

### Module 3 chart specs

- Return histogram: daily-return bins with historical VaR/ES supplied alongside the
  chart. The return values, not a smoothed estimate, determine each bin.
- Normal QQ plot: sorted realized returns against Normal quantiles fit with sample
  mean and standard deviation; the 45-degree line is the Normal reference.
- Drawdown duration: local underwater episodes report start, end, deepest point, and
  consecutive periods below the high-water mark; the report highlights the longest.

## Module 4: statistical rigor

| Function | Formula / algorithm |
| --- | --- |
| `probabilistic_sharpe_ratio` | Existing Bailey-Lopez de Prado PSR, adjusted for sample size, skewness, and kurtosis. |
| `newey_west_mean_tstat(values, max_lag=None)` | `mean(x) / sqrt(LRV/T)`, where `LRV=gamma0 + 2*sum((1-k/(L+1))*gammak)` uses Bartlett weights. Default `L=floor(4*(T/100)^(2/9))`. |
| `hac_sharpe_tstat` / `hac_ic_tstat` | Newey-West mean t-test on excess returns / the IC series. |
| `minimum_track_record_length` | `ceil(1 + D*(Phi^-1(c)/(SR-SR*))^2)`, `D=1-skew*SR+((kurtosis-1)/4)*SR^2`. Sharpe values are per-period. |
| `purged_cpcv_pbo` | For each combinatorial held-out block set: remove pre-test purge and post-test embargo from train, select best IS candidate, rank it OOS; `PBO=mean(logit(rank)<=0)`. |
| `white_reality_check` | Stationary-bootstrap the centered candidate excess-return matrix; p-value is the share of bootstrap maximum means at least as large as the observed maximum. |

`minimum_track_record_length` rejects non-finite input by design. A constant return
series makes the Sharpe ratio and its higher moments non-finite, so
`statistical_rigor_summary` checks those inputs first and reports the statistic as
`None` alongside a `statistic_diagnostics` entry and a `degenerate_returns_note`. The
primitive keeps its strict contract; only the summary layer degrades gracefully.
Previously this raised, and the caller's broad exception handler replaced the entire
strategy report with an error string.

The CPCV helper accepts already-evaluated candidate return series. A caller that
refits a model inside each fold must do that point-in-time refit before providing
the matrix; a return matrix alone cannot retroactively remove a model-training leak.
White's Reality Check is opt-in in the interactive report so an ordinary backtest
remains responsive. The standalone `white_reality_check` function defaults to 1,000
stationary-bootstrap draws for an offline report.

### Module 4 chart specs

- PBO logit histogram: calculate the OOS rank logit for the in-sample selected
  candidate on each CPCV combination. Histogram x = logit rank, y = split count;
  draw a vertical zero line for median OOS rank and annotate PBO.
- HAC comparison: grouped bars using naive and HAC t-stats computed on the same
  return/IC sequence. x = statistic name, y = t-stat; annotate HAC bandwidth and
  horizontal `+/-1.96` reference lines.

All functions contain formula-level docstrings and have hand-checkable reference
tests in `tests/test_risk.py`, `tests/test_statistical_rigor.py`, and
`tests/test_strategy_report.py`.
