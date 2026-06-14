# AFML → AlphaForge Implementation Menu

> What's worth implementing from López de Prado's *Advances in Financial Machine Learning*
> (Wiley, 2018), filtered through AlphaForge's **fundamental/quantamental, monthly-rebalance,
> cross-sectional** ICP. Extracted from the full book 2026-06-14. Page/chapter cites are to AFML.
> The seven deep-dive items carry enough algorithm to implement without the book.

**Already implemented** (do not redo): Probabilistic Sharpe, Deflated Sharpe, anchored walk-forward, IC analysis.
**Already spec'd/queued** ([product_improvement_research.md](product_improvement_research.md)): PBO/CSCV, CPCV (purge+embargo), trial-count ledger, CVaR.

**Cross-dependency:** A2, A3, A4, B3, B4 all ride on the `PurgedKFold`/CPCV infrastructure already queued — land that first and four items get cheaper.

---

## (A) TOP PRIORITY — core fit for the ICP

### A1. Fractional Differentiation (FFD) — Ch 5
Make a price/fundamental level series stationary while **preserving memory**, by differencing with a real-valued order `d∈[0,1]` instead of integer `d=1` (returns destroy the level signal fundamental ratios depend on). On-brand: feeding raw non-stationary levels into a model is a look-ahead/overfit risk; FFD keeps the slow-moving signal honest.

```python
def get_weights_ffd(d, thres=1e-5):          # binomial-series weights of (1-B)^d
    w=[1.0]; k=1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres: break
        w.append(w_); k+=1
    return np.array(w[::-1]).reshape(-1,1)

def frac_diff_ffd(series, d, thres=1e-5):     # fixed-width window (constant memory)
    w=get_weights_ffd(d,thres); width=len(w)-1
    # for each col: X̃_t = w·X[t-width:t]   (skip until `width` history exists)
```
Pick `d` = the **minimum** `d` whose FFD series passes ADF at 95% (`statsmodels.adfuller`, crit≈−2.86; most log-price series go stationary at `d<0.6`). Report price↔FFD correlation as "memory retained." **Extends** `fundamentals.py`/`factors.py` preprocessing. **Effort M. Deps: numpy/pandas + statsmodels (ADF only).**

### A2. Feature Importance MDA + SFI, with substitution-effect guard — Ch 8
MDI (fast, tree-only, in-sample), **MDA** (OOS score drop when a column is permuted), **SFI** (each feature fit alone). Correlated factors (value/quality variants) share importance → MDA/MDI understate redundant-but-real features; SFI is immune. Ship MDA+SFI together; flag disagreement as a substitution warning. Must run on **purged CV** with `sample_weight` (ties to A3/CPCV).
```
MDA: for each purged fold: fit; base=score(test);
     for feat j: permute col j in test; imp[j]=base-score(permuted)
     normalize by -score (neg_log_loss preferred over accuracy)
```
**Extends** new `research/feature_importance.py`, reports into `signal_quality.py`. **Effort M. Deps: sklearn.**

### A3. Sample Weights by Uniqueness/Concurrency + Time Decay — Ch 4
Overlapping forward-return labels are **not IID** → effective N ≪ row count, inflating every backtest stat. Down-weight observations whose label-horizon overlaps many others.
```
concurrency c_t = # labels whose [t_in,t_out] spans bar t
avg uniqueness ū_i = mean_{t in span_i}(1/c_t)
return-attrib weight w_i = |Σ_{t in span_i}(r_t / c_t)|   (log-returns), scaled so Σw=N
time decay over CUMULATIVE uniqueness (not chronological), newest=1 oldest=clfLastW
```
Average uniqueness also sets bagging `max_samples`. **Extends** `backtest.py`/`walkforward.py` (effective-N reporting) + any `.fit(sample_weight=)`. **Effort M. Deps: pure numpy/pandas.**

### A4. Triple-Barrier Labeling + Meta-Labeling — Ch 3
Label by which barrier is hit first: profit-take / stop / vertical (max hold) — barriers scaled by realized vol. **Meta-labeling:** primary model picks the *side*, a secondary binary model decides *whether to act & how big* (label = was the side's bet profitable). For a monthly long-only ranker, adopt the **vertical-barrier + meta-labeling** parts first (vertical barrier = "hold to next rebalance"); pt/sl optional. **Extends** new `research/labeling.py`. **Effort M–L. Deps: numpy/pandas + sklearn (secondary clf).**

### A5. Backtest Statistics Suite — Ch 14  ★ cheapest, highest brand payoff
Beyond Sharpe: # bets, avg holding period, **HHI concentration** of returns (are 3 names / one quarter carrying all PnL?), full **drawdown / time-under-water**. This is the literal "won't let you lie to yourself" thesis as numbers.
```
HHI: w=ret/ret.sum(); hhi=(Σw² − 1/n)/(1 − 1/n)   # on +returns, −returns, bets/month
DD/TuW: hwm=series.expanding().max(); per-hwm depth + time between new highs
```
**Extends** `metrics.py` directly. **Effort S. Deps: pure pandas.**

### A6. Hierarchical Risk Parity (HRP) — Ch 16
Portfolio weights without inverting Σ (ill-conditioned → "Markowitz's curse"). (1) tree-cluster on correlation distance `d=√(0.5(1−ρ))`; (2) quasi-diagonalize (dendrogram leaf order); (3) recursive bisection splitting weight by inverse cluster variance. Beats CLA/IVP out-of-sample; monthly-rebalance native.
```python
dist = sqrt(0.5*(1-corr)); link = scipy.cluster.hierarchy.linkage(dist,'single')
# quasi-diagonalize -> sortIx; then recursive bisection:
# getIVP(cov)=1/diag normalized; clusterVar=w'Σw; alpha=1-v0/(v0+v1)
```
**Extends** new `research/allocation.py`, consumed by `backtest.py` as a weighting scheme. **Effort M. Deps: numpy/pandas + scipy.cluster.hierarchy.**

---

## (B) USEFUL LATER

- **B1. Bet sizing from predicted probability — Ch 10.** `m=2·Φ(z)−1`, `z=(p̂−1/K)/√(p̂(1−p̂))`, average concurrent bets, discretize to curb overtrading. v2 once there's a held book. Pairs with A4. **S–M, scipy.**
- **B2. Strategy risk / probability of failure — Ch 15.** Implied precision/frequency to hit a target Sharpe; `probFailure` via 2-Gaussian mixture on returns. On-brand "minimum viable edge." **S, numpy/scipy.**
- **B3. Sequential bootstrap — Ch 4.** Draw bootstrap samples favoring high-uniqueness obs → more-independent training sets. O(n²)/draw, fine at monthly cadence. Build after A3. **M, numpy/pandas.**
- **B4. CPCV purging edge-cases — Ch 7.** The exact three overlap cases to purge + embargo applied only on the right of the test block. Absorb when implementing the queued CPCV. **S, sklearn `_BaseKFold`.**
- **B5. SADF explosiveness / structural-break — Ch 17.** `SADF_t = sup_{t0}{β̂/σ̂_β}` rolling ADF → regime/bubble flag. Borderline (price-series oriented). **M, numpy.**

---

## (C) SKIP — wrong fit for this ICP

- **Tick/volume/dollar/imbalance bars & ETF trick — Ch 2** — microstructure sampling; cadence is monthly. Skip.
- **CUSUM event filter — Ch 2** — intrabar event sampling; irrelevant to fixed monthly rebalance.
- **Microstructural features (Kyle/Amihud/Hasbrouck λ, Roll, VPIN) — Ch 19** — need trade/quote data, HFT horizon. (Note: simple Amihud illiquidity from daily data IS in the factor library and is fine — distinct from Ch 19's tick-level λ.)
- **Entropy features — Ch 18** — data-hungry, marginal on monthly fundamentals.
- **HPC/multiprocessing engine — Ch 20–22** — plumbing, not a feature; use joblib only if a routine is slow.
- **Optimal trading rules (O-U heatmaps) — Ch 13** — execution/holding-rule optimizer, itself an overfitting hazard.

---

## Suggested build order
A5 (S, immediate readout payoff) → A3 (honest N, feeds everything) → A1 (FFD preprocessing) → A6 (HRP portfolio layer) → A2 (rides CPCV) → A4 (+ B1 follow-on).
