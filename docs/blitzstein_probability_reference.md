# Blitzstein & Hwang — Introduction to Probability (2nd ed.)
## Master Probability Reference for AlphaForge

> Synthesized June 2026 from a full read of all 636 pages (the authors distribute this book free).
> Per-chapter detail: `books/blitzstein_notes/chunk_00..05.md`. This is the deduplicated,
> quant-oriented synthesis: every core definition/theorem/distribution stated for reference,
> plus the AlphaForge application map (§10). **KEY** = memorize.

---

## 1. Foundations (Ch. 1–2)

- **Probability axioms:** `P(∅)=0, P(S)=1`, countable additivity for disjoint events. `P(Aᶜ)=1−P(A)`; `A⊆B ⇒ P(A)≤P(B)`.
- **KEY Inclusion-exclusion:** `P(A∪B)=P(A)+P(B)−P(A∩B)`; general alternating sum.
- **Counting (sampling table):** order+replace `nᵏ`; order/no-replace `n!/(n−k)!`; unordered/no-replace `C(n,k)=n!/((n−k)!k!)`; unordered+replace `C(n+k−1,k)` (Bose-Einstein — NOT equally likely, don't put in naive probability).
- **KEY Conditional probability:** `P(A|B)=P(A∩B)/P(B)`. Chain rule `P(A₁…Aₙ)=∏P(Aᵢ|A₁…Aᵢ₋₁)`.
- **KEY Bayes:** `P(A|B)=P(B|A)P(A)/P(B)`. **Odds form: posterior odds = likelihood ratio × prior odds** (fast sequential updating; avoids computing P(B)).
- **KEY LOTP (law of total probability):** partition `{Aᵢ}` → `P(B)=ΣP(B|Aᵢ)P(Aᵢ)`. Both Bayes & LOTP have an "extra conditioning" variant (append `,E` everywhere).
- **Independence:** `P(A∩B)=P(A)P(B)`. Conditional independence given E ⇎ unconditional independence (both directions fail) — **the formal home of regime-dependent correlation: assets conditionally independent given a latent factor/regime can be strongly correlated unconditionally.**
- **Base-rate fallacy** (rare-disease math): a "95% accurate" test for a 1%-prevalence event is mostly false positives — `P(D|+)≈16%`. **Directly governs how to evaluate a rare-event trading signal: hit-rate ≠ precision.**
- **Simpson's paradox:** an edge present in every subgroup can reverse when pooled, when the confounder's weights differ across groups. **Never pool backtests across regimes/sectors/periods without checking disaggregated results.**
- **Gambler's ruin** (edge p, q=1−p, stake i of N): ruin/​win probabilities `pᵢ=[1−(q/p)ⁱ]/[1−(q/p)ᴺ]` (`=i/N` if fair). A slight negative edge compounds to near-certain ruin → bankroll/drawdown sizing.

## 2. Random variables, expectation, variance (Ch. 3–4)

- **PMF** `P(X=x)`; **CDF** `F(x)=P(X≤x)` (increasing, right-continuous, limits 0/1; jumps = PMF mass). **PDF** `f=F′` (continuous; probability = area).
- **KEY Linearity:** `E(X+Y)=E(X)+E(Y)` — **holds even for dependent X,Y.** `E(cX)=cE(X)`.
- **KEY Fundamental bridge:** `P(A)=E(I_A)`. Write a count as `X=ΣIⱼ`, apply linearity (works for dependent indicators) — the workhorse for E of complicated counts.
- **KEY LOTUS:** `E(g(X))=Σ g(x)P(X=x)` (or ∫). Never swap E and a nonlinear g: `E(g(X))≠g(E(X))`.
- **KEY Variance:** `Var(X)=E(X²)−(EX)²`; `Var(cX)=c²Var(X)`; `Var(X+c)=Var(X)`; independent ⇒ `Var(X+Y)=Var(X)+Var(Y)` (else add `2Cov`).
- Poisson paradigm: sum of many rare weakly-dependent indicators ≈ `Pois(λ)`, `λ=Σpⱼ`.

## 3. The distribution table (the modeling cheat-sheet)

`q=1−p`. MGF `M(t)=E(e^{tX})`.

| Dist | PMF/PDF | Mean | Var | MGF | Quant use |
|---|---|---|---|---|---|
| Bernoulli(p) | `p,q` | `p` | `pq` | `q+pe^t` | up/down tick, default flag |
| Binomial(n,p) | `C(n,k)pᵏq^{n−k}` | `np` | `npq` | `(q+pe^t)ⁿ` | #up-days, #defaults (homog.) |
| Geometric(p) | `qᵏp`, k≥0 | `q/p` | `q/p²` | `p/(1−qe^t)` | #quiet steps before a jump |
| NegBinom(r,p) | `C(n+r−1,r−1)pʳqⁿ` | `rq/p` | `rq/p²` | `(p/(1−qe^t))ʳ` | wait to r-th arrival; overdispersion |
| Hypergeom(w,b,n) | `C(w,k)C(b,n−k)/C(w+b,n)` | `nw/(w+b)` | `np(1−p)·(w+b−n)/(w+b−1)` | — | sampling w/o replacement |
| Poisson(λ) | `e^{−λ}λᵏ/k!` | `λ` | `λ` | `e^{λ(e^t−1)}` | jump/trade/default counts |
| Uniform(a,b) | `1/(b−a)` | `(a+b)/2` | `(b−a)²/12` | `(e^{tb}−e^{ta})/(t(b−a))` | priors, simulation |
| **Normal(μ,σ²)** | `(1/σ√2π)e^{−(x−μ)²/2σ²}` | `μ` | `σ²` | `e^{μt+σ²t²/2}` | log-returns (with caveats) |
| LogNormal(μ,σ²) | `(1/xσ√2π)e^{−(ln x−μ)²/2σ²}` | `e^{μ+σ²/2}` | `m²(e^{σ²}−1)` | ∞ (no MGF) | prices (GBM) |
| Exponential(λ) | `λe^{−λx}` | `1/λ` | `1/λ²` | `λ/(λ−t)` | waiting times (memoryless) |
| Gamma(a,λ) | `(λx)ᵃe^{−λx}/(xΓ(a))` | `a/λ` | `a/λ²` | `(λ/(λ−t))ᵃ` | durations, ΣExpo, Poisson conj. |
| Beta(a,b) | `x^{a−1}(1−x)^{b−1}/β(a,b)` | `a/(a+b)` | `μ(1−μ)/(a+b+1)` | — | bounded qty, Binomial conjugate |
| Chi-square(n) | `Gamma(n/2,1/2)` | `n` | `2n` | `(1−2t)^{−n/2}` | variance inference |
| **Student-t(n)** | `∝(1+x²/n)^{−(n+1)/2}` | `0` (n>1) | `n/(n−2)` (n>2) | undefined | **fat-tailed returns** |

- **Memoryless:** Exponential (continuous) and Geometric (discrete) are the ONLY memoryless distributions: `P(X≥s+t|X≥s)=P(X≥t)`.
- Key relations: `Bin→Pois` (n→∞, p→0, np→λ); `Bin|sum→Hypergeom`; `Pois|sum→Bin`; `Expo=Gamma(1,λ)`; `ΣExpo(λ)=Gamma(n,λ)`; `Chi²(n)=Gamma(n/2,½)`; `Cauchy=t(1)` (no mean/var).

## 4. Moments, MGFs, fat tails (Ch. 6)

- **KEY Best predictors:** constant minimizing MSE `E(X−c)²` is the **mean**; minimizing MAE `E|X−c|` is the **median**. `E(X−c)²=Var(X)+(μ−c)²`.
- **Skewness** = `E((X−μ)/σ)³` (negative = long left tail). **Kurtosis** = `E((X−μ)/σ)⁴−3` (Blitzstein convention subtracts 3 → Normal=0; positive = **leptokurtic / fat tails**).
- **FINANCE CORE:** empirical returns are **leptokurtic and often negatively skewed**; assuming Normal (kurt 0) systematically underestimates tail risk. Student-t (low df) is the canonical fat-tailed swap-in (same mean/var as Normal, far heavier tails).
- **MGF properties:** moments via `E(Xⁿ)=M⁽ⁿ⁾(0)`; **MGF determines the distribution**; independent sum ⇒ MGFs multiply; `M_{a+bX}(t)=e^{at}M_X(bt)`. LogNormal & Student-t have no usable MGF (heavy tails) → use characteristic function `E(e^{itX})`, which always exists.
- **Sample moments:** `E(X̄ₙ)=μ`, `Var(X̄ₙ)=σ²/n` (SE `=σ/√n`); sample variance uses `n−1` (Bessel) for unbiasedness, but **sample SD is biased low** (Jensen on √).

## 5. Joint distributions, covariance, MVN (Ch. 7)

- Joint/marginal/conditional PMFs & PDFs; `f_{X,Y}=f_{Y|X}f_X`; independence ⇔ joint factors into marginals **on all of R²** (not just support).
- **KEY Covariance:** `Cov(X,Y)=E(XY)−E(X)E(Y)`; bilinear; `Var(ΣwᵢXᵢ)=Σwᵢ²Var(Xᵢ)+2Σ_{i<j}wᵢwⱼCov(Xᵢ,Xⱼ)` = **the Markowitz portfolio-variance formula** `wᵀΣw`.
- **KEY Correlation** `∈[−1,1]`. **Independence ⇒ uncorrelated, but NOT conversely** (e.g. `X~N(0,1), Y=X²`: uncorrelated yet fully dependent) — linear correlation misses nonlinear/tail dependence.
- **Multivariate Normal:** linear combos are Normal; marginals Normal; **uncorrelated + jointly Normal ⇒ independent** (special to MVN); fully specified by mean vector + covariance matrix Σ. Parametric VaR / Gaussian copulas lean on MVN — and inherit its thin-tail underestimation of joint crashes (correlations spike toward 1 exactly when diversification is needed).

## 6. Transformations & order statistics (Ch. 8)

- **KEY 1D change of variables:** `f_Y(y)=f_X(x)|dx/dy|` (monotonic g); multivariate uses `|det Jacobian|`. Discrete transforms need NO Jacobian.
- **KEY Convolution** (independent sum): `f_T(t)=∫f_X(x)f_Y(t−x)dx`.
- **Beta-Gamma (bank–post-office):** `X~Gamma(a,λ), Y~Gamma(b,λ)` indep ⇒ `X+Y~Gamma(a+b,λ)`, `X/(X+Y)~Beta(a,b)`, and the two are **independent**.
- **Conjugacy:** Beta is conjugate to Binomial (`Beta(a+k, b+n−k)` after k successes in n); Gamma is conjugate to Poisson (`Gamma(r₀+y, b₀+t)`).
- **Order statistics:** `f_{X(j)}=n·C(n−1,j−1)f(x)F(x)^{j−1}(1−F)^{n−j}`; uniforms: `U_(j)~Beta(j,n−j+1)`. **Empirical quantiles → historical-simulation VaR / Expected Shortfall; max/min → extreme-value & drawdown analysis.**

## 7. Conditional expectation (Ch. 9) — the forecasting engine

- `E(Y|X)` is a **random variable** (a function of X); `E(Y|X=x)=g(x)`.
- Properties: take out what's known `E(h(X)Y|X)=h(X)E(Y|X)`; linearity; independence ⇒ `E(Y|X)=E(Y)`.
- **KEY Adam's Law (iterated expectation):** `E(E(Y|X))=E(Y)`.
- **KEY Eve's Law (total variance):** `Var(Y)=E(Var(Y|X))+Var(E(Y|X))` = **unexplained (noise) + explained (signal) variance** — the variance-decomposition behind factor models and R².
- **KEY:** `E(Y|X)` is the **minimum-MSE predictor** of Y given X (L2 projection). In finance: the optimal forecast of a return given the information set/filtration; conditioning set = features.

## 8. Inequalities & limit theorems (Ch. 10) — the risk-bound + asymptotics core

- **Jensen:** convex g ⇒ `E(g(X))≥g(E(X))`; concave ⇒ ≤. Consequences: `E(X²)≥(EX)²`; **`E(log X)≤log E(X)` ⇒ geometric mean ≤ arithmetic mean** (volatility drag; Kelly/log-wealth); `E(1/X)≥1/E(X)`.
- **Markov:** `P(|X|≥a)≤E|X|/a` (assumption-free, crude). **Chebyshev:** `P(|X−μ|≥cσ)≤1/c²` (distribution-free loss bound from mean+variance). **Chernoff:** `P(X≥a)≤M_X(t)/e^{ta}`, optimize over t>0 → exponentially tight (large-deviation / concentration bounds).
- **Cauchy-Schwarz:** `|E(XY)|≤√(E(X²)E(Y²))` ⇒ `|Corr|≤1`.
- **KEY LLN:** sample mean `X̄ₙ→μ` (weak = in probability via Chebyshev; strong = almost surely). Underpins Monte Carlo and the realization of a long-run edge. **Does NOT imply short-run mean reversion** (gambler's fallacy: outcomes are swamped, not corrected).
- **KEY CLT:** `√n(X̄ₙ−μ)/σ →ᵈ N(0,1)`; rate `O(1/√n)`; **requires finite mean AND variance.** Use continuity correction (±½) for discrete counts.
- **KEY CLT FAILURE = the probabilistic root of tail risk:** infinite-variance / power-law tails (Cauchy: the sample mean of n Cauchys is still Cauchy, forever) violate CLT's premise → Gaussian/CLT-based risk models understate tail risk and converge slowly or never. This is *why* fat-tailed returns break parametric VaR.
- **Volatility-drag paradox:** a stock ×1.7 or ×0.5 daily (equal odds) has `E(wealth)→∞` yet `wealth→0` almost surely (geometric mean < 1). **Always analyze growth in log space.**
- **Chi-square / Student-t:** `(n−1)S²/σ²~χ²_{n−1}`; `t=Z/√(V/n)` → fat-tailed, → Normal as n→∞.

## 9. Markov chains, MCMC, Poisson processes (Ch. 11–13)

- **Markov chain:** transition matrix Q (rows sum to 1); n-step `=Qⁿ`; marginal of Xₙ `=tQⁿ`. States recurrent/transient; chain irreducible (all states communicate) / aperiodic (period gcd =1).
- **KEY Stationary distribution** s: `sQ=s` (left eigenvector, eigenvalue 1). Irreducible ⇒ exists & unique; irreducible+aperiodic ⇒ `P(Xₙ=i)→sᵢ` from any start. `sᵢ=1/rᵢ` (1/mean return time). **Regime-switching & credit-migration models; stationary s = long-run regime occupancy; mean return time = expected regime duration.**
- **KEY Detailed balance (reversibility):** `sᵢqᵢⱼ=sⱼqⱼᵢ` ⇒ s stationary (sufficient, not necessary). Local equations avoid the global eigenproblem — the engine of MCMC.
- **MCMC** samples a target known only up to a normalizing constant (Bayesian posteriors). **Metropolis-Hastings:** propose j from any irreducible P; accept w.p. `min(1, s_j p_ji /(s_i p_ij))` — for symmetric proposals `min(1, s_j/s_i)`, and the normalizing constant cancels. **Gibbs:** cycle through full conditionals (= MH with acceptance 1). Practical: burn-in, correlated draws (autocorrelation inflates estimator variance), tune proposal scale to a moderate acceptance rate. Use: Bayesian estimation of vol/jump/regime/factor params → full posterior ⇒ posterior/parameter risk, not just point estimates. Simulated annealing (`s∝e^{βV}`, β↑) for combinatorial portfolio optimization.
- **Poisson process** (rate λ): counts in length-t interval `~Pois(λt)`, independent on disjoint intervals; **interarrivals i.i.d. Expo(λ); n-th arrival ~Gamma(n,λ)**; given N(t)=n, arrival times = order statistics of n Uniforms. **Superposition** (rates add) & **thinning** (split by prob p into independent PPs). Models trade/order/jump/default arrivals; inhomogeneous λ(t) for intraday seasonality; Cox process (random λ) for stochastic-intensity defaults; Hawkes generalizes to self-exciting order flow.
- Math facts reused constantly: `Γ(n)=(n−1)!`, `Γ(½)=√π`, `Γ(a+1)=aΓ(a)`; geometric series `Σxⁿ=1/(1−x)`; `Σxⁿ/n!=eˣ`.

## 10. AlphaForge application map

1. **Backtest statistics layer (`research/stats.py`):** distribution fitting + the table above; skewness/kurtosis/Jarque-Bera normality tests on returns (expect rejection — leptokurtosis); Student-t and EVT/order-statistics for tails; **deflated/​probabilistic Sharpe needs the sampling distribution of the mean (CLT) and its failure under fat tails** — connects directly to the López de Prado overfitting guards already in the CLAUDE.md moat.
2. **Risk module:** Chebyshev/Chernoff distribution-free loss bounds as a sanity floor under parametric VaR; historical-simulation VaR/ES = empirical order statistics (Ch. 8); covariance matrix `wᵀΣw` for portfolio variance, with the explicit caveat that correlations break in crises (independence ⇏ from uncorrelated; MVN thin tails).
3. **Signal evaluation (ties to Module A LLM signals):** Bayes odds-form for sequential signal updating; **base-rate fallacy as a first-class check on any rare-event signal** (precision vs hit-rate); Simpson's-paradox guard = mandatory per-regime disaggregation of backtest results (reinforces the §8 honesty rules in the E&M reference and CLAUDE.md).
4. **Forecasting:** `E(Y|X)` = the optimal predictor given features → the conceptual basis of every factor/return model; Eve's Law = signal-vs-noise variance decomposition for attribution / R².
5. **Regime modeling:** Markov chains for bull/bear/crash regimes (estimate Q, read stationary occupancy & expected durations) — pairs with the E&M Evaluative-Index exposure dial.
6. **Bayesian estimation (`research/bayes.py`, later):** conjugate updates (Beta-Binomial for win-rates, Gamma-Poisson for arrival intensities) where closed-form; MCMC (MH/Gibbs) for stochastic-vol / hierarchical / regime params → posterior parameter risk feeds position sizing.
7. **Monte Carlo (`research/sim.py`):** LLN justifies MC pricing/strategy simulation; **always simulate compounding in log space (volatility drag)**; Poisson processes for jump/arrival simulation; report MC standard error (`σ/√n`).

### Honesty constraints (consistent with the AlphaForge moat)
- Returns are **not Normal** — never ship a risk number that assumes they are without a fat-tailed alternative beside it.
- A high signal hit-rate is not precision (base rates); a pooled edge is not a per-regime edge (Simpson); a high `E(return)` is not high growth (Jensen/volatility drag). Encode all three as automated checks.
- CLT-based confidence on Sharpe/means degrades with fat tails and finite samples — surface the assumption, don't hide it.

*End of reference. Full detail: books/blitzstein_notes/chunk_00.md … chunk_05.md.*
