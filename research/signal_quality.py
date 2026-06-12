"""Honest signal-quality scorecard — judge the SIGNAL, not just the backtest.

A good backtest can come from a bad signal plus luck. Before a signal (an AI
sentiment score, a fundamental factor, anything) is allowed to count, this module
asks the harder question: *does the signal itself carry information?*

The core metric is the Information Coefficient (IC): the cross-sectional rank
correlation between the signal on date t and the return that FOLLOWS t (t -> t+h).
Using the forward return is correct and is NOT look-ahead — we are measuring the
signal's relationship to the future, not trading on it; the signal must itself be
point-in-time (known at t). We then report:

  * IC mean / std / t-stat / IR  — is the predictive power real and significant?
  * IC decay across horizons      — does the edge persist, or vanish after 1 day
    (a fast-decaying edge that transaction costs will eat)?
  * Out-of-sample IC              — does the signal hold on data it wasn't seen on?
  * Quantile monotonicity         — do higher signal ranks really earn higher returns?
  * Rank autocorrelation          — slow signal (cheap) vs noisy signal (costly)?
  * Coverage                      — how much of the universe the signal actually scores.
  * Re-run stability              — for LLM signals, do repeated extractions agree?

Honesty caveat baked in: for horizon > 1 the forward returns overlap, which
inflates the naive t-stat; we flag that rather than hide it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401  (kept for downstream use / API parity)

TRADING_DAYS = 252


def _forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from t to t+horizon, indexed at t (the last `horizon` rows are NaN)."""
    return close.shift(-horizon) / close - 1.0


def ic_series(signal: pd.DataFrame, close: pd.DataFrame, horizon: int = 1,
              method: str = "spearman", min_names: int = 5) -> pd.Series:
    """Per-date cross-sectional IC between the signal and the forward return.

    Spearman (rank) by default — robust to outliers and the natural choice for a
    monotonic-edge question. Computed only over names that have BOTH a signal and a
    forward return on that date (the common cross-section), so ranks are honest.
    """
    sig = signal.reindex_like(close)
    fwd = _forward_return(close, horizon)
    common = sig.notna() & fwd.notna()
    a = sig.where(common)
    b = fwd.where(common)
    if method == "spearman":
        a = a.rank(axis=1)
        b = b.rank(axis=1)
    A = a.to_numpy(dtype=float)
    B = b.to_numpy(dtype=float)

    mask = ~np.isnan(A)
    count = mask.sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN warmup rows
        am = np.nanmean(A, axis=1, keepdims=True)
        bm = np.nanmean(B, axis=1, keepdims=True)
        da = A - am
        db = B - bm
        num = np.nansum(da * db, axis=1)
        den = np.sqrt(np.nansum(da * da, axis=1) * np.nansum(db * db, axis=1))
        ic = num / den
    ic[(count < min_names) | (den == 0) | ~np.isfinite(ic)] = np.nan
    return pd.Series(ic, index=close.index).dropna()


def ic_summary(signal, close, horizon: int = 1, method: str = "spearman") -> dict:
    ic = ic_series(signal, close, horizon, method)
    n = int(ic.size)
    if n < 3:
        return {"horizon": horizon, "n_obs": n, "mean_ic": None, "ic_std": None,
                "ic_ir": None, "ic_tstat": None, "ic_hit_rate": None,
                "overlap_warning": horizon > 1}
    mean = float(ic.mean())
    sd = float(ic.std(ddof=1))
    ir = mean / sd if sd > 0 else 0.0
    tstat = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {
        "horizon": horizon,
        "n_obs": n,
        "mean_ic": round(mean, 4),
        "ic_std": round(sd, 4),
        "ic_ir": round(ir, 3),                      # IC information ratio (per-period)
        "ic_tstat": round(tstat, 2),
        "ic_hit_rate": round(float((ic > 0).mean()), 3),
        # overlapping forward returns (horizon>1) autocorrelate the IC series and
        # inflate this t-stat — flagged, not hidden.
        "overlap_warning": horizon > 1,
    }


def ic_decay(signal, close, horizons=(1, 5, 10, 21, 63), method: str = "spearman") -> dict:
    """IC at several forward horizons — the decay curve. A signal that only works at
    h=1 is fast-decaying (cost-sensitive); one that persists is more robust."""
    return {int(h): ic_summary(signal, close, int(h), method)["mean_ic"] for h in horizons}


def quantile_forward_returns(signal, close, n_quantiles: int = 5, horizon: int = 1) -> dict:
    """Mean forward return by signal quantile (averaged over dates). A genuine signal
    is roughly MONOTONIC across quantiles and has a positive top-minus-bottom spread."""
    fwd = _forward_return(close, horizon)
    buckets: dict[int, list] = {q: [] for q in range(n_quantiles)}
    for t in close.index:
        s = signal.reindex(columns=close.columns).loc[t] if signal.shape[1] != close.shape[1] else signal.loc[t]
        r = fwd.loc[t]
        m = s.notna() & r.notna()
        if int(m.sum()) < n_quantiles * 2:
            continue
        try:
            # rank first so a low-cardinality / tied signal still buckets evenly
            q = pd.qcut(s[m].rank(method="first"), n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        rr = r[m]
        for qi in range(n_quantiles):
            vals = rr[q == qi]
            if len(vals):
                buckets[qi].append(float(vals.mean()))
    means = [float(np.mean(v)) if v else float("nan") for v in buckets.values()]
    valid = [m for m in means if not np.isnan(m)]
    spread = (means[-1] - means[0]) if (not np.isnan(means[-1]) and not np.isnan(means[0])) else None
    monotone = len(valid) >= 2 and all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1))
    return {"quantile_mean_fwd_returns": [None if np.isnan(m) else round(m, 5) for m in means],
            "top_minus_bottom": None if spread is None else round(spread, 5),
            "monotonic_increasing": bool(monotone)}


def rank_autocorrelation(signal: pd.DataFrame, lag: int = 1) -> float:
    """Cross-sectional rank autocorrelation of the signal (turnover proxy). High =
    slow-moving = cheap to trade; low = noisy = expensive."""
    r = signal.rank(axis=1)
    prev = r.shift(lag)
    acs = []
    for t in r.index:
        a, b = r.loc[t], prev.loc[t]
        m = a.notna() & b.notna()
        if int(m.sum()) < 5:
            continue
        av, bv = a[m].to_numpy(), b[m].to_numpy()
        if av.std() > 0 and bv.std() > 0:
            acs.append(float(np.corrcoef(av, bv)[0, 1]))
    return round(float(np.mean(acs)), 3) if acs else float("nan")


def coverage(signal: pd.DataFrame) -> float:
    """Average fraction of the universe with a valid signal per date."""
    return round(float(signal.notna().mean(axis=1).mean()), 3)


def extraction_stability(repeats) -> dict:
    """Re-run consistency for (e.g.) LLM-extracted signals.

    ``repeats`` is shaped (n_items x n_runs): repeated extractions of the SAME item.
    Returns the within-item dispersion; high std = the model is guessing."""
    arr = np.asarray(repeats, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("repeats must be (n_items x n_runs) with n_runs >= 2")
    per_item_std = arr.std(axis=1, ddof=0)
    return {"mean_within_item_std": round(float(per_item_std.mean()), 4),
            "max_within_item_std": round(float(per_item_std.max()), 4),
            "deterministic": bool(per_item_std.max() < 1e-9)}


def scorecard(signal, close, horizons=(1, 5, 10, 21, 63), method: str = "spearman",
              split: float = 0.7, with_quantiles: bool = True) -> dict:
    """Full signal-quality verdict. A signal is 'predictive' only if its IC is
    statistically significant AND holds its sign out-of-sample."""
    head = ic_summary(signal, close, 1, method)
    decay = ic_decay(signal, close, horizons, method)
    cut = int(len(close) * split)
    is_ic = ic_summary(signal.iloc[:cut], close.iloc[:cut], 1, method)
    oos_ic = ic_summary(signal.iloc[cut:], close.iloc[cut:], 1, method)

    t = head.get("ic_tstat")
    is_m, oos_m = is_ic.get("mean_ic"), oos_ic.get("mean_ic")
    predictive = bool(
        t is not None and abs(t) > 2.0
        and is_m is not None and oos_m is not None
        and np.sign(is_m) == np.sign(oos_m) and abs(oos_m) > 0.0
    )
    verdict = ("PREDICTIVE: IC is significant and holds its sign out-of-sample"
               if predictive else
               "NOT PREDICTIVE: IC is not significant or flips out-of-sample")

    out = {
        "headline_ic": head,
        "ic_decay": decay,
        "in_sample_ic": is_ic.get("mean_ic"),
        "out_sample_ic": oos_ic.get("mean_ic"),
        "rank_autocorr": rank_autocorrelation(signal),
        "coverage": coverage(signal),
        "predictive": predictive,
        "verdict": verdict,
    }
    if with_quantiles:
        out["quantiles"] = quantile_forward_returns(signal, close, 5, 1)
    return out


def compact_scorecard(signal, close, method: str = "spearman") -> dict:
    """Lightweight version for the API response (3-horizon decay, no quantiles)."""
    head = ic_summary(signal, close, 1, method)
    decay = ic_decay(signal, close, (1, 5, 21), method)
    t = head.get("ic_tstat")
    return {
        "mean_ic": head.get("mean_ic"),
        "ic_tstat": t,
        "ic_ir": head.get("ic_ir"),
        "ic_hit_rate": head.get("ic_hit_rate"),
        "ic_decay": decay,
        "coverage": coverage(signal),
        "rank_autocorr": rank_autocorrelation(signal),
        "significant": bool(t is not None and abs(t) > 2.0),
    }
