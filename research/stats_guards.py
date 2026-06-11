"""'Won't let you lie to yourself' guards — the probability lessons, operationalised.

Each guard turns a classic statistical trap (from Blitzstein & Hwang) into an
automated check the platform runs so a user can't accidentally fool themselves:

  * base_rate_precision  — a high-hit-rate signal on a rare event is mostly false
    positives (base-rate fallacy / Bayes).
  * simpson_reversal     — an edge present in every regime can vanish or flip when
    pooled (Simpson's paradox); always disaggregate.
  * lookahead_warning    — a signal that correlates with *same-day* returns more
    than with *next-day* returns is probably peeking.
  * fat_tail_report      — returns are leptokurtic; Gaussian risk numbers lie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def base_rate_precision(base_rate: float, true_positive_rate: float, false_positive_rate: float) -> dict:
    """Given an event base rate and a signal's TPR/FPR, return the *precision*
    P(event | signal fires) via Bayes. The gap between recall (TPR) and precision
    is the base-rate fallacy made concrete."""
    if not 0 <= base_rate <= 1:
        raise ValueError("base_rate must be in [0, 1]")
    p_fire = true_positive_rate * base_rate + false_positive_rate * (1 - base_rate)
    precision = (true_positive_rate * base_rate / p_fire) if p_fire > 0 else float("nan")
    return {
        "base_rate": base_rate,
        "recall_tpr": true_positive_rate,
        "precision": precision,
        "p_signal_fires": p_fire,
        "lift": (precision / base_rate) if base_rate > 0 else float("nan"),
    }


def simpson_reversal(df: pd.DataFrame, group: str, treatment: str, outcome: str) -> dict:
    """Detect a Simpson's-paradox sign reversal between the pooled effect and the
    within-group effects of ``treatment`` on ``outcome``. ``treatment`` must be
    binary (0/1). Returns the pooled diff, the per-group diffs, and whether the
    pooled sign disagrees with the (consistent) within-group sign."""
    def diff(sub: pd.DataFrame) -> float:
        g = sub.groupby(treatment)[outcome].mean()
        if {0, 1}.issubset(g.index):
            return float(g.loc[1] - g.loc[0])
        return np.nan

    pooled = diff(df)
    per_group = {str(k): diff(sub) for k, sub in df.groupby(group)}
    grp_vals = [v for v in per_group.values() if not np.isnan(v)]
    consistent = len(grp_vals) > 0 and (all(v > 0 for v in grp_vals) or all(v < 0 for v in grp_vals))
    reversal = bool(consistent and not np.isnan(pooled) and np.sign(pooled) != np.sign(grp_vals[0]))
    return {"pooled_effect": pooled, "within_group_effects": per_group,
            "within_group_consistent": consistent, "simpson_reversal": reversal}


def lookahead_warning(signal: pd.Series, returns: pd.Series, margin: float = 0.05) -> dict:
    """Flag possible look-ahead: a sound predictive signal should correlate with
    *future* returns at least as much as with *contemporaneous* returns. If the
    same-day correlation materially exceeds the next-day correlation, the signal
    likely embeds information it shouldn't have at decision time."""
    s = signal.reindex(returns.index)
    same_day = float(s.corr(returns))
    next_day = float(s.corr(returns.shift(-1)))
    suspicious = bool(abs(same_day) > abs(next_day) + margin and abs(same_day) > 0.1)
    return {"corr_same_day": same_day, "corr_next_day": next_day,
            "suspected_lookahead": suspicious}


def fat_tail_report(returns) -> dict:
    """Skew, excess kurtosis, Jarque-Bera, and a plain-language verdict on whether
    Gaussian risk estimates (parametric VaR etc.) can be trusted."""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    if r.size < 8:
        return {"verdict": "insufficient data"}
    sk = float(stats.skew(r, bias=False))
    ek = float(stats.kurtosis(r, fisher=True, bias=False))
    jb_p = float(stats.jarque_bera(r).pvalue)
    normal = jb_p >= 0.05
    return {
        "skew": sk,
        "excess_kurtosis": ek,
        "jarque_bera_p": jb_p,
        "returns_are_normal": normal,
        "verdict": ("returns look Normal; Gaussian risk numbers are defensible"
                    if normal else
                    "FAT TAILS / skew detected -- Gaussian VaR understates tail risk; "
                    "use historical-simulation or Student-t"),
    }
