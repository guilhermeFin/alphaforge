"""Per-workspace trial-count ledger — the multiple-testing conscience.

The Deflated Sharpe Ratio (``metrics.deflated_sharpe_ratio``) only deflates as
hard as the ``n_trials`` you feed it. If that number is supplied by hand it is
trivially gamed: run a strategy 200 times, sweeping factors / lookbacks / seeds,
then declare ``n_trials=1`` and report a gorgeous (and entirely spurious) PSR.

This ledger removes that escape hatch. Every backtest request is fingerprinted
over the knobs that constitute a genuine *trial* (a distinct search-space point),
and the count of *distinct* configurations actually run becomes a FLOOR on the
``n_trials`` used to deflate. You can still honestly *over*-declare paper-trials
you ran outside the tool — the floor only forbids under-counting.

Design notes
------------
* Pure stdlib (``hashlib``, ``json``, ``time``, ``dataclasses``) — no numpy /
  pandas. This module must stay importable and cheap everywhere.
* ``trial_fingerprint`` is stable across processes and machines: same logical
  request -> same hash. It deliberately EXCLUDES the declared ``n_trials`` (that
  field is the thing we are policing — letting it into the fingerprint would let
  a user mint a "new" trial just by changing the number they self-report).
* ``seed`` IS part of the fingerprint: sweeping random seeds to fish for a lucky
  draw is real multiple testing and must inflate the haircut.
* Resets are never silent — every reset is appended to an immutable log so a
  cleared count leaves a visible trail.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

# The knobs that define a distinct trial (a point in the search space).
# Deliberately does NOT include "n_trials" — that is the self-reported number we
# are guarding against, and folding it in would let a user fabricate new trials
# simply by changing the count they claim.
TRIAL_KEYS = (
    "provider",
    "factor",
    "lookback",
    "skip",
    "cost_bps",
    "gross",
    "periods",
    "seed",
    "start",
    "symbols",
    # Portfolio controls are genuine additional variants; leaving them out would
    # let a user sweep capacity/cost/rebalance assumptions without a trial count.
    "rebalance_frequency",
    "max_name_weight",
    "max_turnover",
    "target_annual_vol",
    "spread_bps",
    "impact_bps",
    "short_borrow_bps",
    "max_participation",
    "adv_lookback",
    "capital",
)

# Keys whose values are coerced to a numeric (float) canonical form so that, e.g.,
# 126 and 126.0 and "126" fingerprint identically. ``seed`` is kept as an int when
# integral so a seed of 0 vs 0.0 vs "0" all agree.
_NUMERIC_KEYS = ("lookback", "skip", "cost_bps", "gross", "periods", "seed",
                 "max_name_weight", "max_turnover", "target_annual_vol", "spread_bps",
                 "impact_bps", "short_borrow_bps", "max_participation", "adv_lookback", "capital")


def _canon_symbols(value) -> list:
    """Order-insensitive, case-insensitive canonical symbol list.

    Accepts a single symbol string or any iterable of symbols. Each symbol is
    stripped, upper-cased; blanks dropped; the result is sorted so symbol order
    in the request never changes the fingerprint.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    out = []
    for s in items:
        if s is None:
            continue
        token = str(s).strip().upper()
        if token:
            out.append(token)
    return sorted(out)


def _canon_numeric(value):
    """Coerce a numeric knob to a canonical JSON-stable scalar.

    Integral values collapse to ``int`` (so 126 == 126.0 == "126"); genuine
    fractions stay ``float``. Non-coercible / missing values pass through as a
    normalised string so two requests still agree, rather than crashing.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if f != f:  # NaN — never equal to itself; canonicalise to a stable token
        return "nan"
    if f == int(f):
        return int(f)
    return f


def _canon_request(req: dict) -> dict:
    """Project an arbitrary request dict onto the canonical TRIAL_KEYS subspace.

    Only the trial-defining keys survive; everything else (notably ``n_trials``)
    is dropped. Each surviving value is normalised so logically-equal requests
    map to byte-identical JSON.
    """
    req = req or {}
    canon: dict = {}
    for key in TRIAL_KEYS:
        if key == "symbols":
            canon[key] = _canon_symbols(req.get(key))
        elif key in _NUMERIC_KEYS:
            canon[key] = _canon_numeric(req.get(key))
        else:
            value = req.get(key)
            canon[key] = None if value is None else str(value).strip()
    return canon


def trial_fingerprint(req: dict) -> str:
    """Stable short sha256 fingerprint of a backtest request's *trial identity*.

    Two requests share a fingerprint iff they describe the same point in the
    search space (same provider/factor/lookback/skip/cost/gross/periods/seed/start,
    portfolio construction, execution assumptions, and the same *set* of symbols).
    The declared ``n_trials`` is ignored; symbol order and case are ignored; numeric
    knobs are coerced so 126 == 126.0.
    """
    canon = _canon_request(req)
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class TrialLedger:
    """Append-only, per-workspace tally of distinct backtest configurations run.

    The whole point: ``effective_n_trials`` returns a FLOOR (never below the
    number of distinct configs actually executed here), so the DSR haircut cannot
    be dodged by self-reporting a smaller ``n_trials``.
    """

    _counts: dict = field(default_factory=dict)
    total_runs: int = 0
    reset_log: list = field(default_factory=list)
    _started_at: float = field(default_factory=time.time)

    def record(self, req: dict) -> dict:
        """Register one backtest run; return a snapshot of the run's effect.

        Increments ``total_runs`` always. Increments the distinct-config count
        only the first time a given fingerprint is seen. Returns a dict with the
        ``fingerprint``, whether this was a brand-new configuration
        (``is_new_config``), the running ``distinct_count`` and ``total_runs``,
        and ``rerun_of_prior`` (True when this fingerprint has been run before).
        """
        fp = trial_fingerprint(req)
        prior = self._counts.get(fp, 0)
        is_new = prior == 0
        self._counts[fp] = prior + 1
        self.total_runs += 1
        return {
            "fingerprint": fp,
            "is_new_config": is_new,
            "distinct_count": self.distinct_count,
            "total_runs": self.total_runs,
            "rerun_of_prior": not is_new,
        }

    @property
    def distinct_count(self) -> int:
        """Number of distinct trial configurations recorded since the last reset."""
        return len(self._counts)

    def effective_n_trials(self, declared) -> int:
        """The ``n_trials`` to actually deflate by — a floor, not a replacement.

        ``max(int(declared), distinct_count, 1)``. Honest *over*-declaration of
        paper-trials run outside the tool is respected (we never lower the number
        you claim); but you can never under-count below the configurations this
        ledger has actually seen, and the result is always at least 1.
        """
        try:
            declared_int = int(declared)
        except (TypeError, ValueError):
            declared_int = 0
        return max(declared_int, self.distinct_count, 1)

    def reset(self, reason: str = "") -> dict:
        """Zero the counts, but never silently — append an audit record first.

        The discarded ``distinct`` and ``total`` counts, a timestamp, and the
        caller-supplied ``reason`` are appended to the immutable ``reset_log``;
        the live counters are then cleared. Returns the audit record just added.
        """
        entry = {
            "at": time.time(),
            "discarded_distinct": self.distinct_count,
            "discarded_total": self.total_runs,
            "reason": reason,
        }
        self.reset_log.append(entry)
        self._counts = {}
        self.total_runs = 0
        return entry

    def snapshot(self) -> dict:
        """A read-only view of ledger state (no mutation)."""
        return {
            "distinct_count": self.distinct_count,
            "total_runs": self.total_runs,
            "session_age_sec": time.time() - self._started_at,
            "n_resets": len(self.reset_log),
        }
