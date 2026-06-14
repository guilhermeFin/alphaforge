"""Tests for the per-workspace trial-count ledger.

The contract under test: the multiple-testing haircut (Deflated Sharpe) cannot be
silently bypassed. Distinct configurations actually run set a FLOOR on n_trials;
fingerprints are stable and ignore the self-reported n_trials; resets are logged.
"""
from research.trial_ledger import (
    TRIAL_KEYS,
    TrialLedger,
    trial_fingerprint,
)


def _base_req() -> dict:
    return {
        "provider": "synthetic",
        "factor": "momentum",
        "lookback": 126,
        "skip": 21,
        "cost_bps": 5.0,
        "gross": 1.0,
        "periods": 252,
        "seed": 7,
        "start": "2010-01-01",
        "symbols": ["AAPL", "MSFT", "GOOG"],
        "n_trials": 1,
    }


# ---------------------------------------------------------------------------
# Fingerprint behaviour
# ---------------------------------------------------------------------------

def test_fingerprint_ignores_declared_n_trials():
    a = _base_req()
    b = _base_req()
    b["n_trials"] = 999
    assert trial_fingerprint(a) == trial_fingerprint(b)


def test_fingerprint_ignores_unknown_extra_keys():
    a = _base_req()
    b = _base_req()
    b["note"] = "this should not matter"
    b["requested_by"] = "guilherme"
    assert trial_fingerprint(a) == trial_fingerprint(b)


def test_fingerprint_symbol_order_insensitive():
    a = _base_req()
    b = _base_req()
    b["symbols"] = ["GOOG", "AAPL", "MSFT"]  # reordered
    assert trial_fingerprint(a) == trial_fingerprint(b)


def test_fingerprint_symbol_case_and_whitespace_insensitive():
    a = _base_req()
    b = _base_req()
    b["symbols"] = ["  aapl ", "msft", " Goog"]
    assert trial_fingerprint(a) == trial_fingerprint(b)


def test_fingerprint_numeric_coercion():
    a = _base_req()
    b = _base_req()
    b["lookback"] = 126.0   # float vs int
    b["seed"] = "7"          # string vs int
    b["cost_bps"] = 5         # int vs float
    assert trial_fingerprint(a) == trial_fingerprint(b)


def test_fingerprint_changes_with_every_trial_knob():
    base = _base_req()
    base_fp = trial_fingerprint(base)
    perturbations = {
        "provider": "ibkr",
        "factor": "reversal",
        "lookback": 200,
        "skip": 5,
        "cost_bps": 10.0,
        "gross": 2.0,
        "periods": 12,
        "seed": 8,
        "start": "2015-06-01",
        "symbols": ["AAPL", "MSFT"],  # different SET, not just order
    }
    for key in TRIAL_KEYS:
        req = _base_req()
        req[key] = perturbations[key]
        assert trial_fingerprint(req) != base_fp, f"knob {key!r} did not change fingerprint"


def test_fingerprint_seed_change_is_a_new_trial():
    a = _base_req()
    b = _base_req()
    b["seed"] = a["seed"] + 1
    assert trial_fingerprint(a) != trial_fingerprint(b)


# ---------------------------------------------------------------------------
# Ledger recording
# ---------------------------------------------------------------------------

def test_rerun_identical_config_does_not_raise_distinct_but_increments_total():
    led = TrialLedger()
    s1 = led.record(_base_req())
    assert s1["is_new_config"] is True
    assert s1["rerun_of_prior"] is False
    assert s1["distinct_count"] == 1
    assert s1["total_runs"] == 1

    s2 = led.record(_base_req())  # exact same config
    assert s2["is_new_config"] is False
    assert s2["rerun_of_prior"] is True
    assert s2["distinct_count"] == 1          # distinct count unchanged
    assert s2["total_runs"] == 2              # but total runs grew

    assert led.distinct_count == 1
    assert led.total_runs == 2


def test_distinct_count_grows_with_new_configs():
    led = TrialLedger()
    led.record(_base_req())
    r2 = _base_req()
    r2["seed"] = 99
    led.record(r2)
    r3 = _base_req()
    r3["factor"] = "reversal"
    led.record(r3)
    assert led.distinct_count == 3
    assert led.total_runs == 3


def test_snapshot_reports_fingerprint():
    led = TrialLedger()
    snap = led.record(_base_req())
    assert snap["fingerprint"] == trial_fingerprint(_base_req())
    assert isinstance(snap["fingerprint"], str)
    assert len(snap["fingerprint"]) == 16


# ---------------------------------------------------------------------------
# effective_n_trials — the floor
# ---------------------------------------------------------------------------

def test_effective_n_trials_is_a_floor_over_distinct():
    led = TrialLedger()
    for seed in range(5):
        req = _base_req()
        req["seed"] = seed
        led.record(req)
    assert led.distinct_count == 5
    # Declaring fewer than we actually ran is overruled by the floor.
    assert led.effective_n_trials(1) == 5
    assert led.effective_n_trials(3) == 5


def test_effective_n_trials_respects_honest_over_declaration():
    led = TrialLedger()
    led.record(_base_req())
    assert led.distinct_count == 1
    # Honest paper-trials run outside the tool are respected (floor not a cap).
    assert led.effective_n_trials(50) == 50


def test_effective_n_trials_at_least_one():
    led = TrialLedger()  # nothing recorded yet -> distinct_count == 0
    assert led.effective_n_trials(0) >= 1
    assert led.effective_n_trials(0) == 1
    assert led.effective_n_trials(-5) == 1


def test_effective_n_trials_non_numeric_declared_falls_back_to_floor():
    led = TrialLedger()
    led.record(_base_req())
    r2 = _base_req()
    r2["seed"] = 2
    led.record(r2)
    assert led.effective_n_trials(None) == 2
    assert led.effective_n_trials("garbage") == 2


# ---------------------------------------------------------------------------
# reset — never silent
# ---------------------------------------------------------------------------

def test_reset_is_logged_and_counts_zeroed():
    led = TrialLedger()
    led.record(_base_req())
    r2 = _base_req()
    r2["seed"] = 42
    led.record(r2)
    led.record(_base_req())  # rerun -> total 3, distinct 2
    assert led.distinct_count == 2
    assert led.total_runs == 3

    entry = led.reset(reason="new research question")
    # the discarded counts are captured in the audit record
    assert entry["discarded_distinct"] == 2
    assert entry["discarded_total"] == 3
    assert entry["reason"] == "new research question"
    assert "at" in entry

    # counts zeroed
    assert led.distinct_count == 0
    assert led.total_runs == 0
    # but the log grows and snapshot reports it
    assert len(led.reset_log) == 1
    assert led.snapshot()["n_resets"] == 1


def test_reset_log_is_append_only_across_multiple_resets():
    led = TrialLedger()
    led.record(_base_req())
    led.reset(reason="first")
    led.record(_base_req())
    led.reset(reason="second")
    assert len(led.reset_log) == 2
    assert led.reset_log[0]["reason"] == "first"
    assert led.reset_log[1]["reason"] == "second"
    assert led.snapshot()["n_resets"] == 2


def test_snapshot_shape():
    led = TrialLedger()
    led.record(_base_req())
    snap = led.snapshot()
    assert set(snap.keys()) == {"distinct_count", "total_runs", "session_age_sec", "n_resets"}
    assert snap["distinct_count"] == 1
    assert snap["total_runs"] == 1
    assert snap["session_age_sec"] >= 0.0
    assert snap["n_resets"] == 0
