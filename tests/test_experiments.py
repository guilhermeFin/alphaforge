import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")

from research import data
from research.experiments import run_econometrics_ml_experiment
from research.fundamentals import build_fundamentals
from research.trial_ledger import TrialLedger


def _synthetic_inputs(periods=620, n_symbols=18, seed=17):
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    world = data.make_synthetic_world(
        symbols,
        periods=periods,
        seed=seed,
        quality_to_drift=0.0012,
    )
    obs = data.make_synthetic_raw_fundamentals(world, seed=seed)
    fund = build_fundamentals(obs, world.close.index, world.symbols)
    return world, fund


def test_econometrics_ml_experiment_shape_and_separated_outputs():
    world, fund = _synthetic_inputs()

    out = run_econometrics_ml_experiment(
        world.close,
        fund,
        ["gross_profitability", "roe", "book_to_price"],
        horizon=21,
        model_names=["elastic_net"],
        n_splits=4,
        pbo_splits=4,
    )

    assert out["meta"]["baseline"] == "elastic_net"
    assert out["meta"]["synthetic_data_note"].startswith("Synthetic results validate")

    econ = out["econometric_inference"]
    assert {"pooled_ols_hac", "fama_macbeth"} <= set(econ)
    assert econ["pooled_ols_hac"]["covariance_type"] == "newey_west(21)"
    assert econ["fama_macbeth"]["covariance_type"] == "newey_west(21)"

    ml = out["ml_ladder"]
    assert ml["baseline"] == "elastic_net"
    assert len(ml["leaderboard"]) == 1

    rows = out["predictive_oos"]["leaderboard"]
    assert {row["method"] for row in rows} == {
        "pooled_ols_hac",
        "fama_macbeth",
        "elastic_net",
    }
    for row in rows:
        assert row["n_oos_predictions"] > 0
        assert row["costed_scorecard"]["n_trials"] == 3
        assert "walk_forward" in row
        assert row["oos_rank_ic"] is None or np.isfinite(row["oos_rank_ic"])

    assert out["trial_audit"]["candidate_trials"] == 3
    assert out["trial_audit"]["effective_n_trials"] == 3
    assert out["pbo"]["n_strategies"] == 3


def test_experiment_records_each_candidate_in_trial_ledger():
    world, fund = _synthetic_inputs(periods=540, n_symbols=16, seed=23)
    ledger = TrialLedger()

    out = run_econometrics_ml_experiment(
        world.close,
        fund,
        ["gross_profitability", "roe"],
        horizon=21,
        model_names=["elastic_net"],
        n_splits=4,
        pbo_splits=4,
        declared_n_trials=1,
        ledger=ledger,
        request_context={"provider": "synthetic", "seed": 23},
    )

    assert ledger.distinct_count == 3
    assert out["trial_audit"]["effective_n_trials"] == 3
    assert out["trial_audit"]["haircut_was_raised"] is True
    assert len(out["trial_audit"]["records"]) == 3


def test_experiment_rejects_unknown_model_and_spec():
    world, fund = _synthetic_inputs(periods=500, n_symbols=12)

    with pytest.raises(ValueError, match="unknown model"):
        run_econometrics_ml_experiment(
            world.close,
            fund,
            ["gross_profitability"],
            model_names=["elastic_net", "sparkly_tree"],
        )

    with pytest.raises(ValueError, match="unknown econometric spec"):
        run_econometrics_ml_experiment(
            world.close,
            fund,
            ["gross_profitability"],
            model_names=["elastic_net"],
            econometric_specs=["pooled_ols_hac", "magic_beta"],
        )
