"""The research fixture must let factors behave, and the flat one must not.

The standard these tests hold to:

* **Fixture backtest performance is not a research finding.** Six names on one
  deterministic path can come out positive or negative by chance. Neither
  outcome is informative, and neither indicates a problem with the fixture.
* **No test asserts return sign, magnitude, Sharpe, or credibility**, and none
  should ever be changed to.
* The fixture exists to validate *mechanics*: point-in-time handling,
  cross-sectional dispersion, active-position formation, cost behaviour, and
  diagnostic reporting.

Independence from price generation is asserted structurally, by perturbing each
driver and checking what does and does not change, rather than by thresholding a
correlation statistic.
"""
import numpy as np
import pandas as pd
import pytest

from api.service import FACTORLIB_SPECS, run_backtest_workflow
from research import factor_lib, fundamentals, licensed_data
from tests.research_fixtures import (
    ASSET_BASE,
    MARGIN,
    REPORTING_LAG_DAYS,
    SYMBOLS,
    TURNOVER,
    write_degenerate_bundle,
    write_research_bundle,
)

#: Composites the workflow builds directly rather than through FACTORLIB_SPECS.
COMPOSITES = {"value_score": "value_score", "quality_score": "quality_score"}

#: Below this, a cross-section cannot rank names apart in any useful way.
MIN_DISPERSION = 1e-9


def _panels(root, symbols=SYMBOLS):
    close, _volume = licensed_data.load_price_panel(list(symbols), root)
    close = close.dropna(how="all", axis=1)
    observations = licensed_data.load_fundamentals(list(close.columns), root)
    fund = fundamentals.build_fundamentals(observations, close.index, list(close.columns))
    return fund, close


def _mean_dispersion(panel) -> float:
    per_date = pd.DataFrame(panel).std(axis=1, ddof=1).dropna()
    return float(per_date.mean()) if len(per_date) else float("nan")


def _all_specs() -> dict[str, str]:
    specs = {name: spec[0] for name, spec in FACTORLIB_SPECS.items()}
    specs.update(COMPOSITES)
    return specs


# ------------------------------------------------- determinism and shape


def test_bundle_is_byte_identical_across_runs(tmp_path):
    """No randomness: two writes of the same fixture must agree exactly."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    write_research_bundle(first)
    write_research_bundle(second)
    for name in ("prices.csv", "fundamentals.csv", "universe.csv", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_bundle_is_small_enough_to_reason_about(tmp_path):
    write_research_bundle(tmp_path)
    prices = pd.read_csv(tmp_path / "prices.csv")
    facts = pd.read_csv(tmp_path / "fundamentals.csv")
    assert prices["symbol"].nunique() == len(SYMBOLS) == 6
    assert len(prices) == 520 * 6
    # Enough periods for a factor to change its ranking, few enough to inspect.
    assert 5 <= facts["period_end"].nunique() <= 10


def test_fundamental_values_are_hand_checkable(tmp_path):
    """The first period's gross profitability is exactly turnover * margin / 4."""
    write_research_bundle(tmp_path)
    facts = pd.read_csv(tmp_path / "fundamentals.csv")
    first_period = facts[facts["period_end"] == facts["period_end"].min()]
    wide = first_period.pivot_table(index="symbol", columns="metric", values="value")
    for j, symbol in enumerate(SYMBOLS):
        row = wide.loc[symbol]
        observed = (row["revenue"] - row["cogs"]) / row["total_assets"]
        assert observed == pytest.approx(TURNOVER[j] * MARGIN[j] / 4.0)


def test_size_does_not_leak_into_the_ratio(tmp_path):
    """Company size must carry no factor information: that was the old defect."""
    write_research_bundle(tmp_path)
    facts = pd.read_csv(tmp_path / "fundamentals.csv")
    first = facts[facts["period_end"] == facts["period_end"].min()]
    wide = first.pivot_table(index="symbol", columns="metric", values="value")
    gp = ((wide["revenue"] - wide["cogs"]) / wide["total_assets"]).reindex(list(SYMBOLS))
    size = pd.Series(ASSET_BASE, index=list(SYMBOLS))
    # Rank correlation between size and the factor is weak by construction.
    assert abs(gp.rank().corr(size.rank())) < 0.60


# --------------------------------------------------------- point-in-time


def test_fundamentals_are_not_knowable_before_they_are_filed(tmp_path):
    write_research_bundle(tmp_path)
    facts = pd.read_csv(tmp_path / "fundamentals.csv", parse_dates=["period_end", "available_date"])
    lag = (facts["available_date"] - facts["period_end"]).dt.days
    assert (lag == REPORTING_LAG_DAYS).all()
    assert (facts["available_date"] > facts["period_end"]).all()


def test_no_fundamental_is_available_after_the_price_history_ends(tmp_path):
    write_research_bundle(tmp_path)
    prices = pd.read_csv(tmp_path / "prices.csv", parse_dates=["date"])
    facts = pd.read_csv(tmp_path / "fundamentals.csv", parse_dates=["available_date"])
    assert facts["available_date"].max() <= prices["date"].max()


# -------------------------------------------------- factor-library audit


def test_every_registered_factor_has_cross_sectional_dispersion(tmp_path):
    """Audit: name any factor that is constant across the universe.

    A constant factor cannot rank anything, so a test using it proves nothing.
    This asserts the inputs permit ranking. It deliberately says nothing about
    whether any factor predicts returns or earns a profit.
    """
    write_research_bundle(tmp_path)
    fund, close = _panels(tmp_path)
    flat = {}
    for name, fn_name in sorted(_all_specs().items()):
        dispersion = _mean_dispersion(getattr(factor_lib, fn_name)(fund, close))
        if not (np.isfinite(dispersion) and dispersion > MIN_DISPERSION):
            flat[name] = dispersion
    assert not flat, f"factors constant on the realistic fixture: {flat}"


def test_the_audit_would_catch_a_flat_factor(tmp_path):
    """The audit must be able to fail: on the flat fixture, ratio factors collapse."""
    write_degenerate_bundle(tmp_path)
    fund, close = _panels(tmp_path, symbols=("AAA", "BBB", "CCC"))
    flat = [name for name, fn_name in sorted(_all_specs().items())
            if not _mean_dispersion(getattr(factor_lib, fn_name)(fund, close)) > MIN_DISPERSION]
    # Pure fundamental ratios cancel there; price-based ones still vary.
    assert "gross_profitability" in flat
    assert "roe" in flat and "roa" in flat
    assert "book_to_price" not in flat


# ------------------------------------------- factors can actually trade


@pytest.mark.parametrize("factor", ["gross_profitability", "quality"])
def test_factor_forms_non_flat_ranks_and_active_positions(monkeypatch, tmp_path, factor):
    """A ratio factor and a composite must rank names and hold something.

    This asserts the analysis ran on a usable sample. It makes no claim about
    the sign or size of any return.
    """
    write_research_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    result = run_backtest_workflow({
        "provider": "licensed_bundle", "symbols": list(SYMBOLS), "factor": factor,
        "periods": 520, "lookback": 126, "skip": 21, "n_trials": 5,
        "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    })
    validity = result["strategy_report"]["research_validity"]
    assert validity["signal_dispersion"]["is_degenerate"] is False
    assert validity["positions"]["average_active_names"] > 0
    assert validity["returns"]["non_flat_returns"] > 0
    assert validity["supports_performance_claim"] is True
    # Explicitly NOT asserted: any statement about profitability or credibility.
    assert "verdict" in result


def test_momentum_still_behaves_on_the_realistic_fixture(monkeypatch, tmp_path):
    write_research_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    result = run_backtest_workflow({
        "provider": "licensed_bundle", "symbols": list(SYMBOLS), "factor": "momentum",
        "periods": 520, "lookback": 126, "skip": 21, "n_trials": 5,
        "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    })
    assert result["meta"]["provider"] == "licensed_bundle"
    assert result["meta"]["n_symbols"] == len(SYMBOLS)
    validity = result["strategy_report"]["research_validity"]
    assert validity["positions"]["average_active_names"] > 0
    assert validity["returns"]["non_flat_returns"] > 0


def test_diagnostics_separate_a_healthy_sample_from_a_flat_one(monkeypatch, tmp_path):
    healthy, flat = tmp_path / "healthy", tmp_path / "flat"
    healthy.mkdir()
    flat.mkdir()
    write_research_bundle(healthy)
    write_degenerate_bundle(flat)
    request = {
        "provider": "licensed_bundle", "factor": "gross_profitability",
        "periods": 520, "lookback": 126, "skip": 21, "n_trials": 5,
        "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    }
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(healthy))
    good = run_backtest_workflow({**request, "symbols": list(SYMBOLS)})
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(flat))
    bad = run_backtest_workflow({**request, "symbols": ["AAA", "BBB", "CCC"]})

    assert good["strategy_report"]["research_validity"]["supports_performance_claim"] is True
    assert bad["strategy_report"]["research_validity"]["supports_performance_claim"] is False
    assert "insufficient variation" in bad["strategy_report"]["research_validity"]["conclusion"]


# ------------------------------------ the fixture must not plant an edge

#: Every constant that feeds the fundamentals, and a perturbed replacement.
FUNDAMENTAL_DRIVERS = {
    "TURNOVER": (0.41, 1.22, 0.58, 0.97, 0.66, 1.13),
    "MARGIN": (0.51, 0.19, 0.44, 0.23, 0.30, 0.47),
    "TURNOVER_DRIFT": (-0.009, 0.007, -0.003, -0.001, 0.006, -0.008),
    "MARGIN_DRIFT": (-0.006, 0.005, -0.002, -0.007, 0.003, -0.001),
    "ASSET_BASE": (9.1e9, 4.4e8, 1.7e10, 2.2e9, 3.8e10, 7.3e8),
    "ASSET_GROWTH": (0.021, 0.004, 0.017, 0.009, 0.026, 0.002),
    "SHARE_BASE": (2.7e8, 6.1e7, 4.9e8, 8.8e7, 1.3e8, 5.2e8),
    "EQUITY_RATIO": (0.29, 0.71, 0.38, 0.66, 0.44, 0.58),
    "SGA_RATE": (0.22, 0.06, 0.19, 0.08, 0.16, 0.11),
    "OCF_RATIO": (0.77, 1.41, 0.88, 1.29, 0.95, 1.12),
    "CAPEX_RATE": (0.12, 0.02, 0.10, 0.05, 0.08, 0.03),
    "ISSUANCE": (-0.004, 0.003, -0.002, 0.005, 0.001, -0.005),
}

#: Every constant that feeds price generation, and a perturbed replacement.
PRICE_DRIVERS = {
    "PRICE_BASE": (77.0, 31.0, 205.0, 54.0, 19.0, 143.0),
    "PRICE_TREND": (-0.21, 0.27, -0.03, 0.16, 0.08, -0.14),
    "PRICE_AMP": (0.11, 0.04, 0.13, 0.06, 0.08, 0.10),
    "PRICE_PERIOD": (47, 69, 35, 58, 72, 41),
    "PRICE_PHASE": (0.6, 0.1, 0.45, 0.8, 0.25, 0.05),
}


def _bundle_bytes(root) -> tuple[bytes, bytes]:
    return ((root / "prices.csv").read_bytes(), (root / "fundamentals.csv").read_bytes())


@pytest.mark.parametrize("driver", sorted(FUNDAMENTAL_DRIVERS))
def test_price_generation_never_reads_a_fundamental_driver(monkeypatch, tmp_path, driver):
    """Structural independence: no fundamental input can reach a price.

    Perturbing a fundamental driver must change the fundamentals and leave every
    price byte-identical. This asserts the causal wiring directly rather than
    inferring it from a correlation statistic, so there is no threshold to tune
    and no way for an incidental deterministic alignment to fail it.
    """
    from tests import research_fixtures

    baseline, perturbed = tmp_path / "base", tmp_path / "perturbed"
    baseline.mkdir()
    perturbed.mkdir()
    write_research_bundle(baseline)

    monkeypatch.setattr(research_fixtures, driver, FUNDAMENTAL_DRIVERS[driver])
    write_research_bundle(perturbed)

    base_prices, base_facts = _bundle_bytes(baseline)
    new_prices, new_facts = _bundle_bytes(perturbed)
    assert new_prices == base_prices, f"{driver} leaked into price generation"
    # The perturbation must be real, or the test above proves nothing.
    assert new_facts != base_facts, f"{driver} had no effect on the fundamentals"


@pytest.mark.parametrize("driver", sorted(PRICE_DRIVERS))
def test_fundamentals_never_read_a_price_driver(monkeypatch, tmp_path, driver):
    """The converse direction: prices must not feed the reported fundamentals."""
    from tests import research_fixtures

    baseline, perturbed = tmp_path / "base", tmp_path / "perturbed"
    baseline.mkdir()
    perturbed.mkdir()
    write_research_bundle(baseline)

    monkeypatch.setattr(research_fixtures, driver, PRICE_DRIVERS[driver])
    write_research_bundle(perturbed)

    base_prices, base_facts = _bundle_bytes(baseline)
    new_prices, new_facts = _bundle_bytes(perturbed)
    assert new_facts == base_facts, f"{driver} leaked into fundamentals generation"
    assert new_prices != base_prices, f"{driver} had no effect on the prices"


def test_fixture_performance_is_not_treated_as_a_research_finding(monkeypatch, tmp_path):
    """Only mechanics are asserted, never the outcome.

    Six names on one deterministic path can come out positive or negative by
    chance; neither is informative. So this checks that a verdict is reachable
    and honestly labelled, and deliberately asserts nothing about return sign,
    magnitude, Sharpe, or credibility.
    """
    write_research_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    result = run_backtest_workflow({
        "provider": "licensed_bundle", "symbols": list(SYMBOLS),
        "factor": "gross_profitability", "periods": 520, "lookback": 126,
        "skip": 21, "n_trials": 5, "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    })
    assert result["verdict"].startswith(("CREDIBLE", "NOT CREDIBLE"))
    assert result["data_audit"]["status"] in {"supported", "limited", "exploratory"}
    # Present and well-formed; their values are not the fixture's business.
    for field in ("cagr", "ann_sharpe", "deflated_sr"):
        assert field in result["scorecard"]


# ------------------------------- existing integrity constraints hold


def test_cost_and_point_in_time_constraints_survive_the_new_fixture(monkeypatch, tmp_path):
    write_research_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    base = {
        "provider": "licensed_bundle", "symbols": list(SYMBOLS),
        "factor": "gross_profitability", "periods": 520, "lookback": 126,
        "skip": 21, "n_trials": 5, "start": "2020-01-02", "gross": 1.0,
    }
    cheap = run_backtest_workflow({**base, "cost_bps": 1.0})
    dear = run_backtest_workflow({**base, "cost_bps": 50.0})
    # Costs must bite: a higher cost assumption cannot improve the net result.
    assert dear["scorecard"]["cagr"] <= cheap["scorecard"]["cagr"] + 1e-12
    assert dear["scorecard"]["total_return"] <= cheap["scorecard"]["total_return"] + 1e-12
    assert cheap["attribution"]["fundamentals_used"] is True


def test_licensed_bundle_evidence_status_is_unchanged_by_the_fixture(monkeypatch, tmp_path):
    write_research_bundle(tmp_path)
    monkeypatch.setenv(licensed_data.BUNDLE_ENV, str(tmp_path))
    result = run_backtest_workflow({
        "provider": "licensed_bundle", "symbols": list(SYMBOLS), "factor": "momentum",
        "periods": 520, "lookback": 126, "skip": 21, "n_trials": 5,
        "start": "2020-01-02", "cost_bps": 5.0, "gross": 1.0,
    })
    audit = result["data_audit"]
    assert audit["provider"] == "licensed_bundle"
    # The manifest attests all three, so the bundle reaches 'supported'.
    assert audit["historical_universe_membership"] is True
    assert audit["delisted_securities"] is True
    assert audit["corporate_actions_adjusted"] is True
