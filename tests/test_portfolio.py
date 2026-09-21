import numpy as np
import pandas as pd

from research import portfolio


def _scores(periods: int = 50) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    return pd.DataFrame({"A": np.linspace(-2, 2, periods), "B": np.linspace(-1, 1, periods),
                         "C": np.linspace(2, -2, periods), "D": np.linspace(1, -1, periods)}, index=index)


def test_constrained_weights_are_dollar_neutral_and_capped():
    weights = portfolio.constrained_long_short_weights(_scores(), gross=1.0, max_name_weight=0.15)
    assert (weights.abs().max(axis=1) <= 0.15 + 1e-12).all()
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-12).all()


def test_monthly_schedule_does_not_select_an_incomplete_month_end_from_future_rows():
    index = pd.bdate_range("2024-01-02", "2024-02-15")
    weights = pd.DataFrame({"A": np.arange(len(index), dtype=float), "B": -np.arange(len(index), dtype=float)}, index=index)
    scheduled = portfolio.apply_rebalance_schedule(weights, "monthly")
    # The first date is allowed to establish a starting book; the partial February
    # sample never becomes a new monthly rebalance merely because it is the last row.
    assert (scheduled.loc["2024-02-01":] == scheduled.loc["2024-01-31"].to_numpy()).all().all()


def test_turnover_cap_limits_each_step():
    index = pd.bdate_range("2024-01-02", periods=3)
    desired = pd.DataFrame([[0.0, 0.0], [0.8, -0.8], [-0.8, 0.8]], index=index, columns=["A", "B"])
    actual = portfolio.limit_turnover(desired, 0.2)
    assert (actual.diff().abs().sum(axis=1).fillna(0.0) <= 0.2 + 1e-12).all()


def test_volatility_ceiling_never_levers_or_peeks():
    index = pd.bdate_range("2023-01-02", periods=150)
    rng = np.random.default_rng(1)
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + 0.02 * rng.standard_normal(len(index))),
                          "B": 100 * np.cumprod(1 + 0.02 * rng.standard_normal(len(index)))}, index=index)
    target = pd.DataFrame({"A": 0.5, "B": -0.5}, index=index)
    full = portfolio.apply_volatility_ceiling(target, close, target_annual_vol=0.10)
    prefix = portfolio.apply_volatility_ceiling(target.iloc[:100], close.iloc[:100], target_annual_vol=0.10)
    np.testing.assert_allclose(full.iloc[:100], prefix, atol=1e-12)
    assert (full.abs().sum(axis=1) <= target.abs().sum(axis=1) + 1e-12).all()


def test_inverse_volatility_construction_is_capped_and_neutral():
    index = pd.bdate_range("2023-01-02", periods=100)
    rng = np.random.default_rng(8)
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + 0.01 * rng.standard_normal(len(index))),
                          "B": 100 * np.cumprod(1 + 0.03 * rng.standard_normal(len(index))),
                          "C": 100 * np.cumprod(1 + 0.02 * rng.standard_normal(len(index))),
                          "D": 100 * np.cumprod(1 + 0.02 * rng.standard_normal(len(index)))}, index=index)
    weights = portfolio.construct_long_short_weights(_scores(100), close, method="inverse_volatility", max_name_weight=0.20)
    assert (weights.abs().max(axis=1) <= 0.20 + 1e-12).all()
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12)


def test_hierarchical_risk_parity_construction_preserves_caps_and_neutrality():
    index = pd.bdate_range("2023-01-02", periods=130)
    rng = np.random.default_rng(18)
    close = pd.DataFrame({
        symbol: 100 * np.cumprod(1 + scale * rng.standard_normal(len(index)))
        for symbol, scale in zip("ABCDEF", [0.01, 0.012, 0.02, 0.025, 0.018, 0.03])
    }, index=index)
    score = pd.DataFrame(np.tile([-3, -2, -1, 1, 2, 3], (len(index), 1)), index=index, columns=close.columns)
    weights = portfolio.construct_long_short_weights(
        score, close, method="hierarchical_risk_parity", max_name_weight=0.20
    )
    assert (weights.abs().max(axis=1) <= 0.20 + 1e-12).all()
    assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-12)
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-12).all()
