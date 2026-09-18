import numpy as np
import pandas as pd
import pytest

from research.econometrics import fama_macbeth, forward_returns, ols, rolling_ols


def test_ols_recovers_known_linear_relationship():
    rng = np.random.default_rng(7)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.25, size=n)
    y = 1.5 + 2.0 * x1 - 0.75 * x2 + noise

    X = pd.DataFrame({"x1": x1, "x2": x2})
    res = ols(pd.Series(y), X)

    assert res.nobs == n
    assert res.params["const"] == pytest.approx(1.5, abs=0.04)
    assert res.params["x1"] == pytest.approx(2.0, abs=0.04)
    assert res.params["x2"] == pytest.approx(-0.75, abs=0.04)
    assert res.r_squared > 0.97
    assert (res.p_values < 0.01).all()


def test_newey_west_keeps_coefficients_and_changes_inference():
    rng = np.random.default_rng(11)
    n = 500
    x = rng.normal(size=n)
    innovations = rng.normal(size=n)
    error = np.zeros(n)
    for t in range(1, n):
        error[t] = 0.75 * error[t - 1] + innovations[t]
    y = 0.4 + 0.8 * x + error

    classic = ols(y, pd.DataFrame({"x": x}), covariance="classic")
    hac = ols(y, pd.DataFrame({"x": x}), covariance="newey_west", max_lags=5)

    pd.testing.assert_series_equal(classic.params, hac.params)
    assert hac.covariance_type == "newey_west(5)"
    assert not np.allclose(classic.std_errors.to_numpy(), hac.std_errors.to_numpy())


def test_ols_aligns_pandas_and_drops_missing_rows():
    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    y = pd.Series(np.arange(8, dtype=float), index=idx)
    x = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=idx)
    y.iloc[2] = np.nan
    x.iloc[5, 0] = np.nan

    res = ols(y, x)

    assert res.nobs == 6
    assert idx[2] not in res.residuals.index
    assert idx[5] not in res.residuals.index


def test_forward_returns_has_explicit_future_timing():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    one = forward_returns(close, horizon=1)
    two = forward_returns(close, horizon=2)

    assert one.iloc[0] == pytest.approx(0.10)
    assert two.iloc[0] == pytest.approx(0.21)
    assert np.isnan(one.iloc[-1])
    assert two.iloc[-2:].isna().all()


def test_rolling_ols_only_uses_trailing_window():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    x = pd.Series(np.arange(80, dtype=float), index=idx, name="x")
    y = 3.0 + 2.0 * x

    out = rolling_ols(y, x, window=20)

    assert len(out) == 61
    assert out.index[0] == idx[19]
    assert out.index[-1] == idx[-1]
    assert np.allclose(out["x"], 2.0)
    assert np.allclose(out["const"], 3.0)
    assert (out["nobs"] == 20.0).all()


def test_fama_macbeth_recovers_cross_sectional_premium():
    rng = np.random.default_rng(123)
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    assets = [f"A{i:02d}" for i in range(40)]

    size = pd.DataFrame(
        rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets
    )
    value = pd.DataFrame(
        rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets
    )
    eps = pd.DataFrame(
        rng.normal(scale=0.15, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    y = 0.01 + 0.03 * size - 0.02 * value + eps

    res = fama_macbeth(
        y,
        {"size": size, "value": value},
        covariance="newey_west",
        max_lags=3,
    )

    assert res.n_periods == len(dates)
    assert res.average_cross_section_n == pytest.approx(len(assets))
    assert res.params["const"] == pytest.approx(0.01, abs=0.01)
    assert res.params["size"] == pytest.approx(0.03, abs=0.01)
    assert res.params["value"] == pytest.approx(-0.02, abs=0.01)
    assert res.p_values["size"] < 0.01
    assert res.p_values["value"] < 0.01


def test_fama_macbeth_requires_enough_cross_section():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    assets = ["A", "B", "C"]
    y = pd.DataFrame(0.0, index=dates, columns=assets)
    x = pd.DataFrame(1.0, index=dates, columns=assets)

    with pytest.raises(ValueError, match="min_cross_section"):
        fama_macbeth(y, {"x": x}, min_cross_section=2)
