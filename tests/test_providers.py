"""Vendor-agnostic fundamental provider layer — tested WITHOUT any network or key.

The mapping core (vendor_frame_to_obs) is pure, so we exercise the SimFin- and
Sharadar-shaped column layouts against tiny in-memory frames, prove the output
feeds build_fundamentals point-in-time-correctly, and prove the paid adapters
fail loudly (not silently) when their API key is absent.
"""
import numpy as np
import pandas as pd
import pytest

from research import providers as P
from research.fundamentals import build_fundamentals


def test_obs_schema_and_available_date_mapping():
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB"],
        "Report Date": ["2020-03-31", "2020-03-31"],
        "Publish Date": ["2020-05-05", "2020-05-10"],
        "Revenue": [100.0, 200.0],
        "Net Income": [10.0, np.nan],          # NaN must be dropped, not zero-filled
        "Irrelevant": [1, 2],                   # unmapped column ignored
    })
    obs = P.vendor_frame_to_obs(
        df, field_map=P.SIMFIN_INCOME_MAP, symbol_col="Ticker",
        period_end_col="Report Date", available_date_col="Publish Date")

    assert list(obs.columns) == P.OBS_COLUMNS
    # canonical metric names, not vendor names
    assert set(obs["metric"]) == {"revenue", "net_income"}
    # the NaN net_income for BBB was dropped -> only AAA has net_income
    ni = obs[obs["metric"] == "net_income"]
    assert list(ni["symbol"]) == ["AAA"] and ni["value"].iloc[0] == 10.0
    # available_date is the PUBLISH date, never the period end
    aaa_rev = obs[(obs.symbol == "AAA") & (obs.metric == "revenue")].iloc[0]
    assert aaa_rev["available_date"] == pd.Timestamp("2020-05-05")
    assert aaa_rev["period_end"] == pd.Timestamp("2020-03-31")


def test_sharadar_shape_uses_datekey_as_available():
    df = pd.DataFrame({
        "ticker": ["AAA"],
        "reportperiod": ["2021-06-30"],
        "datekey": ["2021-08-04"],     # the date the filing became public = PIT anchor
        "revenue": [500.0],
        "assets": [3000.0],
        "ncfo": [80.0],
    })
    obs = P.vendor_frame_to_obs(
        df, field_map=P.SHARADAR_SF1_MAP, symbol_col="ticker",
        period_end_col="reportperiod", available_date_col="datekey")
    assert set(obs["metric"]) == {"revenue", "total_assets", "op_cash_flow"}
    assert (obs["available_date"] == pd.Timestamp("2021-08-04")).all()


def test_obs_feeds_build_fundamentals_point_in_time():
    """The whole point: a value appears ONLY from its available_date forward."""
    idx = pd.bdate_range("2020-01-01", periods=120)
    avail = idx[60]
    df = pd.DataFrame({
        "ticker": ["AAA"], "reportperiod": ["2019-12-31"],
        "datekey": [avail.strftime("%Y-%m-%d")], "revenue": [42.0],
    })
    obs = P.vendor_frame_to_obs(
        df, field_map=P.SHARADAR_SF1_MAP, symbol_col="ticker",
        period_end_col="reportperiod", available_date_col="datekey")
    fund = build_fundamentals(obs, idx, ["AAA"])
    panel = fund["revenue"]
    assert np.isnan(panel.loc[idx[59], "AAA"])     # not known yet
    assert panel.loc[idx[60], "AAA"] == 42.0       # known from the filing date
    assert panel.loc[idx[90], "AAA"] == 42.0       # forward-filled, never backward


def test_empty_when_no_mapped_columns():
    df = pd.DataFrame({"Ticker": ["AAA"], "Report Date": ["2020-03-31"],
                       "Publish Date": ["2020-05-05"], "Nothing": [1.0]})
    obs = P.vendor_frame_to_obs(df, field_map=P.SIMFIN_INCOME_MAP, symbol_col="Ticker",
                                period_end_col="Report Date", available_date_col="Publish Date")
    assert obs.empty and list(obs.columns) == P.OBS_COLUMNS


def test_missing_required_column_raises():
    df = pd.DataFrame({"Ticker": ["AAA"], "Revenue": [1.0]})  # no date columns
    with pytest.raises(KeyError):
        P.vendor_frame_to_obs(df, field_map=P.SIMFIN_INCOME_MAP, symbol_col="Ticker",
                              period_end_col="Report Date", available_date_col="Publish Date")


def test_factory_and_pit_flags():
    assert isinstance(P.get_fundamental_provider("synthetic"), P.SyntheticFundamentalProvider)
    assert P.get_fundamental_provider("sharadar").is_point_in_time is True
    assert P.get_fundamental_provider("simfin").is_point_in_time is False   # restated, flagged
    with pytest.raises(ValueError):
        P.get_fundamental_provider("bloomberg")


def test_paid_adapters_fail_loudly_without_key(monkeypatch):
    monkeypatch.delenv("SIMFIN_API_KEY", raising=False)
    monkeypatch.delenv("NASDAQ_DATA_LINK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SIMFIN_API_KEY"):
        P.SimFinProvider().fundamentals(["AAPL"])
    with pytest.raises(RuntimeError, match="NASDAQ_DATA_LINK_API_KEY"):
        P.SharadarProvider().fundamentals(["AAPL"])


def test_synthetic_provider_emits_obs_schema():
    obs = P.get_fundamental_provider("synthetic", periods=400, seed=1).fundamentals(
        ["AAA", "BBB", "CCC"])
    assert list(obs.columns) == P.OBS_COLUMNS
    assert not obs.empty
    # available_date is strictly after period_end (the reporting lag) — PIT-honest
    assert (obs["available_date"] > obs["period_end"]).all()
