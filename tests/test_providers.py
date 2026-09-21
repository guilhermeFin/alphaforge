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
    assert P.get_fundamental_provider("sec_edgar").is_point_in_time is True
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


def test_sec_company_facts_are_available_only_from_filing_date():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2023-03-31", "filed": "2023-05-02", "form": "10-Q", "val": 100.0},
                            {"end": "2023-03-31", "filed": "2023-08-01", "form": "10-Q", "val": 101.0},
                            {"end": "2023-06-30", "filed": "2023-08-01", "form": "8-K", "val": 999.0},
                        ]
                    }
                },
                "Assets": {"units": {"USD": [
                    {"end": "2023-03-31", "filed": "2023-05-02", "form": "10-Q", "val": 500.0}
                ]}},
            }
        }
    }
    obs = P.sec_company_facts_to_obs(payload, "abc")
    assert set(obs["metric"]) == {"revenue", "total_assets"}
    revenue = obs[obs["metric"] == "revenue"]
    assert list(revenue["available_date"]) == [pd.Timestamp("2023-05-02"), pd.Timestamp("2023-08-01")]

    index = pd.bdate_range("2023-04-28", "2023-05-05")
    fund = build_fundamentals(obs, index, ["ABC"])
    assert fund["revenue"].loc["2023-05-01", "ABC"] != fund["revenue"].loc["2023-05-01", "ABC"]
    assert fund["revenue"].loc["2023-05-02", "ABC"] == 100.0


def test_sec_provider_uses_declared_agent_and_mocked_endpoints():
    calls = []
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2023-03-31", "filed": "2023-05-02", "form": "10-Q", "val": 500.0}
    ]}}}}}

    def fetch(url, headers):
        calls.append((url, headers))
        if "company_tickers" in url:
            return {"0": {"ticker": "ABC", "cik_str": 1234}}
        return facts

    obs = P.SecEdgarProvider(
        user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch, request_interval=0
    ).fundamentals(["ABC"])
    assert list(obs["metric"]) == ["total_assets"]
    assert len(calls) == 2
    assert calls[0][1]["User-Agent"] == "AlphaForge test@alphaforge.local"


def test_sec_provider_keeps_xom_predecessor_filing_history():
    def facts(value, filed):
        return {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
            {"end": "2026-03-31", "filed": filed, "form": "10-Q", "val": value}
        ]}}}}}

    calls = []

    def fetch(url, _headers):
        calls.append(url)
        if "company_tickers" in url:
            return {"0": {"ticker": "XOM", "cik_str": 2115436}}
        if "CIK0000034088" in url:
            return facts(100.0, "2026-05-04")
        return facts(110.0, "2026-08-03")

    obs = P.SecEdgarProvider(
        user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch, request_interval=0
    ).fundamentals(["XOM"])
    assert set(obs["symbol"]) == {"XOM"}
    assert list(obs["value"]) == [100.0, 110.0]
    assert len(calls) == 3
