import pandas as pd
import pytest
from io import BytesIO
from urllib.error import HTTPError

from research.macro import FredAlfredProvider, _fetch_json, fred_vintages_to_observations, macro_asof_panel


def _payload():
    return {
        "observations": [
            {"date": "2024-01-01", "realtime_start": "2024-02-01", "realtime_end": "2024-03-14", "value": "3.0"},
            {"date": "2024-01-01", "realtime_start": "2024-03-15", "realtime_end": "9999-12-31", "value": "3.2"},
            {"date": "2024-02-01", "realtime_start": "2024-03-15", "realtime_end": "9999-12-31", "value": "3.4"},
            {"date": "2024-03-01", "realtime_start": "2024-04-15", "realtime_end": "9999-12-31", "value": "."},
        ]
    }


def test_alfred_vintage_is_not_visible_before_its_release_date():
    obs = fred_vintages_to_observations(_payload(), "CPI")
    index = pd.DatetimeIndex(["2024-01-31", "2024-02-02", "2024-03-14", "2024-03-15", "2024-03-18"])
    panel = macro_asof_panel(obs, index)
    assert pd.isna(panel.loc["2024-01-31", "CPI"])
    assert panel.loc["2024-02-02", "CPI"] == 3.0
    assert panel.loc["2024-03-14", "CPI"] == 3.0
    # New February observation wins on its release date; the January revision is
    # retained in the raw table but does not backdate into the February value.
    assert panel.loc["2024-03-15", "CPI"] == 3.4


def test_fred_provider_requests_all_vintages_and_needs_a_key(monkeypatch):
    captured = []

    def fetch(url):
        captured.append(url)
        return _payload()

    out = FredAlfredProvider(api_key="test-key", fetch_json=fetch).observations("CPIAUCSL")
    assert not out.empty
    assert "output_type=1" in captured[0]
    assert "realtime_start=1776-07-04" in captured[0]
    assert "limit=100000" in captured[0]

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        FredAlfredProvider().observations("CPIAUCSL")


def test_fred_provider_pages_long_vintage_histories():
    captured = []

    def fetch(url):
        captured.append(url)
        if "offset=0" in url:
            return {
                "count": 2,
                "observations": [{"date": "2024-01-01", "realtime_start": "2024-02-01", "value": "1.0"}],
            }
        return {
            "count": 2,
            "observations": [{"date": "2024-02-01", "realtime_start": "2024-03-01", "value": "2.0"}],
        }

    out = FredAlfredProvider(api_key="test-key", fetch_json=fetch).observations("TEST")
    assert len(captured) == 2
    assert "offset=0" in captured[0] and "offset=1" in captured[1]
    assert list(out["value"]) == [1.0, 2.0]


def test_fred_provider_retries_large_daily_series_in_realtime_chunks():
    captured = []

    def fetch(url):
        captured.append(url)
        if "realtime_start=1776-07-04" in url:
            raise HTTPError(url, 400, "Bad Request", None, BytesIO())
        return {
            "count": 1,
            "observations": [
                {"date": "2024-01-02", "realtime_start": "2024-01-03", "value": "4.0"}
            ],
        }

    out = FredAlfredProvider(api_key="test-key", fetch_json=fetch).observations(
        "DGS10", start="2024-01-01", end="2024-12-30"
    )
    assert len(captured) == 2
    assert "realtime_start=2024-01-01" in captured[1]
    assert list(out["value"]) == [4.0]


def test_fred_fetch_explains_an_unregistered_api_key(monkeypatch):
    def fail(*_args, **_kwargs):
        raise HTTPError(
            "https://api.stlouisfed.org/fred/series/observations",
            400,
            "Bad Request",
            None,
            BytesIO(b'{"error_message":"The value for variable api_key is not registered."}'),
        )

    monkeypatch.setattr("research.macro.urlopen", fail)
    with pytest.raises(RuntimeError, match="FRED rejected FRED_API_KEY"):
        _fetch_json("https://api.stlouisfed.org/fred/series/observations")
