import pandas as pd
import pytest

from research.macro import FredAlfredProvider, fred_vintages_to_observations, macro_asof_panel


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
    assert "output_type=2" in captured[0]
    assert "realtime_start=1776-07-04" in captured[0]

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        FredAlfredProvider().observations("CPIAUCSL")
