from dataclasses import dataclass

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api import service
from api.main import app


def _sec_obs():
    return pd.DataFrame([
        ("AAPL", "2024-03-31", "2024-05-02", "revenue", 100.0),
        ("AAPL", "2024-03-31", "2024-05-02", "total_assets", 500.0),
    ], columns=["symbol", "period_end", "available_date", "metric", "value"])


def _macro_obs():
    return pd.DataFrame([
        ("CPIAUCSL", "2024-03-01", "2024-04-10", "9999-12-31", 315.0),
        ("CPIAUCSL", "2024-04-01", "2024-05-10", "9999-12-31", 316.0),
    ], columns=["series_id", "observation_date", "available_date", "vintage_end", "value"])


class FakeSecProvider:
    def fundamentals(self, symbols):
        assert symbols == ["AAPL"]
        return _sec_obs()


class FakeMacroProvider:
    def observations(self, series_id, start, end):
        assert series_id == "CPIAUCSL"
        assert start == "2024-01-01" and end == "2024-06-01"
        return _macro_obs()


@dataclass
class FakeFeature:
    symbol: str = "AAPL"
    available_at: pd.Timestamp = pd.Timestamp("2024-05-01 16:30:00")
    source: str = "earnings release"
    document_id: str = "aapl-q1"
    model: str = "ProsusAI/finbert"
    sentiment: float = 0.5
    positive_probability: float = 0.6
    negative_probability: float = 0.1
    neutral_probability: float = 0.3


class FakeTextExtractor:
    def extract(self, document):
        assert document.document_id == "aapl-q1"
        return FakeFeature()


def _request(with_text=False):
    req = {
        "symbols": ["AAPL"],
        "macro_series": ["CPIAUCSL"],
        "macro_start": "2024-01-01",
        "as_of": "2024-06-01",
    }
    if with_text:
        req["text_document"] = {
            "symbol": "AAPL",
            "available_at": "2024-05-01T16:30:00",
            "source": "earnings release",
            "document_id": "aapl-q1",
            "text": "Revenue and margins improved.",
        }
    return req


def test_public_data_pilot_summarizes_point_in_time_sources_without_backtest():
    out = service.run_public_data_pilot(
        _request(with_text=True),
        sec_provider=FakeSecProvider(),
        macro_provider=FakeMacroProvider(),
        text_extractor=FakeTextExtractor(),
    )
    assert out["sec"][0]["observations"] == 2
    assert out["sec"][0]["metrics"] == 2
    assert out["macro"][0]["latest_value"] == 316.0
    assert out["macro"][0]["available_date"] == "2024-05-10T00:00:00"
    assert out["text_feature"]["sentiment"] == 0.5
    assert "not a trading result" in out["note"]


def test_public_data_pilot_rejects_unbounded_input_before_networking():
    with pytest.raises(service.WorkflowError, match="1 to 5 tickers"):
        service.run_public_data_pilot({**_request(), "symbols": ["A", "B", "C", "D", "E", "F"]})
    with pytest.raises(service.WorkflowError, match="macro start"):
        service.run_public_data_pilot({**_request(), "macro_start": "2024-07-01"})


def test_public_data_pilot_api_rejects_unknown_fields():
    response = TestClient(app).post("/public-data-pilot", json={**_request(), "surprise": True})
    assert response.status_code == 422


def test_public_data_pilot_page_renders_without_networking(monkeypatch):
    page = str((__import__("pathlib").Path(__file__).resolve().parent.parent / "app" / "pages" / "1_Public_Data_Pilot.py"))
    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    at = AppTest.from_file(page, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Public Data Pilot"
    assert any("Run public-data pilot" in button.label for button in at.button)
