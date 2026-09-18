from dataclasses import dataclass

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api import service
from api.main import app
from research.sec_documents import SecFilingDocument


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


class FakeDocumentProvider:
    def documents(self, symbols, forms, filed_before, per_symbol):
        assert symbols == ["AAPL"] and forms == {"8-K"} and per_symbol == 1
        assert filed_before == pd.Timestamp("2024-06-01")
        return [SecFilingDocument(
            symbol="AAPL", available_at=pd.Timestamp("2024-05-02"), form="8-K",
            accession_number="0001", url="https://www.sec.gov/example", text="Revenue improved.",
        )], []


class FakeBatchTextExtractor:
    def extract(self, document):
        return FakeFeature(
            symbol=document.symbol, available_at=document.available_at, source=document.source,
            document_id=document.document_id,
        )


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
    assert "AAPL: SEC coverage begins" in out["warnings"][0]
    assert out["text_feature"]["sentiment"] == 0.5
    assert "not a trading result" in out["note"]


def test_real_document_batch_preserves_sec_filing_provenance():
    out = service.run_real_document_batch(
        {"symbols": ["AAPL"], "forms": ["8-K"], "as_of": "2024-06-01", "per_symbol": 1},
        document_provider=FakeDocumentProvider(), text_extractor=FakeBatchTextExtractor(),
    )
    assert out["features"][0]["available_at"] == "2024-05-02T00:00:00"
    assert out["features"][0]["form"] == "8-K"
    assert out["features"][0]["source_url"] == "https://www.sec.gov/example"


def test_public_data_pilot_rejects_unbounded_input_before_networking():
    with pytest.raises(service.WorkflowError, match="1 to 5 tickers"):
        service.run_public_data_pilot({**_request(), "symbols": ["A", "B", "C", "D", "E", "F"]})
    with pytest.raises(service.WorkflowError, match="macro start"):
        service.run_public_data_pilot({**_request(), "macro_start": "2024-07-01"})


def test_public_data_pilot_rejects_empty_macro_history():
    class EmptyMacroProvider:
        def observations(self, *_args, **_kwargs):
            return pd.DataFrame(columns=["series_id", "observation_date", "available_date", "vintage_end", "value"])

    with pytest.raises(service.WorkflowError, match="no point-in-time observations"):
        service.run_public_data_pilot(
            _request(), sec_provider=FakeSecProvider(), macro_provider=EmptyMacroProvider()
        )


def test_public_data_pilot_rejects_a_document_after_its_as_of_date():
    class NeverCalled:
        def fundamentals(self, *_args, **_kwargs):
            raise AssertionError("future document must be rejected before source calls")

    req = _request(with_text=True)
    req["text_document"]["available_at"] = "2024-06-02T16:30:00"
    with pytest.raises(service.WorkflowError, match="on or before"):
        service.run_public_data_pilot(req, sec_provider=NeverCalled())


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


def test_real_document_batch_page_renders_without_networking(monkeypatch):
    page = str((__import__("pathlib").Path(__file__).resolve().parent.parent / "app" / "pages" / "2_Real_Document_Batch.py"))
    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    at = AppTest.from_file(page, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Real Document Batch"
    assert any("Classify real filing batch" in button.label for button in at.button)
