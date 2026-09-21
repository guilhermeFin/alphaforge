from dataclasses import dataclass

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api import service
from api.main import app
from research.local_history import LocalResearchHistory
from research.sec_documents import SecFilingDocument
from research.text_features import FinBertExtractor


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
    analyzed_text: str = ""
    input_char_count: int | None = None


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
            analyzed_text=document.text, input_char_count=len(document.text),
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
    review = out["features"][0]["review"]
    assert review["analyzed_text"] == "Revenue improved."
    assert review["model_input_shortened"] is False
    assert review["source_text_chars"] is None


def test_batch_api_returns_the_captured_model_input(monkeypatch):
    passage = "Item 2.02 Revenue improved. " * 80

    class LongDocumentProvider:
        def documents(self, *_args):
            return [SecFilingDocument(
                symbol="AAPL", available_at=pd.Timestamp("2024-05-02"), form="8-K",
                accession_number="0001", url="https://www.sec.gov/example", text=passage,
                source_text_chars=8000, excerpt_start_char=200,
            )], []

    class CapturingClient:
        def text_classification(self, text, model):
            self.text = text
            return [
                {"label": "positive", "score": 0.6}, {"label": "negative", "score": 0.1},
                {"label": "neutral", "score": 0.3},
            ]

    client = CapturingClient()
    monkeypatch.setattr(service, "SecDocumentProvider", LongDocumentProvider)
    monkeypatch.setattr(service, "FinBertExtractor", lambda: FinBertExtractor(client=client))
    response = TestClient(app).post("/real-document-batch", json={
        "symbols": ["AAPL"], "forms": ["8-K"], "as_of": "2024-06-01", "per_symbol": 1,
    })
    assert response.status_code == 200
    review = response.json()["features"][0]["review"]
    assert review["analyzed_text"] == client.text
    assert review["retrieved_excerpt"] == passage
    assert review["analyzed_char_count"] == len(client.text)
    assert review["input_char_count"] == len(passage)
    assert review["model_input_shortened"] is True
    assert review["source_text_chars"] == 8000
    assert review["excerpt_start_char"] == 200


def test_mda_batch_audit_keeps_section_selection_metadata():
    class MdaDocumentProvider:
        def documents(self, *_args):
            return [SecFilingDocument(
                symbol="AAPL", available_at=pd.Timestamp("2024-05-02"), form="10-K",
                accession_number="0001", url="https://www.sec.gov/example", text="Management discusses liquidity and operating results.",
                source_text_chars=8_000, excerpt_start_char=1_000, section_end_char=5_000,
                content_kind="mda_excerpt", selection_method="mda-section-v1", selection_label="MD&A (Item 7)",
                selection_quality_score=82.5, selection_quality_note="Narrative-quality check passed.",
            )], []

    out = service.run_real_document_batch(
        {"symbols": ["AAPL"], "forms": ["10-K"], "as_of": "2024-06-01", "per_symbol": 1},
        document_provider=MdaDocumentProvider(), text_extractor=FakeBatchTextExtractor(),
    )

    feature = out["features"][0]
    assert feature["content_kind"] == "mda_excerpt"
    assert feature["selection_label"] == "MD&A (Item 7)"
    assert feature["extraction_version"] == "mda-section-v1"
    assert feature["review"]["selection"]["method"] == "mda-section-v1"
    assert feature["review"]["section_end_char"] == 5_000
    assert feature["review"]["passages"][0]["section"] == "MD&A (Item 7)"
    assert out["source_validation"]["mda_excerpts"] == 1
    assert out["source_validation"]["filing_excerpts"] == 0


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
    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    at = _page_app("1_Public_Data_Pilot.py")
    assert not at.exception
    assert at.title[0].value == "Data quality"
    assert any("Check data coverage" in button.label for button in at.button)


def test_real_document_batch_page_renders_without_networking(monkeypatch):
    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    at = _page_app("2_Real_Document_Batch.py")
    assert not at.exception
    assert at.title[0].value == "Filing research"
    assert any("Analyze filings" in button.label for button in at.button)


def test_research_history_page_renders_without_networking(monkeypatch):
    monkeypatch.setenv("ALPHAFORGE_API", "http://127.0.0.1:1")
    at = _page_app("3_Research_History.py")
    assert not at.exception
    assert at.title[0].value == "Research history"


def test_research_history_handles_an_insufficient_event_study(monkeypatch, tmp_path):
    path = tmp_path / "history.db"
    LocalResearchHistory(path).record("event_study", "two filings", {
        "status": "insufficient_data", "n_events": 2, "n_oos": None,
        "mse_improvement": None, "oos_pearson": None, "evidence_established": False,
    })
    monkeypatch.setenv("ALPHAFORGE_HISTORY_PATH", str(path))
    at = _page_app("3_Research_History.py")
    assert not at.exception
    assert any("not available" in str(caption.value) for caption in at.caption)


def _batch_review_result():
    result = service.run_real_document_batch(
        {"symbols": ["AAPL"], "forms": ["8-K"], "as_of": "2024-06-01", "per_symbol": 1},
        document_provider=FakeDocumentProvider(), text_extractor=FakeBatchTextExtractor(),
    )
    first = result["features"][0]
    first["review"].update({"source_text_chars": 200, "excerpt_start_char": 40})
    second = {**first, "symbol": "MSFT", "document_id": "0002", "review": {
        "analyzed_text": "Demand weakened.", "retrieved_excerpt": "Demand weakened. More text.",
        "analyzed_char_count": len("Demand weakened."), "input_char_count": len("Demand weakened. More text."),
        "model_input_shortened": True, "source_text_chars": 300, "excerpt_start_char": 0,
    }}
    result["features"].append(second)
    return result


def _page_app(filename, result=None):
    from pathlib import Path

    page = str(Path(__file__).resolve().parent.parent / "app" / "Home.py")
    at = AppTest.from_file(page, default_timeout=30)
    at.session_state["api_mode"] = False
    if result is not None:
        at.session_state["real_document_batch"] = result
    at.run()
    return at.switch_page(f"pages/{filename}").run()


def _review_app(result):
    return _page_app("2_Real_Document_Batch.py", result)


def test_review_selection_shows_exact_input_without_new_model_calls(monkeypatch):
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("Review must not request another classification")

    monkeypatch.setattr("httpx.post", unexpected_call)
    at = _review_app(_batch_review_result())
    assert not at.exception
    assert at.code[0].value == "Revenue improved."
    assert any("No further shortening" in caption.value for caption in at.caption)
    at.selectbox[0].select(1).run()
    assert not at.exception
    assert at.code[0].value == "Demand weakened."
    assert at.code[1].value == "Demand weakened. More text."
    assert any("Shortened for FinBERT" in caption.value for caption in at.caption)
    at.selectbox[0].select(0).run()
    assert at.code[0].value == "Revenue improved."


def test_legacy_result_does_not_invent_an_excerpt():
    result = _batch_review_result()
    result["features"][0].pop("review")
    at = _review_app(result)
    assert not at.exception
    assert not at.code
    assert any("Exact excerpt unavailable" in notice.value for notice in at.info)


def test_failed_new_batch_clears_the_old_review(monkeypatch):
    import httpx

    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: httpx.Response(
        500, json={"detail": "SEC temporarily unavailable"},
    ))
    at = _review_app(_batch_review_result())
    at.button[0].click().run()
    assert not at.exception
    assert not at.code
    assert "SEC temporarily unavailable" in at.error[0].value


def test_multi_passage_review_and_navigation_preserve_results(monkeypatch):
    result = _batch_review_result()
    first = result["features"][0]
    first["content_kind"] = "earnings_release"
    first["passage_count"] = 2
    first["review"]["passages"] = [
        {"section": "Results", "analyzed_text": "Revenue improved.", "retrieved_text": "Revenue improved.", "sentiment": 0.5},
        {"section": "Outlook", "analyzed_text": "Demand weakened.", "retrieved_text": "Demand weakened.", "sentiment": -0.5},
    ]
    monkeypatch.setattr("httpx.post", lambda *_a, **_k: pytest.fail("Review must not make a provider request"))
    at = _review_app(result)
    at.radio[0].set_value(1).run()
    assert not at.exception
    assert at.code[0].value == "Demand weakened."
    at.switch_page("pages/1_Public_Data_Pilot.py").run()
    assert not at.exception
    at.switch_page("pages/2_Real_Document_Batch.py").run()
    assert not at.exception
    assert len(at.session_state["real_document_batch"]["features"]) == 2


def test_mda_review_renders_quality_audit_without_another_model_call(monkeypatch):
    result = _batch_review_result()
    record = result["features"][0]
    record.update({
        "form": "10-K",
        "content_kind": "mda_excerpt",
        "selection_method": "mda-section-v1",
        "selection_label": "MD&A (Item 7)",
        "selection_quality_score": 82.5,
        "selection_quality_note": "Narrative-quality check passed.",
        "selection_note": "MD&A (Item 7) selected from the primary filing using deterministic heading and narrative-quality checks.",
    })
    record["review"].update({
        "section_end_char": 180,
        "selection": {
            "method": "mda-section-v1",
            "label": "MD&A (Item 7)",
            "quality_score": 82.5,
            "quality_note": "Narrative-quality check passed.",
        },
    })
    monkeypatch.setattr("httpx.post", lambda *_a, **_k: pytest.fail("Review must not make a provider request"))

    at = _review_app(result)

    assert not at.exception
    assert "MD&A (Item 7) selected" in at.success[0].value
    assert any("Narrative-quality check passed" in caption.value for caption in at.caption)
    assert any("MD&A (Item 7) spans characters" in caption.value for caption in at.caption)


def test_coverage_result_and_failure_states(monkeypatch):
    import httpx

    result = service.run_public_data_pilot(_request(), sec_provider=FakeSecProvider(), macro_provider=FakeMacroProvider())
    monkeypatch.setattr("httpx.post", lambda *_a, **_k: httpx.Response(200, json=result))
    at = _page_app("1_Public_Data_Pilot.py")
    at.button[0].click().run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Companies checked", "SEC observations", "Economic vintages"]
    assert len(at.dataframe) == 2
    monkeypatch.setattr("httpx.post", lambda *_a, **_k: httpx.Response(502, text="Bad gateway"))
    at.button[0].click().run()
    assert not at.exception
    assert not at.dataframe
    assert "502" in at.error[0].value
