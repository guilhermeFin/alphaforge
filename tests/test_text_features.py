import pandas as pd
import pytest

from research.text_features import FinBertExtractor, TextDocument


class FakeFinBertClient:
    def text_classification(self, text, model):
        assert text == "Demand strengthened and margins improved."
        assert model == "ProsusAI/finbert"
        return [
            {"label": "positive", "score": 0.70},
            {"label": "negative", "score": 0.10},
            {"label": "neutral", "score": 0.20},
        ]


def test_finbert_feature_keeps_source_and_availability_metadata():
    doc = TextDocument(
        symbol="ABC",
        available_at=pd.Timestamp("2024-05-01 16:30:00"),
        source="earnings_call",
        document_id="abc-q1-2024",
        text="Demand strengthened and margins improved.",
    )
    feature = FinBertExtractor(client=FakeFinBertClient()).extract(doc)
    assert feature.sentiment == pytest.approx(0.60)
    assert feature.available_at == doc.available_at
    assert feature.document_id == "abc-q1-2024"
    assert feature.positive_probability + feature.negative_probability + feature.neutral_probability == pytest.approx(1.0)


def test_finbert_requires_a_token_when_no_client_is_supplied(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        FinBertExtractor()


def test_text_document_rejects_missing_point_in_time_metadata():
    with pytest.raises(ValueError, match="required"):
        TextDocument("ABC", pd.Timestamp("2024-01-01"), "", "doc", "text")
