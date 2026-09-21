import pandas as pd
import pytest

from research.text_features import FINBERT_FALLBACK_MAX_CHARS, FINBERT_MAX_CHARS, FINBERT_MAX_ESTIMATED_WORDPIECES, FinBertExtractor, TextDocument, _wordpiece_cost


class FakeFinBertClient:
    def text_classification(self, text, model):
        assert text == "Demand strengthened and margins improved."
        assert model == "ProsusAI/finbert"
        return [
            {"label": "positive", "score": 0.70},
            {"label": "negative", "score": 0.10},
            {"label": "neutral", "score": 0.20},
        ]


class FakeGatewayError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class RetryableFinBertClient(FakeFinBertClient):
    def __init__(self):
        self.calls = 0

    def text_classification(self, text, model):
        self.calls += 1
        if self.calls < 3:
            raise FakeGatewayError(502)
        return super().text_classification(text, model)


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
    assert feature.analyzed_text == doc.text
    assert feature.input_char_count == len(doc.text)
    assert feature.positive_probability + feature.negative_probability + feature.neutral_probability == pytest.approx(1.0)


def test_finbert_retries_a_temporary_gateway_failure():
    doc = TextDocument("ABC", pd.Timestamp("2024-05-01 16:30:00"), "manual note", "abc-001", "Demand strengthened and margins improved.")
    client = RetryableFinBertClient()
    feature = FinBertExtractor(client=client, sleep=lambda _seconds: None).extract(doc)
    assert client.calls == 3
    assert feature.sentiment == pytest.approx(0.60)


def test_finbert_bounds_long_text_before_the_model_call():
    class CapturingClient(FakeFinBertClient):
        def text_classification(self, text, model):
            self.text = text
            return [
                {"label": "positive", "score": 0.70},
                {"label": "negative", "score": 0.10},
                {"label": "neutral", "score": 0.20},
            ]

    client = CapturingClient()
    document = TextDocument("ABC", pd.Timestamp("2024-05-01"), "SEC 8-K", "doc", "word " * 1_000)
    feature = FinBertExtractor(client=client).extract(document)
    assert len(client.text) <= FINBERT_MAX_CHARS
    assert client.text.endswith("word")
    assert feature.analyzed_text == client.text
    assert feature.input_char_count == len(document.text)
    assert len(feature.analyzed_text) < feature.input_char_count


def test_finbert_bounds_xbrl_like_text_by_estimated_tokens_before_the_model_call():
    class CapturingClient:
        def text_classification(self, text, model):
            self.text = text
            return [
                {"label": "positive", "score": 0.70},
                {"label": "negative", "score": 0.10},
                {"label": "neutral", "score": 0.20},
            ]

    client = CapturingClient()
    xbrl = "us-gaap:AccruedLiabilitiesCurrent 0001045810 iso4217:USD " * 100
    feature = FinBertExtractor(client=client).extract(TextDocument("ABC", pd.Timestamp("2024-05-01"), "SEC 10-Q", "doc", xbrl))
    assert sum(_wordpiece_cost(piece) for piece in client.text.split()) <= FINBERT_MAX_ESTIMATED_WORDPIECES
    assert len(feature.analyzed_text) < FINBERT_MAX_CHARS


def test_finbert_retries_token_limit_bad_request_with_safe_shorter_input():
    class TokenLimitClient:
        def __init__(self):
            self.calls = []

        def text_classification(self, text, model):
            self.calls.append(text)
            if len(text) > FINBERT_FALLBACK_MAX_CHARS:
                raise FakeGatewayError(400)
            return [
                {"label": "positive", "score": 0.70},
                {"label": "negative", "score": 0.10},
                {"label": "neutral", "score": 0.20},
            ]

    client = TokenLimitClient()
    xbrl = "us-gaap:AccruedLiabilitiesCurrent 0001045810 iso4217:USD " * 100
    feature = FinBertExtractor(client=client).extract(TextDocument("ABC", pd.Timestamp("2024-05-01"), "SEC 10-Q", "doc", xbrl))
    assert len(client.calls) == 2
    assert len(client.calls[1]) <= FINBERT_FALLBACK_MAX_CHARS
    assert feature.analyzed_text == client.calls[1]


def test_finbert_requires_a_token_when_no_client_is_supplied(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        FinBertExtractor()


def test_text_document_rejects_missing_point_in_time_metadata():
    with pytest.raises(ValueError, match="required"):
        TextDocument("ABC", pd.Timestamp("2024-01-01"), "", "doc", "text")
