"""Bounded Hugging Face text features with source-time provenance.

This is deliberately a feature extractor, not a trade recommender.  Every input
must carry the time it became available; the caller decides the later portfolio
join and can retain the document ID and model identifier in its trial metadata.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TextDocument:
    symbol: str
    available_at: pd.Timestamp
    source: str
    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.symbol or not self.source or not self.document_id or not self.text.strip():
            raise ValueError("symbol, source, document_id, and text are required")
        if pd.isna(pd.Timestamp(self.available_at)):
            raise ValueError("available_at is required for point-in-time text features")


@dataclass(frozen=True)
class FinBertFeature:
    symbol: str
    available_at: pd.Timestamp
    source: str
    document_id: str
    model: str
    sentiment: float
    positive_probability: float
    negative_probability: float
    neutral_probability: float


def _classification_scores(result: Any) -> dict[str, float]:
    """Normalize Hugging Face classification payloads into FinBERT probabilities."""
    rows = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Hugging Face classification response must be a list")
    scores: dict[str, float] = {}
    for row in rows:
        label = row.get("label") if isinstance(row, dict) else getattr(row, "label", None)
        score = row.get("score") if isinstance(row, dict) else getattr(row, "score", None)
        if label is not None and score is not None:
            scores[str(label).lower()] = float(score)
    missing = {"positive", "negative", "neutral"}.difference(scores)
    if missing:
        raise ValueError(f"FinBERT response missing labels: {sorted(missing)}")
    return scores


class FinBertExtractor:
    """Use FinBERT through Hugging Face Inference Providers, only when invoked."""

    def __init__(self, token: str | None = None, client: Any | None = None, model: str = "ProsusAI/finbert"):
        self.model = model
        if client is not None:
            self.client = client
            return
        token = (token or os.environ.get("HF_TOKEN", "")).strip()
        if not token:
            raise RuntimeError("Hugging Face token not found. Set $HF_TOKEN in alphaforge/.env.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as error:  # pragma: no cover - dependency installation path
            raise ImportError("Install Hugging Face support with `pip install alphaforge[llm]`.") from error
        self.client = InferenceClient(provider="hf-inference", api_key=token)

    def extract(self, document: TextDocument) -> FinBertFeature:
        result = self.client.text_classification(document.text, model=self.model)
        scores = _classification_scores(result)
        sentiment = scores["positive"] - scores["negative"]
        return FinBertFeature(
            symbol=document.symbol,
            available_at=pd.Timestamp(document.available_at),
            source=document.source,
            document_id=document.document_id,
            model=self.model,
            sentiment=float(sentiment),
            positive_probability=scores["positive"],
            negative_probability=scores["negative"],
            neutral_probability=scores["neutral"],
        )
