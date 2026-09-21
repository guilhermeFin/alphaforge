"""Bounded Hugging Face text features with source-time provenance.

This is deliberately a feature extractor, not a trade recommender.  Every input
must carry the time it became available; the caller decides the later portfolio
join and can retain the document ID and model identifier in its trial metadata.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

FINBERT_MAX_CHARS = 1_200
FINBERT_FALLBACK_MAX_CHARS = 600
# FinBERT accepts at most 512 WordPiece tokens. Character count alone is not a
# sufficient guard: SEC inline-XBRL identifiers can split into many tokens.
FINBERT_MAX_ESTIMATED_WORDPIECES = 360


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
    analyzed_text: str = ""
    input_char_count: int | None = None


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


def _retryable_status(error: Exception) -> int | None:
    """Return a temporary gateway status from common Hugging Face HTTP errors."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status or getattr(error, "status_code", None) or getattr(error, "code", None)


def _wordpiece_cost(fragment: str) -> int:
    """Conservative local estimate for a BERT-style token budget.

    This deliberately needs no tokenizer download at runtime. Long identifiers
    and numeric/XBRL fragments are treated as several pieces, which keeps the
    provider request comfortably below FinBERT's 512-token maximum.
    """
    if not re.search(r"[A-Za-z0-9]", fragment):
        return 1
    pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", fragment)
    return max(1, sum(max(1, (len(piece) + 5) // 6) for piece in pieces))


def _finbert_input(text: str, *, max_chars: int = FINBERT_MAX_CHARS) -> str:
    """Bound a normalized provider input by characters and estimated tokens.

    This character limit reduces typical excerpt sizes; it is not an exact token
    count. Preserve the result alongside the score for review.
    """
    normalized = " ".join(text.split())
    clipped = normalized[:max_chars]
    if len(normalized) > max_chars:
        clipped = clipped.rsplit(" ", 1)[0] or clipped

    selected: list[str] = []
    budget = 0
    for fragment in re.findall(r"\S+", clipped):
        cost = _wordpiece_cost(fragment)
        if selected and budget + cost > FINBERT_MAX_ESTIMATED_WORDPIECES:
            break
        selected.append(fragment)
        budget += cost
    return " ".join(selected) or clipped[:1]


class FinBertExtractor:
    """Use FinBERT through Hugging Face Inference Providers, only when invoked."""

    def __init__(
        self,
        token: str | None = None,
        client: Any | None = None,
        model: str = "ProsusAI/finbert",
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self._sleep = sleep
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
        analyzed_text = _finbert_input(document.text)
        candidates = [analyzed_text]
        fallback = _finbert_input(document.text, max_chars=FINBERT_FALLBACK_MAX_CHARS)
        if fallback != analyzed_text:
            candidates.append(fallback)
        last_error: Exception | None = None
        result = None
        for candidate in candidates:
            for attempt in range(3):
                try:
                    result = self.client.text_classification(candidate, model=self.model)
                    analyzed_text = candidate
                    break
                except Exception as error:
                    last_error = error
                    status = _retryable_status(error)
                    # Some SEC inline-XBRL tokens expand past the model limit even
                    # after character bounding. Try the safe shorter input once.
                    if status == 400:
                        break
                    if status not in {429, 500, 502, 503, 504}:
                        raise
                    if attempt == 2:
                        raise RuntimeError(
                            "Hugging Face inference is temporarily unavailable after 3 attempts. "
                            "Wait a minute and run the document check again."
                        ) from error
                    self._sleep(float(2**attempt))
            if result is not None:
                break
        if result is None:
            if last_error is None:
                raise RuntimeError("Hugging Face inference returned no classification result.")
            raise last_error
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
            analyzed_text=analyzed_text,
            input_char_count=len(document.text),
        )
