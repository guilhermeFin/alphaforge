"""LLM signal extraction — structured, validated, source-cited.

Module A of AlphaForge: turn unstructured text (transcripts, filings, news) into
*bounded, numeric, auditable* signals that drop straight into a backtest. Two hard
rules from the project guardrails:
  * Output is forced into a validated schema — bounded numeric fields + a reason.
    An out-of-range or malformed number is rejected, never silently trusted.
  * No network call happens unless the caller passes a client. This module is
    import-safe and test-safe offline; the engine never depends on an API.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, Field, ValidationError

PROMPT_TEMPLATE = (
    "You are a buy-side analyst. Read the text and return ONLY a JSON object with keys:\n"
    '  "guidance_change" (float in [-1, 1]; negative = lowered guidance),\n'
    '  "tone" (float in [-1, 1]; management tone),\n'
    '  "reason" (one sentence citing the specific language that drove your scores).\n'
    "Return nothing but the JSON.\n\nTEXT:\n{text}"
)


class TranscriptSignal(BaseModel):
    """A validated, bounded signal. Field constraints are the contract."""
    guidance_change: float = Field(ge=-1.0, le=1.0)
    tone: float = Field(ge=-1.0, le=1.0)
    reason: str = Field(min_length=1)


def validate_signal(payload: dict) -> TranscriptSignal:
    """Raise if the LLM returned anything outside the schema. Never present an
    unvalidated AI number as fact."""
    return TranscriptSignal(**payload)


def parse_signal(raw_text: str) -> TranscriptSignal:
    """Parse a model's raw text response into a validated signal."""
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return validate_signal(json.loads(raw_text[start:end + 1]))


def extract_signal(text: str, client=None, model: str = "claude-sonnet-4-6", max_tokens: int = 300) -> TranscriptSignal:
    """Extract a validated signal from ``text``.

    Offline by design: ``client`` must be an Anthropic client. We raise a clear
    error rather than reach for a global API key, so nothing in the research
    engine silently makes paid calls.
    """
    if client is None:
        raise RuntimeError(
            "extract_signal needs an Anthropic client: "
            "extract_signal(text, client=anthropic.Anthropic())"
        )
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text[:12000])}],
    )
    return parse_signal(msg.content[0].text)
