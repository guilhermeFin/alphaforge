"""Quantamental pipeline — turn timestamped text into a point-in-time signal panel.

This is where Module A (LLM signals) meets the moat (the backtester). The single
most important property is POINT-IN-TIME CORRECTNESS: a document dated D can only
influence positions held from D onward. ``build_signal_panel`` enforces this by
placing each signal at its document date and forward-filling (never backward), and
the backtester then applies its own execution lag on top. tests/test_quantamental.py
proves there is no leak.

The pipeline is extractor-agnostic: pass any ``extract_fn(text) -> dict`` with keys
{guidance_change, tone, reason}. Two are provided:
  * keyword_extractor    — deterministic, offline (tests & demos, $0).
  * make_llm_extractor   — wraps the real Claude extractor, with on-disk caching so
    you never pay twice for the same document.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable

import numpy as np
import pandas as pd

from .signals_llm import validate_signal, extract_signal

# --- vocabulary for the deterministic extractor and the synthetic text generator ---
POS_WORDS = ["raised", "record", "accelerating", "strong", "beat", "expansion",
             "robust", "upgraded", "outperform", "momentum", "confident"]
NEG_WORDS = ["lowered", "withdrawing", "declining", "weak", "miss", "contraction",
             "headwinds", "downgraded", "restructuring", "cautious", "softened"]

ExtractFn = Callable[[str], dict]


# ----------------------------- extractors -----------------------------
def keyword_extractor(text: str) -> dict:
    """Deterministic, offline signal: net positive-vs-negative term balance in [-1, 1]."""
    t = text.lower()
    p = sum(t.count(w) for w in POS_WORDS)
    n = sum(t.count(w) for w in NEG_WORDS)
    score = (p - n) / (p + n) if (p + n) else 0.0
    return {"guidance_change": float(score), "tone": float(score),
            "reason": f"{p} positive / {n} negative terms"}


def make_llm_extractor(client, model: str = "claude-sonnet-4-6", cache_path: str | None = None) -> ExtractFn:
    """Real LLM extractor with a JSON cache keyed by text hash (never pay twice)."""
    cache: dict[str, dict] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    def _extract(text: str) -> dict:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]
        sig = extract_signal(text, client=client, model=model).model_dump()
        cache[key] = sig
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        return sig

    return _extract


# ----------------------------- synthetic text -----------------------------
def sentiment_to_text(sentiment: float, rng: np.random.Generator) -> str:
    """Generate an earnings-call snippet whose word balance reflects ``sentiment``."""
    k = int(round(abs(sentiment) * 3))  # 0..3 emphasis words
    fillers = ["Management commented on the quarter.", "Revenue was discussed.",
               "The outlook was reviewed.", "Operations were covered."]
    if sentiment > 0.1:
        words = list(rng.choice(POS_WORDS, size=max(1, k), replace=True))
        body = f"Results were {words[0]}; the team highlighted " + ", ".join(words) + " trends."
    elif sentiment < -0.1:
        words = list(rng.choice(NEG_WORDS, size=max(1, k), replace=True))
        body = f"Results were {words[0]}; the team flagged " + ", ".join(words) + " conditions."
    else:
        body = "Results were broadly in line; the outlook was reaffirmed with no major changes."
    return rng.choice(fillers) + " " + body


def make_transcripts(events: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add a ``text`` column to an events frame (symbol, date, latent_sentiment)."""
    rng = np.random.default_rng(seed)
    docs = events.copy()
    docs["text"] = [sentiment_to_text(s, rng) for s in docs["latent_sentiment"].to_numpy()]
    return docs


# ----------------------------- the panel builder -----------------------------
def build_signal_panel(
    docs: pd.DataFrame,
    index: pd.DatetimeIndex,
    symbols: list[str],
    extract_fn: ExtractFn,
    field: str = "tone",
    horizon: int = 63,
) -> pd.DataFrame:
    """Build a point-in-time signal panel (dates x symbols) from timestamped docs.

    Each document's extracted, *validated* signal value is placed at the first
    trading date >= the document date, then forward-filled for up to ``horizon``
    trading days (the signal decays to 0 if no fresh document arrives). Nothing is
    ever filled backward, so a signal cannot influence an earlier date.
    """
    panel = pd.DataFrame(np.nan, index=index, columns=symbols, dtype=float)
    for doc in docs.itertuples(index=False):
        if doc.symbol not in panel.columns:
            continue
        pos = index.searchsorted(pd.Timestamp(doc.date))
        if pos >= len(index):
            continue  # document dated after the sample — cannot be used
        sig = validate_signal(extract_fn(doc.text))  # enforce schema/bounds, always
        panel.iat[pos, panel.columns.get_loc(doc.symbol)] = getattr(sig, field)

    panel = panel.ffill(limit=horizon)   # persist forward, decay after `horizon`
    return panel.fillna(0.0)
