"""Deterministic, reviewable narrative-section selection for SEC filings.

The selector is intentionally independent of model output. It identifies the
MD&A section by form heading, rejects table-of-contents and inline-XBRL-heavy
candidates, and leaves the caller to use an explicit generic fallback when a
clean section cannot be established.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


MDA_EXTRACTION_VERSION = "mda-section-v1"
MDA_MAX_EXCERPT_CHARS = 2_400


@dataclass(frozen=True)
class MdaSection:
    """A selected MD&A span and the deterministic audit details behind it."""

    text: str
    start_char: int
    end_char: int
    label: str
    quality_score: float
    quality_note: str
    candidate_count: int


@dataclass(frozen=True)
class MdaSelection:
    """A successful MD&A selection or an explicit reason for its absence."""

    section: MdaSection | None
    fallback_reason: str | None = None


_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'"})
_START_PATTERNS = {
    "10-K": re.compile(
        r"\bitem\s*7\s*[.:-]?\s*management'?s\s+discussion\s+and\s+analysis\b",
        re.I,
    ),
    "10-Q": re.compile(
        r"\bitem\s*2\s*[.:-]?\s*management'?s\s+discussion\s+and\s+analysis\b",
        re.I,
    ),
}
_FALLBACK_START = re.compile(r"\bmanagement'?s\s+discussion\s+and\s+analysis\b", re.I)
_END_PATTERNS = {
    "10-K": re.compile(r"\bitem\s*(?:7a|8)\b(?:\s*[.:-]|\s+(?:quantitative|financial))", re.I),
    "10-Q": re.compile(r"\bitem\s*(?:3|4)\b(?:\s*[.:-]|\s+(?:quantitative|controls))", re.I),
}
_XBRL_MARKERS = re.compile(r"\b(?:us-gaap|dei|xbrli|iso4217|srt|linkbase)\s*:", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_NUMBER = re.compile(r"\b\d+(?:[,.]\d+)?\b")


def _normalized_for_match(text: str) -> str:
    return text.translate(_APOSTROPHES).replace("\xa0", " ")


def _quality(text: str) -> tuple[float, str | None, str]:
    """Score readable prose without using financial direction or sentiment."""
    stripped = text.strip()
    words = _WORD.findall(stripped)
    word_count = len(words)
    nonspace = re.sub(r"\s+", "", stripped)
    alpha_ratio = sum(char.isalpha() for char in nonspace) / max(1, len(nonspace))
    number_ratio = len(_NUMBER.findall(stripped)) / max(1, word_count)
    sentences = len(re.findall(r"[.!?](?:\s|$)", stripped))
    xbrl_markers = len(_XBRL_MARKERS.findall(stripped))

    if word_count < 90:
        return 0.0, "the candidate is too short to establish a narrative section", "too_short"
    if xbrl_markers >= 3 or alpha_ratio < 0.48:
        return 0.0, "the candidate is dominated by inline-XBRL or tabular text", "non_narrative"
    if number_ratio > 0.42 and sentences < 8:
        return 0.0, "the candidate is dominated by numeric table content", "table_heavy"

    # More readable prose and a longer coherent span rank above table-of-contents
    # fragments, while the cap prevents raw filing length from dominating.
    score = (
        55.0 * alpha_ratio
        + 20.0 * min(1.0, sentences / 20.0)
        + 20.0 * min(1.0, word_count / 1_000.0)
        - 10.0 * min(1.0, number_ratio)
        - min(5.0, float(xbrl_markers))
    )
    note = (
        f"Narrative-quality check passed: {word_count:,} words, {alpha_ratio:.0%} alphabetic characters, "
        f"{number_ratio:.0%} numeric tokens."
    )
    return score, None, note


def _section_end(text: str, start: int, pattern: re.Pattern[str]) -> int | None:
    for match in pattern.finditer(text, start + 1):
        return match.start()
    return None


def select_mda_section(
    visible_text: str,
    form: str,
    *,
    max_excerpt_chars: int = MDA_MAX_EXCERPT_CHARS,
) -> MdaSelection:
    """Select a readable MD&A excerpt from normalized SEC visible text.

    The returned text is bounded for later model-input handling, while
    ``start_char`` and ``end_char`` retain the full MD&A span inside the source.
    """
    form = str(form).upper().strip()
    if form not in _START_PATTERNS:
        return MdaSelection(None, "MD&A selection applies only to 10-Q and 10-K filings")
    if max_excerpt_chars < 200:
        raise ValueError("max_excerpt_chars must be at least 200")

    text = _normalized_for_match(visible_text)
    starts = [match.start() for match in _START_PATTERNS[form].finditer(text)]
    if not starts:
        # Some legacy filings omit the item number from the rendered heading.
        starts = [match.start() for match in _FALLBACK_START.finditer(text)]
    if not starts:
        return MdaSelection(None, "no MD&A heading was found in the filing text")

    candidates: list[MdaSection] = []
    reasons: list[str] = []
    item = "Item 7" if form == "10-K" else "Item 2"
    for start in starts:
        end = _section_end(text, start, _END_PATTERNS[form])
        if end is None:
            reasons.append("no later MD&A section boundary was found")
            continue
        section_text = visible_text[start:end].strip()
        score, rejected, note = _quality(section_text)
        if rejected:
            reasons.append(rejected)
            continue
        excerpt_end = min(start + max_excerpt_chars, end)
        excerpt = visible_text[start:excerpt_end].strip()
        if len(excerpt) < 200:
            reasons.append("the MD&A excerpt is too short after normalization")
            continue
        candidates.append(MdaSection(
            text=excerpt,
            start_char=start,
            end_char=end,
            label=f"MD&A ({item})",
            quality_score=score,
            quality_note=note,
            candidate_count=len(starts),
        ))

    if not candidates:
        reason = reasons[0] if reasons else "no readable MD&A candidate passed the quality check"
        return MdaSelection(None, f"MD&A could not be selected because {reason}")
    # A later valid heading wins a score tie, which favors the actual section
    # over an otherwise similar table-of-contents fragment.
    return MdaSelection(max(candidates, key=lambda item: (item.quality_score, item.start_char)))
