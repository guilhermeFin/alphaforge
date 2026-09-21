"""Deterministic earnings-exhibit discovery and bounded, sentiment-blind selection."""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

EXTRACTION_VERSION = "earnings-v1"
MAX_EXHIBITS = 3
MAX_PARAGRAPHS = 3
_FINANCIAL = re.compile(r"\b(revenue|sales|earnings|income|margin|cash flow|profit|outlook|guidance)\b", re.I)
_OUTLOOK = re.compile(r"\b(outlook|guidance|expects?|forecast|anticipates?)\b", re.I)
_BOILERPLATE = re.compile(
    r"forward[- ]looking statements|safe harbor|undue reliance|actual results may|"
    r"risks and uncertainties|shall not be deemed|incorporated by reference|"
    r"^about\b|^investor contacts?\b|^media contacts?\b", re.I,
)


@dataclass(frozen=True)
class EarningsPassage:
    section: str
    text: str


def safe_attachment_url(filing_url: str, href: str) -> str | None:
    """Only HTML documents in the exact SEC filing directory are eligible."""
    target = urlsplit(urljoin(filing_url, href))
    if target.scheme != "https" or target.netloc.lower() != "www.sec.gov":
        return None
    if target.path in {"/ixviewer/doc/action", "/ixviewer/doc/action/", "/ix"}:
        doc = parse_qs(target.query).get("doc", [""])[0]
        return safe_attachment_url(filing_url, doc) if doc.startswith("/Archives/") else None
    if target.query or target.fragment:
        return None
    path = posixpath.normpath(unquote(target.path))
    directory = posixpath.dirname(urlsplit(filing_url).path)
    if posixpath.dirname(path) != directory or not path.lower().endswith((".htm", ".html", ".xhtml")):
        return None
    if path == urlsplit(filing_url).path:
        return None
    return f"https://www.sec.gov{path}"


def release_links(raw_html: str, filing_url: str, *, earnings_item: bool) -> list[str]:
    """Rank release descriptions, not sentiment; never assume every EX-99 is earnings."""
    soup = BeautifulSoup(raw_html, "html.parser")
    candidates: list[tuple[int, int, str]] = []
    for order, anchor in enumerate(soup.find_all("a", href=True)):
        url = safe_attachment_url(filing_url, anchor["href"])
        if not url:
            continue
        row = anchor.find_parent("tr")
        context = " ".join((row or anchor).stripped_strings).lower()
        if any(word in context for word in ("presentation", "slides", "transcript")):
            continue
        explicit = bool(re.search(r"earnings\s+(release|announcement)|(?:financial|quarterly|annual)\s+results", context))
        press = "press release" in context or "news release" in context
        exhibit = bool(re.search(r"(?:ex(?:hibit)?[\s-]*)?99[.\s-][12]\b", context))
        if explicit or (earnings_item and (press or exhibit)):
            candidates.append((0 if explicit else 1 if press else 2, order, url))
    return list(dict.fromkeys(url for _, _, url in sorted(candidates)))[:MAX_EXHIBITS]


def earnings_passages(raw_html: str) -> tuple[tuple[EarningsPassage, ...], int]:
    """First two financial-result paragraphs plus first outlook paragraph, in source order.

    No model output enters selection. This is a bounded narrative sample, not a
    full-document summary; numerical tables and legal disclaimers are excluded.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for node in soup.select("script, style, head, [hidden], [aria-hidden='true'], ix\\:hidden"):
        node.decompose()
    for node in list(soup.find_all(style=True)):
        if node.attrs and re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", node.get("style", ""), re.I):
            node.decompose()
    full_length = len(" ".join(soup.stripped_strings))
    results, outlook, seen = [], [], set()
    for index, block in enumerate(soup.find_all(["p", "div", "li"])):
        if block.find(["p", "div", "li"]):
            continue
        text = " ".join(block.stripped_strings)
        text = " ".join(text.split())
        if text in seen or len(text) < 70 or len(text.split()) < 12:
            continue
        seen.add(text)
        if _BOILERPLATE.search(text) or not _FINANCIAL.search(text):
            continue
        if _OUTLOOK.search(text):
            if not outlook:
                outlook.append((index, EarningsPassage("Outlook", text)))
        elif len(results) < 2:
            results.append((index, EarningsPassage("Results", text)))
    # A forecast alone does not establish that a candidate is an earnings release.
    if not results:
        return (), full_length
    chosen = sorted(results + outlook, key=lambda item: item[0])
    return tuple(passage for _, passage in chosen[:MAX_PARAGRAPHS]), full_length
