"""Small, date-provenanced SEC filing excerpts for batch text research."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
from bs4 import BeautifulSoup

from .http import decode_json_bytes, decode_response_bytes
from .providers import SEC_TICKERS_URL, sec_ticker_map
from .text_features import TextDocument
from .earnings import EXTRACTION_VERSION, EarningsPassage, earnings_passages, release_links
from .filing_sections import MDA_EXTRACTION_VERSION, select_mda_section

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SUPPORTED_FORMS = {"8-K", "10-Q", "10-K"}
SEC_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
SEC_MAX_EXPANDED_TEXT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class SecFilingDocument:
    symbol: str
    available_at: pd.Timestamp
    form: str
    accession_number: str
    url: str
    text: str
    source_text_chars: int | None = None
    excerpt_start_char: int | None = None
    filing_url: str | None = None
    content_kind: str = "filing_excerpt"
    selection_note: str = "Primary filing excerpt; no earnings attachment selected."
    selection_method: str = "generic-filing-excerpt-v1"
    selection_label: str = "Filing excerpt"
    selection_quality_score: float | None = None
    selection_quality_note: str | None = None
    section_end_char: int | None = None
    passages: tuple[EarningsPassage, ...] = ()
    extraction_version: str = EXTRACTION_VERSION

    def as_text_document(self) -> TextDocument:
        source = (
            f"SEC {self.form} earnings release" if self.content_kind == "earnings_release"
            else f"SEC {self.form} MD&A" if self.content_kind == "mda_excerpt"
            else f"SEC {self.form}"
        )
        return TextDocument(
            symbol=self.symbol,
            available_at=self.available_at,
            source=source,
            document_id=self.accession_number,
            text=self.text,
        )


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._ignored += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data):
        if not self._ignored:
            self.parts.append(data)


@dataclass(frozen=True)
class FilingExcerpt:
    text: str
    source_text_chars: int
    start_char: int
    selection_method: str = "generic-filing-excerpt-v1"
    selection_label: str = "Filing excerpt"
    selection_quality_score: float | None = None
    selection_quality_note: str | None = None
    section_end_char: int | None = None


def _decode_sec_text(raw: bytes) -> str:
    """Decode SEC HTML without turning Windows-1252 punctuation into noise."""
    utf8 = raw.decode("utf-8", errors="replace")
    if "\ufffd" not in utf8:
        return utf8
    windows_1252 = raw.decode("cp1252", errors="replace")
    return windows_1252 if windows_1252.count("\ufffd") < utf8.count("\ufffd") else utf8


def _visible(raw_html: str) -> str:
    """Return readable SEC document text, excluding inline-XBRL metadata."""
    soup = BeautifulSoup(raw_html, "html.parser")
    hidden_names = {
        "script", "style", "head", "meta", "link", "noscript", "svg",
        "ix:header", "ix:hidden", "xbrli:context", "xbrli:unit", "xbrli:measure",
        "xbrli:entity", "xbrli:identifier", "xbrli:period", "xbrli:segment",
        "xbrli:scenario", "link:schemaref",
    }
    for node in soup.find_all():
        if node.parent is None:
            continue
        name = (node.name or "").lower()
        style = str((node.attrs or {}).get("style", ""))
        if name in hidden_names or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", style, re.I):
            node.decompose()
    return " ".join(soup.stripped_strings)


def filing_excerpt_details(raw_html: str, max_chars: int = 2_400, form: str | None = None) -> FilingExcerpt:
    """Record excerpt offsets in the normalized visible text of the filing."""
    text = _visible(raw_html)
    lowered = text.lower().translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'"}))
    form = (form or "").upper()
    markers = {
        "8-K": ("item 2.02", "item 7.01", "item 1.01"),
        "10-Q": ("management's discussion and analysis", "item 2. management"),
        "10-K": ("management's discussion and analysis", "item 7. management"),
    }.get(form, ("item 2.02", "item 7.01", "item 1.01"))
    start = 0
    for marker in markers:
        matches = [match.start() for match in re.finditer(re.escape(marker), lowered)]
        if matches:
            # SEC tables of contents usually repeat the real section heading.
            start = matches[1] if len(matches) > 1 and matches[0] < 5_000 else matches[0]
            break
    return FilingExcerpt(text[start:start + max_chars], len(text), start)


def filing_excerpt(raw_html: str, max_chars: int = 2_400, form: str | None = None) -> str:
    """Return a bounded visible-text excerpt, preferring earnings-relevant 8-K items."""
    return filing_excerpt_details(raw_html, max_chars, form).text


def recent_filing_rows(payload: dict, forms: set[str], filed_before: pd.Timestamp) -> list[dict]:
    """Normalize the SEC submissions `recent` column arrays without network I/O."""
    recent = payload.get("filings", {}).get("recent", {})
    fields = ("accessionNumber", "filingDate", "form", "primaryDocument")
    rows = [dict(zip(fields, values)) for values in zip(*(recent.get(field, []) for field in fields))]
    return sorted(
        (
            row for row in rows
            if row["form"] in forms
            and row["primaryDocument"]
            and pd.Timestamp(row["filingDate"]) <= filed_before
        ),
        key=lambda row: row["filingDate"],
        reverse=True,
    )


class _SecOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.netloc.lower() != "www.sec.gov" or not target.path.startswith("/Archives/edgar/data/"):
            raise ValueError("SEC document redirected outside the archive")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class SecDocumentProvider:
    """Retrieve bounded real SEC filing excerpts with filing-date availability."""

    def __init__(
        self,
        user_agent: str | None = None,
        fetch_json: Callable[[str, dict[str, str]], dict] | None = None,
        fetch_text: Callable[[str, dict[str, str]], str] | None = None,
        request_interval: float = 0.12,
    ):
        self.user_agent = user_agent
        self._fetch_json = fetch_json or self._network_json
        self._fetch_text = fetch_text or self._network_text
        self.request_interval = request_interval
        self._last_request_at: float | None = None

    def _headers(self) -> dict[str, str]:
        user_agent = (self.user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
        if not user_agent or "example.com" in user_agent:
            raise RuntimeError("SEC user agent not configured. Set $SEC_USER_AGENT in alphaforge/.env.")
        return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    def _wait(self) -> None:
        if self._last_request_at is not None:
            remaining = self.request_interval - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _network_json(self, url: str, headers: dict[str, str]) -> dict:
        with urlopen(Request(url, headers=headers), timeout=30) as response:  # nosec B310 - fixed SEC HTTPS endpoints
            return decode_json_bytes(response.read(), response.headers.get("Content-Encoding"))

    def _network_text(self, url: str, headers: dict[str, str]) -> str:
        with build_opener(_SecOnlyRedirect()).open(Request(url, headers=headers), timeout=30) as response:
            body = response.read(SEC_MAX_DOWNLOAD_BYTES + 1)
            if len(body) > SEC_MAX_DOWNLOAD_BYTES:
                raise ValueError("SEC document exceeds the 8 MB research limit")
            raw = decode_response_bytes(
                body,
                response.headers.get("Content-Encoding"),
                max_decoded_bytes=SEC_MAX_EXPANDED_TEXT_BYTES,
            )
            return _decode_sec_text(raw)

    def _json(self, url: str) -> dict:
        self._wait()
        return self._fetch_json(url, self._headers())

    def _text(self, url: str) -> str:
        for attempt in range(3):
            self._wait()
            try:
                return self._fetch_text(url, self._headers())
            except (HTTPError, URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                    raise
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _document(self, symbol: str, row: dict, url: str) -> SecFilingDocument:
        raw = self._text(url)
        visible = _visible(raw)
        excerpt = filing_excerpt_details(raw, form=row["form"])
        chosen_url, kind, passages = url, "filing_excerpt", ()
        note = "Primary filing excerpt; no earnings release identified."
        if row["form"] in {"10-Q", "10-K"}:
            selected_mda = select_mda_section(visible, row["form"])
            if selected_mda.section is not None:
                section = selected_mda.section
                excerpt = FilingExcerpt(
                    text=section.text,
                    source_text_chars=len(visible),
                    start_char=section.start_char,
                    selection_method=MDA_EXTRACTION_VERSION,
                    selection_label=section.label,
                    selection_quality_score=section.quality_score,
                    selection_quality_note=section.quality_note,
                    section_end_char=section.end_char,
                )
                kind = "mda_excerpt"
                note = f"{section.label} selected from the primary filing using deterministic heading and narrative-quality checks."
            else:
                note = f"Primary filing excerpt used; {selected_mda.fallback_reason}."
        if row["form"] == "8-K":
            is_earnings = bool(re.search(r"item\s+2\.02\b", _visible(raw), re.I))
            links = release_links(raw, url, earnings_item=is_earnings)
            if not links and is_earnings:
                index_url = url.rsplit("/", 1)[0] + f"/{row['accessionNumber']}-index.htm"
                try:
                    links = release_links(self._text(index_url), url, earnings_item=True)
                except (HTTPError, URLError, TimeoutError, ValueError):
                    note = "Filing index unavailable; primary filing excerpt used."
            failures = 0
            for link in links:
                try:
                    selected, full_length = earnings_passages(self._text(link))
                except (HTTPError, URLError, TimeoutError, ValueError):
                    failures += 1
                    continue
                if selected:
                    chosen_url, kind, passages = link, "earnings_release", selected
                    excerpt = FilingExcerpt(
                        "\n\n".join(p.text for p in passages), full_length, 0,
                        selection_method=EXTRACTION_VERSION,
                        selection_label="Earnings release passages",
                        selection_quality_note="Financial results and outlook paragraphs selected without model output.",
                    )
                    note = "First two results paragraphs and first outlook paragraph, where available; equal paragraph weights."
                    break
            if kind != "earnings_release" and links:
                note = ("Earnings attachment unavailable; primary filing excerpt used." if failures == len(links)
                        else "No usable earnings narrative in eligible attachments; primary filing excerpt used.")
        return SecFilingDocument(
            symbol=symbol.upper(), available_at=pd.Timestamp(row["filingDate"]), form=row["form"],
            accession_number=row["accessionNumber"], url=chosen_url, text=excerpt.text,
            source_text_chars=excerpt.source_text_chars,
            excerpt_start_char=None if passages else excerpt.start_char,
            filing_url=url, content_kind=kind, selection_note=note, passages=passages,
            selection_method=excerpt.selection_method,
            selection_label=excerpt.selection_label,
            selection_quality_score=excerpt.selection_quality_score,
            selection_quality_note=excerpt.selection_quality_note,
            section_end_char=excerpt.section_end_char,
            extraction_version=excerpt.selection_method,
        )

    def documents(
        self,
        symbols: list[str],
        forms: set[str],
        filed_before: pd.Timestamp,
        per_symbol: int,
    ) -> tuple[list[SecFilingDocument], list[str]]:
        tickers = sec_ticker_map(self._json(SEC_TICKERS_URL))
        documents: list[SecFilingDocument] = []
        warnings: list[str] = []
        for symbol in symbols:
            cik = tickers.get(symbol.upper())
            if cik is None:
                warnings.append(f"{symbol}: no current SEC ticker mapping was found.")
                continue
            try:
                payload = self._json(SEC_SUBMISSIONS_URL.format(cik=cik))
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                warnings.append(f"{symbol}: SEC filing history unavailable ({type(error).__name__}).")
                continue
            rows = recent_filing_rows(payload, forms, filed_before)[:per_symbol]
            if not rows:
                warnings.append(f"{symbol}: no selected SEC filing was available by {filed_before.date()}.")
                continue
            if len(rows) < per_symbol:
                warnings.append(
                    f"{symbol}: only {len(rows)} of {per_symbol} requested qualifying filings "
                    f"were available by {filed_before.date()}."
                )
            for row in rows:
                accession = row["accessionNumber"]
                url = SEC_ARCHIVES_URL.format(
                    cik=cik,
                    accession=accession.replace("-", ""),
                    document=quote(row["primaryDocument"]),
                )
                try:
                    document = self._document(symbol, row, url)
                except (HTTPError, URLError, TimeoutError, ValueError) as error:
                    reason = str(error).strip() or type(error).__name__
                    warnings.append(f"{symbol}: filing {accession} unavailable ({reason}).")
                    continue
                if not document.text:
                    warnings.append(f"{symbol}: {accession} did not contain usable visible text.")
                    continue
                documents.append(document)
        return documents, warnings
