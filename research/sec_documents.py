"""Small, date-provenanced SEC filing excerpts for batch text research."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from .http import decode_json_bytes, decode_response_bytes
from .providers import SEC_TICKERS_URL, sec_ticker_map
from .text_features import TextDocument

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SUPPORTED_FORMS = {"8-K", "10-Q", "10-K"}


@dataclass(frozen=True)
class SecFilingDocument:
    symbol: str
    available_at: pd.Timestamp
    form: str
    accession_number: str
    url: str
    text: str

    def as_text_document(self) -> TextDocument:
        return TextDocument(
            symbol=self.symbol,
            available_at=self.available_at,
            source=f"SEC {self.form}",
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


def filing_excerpt(raw_html: str, max_chars: int = 2_400) -> str:
    """Return a bounded visible-text excerpt, preferring earnings-relevant 8-K items."""
    parser = _VisibleText()
    parser.feed(raw_html)
    text = " ".join(" ".join(parser.parts).split())
    lowered = text.lower()
    starts = [lowered.find(marker) for marker in ("item 2.02", "item 7.01", "item 1.01")]
    start = min((value for value in starts if value >= 0), default=0)
    return text[start:start + max_chars]


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
        with urlopen(Request(url, headers=headers), timeout=30) as response:  # nosec B310 - fixed SEC HTTPS endpoints
            raw = decode_response_bytes(response.read(), response.headers.get("Content-Encoding"))
            return raw.decode("utf-8", errors="replace")

    def _json(self, url: str) -> dict:
        self._wait()
        return self._fetch_json(url, self._headers())

    def _text(self, url: str) -> str:
        self._wait()
        return self._fetch_text(url, self._headers())

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
            payload = self._json(SEC_SUBMISSIONS_URL.format(cik=cik))
            rows = recent_filing_rows(payload, forms, filed_before)[:per_symbol]
            if not rows:
                warnings.append(f"{symbol}: no selected SEC filing was available by {filed_before.date()}.")
                continue
            for row in rows:
                accession = row["accessionNumber"]
                url = SEC_ARCHIVES_URL.format(
                    cik=cik,
                    accession=accession.replace("-", ""),
                    document=quote(row["primaryDocument"]),
                )
                excerpt = filing_excerpt(self._text(url))
                if not excerpt:
                    warnings.append(f"{symbol}: {accession} did not contain usable visible text.")
                    continue
                documents.append(SecFilingDocument(
                    symbol=symbol.upper(), available_at=pd.Timestamp(row["filingDate"]), form=row["form"],
                    accession_number=accession, url=url, text=excerpt,
                ))
        return documents, warnings
