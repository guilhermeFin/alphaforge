from dataclasses import replace
from urllib.error import HTTPError

import pandas as pd
import pytest

from api.service import run_real_document_batch
from research.earnings import earnings_passages, release_links, safe_attachment_url
from research.sec_documents import SecDocumentProvider
from research.text_features import FinBertExtractor

BASE = "https://www.sec.gov/Archives/edgar/data/1234/000123456724000001/report.htm"
RELEASE = BASE.replace("report.htm", "release.htm")
RESULT_ONE = "The company reported quarterly revenue of $100 million, an increase of 10 percent compared with the prior year."
RESULT_TWO = "Operating income fell to $20 million as higher production costs reduced the company's operating margin during the quarter."
OUTLOOK = "For the next quarter, the company expects revenue between $90 million and $110 million, with operating margins remaining stable."
PRIMARY = '<p>Item 2.02 Results of Operations and Financial Condition.</p><table><tr><td>99.1</td><td><a href="release.htm">Earnings release</a></td></tr></table>'
RELEASE_HTML = f"<h1>Quarterly results</h1><p>{RESULT_ONE}</p><p>{RESULT_TWO}</p><h2>Outlook</h2><p>{OUTLOOK}</p>"


@pytest.mark.parametrize("href", [
    "https://example.com/release.htm", "//evil.test/release.htm", "https://www.sec.gov.evil.test/release.htm",
    "../other/release.htm", "%2e%2e/other/release.htm", "https://user@www.sec.gov/release.htm",
    "release.pdf", "release.htm?url=http://localhost", "javascript:alert(1)",
])
def test_only_same_filing_html_is_allowed(href):
    assert safe_attachment_url(BASE, href) is None


def test_inline_xbrl_viewer_is_resolved_to_same_filing():
    assert safe_attachment_url(BASE, "/ix?doc=" + RELEASE.split("www.sec.gov")[1]) == RELEASE


def test_release_selection_rejects_unrelated_exhibits_and_slides():
    html = PRIMARY + '<a href="slides.htm">Earnings release presentation</a><a href="contract.htm">99.2 Material contract</a>'
    assert release_links(html, BASE, earnings_item=False) == [RELEASE]
    assert release_links('<a href="release.htm">99.1 Press release</a>', BASE, earnings_item=False) == []


def test_paragraph_selection_is_fixed_and_includes_mixed_tone():
    html = RELEASE_HTML + f"<p>{RESULT_ONE}</p><p>{RESULT_TWO} Much later paragraph.</p>"
    selected, size = earnings_passages(html)
    assert [p.text for p in selected] == [RESULT_ONE, RESULT_TWO, OUTLOOK]
    assert [p.section for p in selected] == ["Results", "Results", "Outlook"]
    assert size > sum(len(p.text) for p in selected)


def test_hidden_content_and_legal_language_are_not_selected():
    html = f'<div style="display:none"><p>{RESULT_TWO}</p></div><ix:hidden><p>{RESULT_TWO}</p></ix:hidden>'
    html += '<p>Forward-looking statements about revenue are subject to risks and uncertainties and actual results may differ materially from our expectations.</p>'
    html += f'<div><p><span>{RESULT_ONE}</span></p></div>'
    selected, _ = earnings_passages(html)
    assert [p.text for p in selected] == [RESULT_ONE]


def test_table_only_or_outlook_only_does_not_masquerade_as_earnings():
    assert earnings_passages(f"<table><tr><td>Revenue</td><td>100</td></tr></table><p>{OUTLOOK}</p>")[0] == ()


def provider(fetch_text):
    def fetch_json(url, _headers):
        if "company_tickers" in url:
            return {"0": {"ticker": "ABC", "cik_str": 1234}}
        return {"filings": {"recent": {
            "accessionNumber": ["0001234567-24-000001"], "filingDate": ["2024-05-02"],
            "form": ["8-K"], "primaryDocument": ["report.htm"],
        }}}
    return SecDocumentProvider(user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch_json,
                               fetch_text=fetch_text, request_interval=0)


def test_attachment_preserves_parent_date_and_both_source_urls():
    p = provider(lambda url, _: PRIMARY if url == BASE else RELEASE_HTML)
    docs, _ = p.documents(["ABC"], {"8-K"}, pd.Timestamp("2024-06-01"), 1)
    doc = docs[0]
    assert doc.content_kind == "earnings_release"
    assert doc.url == RELEASE and doc.filing_url == BASE
    assert doc.available_at == pd.Timestamp("2024-05-02")
    assert doc.excerpt_start_char is None
    assert len(doc.passages) == 3


def test_failed_attachment_returns_explicit_fallback():
    def fetch(url, _):
        if url == BASE:
            return PRIMARY
        raise HTTPError(url, 404, "missing", {}, None)
    docs, _ = provider(fetch).documents(["ABC"], {"8-K"}, pd.Timestamp("2024-06-01"), 1)
    assert docs[0].url == BASE
    assert docs[0].content_kind == "filing_excerpt"
    assert "unavailable" in docs[0].selection_note


def test_filing_index_is_a_bounded_fallback_for_missing_links():
    fetched = []
    def fetch(url, _):
        fetched.append(url)
        if url == BASE:
            return "<p>Item 2.02 Results of Operations.</p>"
        if url.endswith("-index.htm"):
            return PRIMARY
        return RELEASE_HTML
    docs, _ = provider(fetch).documents(["ABC"], {"8-K"}, pd.Timestamp("2024-06-01"), 1)
    assert len(fetched) == 3 and docs[0].url == RELEASE


def test_batch_averages_every_selected_passage_and_records_exact_inputs():
    calls = []
    class Client:
        def text_classification(self, text, model):
            calls.append(text)
            positive = 0.1 if "fell" in text else 0.7
            return [{"label": "positive", "score": positive}, {"label": "negative", "score": 0.8 - positive},
                    {"label": "neutral", "score": 0.2}]
    out = run_real_document_batch({"symbols": ["ABC"], "as_of": "2024-06-01"},
        document_provider=provider(lambda url, _: PRIMARY if url == BASE else RELEASE_HTML),
        text_extractor=FinBertExtractor(client=Client()))
    record = out["features"][0]
    assert calls == [RESULT_ONE, RESULT_TWO, OUTLOOK]
    assert record["positive_probability"] == pytest.approx(0.5)
    assert record["sentiment"] == pytest.approx(0.2)
    assert record["passage_count"] == 3
    assert [p["analyzed_text"] for p in record["review"]["passages"]] == calls
    assert record["filing_url"] == BASE and record["source_url"] == RELEASE
    assert out["source_validation"]["coverage_rate"] == 1.0


def test_failed_passage_never_produces_a_partial_optimistic_average():
    p = provider(lambda url, _: PRIMARY if url == BASE else RELEASE_HTML)
    documents, _ = p.documents(["ABC"], {"8-K"}, pd.Timestamp("2024-06-01"), 1)
    class TwoDocuments:
        def documents(self, *_args):
            return documents + [replace(documents[0], symbol="DEF", passages=(), text=RESULT_ONE)], []
    class Client:
        def text_classification(self, text, model):
            if "fell" in text:
                raise ValueError("provider rejection")
            return [{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.2}]
    out = run_real_document_batch({"symbols": ["ABC", "DEF"], "as_of": "2024-06-01"},
                                 document_provider=TwoDocuments(), text_extractor=FinBertExtractor(client=Client()))
    assert [r["symbol"] for r in out["features"]] == ["DEF"]
    assert "no score substituted" in out["warnings"][0]
