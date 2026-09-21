import pandas as pd

from research.sec_documents import SecDocumentProvider, _decode_sec_text, filing_excerpt, filing_excerpt_details, recent_filing_rows


def test_recent_filing_rows_filters_forms_dates_and_sorts_newest_first():
    payload = {"filings": {"recent": {
        "accessionNumber": ["old", "new", "ignored"],
        "filingDate": ["2024-05-01", "2024-06-01", "2024-07-01"],
        "form": ["8-K", "10-Q", "10-K"],
        "primaryDocument": ["old.htm", "new.htm", "ignored.htm"],
    }}}
    rows = recent_filing_rows(payload, {"8-K", "10-Q"}, pd.Timestamp("2024-06-15"))
    assert [row["accessionNumber"] for row in rows] == ["new", "old"]


def test_filing_excerpt_prefers_earnings_item_and_strips_markup():
    excerpt = filing_excerpt("<html><body>Header <b>Item 2.02</b> Revenue improved. <script>bad()</script></body></html>")
    assert excerpt.startswith("Item 2.02 Revenue improved.")
    assert "bad" not in excerpt


def test_filing_excerpt_removes_inline_xbrl_header_before_selecting_10q_narrative():
    html = """
    <ix:header><xbrli:context>us-gaap:RetainedEarningsMember 0001045810</xbrli:context></ix:header>
    <p>Table of contents Item 2. Management's discussion</p>
    <p>Other material.</p><p>Management's discussion and analysis. Revenue improved during the quarter.</p>
    """
    excerpt = filing_excerpt(html, form="10-Q")
    assert excerpt.startswith("Management's discussion and analysis.")
    assert "us-gaap" not in excerpt


def test_sec_text_decoder_preserves_windows_1252_curly_apostrophes_for_section_matching():
    raw = b"Management" + bytes([0x92]) + b"s discussion"
    assert _decode_sec_text(raw) == "Management\u2019s discussion"


def test_excerpt_position_accounts_for_skipped_header_and_omitted_tail():
    visible = "Header Item 2.02 Revenue improved. " + "Other details. " * 200
    details = filing_excerpt_details(f"<p>{visible}</p>", max_chars=40)
    assert details.source_text_chars == len(visible.strip())
    assert details.start_char == len("Header ")
    assert visible[details.start_char:details.start_char + len(details.text)] == details.text
    assert len(details.text) == 40


def test_complete_short_filing_has_no_omitted_text():
    details = filing_excerpt_details("<p>Revenue improved.</p>")
    assert details.start_char == 0
    assert details.source_text_chars == len(details.text)


def test_document_provider_keeps_filing_date_and_sec_url():
    def fetch_json(url, _headers):
        if "company_tickers" in url:
            return {"0": {"ticker": "ABC", "cik_str": 1234}}
        return {"filings": {"recent": {
            "accessionNumber": ["0001234567-24-000001"],
            "filingDate": ["2024-05-02"],
            "form": ["8-K"],
            "primaryDocument": ["earnings.htm"],
        }}}

    provider = SecDocumentProvider(
        user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch_json,
        fetch_text=lambda *_args: "<p>Item 2.02 Revenue improved.</p>", request_interval=0,
    )
    documents, warnings = provider.documents(["ABC"], {"8-K"}, pd.Timestamp("2024-06-01"), 1)
    assert not warnings and len(documents) == 1
    assert documents[0].available_at == pd.Timestamp("2024-05-02")
    assert documents[0].accession_number == "0001234567-24-000001"
    assert "1234/000123456724000001/earnings.htm" in documents[0].url
    assert documents[0].source_text_chars == len("Item 2.02 Revenue improved.")
    assert documents[0].excerpt_start_char == 0


def test_document_provider_reports_when_fewer_qualifying_filings_exist_than_requested():
    def fetch_json(url, _headers):
        if "company_tickers" in url:
            return {"0": {"ticker": "ABC", "cik_str": 1234}}
        return {"filings": {"recent": {
            "accessionNumber": ["0001234567-24-000001"], "filingDate": ["2024-05-02"],
            "form": ["8-K"], "primaryDocument": ["earnings.htm"],
        }}}

    provider = SecDocumentProvider(user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch_json,
                                   fetch_text=lambda *_args: "<p>Item 2.02 Revenue improved.</p>", request_interval=0)
    documents, warnings = provider.documents(["ABC"], {"8-K"}, pd.Timestamp("2024-06-01"), 2)
    assert len(documents) == 1
    assert "only 1 of 2 requested" in warnings[0]


def test_document_provider_uses_clean_mda_section_for_10k_with_a_visible_audit():
    narrative = " ".join(
        "Management discusses operating performance, liquidity, costs, demand, and capital allocation."
        for _ in range(100)
    )
    html = (
        "<p>Table of contents Item 7. Management's Discussion and Analysis. Item 7A. Market Risk.</p>"
        "<p>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations. "
        f"{narrative} Item 7A. Quantitative and Qualitative Disclosures About Market Risk.</p>"
    )

    def fetch_json(url, _headers):
        if "company_tickers" in url:
            return {"0": {"ticker": "ABC", "cik_str": 1234}}
        return {"filings": {"recent": {
            "accessionNumber": ["0001234567-24-000001"], "filingDate": ["2024-05-02"],
            "form": ["10-K"], "primaryDocument": ["annual.htm"],
        }}}

    provider = SecDocumentProvider(
        user_agent="AlphaForge test@alphaforge.local", fetch_json=fetch_json,
        fetch_text=lambda *_args: html, request_interval=0,
    )
    documents, warnings = provider.documents(["ABC"], {"10-K"}, pd.Timestamp("2024-06-01"), 1)

    assert not warnings
    document = documents[0]
    assert document.content_kind == "mda_excerpt"
    assert document.selection_method == "mda-section-v1"
    assert document.extraction_version == "mda-section-v1"
    assert document.selection_label == "MD&A (Item 7)"
    assert document.selection_quality_score is not None
    assert document.excerpt_start_char is not None
    assert document.section_end_char is not None
    assert document.excerpt_start_char < document.section_end_char
    assert document.text.startswith("Item 7. Management's Discussion")
