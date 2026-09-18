import pandas as pd

from research.sec_documents import SecDocumentProvider, filing_excerpt, recent_filing_rows


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
