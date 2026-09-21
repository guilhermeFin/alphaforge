# Earnings Research

## Scope

The filing research workspace selects the latest one or two requested filings per
company, through the declared filing-date cutoff. It does not search further back
until it finds positive news, and it does not claim that every 8-K contains an
earnings announcement. The default batch is five companies; a bounded validation
run can use up to 25 companies and 50 requested documents.

## Fixed Extraction Rules: earnings-v1 and mda-section-v1

1. Inspect linked HTML documents in the same SEC accession directory. Prefer
   earnings-release descriptions, then press releases associated with Item 2.02.
   Presentations and slides are excluded. The filing index is a fallback when
   an Item 2.02 filing has no eligible attachment link.
2. Fetch at most three candidate attachments. Off-domain links, other filings,
   executable URLs, non-HTML files, and off-archive redirects are rejected.
3. Select the first two financial-results narrative paragraphs and first outlook
   paragraph, where available, in their original order. Remove hidden content,
   legal disclaimers, short labels, and numerical tables. Rules use financial
   vocabulary, never sentiment scores. This is a bounded narrative sample, not a
   complete earnings summary. Some HTML layouts will not yield usable paragraphs.
4. Apply the existing 1,200-character FinBERT input cap to each paragraph. This
   is not an exact token-count guarantee. Capture the precise transmitted text,
   original passage, model name, and each classification score.
5. Average all selected paragraph scores equally. If any selected paragraph fails,
   omit the document score and report the failure; never average only the successes.

For 10-Q and 10-K primary filings, AlphaForge first searches for the appropriate
MD&A heading: Item 2 for a quarterly filing and Item 7 for an annual filing. It
uses the next statutory section boundary to define the candidate, rejects sections
that are too short or dominated by inline-XBRL, numeric table text, or a table of
contents, and scores a bounded opening excerpt. Heading choice and quality checks
never depend on a FinBERT score. The audit preserves the full-source character
range, selection version, quality note, and exact model input.

If no clean MD&A section or suitable earnings attachment can be read, the primary
document remains an explicitly labeled **Filing excerpt**; the fallback reason is
shown rather than replacing it with a substitute score. Source and parent-filing
URLs are separate. Availability is the parent filing date, with **no verified
intraday time**. These records must not be treated as midnight-tradable historical
signals.

## Review And Export

The workspace shows release/fallback counts, per-document scores, source links,
paragraph selection, shortening information, and exact model inputs. CSV exports
contain displayed scores. JSON audit exports also contain passage-level scores,
source metadata, warnings, and the extraction-rule version. Changing the selected
document or passage does not make an inference request.

## Event Study And Saved History

The optional post-filing event study uses adjusted company prices and SPY only
after a filing's date. Because SEC filing dates do not establish an intraday
publication time, the first eligible price is the **next trading session**. It
compares a simple training-only text relationship with a no-text average-return
baseline on a later chronological holdout. Labels that cross the holdout boundary
are purged. A result is exploratory unless it has at least 12 held-out events,
lower held-out squared error than the baseline, and a two-sided correlation
test below 0.05. This measures explanatory association, not tradable performance.

The headline cohort is screened for **model-time integrity**: an observation whose
scoring model did not yet exist when the document appeared is held out, because no
researcher could have produced that score at the time. A descriptive
all-observations view is retained separately, with its verdict fields removed so it
cannot be read as a conclusion. The two cohorts are never blended, and the coverage
consequence is shown in the workspace. Filing dates, training-corpus dates, and
model-release dates are three distinct facts; see
[model_time_integrity.md](model_time_integrity.md) for the registered values, their
sources, and the limits of the check.

Completed backtests, coverage checks, filing batches, and event studies are saved
locally in `data/alphaforge-workspace.db` by default. The Research history workspace
compares their native metrics side by side and exports an audit JSON. It does not
invent a score that would compare unrelated research types.

## App Workspaces

- **Strategy lab:** configured backtests, trial ledger, validation, factor attribution,
  model comparisons, and result export. Synthetic results carry a visible demo warning.
- **Data quality:** SEC company coverage, ALFRED vintage availability, an optional dated
  text check, and downloadable coverage reports.
- **Filing research:** filing and earnings-attachment analysis with source review.
- **Research history:** locally saved research records and side-by-side comparison.

The three pages retain their existing URLs and native Streamlit session navigation.
Shared theme and presentation live in `app/ui.py`, `app/styles.css`, and
`.streamlit/config.toml`. API payloads for existing consumers remain compatible;
filing responses add source-selection and passage-level audit fields.

## References

- [SEC Form 8-K](https://www.sec.gov/files/form8-k.pdf): Item 2.02 and exhibits.
- [SEC filing index example](https://www.sec.gov/Archives/edgar/data/1576280/000157628026000024/0001576280-26-000024-index.htm).
- [Streamlit page navigation](https://docs.streamlit.io/develop/api-reference/widgets/st.page_link).
- [FinBERT](https://arxiv.org/abs/1908.10063) (Araci 2019) and its [Hugging Face weights](https://huggingface.co/ProsusAI/finbert): the scoring model whose availability is screened.
- [Look-Ahead-Bench](https://arxiv.org/abs/2601.13770) (Benhenda 2026): motivation for treating model availability as a point-in-time question.

Neutral or negative scores are legitimate outcomes. Better extraction must be judged
by source relevance and reproducibility, not by whether scores become stronger.
