from research.filing_sections import MDA_EXTRACTION_VERSION, select_mda_section


def _narrative(sentence: str, count: int = 90) -> str:
    return " ".join(f"{sentence}." for _ in range(count))


def test_10k_selector_skips_table_of_contents_and_retains_full_section_range():
    toc = "Table of contents. Item 7. Management's Discussion and Analysis. Item 7A. Market Risk."
    body = (
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations. "
        + _narrative("Management discusses operating results, liquidity, customer demand, and capital allocation")
        + " Item 7A. Quantitative and Qualitative Disclosures About Market Risk."
    )
    text = toc + " " + body

    selection = select_mda_section(text, "10-K")

    assert selection.section is not None
    assert selection.section.text.startswith("Item 7. Management's Discussion")
    assert selection.section.start_char == text.index("Item 7. Management", text.index("Item 7A") + 1)
    assert selection.section.end_char == text.index("Item 7A. Quantitative")
    assert selection.section.candidate_count == 2
    assert selection.section.label == "MD&A (Item 7)"
    assert selection.section.quality_score > 0


def test_10q_selector_uses_item_two_and_stops_before_item_three():
    text = (
        "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations. "
        + _narrative("The company explains quarterly revenue trends, costs, liquidity, and operating performance")
        + " Item 3 Quantitative and Qualitative Disclosures About Market Risk."
    )

    selection = select_mda_section(text, "10-Q")

    assert selection.section is not None
    assert selection.section.label == "MD&A (Item 2)"
    assert "Item 3 Quantitative" not in selection.section.text
    assert selection.section.end_char == text.index("Item 3 Quantitative")


def test_table_or_xbrl_heavy_mda_candidate_is_rejected_with_a_visible_reason():
    table_like = " ".join(
        f"us-gaap:Revenue {index} iso4217:USD {index * 10}" for index in range(120)
    )
    text = (
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations. "
        + table_like
        + " Item 7A. Quantitative and Qualitative Disclosures About Market Risk."
    )

    selection = select_mda_section(text, "10-K")

    assert selection.section is None
    assert "inline-XBRL" in (selection.fallback_reason or "")


def test_unsupported_forms_are_an_explicit_non_selection():
    selection = select_mda_section("Item 7. Management's Discussion and Analysis.", "8-K")

    assert selection.section is None
    assert "10-Q and 10-K" in (selection.fallback_reason or "")
    assert MDA_EXTRACTION_VERSION == "mda-section-v1"
