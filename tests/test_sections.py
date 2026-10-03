"""Section isolation: synthetic trap cases plus golden checks on 20 recorded 10-K filings."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from edgar_extract.htmltext import Block, html_to_blocks, normalize_text
from edgar_extract.pipeline import section_report
from edgar_extract.sections import (
    Method,
    SectionStatus,
    boilerplate_mask,
    extract_sections,
    find_headings,
    isolate,
    page_numbers,
    parse_page_list,
    referenced_pages,
    title_heading,
)

from .conftest import FIXTURES, fixture_bytes, fixture_filings

LOREM = (
    "Revenue increased 12% to $1.2 billion driven by higher demand across all regions and "
    "continued growth in our services business, partially offset by currency headwinds. "
) * 30  # ~5k chars, well above the stub threshold
RISK = "We face intense competition that could reduce our revenue and margins. " * 40

TOC = """
<table>
 <tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
 <tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>12</td></tr>
 <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>20</td></tr>
 <tr><td>Item 7A.</td><td>Quantitative and Qualitative Disclosures</td><td>30</td></tr>
 <tr><td>Item 8.</td><td>Financial Statements</td><td>31</td></tr>
</table>
"""


def doc(body: str) -> str:
    return f"<html><head><title>x</title><style>p{{}}</style></head><body>{body}</body></html>"


def standard_filing(
    item7_heading: str = "<p>Item 7. Management's Discussion and Analysis</p>",
) -> str:
    return doc(
        f"""
        <p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
        {TOC}
        <p>Item 1. Business</p><p>We make things.</p>
        <p>Item 1A. Risk Factors</p><p>{RISK}</p>
        <p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
        <p>Item 2. Properties</p><p>We own buildings.</p>
        {item7_heading}<p>{LOREM}</p><p>Second MD&amp;A paragraph.</p>
        <p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p><p>Rates.</p>
        <p>Item 8. Financial Statements and Supplementary Data</p><p>Statements.</p>
        """
    )


# -- rendering -------------------------------------------------------------------------


def test_normalize_text_folds_typography() -> None:
    raw = (
        "Item\N{NO-BREAK SPACE}7\N{EM DASH}Management\N{RIGHT SINGLE QUOTATION MARK}s"
        "\N{ZERO WIDTH SPACE}  Discussion \N{MINUS SIGN}(3.1)"
    )
    assert normalize_text(raw) == "Item 7-Management's Discussion -(3.1)"


def test_hidden_content_and_ixbrl_header_dropped() -> None:
    blocks = html_to_blocks(
        doc(
            '<div style="display:none"><ix:header>Revenues 999</ix:header></div>'
            "<script>var x=1;</script><p>Visible text.</p>"
            '<p>Revenue was $<ix:nonfraction name="us-gaap:Revenues">5.0</ix:nonfraction>'
            " billion.</p>"
        )
    )
    assert [b.text for b in blocks] == ["Visible text.", "Revenue was $5.0 billion."]


def test_table_row_cells_join_into_one_line() -> None:
    blocks = html_to_blocks(
        doc("<table><tr><td>Item 7.</td><td></td><td>MD&amp;A</td></tr></table>")
    )
    assert blocks == [Block("table_row", "Item 7. MD&A", 0)]


def test_split_inline_heading_reads_as_one_block() -> None:
    blocks = html_to_blocks(
        doc("<p><b>I</b><b>TEM</b><span>&nbsp;7.</span> <i>Management&#8217;s</i></p>")
    )
    assert blocks[0].text == "ITEM 7. Management's"


def test_nested_table_rows_are_rendered_individually() -> None:
    html = doc(
        "<table><tr><td><p>Outer paragraph.</p>"
        "<table><tr><td>Net sales</td><td>$</td><td>391,035</td></tr></table>"
        "</td></tr></table>"
    )
    texts = [(b.kind, b.text) for b in html_to_blocks(html)]
    assert ("p", "Outer paragraph.") in texts
    assert ("table_row", "Net sales $ 391,035") in texts


def test_page_breaks_counted() -> None:
    html = doc(
        '<p>one</p><hr style="page-break-after:always"/><p>two</p>'
        '<div style="break-before: page"><p>three</p></div>'
    )
    assert [(b.text, b.page) for b in html_to_blocks(html)] == [
        ("one", 0),
        ("two", 1),
        ("three", 2),
    ]


# -- headings ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Item 7. Management's Discussion", "7"),
        ("ITEM 7 - MANAGEMENT'S DISCUSSION", "7"),
        ("Item 7: MD&A", "7"),
        ("Item 7", "7"),
        ("ITEM 1A.RISK FACTORS", "1A"),
        ("Item 7A. Quantitative and Qualitative", "7A"),
        ("PART II - Item 7. MD&A", "7"),
        ("Item 1B, 1C", "1B"),
        ("Items 7 And 7A", "7"),
    ],
)
def test_heading_variants(text: str, key: str) -> None:
    (head,) = find_headings([Block("p", text)])
    assert head.key == key


@pytest.mark.parametrize(
    "text",
    [
        "Item 7 of this Annual Report discusses our results.",
        "Itemized deductions were $5 million.",
        "Items 7 and 7A are incorporated by reference.",
        "Item 7. " + "x" * 400,
    ],
)
def test_not_headings(text: str) -> None:
    assert find_headings([Block("p", text)]) == []


# -- isolation traps ------------------------------------------------------------------


def test_takes_body_not_table_of_contents() -> None:
    sections = extract_sections(standard_filing())
    s7, s1a = sections["7"], sections["1A"]
    assert s7.status is SectionStatus.OK and s7.method is Method.ITEM_HEADING
    assert s7.start_heading == "Item 7. Management's Discussion and Analysis"
    assert s7.end_heading is not None and s7.end_heading.startswith("Item 7A.")
    assert s7.text.startswith("Revenue increased 12%")
    assert s7.text.endswith("Second MD&A paragraph.")
    assert "Rates." not in s7.text  # Item 7A is the end marker, not part of Item 7
    assert s1a.text.startswith("We face intense competition")
    assert s1a.end_heading == "Item 1B. Unresolved Staff Comments"


@pytest.mark.parametrize(
    "heading",
    [
        "<p>ITEM 7 &#8212; MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS</p>",
        "<p><b>Item</b><b>&nbsp;7.</b></p><p>Management's Discussion and Analysis</p>",
        "<table><tr><td>Item&nbsp;7.</td><td>Management&#8217;s Discussion</td></tr></table>",
        "<div><span>ITEM 7:</span><span> MD&amp;A</span></div>",
    ],
)
def test_heading_variants_isolate_same_body(heading: str) -> None:
    s7 = extract_sections(standard_filing(heading))["7"]
    assert s7.status is SectionStatus.OK
    assert "Revenue increased 12%" in s7.text
    assert "Rates." not in s7.text and "We own buildings." not in s7.text


def test_paragraph_boundaries_and_offsets() -> None:
    s7 = extract_sections(standard_filing())["7"]
    assert "\n\n" in s7.text
    assert s7.block_kinds[0] == "p"
    for start in s7.block_starts:
        assert start == 0 or s7.text[start - 2 : start] == "\n\n"


def test_incorporated_by_reference_flagged_not_guessed() -> None:
    html = doc(
        f"""{TOC}<p>Item 1A. Risk Factors</p><p>{RISK}</p><p>Item 1B. Unresolved Staff Comments</p>
        <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
        <p>Information in response to this Item 7 can be found in the 2024 Annual Report to
        Shareholders under "Financial Review." That information is incorporated into this item by
        reference.</p>
        <p>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p>
        <p>Financial Review. {LOREM}</p>"""
    )
    s7 = extract_sections(html)["7"]
    assert s7.status is SectionStatus.MISSING_BY_REFERENCE
    assert "Financial Review. Revenue" not in s7.text  # never pulls text from elsewhere
    assert s7.block_starts == []


def test_exhibit_13_reference_flagged() -> None:
    html = doc(
        "<p>Item 7. Management's Discussion and Analysis</p><p>The information required by this "
        "item is set forth in Exhibit 13 to this report.</p><p>Item 7A. Market Risk</p>"
    )
    assert extract_sections(html)["7"].status is SectionStatus.MISSING_BY_REFERENCE


def test_running_page_headers_removed_and_do_not_split_section() -> None:
    pages = "".join(
        f'<div style="page-break-before:always"><p>Item 7</p>'
        f"<p>{LOREM[:800]} part {i}.</p><p>{i + 20}</p></div>"
        for i in range(6)
    )
    html = doc(
        f"{TOC}<p>Item 7. Management's Discussion and Analysis</p>{pages}"
        "<p>Item 7A. Market Risk</p>"
    )
    s7 = extract_sections(html)["7"]
    assert s7.status is SectionStatus.OK
    assert "part 0." in s7.text and "part 5." in s7.text
    assert "\n\nItem 7\n\n" not in s7.text
    assert "\n\n21\n\n" not in s7.text  # bare page numbers stripped


def test_item_1a_ends_at_1c_when_1b_absent() -> None:
    html = doc(
        f"<p>Item 1A. Risk Factors</p><p>{RISK}</p><p>Item 1C. Cybersecurity</p><p>Cyber.</p>"
    )
    s = extract_sections(html)["1A"]
    assert s.end_heading == "Item 1C. Cybersecurity" and "Cyber." not in s.text


def test_runs_to_end_when_no_later_heading() -> None:
    html = doc(f"<p>Item 7. MD&amp;A</p><p>{LOREM}</p>")
    s = extract_sections(html)["7"]
    assert s.status is SectionStatus.OK and s.end_heading is None
    assert "end of document" in s.detail


def test_missing_item_reported() -> None:
    s = extract_sections(doc("<p>Part III amendment.</p><p>Item 10. Directors</p>"))["7"]
    assert s.status is SectionStatus.NOT_FOUND and s.text == ""


def test_not_applicable_stub() -> None:
    html = doc("<p>Item 1A. Risk Factors</p><p>Not applicable.</p><p>Item 1B. None</p>")
    assert extract_sections(html)["1A"].status is SectionStatus.NOT_APPLICABLE


def _paged(pages: list[str], footer: str = "2024 FORM 10-K {n}") -> str:
    out = []
    for n, body in enumerate(pages, start=1):
        out.append(f'<div style="page-break-after:always">{body}<p>{footer.format(n=n)}</p></div>')
    return doc("".join(out))


def test_page_reference_resolved_from_printed_page_numbers() -> None:
    html = _paged(
        [
            "<p>Item 1A. Risk Factors</p><p>" + RISK + "</p><p>Item 1B. None</p>"
            "<p>Item 7. Management's Discussion and Analysis</p>"
            "<p>Management's discussion and analysis appears on pages 3-4. Financial statements "
            "appear on pages 5-6.</p><p>Item 7A. Market Risk</p>",
            "<p>Other material.</p>",
            "<p>Management's Discussion and Analysis</p><p>MD&amp;A page three. "
            + LOREM[:500]
            + "</p>",
            "<p>MD&amp;A page four.</p>",
            "<p>Consolidated statements page five.</p>",
            "<p>Notes page six.</p>",
        ]
    )
    s = extract_sections(html)["7"]
    assert s.status is SectionStatus.OK and s.method is Method.PAGE_REFERENCE
    assert s.pages == [3, 4]
    assert s.text.startswith("Management's Discussion and Analysis\n\nMD&A page three.")
    assert s.text.endswith("MD&A page four.")
    assert "page five" not in s.text and "FORM 10-K" not in s.text


def test_cross_reference_index_row_resolved_and_trimmed() -> None:
    html = _paged(
        [
            "<p>Business overview.</p>",
            "<p>Tail of business section.</p><p>RISK FACTORS. Our business faces risks. "
            + RISK
            + "</p>",
            "<p>More risks.</p><p>LEGAL PROCEEDINGS. See Note 20.</p>",
            "<table><tr><td>Item 1A. Risk Factors</td><td>2-3</td></tr>"
            "<tr><td>Item 1B. Unresolved Staff Comments</td><td>Not applicable</td></tr></table>",
        ]
    )
    s = extract_sections(html)["1A"]
    assert s.method is Method.PAGE_REFERENCE and s.pages == [2, 3]
    assert s.text.startswith("RISK FACTORS. Our business faces risks.")
    assert s.text.endswith("More risks.")
    assert "trimmed" in s.detail


def test_named_section_reference_resolved_from_running_headers() -> None:
    title = "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    pages = [
        "<p>Item 7. Management's Discussion and Analysis</p><p>Reference is made to the section "
        "entitled \"Management's Discussion and Analysis of Financial Condition and Results of "
        'Operations" in the Financial Section of this report.</p><p>Item 7A. Market Risk</p>',
        f"<p>Prior section text.</p><p>{title}</p><p>FUNCTIONAL EARNINGS were strong.</p>",
    ] + [f"<p>{title}</p><p>MD&amp;A page {n}. {LOREM[:300]}</p>" for n in range(3, 7)]
    s = extract_sections(_paged(pages))["7"]
    assert s.status is SectionStatus.OK and s.method is Method.NAMED_SECTION
    assert s.text.startswith(title)
    assert "Prior section text." not in s.text
    assert "MD&A page 6." in s.text


def test_named_reference_unresolved_is_reported() -> None:
    html = doc(
        '<p>Item 7. MD&amp;A</p><p>Reference is made to the section entitled "Nowhere To Be '
        'Found" of this report.</p><p>Item 7A. Market Risk</p>'
    )
    assert extract_sections(html)["7"].status is SectionStatus.REFERENCE_UNRESOLVED


# -- helpers ---------------------------------------------------------------------------


def test_parse_page_list() -> None:
    assert parse_page_list("28-30, 30") == [28, 29, 30, 30]
    assert parse_page_list("5-6, 21-23, 79") == [5, 6, 21, 22, 23, 79]
    assert parse_page_list("52 through 54") == [52, 53, 54]


def test_referenced_pages_prefers_first_sentence_reference() -> None:
    stub = [
        "Management's discussion and analysis appears on pages 52-54. Such information should be "
        "read with the financial statements, which appear on pages 172-321."
    ]
    assert referenced_pages("Item 7. MD&A", stub) == [52, 53, 54]


def test_referenced_pages_ignores_single_toc_page_number() -> None:
    assert referenced_pages("Item 7. Management's Discussion and Analysis 23", []) == []
    assert referenced_pages("Item 7. Management's Discussion and Analysis 8-10", []) == [8, 9, 10]


def test_page_numbers_need_neighbor_agreement() -> None:
    blocks = [
        Block("p", "body", 0),
        Block("p", "1", 0),
        Block("p", "body", 1),
        Block("p", "Total 77", 1),  # a table total, not page 2
        Block("p", "body", 2),
        Block("p", "2024 FORM 10-K 3", 2),
        Block("p", "4 2024 FORM 10-K", 3),
    ]
    numbers = page_numbers(blocks)
    assert numbers[2] == 3 and numbers[3] == 4
    assert 1 not in numbers  # "Total 77" has no neighbor at 76/78


def test_boilerplate_keeps_real_heading_drops_running_header() -> None:
    blocks = [Block("p", "Item 7", p) for p in range(5)] + [
        Block("p", "Item 7. Management's Discussion", 5),
        Block("p", "body text", 5),
    ]
    mask = boilerplate_mask(blocks)
    assert mask[:5] == [True] * 5 and mask[5:] == [False, False]


def test_title_heading_rules() -> None:
    assert title_heading("RISK FACTORS. The following discussion " + "x" * 300, "1A")
    assert title_heading("Risk Factors", "1A")
    assert not title_heading("Risk factors that could affect us include " + "x" * 300, "1A")
    assert not title_heading("Legal proceedings - We are subject to " + "x" * 300, "3")


def test_isolate_on_empty_blocks() -> None:
    assert isolate([], "7").status is SectionStatus.NOT_FOUND


# -- golden checks on recorded filings ----------------------------------------------------

EXPECTATIONS = json.loads((FIXTURES / "section_expectations.json").read_text())


def test_at_least_15_fixture_filings() -> None:
    assert len(fixture_filings()) >= 15
    ok = {(e["url"], e["item"]) for e in EXPECTATIONS if e["status"] == "ok"}
    assert len({u for u, _ in ok}) >= 15


@pytest.mark.parametrize(
    "exp",
    EXPECTATIONS,
    ids=[f"{e['ticker']}-{e['fiscal_year']}-{e['form']}-{e['item']}" for e in EXPECTATIONS],
)
def test_recorded_filing_sections(exp: dict[str, object]) -> None:
    html = fixture_bytes(str(exp["url"]))
    assert html is not None
    sec = extract_sections(html, items=(str(exp["item"]),))[str(exp["item"])]
    assert sec.status.value == exp["status"]
    assert sec.method.value == exp["method"]
    assert (sec.start_heading or "")[:40] == exp["start_heading"]
    assert (sec.end_heading or "")[:40] == exp["end_heading"]
    assert sec.text[:60] == exp["first_text"]
    assert sec.text[-60:] == exp["last_text"]
    if exp["status"] == "ok":
        assert sec.n_chars >= 10_000  # non-empty, real section
        if exp["method"] == "item_heading":
            assert (
                str(exp["start_heading"])
                .lower()
                .replace(" ", "")
                .startswith(("item1a", "item7", "part"))
            )


# -- section report ---------------------------------------------------------------------


def test_section_report_flags_outliers() -> None:
    rows = []
    for t in ("A", "B", "C"):
        for fy in range(2020, 2025):
            n = 50_000
            if (t, fy) == ("A", 2024):
                n = 3_000  # far too short
            if (t, fy) == ("B", 2024):
                n = 120_000  # 2.4x the company's median
            rows.append(
                {
                    "ticker": t,
                    "fiscal_year": fy,
                    "form": "10-K",
                    "item": "7",
                    "status": "ok",
                    "n_chars": n,
                }
            )
    rows.append(
        {
            "ticker": "C",
            "fiscal_year": 2025,
            "form": "10-K",
            "item": "7",
            "status": "section_missing_by_reference",
            "n_chars": 150,
        }
    )
    df = pd.DataFrame(rows)
    for col in (
        "cik",
        "is_amendment",
        "accession",
        "filing_date",
        "report_date",
        "primary_doc_url",
        "method",
        "text",
        "n_paragraphs",
        "n_table_rows",
        "start_heading",
        "end_heading",
        "pages",
        "detail",
        "block_starts",
        "block_kinds",
    ):
        df[col] = None
    report = section_report(df)
    flagged = report[report["outlier"]].set_index(["ticker", "fiscal_year"])["outlier_reason"]
    assert set(flagged.index) == {("A", 2024), ("B", 2024)}
    assert "fewer than 5000 chars" in flagged[("A", 2024)]
    assert "company median" in flagged[("B", 2024)]
    assert not report[report["status"] != "ok"]["outlier"].any()
