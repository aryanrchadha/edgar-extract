from __future__ import annotations

import pandas as pd
import pytest

from edgar_extract.baseline_extract import NER_ROLE, Role, assign_role
from edgar_extract.numeric_spans import find_numeric_mentions
from edgar_extract.pipeline import EXTRACTION_COLUMNS, extract_regex
from edgar_extract.sections import extract_sections

from .conftest import fixture_bytes, fixture_filings


def roles(sentence: str) -> list[str]:
    return [assign_role(sentence, m).role.value for m in find_numeric_mentions(sentence)]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Google Cloud revenues were $10.1 billion, up 31%.", ["REVENUE", "OTHER_NUMERIC"]),
        ("Net sales increased 2% or $7.8 billion during 2024.", ["OTHER_NUMERIC", "REVENUE"]),
        ("We had $3.5 billion of revenue from Paxlovid.", ["REVENUE"]),
        ("Research and development expenses increased $1.7 billion.", ["RD_EXPENSE"]),
        ("We recognized restructuring charges of $2.8 billion in 2024.", ["ONE_TIME_CHARGE"]),
        ("We recorded a goodwill impairment of $1.2 billion.", ["ONE_TIME_CHARGE"]),
        (
            "Fiscal 2023 includes a reserve for talc matters of $7.0 billion.",
            ["LITIGATION_RESERVE"],
        ),
        ("Firmwide legal expense was $1.1 billion.", ["LITIGATION_RESERVE"]),
        ("We expect revenue of $1.2 to $1.4 billion in 2026.", ["GUIDANCE", "GUIDANCE"]),
        (
            "Net sales are expected to be between $151.0 billion and $155.5 billion.",
            ["GUIDANCE", "GUIDANCE"],
        ),
        ("Revenue for 2022, as restated, was $5.1 billion.", ["RESTATED_PRIOR"]),
        ("Cost of sales was $5.0 billion.", ["OTHER_NUMERIC"]),
        ("Sales and marketing expense was $2.0 billion.", ["OTHER_NUMERIC"]),
        ("Cash and cash equivalents were $8.0 billion.", ["OTHER_NUMERIC"]),
    ],
)
def test_role_rules(sentence: str, expected: list[str]) -> None:
    assert roles(sentence) == expected


@pytest.mark.parametrize(
    "sentence",
    [
        "The five-year facility, as amended and restated in 2021, is $4.62 billion.",
        "The allowance for expected credit losses was $3.2 billion.",
        "This new accounting guidance had no impact; revenue was $5.0 billion.",
    ],
)
def test_role_false_positive_guards(sentence: str) -> None:
    assert "RESTATED_PRIOR" not in roles(sentence) and "GUIDANCE" not in roles(sentence)


def test_nearest_left_cue_wins() -> None:
    sentence = "Research and development expense was $2.0 billion while revenue was $9.0 billion."
    assert roles(sentence) == ["RD_EXPENSE", "REVENUE"]


def test_percentages_only_get_guidance_or_other() -> None:
    assert roles("Revenue grew 12% in the year.") == ["OTHER_NUMERIC"]
    assert roles("We expect revenue growth of 5% next year.") == ["GUIDANCE"]


def test_ner_role_mapping_covers_all_roles() -> None:
    assert set(NER_ROLE) == set(Role)
    assert set(NER_ROLE.values()) == {
        "REVENUE",
        "GUIDANCE",
        "ONE_TIME_CHARGE",
        "RESTATED_PRIOR",
        "OTHER_NUMERIC",
    }


def test_extract_regex_on_recorded_filing_has_exact_lineage() -> None:
    filing = next(f for f in fixture_filings() if f["ticker"] == "GOOGL")
    html = fixture_bytes(filing["url"])
    assert html is not None
    rows = []
    for sec in extract_sections(html).values():
        rows.append(
            {
                "ticker": "GOOGL", "cik": filing["cik"], "fiscal_year": filing["fiscal_year"],
                "form": "10-K", "accession": filing["accession"],
                "filing_date": pd.Timestamp(filing["filing_date"]), "report_date": None,
                "primary_doc_url": filing["url"], "item": sec.item, "status": sec.status.value,
                "text": sec.text, "block_starts": sec.block_starts, "block_kinds": sec.block_kinds,
            }
        )  # fmt: skip
    sections = pd.DataFrame(rows)
    ex = extract_regex(sections)
    assert list(ex.columns) == EXTRACTION_COLUMNS
    assert len(ex) > 100
    texts = dict(zip(sections["item"], sections["text"], strict=True))
    for r in ex.itertuples():
        text = texts[r.section]
        assert text[r.span_start : r.span_end] == r.raw_text
        assert text[r.sentence_start : r.sentence_end] == r.sentence
        assert r.primary_doc_url == filing["url"]
    usd = ex[ex["unit"] == "USD"]
    assert (usd["value_usd"] == usd["value"]).all()
    assert set(ex["extractor"]) == {"regex_v1"}
    # GOOGL's MD&A discusses segment revenues in prose.
    assert (ex["role"] == "REVENUE").sum() >= 10
