"""Reconciliation: one synthetic test per status and mismatch cause, plus the helper rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from edgar_extract.reconcile import (
    Cause,
    Status,
    candidate_fields,
    member_words,
    mention_period,
    reconcile,
)
from edgar_extract.xbrl import load_tag_map

TAG_MAP = load_tag_map()
CIK = 1
ACC_2024 = "0000000001-25-000001"  # FY2024 10-K, filed 2025-02-01
ACC_2025 = "0000000001-26-000001"  # FY2025 10-K, filed 2026-02-01


def fact(field: str, tag: str, value: float, fy: int, acc: str, filed: str) -> dict[str, Any]:
    return {
        "cik": CIK, "tag": tag, "unit": "USD", "value": value,
        "period_start": pd.Timestamp(f"{fy}-01-01"), "period_end": pd.Timestamp(f"{fy}-12-31"),
        "duration_days": 365, "period_type": "annual", "fy": fy, "fp": "FY", "form": "10-K",
        "accession": acc, "filed": pd.Timestamp(filed), "frame": None,
        "canonical_field": field, "tag_priority": 0,
    }  # fmt: skip


CANONICAL = pd.DataFrame(
    [
        fact("Revenues", "us-gaap:Revenues", 10_000e6, 2024, ACC_2024, "2025-02-01"),
        fact("Revenues", "us-gaap:Revenues", 9_000e6, 2023, ACC_2024, "2025-02-01"),
        # FY2024 revenue recast in the FY2025 10-K (e.g. discontinued operations)
        fact("Revenues", "us-gaap:Revenues", 9_400e6, 2024, ACC_2025, "2026-02-01"),
        fact("Revenues", "us-gaap:Revenues", 11_000e6, 2025, ACC_2025, "2026-02-01"),
        fact(
            "RestructuringCharges",
            "us-gaap:RestructuringCharges",
            412e6,
            2024,
            ACC_2024,
            "2025-02-01",
        ),
        fact(
            "ImpairmentCharges",
            "us-gaap:GoodwillImpairmentLoss",
            1_500e6,
            2024,
            ACC_2024,
            "2025-02-01",
        ),
        fact(
            "ResearchAndDevelopmentExpense",
            "us-gaap:ResearchAndDevelopmentExpense",
            2_345e6,
            2024,
            ACC_2024,
            "2025-02-01",
        ),
    ]
)
INLINE = pd.DataFrame(
    [
        {
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "value": 3_210e6, "unit": "USD", "decimals": "-6",
            "period_start": pd.Timestamp("2024-01-01"), "period_end": pd.Timestamp("2024-12-31"),
            "dims": "us-gaap:StatementBusinessSegmentsAxis=abc:CloudServicesMember",
            "is_dimensional": True, "context_id": "c1", "cik": CIK, "accession": ACC_2024,
        }
    ]
)  # fmt: skip


def extraction(
    sentence: str, raw: str, value: float, scale: float, role: str = "REVENUE", **kw: Any
) -> dict[str, Any]:
    start = sentence.index(raw)
    row = {
        "ticker": "ABC", "cik": CIK, "fiscal_year": 2024, "form": "10-K", "accession": ACC_2024,
        "filing_date": pd.Timestamp("2025-02-01"), "section": "7", "sentence": sentence,
        "sentence_start": 0, "sentence_end": len(sentence), "span_start": start,
        "span_end": start + len(raw), "raw_text": raw, "value": value, "unit": "USD",
        "numeric_kind": "level", "unit_scale": scale, "role": role,
    }  # fmt: skip
    row.update(kw)
    return row


def run(*rows: dict[str, Any]) -> pd.DataFrame:
    return reconcile(pd.DataFrame(list(rows)), CANONICAL, INLINE, TAG_MAP)


def one(*rows: dict[str, Any]) -> pd.Series:
    return run(*rows).iloc[0]


# -- statuses -----------------------------------------------------------------------------


def test_match_within_rounding() -> None:
    r = one(extraction("Revenue was $10.0 billion in 2024.", "$10.0 billion", 10e9, 1e9))
    assert r["status"] == Status.MATCH
    assert (r["canonical_field"], r["xbrl_value"], r["xbrl_accession"]) == (
        "Revenues",
        10_000e6,
        ACC_2024,
    )
    assert r["period_source"] == "explicit_year" and pd.isna(r["mismatch_cause"])


def test_default_period_is_filing_year_and_fact_is_as_reported_in_same_filing() -> None:
    # FY2024 was later recast to 9,400, but the FY2024 filing reported 10,000.
    r = one(extraction("Revenue was $10,000 million.", "$10,000 million", 10_000e6, 1e6))
    assert r["status"] == Status.MATCH and r["period_source"] == "default_current"


def test_no_counterpart_cases() -> None:
    rows = run(
        extraction(
            "We expect revenue of $12.0 billion next year.",
            "$12.0 billion",
            12e9,
            1e9,
            role="GUIDANCE",
        ),
        extraction(
            "Revenue grew 12% in 2024.", "12%", 12.0, 1.0, unit="percent", numeric_kind="pct_change"
        ),
        extraction("Fourth quarter revenue was $2.6 billion.", "$2.6 billion", 2.6e9, 1e9),
        extraction("Revenue was $5.0 billion in 2019.", "$5.0 billion", 5e9, 1e9),
    )
    assert (rows["status"] == Status.NO_XBRL_COUNTERPART).all()
    assert rows["cause_detail"].tolist()[:3] == [
        "forward-looking figure",
        "not a dollar amount (percent, pct_change)",
        "sub-annual figure; only annual facts are compared",
    ]
    assert "no XBRL fact" in rows["cause_detail"].iloc[3]


def test_other_numeric_is_not_reconciled() -> None:
    assert run(
        extraction("Cash was $9.0 billion.", "$9.0 billion", 9e9, 1e9, role="OTHER_NUMERIC")
    ).empty


# -- one test per mismatch cause ---------------------------------------------------------


def test_cause_restatement() -> None:
    # The FY2025 10-K states FY2024 revenue as originally reported (10,000) while the
    # same filing's XBRL carries the recast 9,400.
    r = one(
        extraction(
            "Revenue was $10,000 million in 2024.", "$10,000 million", 10_000e6, 1e6,
            fiscal_year=2025, accession=ACC_2025, filing_date=pd.Timestamp("2026-02-01"),
        )
    )  # fmt: skip
    assert r["status"] == Status.MISMATCH and r["mismatch_cause"] == Cause.RESTATEMENT
    assert r["xbrl_value"] == 9_400e6 and "earlier version" in r["cause_detail"]


def test_cause_non_gaap_adjustment() -> None:
    # Adjusted revenue excludes the 412 restructuring item: 10,000 + 412 = 10,412.
    r = one(
        extraction(
            "Adjusted revenue, excluding the restructuring impact, was $10,412 million.",
            "$10,412 million",
            10_412e6,
            1e6,
        )
    )
    assert r["mismatch_cause"] == Cause.NON_GAAP_ADJUSTMENT
    assert "RestructuringCharges" in r["cause_detail"]


def test_non_gaap_needs_a_reconcilable_amount() -> None:
    r = one(extraction("Adjusted revenue was $12,345 million.", "$12,345 million", 12_345e6, 1e6))
    assert r["mismatch_cause"] == Cause.EXTRACTION_ERROR


def test_cause_segment_vs_consolidated() -> None:
    r = one(
        extraction("Cloud Services revenue was $3.2 billion in 2024.", "$3.2 billion", 3.2e9, 1e9)
    )
    assert r["mismatch_cause"] == Cause.SEGMENT_VS_CONSOLIDATED
    assert "CloudServicesMember" in r["cause_detail"]


def test_segment_needs_member_name_or_precise_number() -> None:
    # Same value, but nothing in the sentence names the segment and $3.2 billion is coarse.
    r = one(extraction("Some revenue was $3.2 billion in 2024.", "$3.2 billion", 3.2e9, 1e9))
    assert r["mismatch_cause"] == Cause.EXTRACTION_ERROR
    # A precise number is enough on its own.
    r = one(extraction("Some revenue was $3,210 million in 2024.", "$3,210 million", 3_210e6, 1e6))
    assert r["mismatch_cause"] == Cause.SEGMENT_VS_CONSOLIDATED


@pytest.mark.parametrize(
    ("sentence", "raw", "value", "scale", "detail"),
    [
        # 10,000 million read with a thousands scale
        ("Revenue was $10,000 thousand.", "$10,000 thousand", 10_000e3, 1e3, "10^-3"),
        # one unit off in the last digit
        ("Revenue was $10.1 billion.", "$10.1 billion", 10.1e9, 1e9, "last displayed digit"),
    ],
)
def test_cause_scale_or_rounding(
    sentence: str, raw: str, value: float, scale: float, detail: str
) -> None:
    r = one(extraction(sentence, raw, value, scale))
    assert r["mismatch_cause"] == Cause.SCALE_OR_ROUNDING and detail in r["cause_detail"]


def test_cause_extraction_error_with_period_hint() -> None:
    # The value is FY2023's revenue, but the sentence gives no year: wrong period.
    r = one(extraction("Revenue was $9,000 million.", "$9,000 million", 9_000e6, 1e6))
    assert r["mismatch_cause"] == Cause.EXTRACTION_ERROR
    assert "FY2023" in r["cause_detail"]


def test_cause_order_restatement_beats_rounding() -> None:
    # 9,400 (recast) differs from 10,000 by far more than the rounding, so restatement wins.
    r = one(
        extraction(
            "Revenue was $9,400 million in 2024.",
            "$9,400 million",
            9_400e6,
            1e6,
        )
    )
    assert r["mismatch_cause"] == Cause.RESTATEMENT and "later version" in r["cause_detail"]


# -- helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sentence", "raw", "expected"),
    [
        (
            "Revenue was $5.0 billion in 2023, up from 2022.",
            "$5.0 billion",
            (2023, "explicit_year"),
        ),
        ("In fiscal 2022, revenue was $5.0 billion.", "$5.0 billion", (2022, "explicit_year")),
        ("Revenue was $5.0 billion.", "$5.0 billion", (2024, "default_current")),
        (
            "The nonrecurrence of a $1.0 billion charge helped.",
            "$1.0 billion",
            (2023, "prior_year_wording"),
        ),
    ],
)
def test_mention_period(sentence: str, raw: str, expected: tuple[int, str]) -> None:
    s = sentence.index(raw)
    p = mention_period(sentence, s, s + len(raw), 2024, [(s, s + len(raw))])
    assert (p.fiscal_year, p.source) == expected


def test_respectively_lists() -> None:
    sentence = "R&D was $1.6 billion and $1.5 billion in 2024 and 2023, respectively."
    spans = [
        (sentence.index("$1.6"), sentence.index("$1.6") + 12),
        (sentence.index("$1.5"), sentence.index("$1.5") + 12),
    ]
    years = [mention_period(sentence, a, b, 2024, spans).fiscal_year for a, b in spans]
    assert years == [2024, 2023]


def test_year_of_next_amount_is_not_borrowed() -> None:
    sentence = "Revenue was $5.0 billion, compared to $4.0 billion in 2022."
    s = sentence.index("$5.0")
    assert mention_period(sentence, s, s + 12, 2024, []).fiscal_year == 2024


def test_field_choice_by_nearest_keyword() -> None:
    s = "We recorded impairment charges of $1.5 billion and restructuring charges of $412 million."
    assert candidate_fields("ONE_TIME_CHARGE", s, s.index("$1.5"))[0] == "ImpairmentCharges"
    assert candidate_fields("ONE_TIME_CHARGE", s, s.index("$412"))[0] == "RestructuringCharges"
    s2 = "The litigation reserve was $2.0 billion."
    assert candidate_fields("LITIGATION_RESERVE", s2, s2.index("$"))[0] == "LitigationReserve"
    s3 = "Revenue, as restated, was $5.0 billion."
    assert candidate_fields("RESTATED_PRIOR", s3, s3.index("$")) == ["Revenues"]


def test_member_words() -> None:
    dims = (
        "srt:ProductOrServiceAxis=jnj:OrthopaedicsMember;"
        "us-gaap:StatementBusinessSegmentsAxis=jnj:MedTechMember"
    )
    assert member_words(dims) == {"orthopaedics", "med", "tech"}
    assert (
        member_words("us-gaap:StatementBusinessSegmentsAxis=us-gaap:AllOtherSegmentsMember")
        == set()
    )


def test_period_hint_ignores_years_outside_the_comparative_window() -> None:
    old = CANONICAL.copy()
    ancient = fact(
        "Revenues", "us-gaap:Revenues", 7_777e6, 2013, "0000000001-14-000001", "2014-02-01"
    )
    canonical = pd.concat([old, pd.DataFrame([ancient])], ignore_index=True)
    row = extraction("Revenue was $7,777 million.", "$7,777 million", 7_777e6, 1e6)
    r = reconcile(pd.DataFrame([row]), canonical, INLINE, TAG_MAP).iloc[0]
    assert r["mismatch_cause"] == Cause.EXTRACTION_ERROR and r["cause_detail"] == ""


# -- change amounts --------------------------------------------------------------------------


def change(sentence: str, raw: str, value: float, scale: float, **kw: Any) -> dict[str, Any]:
    return extraction(sentence, raw, value, scale, numeric_kind="delta", **kw)


def test_change_matches_difference_of_two_years() -> None:
    # FY2024 10,000 - FY2023 9,000 = 1,000, both as reported in the FY2024 10-K
    r = one(change("Revenue increased $1.0 billion in 2024.", "$1.0 billion", 1e9, 1e9))
    assert (r["status"], r["comparison"], r["xbrl_value"]) == (Status.MATCH, "change", 1_000e6)
    assert "FY2024 10,000,000,000 - FY2023 9,000,000,000" in r["cause_detail"]


def test_change_base_year_wording() -> None:
    r = one(change("Revenue increased $1.0 billion compared to 2023.", "$1.0 billion", 1e9, 1e9))
    assert (r["fiscal_year_ref"], r["period_source"], r["status"]) == (
        2024,
        "base_year",
        Status.MATCH,
    )


def test_change_restatement_cause() -> None:
    # In the FY2025 10-K, FY2025 = 11,000 and FY2024 (recast) = 9,400: change 1,600.
    # The text quotes 1,000 = 11,000 - 10,000, i.e. against FY2024 as originally reported.
    r = one(
        change(
            "Revenue increased $1,000 million in 2025.", "$1,000 million", 1_000e6, 1e6,
            fiscal_year=2025, accession=ACC_2025, filing_date=pd.Timestamp("2026-02-01"),
        )
    )  # fmt: skip
    assert r["status"] == Status.MISMATCH and r["mismatch_cause"] == Cause.RESTATEMENT
    assert r["xbrl_value"] == 1_600e6


def test_level_read_as_change_is_an_extraction_error() -> None:
    r = one(change("Revenue increased to $10,000 million.", "$10,000 million", 10_000e6, 1e6))
    assert r["mismatch_cause"] == Cause.EXTRACTION_ERROR
    assert "equals the FY t level" in r["cause_detail"]


def test_change_rows_are_not_ingested_into_the_pit_db(tmp_path: Path) -> None:
    from edgar_extract.pit_db import PointInTimeDB, build_pit_db

    rec = run(
        extraction("Revenue was $10,000 million.", "$10,000 million", 10_000e6, 1e6),
        change("Revenue increased $1.0 billion in 2024.", "$1.0 billion", 1e9, 1e9),
    ).assign(
        extractor="regex_v1", role_cue="revenue", form="10-K",
        primary_doc_url="https://www.sec.gov/Archives/x.htm",
    )  # fmt: skip
    companies = pd.DataFrame([{"cik": CIK, "ticker": "ABC", "name": "ABC"}])
    stats = build_pit_db(tmp_path / "p.duckdb", CANONICAL.iloc[:0], companies, [rec])
    assert stats["new_text_rows"] == 1
    with PointInTimeDB(tmp_path / "p.duckdb", read_only=True) as pit:
        rows = pit.as_of("2030-01-01", ["ABC"], ["Revenues"], sources=("text",))
    assert rows["value_usd"].tolist() == [10_000e6]


def test_segment_change_named_in_sentence() -> None:
    inline = pd.concat(
        [
            INLINE,
            INLINE.assign(
                value=2_000e6,
                period_start=pd.Timestamp("2023-01-01"),
                period_end=pd.Timestamp("2023-12-31"),
            ),
        ],
        ignore_index=True,
    )
    row = change(
        "Cloud Services revenue increased $1.2 billion in 2024.", "$1.2 billion", 1.2e9, 1e9
    )
    r = reconcile(pd.DataFrame([row]), CANONICAL, inline, TAG_MAP).iloc[0]
    assert r["mismatch_cause"] == Cause.SEGMENT_VS_CONSOLIDATED  # 3,210 - 2,000 = 1,210
    assert "member named in sentence" in r["cause_detail"]


def test_segment_named_only_in_heading() -> None:
    row = extraction(
        "Net sales were $3.2 billion in 2024.", "$3.2 billion", 3.2e9, 1e9, role="REVENUE",
        context_heading="Cloud Services: | Fiscal Year 2024 Compared to Fiscal Year 2023:",
    )  # fmt: skip
    r = one(row)
    assert r["mismatch_cause"] == Cause.SEGMENT_VS_CONSOLIDATED
    assert "member named in heading" in r["cause_detail"]


def test_heading_needs_a_reasonably_precise_number() -> None:
    inline = INLINE.assign(value=0.3e9)
    row = extraction(
        "Net sales were $0.3 billion in 2024.", "$0.3 billion", 0.3e9, 1e9,
        context_heading="Cloud Services:",
    )  # fmt: skip
    r = reconcile(pd.DataFrame([row]), CANONICAL, inline, TAG_MAP).iloc[0]
    assert r["mismatch_cause"] == Cause.EXTRACTION_ERROR  # +/-17%: too coarse for a heading


def test_generic_words_do_not_corroborate() -> None:
    from edgar_extract.reconcile import member_words

    words = member_words("srt:ProductOrServiceAxis=xom:IncomeFromEquityAffiliatesMember")
    assert words == {"equity", "affiliates"}
    assert not (words & {"cash", "flow", "from", "operating", "activities"})
