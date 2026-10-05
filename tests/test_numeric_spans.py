from __future__ import annotations

import pytest

from edgar_extract.numeric_spans import (
    NumericMention,
    find_numeric_mentions,
    is_prose,
    iter_sentences,
    split_sentences,
)


def one(sentence: str) -> NumericMention:
    (m,) = find_numeric_mentions(sentence)
    return m


@pytest.mark.parametrize(
    ("sentence", "raw", "value", "scale"),
    [
        ("Revenue was $1.2 billion.", "$1.2 billion", 1.2e9, 1e9),
        ("Revenue was $845 million.", "$845 million", 845e6, 1e6),
        ("Revenue was $1.07 billion.", "$1.07 billion", 1_070_000_000.0, 1e9),
        ("Net sales were $ 391,035 for the year.", "$ 391,035", 391_035.0, 1.0),
        ("A charge of $2.0B was recorded.", "$2.0B", 2e9, 1e9),
        ("Proceeds were US$450 mn in total.", "US$450 mn", 450e6, 1e6),
        ("Costs were $12.5 thousand in total.", "$12.5 thousand", 12_500.0, 1e3),
    ],
)
def test_currency_amounts_normalized_to_usd(
    sentence: str, raw: str, value: float, scale: float
) -> None:
    m = one(sentence)
    assert (m.raw_text, m.value, m.value_usd, m.unit, m.unit_scale) == (
        raw,
        value,
        value,
        "USD",
        scale,
    )
    assert sentence[m.start : m.end] == raw
    assert m.scale_explicit is (scale != 1.0)


def test_parenthesized_negatives() -> None:
    m = one("Operating loss was $(3.1) million in the quarter.")
    assert m.value == -3.1e6 and m.is_negative
    m = one("The adjustment was (3.1) in the prior period.")
    assert (m.value, m.unit, m.numeric_kind) == (-3.1, "none", "number")
    m = one("Revenue was down (2)% year over year.")
    assert (m.value, m.unit, m.numeric_kind) == (-2.0, "percent", "pct_change")


def test_opening_parenthesis_is_not_a_negative() -> None:
    m = one("Charges ($5 million in total) were recorded.")
    assert m.value == 5e6 and not m.is_negative


@pytest.mark.parametrize(
    ("sentence", "value", "unit"),
    [
        ("Margin was 12.4% for the year.", 12.4, "percent"),
        ("Margin was 12 percent for the year.", 12.0, "percent"),
        ("Margin declined 120 basis points.", 120.0, "bp"),
        ("Margin declined 2.5 percentage points.", 2.5, "pp"),
    ],
)
def test_percentages_and_points(sentence: str, value: float, unit: str) -> None:
    m = one(sentence)
    assert (m.value, m.unit, m.value_usd) == (value, unit, None)


def test_per_share() -> None:
    m = one("Diluted EPS was $6.08 per diluted share for 2024.")
    assert (m.raw_text, m.unit, m.numeric_kind, m.value_usd) == (
        "$6.08 per diluted share",
        "USD/share",
        "per_share",
        6.08,
    )


def test_increased_x_pct_to_y_level() -> None:
    pct, level, prior = find_numeric_mentions(
        "Revenue increased 12.4% to $1.2 billion, from $1.07 billion in fiscal 2023."
    )
    assert (pct.numeric_kind, pct.value) == ("pct_change", 12.4)
    assert (level.numeric_kind, level.value) == ("level", 1.2e9)
    assert (prior.numeric_kind, prior.value) == ("level", 1.07e9)


def test_increased_by_dollars_is_a_delta() -> None:
    delta, level = find_numeric_mentions("Other expenses increased $142 million, to $941 million.")
    assert delta.numeric_kind == "delta" and level.numeric_kind == "level"
    pct, amount = find_numeric_mentions("Total net sales increased 2% or $7.8 billion during 2024.")
    assert pct.numeric_kind == "pct_change" and amount.numeric_kind == "delta"


@pytest.mark.parametrize(
    ("sentence", "low", "high", "unit"),
    [
        ("We expect revenue of $1.2 to $1.4 billion next year.", 1.2e9, 1.4e9, "USD"),
        ("We expect revenue of $1.2 to 1.4 billion next year.", 1.2e9, 1.4e9, "USD"),
        ("Charges of $5-$6 billion are expected.", 5e9, 6e9, "USD"),
        (
            "Sales are expected to be between $151.0 billion and $155.5 billion.",
            151e9,
            155.5e9,
            "USD",
        ),
        ("Growth of 12 to 14% is expected.", 12.0, 14.0, "percent"),
        ("Margins between 45% and 46% are expected.", 45.0, 46.0, "percent"),
    ],
)
def test_ranges(sentence: str, low: float, high: float, unit: str) -> None:
    lo, hi = find_numeric_mentions(sentence)
    assert (lo.value, hi.value, lo.unit, hi.unit) == (low, high, unit, unit)
    assert (lo.range_part, hi.range_part) == ("low", "high")
    assert lo.range_id == hi.range_id is not None
    assert sentence[lo.start : lo.end] == lo.raw_text and sentence[hi.start : hi.end] == hi.raw_text


def test_list_with_and_is_not_a_range() -> None:
    a, b = find_numeric_mentions("We recorded $5 million and $6 million of charges.")
    assert a.range_part is None and b.range_part is None


@pytest.mark.parametrize(
    "sentence",
    [
        "See Note 7 and Item 1A of our 2024 Form 10-K.",
        "ASU 2016-13 was adopted on January 1, 2020.",
        "Section 13(r) of the Exchange Act and Rule 10b5-1 apply.",
        "We operate 3 segments in 12 countries.",
    ],
)
def test_non_financial_numbers_ignored(sentence: str) -> None:
    assert find_numeric_mentions(sentence) == []


def test_counts_and_foreign_currency() -> None:
    a, b = find_numeric_mentions("We had 164,000 employees and repurchased 1.2 billion shares.")
    assert (a.value, a.unit, b.value, b.unit) == (164_000.0, "count", 1.2e9, "count")
    m = one("We issued €13.25 billion of euro-denominated notes.")
    assert (m.unit, m.value, m.value_usd) == ("other_currency", 13.25e9, None)


def test_is_prose() -> None:
    assert is_prose("Revenue increased 12% to $1.2 billion due to strong demand in all regions.")
    assert not is_prose("Total net sales $ 416,161 6 % $ 391,035 2 % $ 383,285")
    assert not is_prose("Income before income tax expense 28,430 19,793 27,840")
    assert not is_prose("Overview")


def test_sentence_offsets_with_repeated_sentences() -> None:
    text = "Revenue rose. Revenue rose. Costs fell by $5 million in the U.S. market."
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == [
        "Revenue rose.",
        "Revenue rose.",
        "Costs fell by $5 million in the U.S. market.",
    ]
    assert spans[1][0] == 14  # the second copy, not the first


def test_iter_sentences_respects_blocks_and_skips_tables() -> None:
    blocks = [
        "Revenue increased 12% to $1.2 billion in the year. Costs were flat for the year.",
        "Net sales $ 416,161 $ 391,035",
        "Margins improved by 50 basis points over the prior year period.",
    ]
    text = "\n\n".join(blocks)
    starts = [0, len(blocks[0]) + 2, len(blocks[0]) + len(blocks[1]) + 4]
    sents = list(iter_sentences(text, starts, ["p", "table_row", "p"]))
    assert [s.text for s in sents] == [
        "Revenue increased 12% to $1.2 billion in the year.",
        "Costs were flat for the year.",
        "Margins improved by 50 basis points over the prior year period.",
    ]
    for s in sents:
        assert text[s.start : s.end] == s.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("North America:", True),
        ("Fiscal Year 2023 Compared to Fiscal Year 2022:", True),
        ("Results of Operations", True),
        ("Total income from operations $ 74,842 $ 84,293", False),
        ("n the sale of securities of $0.2 billion in 2025", False),
        ("(1) Inclusive of VELCADE which was previously disclosed", False),
        ("Revenue increased in all regions.", False),
        ("Segment results for the year by business, region, product line and channel", False),
    ],
)
def test_is_heading_like(text: str, expected: bool) -> None:
    from edgar_extract.numeric_spans import is_heading_like

    assert is_heading_like(text) is expected


def test_sentences_carry_recent_headings() -> None:
    from edgar_extract.numeric_spans import iter_sentences

    blocks = [
        "North America:",
        "Net sales $ 20,126 $ 20,340",
        "Fiscal Year 2023 Compared to Fiscal Year 2022:",
        "Net sales decreased 1.0% to $20.1 billion in 2023 compared to $20.3 billion in 2022.",
    ]
    text = "\n\n".join(blocks)
    starts, pos = [], 0
    for b in blocks:
        starts.append(pos)
        pos += len(b) + 2
    sents = list(iter_sentences(text, starts, ["p", "table_row", "p", "p"]))
    sent = sents[-1]  # the heading line is also prose-like, but carries no numbers
    assert sent.text.startswith("Net sales decreased")
    assert sent.context == "North America: | Fiscal Year 2023 Compared to Fiscal Year 2022:"
