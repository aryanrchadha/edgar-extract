"""Sentence segmentation and detection of numeric entities in section prose.

Scope (what counts as a numeric entity), chosen to cover financial quantities and skip
noise such as years, note numbers and dates:

* currency amounts: ``$1.2 billion``, ``$845 million``, ``$ 391,035``, ``US$5bn``,
  ``$(3.1) million``
* per-share amounts: ``$6.08 per diluted share``
* percentages: ``12.4%``, ``(2)%``, ``12 percent``; percentage points and basis points
* scaled counts: ``1.2 billion shares``, ``164,000 employees`` (comma-grouped, not a year)
* parenthesized decimals in prose: ``(3.1)`` (negative, unit unknown)
* ranges: ``$1.2 to $1.4 billion``, ``between 12% and 14%``, ``$5-$6 billion``; both ends
  are emitted with ``range_part`` low/high and share the scale written on either end

Each mention carries a ``numeric_kind``: ``level``, ``delta`` (``increased $X``),
``pct``, ``pct_change`` (``increased 12%``), ``per_share``, ``count``, ``points``, or
``number``. In ``increased 12% to $1.2 billion`` the 12% is a ``pct_change`` and the $1.2
billion a ``level``.

Values are normalized to base units with the scale kept explicit: ``$1.2 billion`` becomes
``value=1.2e9, unit="USD", unit_scale=1e9, scale_explicit=True``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal

with warnings.catch_warnings():
    # pysbd 0.3.4 compiles regexes with invalid escape sequences; harmless but noisy.
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

Unit = Literal["USD", "USD/share", "other_currency", "percent", "pp", "bp", "count", "none"]
NumericKind = Literal[
    "level", "delta", "pct", "pct_change", "per_share", "count", "points", "number"
]

SCALES: dict[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_SCALE_WORD = r"thousand|million|billion|trillion"
_SCALE_ABBR = r"bn|mn|mm|tn|[kmb]"
_CUR = r"(?:US\s?\$|U\.S\.\s?\$|\$)"

# Money: optional sign/parens, currency symbol, number, optional scale.
_MONEY = (
    rf"(?P<neg>-|\()?\s*{_CUR}\s*(?P<neg2>-|\()?\s*(?P<num>{_NUM})\s*(?P<close>\))?"
    rf"(?:\s*(?P<scale>{_SCALE_WORD})\b|(?P<abbr>{_SCALE_ABBR})\b)?"
)
_PER_SHARE = r"\s*per\s+(?:basic\s+(?:and|or)\s+diluted\s+|diluted\s+|basic\s+)?(?:common\s+)?share"
_PCT = rf"(?P<pneg>\()?(?P<pnum>-?(?:{_NUM}))\s*(?P<pclose>\))?\s*(?:%|percent\b|per\s+cent\b)"
_POINTS = rf"(?P<qnum>{_NUM})\s*(?P<qunit>percentage\s+points?|basis\s+points?|bps)\b"
_COUNT = (
    rf"(?<![\w$.,])(?P<cnum>{_NUM})\s*(?P<cscale>{_SCALE_WORD})\b(?!\s*(?:%|percent))"
    r"|(?<![\w$.,])(?P<cgrp>\d{1,3}(?:,\d{3})+)(?![\d.,]*\s*(?:%|percent))(?![\w])"
)
_PAREN_DEC = r"(?<![\w$])\((?P<dnum>\d+\.\d+)\)(?!\s*%)"
# Non-dollar currencies are detected so they are not mistaken for counts, but never
# converted: there is no FX rate in this pipeline.
_FX = (
    rf"(?:€|£|¥|EUR\s?|GBP\s?|JPY\s?|CHF\s?)\s*(?P<fnum>{_NUM})(?:\s*(?P<fscale>{_SCALE_WORD})\b)?"
)

MENTION_RE = re.compile(
    rf"(?P<money>{_MONEY}(?P<pershare>{_PER_SHARE})?)"
    rf"|(?P<pct>{_PCT})"
    rf"|(?P<points>{_POINTS})"
    rf"|(?P<fx>{_FX})"
    rf"|(?P<count>{_COUNT})"
    rf"|(?P<paren>{_PAREN_DEC})",
    re.I,
)

# "$1.2 to $1.4 billion", "12% to 14%", "between $5 and $6 billion", "$5-$6 billion"
_RANGE_JOIN_RE = re.compile(r"^\s*(?:to|-|through|and)\s*$", re.I)
_RANGE_INTRO_RE = re.compile(r"(?:between|range\s+of|ranging\s+from|from)\s*$", re.I)
# "12 to 14%": the first number has no unit of its own.
_PCT_RANGE_RE = re.compile(
    rf"(?<![\w$.,])(?P<a>{_NUM})\s*(?:to|-)\s*(?P<b>{_NUM})\s*(?:%|percent\b)", re.I
)
_MONEY_RANGE_RE = re.compile(
    rf"{_CUR}\s*(?P<a>{_NUM})\s*(?:to|-|and)\s*{_CUR}?\s*(?P<b>{_NUM})\s*(?P<scale>{_SCALE_WORD})\b",
    re.I,
)

_CHANGE_VERB_RE = re.compile(
    r"(?:increase[sd]?|decrease[sd]?|grew|grow(?:th|s)?|declined?|declines|rose|rise[sn]?|"
    r"fell|fall(?:s|en)?|up|down|higher|lower|improved?|improvement|reduction|reduced|"
    r"change[sd]?|gain(?:ed)?|drop(?:ped)?)\s+(?:of\s+|by\s+)?(?:approximately\s+|about\s+|nearly\s+|almost\s+)?$",
    re.I,
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

# Section text blocks with fewer alphabetic words, or a lower letter share, are tables.
PROSE_MIN_WORDS = 5
PROSE_MIN_ALPHA = 0.5


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int  # offset in the section text
    end: int
    block_kind: str
    context: str = ""  # recent headings above the sentence, " | "-separated


@dataclass(frozen=True)
class NumericMention:
    start: int  # offsets within the sentence
    end: int
    raw_text: str
    value: float  # in base units (USD, percent points, count)
    unit: Unit
    unit_scale: float
    scale_explicit: bool
    is_negative: bool
    numeric_kind: NumericKind
    range_id: int | None = None
    range_part: Literal["low", "high"] | None = None

    @property
    def value_usd(self) -> float | None:
        return self.value if self.unit in ("USD", "USD/share") else None


# -- sentences -------------------------------------------------------------------------


_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])\(?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?(?![\w])")
_SENTENCE_END_RE = re.compile(r"[.:;?!][\"')\]]*$")


def is_prose(text: str) -> bool:
    """True for running text; False for table rows and table-like lines.

    A line is prose when it has at least 5 alphabetic words, letters make up at least half of
    it, and it does not look like a row of figures (2+ numbers and no sentence-ending
    punctuation, e.g. "Income before income tax expense 28,430 19,793 27,840").
    """
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < PROSE_MIN_WORDS:
        return False
    if sum(len(w) for w in words) / max(len(text), 1) < PROSE_MIN_ALPHA:
        return False
    numbers = len(_NUMBER_TOKEN_RE.findall(text))
    return not (numbers >= 2 and not _SENTENCE_END_RE.search(text.strip()))


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence (start, end) offsets within ``text``, located left to right.

    pysbd's own char spans can point at an earlier duplicate of a repeated sentence, so
    each segment is found by searching forward from the previous one.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for seg in _SEGMENTER.segment(text):
        piece = seg.strip()
        if not piece:
            continue
        pos = text.find(piece, cursor)
        if pos < 0:  # pysbd normalized something; fall back to the cursor
            pos = cursor
        spans.append((pos, pos + len(piece)))
        cursor = pos + len(piece)
    return spans


HEADING_MAX_CHARS = 80
HEADING_MAX_WORDS = 10
CONTEXT_HEADINGS = 4
CONTEXT_LOOKBACK_BLOCKS = 40


_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def is_heading_like(text: str) -> bool:
    """A short label line such as "North America:" or "Results of Operations".

    It starts with a capital letter, has at most 10 words, no figures other than years, and
    no sentence-ending period. That excludes table rows ("Total income $ 74,842"), prose
    fragments that start mid-word, and footnotes ("(1) Inclusive of ...").
    """
    stripped = text.strip()
    if not stripped or len(stripped) > HEADING_MAX_CHARS or stripped.endswith("."):
        return False
    if not stripped[0].isupper() or len(stripped.split()) > HEADING_MAX_WORDS:
        return False
    return not re.search(r"\d", _YEAR_TOKEN_RE.sub("", stripped))


def heading_context(
    text: str, block_starts: list[int], index: int, *, limit: int = CONTEXT_HEADINGS
) -> str:
    """Up to ``limit`` heading-like blocks above block ``index``, nearest last.

    Segment discussions often name the segment only in a heading several blocks up
    ("North America:" above a table and a "Fiscal Year 2023 Compared to 2022:" sub-head).
    """
    found: list[str] = []
    for k in range(index - 1, max(-1, index - 1 - CONTEXT_LOOKBACK_BLOCKS), -1):
        end = block_starts[k + 1] - 2 if k + 1 < len(block_starts) else len(text)
        block = text[block_starts[k] : end]
        if is_heading_like(block):
            found.append(block.strip())
            if len(found) == limit:
                break
    return " | ".join(reversed(found))


def iter_sentences(
    text: str, block_starts: list[int], block_kinds: list[str], *, prose_only: bool = True
) -> Iterator[Sentence]:
    """Sentences of a section, never crossing paragraph (block) boundaries."""
    for i, start in enumerate(block_starts):
        end = block_starts[i + 1] - 2 if i + 1 < len(block_starts) else len(text)
        block = text[start:end]
        if prose_only and not is_prose(block):
            continue
        context = heading_context(text, block_starts, i)
        for s, e in split_sentences(block):
            yield Sentence(block[s:e], start + s, start + e, block_kinds[i], context)


# -- numeric mentions ------------------------------------------------------------------


def _to_float(num: str, scale: float = 1.0) -> float:
    """Exact decimal scaling, so "$1.07 billion" is 1070000000.0, not ...0000001."""
    return float(Decimal(num.replace(",", "")) * Decimal(int(scale)))


def _scale_of(word: str | None) -> tuple[float, bool]:
    if not word:
        return 1.0, False
    return SCALES[word.lower()], True


def _money(m: re.Match[str]) -> NumericMention | None:
    num = m.group("num")
    raw_scale = m.group("scale") or m.group("abbr")
    scale, explicit = _scale_of(raw_scale)
    neg = bool(m.group("neg") or m.group("neg2"))
    if (m.group("neg") == "(" or m.group("neg2") == "(") and not m.group("close"):
        neg = False  # "($5 million" is an opening parenthesis, not a negative
    per_share = bool(m.group("pershare"))
    value = _to_float(num, scale) * (-1 if neg else 1)
    group = m.group("money")
    start = m.start("money") + (len(group) - len(group.lstrip()))
    end = m.end("money") - (len(group) - len(group.rstrip()))
    return NumericMention(
        start=start,
        end=end,
        raw_text=group.strip(),
        value=value,
        unit="USD/share" if per_share else "USD",
        unit_scale=scale,
        scale_explicit=explicit,
        is_negative=neg,
        numeric_kind="per_share" if per_share else "level",
    )


def _mention(m: re.Match[str]) -> NumericMention | None:
    if m.group("money"):
        return _money(m)
    if m.group("pct"):
        num = m.group("pnum")
        neg = num.startswith("-") or bool(m.group("pneg") and m.group("pclose"))
        value = abs(_to_float(num.lstrip("-"))) * (-1 if neg else 1)
        return NumericMention(
            m.start("pct"), m.end("pct"), m.group("pct"), value, "percent", 1.0, False, neg, "pct"
        )
    if m.group("points"):
        unit: Unit = "bp" if m.group("qunit").lower().startswith(("basis", "bps")) else "pp"
        return NumericMention(
            m.start("points"), m.end("points"), m.group("points"),
            _to_float(m.group("qnum")), unit, 1.0, False, False, "points",
        )  # fmt: skip
    if m.group("fx"):
        scale, explicit = _scale_of(m.group("fscale"))
        return NumericMention(
            m.start("fx"), m.end("fx"), m.group("fx"),
            _to_float(m.group("fnum"), scale), "other_currency", scale, explicit, False, "level",
        )  # fmt: skip
    if m.group("count"):
        if m.group("cnum"):
            scale, _ = _scale_of(m.group("cscale"))
            value = _to_float(m.group("cnum"), scale)
            explicit = True
        else:
            grp = m.group("cgrp")
            if _YEAR_RE.match(grp.replace(",", "")):
                return None
            value, scale, explicit = _to_float(grp), 1.0, False
        return NumericMention(
            m.start("count"), m.end("count"), m.group("count"),
            value, "count", scale, explicit, False, "count",
        )  # fmt: skip
    if m.group("paren"):
        return NumericMention(
            m.start("paren"), m.end("paren"), m.group("paren"),
            -_to_float(m.group("dnum")), "none", 1.0, False, True, "number",
        )  # fmt: skip
    return None


def _special_ranges(sentence: str) -> list[NumericMention]:
    """Ranges whose first number has no unit or scale of its own."""
    out: list[NumericMention] = []
    for m in _PCT_RANGE_RE.finditer(sentence):
        bounds = {"low": (m.start("a"), m.end("a")), "high": (m.start("b"), m.end())}
        for part, grp in (("low", "a"), ("high", "b")):
            start, end = bounds[part]
            out.append(
                NumericMention(
                    start, end, sentence[start:end], _to_float(m.group(grp)),
                    "percent", 1.0, False, False, "pct", range_part=part,  # type: ignore[arg-type]
                )
            )  # fmt: skip
    for m in _MONEY_RANGE_RE.finditer(sentence):
        scale, _ = _scale_of(m.group("scale"))
        # The high end starts at its own "$" when it has one ("$5-$6 billion").
        dollar = sentence.rfind("$", m.end("a"), m.start("b"))
        bounds = {
            "low": (m.start(), m.end("a")),
            "high": (dollar if dollar >= 0 else m.start("b"), m.end()),
        }
        for part, grp in (("low", "a"), ("high", "b")):
            start, end = bounds[part]
            out.append(
                NumericMention(
                    start, end, sentence[start:end].strip(), _to_float(m.group(grp), scale),
                    "USD", scale, True, False, "level", range_part=part,  # type: ignore[arg-type]
                )
            )  # fmt: skip
    return out


def find_numeric_mentions(sentence: str) -> list[NumericMention]:
    """All numeric mentions in a sentence, left to right, with ranges and kinds resolved."""
    special = _special_ranges(sentence)
    taken = [(s.start, s.end) for s in special]
    mentions = list(special)
    for m in MENTION_RE.finditer(sentence):
        mention = _mention(m)
        if mention is None:
            continue
        if any(not (mention.end <= a or mention.start >= b) for a, b in taken):
            continue
        mentions.append(mention)
    mentions.sort(key=lambda x: x.start)
    mentions = _link_ranges(sentence, mentions)
    return [_classify_kind(sentence, i, mentions) for i in range(len(mentions))]


def _link_ranges(sentence: str, mentions: list[NumericMention]) -> list[NumericMention]:
    out = list(mentions)
    range_id = 0
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if a.range_part == "low" and b.range_part == "high":
            out[i], out[i + 1] = replace(a, range_id=range_id), replace(b, range_id=range_id)
            range_id += 1
            continue
        if a.range_part or b.range_part or a.unit != b.unit:
            continue
        joiner = sentence[a.end : b.start]
        if not _RANGE_JOIN_RE.match(joiner):
            continue
        if joiner.strip().lower() == "and" and not _RANGE_INTRO_RE.search(sentence[: a.start]):
            continue  # "$5 million and $6 million" is a list, not a range
        if not a.scale_explicit and b.scale_explicit:
            scaled = float(Decimal(str(a.value)) * Decimal(int(b.unit_scale)))
            a = replace(a, value=scaled, unit_scale=b.unit_scale, scale_explicit=True)
        out[i] = replace(a, range_id=range_id, range_part="low")
        out[i + 1] = replace(b, range_id=range_id, range_part="high")
        range_id += 1
    return out


_OR_AFTER_PCT_RE = re.compile(r"^\s*,?\s*or\s+$", re.I)


def _classify_kind(sentence: str, i: int, mentions: list[NumericMention]) -> NumericMention:
    m = mentions[i]
    left = sentence[max(0, m.start - 60) : m.start]
    changed = bool(_CHANGE_VERB_RE.search(left))
    if m.numeric_kind == "pct" and changed:
        return replace(m, numeric_kind="pct_change")
    if m.numeric_kind == "level" and m.range_part is None:
        # "increased 2% or $7.8 billion": the amount restates the change in dollars.
        if i > 0 and mentions[i - 1].unit == "percent":
            prev = mentions[i - 1]
            if _OR_AFTER_PCT_RE.match(sentence[prev.end : m.start]) and _CHANGE_VERB_RE.search(
                sentence[max(0, prev.start - 60) : prev.start]
            ):
                return replace(m, numeric_kind="delta")
        if changed:
            return replace(m, numeric_kind="delta")
    return m
