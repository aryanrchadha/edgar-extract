"""Isolate Item 1A (Risk Factors) and Item 7 (MD&A) from a 10-K's HTML.

Rules, in the order they are applied:

1. **Render** the HTML into blocks (paragraphs and table rows) with physical page numbers
   (:mod:`edgar_extract.htmltext`).
2. **Strip boilerplate**: running page headers and footers (``2024 FORM 10-K 81``,
   ``Table of Contents``, MSFT-style ``Item 7`` page headers) and bare page numbers that sit
   at the top or bottom of a page and repeat across pages.
3. **Find Item headings**: a short block that starts with ``Item <n>[A-D]`` (optionally
   after ``Part II``), whatever the punctuation (``Item 7.``, ``ITEM 7 -``, ``Item 7:``,
   ``ITEM 1A.RISK FACTORS``). A block whose text continues in lower case
   (``Item 7 of this report...``) is a cross-reference, not a heading.
4. **Pick the span**: for each heading of the target item, the span runs to the next heading
   of a *later* item (``Item 7A``/``8`` for Item 7, ``Item 1B``/``1C``/``2`` for Item 1A).
   The longest such span wins. This discards table-of-contents hits, which end at the next
   TOC line.
5. **Short spans are stubs**, and the stub text decides the outcome:
   * incorporation by reference to the annual report / Exhibit 13 gives
     ``section_missing_by_reference``. Nothing is guessed.
   * a page reference into this same document (``appears on pages 52-167``, or a
     "Form 10-K Cross-Reference Index" row ``Item 7 ... 8-24``) gives text resolved from
     those pages via the printed page numbers (``method="page_reference"``).
   * otherwise ``section_not_found``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from edgar_extract.htmltext import Block, html_to_blocks

TARGET_ITEMS = ("1A", "7")

# Spans whose body is shorter than this are treated as stubs and inspected for references.
STUB_MAX_CHARS = 2000
# A heading line longer than this is prose that happens to start with "Item".
HEADING_MAX_CHARS = 300
# Running headers/footers are short.
BOILERPLATE_MAX_CHARS = 80
BOILERPLATE_MIN_REPEATS = 4

_HEADING_RE = re.compile(
    r"^(?:part\s+[iv]{1,3}\s*[,.:\-]?\s*)?items?\s*(\d{1,2})\s*([a-d])?(?![a-z0-9])\s*[.:\-]?\s*(.*)$",
    re.I | re.S,
)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^(?:page\s*)?-?\s*(?:\d{1,3}|[ivxlc]{1,7})\s*-?$", re.I)
_EDGE_NUMBER_RE = re.compile(r"^(\d{1,3})(?![\d,.])\b|\b(?<![\d,.$])(\d{1,3})$")
_BYREF_RE = re.compile(
    r"incorporated\s+(?:herein\s+|into\s+this\s+item\s+)?by\s+reference"
    r"|annual\s+report\s+to\s+(?:share|stock)holders"
    r"|exhibit\s+13\b",
    re.I,
)
_NOT_APPLICABLE_RE = re.compile(r"^\W*(?:not\s+applicable|none|\[?reserved\]?)\W*$", re.I)
# "pages 52-167", "page 30", "Pages 28-30, 30" (sentences and index rows)
_PAGES_PHRASE_RE = re.compile(
    r"pages?\s+((?:\d{1,3}\s*(?:(?:-|through|to)\s*\d{1,3})?\s*(?:,|and)?\s*)+)", re.I
)
# Bare trailing list in an index row: "... Results of Operations 8-24" or "14, 70-71"
_TRAILING_PAGES_RE = re.compile(
    r"(?<![\w$.,])(\d{1,3}(?:\s*-\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*-\s*\d{1,3})?)*)\s*$"
)
_RANGE_RE = re.compile(r"(\d{1,3})\s*(?:(?:-|through|to)\s*(\d{1,3}))?")
# 'Reference is made to the section entitled "Management's Discussion ..." in the Financial
# Section of this report.'
_NAMED_REF_RE = re.compile(
    r"(?:section|caption|heading)s?\s+(?:entitled|captioned|titled|headed|called)\s+"
    r"\"([^\"]{5,200})\"",
    re.I,
)

# Regulation S-K item titles, used to trim page-resolved sections to the item's own heading
# on the first page and to stop before the next item's heading on the last page.
ITEM_TITLES: dict[str, str] = {
    "1A": r"risk\s+factors",
    "1B": r"unresolved\s+staff\s+comments",
    "1C": r"cybersecurity",
    "2": r"properties",
    "3": r"legal\s+proceedings",
    "4": r"mine\s+safety\s+disclosures",
    "5": r"market\s+for\s+(?:the\s+)?registrant",
    "7": r"management'?s\s+discussion\s+and\s+analysis",
    "7A": r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk",
    "8": r"financial\s+statements\s+and\s+supplementary\s+data",
    "9": r"changes\s+in\s+and\s+disagreements\s+with\s+accountants",
    "9A": r"controls\s+and\s+procedures",
}
TITLE_HEADING_MAX_CHARS = 150


class SectionStatus(StrEnum):
    OK = "ok"
    MISSING_BY_REFERENCE = "section_missing_by_reference"
    NOT_APPLICABLE = "section_not_applicable"
    REFERENCE_UNRESOLVED = "reference_unresolved"
    NOT_FOUND = "section_not_found"


class Method(StrEnum):
    ITEM_HEADING = "item_heading"
    PAGE_REFERENCE = "page_reference"
    NAMED_SECTION = "named_section_reference"
    NONE = "none"


def item_order(key: str) -> tuple[int, str]:
    m = re.fullmatch(r"(\d{1,2})([A-D]?)", key)
    if not m:
        raise ValueError(f"bad item key {key!r}")
    return int(m.group(1)), m.group(2)


@dataclass(frozen=True)
class Heading:
    index: int  # position in the cleaned block list
    key: str  # "1A", "7", "7A", ...
    text: str


@dataclass
class Section:
    item: str
    status: SectionStatus
    method: Method = Method.NONE
    text: str = ""
    start_heading: str | None = None
    end_heading: str | None = None
    pages: list[int] = field(default_factory=list)
    detail: str = ""
    block_starts: list[int] = field(default_factory=list)
    block_kinds: list[str] = field(default_factory=list)
    # Inline-XBRL facts displayed in the section: dicts with the block index and IxFact fields.
    ix_facts: list[dict[str, object]] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.text)


# -- boilerplate -------------------------------------------------------------------------


def _template(text: str) -> str:
    """Running header/footer template: page numbers at either end replaced by '#'."""
    t = text.lower().strip()
    t = re.sub(r"^\d{1,3}(?=\s|$)", "#", t)
    return re.sub(r"(?<=\s)\d{1,3}$|^\d{1,3}$", "#", t)


def boilerplate_mask(blocks: list[Block]) -> list[bool]:
    """True for running headers/footers and page numbers at page edges."""
    by_page: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(blocks):
        by_page[b.page].append(i)
    edges: set[int] = set()
    for idx in by_page.values():
        edges.update(idx[:2])
        edges.update(idx[-2:])
    paged = len(by_page) > 1

    def short(i: int) -> bool:
        return len(blocks[i].text) <= BOILERPLATE_MAX_CHARS

    pool = [i for i in (edges if paged else range(len(blocks))) if short(i)]
    exact = Counter(blocks[i].text.lower() for i in pool)
    templ = Counter(_template(blocks[i].text) for i in pool)

    mask = [False] * len(blocks)
    for i in pool:
        text = blocks[i].text
        if _PAGE_NUMBER_ONLY_RE.match(text):
            mask[i] = True
        elif _HEADING_RE.match(text):
            # A bare "Item 7" repeated as a page header goes; a real heading occurs once or
            # twice (TOC + body) and stays.
            mask[i] = exact[text.lower()] >= BOILERPLATE_MIN_REPEATS
        else:
            mask[i] = templ[_template(text)] >= BOILERPLATE_MIN_REPEATS
    return mask


# -- page numbers ----------------------------------------------------------------------


def page_numbers(blocks: list[Block]) -> dict[int, int]:
    """Map physical page -> printed page number, read from header/footer edge blocks.

    A candidate number is accepted only when a neighboring page agrees (n-1 before or n+1
    after), so a table cell that happens to end a page with "Total 12" is not mistaken for
    a page number.
    """
    by_page: dict[int, list[Block]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)
    cands: dict[int, set[int]] = {}
    for page, bl in by_page.items():
        found: set[int] = set()
        for b in bl[:2] + bl[-2:]:
            if len(b.text) > BOILERPLATE_MAX_CHARS:
                continue
            for m in _EDGE_NUMBER_RE.finditer(b.text):
                found.add(int(m.group(1) or m.group(2)))
        cands[page] = found
    result: dict[int, int] = {}
    for page in sorted(cands):
        for n in sorted(cands[page]):
            if (n - 1) in cands.get(page - 1, set()) or (n + 1) in cands.get(page + 1, set()):
                result[page] = n
                break
    return result


def parse_page_list(text: str) -> list[int]:
    pages: list[int] = []
    for m in _RANGE_RE.finditer(text):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if hi < lo or hi - lo > 400:
            continue
        pages.extend(range(lo, hi + 1))
    return pages


def referenced_pages(heading_text: str, stub_texts: list[str]) -> list[int]:
    """Pages named by a stub sentence or by cross-reference index rows."""
    pages: list[int] = []
    for text in stub_texts:
        m = _PAGES_PHRASE_RE.search(text)
        if m:
            pages.extend(parse_page_list(m.group(1)))
            # A sentence names the section's pages first ("appears on pages 52-167. ...
            # financial statements ... on pages 172-321"); later mentions are other things.
            if len(text) > BOILERPLATE_MAX_CHARS:
                break
            continue
        m2 = _TRAILING_PAGES_RE.search(text)
        if m2 and len(text) <= HEADING_MAX_CHARS:
            pages.extend(parse_page_list(m2.group(1)))
    if not pages:
        title = _HEADING_RE.match(heading_text)
        rest = title.group(3) if title else heading_text
        m = _PAGES_PHRASE_RE.search(rest) or _TRAILING_PAGES_RE.search(rest)
        # A single trailing number is a table-of-contents page, not a reference: require a
        # range or an explicit "Page(s)".
        if m and ("-" in m.group(0) or "page" in m.group(0).lower()):
            pages.extend(parse_page_list(m.group(1)))
    return sorted(set(pages))


# -- headings and spans ------------------------------------------------------------------


def find_headings(blocks: list[Block]) -> list[Heading]:
    heads: list[Heading] = []
    for i, b in enumerate(blocks):
        if len(b.text) > HEADING_MAX_CHARS:
            continue
        m = _HEADING_RE.match(b.text)
        if not m:
            continue
        rest = m.group(3).strip()
        if rest and rest[0].islower():
            continue  # "Item 7 of this Form 10-K ..." is a cross-reference
        key = f"{int(m.group(1))}{(m.group(2) or '').upper()}"
        heads.append(Heading(i, key, b.text))
    return heads


@dataclass(frozen=True)
class _Span:
    start: Heading
    end: Heading | None
    first: int  # first body block
    stop: int  # one past the last body block
    chars: int


def best_span(blocks: list[Block], heads: list[Heading], item: str) -> _Span | None:
    target = item_order(item)
    spans: list[_Span] = []
    for n, h in enumerate(heads):
        if h.key != item:
            continue
        end = next((e for e in heads[n + 1 :] if item_order(e.key) > target), None)
        stop = end.index if end else len(blocks)
        chars = sum(len(b.text) for b in blocks[h.index + 1 : stop])
        spans.append(_Span(h, end, h.index + 1, stop, chars))
    if not spans:
        return None
    # Longest span; ties go to the later one (the body follows the table of contents).
    return max(spans, key=lambda s: (s.chars, s.start.index))


def _ix_facts(blocks: list[Block]) -> list[dict[str, object]]:
    return [
        {
            "block": i,
            **{
                k: getattr(f, k)
                for k in ("name", "context", "unit", "scale", "decimals", "sign", "text")
            },
        }
        for i, b in enumerate(blocks)
        for f in b.ix
    ]


def _assemble(blocks: list[Block]) -> tuple[str, list[int], list[str]]:
    parts: list[str] = []
    starts: list[int] = []
    kinds: list[str] = []
    pos = 0
    for b in blocks:
        if parts:
            pos += 2  # "\n\n"
        starts.append(pos)
        kinds.append(b.kind)
        parts.append(b.text)
        pos += len(b.text)
    return "\n\n".join(parts), starts, kinds


def _section_from_blocks(
    item: str,
    blocks: list[Block],
    method: Method,
    start_heading: str | None,
    end_heading: str | None,
    pages: list[int] | None = None,
    detail: str = "",
) -> Section:
    text, starts, kinds = _assemble(blocks)
    return Section(
        item=item,
        status=SectionStatus.OK,
        method=method,
        text=text,
        start_heading=start_heading,
        end_heading=end_heading,
        pages=pages or [],
        detail=detail,
        block_starts=starts,
        block_kinds=kinds,
        ix_facts=_ix_facts(blocks),
    )


def _resolve_pages(
    item: str, clean: list[Block], numbers: dict[int, int], pages: list[int], span: _Span
) -> Section:
    wanted = set(pages)
    physical = sorted(p for p, n in numbers.items() if n in wanted)
    found = {numbers[p] for p in physical}
    if not physical:
        return Section(
            item,
            SectionStatus.REFERENCE_UNRESOLVED,
            Method.PAGE_REFERENCE,
            start_heading=span.start.text,
            pages=pages,
            detail="referenced pages could not be matched to printed page numbers",
        )
    chosen = [b for b in clean if b.page in set(physical)]
    chosen, notes = trim_to_titles(item, chosen)
    missing = sorted(wanted - found)
    detail = f"pages {_compact(sorted(found))}"
    if missing:
        detail += f"; printed page numbers not found for {_compact(missing)}"
    if notes:
        detail += f"; {notes}"
    return _section_from_blocks(
        item,
        chosen,
        Method.PAGE_REFERENCE,
        start_heading=chosen[0].text if chosen else None,
        end_heading=chosen[-1].text if chosen else None,
        pages=sorted(found),
        detail=detail,
    )


def title_heading(text: str, item: str) -> bool:
    """Does ``text`` open with item ``item``'s title, styled as a heading?

    Headings are either short blocks or upper-case run-in titles ("RISK FACTORS. The ...").
    """
    pattern = ITEM_TITLES.get(item)
    if pattern is None:
        return False
    m = re.match(rf"^(?:item\s*\d{{1,2}}[a-d]?\W*)?({pattern})\b", text, re.I)
    if not m:
        return False
    return len(text) <= TITLE_HEADING_MAX_CHARS or m.group(1).isupper()


def trim_to_titles(item: str, blocks: list[Block]) -> tuple[list[Block], str]:
    """Trim a page-resolved block list to the item's title and the next item's title.

    Only the first and last pages are trimmed; everything between is kept as referenced.
    """
    if not blocks:
        return blocks, ""
    first_page, last_page = blocks[0].page, blocks[-1].page
    start = next(
        (i for i, b in enumerate(blocks) if b.page == first_page and title_heading(b.text, item)),
        0,
    )
    end = len(blocks)
    others = [k for k in ITEM_TITLES if k != item]
    for i in range(start + 1, len(blocks)):
        b = blocks[i]
        if b.page == last_page and any(title_heading(b.text, k) for k in others):
            end = i
            break
    notes = []
    if start:
        notes.append("first page trimmed to item title")
    if end < len(blocks):
        notes.append("last page trimmed at next item title")
    return blocks[start:end], "; ".join(notes)


def _resolve_named_section(
    item: str, blocks: list[Block], clean: list[Block], title: str, span: _Span
) -> Section:
    """Resolve 'the section entitled "<title>" ... of this report'.

    The referenced section is the longest run of consecutive pages whose running
    header/footer is the title, plus the page before it when that page carries the title
    as a heading (sections usually open mid-page under a large title).
    """
    wanted = normalize_title(title)
    by_page: dict[int, list[Block]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)
    hits = sorted(
        p
        for p, bl in by_page.items()
        if any(normalize_title(b.text) == wanted for b in bl[:2] + bl[-2:])
    )
    runs: list[list[int]] = []
    for p in hits:
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    if not runs:
        return Section(
            item,
            SectionStatus.REFERENCE_UNRESOLVED,
            Method.NAMED_SECTION,
            start_heading=span.start.text,
            detail=f'no pages carry the referenced section title "{title}"',
        )
    run = max(runs, key=len)
    prev = run[0] - 1
    if any(normalize_title(b.text) == wanted for b in by_page.get(prev, [])):
        run.insert(0, prev)
    pages = set(run)
    chosen = [b for b in clean if b.page in pages]
    chosen, notes = trim_to_titles(item, chosen)
    detail = f'section "{title}" on physical pages {_compact(sorted(pages))}'
    return _section_from_blocks(
        item,
        chosen,
        Method.NAMED_SECTION,
        start_heading=chosen[0].text if chosen else None,
        end_heading=chosen[-1].text if chosen else None,
        detail=f"{detail}; {notes}" if notes else detail,
    )


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _compact(pages: list[int]) -> str:
    out: list[str] = []
    for p in pages:
        if out and "-" not in out[-1] and int(out[-1]) == p - 1:
            out[-1] = f"{out[-1]}-{p}"
        elif out and "-" in out[-1] and int(out[-1].split("-")[1]) == p - 1:
            out[-1] = f"{out[-1].split('-')[0]}-{p}"
        else:
            out.append(str(p))
    return ", ".join(out)


def isolate(blocks: list[Block], item: str) -> Section:
    """Isolate one item from rendered blocks. See the module docstring for the rules."""
    mask = boilerplate_mask(blocks)
    clean = [b for b, drop in zip(blocks, mask, strict=True) if not drop]
    heads = find_headings(clean)
    span = best_span(clean, heads, item)
    if span is None:
        return Section(item, SectionStatus.NOT_FOUND, detail=f"no 'Item {item}' heading found")

    body = clean[span.first : span.stop]
    end_text = span.end.text if span.end else None
    if span.chars >= STUB_MAX_CHARS:
        detail = "" if span.end else "no later Item heading; ran to end of document"
        return _section_from_blocks(
            item, body, Method.ITEM_HEADING, span.start.text, end_text, detail=detail
        )

    stub = " ".join(b.text for b in body)
    if _BYREF_RE.search(stub) or _BYREF_RE.search(span.start.text):
        return Section(
            item,
            SectionStatus.MISSING_BY_REFERENCE,
            Method.NONE,
            text=stub,
            start_heading=span.start.text,
            end_heading=end_text,
            detail="incorporated by reference to a document outside this filing",
        )
    pages = referenced_pages(span.start.text, [b.text for b in body])
    if pages:
        # Printed page numbers live in the footers that boilerplate removal just dropped,
        # so read them from the raw blocks.
        return _resolve_pages(item, clean, page_numbers(blocks), pages, span)
    named = _NAMED_REF_RE.search(stub)
    if named:
        return _resolve_named_section(item, blocks, clean, named.group(1), span)
    if _NOT_APPLICABLE_RE.match(stub or ""):
        return Section(
            item,
            SectionStatus.NOT_APPLICABLE,
            text=stub,
            start_heading=span.start.text,
            end_heading=end_text,
        )
    if span.chars == 0:
        return Section(
            item,
            SectionStatus.NOT_FOUND,
            start_heading=span.start.text,
            end_heading=end_text,
            detail="only table-of-contents style headings found; no section body",
        )
    return _section_from_blocks(
        item,
        body,
        Method.ITEM_HEADING,
        span.start.text,
        end_text,
        detail=f"short section ({span.chars} chars)",
    )


def extract_sections(
    html: bytes | str, items: tuple[str, ...] = TARGET_ITEMS
) -> dict[str, Section]:
    blocks = html_to_blocks(html)
    return {item: isolate(blocks, item) for item in items}
