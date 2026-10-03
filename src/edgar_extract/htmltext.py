"""Filing HTML -> ordered text blocks that keep paragraph and table-row boundaries.

The renderer is deliberately simple and deterministic:

* hidden content is dropped (``display:none``, the inline-XBRL ``ix:header``, scripts),
* block elements (``p``, ``div``, ``br``, ``li``, headings...) end a paragraph,
* a table row whose cells contain no nested table becomes one ``table_row`` block with the
  cell texts joined by single spaces, so a heading laid out as
  ``<td>Item 7.</td><td>Management's Discussion...</td>`` reads as one line,
* rows that contain nested tables are treated as containers and rendered recursively,
* whitespace and typography are normalized (``&nbsp;``, zero-width characters, unicode
  dashes and minus signs, curly quotes) so regexes see one canonical form.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from lxml import etree, html

BlockKind = Literal["p", "table_row"]

BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "br", "caption", "center",
        "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "html", "li", "main", "nav",
        "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr",
        "ul",
    }
)  # fmt: skip
SKIP_TAGS = frozenset({"script", "style", "head", "title", "noscript", "ix:header", "xml"})
CELL_TAGS = frozenset({"td", "th"})

_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
_BREAK_BEFORE_RE = re.compile(r"(?:page-break-before\s*:\s*always|break-before\s*:\s*page)", re.I)
_BREAK_AFTER_RE = re.compile(r"(?:page-break-after\s*:\s*always|break-after\s*:\s*page)", re.I)
_WS_RE = re.compile(r"[ \t\r\n\f\v]+")

# Typography folded to ASCII so every downstream regex sees one form. Code points rather
# than literals, so the table is unambiguous to read.
_SPACES = (0x00A0, *range(0x2002, 0x200B), 0x202F, 0x205F, 0x3000)
_DROPPED = (0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)
_DASHES = (*range(0x2010, 0x2016), 0x2212, 0xFE58, 0xFE63, 0xFF0D)
_SINGLE_QUOTES = (0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_DOUBLE_QUOTES = (0x201C, 0x201D, 0x201E, 0x201F, 0x2033)
_BULLETS = (0x2022, 0x25CF, 0x25AA)
_TRANSLATE: dict[int, str | None] = {
    **{cp: " " for cp in _SPACES},
    **{cp: None for cp in _DROPPED},
    **{cp: "-" for cp in _DASHES},
    **{cp: "'" for cp in _SINGLE_QUOTES},
    **{cp: '"' for cp in _DOUBLE_QUOTES},
    **{cp: "*" for cp in _BULLETS},
}


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text.translate(_TRANSLATE)).strip()


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    page: int = 0  # physical page (0-based), counted from CSS page breaks


def _tag(el: etree._Element) -> str:
    tag = el.tag
    if not isinstance(tag, str):  # comments, processing instructions
        return ""
    return tag.lower()


def _is_hidden(el: etree._Element) -> bool:
    style = el.get("style")
    return bool(style and _HIDDEN_RE.search(style))


def _skip(el: etree._Element) -> bool:
    tag = _tag(el)
    return tag == "" or tag in SKIP_TAGS or _is_hidden(el)


def _has_nested_table(row: etree._Element) -> bool:
    return any(_tag(d) == "table" for d in row.iterdescendants())


def _inline_text(el: etree._Element) -> str:
    """Visible text of ``el`` with block boundaries turned into spaces."""
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        if _skip(node):
            return
        block = _tag(node) in BLOCK_TAGS
        if block:
            parts.append(" ")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if block:
            parts.append(" ")

    walk(el)
    return "".join(parts)


class _Renderer:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.buf: list[str] = []
        self.page = 0

    def flush(self) -> None:
        text = normalize_text("".join(self.buf))
        self.buf.clear()
        if text:
            self.blocks.append(Block("p", text, self.page))

    def page_break(self) -> None:
        self.flush()
        self.page += 1

    def walk(self, el: etree._Element) -> None:
        style = el.get("style") or ""
        if style and _BREAK_BEFORE_RE.search(style):
            self.page_break()
        self._walk_content(el)
        if style and _BREAK_AFTER_RE.search(style):
            self.page_break()

    def _walk_content(self, el: etree._Element) -> None:
        tag = _tag(el)
        if tag == "tr" and not _has_nested_table(el):
            self.flush()
            cells = [normalize_text(_inline_text(c)) for c in el if _tag(c) in CELL_TAGS]
            row = " ".join(c for c in cells if c)
            if row:
                self.blocks.append(Block("table_row", row, self.page))
            return
        block = tag in BLOCK_TAGS
        if block:
            self.flush()
        if el.text:
            self.buf.append(el.text)
        for child in el:
            if not _skip(child):
                self.walk(child)
            if child.tail:
                self.buf.append(child.tail)
        if block:
            self.flush()


def parse_html(content: bytes | str) -> etree._Element:
    if isinstance(content, str):
        content = content.encode("utf-8")
    # Strip an XML declaration: lxml.html rejects str input with one and some SEC
    # documents are XHTML.
    content = re.sub(rb"^\s*<\?xml[^>]*\?>", b"", content, count=1)
    parser = html.HTMLParser(recover=True, huge_tree=True, remove_comments=True)
    root = html.document_fromstring(content, parser=parser)
    return root


def html_to_blocks(content: bytes | str) -> list[Block]:
    root = parse_html(content)
    renderer = _Renderer()
    if not _skip(root):
        renderer.walk(root)
    renderer.flush()
    return renderer.blocks


def iter_text(blocks: list[Block]) -> Iterator[str]:
    for b in blocks:
        yield b.text
