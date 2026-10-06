"""Silver-label builder, gold-candidate sampler, and the gold labeling tool.

**Silver labels** are built automatically from two sources, both tied to the same filing:

1. *XBRL value match*: a dollar ``level`` mention whose value equals a fact reported in the
   same filing (same accession, unit USD) within the rounding its text implies
   (``$8,310 million`` matches facts in [8,309.5e6, 8,310.5e6]), and only when that
   rounding is at most 0.1% of the value; ``$1.2 billion`` (+/-4%) is too coarse to tell a
   real match from a coincidence. The role is the one the fact's tag
   implies via ``tag_map.yaml`` (REVENUE, ONE_TIME_CHARGE or OTHER_NUMERIC for any other
   tag). When matching facts imply different roles the mention stays unlabeled.
2. *Inline XBRL*: a mention whose displayed number is an ``ix:nonFraction`` in the same
   paragraph gets the role of that concept. Coverage is tiny in MD&A prose (measured by
   :func:`ixbrl_coverage`, reported in ``reports/ixbrl_coverage.md``).

Silver labels never produce GUIDANCE or RESTATED_PRIOR: future values and restatement
status are not implied by a tag. Mentions without a silver label are ignored by the loss.

**Gold labels** come only from a person, via :func:`run_labeling`. Candidates are a
stratified sample (by keyword stratum and company) written once to
``labels/gold_candidates.jsonl``; the tool shows each candidate span and records one
keypress per span to ``labels/gold_labels.jsonl``.
"""

from __future__ import annotations

import bisect
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from edgar_extract.xbrl import TagMap

NER_LABELS = ("REVENUE", "GUIDANCE", "ONE_TIME_CHARGE", "RESTATED_PRIOR", "OTHER_NUMERIC")
NOT_NUMERIC = "NOT_NUMERIC"  # gold only: the candidate span is not a numeric quantity

KEYMAP: dict[str, str] = {
    "r": "REVENUE",
    "g": "GUIDANCE",
    "c": "ONE_TIME_CHARGE",
    "p": "RESTATED_PRIOR",
    "o": "OTHER_NUMERIC",
    "x": NOT_NUMERIC,
}

# A rounding tolerance wider than this share of the value is too coarse to call a match.
# A filing reports ~1,500 USD facts over ~6 orders of magnitude, so a window of relative
# width r catches about 217*r unrelated facts by chance: r=4% ("$1.2 billion") expects ~8
# coincidental matches, r=0.1% ("$8,310 million") about 0.2. Measured on this universe
# (see README), matches looser than 0.1% were mostly coincidences.
MAX_RELATIVE_TOLERANCE = 0.001
MAX_SPANS_PER_CANDIDATE = 6
MAX_SENTENCE_CHARS = 500


def sentence_id(accession: str, section: str, sentence_start: int) -> str:
    return f"{accession}:{section}:{sentence_start}"


# -- silver labels ---------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.(\d+))?")


def rounding_tolerance(raw_text: str, unit_scale: float) -> float | None:
    """Half a unit of the last displayed digit, in base units ($1.2 billion -> 0.05e9)."""
    m = _NUMBER_RE.search(raw_text)
    if m is None:
        return None
    decimals = len(m.group(1) or "")
    return 0.5 * 10.0 ** (-decimals) * unit_scale


@dataclass(frozen=True)
class SilverLabel:
    role: str
    source: str  # "xbrl_value_match" or "ixbrl_tag"
    tags: tuple[str, ...]


def _role_for_tag(tag: str, tag_index: dict[str, tuple[str, int]], tag_map: TagMap) -> str:
    if tag in tag_index:
        return tag_map.fields[tag_index[tag][0]].role
    return "OTHER_NUMERIC"


def value_match_label(
    value: float, tolerance: float, facts: pd.DataFrame, tag_map: TagMap
) -> SilverLabel | None:
    """Silver label from facts of the same filing whose value is within ``tolerance``."""
    if tolerance <= 0 or value == 0 or tolerance / abs(value) > MAX_RELATIVE_TOLERANCE:
        return None
    hits = facts[(facts["value"].abs() - abs(value)).abs() <= tolerance]
    if hits.empty:
        return None
    index = tag_map.tag_index()
    tags = tuple(sorted(set(hits["tag"])))
    roles = {_role_for_tag(t, index, tag_map) for t in tags}
    if len(roles) != 1:
        return None
    return SilverLabel(roles.pop(), "xbrl_value_match", tags)


def ixbrl_label(
    raw_text: str, ix_in_block: list[dict[str, Any]], tag_map: TagMap
) -> SilverLabel | None:
    m = _NUMBER_RE.search(raw_text)
    if m is None:
        return None
    hits = [f for f in ix_in_block if f["text"] == m.group(0)]
    if not hits:
        return None
    index = tag_map.tag_index()
    tags = tuple(sorted({str(f["name"]) for f in hits}))
    roles = {_role_for_tag(t, index, tag_map) for t in tags}
    if len(roles) != 1:
        return None
    return SilverLabel(roles.pop(), "ixbrl_tag", tags)


def build_silver(
    extractions: pd.DataFrame,
    sections: pd.DataFrame,
    facts: pd.DataFrame,
    tag_map: TagMap,
) -> pd.DataFrame:
    """Add ``silver_role``, ``silver_source`` and ``silver_tags`` to each extraction row.

    ``facts`` is the flattened companyfacts frame for the companies involved.
    """
    usd_facts = facts[facts["unit"] == "USD"]
    by_accession = {str(acc): grp for acc, grp in usd_facts.groupby("accession")}
    sec_index: dict[tuple[str, str], tuple[list[int], list[dict[str, Any]]]] = {
        (str(r["accession"]), str(r["item"])): (list(r["block_starts"]), list(r["ix_facts"]))
        for r in sections.to_dict("records")
    }
    roles: list[str | None] = []
    sources: list[str | None] = []
    tags: list[str | None] = []
    for r in extractions.to_dict("records"):
        label: SilverLabel | None = None
        starts, ix = sec_index.get((str(r["accession"]), str(r["section"])), ([], []))
        if ix:
            block = bisect.bisect_right(starts, int(r["sentence_start"])) - 1
            in_block = [f for f in ix if int(f["block"]) == block]
            if in_block:
                label = ixbrl_label(str(r["raw_text"]), in_block, tag_map)
        if label is None and r["unit"] == "USD" and r["numeric_kind"] == "level":
            tol = rounding_tolerance(str(r["raw_text"]), float(r["unit_scale"]))
            filing_facts = by_accession.get(str(r["accession"]))
            if tol is not None and filing_facts is not None:
                label = value_match_label(float(r["value"]), tol, filing_facts, tag_map)
        roles.append(label.role if label else None)
        sources.append(label.source if label else None)
        tags.append(",".join(label.tags) if label else None)
    out = extractions.copy()
    out["silver_role"] = roles
    out["silver_source"] = sources
    out["silver_tags"] = tags
    return out


def ixbrl_coverage(sections: pd.DataFrame, extractions: pd.DataFrame) -> pd.DataFrame:
    """How many prose dollar mentions per item are displayed as inline-XBRL facts."""
    rows = []
    for item, grp in extractions[extractions["unit"] == "USD"].groupby("section"):
        secs = sections[(sections["item"] == item) & (sections["status"] == "ok")]
        n_ix = int(sum(len(x) for x in secs["ix_facts"]))
        rows.append({"item": item, "ix_facts_in_section": n_ix, "usd_prose_mentions": len(grp)})
    return pd.DataFrame(rows)


def _spans(grp: pd.DataFrame, label_col: str | None) -> list[dict[str, Any]]:
    spans = []
    for r in grp.to_dict("records"):
        span: dict[str, Any] = {
            "start": int(r["span_start"]) - int(r["sentence_start"]),
            "end": int(r["span_end"]) - int(r["sentence_start"]),
            "raw_text": r["raw_text"],
        }
        if label_col is not None:
            value = r[label_col]
            span["label"] = value if isinstance(value, str) else None
        spans.append(span)
    return spans


def _sentence_record(key: tuple[Any, ...], grp: pd.DataFrame) -> dict[str, Any]:
    ticker, cik, fiscal_year, accession, section, start = key
    return {
        "sentence_id": sentence_id(str(accession), str(section), int(start)),
        "ticker": str(ticker),
        "cik": int(cik),
        "fiscal_year": int(fiscal_year),
        "accession": str(accession),
        "section": str(section),
        "sentence_start": int(start),
        "sentence": str(grp["sentence"].iloc[0]),
    }


SENTENCE_KEYS = ["ticker", "cik", "fiscal_year", "accession", "section", "sentence_start"]


def silver_sentences(silver: pd.DataFrame) -> list[dict[str, Any]]:
    """Sentences with at least one silver-labeled span; other spans carry label None."""
    out = []
    for key, grp in silver.groupby(SENTENCE_KEYS, sort=False):
        if grp["silver_role"].isna().all():
            continue
        record = _sentence_record(cast("tuple[Any, ...]", key), grp)
        record["spans"] = _spans(grp, "silver_role")
        out.append(record)
    return out


# -- gold candidate sampling -----------------------------------------------------------

STRATA: list[tuple[str, int, str]] = [
    # (name, target count, regex over the sentence; "" means uniform random)
    (
        "restatement_language",
        30,
        r"\b(?:restat\w*|recast|previously\s+reported|as\s+revised|revised\s+to\s+reflect)\b",
    ),
    (
        "forward_looking",
        45,
        r"\b(?:expect\w*|anticipat\w*|outlook|guidance|forecast\w*|we\s+plan|we\s+intend|target)\b",
    ),
    (
        "charges",
        45,
        r"\b(?:impairment|restructuring|severance|write-?(?:off|down)|charge[sd]?|settlement|"
        r"litigation|one-time|non-?recurring)\b",
    ),
    ("revenue", 50, r"\b(?:revenues?|net\s+sales|sales)\b"),
    ("random", 70, ""),
]


def sample_gold_candidates(extractions: pd.DataFrame, seed: int = 13) -> list[dict[str, Any]]:
    """Stratified, company-balanced sample of sentences for gold labeling.

    Strata are defined by **keywords in the sentence**, not by either extractor's output,
    so the sample does not favor one extractor's mistakes. Within a stratum sentences are
    drawn round-robin across companies. Duplicate sentence texts (boilerplate repeated
    across years) appear once. Sentences longer than 500 characters or with more than six
    candidate spans are skipped to keep labeling tractable.
    """
    rng = random.Random(seed)
    pool: dict[str, dict[str, Any]] = {}
    seen_text: set[str] = set()
    for key, grp in extractions.groupby(SENTENCE_KEYS, sort=True):
        sent = str(grp["sentence"].iloc[0])
        norm = re.sub(r"\s+", " ", sent.lower())
        if (
            norm in seen_text
            or len(sent) > MAX_SENTENCE_CHARS
            or len(grp) > MAX_SPANS_PER_CANDIDATE
        ):
            continue
        seen_text.add(norm)
        record = _sentence_record(cast("tuple[Any, ...]", key), grp)
        record["spans"] = _spans(grp, None)
        pool[record["sentence_id"]] = record
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for name, target, pattern in STRATA:
        rx = re.compile(pattern, re.I) if pattern else None
        eligible = [
            c
            for sid, c in pool.items()
            if sid not in used and (rx is None or rx.search(c["sentence"]))
        ]
        by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in eligible:
            by_company[c["ticker"]].append(c)
        for lst in by_company.values():
            rng.shuffle(lst)
        companies = sorted(by_company)
        rng.shuffle(companies)
        taken = 0
        while taken < target and any(by_company.values()):
            for t in companies:
                if by_company[t] and taken < target:
                    c = by_company[t].pop()
                    chosen.append({**c, "stratum": name})
                    used.add(c["sentence_id"])
                    taken += 1
    rng.shuffle(chosen)
    return chosen


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- interactive gold labeling ---------------------------------------------------------


def flush_pending_input() -> None:
    """Discard keys typed before the current span was shown.

    Without this, a held-down key (auto-repeat) or pasted text is read as a stream of
    answers to spans nobody looked at. A no-op when stdin is not a terminal.
    """
    if not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError):
        return


def read_key() -> str:
    """Read one keypress from the terminal without waiting for Enter (POSIX)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        flush_pending_input()
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":  # Ctrl-C
        raise KeyboardInterrupt
    if ch == "":
        raise EOFError("stdin closed")
    return ch


def read_line_key(prompt: str = "key + Enter> ") -> str:
    """Line-mode fallback: type the letter and press Enter (works in any terminal)."""
    flush_pending_input()
    line = input(prompt).strip().lower()
    return line[:1]


def pick_key_reader(line_mode: bool = False) -> Callable[[], str]:
    """Single-keypress reading when the terminal supports it, otherwise line mode."""
    if line_mode or not sys.stdin.isatty():
        return read_line_key
    try:
        import termios

        termios.tcgetattr(sys.stdin.fileno())
    except (ImportError, OSError, termios.error):
        return read_line_key
    return read_key


HELP = (
    "[r] REVENUE  [g] GUIDANCE  [c] ONE_TIME_CHARGE  [p] RESTATED_PRIOR  [o] OTHER_NUMERIC  "
    "[x] not a number / bad span   |   [b] back one span  [s] skip sentence  "
    "[u] undo last saved sentence  [q] save & quit"
)


def _render(sentence: str, spans: list[dict[str, Any]], current: int) -> str:
    """Sentence (rich markup) with every candidate bracketed and the current one highlighted.

    All filing text is escaped: a literal "[r]" or "[$0.2 billion]" must not be read as a
    rich style tag.
    """
    from rich.markup import escape

    out, pos = [], 0
    for i, sp in enumerate(spans):
        out.append(escape(sentence[pos : sp["start"]]))
        text = escape(sentence[sp["start"] : sp["end"]])
        if i == current:
            out.append(f"[bold black on yellow]>>{text}<<[/]")
        elif "label" in sp:
            out.append(f"[green]\\[{text} = {sp['label']}][/]")
        else:
            out.append(f"[cyan]\\[{text}][/]")
        pos = sp["end"]
    out.append(escape(sentence[pos:]))
    return "".join(out)


def _remove_last_line(path: Path) -> dict[str, Any] | None:
    rows = read_jsonl(path)
    if not rows:
        return None
    write_jsonl(path, rows[:-1])
    return rows[-1]


def run_labeling(
    candidates_path: Path,
    out_path: Path,
    labeler: str,
    *,
    target: int = 200,
    key_reader: Callable[[], str] = read_key,
    printer: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Label candidate spans one keypress at a time; returns the number of sentences saved.

    Progress is appended to ``out_path`` after every sentence, so quitting (q or Ctrl-C)
    never loses work, and rerunning resumes where you stopped. Each record stores
    ``seconds``, the time from first showing the sentence to saving it, which
    :func:`audit_gold` uses to flag sentences labeled faster than they can be read.
    """
    from rich.markup import escape

    if printer is None:
        from rich.console import Console

        console = Console(highlight=False)
        printer = console.print
    candidates = read_jsonl(candidates_path)
    if not candidates:
        raise FileNotFoundError(
            f"no candidates in {candidates_path}; run `edgar-extract label --prepare`"
        )
    done = {r["sentence_id"] for r in read_jsonl(out_path)}
    queue = [c for c in candidates if c["sentence_id"] not in done]
    saved = len(done)
    i = 0
    while i < len(queue):
        cand = queue[i]
        spans = [dict(sp) for sp in cand["spans"]]
        j = 0
        skipped = False
        shown_at = clock()
        while j < len(spans):
            printer("")
            where = f"{cand['ticker']} FY{cand['fiscal_year']} Item {cand['section']}"
            printer(
                f"[dim]{saved}/{target} saved · {where} · "
                f"stratum {cand.get('stratum', '-')} · span {j + 1}/{len(spans)}[/]"
            )
            printer(_render(cand["sentence"], spans, j))
            printer(f"[dim]{escape(HELP)}[/]")
            key = key_reader().lower()
            if key in KEYMAP:
                spans[j]["label"] = KEYMAP[key]
                j += 1
            elif key == "b" and j > 0:
                j -= 1
                spans[j].pop("label", None)
            elif key == "s":
                skipped = True
                break
            elif key == "u":
                last = _remove_last_line(out_path)
                if last is not None:
                    saved -= 1
                    queue.insert(
                        i, next(c for c in candidates if c["sentence_id"] == last["sentence_id"])
                    )
                    printer("[yellow]removed the last saved sentence; it is next in the queue[/]")
                    break
            elif key == "q":
                printer(f"saved {saved} sentences to {out_path}")
                return saved
        else:
            record = {
                **{
                    k: cand[k]
                    for k in (
                        "sentence_id",
                        "ticker",
                        "cik",
                        "fiscal_year",
                        "accession",
                        "section",
                        "sentence",
                        "sentence_start",
                    )
                },
                "stratum": cand.get("stratum"),
                "spans": [
                    {k: sp[k] for k in ("start", "end", "raw_text", "label")} for sp in spans
                ],
                "labeler": labeler,
                "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "seconds": round(clock() - shown_at, 2),
            }
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            saved += 1
            i += 1
            continue
        if skipped:
            i += 1
        # after "u" the loop re-reads queue[i], which is now the restored sentence
    printer(f"all candidates done; saved {saved} sentences to {out_path}")
    return saved


# -- gold label audit ------------------------------------------------------------------

# Reading a 10-K sentence and deciding a role takes seconds per number. A sentence saved
# in less than this many seconds per span was not read. Measured on the first labeling
# session: its first 20 sentences took at least 2.0 s per span.
MIN_SECONDS_PER_SPAN = 1.5

# The guide's RESTATED_PRIOR definition: the sentence itself calls the figure restated.
_RESTATEMENT_WORDS = re.compile(
    r"restat|recast|revis|as previously reported|as originally reported", re.IGNORECASE
)


def _sentence_seconds(gold: list[dict[str, Any]]) -> list[float | None]:
    """Seconds spent on each sentence: the stored ``seconds``, else the gap since the
    previously saved sentence (``labeled_at`` order). ``None`` when neither exists (the
    first sentence of a file without timings)."""
    order = sorted(range(len(gold)), key=lambda i: str(gold[i].get("labeled_at", "")))
    out: list[float | None] = [None] * len(gold)
    prev: datetime | None = None
    for i in order:
        stamp = gold[i].get("labeled_at")
        now = datetime.fromisoformat(stamp) if stamp else None
        if gold[i].get("seconds") is not None:
            out[i] = float(gold[i]["seconds"])
        elif now is not None and prev is not None:
            out[i] = (now - prev).total_seconds()
        prev = now
    return out


def audit_gold(
    gold: list[dict[str, Any]], *, min_seconds_per_span: float = MIN_SECONDS_PER_SPAN
) -> list[dict[str, Any]]:
    """One row per gold sentence with any warning; ``fast`` rows are unreliable.

    Warnings:
    - ``fast``: saved in under ``min_seconds_per_span`` seconds per span.
    - ``guide``: a label the guide rules out from the sentence alone: RESTATED_PRIOR in
      a sentence that never says restated/recast/revised/previously reported, or a
      percentage labeled REVENUE or ONE_TIME_CHARGE.
    """
    rows = []
    for g, secs in zip(gold, _sentence_seconds(gold), strict=True):
        n_spans = max(1, len(g["spans"]))
        fast = secs is not None and secs < min_seconds_per_span * n_spans
        guide = []
        for sp in g["spans"]:
            label = sp.get("label")
            if label == "RESTATED_PRIOR" and not _RESTATEMENT_WORDS.search(g["sentence"]):
                guide.append(f"{sp['raw_text']}: RESTATED_PRIOR without restatement wording")
            if label in ("REVENUE", "ONE_TIME_CHARGE") and "%" in sp["raw_text"]:
                guide.append(f"{sp['raw_text']}: a percentage labeled {label}")
        if fast or guide:
            rows.append(
                {
                    "sentence_id": g["sentence_id"],
                    "ticker": g["ticker"],
                    "labeled_at": g.get("labeled_at"),
                    "seconds": secs,
                    "spans": len(g["spans"]),
                    "fast": fast,
                    "guide": guide,
                }
            )
    return rows


def split_fast(
    gold: list[dict[str, Any]], *, min_seconds_per_span: float = MIN_SECONDS_PER_SPAN
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, fast): gold sentences split by the ``fast`` audit warning, order kept."""
    fast_ids = {
        r["sentence_id"]
        for r in audit_gold(gold, min_seconds_per_span=min_seconds_per_span)
        if r["fast"]
    }
    kept = [g for g in gold if g["sentence_id"] not in fast_ids]
    return kept, [g for g in gold if g["sentence_id"] in fast_ids]


def gold_summary(gold: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(sp["label"] for g in gold for sp in g["spans"])
    return {
        "sentences": len(gold),
        "spans": sum(labels.values()),
        "labels": dict(labels),
        "companies": dict(Counter(g["ticker"] for g in gold)),
    }
