"""Span-level evaluation of the regex and NER extractors on the gold **test** split only.

For each role r: precision = TP / predicted r, recall = TP / gold r, F1, with gold support
and 95% Wilson intervals for precision and recall. Gold spans marked "not a number" count
as gold ``O``; predicting any role on them is a false positive for that role. Both
extractors are scored on exactly the same candidate spans.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from edgar_extract.baseline_extract import NER_ROLE, assign_role
from edgar_extract.ner.labeling import NER_LABELS, NOT_NUMERIC
from edgar_extract.numeric_spans import find_numeric_mentions

SMALL_N = 30


@dataclass(frozen=True)
class RoleMetrics:
    role: str
    support: int  # gold spans with this role
    predicted: int
    tp: int
    precision: float | None
    recall: float | None
    f1: float | None
    precision_ci: tuple[float, float] | None
    recall_ci: tuple[float, float] | None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def gold_pairs(gold: Sequence[dict[str, Any]]) -> list[tuple[str, int, int, str, str]]:
    """(sentence_id, start, end, sentence, gold_label) with "not a number" mapped to O."""
    return [
        (
            g["sentence_id"],
            sp["start"],
            sp["end"],
            g["sentence"],
            "O" if sp["label"] == NOT_NUMERIC else sp["label"],
        )
        for g in gold
        for sp in g["spans"]
    ]


def regex_predictor(sentence: str, spans: Sequence[tuple[int, int]]) -> list[str]:
    """Regex baseline role for each span, mapped to the NER label set."""
    by_offset = {(m.start, m.end): m for m in find_numeric_mentions(sentence)}
    out = []
    for s, e in spans:
        m = by_offset.get((s, e))
        out.append(NER_ROLE[assign_role(sentence, m).role] if m else "OTHER_NUMERIC")
    return out


def predict_all(
    gold: Sequence[dict[str, Any]],
    predictor: Callable[[str, Sequence[tuple[int, int]]], list[str]],
) -> list[tuple[str, str]]:
    """(gold label, predicted label) for every gold span."""
    pairs = []
    for g in gold:
        spans = [(int(sp["start"]), int(sp["end"])) for sp in g["spans"]]
        preds = predictor(g["sentence"], spans)
        for sp, pred in zip(g["spans"], preds, strict=True):
            truth = "O" if sp["label"] == NOT_NUMERIC else sp["label"]
            pairs.append((truth, pred))
    return pairs


def role_metrics(
    pairs: Sequence[tuple[str, str]], roles: Sequence[str] = NER_LABELS
) -> list[RoleMetrics]:
    out = []
    for r in roles:
        tp = sum(1 for t, p in pairs if t == r and p == r)
        pred = sum(1 for _, p in pairs if p == r)
        sup = sum(1 for t, _ in pairs if t == r)
        prec = tp / pred if pred else None
        rec = tp / sup if sup else None
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec and rec
            else (0.0 if prec is not None and rec is not None else None)
        )
        out.append(RoleMetrics(r, sup, pred, tp, prec, rec, f1, wilson(tp, pred), wilson(tp, sup)))
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _ci(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f" [{ci[0]:.2f}-{ci[1]:.2f}]"


def metrics_table(name: str, metrics: Sequence[RoleMetrics]) -> list[str]:
    lines = [
        f"### {name}",
        "",
        "| role | gold n | predicted n | precision [95% CI] | recall [95% CI] | F1 | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        note = f"n < {SMALL_N}: unreliable" if m.support < SMALL_N else ""
        lines.append(
            f"| {m.role} | {m.support} | {m.predicted} | {_fmt(m.precision)}{_ci(m.precision_ci)} "
            f"| {_fmt(m.recall)}{_ci(m.recall_ci)} | {_fmt(m.f1)} | {note} |"
        )
    return [*lines, ""]
