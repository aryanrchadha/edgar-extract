"""Silver-label rules, gold candidate sampling, and the labeling tool (scripted keys)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from edgar_extract.ner.labeling import (
    build_silver,
    gold_summary,
    ixbrl_label,
    read_jsonl,
    rounding_tolerance,
    run_labeling,
    sample_gold_candidates,
    silver_sentences,
    value_match_label,
    write_jsonl,
)
from edgar_extract.xbrl import load_tag_map

TAG_MAP = load_tag_map()


def facts(*rows: tuple[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"tag": t, "value": v, "unit": "USD", "accession": "A-1"} for t, v in rows]
    )


@pytest.mark.parametrize(
    ("raw", "scale", "tol"),
    [
        ("$1.2 billion", 1e9, 0.05e9),
        ("$8,310 million", 1e6, 0.5e6),
        ("$1.215 billion", 1e9, 0.0005e9),
        ("$ 391,035", 1.0, 0.5),
    ],
)
def test_rounding_tolerance(raw: str, scale: float, tol: float) -> None:
    assert rounding_tolerance(raw, scale) == pytest.approx(tol)


def test_value_match_requires_precise_text() -> None:
    f = facts(("us-gaap:Revenues", 45_754e6), ("us-gaap:Assets", 1.21e9))
    hit = value_match_label(45_754e6, 0.5e6, f, TAG_MAP)
    assert hit is not None and hit.role == "REVENUE" and hit.tags == ("us-gaap:Revenues",)
    # "$1.2 billion" is +/-4%: too coarse to separate a match from a coincidence.
    assert value_match_label(1.2e9, 0.05e9, f, TAG_MAP) is None


def test_value_match_uses_absolute_values_and_unmapped_tags_are_other() -> None:
    f = facts(("us-gaap:NetCashProvidedByUsedInInvestingActivities", -94_312e6))
    hit = value_match_label(-94_312e6, 0.5e6, f, TAG_MAP)
    assert hit is not None and hit.role == "OTHER_NUMERIC"
    f = facts(("us-gaap:PaymentsForRepurchaseOfCommonStock", 1_795e6))
    hit = value_match_label(1_795e6, 0.5e6, f, TAG_MAP)
    assert hit is not None and hit.role == "OTHER_NUMERIC"


def test_value_match_with_conflicting_roles_is_dropped() -> None:
    f = facts(("us-gaap:Revenues", 3_595e6), ("us-gaap:RestructuringCharges", 3_595e6))
    assert value_match_label(3_595e6, 0.5e6, f, TAG_MAP) is None


def test_ixbrl_label() -> None:
    ix = [{"text": "159", "name": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"}]
    hit = ixbrl_label("$159 million", ix, TAG_MAP)
    assert hit is not None and (hit.role, hit.source) == ("REVENUE", "ixbrl_tag")
    assert ixbrl_label("$160 million", ix, TAG_MAP) is None


def _extraction(
    acc: str, sentence: str, raw: str, value: float, scale: float, **kw: object
) -> dict[str, object]:
    sent_start = int(kw.pop("sentence_start", 100))  # type: ignore[call-overload]
    start = sentence.index(raw)
    row: dict[str, object] = {
        "ticker": "KO", "cik": 21344, "fiscal_year": 2023, "accession": acc, "section": "7",
        "sentence": sentence, "sentence_start": sent_start,
        "sentence_end": sent_start + len(sentence), "span_start": sent_start + start,
        "span_end": sent_start + start + len(raw), "raw_text": raw, "value": value,
        "unit": "USD", "numeric_kind": "level", "unit_scale": scale, "role": "REVENUE",
    }  # fmt: skip
    row.update(kw)
    return row


def test_build_silver_and_sentences() -> None:
    s = "Net operating revenues were $45,754 million in 2023, compared to $1.2 billion."
    ex = pd.DataFrame(
        [
            _extraction("A-1", s, "$45,754 million", 45_754e6, 1e6),
            _extraction("A-1", s, "$1.2 billion", 1.2e9, 1e9),
        ]
    )
    sections = pd.DataFrame(
        [{"accession": "A-1", "item": "7", "block_starts": [0], "ix_facts": []}]
    )
    f = pd.DataFrame(
        [
            {"tag": "us-gaap:Revenues", "value": 45_754e6, "unit": "USD", "accession": "A-1"},
            {"tag": "us-gaap:Revenues", "value": 45_754e6, "unit": "USD", "accession": "OTHER"},
        ]
    )
    silver = build_silver(ex, sections, f, TAG_MAP)
    assert silver["silver_role"].iloc[0] == "REVENUE" and pd.isna(silver["silver_role"].iloc[1])
    assert silver["silver_source"].iloc[0] == "xbrl_value_match"
    (sent,) = silver_sentences(silver)
    assert sent["sentence_id"] == "A-1:7:100"
    assert [sp["label"] for sp in sent["spans"]] == ["REVENUE", None]
    assert s[sent["spans"][0]["start"] : sent["spans"][0]["end"]] == "$45,754 million"


def test_silver_only_from_same_filing() -> None:
    s = "Revenues were $45,754 million."
    ex = pd.DataFrame([_extraction("A-1", s, "$45,754 million", 45_754e6, 1e6)])
    sections = pd.DataFrame(
        [{"accession": "A-1", "item": "7", "block_starts": [0], "ix_facts": []}]
    )
    f = pd.DataFrame(
        [{"tag": "us-gaap:Revenues", "value": 45_754e6, "unit": "USD", "accession": "B-2"}]
    )
    assert build_silver(ex, sections, f, TAG_MAP)["silver_role"].isna().all()


# -- gold candidates --------------------------------------------------------------------


def _pool() -> pd.DataFrame:
    rows = []
    sentences = {
        "revenue": "Revenue increased to $5.0 billion in the year due to demand.",
        "charges": "We recorded impairment charges of $1.0 billion in the year.",
        "forward_looking": "We expect capital expenditures of $2.0 billion next year.",
        "restatement_language": "Prior amounts have been recast to $3.0 billion for comparability.",
        "random": "Cash and equivalents were $9.0 billion at year end.",
    }
    for t_i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for fy in (2022, 2023):
            for k_i, (_key, base) in enumerate(sentences.items()):
                sent = base.replace(" in the year", f" in fiscal {fy} for {ticker}")
                raw = sent[sent.index("$") : sent.index("billion") + 7]
                rows.append(
                    _extraction(
                        f"{t_i}-{fy}", sent, raw, 1e9, 1e9,
                        ticker=ticker, fiscal_year=fy, sentence_start=k_i * 1000,
                    )
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_sampling_is_deterministic_deduplicated_and_spread() -> None:
    pool = _pool()
    a, b = sample_gold_candidates(pool), sample_gold_candidates(pool)
    assert a == b
    ids = [c["sentence_id"] for c in a]
    assert len(ids) == len(set(ids))
    texts = [c["sentence"].lower() for c in a]
    assert len(texts) == len(set(texts))
    assert {c["ticker"] for c in a} == {"AAA", "BBB", "CCC", "DDD"}
    by_stratum = {c["stratum"] for c in a}
    assert "restatement_language" in by_stratum and "random" in by_stratum
    for c in a:
        for sp in c["spans"]:
            assert c["sentence"][sp["start"] : sp["end"]] == sp["raw_text"]
            assert "label" not in sp  # candidates never carry a suggestion


def test_sampling_skips_long_or_crowded_sentences() -> None:
    long = "Revenue was $1.0 billion " + "and more words " * 60 + "."
    pool = pd.DataFrame([_extraction("A", long, "$1.0 billion", 1e9, 1e9)])
    assert sample_gold_candidates(pool) == []


# -- interactive tool with scripted keys ----------------------------------------------------


def keys(seq: str) -> Iterator[str]:
    yield from seq


@pytest.fixture
def candidates(tmp_path: Path) -> Path:
    path = tmp_path / "gold_candidates.jsonl"
    rows = []
    for n in range(3):
        sent = f"Revenue was $1.{n} billion and margin was 4{n}%."
        rows.append(
            {
                "sentence_id": f"A:7:{n}", "ticker": "AAA", "cik": 1, "fiscal_year": 2024,
                "accession": "A", "section": "7", "sentence_start": n, "sentence": sent,
                "stratum": "revenue",
                "spans": [
                    {"start": sent.index("$"), "end": sent.index(" billion") + 8,
                     "raw_text": f"$1.{n} billion"},
                    {"start": sent.index("4"), "end": sent.index("%") + 1, "raw_text": f"4{n}%"},
                ],
            }
        )  # fmt: skip
    write_jsonl(path, rows)
    return path


def run(candidates: Path, out: Path, seq: str) -> int:
    it = keys(seq)
    return run_labeling(
        candidates, out, "tester", key_reader=lambda: next(it), printer=lambda s: None
    )


def test_labels_every_span_and_saves_per_sentence(candidates: Path, tmp_path: Path) -> None:
    out = tmp_path / "gold.jsonl"
    assert run(candidates, out, "rorogo") == 3
    gold = read_jsonl(out)
    assert [[sp["label"] for sp in g["spans"]] for g in gold] == [
        ["REVENUE", "OTHER_NUMERIC"],
        ["REVENUE", "OTHER_NUMERIC"],
        ["GUIDANCE", "OTHER_NUMERIC"],
    ]
    assert all(g["labeler"] == "tester" and g["labeled_at"] for g in gold)
    assert gold_summary(gold)["labels"] == {"REVENUE": 2, "OTHER_NUMERIC": 3, "GUIDANCE": 1}


def test_back_skip_quit_and_resume(candidates: Path, tmp_path: Path) -> None:
    out = tmp_path / "gold.jsonl"
    # sentence 0: r, back, c, o -> saved [c, o]; sentence 1: skipped; sentence 2: r then quit
    assert run(candidates, out, "rbcosrq") == 1
    (first,) = read_jsonl(out)
    assert [sp["label"] for sp in first["spans"]] == ["ONE_TIME_CHARGE", "OTHER_NUMERIC"]
    # resume: sentence 0 is done; skipped sentence 1 comes back, then 2
    assert run(candidates, out, "roxo") == 3
    assert [g["sentence_id"] for g in read_jsonl(out)] == ["A:7:0", "A:7:1", "A:7:2"]
    assert read_jsonl(out)[2]["spans"][0]["label"] == "NOT_NUMERIC"


def test_undo_last_saved_sentence(candidates: Path, tmp_path: Path) -> None:
    out = tmp_path / "gold.jsonl"
    # label sentence 0 as r,o; then undo; relabel sentence 0 as c,o; then quit
    assert run(candidates, out, "rouco" + "q") == 1
    (only,) = read_jsonl(out)
    assert only["sentence_id"] == "A:7:0"
    assert [sp["label"] for sp in only["spans"]] == ["ONE_TIME_CHARGE", "OTHER_NUMERIC"]


def test_unknown_keys_are_ignored(candidates: Path, tmp_path: Path) -> None:
    out = tmp_path / "gold.jsonl"
    assert run(candidates, out, "z?r!oq") == 1
    assert [sp["label"] for sp in read_jsonl(out)[0]["spans"]] == ["REVENUE", "OTHER_NUMERIC"]


def test_repo_candidates_file_is_well_formed() -> None:
    path = Path(__file__).parents[1] / "labels" / "gold_candidates.jsonl"
    rows = read_jsonl(path)
    assert len(rows) >= 200
    for r in rows:
        assert r["sentence_id"] and r["spans"] and r["stratum"]
        for sp in r["spans"]:
            assert r["sentence"][sp["start"] : sp["end"]] == sp["raw_text"]
    assert json.dumps(rows[0])  # serializable


def test_line_mode_reader_takes_first_letter(monkeypatch: pytest.MonkeyPatch) -> None:
    from edgar_extract.ner import labeling

    answers = iter(["  R  ", "", "other"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert [labeling.read_line_key() for _ in range(3)] == ["r", "", "o"]


def test_non_tty_stdin_falls_back_to_line_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from edgar_extract.ner import labeling

    monkeypatch.setattr("sys.stdin", io.StringIO("r\n"))
    assert labeling.pick_key_reader() is labeling.read_line_key
    assert labeling.pick_key_reader(line_mode=True) is labeling.read_line_key


def test_line_mode_session_saves_labels(
    candidates: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edgar_extract.ner import labeling

    answers = iter(["r", "o", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    out = tmp_path / "gold.jsonl"
    saved = run_labeling(
        candidates, out, "tester", key_reader=labeling.read_line_key, printer=lambda s: None
    )
    assert saved == 1
    assert [sp["label"] for sp in read_jsonl(out)[0]["spans"]] == ["REVENUE", "OTHER_NUMERIC"]
