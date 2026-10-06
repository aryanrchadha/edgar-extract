"""Orchestration for `edgar-extract ner train` and `edgar-extract ner evaluate`."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgar_extract.config import Settings
from edgar_extract.ner.evaluate import (
    SMALL_N,
    metrics_table,
    predict_all,
    regex_predictor,
    role_metrics,
)
from edgar_extract.ner.labeling import (
    MIN_SECONDS_PER_SPAN,
    NOT_NUMERIC,
    gold_summary,
    read_jsonl,
    split_fast,
)

# Below this many labeled sentences a held-out evaluation is not meaningful.
MIN_GOLD_SENTENCES = 50


def _paths(settings: Settings) -> dict[str, Path]:
    from edgar_extract.ner.infer import model_dir
    from edgar_extract.pipeline import labels_dir

    return {
        "gold": labels_dir(settings) / "gold_labels.jsonl",
        "silver": settings.paths.interim_dir / "silver_sentences.jsonl",
        "model": model_dir(settings),
        "report": settings.paths.reports_dir / "ner_metrics.md",
        "card": model_dir(settings) / "MODEL_CARD.md",
        "train_report": model_dir(settings) / "train_report.json",
    }


def load_gold(settings: Settings) -> tuple[list[dict[str, Any]], int]:
    """(usable gold sentences, number excluded as labeled too fast to have been read)."""
    gold, fast = split_fast(read_jsonl(_paths(settings)["gold"]))
    if len(gold) < MIN_GOLD_SENTENCES:
        excluded = (
            f"; {len(fast)} more were labeled in under {MIN_SECONDS_PER_SPAN} s per span and "
            "are excluded (see `edgar-extract label --audit`)"
            if fast
            else ""
        )
        raise RuntimeError(
            f"only {len(gold)} usable gold sentences in {_paths(settings)['gold']}{excluded} "
            f"(need >= {MIN_GOLD_SENTENCES}). Label with `edgar-extract label` first; "
            "metrics are never computed on silver labels."
        )
    return gold, len(fast)


def run_training(settings: Settings, *, epochs: int | None = None) -> dict[str, Any]:
    from edgar_extract.ner.train import (
        TrainConfig,
        TrainReport,
        build_training_set,
        company_split,
        pick_device,
        train,
    )

    paths = _paths(settings)
    gold, n_fast = load_gold(settings)
    silver = read_jsonl(paths["silver"])
    if not silver:
        raise FileNotFoundError(f"{paths['silver']} missing; run `edgar-extract label --prepare`")
    cfg = TrainConfig()
    if epochs is not None:
        cfg.epochs = epochs
    train_cos, test_cos = company_split(gold, cfg.test_fraction, cfg.seed)
    examples = build_training_set(silver, gold, test_cos, cfg)
    losses = train(examples, cfg, paths["model"])
    gold_train = [g for g in gold if g["ticker"] in set(train_cos)]
    silver_used = [e for e in examples if e.source == "silver"]
    report = TrainReport(
        config=asdict(cfg),
        n_silver_sentences=len(silver_used),
        n_silver_spans=sum(1 for e in silver_used for *_, r in e.spans if r is not None),
        n_gold_train_sentences=len(gold_train),
        n_gold_train_spans=sum(len(g["spans"]) for g in gold_train),
        train_companies=train_cos,
        test_companies=test_cos,
        device=pick_device(),
        epochs_loss=losses,
        n_gold_excluded_fast=n_fast,
    )
    paths["train_report"].write_text(json.dumps(asdict(report), indent=1))
    return asdict(report)


def run_evaluation(settings: Settings) -> Path:
    from edgar_extract.ner.infer import NerModel
    from edgar_extract.ner.train import company_split

    paths = _paths(settings)
    gold, n_fast = load_gold(settings)
    train_report = json.loads(paths["train_report"].read_text())
    cfg = train_report["config"]
    _, test_cos = company_split(gold, cfg["test_fraction"], cfg["seed"])
    if sorted(test_cos) != sorted(train_report["test_companies"]):
        raise RuntimeError("gold labels changed since training; retrain before evaluating")
    test = [g for g in gold if g["ticker"] in set(test_cos)]
    model = NerModel.load(paths["model"])

    def ner_predictor(sentence: str, spans: Any) -> list[str]:
        return [label for label, _ in model.predict(sentence, spans)]

    regex_m = role_metrics(predict_all(test, regex_predictor))
    ner_m = role_metrics(predict_all(test, ner_predictor))
    summary = gold_summary(gold)
    test_spans = sum(len(g["spans"]) for g in test)
    not_numeric = sum(1 for g in test for sp in g["spans"] if sp["label"] == NOT_NUMERIC)
    lines = [
        "# NER vs regex baseline: span-level role metrics on the gold test set",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} by `edgar-extract ner evaluate`.",
        "",
        "## Data",
        "",
        f"- Gold labels: {summary['sentences']} sentences, {summary['spans']} spans, labeled by "
        f"{', '.join(sorted({g['labeler'] for g in gold}))} with `edgar-extract label`"
        + (
            f"; {n_fast} further sentences were excluded as labeled in under "
            f"{MIN_SECONDS_PER_SPAN} s per span (`edgar-extract label --audit`)."
            if n_fast
            else "."
        ),
        f"- Held-out **test** companies ({len(test_cos)}): {', '.join(test_cos)}: "
        f"{len(test)} sentences, {test_spans} spans ({not_numeric} marked not-a-number).",
        f"- Training: {train_report['n_silver_sentences']} silver sentences "
        f"({train_report['n_silver_spans']} labeled spans) + "
        f"{train_report['n_gold_train_sentences']} gold training sentences "
        f"({train_report['n_gold_train_spans']} spans) x{cfg['gold_weight']} weight. "
        "Test companies are excluded from silver training data too.",
        f"- Model: `{cfg['base_model']}`, {cfg['epochs']} epochs, lr {cfg['learning_rate']}, "
        f"batch {cfg['batch_size']}, trained on `{train_report['device']}`. Hyperparameters "
        "were fixed before evaluation; nothing was tuned on the test set.",
        "- Both extractors are scored on the same candidate spans. The regex roles are mapped "
        "to the NER label set (RD_EXPENSE and LITIGATION_RESERVE count as OTHER_NUMERIC).",
        f"- Rows with gold n < {SMALL_N} are flagged: their metrics are too noisy to rank the "
        "extractors.",
        "",
        "## Results",
        "",
        *metrics_table("Regex baseline (regex_v1)", regex_m),
        *metrics_table("DistilBERT token classifier (ner_distilbert_v1)", ner_m),
        "## Caveats",
        "",
        "- Only spans found by the shared candidate detector are evaluated; numbers the "
        "detector misses are invisible to both extractors and to this table.",
        "- The gold sample is keyword-stratified, not a random sample of all sentences, so "
        "these are not population-weighted accuracy estimates.",
        "- Silver labels contain no GUIDANCE or RESTATED_PRIOR examples, so the model learns "
        "those roles only from the gold training split.",
    ]
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines) + "\n")
    write_model_card(paths["card"], train_report, regex_m, ner_m, len(test), test_spans)
    return paths["report"]


def write_model_card(
    path: Path, train_report: dict[str, Any], regex_m: Any, ner_m: Any, n_test: int, n_spans: int
) -> None:
    cfg = train_report["config"]
    lines = [
        "# Model card: ner_distilbert_v1",
        "",
        "**Task.** Assign a role (REVENUE, GUIDANCE, ONE_TIME_CHARGE, RESTATED_PRIOR, "
        "OTHER_NUMERIC, or O for not-a-number) to numeric spans in 10-K Item 1A / Item 7 "
        "prose. Spans come from `numeric_spans.py`; the model labels tokens.",
        "",
        f"**Base model.** `{cfg['base_model']}` fine-tuned for token classification "
        f"({cfg['epochs']} epochs, lr {cfg['learning_rate']}, max length {cfg['max_length']}, "
        f"seed {cfg['seed']}) on `{train_report['device']}`.",
        "",
        "**Training data.**",
        f"- Silver: {train_report['n_silver_sentences']} sentences / "
        f"{train_report['n_silver_spans']} spans labeled by exact XBRL value matches (<=0.1% "
        "rounding) and inline-XBRL tags in the same filing.",
        f"- Gold (train split): {train_report['n_gold_train_sentences']} sentences / "
        f"{train_report['n_gold_train_spans']} spans, labeled by a person.",
        f"- Train companies: {', '.join(train_report['train_companies'])}.",
        "",
        f"**Evaluation.** Gold test split, held out by company "
        f"({', '.join(train_report['test_companies'])}): {n_test} sentences, {n_spans} spans. "
        "Full table in `reports/ner_metrics.md`.",
        "",
        "| role | gold n | regex F1 | NER F1 |",
        "|---|---|---|---|",
        *[
            f"| {r.role} | {r.support} | {'n/a' if r.f1 is None else f'{r.f1:.2f}'} | "
            f"{'n/a' if n.f1 is None else f'{n.f1:.2f}'} |"
            for r, n in zip(regex_m, ner_m, strict=True)
        ],
        "",
        "**Known failure modes and limits.**",
        "- Small gold set: per-role metrics on fewer than 30 test spans are noise.",
        "- Silver labels are biased toward numbers that also appear in XBRL facts "
        "(financial-statement amounts) and contain no GUIDANCE or RESTATED_PRIOR examples.",
        "- Boundaries are inherited from the regex candidate detector; the model cannot "
        "recover a number the detector misses.",
        "- Trained on 20 large-cap US filers, FY2021-FY2025; other industries, sizes and "
        "years are out of distribution.",
        "",
        "Weights are not committed (size); retrain with `edgar-extract ner train`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
