"""Fine-tune a DistilBERT token classifier that tags numeric entities with a role.

Label scheme (token level): ``O`` plus the five roles (``REVENUE``, ``GUIDANCE``,
``ONE_TIME_CHARGE``, ``RESTATED_PRIOR``, ``OTHER_NUMERIC``). Entity boundaries come from the
candidate detector in :mod:`edgar_extract.numeric_spans`, so the model's job is the role,
and the regex and NER extractors are compared on the same spans.

* tokens inside a labeled span get that span's role,
* tokens inside an **unlabeled** silver span get ``-100`` (ignored by the loss),
* tokens outside spans get ``O``, and gold spans marked "not a number" get ``O``.

Training data is silver sentences plus the gold *training* split. Gold test companies
(30% of gold sentences, chosen by company) are excluded from **both** gold and silver
training data, so no test company is seen in training. Hyperparameters are fixed in
:class:`TrainConfig` before any evaluation; nothing is tuned on the test set.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from edgar_extract.ner.labeling import NER_LABELS, NOT_NUMERIC

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

LABELS: tuple[str, ...] = ("O", *NER_LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
IGNORE = -100


@dataclass
class TrainConfig:
    base_model: str = "distilbert-base-cased"
    max_length: int = 256
    stride: int = 64
    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gold_weight: int = 3  # each gold training sentence is repeated this many times
    test_fraction: float = 0.3
    seed: int = 13
    max_silver_other_only: int = 1500  # cap silver sentences whose only labels are OTHER


@dataclass
class Example:
    sentence_id: str
    ticker: str
    text: str
    spans: list[tuple[int, int, str | None]]  # (start, end, label or None if unknown)
    source: str  # "silver" or "gold"


@dataclass
class TrainReport:
    config: dict[str, Any]
    n_silver_sentences: int
    n_silver_spans: int
    n_gold_train_sentences: int
    n_gold_train_spans: int
    train_companies: list[str]
    test_companies: list[str]
    device: str
    epochs_loss: list[float] = field(default_factory=list)


def gold_to_examples(gold: Sequence[dict[str, Any]]) -> list[Example]:
    out = []
    for g in gold:
        spans = [
            (int(sp["start"]), int(sp["end"]), "O" if sp["label"] == NOT_NUMERIC else sp["label"])
            for sp in g["spans"]
        ]
        out.append(Example(g["sentence_id"], g["ticker"], g["sentence"], spans, "gold"))
    return out


def silver_to_examples(silver: Sequence[dict[str, Any]]) -> list[Example]:
    return [
        Example(
            s["sentence_id"],
            s["ticker"],
            s["sentence"],
            [(int(sp["start"]), int(sp["end"]), sp["label"]) for sp in s["spans"]],
            "silver",
        )
        for s in silver
    ]


def company_split(
    gold: Sequence[dict[str, Any]], test_fraction: float = 0.3, seed: int = 13
) -> tuple[list[str], list[str]]:
    """Hold out whole companies until at least ``test_fraction`` of gold sentences are test.

    Companies are shuffled with a fixed seed; the split depends only on the labeled data,
    never on model results.
    """
    counts: dict[str, int] = {}
    for g in gold:
        counts[g["ticker"]] = counts.get(g["ticker"], 0) + 1
    companies = sorted(counts)
    random.Random(seed).shuffle(companies)
    total = sum(counts.values())
    test: list[str] = []
    taken = 0
    for c in companies:
        if taken >= test_fraction * total:
            break
        test.append(c)
        taken += counts[c]
    train = [c for c in companies if c not in test]
    return sorted(train), sorted(test)


def token_windows(
    tokenizer: PreTrainedTokenizerBase, text: str, max_length: int, stride: int
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Overlapping windows of ``(input_ids, char offsets)``, each wrapped in [CLS] ... [SEP].

    Windowing is done here rather than with ``return_overflowing_tokens`` because that
    option returned only the first two windows for long inputs in transformers 5.x,
    silently dropping the end of long sentences.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    body = max_length - 2
    step = max(body - stride, 1)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("tokenizer needs [CLS] and [SEP] tokens")
    out = []
    start = 0
    while True:
        chunk = slice(start, start + body)
        out.append(([cls_id, *ids[chunk], sep_id], [(0, 0), *offsets[chunk], (0, 0)]))
        if start + body >= len(ids):
            break
        start += step
    return out


def encode(
    tokenizer: PreTrainedTokenizerBase, example: Example, max_length: int, stride: int
) -> list[dict[str, list[int]]]:
    """Tokenize one sentence into one or more overlapping windows with token labels."""
    windows = []
    for ids, offsets in token_windows(tokenizer, example.text, max_length, stride):
        labels = []
        for start, end in offsets:
            if start == end:  # [CLS] / [SEP]
                labels.append(IGNORE)
                continue
            label: int = LABEL2ID["O"]
            for s, e, role in example.spans:
                if start < e and end > s:
                    label = IGNORE if role is None else LABEL2ID[role]
                    break
            labels.append(label)
        windows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return windows


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    import torch

    width = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, value: int) -> torch.Tensor:
        return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "labels": pad("labels", IGNORE),
    }


def build_training_set(
    silver: Sequence[dict[str, Any]],
    gold: Sequence[dict[str, Any]],
    test_companies: Sequence[str],
    cfg: TrainConfig,
) -> list[Example]:
    rng = random.Random(cfg.seed)
    held_out = set(test_companies)
    silver_ex = [e for e in silver_to_examples(silver) if e.ticker not in held_out]
    other_only = [e for e in silver_ex if {r for *_, r in e.spans if r} == {"OTHER_NUMERIC"}]
    informative = [e for e in silver_ex if e not in other_only]
    rng.shuffle(other_only)
    silver_ex = informative + other_only[: cfg.max_silver_other_only]
    gold_ex = [e for e in gold_to_examples(gold) if e.ticker not in held_out]
    examples = silver_ex + gold_ex * cfg.gold_weight
    rng.shuffle(examples)
    return examples


def train(
    examples: Sequence[Example],
    cfg: TrainConfig,
    out_dir: Path,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model: PreTrainedModel | None = None,
    device: str | None = None,
    max_steps: int | None = None,
) -> list[float]:
    """Fine-tune and save the model to ``out_dir``. Returns mean loss per epoch."""
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = device or pick_device()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if model is None:
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.base_model,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id=LABEL2ID,
        )
    windows = [w for ex in examples for w in encode(tokenizer, ex, cfg.max_length, cfg.stride)]
    if not windows:
        raise ValueError("no training examples")
    # transformers 5 types .to() through a decorator mypy cannot see through.
    cast(Any, model).to(device)
    model.train()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = (len(windows) + cfg.batch_size - 1) // cfg.batch_size
    total = steps_per_epoch * cfg.epochs if max_steps is None else max_steps
    sched = cast(Any, get_linear_schedule_with_warmup)(optim, int(cfg.warmup_ratio * total), total)
    pad_id = tokenizer.pad_token_id or 0
    rng = random.Random(cfg.seed)
    losses: list[float] = []
    step = 0
    for epoch in range(cfg.epochs):
        rng.shuffle(windows)
        running, n = 0.0, 0
        for i in range(0, len(windows), cfg.batch_size):
            batch = _collate(windows[i : i + cfg.batch_size], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += float(out.loss.detach().cpu())
            n += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        losses.append(running / max(n, 1))
        log.info("epoch %d mean loss %.4f", epoch + 1, losses[-1])
        if max_steps is not None and step >= max_steps:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return losses
