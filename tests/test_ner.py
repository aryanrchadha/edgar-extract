"""NER training/inference/evaluation logic with a tiny, randomly initialized model (offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edgar_extract.ner.evaluate import predict_all, regex_predictor, role_metrics, wilson
from edgar_extract.ner.labeling import NOT_NUMERIC
from edgar_extract.ner.train import (
    IGNORE,
    LABEL2ID,
    LABELS,
    Example,
    TrainConfig,
    build_training_set,
    company_split,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

CORPUS = [
    "Revenue was $5.0 billion in 2024.",
    "Net sales were $3.0 billion in 2023.",
    "We recorded impairment charges of $1.0 billion.",
    "Cash was $9.0 billion at year end.",
    "We expect capital expenditures of $2.0 billion next year.",
]


@pytest.fixture(scope="module")
def tiny() -> tuple[Any, Any]:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import (
        DistilBertConfig,
        DistilBertForTokenClassification,
        PreTrainedTokenizerFast,
    )

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=200, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tok.train_from_iterator(CORPUS * 20, trainer)
    from tokenizers.processors import TemplateProcessing

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )  # fmt: skip
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=fast.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
        max_position_embeddings=128, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
    )  # fmt: skip
    return fast, DistilBertForTokenClassification(config)


def ex(text: str, raw: str, label: str | None, ticker: str = "AAA") -> Example:
    s = text.index(raw)
    return Example(f"{ticker}:{text}", ticker, text, [(s, s + len(raw), label)], "gold")


def test_encode_labels(tiny: tuple[Any, Any]) -> None:
    from edgar_extract.ner.train import encode

    tok, _ = tiny
    e = Example(
        "x", "AAA", "Revenue was $5.0 billion and cash was $9.0 billion.",
        [(12, 24, "REVENUE"), (38, 50, None)], "silver",
    )  # fmt: skip
    (w,) = encode(tok, e, max_length=64, stride=8)
    assert w["labels"][0] == IGNORE and w["labels"][-1] == IGNORE  # [CLS], [SEP]
    enc = tok(e.text, return_offsets_mapping=True)
    for (a, b), lab in zip(enc["offset_mapping"], w["labels"], strict=True):
        if a == b:
            continue
        if 12 <= a < 24:
            assert lab == LABEL2ID["REVENUE"]
        elif 38 <= a < 50:
            assert lab == IGNORE  # unlabeled silver span is ignored by the loss
        else:
            assert lab == LABEL2ID["O"]


def test_long_sentence_is_windowed(tiny: tuple[Any, Any]) -> None:
    from edgar_extract.ner.train import encode

    tok, _ = tiny
    text = "Cash was $9.0 billion at year end. " * 20 + "Revenue was $5.0 billion in 2024."
    e = ex(text, "$5.0 billion", "REVENUE")
    windows = encode(tok, e, max_length=32, stride=8)
    assert len(windows) > 1
    assert any(LABEL2ID["REVENUE"] in w["labels"] for w in windows)


def test_company_split_is_whole_companies_and_deterministic() -> None:
    gold = [{"ticker": t} for t in ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10]
    train, test = company_split(gold, 0.3, 13)
    assert company_split(gold, 0.3, 13) == (train, test)
    assert set(train).isdisjoint(test) and set(train) | set(test) == {"A", "B", "C", "D"}
    share = sum(1 for g in gold if g["ticker"] in test) / len(gold)
    assert 0.3 <= share <= 0.5


def test_test_companies_excluded_from_silver_and_gold_training() -> None:
    silver = [
        {"sentence_id": f"s{i}", "ticker": t, "sentence": "Revenue was $5.0 billion.",
         "spans": [{"start": 12, "end": 24, "label": "REVENUE"}]}
        for i, t in enumerate(["AAA", "BBB", "TEST"])
    ]  # fmt: skip
    gold = [
        {"sentence_id": f"g{i}", "ticker": t, "sentence": "Cash was $9.0 billion.",
         "spans": [{"start": 9, "end": 21, "label": NOT_NUMERIC}]}
        for i, t in enumerate(["AAA", "TEST"])
    ]  # fmt: skip
    cfg = TrainConfig(gold_weight=2)
    out = build_training_set(silver, gold, ["TEST"], cfg)
    assert all(e.ticker != "TEST" for e in out)
    golds = [e for e in out if e.source == "gold"]
    assert len(golds) == 2 and golds[0].spans[0][2] == "O"  # not-a-number -> O


def test_train_and_predict_tiny_model(tiny: tuple[Any, Any], tmp_path: Path) -> None:
    from edgar_extract.ner.infer import NerModel
    from edgar_extract.ner.train import train

    tok, model = tiny
    data = [
        ex("Revenue was $5.0 billion in 2024.", "$5.0 billion", "REVENUE"),
        ex("Net sales were $3.0 billion in 2023.", "$3.0 billion", "REVENUE"),
        ex("Cash was $9.0 billion at year end.", "$9.0 billion", "OTHER_NUMERIC"),
    ] * 30
    cfg = TrainConfig(epochs=8, batch_size=8, learning_rate=5e-3, warmup_ratio=0.0)
    losses = train(data, cfg, tmp_path / "m", tokenizer=tok, model=model, device="cpu")
    assert losses[-1] < losses[0]
    loaded = NerModel.load(tmp_path / "m", device="cpu")
    preds = loaded.predict("Revenue was $5.0 billion in 2024.", [(12, 24)])
    assert preds[0][0] in LABELS and 0.0 <= preds[0][1] <= 1.0
    assert loaded.predict("Revenue was $5.0 billion.", []) == []
    assert (tmp_path / "m" / "train_config.json").is_file()


def test_missing_model_dir_is_a_clear_error(tmp_path: Path) -> None:
    from edgar_extract.ner.infer import NerModel

    with pytest.raises(FileNotFoundError, match="ner train"):
        NerModel.load(tmp_path)


def test_wilson_and_role_metrics() -> None:
    lo, hi = wilson(8, 10)  # type: ignore[misc]
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95
    assert wilson(0, 0) is None
    pairs = [("REVENUE", "REVENUE"), ("REVENUE", "OTHER_NUMERIC"), ("O", "REVENUE"),
             ("OTHER_NUMERIC", "OTHER_NUMERIC")]  # fmt: skip
    m = {r.role: r for r in role_metrics(pairs)}
    assert (m["REVENUE"].tp, m["REVENUE"].predicted, m["REVENUE"].support) == (1, 2, 2)
    assert m["REVENUE"].precision == 0.5 and m["REVENUE"].recall == 0.5
    assert m["GUIDANCE"].precision is None and m["GUIDANCE"].support == 0


def test_regex_predictor_maps_to_ner_labels() -> None:
    gold = [
        {
            "sentence": "Research and development expense was $2.0 billion while revenue was "
            "$9.0 billion.",
            "spans": [{"start": 37, "end": 49, "label": "OTHER_NUMERIC"},
                      {"start": 68, "end": 80, "label": "REVENUE"}],
        }
    ]  # fmt: skip
    assert predict_all(gold, regex_predictor) == [
        ("OTHER_NUMERIC", "OTHER_NUMERIC"),  # RD_EXPENSE maps to OTHER_NUMERIC
        ("REVENUE", "REVENUE"),
    ]


def test_evaluation_refuses_without_gold(tmp_path: Path) -> None:
    import yaml

    from edgar_extract.config import load_settings
    from edgar_extract.ner.run import load_gold

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"sec": {"contact_email": "a@b.edu"}, "universe": {"tickers": ["X"]}})
    )
    with pytest.raises(RuntimeError, match="never computed on silver"):
        load_gold(load_settings(cfg))


def test_evaluation_excludes_fast_gold(tmp_path: Path) -> None:
    import yaml

    from edgar_extract.config import load_settings
    from edgar_extract.ner.labeling import write_jsonl
    from edgar_extract.ner.run import load_gold

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"sec": {"contact_email": "a@b.edu"}, "universe": {"tickers": ["X"]}})
    )
    span = {"start": 0, "end": 1, "raw_text": "$1", "label": "OTHER_NUMERIC"}
    rows = [
        {"sentence_id": f"S{i}", "ticker": "X", "sentence": "$1", "spans": [span],
         "labeled_at": "", "seconds": 5.0 if i < 10 else 0.2}
        for i in range(60)
    ]  # fmt: skip
    write_jsonl(tmp_path / "labels" / "gold_labels.jsonl", rows)
    with pytest.raises(RuntimeError, match=r"only 10 usable gold sentences.*50 more were labeled"):
        load_gold(load_settings(cfg))
