"""Inference with the fine-tuned token classifier.

For each candidate span the model's token probabilities are averaged over the span's
tokens (and over overlapping windows for long sentences). The span is ``O`` ("not a
numeric entity") when the mean ``O`` probability exceeds 0.5; otherwise its role is the
most probable non-``O`` label.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from edgar_extract.config import Settings
from edgar_extract.ner.train import LABELS, token_windows

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

EXTRACTOR_NAME = "ner_distilbert_v1"


class NerModel:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str = "cpu",
        max_length: int = 256,
        stride: int = 64,
    ) -> None:
        # transformers 5 types .to() through a decorator mypy cannot see through.
        self.model = cast(Any, model).to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.stride = stride

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> NerModel:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        from edgar_extract.ner.train import pick_device

        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"no trained NER model at {path}; train it with `edgar-extract ner train`"
            )
        model = AutoModelForTokenClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model, tokenizer, device=device or pick_device())

    def predict(self, sentence: str, spans: Sequence[tuple[int, int]]) -> list[tuple[str, float]]:
        """(label, probability) for each candidate span of one sentence."""
        import torch

        if not spans:
            return []
        windows = token_windows(self.tokenizer, sentence, self.max_length, self.stride)
        sums = torch.zeros(len(spans), len(LABELS))
        counts = torch.zeros(len(spans))
        for ids, offsets in windows:
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits
            probs = torch.softmax(logits[0].float(), dim=-1).cpu()
            for t, (a, b) in enumerate(offsets):
                if a == b:
                    continue
                for k, (s, e) in enumerate(spans):
                    if a < e and b > s:
                        sums[k] += probs[t]
                        counts[k] += 1
        out = []
        o_id = LABELS.index("O")
        for k in range(len(spans)):
            if counts[k] == 0:
                out.append(("OTHER_NUMERIC", 0.0))  # span fell outside every window
                continue
            mean = sums[k] / counts[k]
            if float(mean[o_id]) > 0.5:
                out.append(("O", float(mean[o_id])))
                continue
            role_probs = mean.clone()
            role_probs[o_id] = -1
            best = int(role_probs.argmax())
            out.append((LABELS[best], float(mean[best])))
        return out


def model_dir(settings: Settings) -> Path:
    return settings.paths.models_dir / "ner"


def extract_ner(sections: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Same rows and columns as the regex extractor, with roles from the model.

    Spans the model labels ``O`` (not a numeric entity) are dropped from the output.
    """
    from edgar_extract.pipeline import extract_regex

    model = NerModel.load(model_dir(settings))
    base = extract_regex(sections)
    if base.empty:
        return base.assign(extractor=EXTRACTOR_NAME)
    roles = pd.Series(index=base.index, dtype="object")
    cues = pd.Series(index=base.index, dtype="object")
    for _, grp in base.groupby(["accession", "section", "sentence_start"], sort=False):
        first = grp.iloc[0]
        start = int(cast(Any, first["sentence_start"]))
        spans = [
            (int(a) - start, int(b) - start)
            for a, b in zip(grp["span_start"], grp["span_end"], strict=True)
        ]
        for idx, (label, prob) in zip(
            grp.index, model.predict(str(first["sentence"]), spans), strict=True
        ):
            roles[idx] = label
            cues[idx] = f"p={prob:.2f}"
    out = base.assign(role=roles, role_cue=cues, extractor=EXTRACTOR_NAME)
    return out[out["role"] != "O"].reset_index(drop=True)
