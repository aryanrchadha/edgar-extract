"""Inference with the fine-tuned token classifier (implemented in M4)."""

from __future__ import annotations

import pandas as pd

from edgar_extract.config import Settings


def extract_ner(sections: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    model_dir = settings.paths.models_dir / "ner"
    raise FileNotFoundError(
        f"no trained NER model at {model_dir}. The NER extractor lands in M4, after gold "
        "labels exist; use --extractor regex until then."
    )
