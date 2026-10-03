"""Record every skipped company or filing with a reason. Silent drops are bugs."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Skip:
    stage: str
    ticker: str
    reason: str
    accession: str = ""
    detail: str = ""
    logged_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class SkipLog:
    def __init__(self) -> None:
        self.entries: list[Skip] = []

    def add(
        self, stage: str, ticker: str, reason: str, accession: str = "", detail: str = ""
    ) -> None:
        entry = Skip(stage, ticker, reason, accession, detail)
        self.entries.append(entry)
        log.warning("skip [%s] %s %s: %s %s", stage, ticker, accession, reason, detail)

    def to_frame(self) -> pd.DataFrame:
        cols = list(Skip.__dataclass_fields__)
        return pd.DataFrame([asdict(e) for e in self.entries], columns=cols)

    def write(self, path: Path) -> None:
        """Append to a CSV so skips from all stages accumulate in one place."""
        if not self.entries:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, mode="a", header=not path.exists(), index=False)
