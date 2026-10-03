"""Record of everything the pipeline chose not to process, and why.

Silent drops are bugs: any stage that skips a ticker, filing or section calls
:meth:`SkipLog.record`. Entries are appended to a JSONL file so they survive reruns and
can be inspected by the dashboard.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SkipEntry:
    stage: str
    reason: str
    ticker: str | None = None
    cik: int | None = None
    accession: str | None = None
    detail: str | None = None
    logged_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class SkipLog:
    """In-memory list of skips, optionally mirrored to an append-only JSONL file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.entries: list[SkipEntry] = []

    def record(
        self,
        stage: str,
        reason: str,
        *,
        ticker: str | None = None,
        cik: int | None = None,
        accession: str | None = None,
        detail: str | None = None,
    ) -> SkipEntry:
        entry = SkipEntry(
            stage=stage, reason=reason, ticker=ticker, cik=cik, accession=accession, detail=detail
        )
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def clear_stage(self, stage: str) -> None:
        """Drop earlier entries for ``stage`` (from memory and file) before rerunning it."""
        self.entries = [e for e in self.entries if e.stage != stage]
        if self.path is not None and self.path.is_file():
            kept = [e for e in self.read(self.path) if e.stage != stage]
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for e in kept:
                    fh.write(json.dumps(asdict(e), sort_keys=True) + "\n")
            tmp.replace(self.path)

    def for_stage(self, stage: str) -> list[SkipEntry]:
        return [e for e in self.entries if e.stage == stage]

    def summary(self, stage: str | None = None) -> Counter[str]:
        entries = self.entries if stage is None else self.for_stage(stage)
        return Counter(e.reason for e in entries)

    @staticmethod
    def read(path: Path) -> list[SkipEntry]:
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as fh:
            return [SkipEntry(**json.loads(line)) for line in fh if line.strip()]
