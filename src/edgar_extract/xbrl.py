"""XBRL company facts: loading, flattening, and normalization to canonical fields.

``companyfacts`` lists every fact a company has reported, once per filing that reported
it: the FY2022 revenue appears in the FY2022 10-K and again as a comparative in the
FY2023 and FY2024 10-Ks, each with its own ``accn`` and ``filed`` date. That repetition is
exactly what point-in-time versioning needs, so it is kept.

Periods are identified by dates. ``fy``/``fp`` describe the *filing*, not the period (the
FY2024 10-K tags FY2022 comparatives ``fy=2024``), so they are carried along but never
used as a key.

Limitation: companyfacts aggregates only non-dimensional (entity-wide) facts. Segment
and other dimensional facts must come from the filing's own inline XBRL.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from edgar_extract.edgar_client import EdgarClient

FACT_COLUMNS = [
    "cik",
    "tag",
    "unit",
    "value",
    "period_start",
    "period_end",
    "duration_days",
    "period_type",
    "fy",
    "fp",
    "form",
    "accession",
    "filed",
    "frame",
]

# Durations SEC itself uses for frames: annual 365 +/- 30 days, quarterly 91 +/- 30.
ANNUAL_DAYS = (335, 395)
QUARTER_DAYS = (61, 121)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    period: Literal["duration", "instant"]
    unit: str
    role: str
    baseline_role: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class TagMap:
    fields: dict[str, FieldSpec]

    def tag_index(self) -> dict[str, tuple[str, int]]:
        """tag -> (canonical field, priority); lower priority number wins."""
        index: dict[str, tuple[str, int]] = {}
        for spec in self.fields.values():
            for prio, tag in enumerate(spec.tags):
                if tag in index:
                    raise ValueError(f"{tag} is mapped to both {index[tag][0]} and {spec.name}")
                index[tag] = (spec.name, prio)
        return index


def load_tag_map(path: Path | None = None) -> TagMap:
    """Load ``tag_map.yaml`` (the packaged copy unless a path is given)."""
    if path is None:
        text = resources.files("edgar_extract").joinpath("tag_map.yaml").read_text("utf-8")
    else:
        text = path.read_text("utf-8")
    raw = yaml.safe_load(text)
    fields = {}
    for name, spec in raw["fields"].items():
        if spec["period"] not in ("duration", "instant"):
            raise ValueError(f"{name}: period must be duration or instant")
        fields[name] = FieldSpec(
            name=name,
            period=spec["period"],
            unit=spec["unit"],
            role=spec["role"],
            baseline_role=spec["baseline_role"],
            tags=tuple(spec["tags"]),
        )
    tag_map = TagMap(fields)
    tag_map.tag_index()  # validates uniqueness
    return tag_map


def period_type(start: pd.Timestamp | None, end: pd.Timestamp) -> str:
    if start is None or pd.isna(start):
        return "instant"
    days = (end - start).days + 1
    if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
        return "annual"
    if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
        return "quarter"
    return "other"


def flatten_company_facts(payload: dict[str, Any]) -> pd.DataFrame:
    """One row per (fact, reporting filing) from a companyfacts JSON payload."""
    cik = int(payload["cik"])
    rows: list[dict[str, Any]] = []
    for taxonomy, concepts in payload.get("facts", {}).items():
        for concept, body in concepts.items():
            tag = f"{taxonomy}:{concept}"
            for unit, facts in body.get("units", {}).items():
                for f in facts:
                    rows.append(
                        {
                            "cik": cik,
                            "tag": tag,
                            "unit": unit,
                            "value": f.get("val"),
                            "period_start": f.get("start"),
                            "period_end": f.get("end"),
                            "fy": f.get("fy"),
                            "fp": f.get("fp"),
                            "form": f.get("form"),
                            "accession": f.get("accn"),
                            "filed": f.get("filed"),
                            "frame": f.get("frame"),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["duration_days"] = (df["period_end"] - df["period_start"]).dt.days + 1
    df["period_type"] = [
        period_type(s, e) for s, e in zip(df["period_start"], df["period_end"], strict=True)
    ]
    df["fy"] = df["fy"].astype("Int64")
    df["duration_days"] = df["duration_days"].astype("Int64")
    return df[FACT_COLUMNS]


def load_company_facts(client: EdgarClient, cik: int) -> pd.DataFrame:
    return flatten_company_facts(client.company_facts(cik))


def canonicalize(facts: pd.DataFrame, tag_map: TagMap) -> pd.DataFrame:
    """Map facts to canonical fields, keeping one tag per field, period and filing.

    When a filing reports several mapped tags for the same field and period, the
    highest-priority tag in ``tag_map.yaml`` is kept. Facts whose unit or period kind does
    not match the field spec are dropped (e.g. a USD/shares value under a USD field).
    """
    index = tag_map.tag_index()
    mapped = facts[facts["tag"].isin(index)].copy()
    if mapped.empty:
        return mapped.assign(
            canonical_field=pd.Series(dtype="string"), tag_priority=pd.Series(dtype="int64")
        )
    mapped["canonical_field"] = mapped["tag"].map(lambda t: index[t][0])
    mapped["tag_priority"] = mapped["tag"].map(lambda t: index[t][1])
    spec_unit = mapped["canonical_field"].map(lambda f: tag_map.fields[f].unit)
    spec_period = mapped["canonical_field"].map(lambda f: tag_map.fields[f].period)
    is_instant = mapped["period_type"] == "instant"
    keep = (mapped["unit"] == spec_unit) & ((spec_period == "instant") == is_instant)
    mapped = mapped[keep]
    key = ["cik", "canonical_field", "period_start", "period_end", "accession"]
    mapped = mapped.sort_values([*key, "tag_priority"], na_position="first")
    # dropna=False keeps instants (period_start is NaT) as their own groups.
    best = mapped.groupby(key, dropna=False, sort=False).head(1)
    return best.reset_index(drop=True)
