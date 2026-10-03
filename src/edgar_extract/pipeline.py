"""Pipeline stages that turn cached SEC data into Parquet artifacts.

Each stage reads the previous stage's artifact, writes its own, and records every filing or
section it could not process in the skip log.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from edgar_extract.baseline_extract import EXTRACTOR_NAME, assign_role
from edgar_extract.config import Settings
from edgar_extract.edgar_client import EdgarClient, EdgarError
from edgar_extract.numeric_spans import find_numeric_mentions, iter_sentences
from edgar_extract.sections import Section, SectionStatus, extract_sections
from edgar_extract.skiplog import SkipLog

log = logging.getLogger(__name__)

SECTIONS_STAGE = "sections"
EXTRACT_STAGE = "extract"

SECTION_COLUMNS = [
    "ticker",
    "cik",
    "fiscal_year",
    "form",
    "is_amendment",
    "accession",
    "filing_date",
    "report_date",
    "primary_doc_url",
    "item",
    "status",
    "method",
    "text",
    "n_chars",
    "n_paragraphs",
    "n_table_rows",
    "start_heading",
    "end_heading",
    "pages",
    "detail",
    "block_starts",
    "block_kinds",
]

# Outlier rules for section_report.csv (only sections with status "ok").
OUTLIER_ROBUST_Z = 3.5
OUTLIER_COMPANY_RATIO = 2.0
OUTLIER_MIN_CHARS = 5000


def sections_path(settings: Settings) -> Path:
    return settings.paths.interim_dir / "sections.parquet"


def section_report_path(settings: Settings) -> Path:
    return settings.paths.reports_dir / "section_report.csv"


def load_universe(settings: Settings) -> pd.DataFrame:
    path = settings.paths.universe_parquet
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `edgar-extract run --stage universe` first")
    return pd.read_parquet(path)


def _section_row(filing: pd.Series, sec: Section) -> dict[str, object]:
    kinds = sec.block_kinds
    return {
        "ticker": filing["ticker"],
        "cik": int(filing["cik"]),
        "fiscal_year": int(filing["fiscal_year"]),
        "form": filing["form"],
        "is_amendment": bool(filing["is_amendment"]),
        "accession": filing["accession"],
        "filing_date": filing["filing_date"],
        "report_date": filing["report_date"],
        "primary_doc_url": filing["primary_doc_url"],
        "item": sec.item,
        "status": sec.status.value,
        "method": sec.method.value,
        "text": sec.text,
        "n_chars": sec.n_chars,
        "n_paragraphs": sum(1 for k in kinds if k == "p"),
        "n_table_rows": sum(1 for k in kinds if k == "table_row"),
        "start_heading": sec.start_heading,
        "end_heading": sec.end_heading,
        "pages": ",".join(map(str, sec.pages)),
        "detail": sec.detail,
        "block_starts": sec.block_starts,
        "block_kinds": kinds,
    }


def run_sections_stage(settings: Settings, client: EdgarClient) -> pd.DataFrame:
    """Download (or read from cache) every primary document and isolate Items 1A and 7."""
    skiplog = SkipLog(settings.paths.skip_log)
    skiplog.clear_stage(SECTIONS_STAGE)
    universe = load_universe(settings)
    rows: list[dict[str, object]] = []
    for _, filing in universe.iterrows():
        ticker, cik, accession = str(filing["ticker"]), int(filing["cik"]), str(filing["accession"])
        try:
            html = client.filing_document(cik, accession, str(filing["primary_document"]))
        except EdgarError as exc:
            skiplog.record(
                SECTIONS_STAGE,
                "filing document unavailable",
                ticker=ticker,
                cik=cik,
                accession=accession,
                detail=str(exc),
            )
            continue
        for sec in extract_sections(html).values():
            if sec.status is not SectionStatus.OK:
                detail = sec.detail
                if filing["is_amendment"] and sec.status is SectionStatus.NOT_FOUND:
                    detail = f"{detail}; 10-K/A amendments often contain only Part III"
                skiplog.record(
                    SECTIONS_STAGE,
                    f"Item {sec.item}: {sec.status.value}",
                    ticker=ticker,
                    cik=cik,
                    accession=accession,
                    detail=detail,
                )
            rows.append(_section_row(filing, sec))
        log.info("%s %s: sections isolated", filing["ticker"], filing["accession"])

    df = pd.DataFrame(rows, columns=SECTION_COLUMNS)
    out = sections_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    report = section_report(df)
    report_path = section_report_path(settings)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    return df


def section_report(sections: pd.DataFrame) -> pd.DataFrame:
    """Character counts per section per filing, with outliers flagged.

    A section with status "ok" is flagged when any of these hold:

    * ``n_chars < 5000`` (very short for a large-cap 10-K section),
    * ``|robust z| > 3.5`` on log(n_chars) across all filings for that item
      (median/MAD, scaled by 1.4826),
    * ``n_chars`` is more than 2x above or below the company's own median for that item.
    """
    cols = [c for c in SECTION_COLUMNS if c not in ("text", "block_starts", "block_kinds")]
    report = sections[cols].copy()
    report["outlier"] = False
    report["outlier_reason"] = ""
    ok = report["status"] == SectionStatus.OK.value

    def flag(mask: pd.Series | np.ndarray, reason: str) -> None:
        report.loc[mask, "outlier"] = True
        report.loc[mask, "outlier_reason"] = report.loc[mask, "outlier_reason"].map(
            lambda r: f"{r}; {reason}" if r else reason
        )

    flag(ok & (report["n_chars"] < OUTLIER_MIN_CHARS), f"fewer than {OUTLIER_MIN_CHARS} chars")
    for item, grp in report[ok].groupby("item"):
        logs = grp["n_chars"].clip(lower=1).map(math.log)
        median = logs.median()
        mad = (logs - median).abs().median() * 1.4826
        if mad > 0:
            z = (logs - median) / mad
            flag(
                report.index.isin(z[z.abs() > OUTLIER_ROBUST_Z].index),
                f"robust z > {OUTLIER_ROBUST_Z} for Item {item}",
            )
        company_median = grp.groupby("ticker")["n_chars"].transform("median")
        ratio = grp["n_chars"] / company_median
        off = ratio[(ratio > OUTLIER_COMPANY_RATIO) | (ratio < 1 / OUTLIER_COMPANY_RATIO)]
        flag(
            report.index.isin(off.index), f"more than {OUTLIER_COMPANY_RATIO:g}x off company median"
        )
    return report.sort_values(["ticker", "fiscal_year", "form", "item"]).reset_index(drop=True)


# -- extraction --------------------------------------------------------------------------

EXTRACTION_COLUMNS = [
    "ticker",
    "cik",
    "fiscal_year",
    "form",
    "accession",
    "filing_date",
    "report_date",
    "primary_doc_url",
    "section",
    "sentence",
    "sentence_start",
    "sentence_end",
    "span_start",
    "span_end",
    "raw_text",
    "value",
    "unit",
    "numeric_kind",
    "value_usd",
    "unit_scale",
    "scale_explicit",
    "is_negative",
    "range_id",
    "range_part",
    "role",
    "role_cue",
    "extractor",
    "block_kind",
]


def extractions_path(settings: Settings, name: str = "extractions") -> Path:
    return settings.paths.interim_dir / f"{name}.parquet"


def extract_regex(sections: pd.DataFrame) -> pd.DataFrame:
    """Run the regex baseline over every section with status "ok".

    ``span_start``/``span_end`` and ``sentence_start``/``sentence_end`` are offsets into
    the section text stored in sections.parquet (keyed by accession and item), so every
    value can be traced to the exact characters it came from.
    """
    rows: list[dict[str, object]] = []
    ok = sections[sections["status"] == SectionStatus.OK.value]
    for sec in ok.itertuples(index=False):
        text = str(sec.text)
        starts = [int(x) for x in cast("Sequence[int]", sec.block_starts)]
        kinds = [str(x) for x in cast("Sequence[str]", sec.block_kinds)]
        for sent in iter_sentences(text, starts, kinds):
            for m in find_numeric_mentions(sent.text):
                decision = assign_role(sent.text, m)
                rows.append(
                    {
                        "ticker": sec.ticker,
                        "cik": sec.cik,
                        "fiscal_year": sec.fiscal_year,
                        "form": sec.form,
                        "accession": sec.accession,
                        "filing_date": sec.filing_date,
                        "report_date": sec.report_date,
                        "primary_doc_url": sec.primary_doc_url,
                        "section": sec.item,
                        "sentence": sent.text,
                        "sentence_start": sent.start,
                        "sentence_end": sent.end,
                        "span_start": sent.start + m.start,
                        "span_end": sent.start + m.end,
                        "raw_text": m.raw_text,
                        "value": m.value,
                        "unit": m.unit,
                        "numeric_kind": m.numeric_kind,
                        "value_usd": m.value_usd,
                        "unit_scale": m.unit_scale,
                        "scale_explicit": m.scale_explicit,
                        "is_negative": m.is_negative,
                        "range_id": m.range_id,
                        "range_part": m.range_part,
                        "role": decision.role.value,
                        "role_cue": decision.cue,
                        "extractor": EXTRACTOR_NAME,
                        "block_kind": sent.block_kind,
                    }
                )
    df = pd.DataFrame(rows, columns=EXTRACTION_COLUMNS)
    return df.astype({"range_id": "Int64", "value_usd": "Float64"})


def load_sections(settings: Settings) -> pd.DataFrame:
    path = sections_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `edgar-extract run --stage sections` first")
    return pd.read_parquet(path)


def run_extract_stage(
    settings: Settings,
    *,
    extractor: str = "regex",
    tickers: list[str] | None = None,
    years: int | None = None,
    output_name: str = "extractions",
) -> pd.DataFrame:
    """Extract numeric entities from isolated sections and write ``<output_name>.parquet``."""
    sections = load_sections(settings)
    if tickers:
        sections = sections[sections["ticker"].isin(tickers)]
    if years:
        latest = sections.groupby("ticker")["fiscal_year"].transform("max")
        sections = sections[sections["fiscal_year"] > latest - years]
    if extractor == "regex":
        df = extract_regex(sections)
    elif extractor == "ner":
        from edgar_extract.ner.infer import extract_ner  # heavy imports, load lazily

        df = extract_ner(sections, settings)
    else:
        raise ValueError(f"unknown extractor {extractor!r}; use 'regex' or 'ner'")
    out = extractions_path(settings, output_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
