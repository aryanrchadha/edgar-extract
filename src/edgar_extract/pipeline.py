"""Pipeline stages that turn cached SEC data into Parquet artifacts.

Each stage reads the previous stage's artifact, writes its own, and records every filing or
section it could not process in the skip log.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from edgar_extract.config import Settings
from edgar_extract.edgar_client import EdgarClient, EdgarError
from edgar_extract.sections import Section, SectionStatus, extract_sections
from edgar_extract.skiplog import SkipLog

log = logging.getLogger(__name__)

SECTIONS_STAGE = "sections"

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
