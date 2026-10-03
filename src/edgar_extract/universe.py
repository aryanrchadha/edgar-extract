"""Universe construction: ticker -> CIK, optional SIC filter, 10-K accession list.

For each company we select the most recent ``years`` fiscal years that have an original
annual report (``10-K`` or ``10-KT``). Amendments for those fiscal years are kept and
flagged with ``is_amendment``; they never count toward the year total.

Fiscal year convention: ``year(reportDate)``, minus one when the period ends in the first
seven days of January (52/53-week years ending on the Saturday nearest Dec 31). This is a
documented convention, not each company's own label.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from edgar_extract.config import SicFilter
from edgar_extract.edgar_client import (
    EdgarClient,
    EdgarError,
    EdgarNotFoundError,
    archive_url,
)
from edgar_extract.skiplog import SkipLog

log = logging.getLogger(__name__)

STAGE = "universe"
ORIGINAL_FORMS = frozenset({"10-K", "10-KT"})
AMENDMENT_FORMS = frozenset({"10-K/A", "10-KT/A"})
ANNUAL_FORMS = ORIGINAL_FORMS | AMENDMENT_FORMS

UNIVERSE_COLUMNS = [
    "ticker",
    "cik",
    "company_name",
    "sic",
    "fiscal_year",
    "form",
    "is_amendment",
    "accession",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_document",
    "primary_doc_url",
    "is_inline_xbrl",
]


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str
    sic: int | None
    sic_description: str
    fiscal_year_end: str | None


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    company_name: str
    sic: int | None
    fiscal_year: int
    form: str
    is_amendment: bool
    accession: str
    filing_date: date
    report_date: date
    acceptance_datetime: str
    primary_document: str
    primary_doc_url: str
    is_inline_xbrl: bool


def fiscal_year_from_period_end(period_end: date) -> int:
    if period_end.month == 1 and period_end.day <= 7:
        return period_end.year - 1
    return period_end.year


def load_ticker_map(client: EdgarClient) -> dict[str, int]:
    """Map upper-case ticker -> CIK from SEC's ``company_tickers.json``."""
    mapping: dict[str, int] = {}
    for row in client.company_tickers().values():
        ticker = str(row["ticker"]).upper()
        # The file lists a CIK once per ticker; keep the first if a ticker repeats.
        mapping.setdefault(ticker, int(row["cik_str"]))
    return mapping


def resolve_ciks(
    tickers: Iterable[str],
    ticker_map: dict[str, int],
    overrides: dict[str, int],
    skiplog: SkipLog,
) -> list[tuple[str, int]]:
    """Resolve tickers to CIKs, de-duplicating share classes that map to one CIK."""
    resolved: list[tuple[str, int]] = []
    seen: dict[int, str] = {}
    for ticker in tickers:
        cik = overrides.get(ticker, ticker_map.get(ticker))
        if cik is None:
            skiplog.record(STAGE, "ticker not found in SEC company_tickers.json", ticker=ticker)
            continue
        if cik in seen:
            skiplog.record(
                STAGE,
                "duplicate CIK (another ticker for the same registrant is already in the universe)",
                ticker=ticker,
                cik=cik,
                detail=f"kept {seen[cik]}",
            )
            continue
        seen[cik] = ticker
        resolved.append((ticker, cik))
    return resolved


def company_from_submissions(ticker: str, cik: int, sub: dict[str, Any]) -> Company:
    sic_raw = str(sub.get("sic") or "").strip()
    return Company(
        ticker=ticker,
        cik=cik,
        name=str(sub.get("name") or ""),
        sic=int(sic_raw) if sic_raw.isdigit() else None,
        sic_description=str(sub.get("sicDescription") or ""),
        fiscal_year_end=sub.get("fiscalYearEnd"),
    )


def successor_hint(sub: dict[str, Any]) -> str | None:
    """Explain a missing 10-K history when the registrant is a successor issuer.

    After a holding-company reorganization the ticker moves to a new CIK whose history
    starts with an 8-K12B; the historical 10-Ks stay under the predecessor's CIK.
    """
    for row in _columnar_rows(sub["filings"]["recent"]):
        if row.get("form") == "8-K12B":
            return (
                f"registrant became a successor issuer (8-K12B filed {row.get('filingDate')}); "
                "its earlier 10-Ks are under the predecessor CIK. "
                "Add the predecessor to universe.cik_overrides."
            )
    return None


def _columnar_rows(block: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Turn SEC's columnar arrays (one list per field) into row dicts."""
    keys = [k for k, v in block.items() if isinstance(v, list)]
    if not keys:
        return
    n = len(block[keys[0]])
    for k in keys:
        if len(block[k]) != n:
            raise EdgarError(f"submissions column {k!r} has {len(block[k])} rows, expected {n}")
    for i in range(n):
        yield {k: block[k][i] for k in keys}


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _distinct_original_years(rows: list[dict[str, Any]]) -> set[int]:
    years: set[int] = set()
    for r in rows:
        if r.get("form") in ORIGINAL_FORMS and (d := _parse_date(r.get("reportDate"))):
            years.add(fiscal_year_from_period_end(d))
    return years


def annual_report_rows(
    client: EdgarClient, sub: dict[str, Any], years: int, skiplog: SkipLog, ticker: str
) -> list[dict[str, Any]]:
    """All 10-K family rows, pulling older paginated history only when needed.

    Pages are fetched in order until we have ``years + 1`` distinct original fiscal years
    (one extra so an amendment for the oldest selected year is not cut off at a page
    boundary) or the pages run out.
    """
    rows = [r for r in _columnar_rows(sub["filings"]["recent"]) if r.get("form") in ANNUAL_FORMS]
    for page in sub["filings"].get("files", []):
        if len(_distinct_original_years(rows)) > years:
            break
        name = page.get("name")
        try:
            data = client.submissions_page(name)
        except EdgarNotFoundError:
            skiplog.record(
                STAGE, "submissions history page missing (HTTP 404)", ticker=ticker, detail=name
            )
            continue
        rows.extend(r for r in _columnar_rows(data) if r.get("form") in ANNUAL_FORMS)
    return rows


def select_filings(
    company: Company, rows: list[dict[str, Any]], years: int, skiplog: SkipLog
) -> list[FilingRef]:
    """Pick the latest ``years`` fiscal years of original 10-Ks plus their amendments."""
    parsed: list[tuple[dict[str, Any], date, date]] = []
    for r in rows:
        acc = r.get("accessionNumber")
        report = _parse_date(r.get("reportDate"))
        filed = _parse_date(r.get("filingDate"))
        if report is None or filed is None:
            skiplog.record(
                STAGE,
                "annual report without a parseable reportDate/filingDate",
                ticker=company.ticker,
                cik=company.cik,
                accession=acc,
                detail=f"form={r.get('form')} reportDate={r.get('reportDate')!r}",
            )
            continue
        if not r.get("primaryDocument"):
            skiplog.record(
                STAGE,
                "annual report without a primaryDocument",
                ticker=company.ticker,
                cik=company.cik,
                accession=acc,
            )
            continue
        parsed.append((r, report, filed))

    originals: dict[int, tuple[dict[str, Any], date, date]] = {}
    for r, report, filed in sorted(parsed, key=lambda t: (t[2], t[0]["accessionNumber"])):
        if r["form"] not in ORIGINAL_FORMS:
            continue
        fy = fiscal_year_from_period_end(report)
        if fy in originals:
            skiplog.record(
                STAGE,
                "second original annual report for the same fiscal year (kept the earliest filed)",
                ticker=company.ticker,
                cik=company.cik,
                accession=r["accessionNumber"],
                detail=f"fiscal_year={fy} kept={originals[fy][0]['accessionNumber']}",
            )
            continue
        originals[fy] = (r, report, filed)

    selected_years = sorted(originals, reverse=True)[:years]
    if len(selected_years) < years:
        skiplog.record(
            STAGE,
            f"fewer than {years} fiscal years of original 10-K filings available",
            ticker=company.ticker,
            cik=company.cik,
            detail=f"found {sorted(selected_years)}",
        )

    chosen: list[tuple[dict[str, Any], date, date, int]] = [
        (*originals[fy], fy) for fy in selected_years
    ]
    oldest_original_filed = min((originals[fy][2] for fy in selected_years), default=None)
    for r, report, filed in parsed:
        if r["form"] not in AMENDMENT_FORMS:
            continue
        fy = fiscal_year_from_period_end(report)
        if fy in selected_years:
            chosen.append((r, report, filed, fy))
        elif oldest_original_filed is not None and filed >= oldest_original_filed:
            skiplog.record(
                STAGE,
                "10-K/A for a fiscal year outside the selected window",
                ticker=company.ticker,
                cik=company.cik,
                accession=r["accessionNumber"],
                detail=f"fiscal_year={fy}",
            )

    refs = [
        FilingRef(
            ticker=company.ticker,
            cik=company.cik,
            company_name=company.name,
            sic=company.sic,
            fiscal_year=fy,
            form=r["form"],
            is_amendment=r["form"] in AMENDMENT_FORMS,
            accession=r["accessionNumber"],
            filing_date=filed,
            report_date=report,
            acceptance_datetime=str(r.get("acceptanceDateTime") or ""),
            primary_document=r["primaryDocument"],
            primary_doc_url=archive_url(company.cik, r["accessionNumber"], r["primaryDocument"]),
            is_inline_xbrl=bool(r.get("isInlineXBRL")),
        )
        for r, report, filed, fy in chosen
    ]
    return sorted(refs, key=lambda f: (f.fiscal_year, f.filing_date, f.accession), reverse=True)


def build_universe(
    client: EdgarClient,
    tickers: Iterable[str],
    *,
    years: int,
    sic_filter: SicFilter,
    cik_overrides: dict[str, int],
    skiplog: SkipLog,
) -> pd.DataFrame:
    """Resolve tickers, apply the SIC filter, and list the 10-K filings to process."""
    ticker_map = load_ticker_map(client)
    refs: list[FilingRef] = []
    for ticker, cik in resolve_ciks(tickers, ticker_map, cik_overrides, skiplog):
        try:
            sub = client.submissions(cik)
        except EdgarNotFoundError:
            skiplog.record(STAGE, "submissions endpoint returned 404", ticker=ticker, cik=cik)
            continue
        company = company_from_submissions(ticker, cik, sub)
        if (reason := sic_filter.reason_rejected(company.sic)) is not None:
            skiplog.record(STAGE, "excluded by SIC filter", ticker=ticker, cik=cik, detail=reason)
            continue
        rows = annual_report_rows(client, sub, years, skiplog, ticker)
        if not rows:
            skiplog.record(
                STAGE,
                "no 10-K filings in submissions history",
                ticker=ticker,
                cik=cik,
                detail=successor_hint(sub),
            )
            continue
        selected = select_filings(company, rows, years, skiplog)
        refs.extend(selected)
        log.info("%s (CIK %s): %d annual filings selected", ticker, cik, len(selected))
    return universe_frame(refs)


def universe_frame(refs: list[FilingRef]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in refs], columns=UNIVERSE_COLUMNS)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df["report_date"] = pd.to_datetime(df["report_date"])
    return df.astype(
        {
            "ticker": "string",
            "cik": "int64",
            "company_name": "string",
            "sic": "Int64",
            "fiscal_year": "int64",
            "form": "string",
            "is_amendment": "bool",
            "accession": "string",
            "filing_date": "datetime64[ns]",
            "report_date": "datetime64[ns]",
            "acceptance_datetime": "string",
            "primary_document": "string",
            "primary_doc_url": "string",
            "is_inline_xbrl": "bool",
        }
    )
