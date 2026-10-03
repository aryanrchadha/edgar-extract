"""Universe construction: ticker -> CIK, optional SIC filter, and the 10-K accession list."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from edgar_extract.config import Settings, normalize_ticker, sic_matches
from edgar_extract.edgar_client import EdgarClient, EdgarError, filing_document_url
from edgar_extract.skiplog import SkipLog

log = logging.getLogger(__name__)

ANNUAL_FORMS = frozenset({"10-K", "10-K405", "10-KT"})
AMENDMENT_FORMS = frozenset({"10-K/A", "10-K405/A", "10-KT/A"})

UNIVERSE_COLUMNS = [
    "ticker",
    "cik",
    "company_name",
    "sic",
    "sic_description",
    "fiscal_year_end",
    "fiscal_year",
    "form",
    "is_amendment",
    "accession",
    "filing_date",
    "acceptance_datetime",
    "period_end",
    "primary_document",
    "primary_doc_url",
    "is_xbrl",
    "is_inline_xbrl",
]


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str


def build_ticker_map(raw: dict[str, Any]) -> dict[str, Company]:
    """Parse SEC ``company_tickers.json`` ({"0": {"cik_str", "ticker", "title"}, ...}).

    When a ticker appears more than once, the first entry wins (SEC orders the file by
    market value), and the collision is logged.
    """
    out: dict[str, Company] = {}
    for row in raw.values():
        t = normalize_ticker(str(row["ticker"]))
        if t in out:
            log.debug("duplicate ticker %s in SEC map; keeping CIK %s", t, out[t].cik)
            continue
        out[t] = Company(ticker=t, cik=int(row["cik_str"]), name=str(row["title"]))
    return out


def resolve_companies(
    settings: Settings, ticker_map: dict[str, Company], skips: SkipLog
) -> list[Company]:
    companies: list[Company] = []
    for t in settings.universe.tickers:
        if t in settings.universe.cik_overrides:
            name = ticker_map[t].name if t in ticker_map else ""
            companies.append(Company(t, settings.universe.cik_overrides[t], name))
        elif t in ticker_map:
            companies.append(ticker_map[t])
        else:
            skips.add("universe", t, "ticker_not_in_sec_map")
    return companies


def fiscal_year_from_period_end(period_end: date) -> int:
    """Label a fiscal year by the calendar year of its period end.

    Exception: a 52/53-week year ending in the first 7 days of January belongs to the
    prior year (e.g. a year ending 2021-01-02 is fiscal 2020). Companies whose fiscal
    year ends in late January or February label years inconsistently (Walmart's year
    ending 2024-01-31 is "fiscal 2024"; Target's ending 2024-02-03 is "fiscal 2023");
    this rule follows the period-end year and is used only as a display label. The PIT
    database keys on ``period_end``, never on this label.
    """
    if period_end.month == 1 and period_end.day <= 7:
        return period_end.year - 1
    return period_end.year


def filings_frame(block: dict[str, Any]) -> pd.DataFrame:
    """Convert a columnar submissions block (``filings.recent`` or a page) to a DataFrame."""
    if not block or "accessionNumber" not in block:
        return pd.DataFrame()
    n = len(block["accessionNumber"])
    cols = {k: v for k, v in block.items() if isinstance(v, list) and len(v) == n}
    return pd.DataFrame(cols)


def all_filings(client: EdgarClient, submissions: dict[str, Any], need_annual: int) -> pd.DataFrame:
    """Recent filings, extended with older pages until ``need_annual`` 10-Ks are found.

    ``filings.recent`` holds at least one year or 1,000 filings, so prolific filers
    (banks issuing 424B notes) need the paginated history to reach five 10-Ks.
    """
    frames = [filings_frame(submissions.get("filings", {}).get("recent", {}))]

    def annual_count() -> int:
        df = pd.concat(frames, ignore_index=True)
        return 0 if df.empty else int(df["form"].isin(ANNUAL_FORMS).sum())

    pages = sorted(
        submissions.get("filings", {}).get("files", []),
        key=lambda f: f.get("filingTo", ""),
        reverse=True,
    )
    for page in pages:
        if annual_count() >= need_annual:
            break
        frames.append(filings_frame(client.submissions_page(page["name"])))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("accessionNumber") if not df.empty else df


def select_annual_filings(
    filings: pd.DataFrame, years: int, ticker: str, skips: SkipLog
) -> pd.DataFrame:
    """Pick original 10-Ks for the latest ``years`` fiscal years, plus their amendments."""
    if filings.empty:
        skips.add("universe", ticker, "no_filings_in_submissions")
        return pd.DataFrame()
    df = filings[filings["form"].isin(ANNUAL_FORMS | AMENDMENT_FORMS)].copy()
    if df.empty:
        skips.add("universe", ticker, "no_10k_filings")
        return df

    df["period_end"] = pd.to_datetime(df["reportDate"], errors="coerce").dt.date
    missing = df["period_end"].isna()
    for acc in df.loc[missing, "accessionNumber"]:
        skips.add("universe", ticker, "missing_report_date", accession=acc)
    df = df[~missing].copy()
    df["fiscal_year"] = df["period_end"].map(fiscal_year_from_period_end)
    df["is_amendment"] = df["form"].isin(AMENDMENT_FORMS)

    originals = df[~df["is_amendment"]].sort_values("filingDate")
    # If two originals cover the same fiscal year (e.g. a 10-KT transition report), keep
    # the latest-filed and log the other.
    dup = originals.duplicated("fiscal_year", keep="last")
    dropped = set(originals.loc[dup, "accessionNumber"])
    for acc in sorted(dropped):
        skips.add("universe", ticker, "duplicate_fiscal_year_original", accession=acc)
    keep_years = set(sorted(originals.loc[~dup, "fiscal_year"], reverse=True)[:years])
    if len(keep_years) < years:
        skips.add(
            "universe",
            ticker,
            "fewer_fiscal_years_than_requested",
            detail=f"found {len(keep_years)} of {years}",
        )
    return df[df["fiscal_year"].isin(keep_years) & ~df["accessionNumber"].isin(dropped)]


def build_universe(
    settings: Settings, client: EdgarClient, skips: SkipLog, *, refresh: bool = False
) -> pd.DataFrame:
    ticker_map = build_ticker_map(client.company_tickers(refresh=refresh))
    rows: list[pd.DataFrame] = []
    for co in resolve_companies(settings, ticker_map, skips):
        try:
            subs = client.submissions(co.cik, refresh=refresh)
        except EdgarError as exc:
            skips.add("universe", co.ticker, "submissions_fetch_failed", detail=str(exc))
            continue
        sic_raw = subs.get("sic")
        sic = int(sic_raw) if str(sic_raw or "").isdigit() else None
        if not sic_matches(sic, settings.universe.sic_include, settings.universe.sic_exclude):
            skips.add("universe", co.ticker, "sic_filtered", detail=f"sic={sic}")
            continue
        try:
            filings = all_filings(client, subs, need_annual=settings.years)
        except EdgarError as exc:
            skips.add("universe", co.ticker, "submissions_page_fetch_failed", detail=str(exc))
            continue
        sel = select_annual_filings(filings, settings.years, co.ticker, skips)
        if sel.empty:
            continue
        out = pd.DataFrame(
            {
                "ticker": co.ticker,
                "cik": co.cik,
                "company_name": subs.get("name") or co.name,
                "sic": sic,
                "sic_description": subs.get("sicDescription", ""),
                "fiscal_year_end": subs.get("fiscalYearEnd", ""),
                "fiscal_year": sel["fiscal_year"].astype(int),
                "form": sel["form"],
                "is_amendment": sel["is_amendment"].astype(bool),
                "accession": sel["accessionNumber"],
                "filing_date": pd.to_datetime(sel["filingDate"]).dt.date,
                "acceptance_datetime": sel.get("acceptanceDateTime", pd.Series(dtype=str)),
                "period_end": sel["period_end"],
                "primary_document": sel["primaryDocument"],
                "primary_doc_url": [
                    filing_document_url(co.cik, a, d)
                    for a, d in zip(sel["accessionNumber"], sel["primaryDocument"], strict=True)
                ],
                "is_xbrl": sel.get("isXBRL", pd.Series(0, index=sel.index)).astype(bool),
                "is_inline_xbrl": sel.get("isInlineXBRL", pd.Series(0, index=sel.index)).astype(
                    bool
                ),
            }
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    uni = pd.concat(rows, ignore_index=True)[UNIVERSE_COLUMNS]
    return uni.sort_values(["ticker", "fiscal_year", "filing_date"]).reset_index(drop=True)
