"""Universe builder tests on trimmed, real SEC responses recorded in tests/fixtures/sec."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from edgar_extract.config import SicFilter, SicRange
from edgar_extract.edgar_client import EdgarClient
from edgar_extract.skiplog import SkipLog
from edgar_extract.universe import (
    UNIVERSE_COLUMNS,
    Company,
    build_universe,
    fiscal_year_from_period_end,
    select_filings,
)

from .conftest import FixtureSession

XOM_OVERRIDE = {"XOM": 34088}


def build(
    client: EdgarClient, tickers: list[str], **kwargs: object
) -> tuple[pd.DataFrame, SkipLog]:
    skiplog = SkipLog()
    params: dict[str, object] = {"years": 5, "sic_filter": SicFilter(), "cik_overrides": {}}
    params.update(kwargs)
    df = build_universe(client, tickers, skiplog=skiplog, **params)  # type: ignore[arg-type]
    return df, skiplog


@pytest.mark.parametrize(
    ("period_end", "fy"),
    [
        (date(2025, 9, 27), 2025),  # AAPL
        (date(2022, 1, 2), 2021),  # JNJ 52/53-week year ending the Sunday after Dec 31
        (date(2025, 1, 26), 2025),  # NVDA, fiscal year named for the year it ends
        (date(2024, 12, 31), 2024),
        (date(2025, 1, 8), 2025),
    ],
)
def test_fiscal_year_convention(period_end: date, fy: int) -> None:
    assert fiscal_year_from_period_end(period_end) == fy


def test_apple_last_five_fiscal_years(client: EdgarClient, session: FixtureSession) -> None:
    df, skiplog = build(client, ["AAPL"])
    assert list(df.columns) == UNIVERSE_COLUMNS
    assert sorted(df["fiscal_year"]) == [2021, 2022, 2023, 2024, 2025]
    assert not df["is_amendment"].any()
    latest = df[df["fiscal_year"] == 2025].iloc[0]
    assert latest["accession"] == "0000320193-25-000079"
    assert latest["filing_date"] == pd.Timestamp("2025-10-31")
    assert latest["report_date"] == pd.Timestamp("2025-09-27")
    assert latest["primary_doc_url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )
    assert latest["is_inline_xbrl"]
    # Recent history already covers 5+ years, so the 1994-2015 page is never requested.
    assert not any("submissions-001" in u for u in session.urls())
    assert skiplog.entries == []


def test_paginated_history_is_fetched_when_needed(
    client: EdgarClient, session: FixtureSession
) -> None:
    df, _ = build(client, ["META", "WFC"])
    assert any("CIK0001326801-submissions-001.json" in u for u in session.urls())
    assert any("CIK0000072971-submissions-001.json" in u for u in session.urls())
    for ticker in ("META", "WFC"):
        years = df[(df["ticker"] == ticker) & ~df["is_amendment"]]["fiscal_year"]
        assert sorted(years) == [2021, 2022, 2023, 2024, 2025], ticker


def test_amendments_kept_but_not_counted(client: EdgarClient) -> None:
    df, _ = build(client, ["DIS"])
    originals = df[~df["is_amendment"]]
    amendments = df[df["is_amendment"]]
    assert sorted(originals["fiscal_year"]) == [2021, 2022, 2023, 2024, 2025]
    assert set(amendments["form"]) == {"10-K/A"}
    assert set(amendments["fiscal_year"]) <= set(originals["fiscal_year"])
    assert len(amendments) == 2


def test_period_ending_in_early_january(client: EdgarClient) -> None:
    df, _ = build(client, ["JNJ"])
    row = df[df["report_date"] == pd.Timestamp("2022-01-02")].iloc[0]
    assert row["fiscal_year"] == 2021


def test_successor_issuer_explained_and_override_works(client: EdgarClient) -> None:
    df, skiplog = build(client, ["XOM"])
    assert df.empty
    (entry,) = skiplog.entries
    assert entry.reason == "no 10-K filings in submissions history"
    assert entry.cik == 2115436
    assert entry.detail is not None and "8-K12B" in entry.detail and "cik_overrides" in entry.detail

    df, skiplog = build(client, ["XOM"], cik_overrides=XOM_OVERRIDE)
    assert sorted(df["fiscal_year"]) == [2021, 2022, 2023, 2024, 2025]
    assert set(df["cik"]) == {34088}
    assert skiplog.entries == []


def test_unknown_ticker_and_share_class_duplicates_are_logged(client: EdgarClient) -> None:
    _, skiplog = build(client, ["GOOGL", "GOOG", "ZZZZ"])
    reasons = {(e.ticker, e.reason.split(" (")[0]) for e in skiplog.entries}
    assert ("ZZZZ", "ticker not found in SEC company_tickers.json") in reasons
    assert ("GOOG", "duplicate CIK") in reasons


def test_sic_filter_excludes_and_logs(client: EdgarClient) -> None:
    banks_only = SicFilter(include=(SicRange(6000, 6999),))
    df, skiplog = build(client, ["AAPL", "WFC"], sic_filter=banks_only)
    assert set(df["ticker"]) == {"WFC"}
    (entry,) = skiplog.entries
    assert entry.ticker == "AAPL" and entry.reason == "excluded by SIC filter"
    assert entry.detail == "SIC 3571 not in sic_include"


# -- selection rules on hand-built rows ---------------------------------------------------

COMPANY = Company("TST", 1, "Test Co", 1000, "", "1231")


def row(
    form: str, acc: str, report: str | None, filed: str, doc: str = "d.htm"
) -> dict[str, object]:
    return {
        "form": form,
        "accessionNumber": acc,
        "reportDate": report,
        "filingDate": filed,
        "primaryDocument": doc,
        "acceptanceDateTime": "",
        "isInlineXBRL": 1,
    }


def test_duplicate_original_keeps_earliest_and_logs() -> None:
    skiplog = SkipLog()
    rows = [
        row("10-K", "0000000001-24-000002", "2023-12-31", "2024-03-01"),
        row("10-K", "0000000001-24-000001", "2023-12-31", "2024-02-01"),
    ]
    (ref,) = select_filings(COMPANY, rows, 1, skiplog)
    assert ref.accession == "0000000001-24-000001"
    assert skiplog.entries[0].accession == "0000000001-24-000002"


def test_unparseable_rows_logged_and_short_history_flagged() -> None:
    skiplog = SkipLog()
    rows = [
        row("10-K", "0000000001-24-000001", "", "2024-02-01"),
        row("10-K", "0000000001-23-000001", "2022-12-31", "2023-02-01", doc=""),
        row("10-K", "0000000001-22-000001", "2021-12-31", "2022-02-01"),
    ]
    refs = select_filings(COMPANY, rows, 5, skiplog)
    assert [r.fiscal_year for r in refs] == [2021]
    reasons = [e.reason for e in skiplog.entries]
    assert "annual report without a parseable reportDate/filingDate" in reasons
    assert "annual report without a primaryDocument" in reasons
    assert "fewer than 5 fiscal years of original 10-K filings available" in reasons


def test_amendment_outside_window_logged() -> None:
    skiplog = SkipLog()
    rows = [
        row("10-K", "0000000001-25-000001", "2024-12-31", "2025-02-01"),
        row("10-K", "0000000001-24-000001", "2023-12-31", "2024-02-01"),
        row("10-K/A", "0000000001-25-000009", "2023-12-31", "2025-04-01"),
    ]
    refs = select_filings(COMPANY, rows, 1, skiplog)
    assert [r.accession for r in refs] == ["0000000001-25-000001"]
    assert skiplog.entries[0].reason == "10-K/A for a fiscal year outside the selected window"
