from __future__ import annotations

import json
from datetime import date

import pandas as pd

from edgar_extract.config import Settings
from edgar_extract.edgar_client import EdgarClient
from edgar_extract.skiplog import SkipLog
from edgar_extract.universe import (
    UNIVERSE_COLUMNS,
    build_ticker_map,
    build_universe,
    filings_frame,
    fiscal_year_from_period_end,
    select_annual_filings,
)

from .conftest import SEC


def test_ticker_map_first_entry_wins() -> None:
    m = build_ticker_map(json.loads((SEC / "synthetic_company_tickers.json").read_text()))
    assert m["AAA"].cik == 1111
    assert m["BRK-B"].cik == 2222


def test_fiscal_year_rule() -> None:
    assert fiscal_year_from_period_end(date(2023, 9, 30)) == 2023
    assert fiscal_year_from_period_end(date(2021, 1, 2)) == 2020
    assert fiscal_year_from_period_end(date(2024, 1, 31)) == 2024


def test_select_annual_filings_keeps_amendments_and_latest_years() -> None:
    subs = json.loads((SEC / "synthetic_submissions_alpha.json").read_text())
    df = filings_frame(subs["filings"]["recent"])
    skips = SkipLog()
    sel = select_annual_filings(df, 5, "AAA", skips)
    originals = sel[~sel["is_amendment"]]
    assert sorted(originals["fiscal_year"]) == [2021, 2022, 2023, 2024, 2025]
    assert sel["is_amendment"].sum() == 1  # the FY2024 10-K/A
    assert "10-Q" not in set(sel["form"])
    assert skips.entries == []


def test_universe_end_to_end_with_pagination(
    settings: Settings, offline_client: EdgarClient
) -> None:
    skips = SkipLog()
    uni = build_universe(settings, offline_client, skips)
    assert list(uni.columns) == UNIVERSE_COLUMNS
    bank = uni[uni["ticker"] == "BNK"]
    # Only one 10-K is in `recent`; the other four come from the paginated history file.
    assert sorted(bank["fiscal_year"]) == [2021, 2022, 2023, 2024, 2025]
    assert (
        bank["primary_doc_url"].iloc[0].startswith("https://www.sec.gov/Archives/edgar/data/3333/")
    )
    assert uni["filing_date"].map(type).eq(date).all()
    assert offline_client.stats["network_requests"] == 0


def test_universe_logs_unknown_tickers_and_sic_filter(
    settings: Settings, offline_client: EdgarClient
) -> None:
    settings.universe.tickers = ["AAA", "BNK", "ZZZZ"]
    settings.universe.sic_include = ["6000-6999"]
    skips = SkipLog()
    uni = build_universe(settings, offline_client, skips)
    assert set(uni["ticker"]) == {"BNK"}
    reasons = {(s.ticker, s.reason) for s in skips.entries}
    assert ("ZZZZ", "ticker_not_in_sec_map") in reasons
    assert ("AAA", "sic_filtered") in reasons


def test_fewer_years_than_requested_is_logged() -> None:
    subs = json.loads((SEC / "synthetic_submissions_alpha.json").read_text())
    skips = SkipLog()
    select_annual_filings(filings_frame(subs["filings"]["recent"]), 10, "AAA", skips)
    assert any(s.reason == "fewer_fiscal_years_than_requested" for s in skips.entries)


def test_skiplog_writes_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    skips = SkipLog()
    skips.add("universe", "X", "why")
    skips.write(tmp_path / "s.csv")
    skips.write(tmp_path / "s.csv")
    assert len(pd.read_csv(tmp_path / "s.csv")) == 2
