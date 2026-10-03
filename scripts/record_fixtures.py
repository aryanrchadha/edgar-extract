"""Record small, trimmed SEC responses into tests/fixtures/sec/ for offline tests.

Run from the repo root (uses the disk cache; hits SEC only for URLs not cached yet):

    uv run python scripts/record_fixtures.py

Trimming keeps the real response structure and values but drops rows the tests do not
need: submissions arrays keep every 10-K-family row, any 8-K12B, and the first few
other rows. Nothing is edited or synthesized.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from edgar_extract.config import load_settings
from edgar_extract.edgar_client import (
    COMPANY_TICKERS_URL,
    EdgarClient,
    submissions_page_url,
    submissions_url,
)
from edgar_extract.universe import ANNUAL_FORMS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "sec"

# CIKs: AAPL, META, WFC, DIS, JNJ, XOM (predecessor), ExxonMobil Holdings (successor)
SUBMISSION_CIKS = [320193, 1326801, 72971, 1744489, 200406, 34088, 2115436]
TICKERS = {"AAPL", "META", "WFC", "DIS", "JNJ", "XOM", "GOOGL", "GOOG", "MSFT"}
KEEP_OTHER_ROWS = 3

# Filing documents copied verbatim (gzip, as cached) for section-isolation tests: one per
# company, chosen to cover every layout the isolator handles.
FILINGS = [
    ("AAPL", 2025, "10-K"),
    ("MSFT", 2023, "10-K"),  # running "Item 7" page headers; combined "Item 1B, 2, 3, 4"
    ("AMZN", 2021, "10-K"),
    ("GOOGL", 2022, "10-K"),  # "ITEM 1A.RISK FACTORS" without a space
    ("META", 2024, "10-K"),
    ("NVDA", 2023, "10-K"),
    ("JPM", 2024, "10-K"),  # Item 7 -> "appears on pages 52-167"
    ("WFC", 2024, "10-K"),  # Items 1A and 7 incorporated by reference (Annual Report)
    ("JNJ", 2022, "10-K"),
    ("PFE", 2025, "10-K"),  # Item 1A ends at Item 1C
    ("XOM", 2023, "10-K"),  # Item 7 -> 'section entitled "..." in the Financial Section'
    ("WMT", 2026, "10-K"),
    ("PG", 2022, "10-K"),
    ("KO", 2021, "10-K"),
    ("INTC", 2024, "10-K"),  # Form 10-K cross-reference index
    ("GE", 2025, "10-K"),  # Form 10-K cross-reference index
    ("BA", 2022, "10-K"),
    ("CAT", 2023, "10-K"),
    ("DIS", 2022, "10-K/A"),  # Part III amendment: no Item 1A / 7
    ("KHC", 2025, "10-K"),
]


def trim_columnar(block: dict[str, Any]) -> dict[str, Any]:
    keys = [k for k, v in block.items() if isinstance(v, list)]
    n = len(block[keys[0]])
    keep = [
        i
        for i in range(n)
        if block["form"][i] in ANNUAL_FORMS or block["form"][i] == "8-K12B" or i < KEEP_OTHER_ROWS
    ]
    return {k: ([v[i] for i in keep] if k in keys else v) for k, v in block.items()}


def write(url: str, payload: Any) -> None:
    parts = urlsplit(url)
    path = OUT / parts.netloc / parts.path.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


def main() -> None:
    settings = load_settings(ROOT / "config.yaml")
    client = EdgarClient.from_settings(settings)

    tickers = client.company_tickers()
    write(COMPANY_TICKERS_URL, {k: v for k, v in tickers.items() if v["ticker"] in TICKERS})

    for cik in SUBMISSION_CIKS:
        sub = client.submissions(cik)
        sub["filings"]["recent"] = trim_columnar(sub["filings"]["recent"])
        write(submissions_url(cik), sub)
        # Record only the pages the universe builder actually needs (cached ones).
        for page in sub["filings"]["files"]:
            url = submissions_page_url(page["name"])
            if client.is_cached(url):
                write(url, trim_columnar(client.submissions_page(page["name"])))

    universe = pd.read_parquet(settings.paths.universe_parquet)
    manifest = []
    for ticker, fy, form in FILINGS:
        row = universe[
            (universe["ticker"] == ticker)
            & (universe["fiscal_year"] == fy)
            & (universe["form"] == form)
        ].iloc[0]
        url = str(row["primary_doc_url"])
        client.get_bytes(url)  # ensure cached
        src = client.cache.path_for(url)
        parts = urlsplit(url)
        dest = OUT / parts.netloc / f"{parts.path.lstrip('/')}.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"copied {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")
        manifest.append(
            {
                "ticker": ticker,
                "cik": int(row["cik"]),
                "fiscal_year": fy,
                "form": form,
                "accession": str(row["accession"]),
                "filing_date": str(row["filing_date"].date()),
                "primary_document": str(row["primary_document"]),
                "url": url,
            }
        )
    (ROOT / "tests" / "fixtures" / "filings.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
