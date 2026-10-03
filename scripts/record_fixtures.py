"""Record small, trimmed SEC responses into tests/fixtures/sec/ for offline tests.

Run from the repo root (uses the disk cache; hits SEC only for URLs not cached yet):

    uv run python scripts/record_fixtures.py

Trimming keeps the real response structure and values but drops rows the tests do not
need: submissions arrays keep every 10-K-family row, any 8-K12B, and the first few
other rows. Nothing is edited or synthesized.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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


if __name__ == "__main__":
    main()
