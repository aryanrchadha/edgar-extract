"""Write tests/fixtures/section_expectations.json from the current isolator output.

Run only after reviewing the boundaries by hand: the expectations encode what a person
checked (start heading, end heading, first and last text of each section), so tests catch
regressions. Last reviewed 2026-10-03: every fixture section's first/last 100 characters
were read against the filing and judged correct (known page-granularity spill noted below).

    uv run python scripts/freeze_section_expectations.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.parse import urlsplit

from edgar_extract.sections import extract_sections

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"

# Reviewed exceptions: page-referenced sections include whole pages, so a section can end
# with up to one page of the next item's content. Kept as-is and documented in README.
NOTES = {
    ("INTC", "1A"): "page-granular: last page also holds 'Other Key Information' content",
    ("GE", "7"): "page-granular: last page also holds the Item 5 share repurchase table",
}


def main() -> None:
    out = []
    for f in json.loads((FIX / "filings.json").read_text()):
        parts = urlsplit(f["url"])
        path = FIX / "sec" / parts.netloc / f"{parts.path.lstrip('/')}.gz"
        sections = extract_sections(gzip.decompress(path.read_bytes()))
        for item, sec in sections.items():
            out.append(
                {
                    "ticker": f["ticker"],
                    "fiscal_year": f["fiscal_year"],
                    "form": f["form"],
                    "url": f["url"],
                    "item": item,
                    "status": sec.status.value,
                    "method": sec.method.value,
                    "start_heading": (sec.start_heading or "")[:40],
                    "end_heading": (sec.end_heading or "")[:40],
                    "first_text": sec.text[:60],
                    "last_text": sec.text[-60:],
                    "n_chars": sec.n_chars,
                    "note": NOTES.get((f["ticker"], item), ""),
                }
            )
    (FIX / "section_expectations.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out)} expectations")


if __name__ == "__main__":
    main()
