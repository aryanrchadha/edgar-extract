# SEC response fixtures

Real responses from sec.gov / data.sec.gov, recorded on 2026-10-03 by
`scripts/record_fixtures.py` and trimmed to keep the repo small:

- `www.sec.gov/files/company_tickers.json`: only the entries for the tickers the tests use.
- `data.sec.gov/submissions/*.json`: full top-level metadata. The columnar filing arrays
  keep every 10-K / 10-K/A / 10-KT row, any 8-K12B row, and the first 3 other rows.
  Paginated history files (`*-submissions-001.json`) are trimmed the same way.

Values are never edited. Directory layout mirrors the URL (`<host>/<path>`) so the test
session can serve them by URL.
