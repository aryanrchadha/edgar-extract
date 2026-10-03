# SEC response fixtures

Real responses from sec.gov / data.sec.gov, recorded on 2026-10-03 by
`scripts/record_fixtures.py` and trimmed to keep the repo small:

- `www.sec.gov/files/company_tickers.json`: only the entries for the tickers the tests use.
- `data.sec.gov/submissions/*.json`: full top-level metadata. The columnar filing arrays
  keep every 10-K / 10-K/A / 10-KT row, any 8-K12B row, and the first 3 other rows.
  Paginated history files (`*-submissions-001.json`) are trimmed the same way.

- `data.sec.gov/api/xbrl/companyfacts/CIK0000200406.json` (JNJ) and `CIK0000040545.json`
  (GE): companies with known restatements, trimmed to four concepts (Revenues,
  RevenueFromContractWithCustomerExcludingAssessedTax, NetIncomeLoss, Assets) and periods
  ending 2019-01-01 or later. Used by the point-in-time invariant tests.

Values are never edited. Directory layout mirrors the URL (`<host>/<path>`) so the test
session can serve them by URL.

## Filing documents

`www.sec.gov/Archives/.../*.htm.gz` are 20 complete 10-K primary documents (one per
company in the default universe), stored gzip-compressed exactly as cached. They are listed
in `tests/fixtures/filings.json`. `tests/fixtures/section_expectations.json` holds the
reviewed section boundaries (start/end heading, first/last 60 characters) written by
`scripts/freeze_section_expectations.py` after the boundaries were read and checked by hand.
