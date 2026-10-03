# SEC fixtures

Files prefixed `synthetic_` are hand-written to mirror the documented structure of SEC
endpoints (columnar `filings.recent`, paginated `filings.files`, `company_tickers.json`).
Their values are made up for unit tests and are NOT real filing data.

Recorded (real) responses are added under `recorded/` once a contact email is configured,
trimmed to the fields the tests need.
