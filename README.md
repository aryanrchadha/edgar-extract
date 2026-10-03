# edgar-extract

A pipeline that downloads 10-K filings from SEC EDGAR, isolates MD&A (Item 7) and Risk
Factors (Item 1A), extracts numeric entities from the prose, reconciles them against XBRL,
and stores the result in a **point-in-time (PIT) fundamentals database**: a query "as of
date D" returns only what was knowable on D.

> Status: **M1-M3 complete** (SEC client, universe builder, section isolation, regex
> baseline extractor). Later stages
> print "not implemented yet" instead of producing output. See [PLAN.md](PLAN.md) for the
> design, risks and milestone status.

## Architecture

```mermaid
flowchart LR
  cfg[config.yaml] --> conf[config.py]
  conf --> client[edgar_client.py<br/>User-Agent, &lt;8 req/s, disk cache]
  client --> uni[universe.py<br/>ticker→CIK, SIC filter, 10-K list]
  uni --> sec[sections.py<br/>Item 1A / Item 7]
  sec --> spans[numeric_spans.py]
  spans --> rx[baseline_extract.py]
  spans --> ner[ner/infer.py]
  client --> xb[xbrl.py<br/>companyfacts + tag_map.yaml]
  rx --> rec[reconcile.py]
  ner --> rec
  xb --> rec
  rec --> pit[(pit_db.py<br/>DuckDB, append-only)]
  xb --> pit
  rec --> qa[qa.py]
  pit --> qa
  qa --> dash[dashboard/app.py]
  pit --> ex[examples/pit_vs_naive.py]
```

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Configure

Edit `config.yaml`:

- `sec.contact_email`: required. SEC's fair-access policy asks automated clients to declare
  themselves; every request sends `User-Agent: <app_name> <contact_email>`. It can also be set
  with `EDGAR_EXTRACT_SEC__CONTACT_EMAIL`.
- `universe.tickers`: the companies to process. `universe.cik_overrides` pins a ticker to a
  CIK when SEC's ticker file is wrong for historical work (see XOM below).
- `universe.sic_include` / `sic_exclude`: optional SIC filter, as codes (`"3571"`) or
  ranges (`"6000-6999"`).
- `years`: number of most recent fiscal years of 10-K filings per company (default 5).

## Usage

```bash
uv run edgar-extract run --config config.yaml --stage universe
```

This resolves tickers to CIKs, applies the SIC filter, selects the most recent five fiscal
years of original 10-K filings (plus any 10-K/A for those years, flagged as amendments),
prints the table, and writes `data/interim/universe.parquet`. Add `--offline` to run
purely from cache, or `--refresh` to refetch the mutable JSON endpoints.

```bash
uv run edgar-extract run --config config.yaml --stage sections
```

This downloads each filing's primary document (cached), isolates Item 1A and Item 7, writes
`data/interim/sections.parquet` (clean text with paragraph offsets and lineage columns) and
`reports/section_report.csv` (character counts per section per filing, outliers flagged).
`--stage all` runs every implemented stage in order.

```bash
uv run edgar-extract run --stage extract              # whole universe -> extractions.parquet
uv run edgar-extract extract --ticker AAPL --years 5  # one company -> extractions_AAPL_regex.parquet
```

Stage commands that land in later milestones (`asof`, `dashboard`, `--extractor ner`, and
`run --stage xbrl|reconcile|pit|qa`) currently exit non-zero with a message saying which
milestone implements them.

## SEC access

- Endpoints (checked against sec.gov documentation on 2026-10-03):
  `data.sec.gov/submissions/CIK##########.json` (plus paginated older history),
  `data.sec.gov/api/xbrl/companyfacts/CIK##########.json`,
  `www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>`, and
  `www.sec.gov/files/company_tickers.json`.
- **Rate limit:** one process-wide limiter spaces requests evenly (default 7/s; the config
  rejects anything above 8). SEC's ceiling is 10/s.
- **Cache:** every response body is stored gzip-compressed under `data/raw/<host>/<path>.gz`,
  with a `.meta.json` that records when it was fetched. Reruns never re-download.
- **Errors:** 429 and 5xx get exponential backoff that honors `Retry-After`. 403 aborts
  without retrying, because it usually means a missing User-Agent or a block. 404 is logged
  as a skip.

## Section isolation (Item 1A, Item 7)

`sections.py` turns the filing HTML into paragraph and table-row blocks, with hidden
inline-XBRL content removed, typography normalized (`&nbsp;`, unicode dashes and quotes) and
page breaks tracked. It strips running headers and footers, then applies these rules:

1. An **Item heading** is a short block starting with `Item <n>[A-D]` in any punctuation
   style (`Item 7.`, `ITEM 7 -`, `ITEM 1A.RISK FACTORS`, a `<td>Item 7.</td><td>MD&A</td>`
   row, a heading split across `<b>` tags). `Item 7 of this report...` is a cross-reference,
   not a heading.
2. A section runs from its heading to the next heading of a **later** item (`7A` or `8`
   for Item 7; `1B`, `1C` or `2` for Item 1A). Among all candidate headings, **the longest
   span wins**, which discards table-of-contents hits.
3. **Short spans are stubs** and are classified by what they say:
   - incorporation by reference to the Annual Report / Exhibit 13 gives
     `section_missing_by_reference`, and no text is guessed (WFC, all years);
   - a page reference into the same document (JPM: "appears on pages 52-167", or a
     Form 10-K cross-reference index row as at GE and INTC) is resolved through the printed
     page numbers in the running footers (`method=page_reference`);
   - a named-section reference (XOM: 'the section entitled "Management's Discussion..." in
     the Financial Section of this report') is resolved to the pages whose running header
     carries that title (`method=named_section_reference`).

Results on the 102 filings in the default universe (FY2021-FY2025):

| item | ok (heading) | ok (page ref.) | ok (named ref.) | by reference | not found |
|---|---|---|---|---|---|
| 1A | 85 | 10 | 0 | 5 (WFC) | 2 (DIS 10-K/A, Part III only) |
| 7 | 75 | 15 | 5 (XOM) | 5 (WFC) | 2 (DIS 10-K/A, Part III only) |

Every non-ok section is also in the skip log with its reason.

## Numeric extraction (regex baseline)

`numeric_spans.py` splits each prose paragraph into sentences (pysbd, offsets located
left to right) and finds numeric entities: dollar amounts with scale words (`$1.2 billion`,
`$845 million`, `$(3.1) million`), per-share amounts, percentages (`12.4%`, `(2)%`),
percentage and basis points, scaled counts, parenthesized decimals, and ranges
(`$1.2 to $1.4 billion`, where the scale on either end applies to both). Each entity gets a
kind: `level`, `delta` ("increased $142 million"), `pct`, `pct_change` ("increased 12%"),
`per_share`, `count`, `points` or `number`. Values are normalized to base units with the
scale kept explicit (`value_usd=1.2e9, unit_scale=1e9`). Years, note and item numbers,
dates, and accounting-standard codes are not extracted.

Table rows are not prose and are skipped. A block counts as prose when it has 5 or more
alphabetic words, letters make up at least half of it, and it is not a line of figures
without sentence punctuation.

`baseline_extract.py` assigns a role by keyword rules, documented in the module, and
records the deciding cue in `role_cue` for auditing. The order is:
1. `GUIDANCE` for a company forward-looking sentence ("we expect", "is expected to",
   "our ... outlook");
2. `RESTATED_PRIOR` for restatement language ("as restated", "have been recast"), but not
   "amended and restated";
3. otherwise the nearest keyword to the left: `LITIGATION_RESERVE`, `ONE_TIME_CHARGE`,
   `RD_EXPENSE` or `REVENUE`;
4. else `OTHER_NUMERIC`.

`extractions.parquet` has one row per numeric entity. Columns:
- the spec columns: `ticker, accession, section, sentence, span_start, span_end, raw_text,
  value_usd, unit_scale, role, extractor`;
- lineage: `primary_doc_url`, `sentence_start/sentence_end`, `filing_date`;
- `value`, `unit`, `numeric_kind`, `is_negative` and range fields.

All offsets index into the section text in `sections.parquet`, and a test checks that
`text[span_start:span_end] == raw_text` for every row.

Baseline on the default universe: **18,087 entities from 95 filings**. That count is only
how often each rule fired, not accuracy; accuracy is measured against gold labels in M4.

| role | Item 1A | Item 7 |
|---|---|---|
| REVENUE | 10 | 1,678 |
| GUIDANCE | 26 | 516 |
| ONE_TIME_CHARGE | 35 | 808 |
| LITIGATION_RESERVE | 1 | 81 |
| RD_EXPENSE | 12 | 100 |
| RESTATED_PRIOR | 0 | 3 |
| OTHER_NUMERIC | 477 | 14,340 |

## Conventions worth knowing

- **Fiscal year** = year of the 10-K's period end, minus one when the period ends in the
  first 7 days of January (52/53-week years). Some companies name their years differently.
- **Every skip is logged** to `data/interim/skipped.jsonl` with stage and reason. Nothing is
  dropped silently.
- **XOM:** on 2026-07-01 Exxon Mobil reorganized under a new holding company (CIK 2115436),
  and SEC's ticker file now maps XOM there. Its 10-K history is under the predecessor CIK
  34088, so `config.yaml` pins it with `cik_overrides`. When the universe builder meets this
  pattern (a registrant with no 10-Ks whose history starts with an 8-K12B), the skip-log
  entry says so.

## Development

```bash
uv run pytest           # offline: sockets are blocked in tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Test fixtures under `tests/fixtures/sec/` are trimmed real SEC responses, recorded by
`scripts/record_fixtures.py`.

## Known limitations

Collected here as milestones land.

- **Item 7 / 1A incorporated by reference (WFC):** these sections are flagged
  `section_missing_by_reference` and contribute no text. The Exhibit 13 annual report is
  not parsed as a substitute.
- **Page-referenced sections are page-granular.** For JPM, GE, INTC and XOM the section is
  the set of referenced pages, trimmed to the item's own title on the first page and at the
  next item title on the last page. If the last page's other content has no recognizable
  item title, up to one page of it remains: INTC Item 1A ends with "Other Key Information"
  text, and GE Item 7 ends with the Item 5 share-repurchase table.
- Section text includes table rows (marked in `block_kinds`) so offsets stay exact. The
  extractor reads prose only. Some companies (e.g. AAPL) state revenue almost entirely in
  MD&A tables, so their prose yields few REVENUE entities. That is a property of the
  filings, not a parsing failure, and it limits text-vs-XBRL coverage for those companies.
- Non-dollar amounts (EUR, GBP, JPY, CHF) are detected and labeled `other_currency` but
  never converted, since the pipeline has no FX rates.

- Fiscal-year labels follow the period-end convention above, not each company's own naming.
- The PIT database is keyed by CIK. After a holding-company reorganization (XOM, 2026),
  filings after the reorganization arrive under the successor CIK.
