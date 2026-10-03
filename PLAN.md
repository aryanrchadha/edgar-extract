# edgar-extract — design plan

This file records how I read the spec, the design choices that follow from it, and
the risks I can see up front. It is updated as milestones land.

## 1. Interpretation

The deliverable is a **point-in-time (PIT) fundamentals store** that other projects
import to avoid look-ahead bias. Everything else (section isolation, text extraction,
reconciliation, QA) exists to (a) populate that store with lineage and (b) measure how
trustworthy text-derived numbers are relative to XBRL.

Two sources feed the PIT DB:

| source | what | knowable-on date |
|---|---|---|
| `xbrl` | every fact in `companyfacts` for the configured forms, one row per (fact, filing) | the `filed` date of the accession that reported it |
| `text` | numbers extracted from Item 7 / Item 1A prose with an assigned role | the filing date of the 10-K the sentence came from |

A restatement is not an overwrite: when FY2022 revenue is reported again in the FY2023
10-K with a different value, that is a second row with its own `filing_date` and
`accession`. `as_of(D)` picks, per `(cik, field, period_end)`, the row with the greatest
`filing_date <= D` (ties broken by accession, then source priority xbrl > text).

## 2. Architecture

```
config.yaml ─► config.py (pydantic) ─► edgar_client.py (UA header, 8 rps token bucket, disk cache)
                                         │
             universe.py ◄───────────────┤  company_tickers.json, submissions/CIK*.json
                │ accession list         │
                ▼                        ▼
           sections.py (Item 1A, Item 7)   xbrl.py (companyfacts, tag_map.yaml → canonical fields)
                │                              │
           numeric_spans.py ─► baseline_extract.py / ner/infer.py
                │                              │
                └──────────► reconcile.py ◄────┘  MATCH / MISMATCH(+cause) / NO_XBRL_COUNTERPART
                                 │
                             pit_db.py (DuckDB, append-only, as_of / history / latest_vs_asof)
                                 │
                     qa.py ─► dashboard/app.py, reports/*.md, examples/pit_vs_naive.py
```

Artifacts:

- `data/raw/` — raw HTTP responses (filing HTML, JSON), keyed by URL path. Never re-downloaded.
- `data/interim/*.parquet` — universe, sections, extractions, reconciliation.
- `data/pit.duckdb` — PIT database.
- `reports/` — section report, NER metrics, mismatch audit.
- `models/` — trained token classifier + model card.

## 3. Key design decisions

1. **Period identity uses dates, not `fy`/`fp`.** In `companyfacts`, `fy`/`fp` describe the
   *filing* (the FY2023 10-K tags FY2021 comparatives with `fy=2023`). Using them as the
   period key is a classic bug. Periods are identified by `(start, end)`; annual facts are
   durations of ~350–380 days. The PIT key is `period_end` as the spec requires, plus a
   `period_start`/duration column so annual vs quarterly cannot collide.
2. **Append-only PIT table.** Inserts are idempotent on the full key
   `(cik, canonical_field, period_end, filing_date, accession, source, span)`; there is no
   UPDATE or DELETE path in the public API.
3. **Filing date semantics.** `as_of(D)` includes filings with `filing_date <= D`. EDGAR
   filing dates for after-hours acceptances (after 17:30 ET) are next business day, so the
   date already reflects public availability; `acceptance_datetime` is stored for users who
   want intraday precision. This is documented, not hidden.
4. **Hand-rolled cache and rate limiter.** A file cache keyed by URL path under `data/raw/`
   is easier to inspect, commit as fixtures, and reason about than an opaque SQLite cache.
   The limiter is a thread-safe minimum-interval limiter set to 8 rps (SEC limit is 10).
   429/5xx get exponential backoff; 4xx other than 429 are not retried and are logged.
5. **Rules over cleverness for section isolation.** Find all candidate `Item N` heading
   positions in normalized text; for each target item take the longest span between a
   heading and the next item heading. TOC hits produce short spans and lose to the real
   section. Incorporation-by-reference is detected by phrase + span length and flagged as
   `section_missing_by_reference`, never guessed.
6. **Evaluation discipline for M4.** Silver labels come from XBRL matches, so evaluating
   on silver would be circular. Metrics are reported on the gold test split only, split by
   company, with sample sizes, regardless of which extractor wins.
7. **ML dependencies are optional extras** (`pip install .[ner]`) so the PIT DB and
   regex pipeline install without torch.

## 4. Risks and how they are handled

| risk | handling |
|---|---|
| Item 7 incorporated by reference to Exhibit 13 (common for some banks/industrials) | flagged `section_missing_by_reference`, logged, counted in the section report; not parsed from the exhibit in v0.1 |
| TOC duplicates, split headings, `&nbsp;`, unicode dashes, nested tables | normalization pass + longest-span rule + fixture tests per trap |
| Inline XBRL rarely tags MD&A prose | measured (share of Item 7 numeric spans inside `ix:nonFraction`) before relying on it; reported |
| Silver labels are noisy and biased toward numbers that also appear in statements | used for training only; gold test set held out by company |
| Gold set (~200 sentences) is small | metrics carry n; any cell with n < 30 is flagged in reports |
| Matching text to XBRL needs the right period | period inferred from sentence (year mentions, "prior year", "compared to") with the filing's fiscal year as default; ambiguous cases recorded as such |
| "Restatement" in XBRL is often a recast (discontinued ops, segment changes), not an accounting error | the PIT DB records any value change as a new version; the `RESTATEMENT` mismatch cause means "a later filing reports a different value for the same period", which is what matters for look-ahead bias. Documented. |
| Text-extracted "precision vs XBRL" is a proxy, not ground truth | dashboard labels it as agreement-with-XBRL, separate from gold-set precision/recall |
| M8 universe of ~20 names is far too small for decile sorts | use a coarser sort, report n per bucket, label as a bias demonstration, not a trading result |
| `yfinance` is unofficial and survivorship-biased | used only in the example; failures reported, not papered over |
| SEC endpoint changes | endpoint formats verified against sec.gov EDGAR API docs on 2026-10-03; client wraps them in one module |

## 5. Things I need from the user

- Contact email for the SEC `User-Agent` header (blocks M1 live download).
- Review of the 20-ticker universe in `config.yaml` before the first full download.
- ~200 gold labels at M4 (I will provide the exact command; I will not invent labels).
- GPU / time budget for DistilBERT vs FinBERT. Detected hardware: Apple M5 (MPS, no CUDA).

## 6. Milestone status

- [ ] M1 client + universe
- [ ] M2 section isolation
- [ ] M3 regex baseline
- [ ] M4 transformer extractor (blocked on gold labels)
- [ ] M5 reconciliation
- [ ] M6 PIT database
- [ ] M7 dashboard
- [ ] M8 PIT vs naive demo, v0.1.0
