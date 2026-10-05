# Mismatch audit: 25 random reconciliation mismatches

**Who reviewed.** Claude (the AI assistant building this pipeline) reviewed these cases on
2026-10-03 by reading each sentence against the compared XBRL fact. Where a verdict
depended on data, it was checked against the filing's XBRL (noted below). **The project
owner has not yet reviewed them;** treat the verdicts as a second opinion to verify, not
as ground truth. Reproduce the sample with `uv run python scripts/sample_mismatch_audit.py`.

**Sample.** 25 rows drawn uniformly at random (seed 2026) from the 1,767 value
(`comparison="level"`) `MISMATCH` rows of `reconciliation.parquet`, produced by the regex
extractor on the full universe. Change amounts were reconciled later and are not covered
by this audit. **n = 25 is
below 30**, so the accuracy below has a wide interval (95% Wilson: 61%-91%).

**How a verdict is decided.** The true reason for each mismatch is determined first. If it
is one of the four specific causes (restatement, non-GAAP, segment, scale/rounding), the
rules must assign that cause. If the extraction itself is wrong (wrong role, wrong period,
or a change amount read as a level), `EXTRACTION_ERROR` is correct. If the true reason lies
outside the taxonomy (an XBRL concept missing from `tag_map.yaml`, or text that sums
several facts), `EXTRACTION_ERROR` is counted as **incorrect** and flagged as a taxonomy
gap.

## Result

**20 of 25 classified correctly (80%).**

| rule-assigned cause | cases | correct |
|---|---|---|
| SEGMENT_VS_CONSOLIDATED | 7 | 7 |
| EXTRACTION_ERROR | 18 | 13 |
| RESTATEMENT, NON_GAAP_ADJUSTMENT, SCALE_OR_ROUNDING | 0 | - |

True reasons across the 25: 8 role errors by the regex extractor, 5 period-attribution
errors, 10 segment or dimensional amounts (7 caught, 3 missed), 1 concept missing from the
tag map, and 1 aggregation of two facts.

## Cases

| # | company | text value (context) | role / compared fact | rule cause | verdict | true reason |
|---|---|---|---|---|---|---|
| 1 | INTC FY2024 | $922M "Gaudi AI accelerator inventory-related charges" | REVENUE / revenue $53.1B | EXTRACTION_ERROR | correct | role error: an inventory charge tagged REVENUE ("revenue" earlier in sentence) |
| 2 | GE FY2021 | $0.6B "decreased goodwill impairments of" | ONE_TIME_CHARGE / FY2019 goodwill impairment | EXTRACTION_ERROR | correct | a change amount read as a level, and the period taken from "2019 annual ... review" |
| 3 | KO FY2021 | $846M "other operating charges" | ONE_TIME_CHARGE / goodwill impairment $7M | EXTRACTION_ERROR | **incorrect** | taxonomy gap: equals `us-gaap:OtherCostAndExpenseOperating` (846M, verified), which `tag_map.yaml` does not map |
| 4 | INTC FY2025 | $494M impairment "in 2025 and 2024 ... respectively" | ONE_TIME_CHARGE / FY2024 goodwill impairment | EXTRACTION_ERROR | correct | period error: the years precede the amounts, so the "respectively" rule did not apply, and it is a PP&E (not goodwill) impairment |
| 5 | INTC FY2023 | $952M "Revenue was ... driven by higher packaging revenue" | REVENUE / revenue $54.2B | SEGMENT | correct | Intel Foundry Services segment revenue (dimensional fact, precise number) |
| 6 | KHC FY2024 | $20.1B "Net sales decreased 1.0% to ... in 2023" | REVENUE / revenue $26.6B | EXTRACTION_ERROR | **incorrect** | segment: North America net sales 2023 = $20.126B (verified). The sentence never names the segment, so the corroboration rule rejected it |
| 7 | KHC FY2021 | $124M "loss on extinguishment of debt ... in the prior year" | ONE_TIME_CHARGE / impairment | EXTRACTION_ERROR | correct | period error ("in 2021" earlier in the clause beat "prior year"), and extinguishment losses are not a mapped field |
| 8 | KHC FY2021 | $6.5B "carrying amount of brands ... impairment testing" | ONE_TIME_CHARGE / impairment | EXTRACTION_ERROR | correct | role error: a carrying amount, not a charge |
| 9 | GE FY2024 | $0.5B "higher gains on sales of business interests" | REVENUE / revenue | EXTRACTION_ERROR | correct | role error: "sales of business interests" read as sales |
| 10 | PG FY2022 | $2.0B "Net earnings increased 8% to ... due to ... net sales" | REVENUE / revenue | EXTRACTION_ERROR | correct | role error: segment net earnings; "net sales" was the cue to the right |
| 11 | INTC FY2025 | $3.1B "impairments relating to goodwill and acquired intangibles ... in 2024" | ONE_TIME_CHARGE / goodwill impairment $2,984M | EXTRACTION_ERROR | **incorrect** | aggregation: the text sums goodwill and acquired-intangible impairments; the compared fact is goodwill alone. No taxonomy bucket |
| 12 | CAT FY2024 | $4.053B "Financial Products' segment revenues" | REVENUE / revenue $64.8B | SEGMENT | correct | segment fact, member named |
| 13 | KHC FY2023 | $444M "goodwill impairment losses of ... in 2022" | ONE_TIME_CHARGE / FY2023 impairment | EXTRACTION_ERROR | correct | period error: "in 2022" sits after the next amount, so the 2023 earlier in the sentence was used |
| 14 | PFE FY2023 | $1.0B "gross cost savings" | ONE_TIME_CHARGE / restructuring | EXTRACTION_ERROR | correct | role error: savings, not a charge |
| 15 | DIS FY2023 | $6.34 "average monthly revenue per paid subscriber" | REVENUE / revenue | EXTRACTION_ERROR | correct | role error: per-subscriber ARPU, not a revenue total |
| 16 | INTC FY2025 | $103M "non-cash impairments of operating leased assets" | ONE_TIME_CHARGE / FY2025 impairment | EXTRACTION_ERROR | correct | period error: the amount is FY2024 (verified `OperatingLeaseImpairmentLoss` 103M, 2024, restructuring-plan axis). Also an unmapped concept |
| 17 | DIS FY2022 | $5.4B "Cost of products ... increased ... due to higher ... sales" | REVENUE / revenue | EXTRACTION_ERROR | correct | role error: cost of products |
| 18 | PFE FY2023 | $20.1B "Emerging markets revenues ... from $20.1 billion in 2022" | REVENUE / revenue | SEGMENT | correct | geographic dimension, member named |
| 19 | JNJ FY2025 | $35M restructuring "recorded in Cost of products sold" | ONE_TIME_CHARGE / restructuring | SEGMENT | correct | dimensional fact: restructuring by income-statement location (COGS) |
| 20 | KHC FY2023 | $20.3B "compared to ... in 2022" (net sales) | REVENUE / revenue | EXTRACTION_ERROR | **incorrect** | segment: North America net sales 2022 = $20.34B (verified); missed for the same reason as #6 |
| 21 | JNJ FY2022 | $0.9B "Women's Health franchise sales" | REVENUE / revenue | SEGMENT | correct | product dimension, member named |
| 22 | DIS FY2025 | $1,284M "Other revenue increased ... fees received from the Entertainment segment" | REVENUE / revenue | EXTRACTION_ERROR | **incorrect** | segment-level amount: XBRL shows these intersegment fees as a -$1,285M elimination (verified); off by $1M, so not matched |
| 23 | PG FY2024 | $15.2B "Beauty net sales" | REVENUE / revenue | SEGMENT | correct | segment, member named |
| 24 | CAT FY2025 | $25.455B "Construction Industries' total sales ... 2024" | REVENUE / revenue | SEGMENT | correct | segment, member named |
| 25 | XOM FY2024 | $270M "product sales mix decreased earnings by" | REVENUE / revenue | EXTRACTION_ERROR | correct | role error: an earnings driver |

## What this says about the rules

- **EXTRACTION_ERROR is mostly real.** 13 of 18 were genuine extraction mistakes: role
  errors from the regex keyword windows (8) and wrong-year attribution (5). These are
  exactly what the M4 model and better period parsing should fix.
- **The segment rule is precise but misses some cases (7 of 10 found).** It requires the
  segment's name in the sentence, or a precise number, so that coarse figures do not
  match one of hundreds of dimensional facts by chance. That costs recall when the segment
  is named only in a section heading (KHC North America). Carrying the enclosing heading
  into the sentence context would recover these cases.
- **Taxonomy gaps:** concepts outside `tag_map.yaml` (OtherCostAndExpenseOperating,
  OperatingLeaseImpairmentLoss) and text totals that sum several facts have no bucket and
  fall into EXTRACTION_ERROR. The report counts them as rule failures rather than hiding
  them.
- No restatement, non-GAAP or scale case appeared in this sample (those causes are rare:
  1, 33 and 21 rows out of 1,767 mismatches), so this audit says nothing about their
  precision. Their behavior is pinned by the synthetic unit tests in
  `tests/test_reconcile.py`.

## Addendum (2026-10-04): after heading context

Reconciliation now also accepts the segment's name in the headings just above a sentence
(with the number within 5%). It also reconciles change amounts, which this sample does not
contain. Re-running the same 25 cases against the verdicts above:

- #6 and #20 (KHC North America net sales) are now `SEGMENT_VS_CONSOLIDATED`, correct.
- #21 (JNJ Women's Health) is unchanged and correct.
- The remaining misses are #3 (taxonomy gap), #11 (aggregation) and #22 (DIS
  intersegment elimination).

That gives **22 of 25 on this sample**. The change was motivated by these very cases, so
22/25 is an in-sample figure and overstates generalization. An unbiased estimate needs a
fresh sample (a different seed) reviewed blind.
