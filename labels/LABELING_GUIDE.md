# Gold labeling guide

You label **candidate numbers** in sentences taken from Item 1A (Risk Factors) and Item 7
(MD&A) of the 10-Ks in the universe. The tool highlights one candidate at a time
(`>>like this<<`) and you press one key. Each sentence is saved as soon as all of its
candidates are labeled, so you can quit at any time and resume later.

```bash
uv run edgar-extract label --labeler <your initials>
```

| key | label | use it for |
|---|---|---|
| `r` | REVENUE | a **dollar amount of revenue / net sales**, at any level (total, segment, product, region), for a reported period. Both levels ("revenue was $8.0 billion") and dollar changes ("revenue increased $1.0 billion") count. |
| `g` | GUIDANCE | a figure the company states about **its own future** results or plans: expected revenue, capex, charges, margins, targets, ranges. Percentages count. |
| `c` | ONE_TIME_CHARGE | a dollar amount of a **non-recurring charge or gain** recorded in a period: impairments, restructuring, severance, write-offs or write-downs, loss on extinguishment, a litigation **charge or settlement expense**. |
| `p` | RESTATED_PRIOR | a prior-period figure the sentence explicitly calls **restated, recast, revised, or "as previously reported"** (either the old or the new value). |
| `o` | OTHER_NUMERIC | any other real number: other expenses, cash, debt, margins, growth rates, counts, a litigation **reserve balance**, EPS. |
| `x` | not a number | the highlighted text is not a numeric quantity, or the span is cut wrongly (e.g. part of a name or a code). |

Other keys: `b` goes back one span in the current sentence, `s` skips the sentence (it is
not saved), `u` undoes the last saved sentence, and `q` saves and quits.

Rules of thumb:

- **Judge the number, not the sentence.** In "Revenue increased 12% to $1.2 billion", the
  `$1.2 billion` is REVENUE and the `12%` is OTHER_NUMERIC.
- **Percentages are never REVENUE or ONE_TIME_CHARGE.** They are GUIDANCE when
  forward-looking, otherwise OTHER_NUMERIC.
- **Forward-looking beats everything.** "We expect restructuring charges of $1.0 to
  $1.2 billion" is GUIDANCE for both ends of the range.
- **Macro forecasts are not guidance.** "Renewables are expected to reach 20% of
  energy by 2050" is OTHER_NUMERIC; guidance is the company's own outlook.
- **Revenue-like is not revenue.** Cost of sales, deferred revenue balances and
  "sales and marketing" expense are OTHER_NUMERIC.
- When genuinely unsure, choose OTHER_NUMERIC rather than guessing a specific role.

The sample (`gold_candidates.jsonl`, 218 sentences) was drawn once, with a fixed seed,
from keyword strata (restatement language, forward-looking language, charges, revenue,
and a uniform random stratum) and spread across companies. It was **not** drawn from
either extractor's predictions, and the tool never shows a model or regex suggestion.

## Checking your labels

Each saved sentence records how long it was on screen. `edgar-extract label --audit`
lists sentences saved in under 1.5 seconds per number, which is faster than a sentence can
be read, plus labels the rules above exclude (RESTATED_PRIOR where the sentence never says
restated, recast, revised or previously reported; a percentage labeled REVENUE or
ONE_TIME_CHARGE). Training and evaluation drop the too-fast sentences automatically.
`edgar-extract label --discard-fast` moves them to `gold_labels.discarded.jsonl` (nothing
is deleted), and the next `edgar-extract label` run shows them again for relabeling.

The tool discards keys typed before each number appears, so a held-down key or pasted text
can no longer label spans that were never shown.
