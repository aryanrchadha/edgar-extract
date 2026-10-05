"""Draw the 25 random mismatches reviewed in reports/mismatch_audit.md (seed 2026).

    uv run python scripts/sample_mismatch_audit.py

Prints each case with its sentence, compared XBRL fact and rule-assigned cause, so anyone
can re-check the verdicts recorded in the report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
N = 25


def main() -> None:
    rec = pd.read_parquet(ROOT / "data" / "interim" / "reconciliation.parquet")
    mismatches = rec[rec["status"] == "MISMATCH"]
    if "comparison" in mismatches.columns:
        # The audit (2026-10-03) predates change-amount reconciliation: level rows only.
        mismatches = mismatches[mismatches["comparison"] == "level"]
    sample = mismatches.sample(N, random_state=SEED).reset_index(drop=True)
    for i, x in sample.iterrows():
        j = int(x["span_start"]) - int(x["sentence_start"])
        s = str(x["sentence"])
        raw = str(x["raw_text"])
        print(
            f"#{i + 1} {x['ticker']} FY{x['fiscal_year']} Item {x['section']} role={x['role']} "
            f"field={x['canonical_field']} FY_ref={x['fiscal_year_ref']} ({x['period_source']}) "
            f"xbrl={x['xbrl_value']:,.0f} [{x['xbrl_tag']}] cause={x['mismatch_cause']} "
            f"| {x['cause_detail']}"
        )
        print(f"    ...{s[max(0, j - 200) : j]}<<{raw}>>{s[j + len(raw) : j + len(raw) + 120]}")


if __name__ == "__main__":
    main()
