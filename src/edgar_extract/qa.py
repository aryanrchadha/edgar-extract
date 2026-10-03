"""Quality metrics computed from built artifacts (no network). Used by the dashboard and
by ``edgar-extract run --stage qa``, which writes ``reports/qa_summary.md``.

Definitions. Both are measured against XBRL, which is not the same as being correct:

* **Agreement with XBRL** ("precision vs XBRL"): among text extractions that could be
  compared with an XBRL fact (status MATCH or MISMATCH), the share that MATCH. A mismatch
  can be a correct reading of a segment or non-GAAP number, so this understates accuracy
  for those roles. Accuracy against human labels lives in ``reports/ner_metrics.md``.
* **XBRL recall**: for each 10-K, the filing's own key facts for its fiscal year
  (Revenues, R&D, restructuring, impairment, litigation expense) that a text extraction
  from the same filing matched. Prose need not mention every fact, so this measures
  coverage of the facts, not extractor failure.
* **Role coverage**: per filing and section, the share of detected numeric spans that the
  extractor assigned a specific role (anything except OTHER_NUMERIC).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from edgar_extract.universe import fiscal_year_from_period_end

RECALL_FIELDS = (
    "Revenues",
    "ResearchAndDevelopmentExpense",
    "RestructuringCharges",
    "ImpairmentCharges",
    "LitigationExpense",
)
SMALL_N = 30


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rate_table(df: pd.DataFrame, by: list[str], k_col: str, n_col: str, name: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)[[k_col, n_col]].sum().reset_index()
    g[name] = g[k_col] / g[n_col].where(g[n_col] > 0)
    ci = [wilson_interval(int(k), int(n)) for k, n in zip(g[k_col], g[n_col], strict=True)]
    g[f"{name}_lo"] = [c[0] for c in ci]
    g[f"{name}_hi"] = [c[1] for c in ci]
    g["small_n"] = g[n_col] < SMALL_N
    return g


def agreement(rec: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Share of comparable extractions that match XBRL, with n and a Wilson 95% interval."""
    comp = rec[rec["status"].isin(["MATCH", "MISMATCH"])].assign(
        n_match=lambda d: (d["status"] == "MATCH").astype(int), n_comparable=1
    )
    if comp.empty:
        return pd.DataFrame(columns=[*by, "n_match", "n_comparable", "agreement"])
    return _rate_table(comp, by, "n_match", "n_comparable", "agreement")


def key_facts(canonical: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Each original 10-K's own annual facts for RECALL_FIELDS, for its own fiscal year."""
    filings = universe[~universe["is_amendment"]][["ticker", "cik", "fiscal_year", "accession"]]
    c = canonical[
        canonical["canonical_field"].isin(RECALL_FIELDS) & (canonical["period_type"] == "annual")
    ].copy()
    c["fy_label"] = c["period_end"].map(lambda d: fiscal_year_from_period_end(d.date()))
    merged = filings.merge(c, on=["cik", "accession"])
    merged = merged[merged["fy_label"] == merged["fiscal_year"]]
    return merged[["ticker", "cik", "fiscal_year", "accession", "canonical_field", "value"]]


def xbrl_recall(rec: pd.DataFrame, canonical: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    facts = key_facts(canonical, universe)
    matched = rec[rec["status"] == "MATCH"][["accession", "canonical_field", "fiscal_year_ref"]]
    matched = matched.drop_duplicates().assign(found=1)
    facts = facts.merge(
        matched,
        left_on=["accession", "canonical_field", "fiscal_year"],
        right_on=["accession", "canonical_field", "fiscal_year_ref"],
        how="left",
    )
    facts["found"] = facts["found"].fillna(0).astype(int)
    return facts.assign(n_facts=1)


def recall(
    rec: pd.DataFrame, canonical: pd.DataFrame, universe: pd.DataFrame, by: list[str]
) -> pd.DataFrame:
    facts = xbrl_recall(rec, canonical, universe)
    if facts.empty:
        return pd.DataFrame(columns=[*by, "found", "n_facts", "recall"])
    return _rate_table(facts, by, "found", "n_facts", "recall")


def cause_breakdown(rec: pd.DataFrame, by: str = "ticker") -> pd.DataFrame:
    mm = rec[rec["status"] == "MISMATCH"]
    return mm.groupby([by, "mismatch_cause"]).size().reset_index(name="n")


def role_coverage(extractions: pd.DataFrame) -> pd.DataFrame:
    df = extractions.assign(
        structured=(extractions["role"] != "OTHER_NUMERIC").astype(int), n_spans=1
    )
    g = (
        df.groupby(["ticker", "fiscal_year", "accession", "section"])[["structured", "n_spans"]]
        .sum()
        .reset_index()
    )
    g["coverage"] = g["structured"] / g["n_spans"]
    g["filing"] = g["ticker"] + " FY" + g["fiscal_year"].astype(str)
    return g


def pit_coverage(pit_path: Path) -> pd.DataFrame:
    """Rows (versions) and distinct facts per company and fiscal year in the PIT DB."""
    from edgar_extract.pit_db import PointInTimeDB

    with PointInTimeDB(pit_path, read_only=True) as pit:
        df = pit.con.execute(
            """
            SELECT c.ticker, f.period_end, f.canonical_field, f.period_start, f.source
            FROM facts f JOIN companies c USING (cik)
            WHERE f.period_type IN ('annual', 'instant')
            """
        ).df()
    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "fiscal_year", "versions", "facts", "versions_per_fact"]
        )
    df["fiscal_year"] = df["period_end"].map(
        lambda d: fiscal_year_from_period_end(pd.Timestamp(d).date())
    )
    g = df.groupby(["ticker", "fiscal_year"]).size().to_frame("versions")
    distinct = df.drop_duplicates(
        ["ticker", "fiscal_year", "canonical_field", "period_start", "period_end"]
    )
    g["facts"] = distinct.groupby(["ticker", "fiscal_year"]).size()
    g["versions_per_fact"] = g["versions"] / g["facts"]
    return g.reset_index()


def summary_markdown(
    rec: pd.DataFrame,
    canonical: pd.DataFrame,
    universe: pd.DataFrame,
    extractions: pd.DataFrame,
    extractor: str,
) -> str:
    def pct(x: float) -> str:
        return "n/a" if pd.isna(x) else f"{x:.1%}"

    agr = agreement(rec, ["role"])
    rcl = recall(rec, canonical, universe, ["canonical_field"])
    cov = role_coverage(extractions)
    overall_a = agreement(rec.assign(all="all"), ["all"])
    lines = [
        f"# QA summary ({extractor})",
        "",
        "Generated by `edgar-extract run --stage qa` from built artifacts. Definitions are in",
        "`src/edgar_extract/qa.py`; both rates are measured against XBRL, which is not the",
        "same as correctness (see `reports/mismatch_audit.md` and `reports/ner_metrics.md`).",
        "",
        "## Agreement with XBRL (MATCH / comparable), by role",
        "",
        "| role | comparable n | match | agreement [95% CI] |",
        "|---|---|---|---|",
    ]
    for r in agr.to_dict("records"):
        flag = " (n < 30)" if r["small_n"] else ""
        lines.append(
            f"| {r['role']} | {r['n_comparable']} | {r['n_match']} | {pct(r['agreement'])} "
            f"[{pct(r['agreement_lo'])}-{pct(r['agreement_hi'])}]{flag} |"
        )
    if not overall_a.empty:
        o = overall_a.iloc[0]
        lines.append(f"| **all** | {o['n_comparable']} | {o['n_match']} | {pct(o['agreement'])} |")
    lines += [
        "",
        "## XBRL recall: the filing's own key facts matched by text from the same filing",
        "",
        "| field | facts n | found | recall [95% CI] |",
        "|---|---|---|---|",
    ]
    for r in rcl.to_dict("records"):
        flag = " (n < 30)" if r["small_n"] else ""
        lines.append(
            f"| {r['canonical_field']} | {r['n_facts']} | {r['found']} | {pct(r['recall'])} "
            f"[{pct(r['recall_lo'])}-{pct(r['recall_hi'])}]{flag} |"
        )
    lines += [
        "",
        "## Role coverage (share of numeric spans given a specific role)",
        "",
        "| section | filings | median coverage | min | max |",
        "|---|---|---|---|---|",
    ]
    for section, g in cov.groupby("section"):
        lines.append(
            f"| Item {section} | {g['accession'].nunique()} | {pct(g['coverage'].median())} "
            f"| {pct(g['coverage'].min())} | {pct(g['coverage'].max())} |"
        )
    return "\n".join(lines) + "\n"
