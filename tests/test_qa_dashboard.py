"""QA metrics on small synthetic artifacts, and the dashboard rendered headless offline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from edgar_extract import qa
from edgar_extract.pit_db import build_pit_db

ACC = "0000000001-25-000001"


def rec_row(status: str, cause: str | None, **kw: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "ABC", "cik": 1, "fiscal_year": 2024, "accession": ACC, "section": "7",
        "role": "REVENUE", "status": status, "mismatch_cause": cause, "extractor": "regex_v1",
        "canonical_field": "Revenues", "fiscal_year_ref": 2024,
    }  # fmt: skip
    row.update(kw)
    return row


REC = pd.DataFrame(
    [
        rec_row("MATCH", None),
        rec_row("MISMATCH", "SEGMENT_VS_CONSOLIDATED"),
        rec_row("MISMATCH", "EXTRACTION_ERROR"),
        rec_row("NO_XBRL_COUNTERPART", None, role="GUIDANCE"),
    ]
)
UNIVERSE = pd.DataFrame(
    [{"ticker": "ABC", "cik": 1, "fiscal_year": 2024, "accession": ACC, "is_amendment": False,
      "company_name": "ABC Corp"}]
)  # fmt: skip
CANONICAL = pd.DataFrame(
    [
        {"cik": 1, "accession": ACC, "canonical_field": f, "period_type": "annual",
         "period_start": pd.Timestamp("2024-01-01"), "period_end": pd.Timestamp("2024-12-31"),
         "value": v, "tag": "us-gaap:X", "unit": "USD", "form": "10-K",
         "filed": pd.Timestamp("2025-02-01")}
        for f, v in [("Revenues", 10e9), ("ResearchAndDevelopmentExpense", 1e9)]
    ]
)  # fmt: skip
EXTRACTIONS = pd.DataFrame(
    [
        {"ticker": "ABC", "fiscal_year": 2024, "accession": ACC, "section": s, "role": r,
         "extractor": "regex_v1"}
        for s, r in [("7", "REVENUE"), ("7", "OTHER_NUMERIC"), ("7", "OTHER_NUMERIC"),
                     ("1A", "OTHER_NUMERIC")]
    ]
)  # fmt: skip


def test_agreement_counts_only_comparable_rows() -> None:
    a = qa.agreement(REC, ["ticker"]).iloc[0]
    assert (a["n_comparable"], a["n_match"]) == (3, 1)
    assert a["agreement"] == pytest.approx(1 / 3) and a["small_n"]
    assert a["agreement_lo"] < a["agreement"] < a["agreement_hi"]


def test_recall_uses_the_filings_own_key_facts() -> None:
    r = qa.recall(REC, CANONICAL, UNIVERSE, ["canonical_field"]).set_index("canonical_field")
    assert r.loc["Revenues", "found"] == 1 and r.loc["Revenues", "n_facts"] == 1
    assert r.loc["ResearchAndDevelopmentExpense", "found"] == 0


def test_role_coverage_and_cause_breakdown() -> None:
    cov = qa.role_coverage(EXTRACTIONS).set_index("section")
    assert cov.loc["7", "coverage"] == pytest.approx(1 / 3) and cov.loc["1A", "coverage"] == 0
    cb = qa.cause_breakdown(REC)
    assert set(cb["mismatch_cause"]) == {"SEGMENT_VS_CONSOLIDATED", "EXTRACTION_ERROR"}


def test_wilson_interval_bounds() -> None:
    lo, hi = qa.wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"sec": {"contact_email": "a@b.edu"}, "universe": {"tickers": ["ABC"]}})
    )
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    UNIVERSE.to_parquet(interim / "universe.parquet")
    EXTRACTIONS.to_parquet(interim / "extractions.parquet")
    CANONICAL.to_parquet(interim / "canonical_facts.parquet")
    REC.to_parquet(interim / "reconciliation.parquet")
    build_pit_db(
        tmp_path / "data" / "pit.duckdb",
        CANONICAL,
        UNIVERSE.rename(columns={"company_name": "name"}),
    )
    return cfg


def test_pit_coverage(artifacts: Path) -> None:
    pc = qa.pit_coverage(artifacts.parent / "data" / "pit.duckdb")
    assert pc[["ticker", "fiscal_year", "versions", "facts"]].values.tolist() == [
        ["ABC", 2024, 2, 2]
    ]


def test_dashboard_renders_offline(artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("EDGAR_EXTRACT_CONFIG", str(artifacts))
    app = Path(__file__).parents[1] / "src" / "edgar_extract" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]  # sockets are blocked in tests
    headers = [h.value for h in at.header]
    assert headers[0].startswith("1. Precision and recall")
    assert any("Point-in-time" in h for h in headers)
    assert len(at.get("plotly_chart")) >= 7
    assert any("No gold-label metrics yet" in i.value for i in at.info)


def test_recall_ignores_matched_change_amounts() -> None:
    rec = REC.assign(comparison="level")
    rec.loc[0, "comparison"] = "change"  # the only MATCH is a change amount
    r = qa.recall(rec, CANONICAL, UNIVERSE, ["canonical_field"]).set_index("canonical_field")
    assert r.loc["Revenues", "found"] == 0
