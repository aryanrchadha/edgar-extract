"""Point-in-time database: the four required invariants, append-only behavior, and a
property test, on synthetic data and on real restatements (JNJ, GE) from recorded
companyfacts."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edgar_extract.pit_db import PointInTimeDB, build_pit_db
from edgar_extract.xbrl import canonicalize, flatten_company_facts, load_tag_map

from .conftest import SEC_FIXTURES

COMPANIES = pd.DataFrame(
    [
        {"cik": 200406, "ticker": "JNJ", "name": "Johnson & Johnson"},
        {"cik": 40545, "ticker": "GE", "name": "General Electric"},
        {"cik": 1, "ticker": "ABC", "name": "Synthetic Co"},
    ]
)


def canonical_row(
    field: str, value: float, start: str | None, end: str, filed: str, acc: str, **kw: Any
) -> dict[str, Any]:
    ptype = "instant" if start is None else ("annual" if field != "Q" else "quarter")
    row = {
        "cik": 1, "tag": "us-gaap:Revenues", "unit": "USD", "value": value,
        "period_start": pd.Timestamp(start) if start else pd.NaT, "period_end": pd.Timestamp(end),
        "period_type": ptype, "form": "10-K", "accession": acc, "filed": pd.Timestamp(filed),
        "canonical_field": field,
    }  # fmt: skip
    row.update(kw)
    return row


# Synthetic: FY2022 revenue first reported 100 (filed 2023-02-15), restated to 90 in the
# FY2023 10-K (filed 2024-02-15), which also reports FY2023 = 120.
SYNTHETIC = pd.DataFrame(
    [
        canonical_row(
            "Revenues", 100.0, "2022-01-01", "2022-12-31", "2023-02-15", "0000000001-23-000001"
        ),
        canonical_row(
            "Revenues", 90.0, "2022-01-01", "2022-12-31", "2024-02-15", "0000000001-24-000001"
        ),
        canonical_row(
            "Revenues", 120.0, "2023-01-01", "2023-12-31", "2024-02-15", "0000000001-24-000001"
        ),
        canonical_row("Assets", 500.0, None, "2022-12-31", "2023-02-15", "0000000001-23-000001"),
    ]
)


@pytest.fixture
def synthetic_db(tmp_path: Path) -> PointInTimeDB:
    build_pit_db(tmp_path / "pit.duckdb", SYNTHETIC, COMPANIES)
    return PointInTimeDB(tmp_path / "pit.duckdb", read_only=True)


def fy2022(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["canonical_field"] == "Revenues") & (df["period_end"] == pd.Timestamp("2022-12-31"))
    ]


# -- the four invariants on synthetic data ------------------------------------------------


def test_a_before_restatement_returns_original(synthetic_db: PointInTimeDB) -> None:
    row = fy2022(synthetic_db.as_of("2023-06-30", ["ABC"])).iloc[0]
    assert row["value_usd"] == 100.0 and row["accession"] == "0000000001-23-000001"


def test_b_after_restatement_returns_restated(synthetic_db: PointInTimeDB) -> None:
    row = fy2022(synthetic_db.as_of("2024-02-15", ["ABC"])).iloc[0]  # the filing day itself
    assert row["value_usd"] == 90.0 and row["accession"] == "0000000001-24-000001"


def test_c_before_original_filing_returns_nothing(synthetic_db: PointInTimeDB) -> None:
    assert fy2022(synthetic_db.as_of("2023-02-14", ["ABC"])).empty
    assert synthetic_db.as_of("2023-02-14", ["ABC"]).empty


def test_d_no_row_filed_after_query_date(synthetic_db: PointInTimeDB) -> None:
    for d in ("2023-02-15", "2023-12-31", "2024-02-14", "2030-01-01"):
        res = synthetic_db.as_of(d, period_types=None, sources=None, exclude_causes=None)
        assert (res["filing_date"] <= pd.Timestamp(d)).all()


def test_lag_days_delays_knowability(synthetic_db: PointInTimeDB) -> None:
    assert fy2022(synthetic_db.as_of("2023-02-15", ["ABC"], lag_days=1)).empty
    assert not fy2022(synthetic_db.as_of("2023-02-16", ["ABC"], lag_days=1)).empty


def test_history_keeps_every_version(synthetic_db: PointInTimeDB) -> None:
    h = synthetic_db.history("ABC", "Revenues")
    assert list(zip(h["period_end"].dt.year, h["value_usd"], strict=True)) == [
        (2022, 100.0),
        (2022, 90.0),
        (2023, 120.0),
    ]


def test_latest_vs_asof_shows_the_gap(synthetic_db: PointInTimeDB) -> None:
    gap = synthetic_db.latest_vs_asof("ABC", "Revenues", "2023-06-30")
    (row,) = gap.to_dict("records")
    assert (row["asof_value"], row["latest_value"], row["change"]) == (100.0, 90.0, -10.0)
    assert row["restated_since"]


def test_instants_and_period_filters(synthetic_db: PointInTimeDB) -> None:
    res = synthetic_db.as_of("2023-06-30", ["ABC"], ["Assets"])
    assert res["value_usd"].tolist() == [500.0] and res["period_start"].isna().all()
    assert synthetic_db.as_of("2023-06-30", ["ABC"], ["Assets"], period_types=("annual",)).empty


# -- append-only ----------------------------------------------------------------------------


def test_reingest_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "pit.duckdb"
    first = build_pit_db(path, SYNTHETIC, COMPANIES)
    assert first["new_xbrl_rows"] == 4
    assert build_pit_db(path, SYNTHETIC, COMPANIES)["new_xbrl_rows"] == 0
    # A later filing re-reports FY2023 with a new value: a new row, the old one stays.
    later = pd.DataFrame(
        [
            canonical_row(
                "Revenues", 125.0, "2023-01-01", "2023-12-31", "2025-02-15", "0000000001-25-000001"
            )
        ]
    )
    assert build_pit_db(path, later, COMPANIES)["new_xbrl_rows"] == 1
    with PointInTimeDB(path, read_only=True) as pit:
        values = pit.history("ABC", "Revenues")["value_usd"].tolist()
        assert values == [100.0, 90.0, 120.0, 125.0]
        assert pit.count() == 5


def test_api_has_no_update_or_delete() -> None:
    public = {name for name in dir(PointInTimeDB) if not name.startswith("_")}
    assert not any(
        word in name for name in public for word in ("update", "delete", "drop", "remove")
    )


def test_annual_and_q4_with_same_period_end_do_not_collide(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            canonical_row(
                "Revenues", 400.0, "2023-01-01", "2023-12-31", "2024-02-15", "0000000001-24-000001"
            ),
            canonical_row(
                "Revenues",
                110.0,
                "2023-10-01",
                "2023-12-31",
                "2024-02-15",
                "0000000001-24-000001",
                period_type="quarter",
            ),
        ]
    )
    build_pit_db(tmp_path / "p.duckdb", rows, COMPANIES)
    with PointInTimeDB(tmp_path / "p.duckdb", read_only=True) as pit:
        res = pit.as_of("2024-03-01", ["ABC"], ["Revenues"], period_types=None)
        assert sorted(res["value_usd"]) == [110.0, 400.0]


def test_text_rows_need_a_period_and_keep_lineage(tmp_path: Path) -> None:
    base = {
        "cik": 1, "form": "10-K", "accession": "0000000001-24-000001",
        "filing_date": pd.Timestamp("2024-02-15"), "value": 120e0, "extractor": "regex_v1",
        "role_cue": "revenue", "xbrl_tag": "us-gaap:Revenues",
        "primary_doc_url": "https://www.sec.gov/Archives/x.htm", "section": "7", "span_start": 10,
        "span_end": 22, "sentence": "Revenue was $120 in 2023.", "canonical_field": "Revenues",
    }  # fmt: skip
    rec = pd.DataFrame(
        [
            {**base, "status": "MATCH", "mismatch_cause": None,
             "period_start": pd.Timestamp("2023-01-01"), "period_end": pd.Timestamp("2023-12-31")},
            {**base, "status": "NO_XBRL_COUNTERPART", "mismatch_cause": None, "span_start": 30,
             "period_start": pd.NaT, "period_end": pd.NaT},
        ]
    )  # fmt: skip
    stats = build_pit_db(tmp_path / "p.duckdb", SYNTHETIC, COMPANIES, [rec])
    assert stats["new_text_rows"] == 1
    with PointInTimeDB(tmp_path / "p.duckdb", read_only=True) as pit:
        both = pit.as_of("2024-03-01", ["ABC"], ["Revenues"], sources=("xbrl", "text"))
        fy23 = both[both["period_end"] == pd.Timestamp("2023-12-31")]
        assert fy23["source"].tolist() == ["xbrl"]  # same filing: XBRL wins the tie
        text = pit.as_of("2024-03-01", ["ABC"], ["Revenues"], sources=("text",))
        (row,) = text.to_dict("records")
        assert (row["mismatch_cause"], row["lineage_span_start"], row["lineage_section"]) == (
            "MATCH",
            10,
            "7",
        )
        assert row["lineage_url"].endswith("x.htm") and pd.isna(row["confidence"])


# -- real data: JNJ (Kenvue separation) and GE (spin-offs) ---------------------------------


@pytest.fixture(scope="module")
def real_db(tmp_path_factory: pytest.TempPathFactory) -> PointInTimeDB:
    frames = []
    for cik in ("0000200406", "0000040545"):
        path = SEC_FIXTURES / "data.sec.gov" / "api" / "xbrl" / "companyfacts" / f"CIK{cik}.json"
        frames.append(flatten_company_facts(json.loads(path.read_text())))
    canonical = canonicalize(pd.concat(frames, ignore_index=True), load_tag_map())
    db_path = tmp_path_factory.mktemp("pit") / "pit.duckdb"
    build_pit_db(db_path, canonical, COMPANIES)
    return PointInTimeDB(db_path, read_only=True)


def _jnj_fy2022(pit: PointInTimeDB, d: str) -> pd.DataFrame:
    res = pit.as_of(d, ["JNJ"], ["Revenues"])
    return res[res["period_end"] == pd.Timestamp("2023-01-01")]  # JNJ's FY2022 ends 2023-01-01


def test_real_jnj_restatement_invariants(real_db: PointInTimeDB) -> None:
    # Original FY2022 10-K filed 2023-02-16: $94,943M; FY2023 10-K (after the Kenvue
    # separation) filed 2024-02-16 recast FY2022 to $79,990M.
    assert _jnj_fy2022(real_db, "2023-02-15").empty  # (c)
    assert _jnj_fy2022(real_db, "2023-06-01")["value_usd"].tolist() == [94_943e6]  # (a)
    assert _jnj_fy2022(real_db, "2024-06-01")["value_usd"].tolist() == [79_990e6]  # (b)
    hist = real_db.history("JNJ", "Revenues")
    fy22 = hist[hist["period_end"] == pd.Timestamp("2023-01-01")]
    assert {94_943e6, 79_990e6} <= set(fy22["value_usd"])


def test_real_ge_three_versions(real_db: PointInTimeDB) -> None:
    def fy2022(d: str) -> list[float]:
        res = real_db.as_of(d, ["GE"], ["Revenues"])
        return res[res["period_end"] == pd.Timestamp("2022-12-31")]["value_usd"].tolist()

    assert fy2022("2023-02-09") == []
    assert fy2022("2023-03-01") == [76_555e6]  # as first reported
    assert fy2022("2024-03-01") == [58_100e6]  # after the GE HealthCare spin-off
    assert fy2022("2025-03-01") == [29_139e6]  # after the GE Vernova spin-off
    gap = real_db.latest_vs_asof("GE", "Revenues", "2023-03-01")
    row = gap[gap["period_end"] == pd.Timestamp("2022-12-31")].iloc[0]
    assert row["restated_since"] and row["rel_change"] == pytest.approx(29_139 / 76_555 - 1)


def test_real_restatements_listing(real_db: PointInTimeDB) -> None:
    r = real_db.restatements(["GE"], ["Revenues"])
    fy22 = r[r["period_end"] == pd.Timestamp("2022-12-31")].iloc[0]
    assert fy22["n_values"] == 3


# -- property test: invariant (d) for random dates, fields, companies ----------------------

DATES = st.dates(min_value=date(2018, 1, 1), max_value=date(2027, 1, 1))
FIELDS = ["Revenues", "NetIncomeLoss", "Assets"]


@settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    d=DATES,
    fields=st.one_of(
        st.none(),
        st.lists(st.sampled_from(["Revenues", "NetIncomeLoss", "Assets"]), min_size=1, max_size=3),
    ),
    tickers=st.one_of(st.none(), st.lists(st.sampled_from(["JNJ", "GE"]), min_size=1, max_size=2)),
    lag=st.integers(min_value=0, max_value=5),
)
def test_property_no_future_rows_and_matches_brute_force(
    real_db: PointInTimeDB, d: date, fields: list[str] | None, tickers: list[str] | None, lag: int
) -> None:
    res = real_db.as_of(d, tickers, fields, period_types=None, lag_days=lag)
    cutoff = pd.Timestamp(d - timedelta(days=lag))
    assert (res["filing_date"] <= cutoff).all()  # invariant (d)
    # Brute force: latest filing per period among rows filed on or before the cutoff.
    allrows = pd.concat(
        [
            real_db.history(t, f, period_types=None)
            for t in ("JNJ", "GE")
            for f in ("Revenues", "NetIncomeLoss", "Assets")
        ]
    )
    allrows = allrows[allrows["filing_date"] <= cutoff]
    if tickers is not None:
        allrows = allrows[allrows["ticker"].isin(tickers)]
    if fields is not None:
        allrows = allrows[allrows["canonical_field"].isin(fields)]
    key = ["ticker", "canonical_field", "period_start", "period_end"]
    expected = (
        allrows.sort_values([*key, "filing_date", "accession"]).groupby(key, dropna=False).tail(1)
    )
    assert len(res) == len(expected)
    merged = res.merge(expected, on=key, suffixes=("", "_bf"))
    assert len(merged) == len(res)
    assert (merged["value_usd"] == merged["value_usd_bf"]).all()
