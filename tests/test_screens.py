"""Point-in-time Piotroski F-score and Altman Z'' on a hand-computed synthetic company."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from edgar_extract.pit_db import PointInTimeDB, build_pit_db
from edgar_extract.screens import altman_z2, consecutive_year_ends, piotroski, screen

COMPANIES = pd.DataFrame([{"cik": 1, "ticker": "ABC", "name": "ABC Corp"}])
# FY ends 2021-2023; each 10-K filed on Feb 15 of the next year.
ACC = {2021: "0000000001-22-000001", 2022: "0000000001-23-000001", 2023: "0000000001-24-000001"}

# Year-end values chosen so every Piotroski signal is True in FY2023 and easy to check.
YEARS: dict[int, dict[str, float]] = {
    2021: {"NetIncomeLoss": 40.0, "Assets": 1000.0},
    2022: {
        "NetIncomeLoss": 50.0, "NetCashProvidedByOperatingActivities": 60.0, "Revenues": 800.0,
        "GrossProfit": 300.0, "WeightedAverageDilutedShares": 100.0, "Assets": 1100.0,
        "AssetsCurrent": 400.0, "LiabilitiesCurrent": 300.0, "LongTermDebtNoncurrent": 330.0,
        "RetainedEarnings": 200.0, "StockholdersEquity": 500.0, "Liabilities": 600.0,
        "OperatingIncomeLoss": 70.0,
    },
    2023: {
        "NetIncomeLoss": 110.0, "NetCashProvidedByOperatingActivities": 150.0, "Revenues": 990.0,
        "GrossProfit": 400.0, "WeightedAverageDilutedShares": 95.0, "Assets": 1200.0,
        "AssetsCurrent": 500.0, "LiabilitiesCurrent": 300.0, "LongTermDebtNoncurrent": 300.0,
        "RetainedEarnings": 240.0, "StockholdersEquity": 600.0, "Liabilities": 600.0,
        "OperatingIncomeLoss": 120.0,
    },
}  # fmt: skip
DURATION = {"NetIncomeLoss", "NetCashProvidedByOperatingActivities", "Revenues", "GrossProfit",
            "CostOfRevenue", "WeightedAverageDilutedShares", "OperatingIncomeLoss"}  # fmt: skip


def row(
    field: str, value: float, fy: int, filed_fy: int, form: str = "10-K", start: str | None = None
) -> dict[str, Any]:
    duration = field in DURATION
    return {
        "cik": 1, "tag": f"us-gaap:{field}", "unit": "USD", "value": value,
        "period_start": pd.Timestamp(start or f"{fy}-01-01") if duration else pd.NaT,
        "period_end": pd.Timestamp(f"{fy}-12-31"),
        "period_type": "annual" if duration else "instant", "form": form,
        "accession": ACC.get(filed_fy, f"0000000001-{filed_fy % 100:02d}-000009"),
        "filed": pd.Timestamp(f"{filed_fy + 1}-02-15"), "canonical_field": field,
    }  # fmt: skip


def canonical(years: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = [row(f, v, fy, fy) for fy, vals in years.items() for f, v in vals.items()]
    # Each 10-K also re-reports the prior year's balance sheet (normal comparatives).
    rows += [row(f, v, fy, fy + 1) for fy, vals in years.items() if fy + 1 in YEARS
             for f, v in vals.items() if f not in DURATION]  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def pit(tmp_path: Path) -> PointInTimeDB:
    build_pit_db(tmp_path / "p.duckdb", canonical(YEARS), COMPANIES)
    return PointInTimeDB(tmp_path / "p.duckdb", read_only=True)


def test_full_score_and_altman(pit: PointInTimeDB) -> None:
    (r,) = screen(pit, "2024-06-30").to_dict("records")
    assert r["fiscal_year_end"] == pd.Timestamp("2023-12-31") and r["years_available"] == 3
    assert (r["f_score"], r["f_signals"], r["f_complete"]) == (9, 9, True)
    # X1 = 200/1200, X2 = 240/1200, X3 = 120/1200, X4 = 600/600
    expected = 6.56 * 200 / 1200 + 3.26 * 240 / 1200 + 6.72 * 120 / 1200 + 1.05 * 1.0
    assert r["z_double_prime"] == pytest.approx(expected)
    assert r["z_zone"] == "safe" and r["derived"] == "" and r["missing"] == ""
    assert r["knowable_by"] <= pd.Timestamp("2024-06-30")


def test_screen_is_point_in_time(pit: PointInTimeDB) -> None:
    # Before the FY2023 10-K (filed 2024-02-15) the latest fiscal year is FY2022, which
    # has only one prior year: F3 and F9 cannot be evaluated.
    (r,) = screen(pit, "2024-02-14").to_dict("records")
    assert r["fiscal_year_end"] == pd.Timestamp("2022-12-31")
    assert r["years_available"] == 2 and r["f_signals"] < 9
    assert r["knowable_by"] <= pd.Timestamp("2024-02-14")
    assert screen(pit, "2022-02-14").empty  # nothing filed yet


def test_ttm_figures_in_a_10q_do_not_define_a_fiscal_year(tmp_path: Path) -> None:
    ttm = pd.DataFrame([row("NetIncomeLoss", 999.0, 2024, 2023, form="10-Q", start="2023-04-01")])
    ttm["period_end"] = pd.Timestamp("2024-03-31")
    ttm["filed"] = pd.Timestamp("2024-05-01")
    build_pit_db(tmp_path / "p.duckdb", pd.concat([canonical(YEARS), ttm]), COMPANIES)
    with PointInTimeDB(tmp_path / "p.duckdb", read_only=True) as pit:
        (r,) = screen(pit, "2024-06-30").to_dict("records")
    assert r["fiscal_year_end"] == pd.Timestamp("2023-12-31")


def test_fallbacks_are_labeled(tmp_path: Path) -> None:
    years = {fy: dict(v) for fy, v in YEARS.items()}
    for fy in (2022, 2023):
        years[fy]["CostOfRevenue"] = years[fy]["Revenues"] - years[fy].pop("GrossProfit")
        years[fy]["LiabilitiesAndStockholdersEquity"] = (
            years[fy]["Liabilities"] + years[fy]["StockholdersEquity"]
        )
        del years[fy]["Liabilities"]
    build_pit_db(tmp_path / "p.duckdb", canonical(years), COMPANIES)
    with PointInTimeDB(tmp_path / "p.duckdb", read_only=True) as pit:
        (r,) = screen(pit, "2024-06-30").to_dict("records")
    assert "GrossProfit=Revenues-CostOfRevenue" in r["derived"]
    assert "Liabilities=LiabilitiesAndStockholdersEquity-StockholdersEquity" in r["derived"]
    assert r["f_score"] == 9 and r["z_double_prime"] is not None


def test_missing_inputs_give_none_not_a_guess() -> None:
    bank = {"NetIncomeLoss": 10.0, "Assets": 100.0, "Liabilities": 90.0, "StockholdersEquity": 10.0,
            "RetainedEarnings": 5.0, "EBIT": 15.0}  # fmt: skip
    z = altman_z2(bank)
    assert z["x1_working_capital"] is None and z["z_double_prime"] is None and z["z_zone"] is None
    p = piotroski([bank, {"Assets": 90.0}])
    assert p["f6_current_ratio_up"] is None and p["f8_gross_margin_up"] is None
    assert p["f1_roa_positive"] is True and p["f_signals"] < 9 and not p["f_complete"]


def test_consecutive_year_ends_stop_at_a_gap() -> None:
    ends = [pd.Timestamp(d) for d in ("2023-12-31", "2022-12-31", "2020-12-31")]
    assert consecutive_year_ends(ends) == ends[:2]
    weekly = [pd.Timestamp(d) for d in ("2023-12-30", "2022-12-31", "2022-01-01")]
    assert consecutive_year_ends(weekly) == weekly
