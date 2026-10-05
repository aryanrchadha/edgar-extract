"""Point-in-time fundamental screens: Piotroski F-score and Altman Z''.

Both are computed only from :meth:`PointInTimeDB.as_of`, so a screen dated D uses nothing
filed after D. Every result carries ``knowable_by``, the latest filing date among the
inputs it used, which is always ``<= D``.

**Fiscal years.** Year *t* is the latest fiscal year whose 10-K was filed by D (10-Q
trailing-twelve-month and proxy figures do not define fiscal years); *t-1* and *t-2* are
the two fiscal years before it (each ~1 year earlier). Values are as of D:
if the year-*t* 10-K recast year *t-1*, the recast comparative is used, which is what an
investor on D would have compared.

**Inputs and fallbacks** (each fallback is listed in the ``derived`` column):

* gross profit = revenues - cost of revenue, when GrossProfit is not tagged;
* total liabilities = liabilities and equity - stockholders' equity, when Liabilities is
  not tagged;
* EBIT = operating income; else pre-tax income + interest expense;
* shares = weighted-average diluted shares; else shares outstanding at year end.

Inputs that are genuinely absent make the affected signals ``None``: banks report no
current assets or gross profit, and GE tags interest expense with a company-specific
concept that companyfacts omits, which leaves its EBIT undefined. Missing inputs are
listed in ``missing`` and never imputed.

**Piotroski (2000) F-score**, nine binary signals. Year-end assets stand in for the
average assets of the original, a common simplification.

1. ROA > 0, where ROA = NI_t / Assets_{t-1}
2. CFO_t > 0
3. ROA_t > ROA_{t-1}
4. CFO_t / Assets_{t-1} > ROA_t  (earnings backed by cash: low accruals)
5. LTD_t / Assets_t < LTD_{t-1} / Assets_{t-1}
6. current ratio_t > current ratio_{t-1}
7. shares_t <= shares_{t-1}  (no dilution)
8. gross margin_t > gross margin_{t-1}
9. asset turnover_t > asset turnover_{t-1}, where turnover = revenue / prior-year assets

``f_score`` counts the signals that hold; ``f_signals`` counts those that could be
evaluated. Compare scores only when ``f_signals`` is 9 (``f_complete``).

**Altman Z''** (Altman 1995, the version for non-manufacturers that needs no market
value): Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4, where X1 = working capital / total
assets, X2 = retained earnings / total assets, X3 = EBIT / total assets and X4 = book
equity / total liabilities. Zones: > 2.60 safe, 1.10-2.60 grey, < 1.10 distress. The
original Z-score needs market capitalisation and is not computed here. Z'' is not
designed for banks, and the missing working capital already leaves it undefined for them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from edgar_extract.pit_db import DateLike, PointInTimeDB

DURATION_FIELDS = (
    "NetIncomeLoss",
    "NetCashProvidedByOperatingActivities",
    "Revenues",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncomeLoss",
    "PretaxIncome",
    "InterestExpense",
    "WeightedAverageDilutedShares",
)
INSTANT_FIELDS = (
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "LongTermDebtNoncurrent",
    "RetainedEarnings",
    "StockholdersEquity",
    "Liabilities",
    "LiabilitiesAndStockholdersEquity",
    "SharesOutstanding",
)
YEAR_DAYS = (330, 400)  # spacing between consecutive fiscal year ends

Z_SAFE = 2.60
Z_DISTRESS = 1.10


def _annual_value(rows: pd.DataFrame, field: str, end: pd.Timestamp) -> pd.Series | None:
    """The duration row for ``field`` ending on ``end`` that is closest to 365 days."""
    r = rows[(rows["canonical_field"] == field) & (rows["period_end"] == end)]
    r = r[r["period_start"].notna()]
    if r.empty:
        return None
    days = [int(d) for d in (r["period_end"] - r["period_start"]).dt.days]
    best = min(range(len(days)), key=lambda i: abs(days[i] - 364))
    return r.iloc[best]


def _instant_value(rows: pd.DataFrame, field: str, end: pd.Timestamp) -> pd.Series | None:
    r = rows[
        (rows["canonical_field"] == field)
        & (rows["period_end"] == end)
        & rows["period_start"].isna()
    ]
    return None if r.empty else r.iloc[0]


def consecutive_year_ends(candidates: Sequence[pd.Timestamp], n: int = 3) -> list[pd.Timestamp]:
    """Up to ``n`` fiscal-year ends, latest first, each about a year before the previous."""
    ends = sorted(set(candidates), reverse=True)
    chosen: list[pd.Timestamp] = []
    for e in ends:
        if not chosen:
            chosen.append(e)
            continue
        gap = (chosen[-1] - e).days
        if YEAR_DAYS[0] <= gap <= YEAR_DAYS[1]:
            chosen.append(e)
        elif gap > YEAR_DAYS[1]:
            break  # a missing year: stop rather than compare non-adjacent years
        if len(chosen) == n:
            break
    return chosen


def year_values(
    rows: pd.DataFrame, end: pd.Timestamp
) -> tuple[dict[str, float], list[str], list[pd.Timestamp]]:
    """Inputs for one fiscal year: values, the fallbacks used, and the filing dates read."""
    vals: dict[str, float] = {}
    filed: list[pd.Timestamp] = []
    for field in DURATION_FIELDS:
        row = _annual_value(rows, field, end)
        if row is not None:
            vals[field] = float(row["value_usd"])
            filed.append(pd.Timestamp(row["filing_date"]))
    for field in INSTANT_FIELDS:
        row = _instant_value(rows, field, end)
        if row is not None:
            vals[field] = float(row["value_usd"])
            filed.append(pd.Timestamp(row["filing_date"]))
    derived: list[str] = []
    if "GrossProfit" not in vals and {"Revenues", "CostOfRevenue"} <= vals.keys():
        vals["GrossProfit"] = vals["Revenues"] - vals["CostOfRevenue"]
        derived.append("GrossProfit=Revenues-CostOfRevenue")
    if (
        "Liabilities" not in vals
        and {"LiabilitiesAndStockholdersEquity", "StockholdersEquity"} <= vals.keys()
    ):
        vals["Liabilities"] = vals["LiabilitiesAndStockholdersEquity"] - vals["StockholdersEquity"]
        derived.append("Liabilities=LiabilitiesAndStockholdersEquity-StockholdersEquity")
    if "OperatingIncomeLoss" in vals:
        vals["EBIT"] = vals["OperatingIncomeLoss"]
    elif {"PretaxIncome", "InterestExpense"} <= vals.keys():
        vals["EBIT"] = vals["PretaxIncome"] + vals["InterestExpense"]
        derived.append("EBIT=PretaxIncome+InterestExpense")
    if "WeightedAverageDilutedShares" in vals:
        vals["Shares"] = vals["WeightedAverageDilutedShares"]
    elif "SharesOutstanding" in vals:
        vals["Shares"] = vals["SharesOutstanding"]
        derived.append("Shares=SharesOutstanding")
    return vals, derived, filed


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _gt(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a > b


def piotroski(y: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Nine signals from up to three years of inputs, latest first."""
    t = y[0] if len(y) > 0 else {}
    p = y[1] if len(y) > 1 else {}
    pp = y[2] if len(y) > 2 else {}
    roa_t = _ratio(t.get("NetIncomeLoss"), p.get("Assets"))
    roa_p = _ratio(p.get("NetIncomeLoss"), pp.get("Assets"))
    cfo_ta = _ratio(t.get("NetCashProvidedByOperatingActivities"), p.get("Assets"))
    lev_t = _ratio(t.get("LongTermDebtNoncurrent"), t.get("Assets"))
    lev_p = _ratio(p.get("LongTermDebtNoncurrent"), p.get("Assets"))
    cr_t = _ratio(t.get("AssetsCurrent"), t.get("LiabilitiesCurrent"))
    cr_p = _ratio(p.get("AssetsCurrent"), p.get("LiabilitiesCurrent"))
    gm_t = _ratio(t.get("GrossProfit"), t.get("Revenues"))
    gm_p = _ratio(p.get("GrossProfit"), p.get("Revenues"))
    at_t = _ratio(t.get("Revenues"), p.get("Assets"))
    at_p = _ratio(p.get("Revenues"), pp.get("Assets"))
    cfo = t.get("NetCashProvidedByOperatingActivities")
    shares_t, shares_p = t.get("Shares"), p.get("Shares")
    signals: dict[str, bool | None] = {
        "f1_roa_positive": None if roa_t is None else roa_t > 0,
        "f2_cfo_positive": None if cfo is None else cfo > 0,
        "f3_roa_up": _gt(roa_t, roa_p),
        "f4_cfo_above_roa": _gt(cfo_ta, roa_t),
        "f5_leverage_down": _gt(lev_p, lev_t),
        "f6_current_ratio_up": _gt(cr_t, cr_p),
        "f7_no_dilution": None if shares_t is None or shares_p is None else shares_t <= shares_p,
        "f8_gross_margin_up": _gt(gm_t, gm_p),
        "f9_turnover_up": _gt(at_t, at_p),
    }
    evaluated = [v for v in signals.values() if v is not None]
    return {
        **signals,
        "f_score": sum(evaluated),
        "f_signals": len(evaluated),
        "f_complete": len(evaluated) == len(signals),
    }


def altman_z2(t: dict[str, float]) -> dict[str, Any]:
    ta, tl = t.get("Assets"), t.get("Liabilities")
    wc = None
    if "AssetsCurrent" in t and "LiabilitiesCurrent" in t:
        wc = t["AssetsCurrent"] - t["LiabilitiesCurrent"]
    x = {
        "x1_working_capital": _ratio(wc, ta),
        "x2_retained_earnings": _ratio(t.get("RetainedEarnings"), ta),
        "x3_ebit": _ratio(t.get("EBIT"), ta),
        "x4_equity_to_liabilities": _ratio(t.get("StockholdersEquity"), tl),
    }
    if any(v is None for v in x.values()):
        return {**x, "z_double_prime": None, "z_zone": None}
    z = (
        6.56 * x["x1_working_capital"]  # type: ignore[operator]
        + 3.26 * x["x2_retained_earnings"]  # type: ignore[operator]
        + 6.72 * x["x3_ebit"]  # type: ignore[operator]
        + 1.05 * x["x4_equity_to_liabilities"]  # type: ignore[operator]
    )
    zone = "safe" if z > Z_SAFE else ("distress" if z < Z_DISTRESS else "grey")
    return {**x, "z_double_prime": z, "z_zone": zone}


NEEDED = {
    "piotroski": ("NetIncomeLoss", "NetCashProvidedByOperatingActivities", "Assets",
                  "LongTermDebtNoncurrent", "AssetsCurrent", "LiabilitiesCurrent", "Shares",
                  "GrossProfit", "Revenues"),
    "altman": ("Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent",
               "RetainedEarnings", "EBIT", "StockholdersEquity"),
}  # fmt: skip


def screen(
    pit: PointInTimeDB,
    date: DateLike,
    tickers: Sequence[str] | None = None,
    *,
    lag_days: int = 0,
) -> pd.DataFrame:
    """Piotroski F-score and Altman Z'' per company, using only data knowable on ``date``."""
    fields = [*DURATION_FIELDS, *INSTANT_FIELDS]
    rows = pit.as_of(date, tickers, fields, lag_days=lag_days)
    fy_ends = pit.fiscal_year_ends(date, tickers, lag_days=lag_days)
    out = []
    for ticker, grp in rows.groupby("ticker"):
        ends = consecutive_year_ends(list(fy_ends[fy_ends["ticker"] == ticker]["period_end"]))
        if not ends:
            continue
        years, derived, filed = [], [], []
        for e in ends:
            vals, d, f = year_values(grp, e)
            years.append(vals)
            derived.extend(d if e == ends[0] else [])
            filed.extend(f)
        missing = sorted({k for k in NEEDED["piotroski"] + NEEDED["altman"] if k not in years[0]})
        out.append(
            {
                "ticker": ticker,
                "fiscal_year_end": ends[0],
                "years_available": len(ends),
                "knowable_by": max(filed),
                **piotroski(years),
                **altman_z2(years[0]),
                "derived": "; ".join(derived),
                "missing": ", ".join(missing),
            }
        )
    return pd.DataFrame(out)
