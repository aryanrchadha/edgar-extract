"""Why point-in-time data matters: one ROA signal, backtested with and without look-ahead.

THIS IS A DEMONSTRATION OF LOOK-AHEAD BIAS, NOT A TRADING CLAIM. The sample is 20
large-cap US stocks chosen in 2026 (survivorship bias), rebalanced once a year, equal
weighted, with no costs, over about 15 annual periods. None of that supports a statement
about alpha.

The same signal is computed three ways each May 1:

* **PIT**: ``PointInTimeDB.as_of(date)``. Each company's latest *annual* net income and
  total assets whose 10-K was **filed** on or before the rebalance date, as originally
  reported.
* **Naive, restated values only**: the same fiscal periods as PIT, but with today's
  latest restated values (later recasts applied retroactively).
* **Naive (typical)**: the latest fiscal year that **ended** on or before the rebalance
  date (ignoring the filing lag) with today's restated values. This is what a backtest
  built on a "latest fundamentals" table does.

Signal: ROA = net income / total assets at the same period end. Long the top quintile and
short the bottom quintile; a decile sort is meaningless with 20 names. Hold 12 months.
Prices are yfinance dividend- and split-adjusted closes, cached under ``data/raw/yfinance``.

    uv sync --extra examples
    uv run python examples/pit_vs_naive.py        # writes reports/pit_vs_naive.md
"""

from __future__ import annotations

import hashlib
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from edgar_extract.config import load_settings
from edgar_extract.pit_db import PointInTimeDB

ROOT = Path(__file__).resolve().parents[1]
YEARS = range(2011, 2026)
# The primary schedule was fixed before running anything. The second one is a sensitivity
# check inside 10-K filing season, where the timing part of look-ahead is largest. Both
# are always reported together.
SCHEDULES = {
    "May 1 (primary)": (5, 1),
    "Feb 15 (sensitivity: inside 10-K filing season)": (2, 15),
}
PRICE_START = date(2011, 1, 1)
PRICE_END = date(2026, 6, 1)
QUANTILES = 5
FIELDS = ["NetIncomeLoss", "Assets"]


# -- fundamentals --------------------------------------------------------------------------


def roa_table(
    rows: pd.DataFrame, chosen_periods: pd.DataFrame | None, cutoff: date | None
) -> pd.DataFrame:
    """ROA per ticker from as_of-style rows.

    ``chosen_periods`` (ticker, period_end) fixes which fiscal year to use. Otherwise the
    latest annual period is used, optionally limited to ``period_end <= cutoff``.
    """
    ni = rows[(rows["canonical_field"] == "NetIncomeLoss") & (rows["period_type"] == "annual")]
    # Two "annual" durations can end on the same day (a 52/53-week year next to a transition
    # or 12-month figure): keep the one closest to 365 days.
    ni = ni.assign(off=((ni["period_end"] - ni["period_start"]).dt.days - 364).abs())
    ni = ni.sort_values("off").drop_duplicates(["ticker", "period_end"])
    assets = rows[rows["canonical_field"] == "Assets"][
        ["ticker", "period_end", "value_usd", "filing_date"]
    ]
    if cutoff is not None:
        ni = ni[ni["period_end"] <= pd.Timestamp(cutoff)]
    if chosen_periods is not None:
        ni = ni.merge(chosen_periods, on=["ticker", "period_end"])
    else:
        ni = ni.sort_values("period_end").groupby("ticker").tail(1)
    out = ni[["ticker", "period_end", "value_usd", "filing_date"]].merge(
        assets, on=["ticker", "period_end"], suffixes=("_ni", "_assets")
    )
    out["roa"] = out["value_usd_ni"] / out["value_usd_assets"]
    return out.rename(columns={"value_usd_ni": "net_income", "value_usd_assets": "assets"})


def signals(pit: PointInTimeDB, d: date, latest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    as_of = pit.as_of(d, fields=FIELDS)
    pit_roa = roa_table(as_of, None, None)
    periods = pit_roa[["ticker", "period_end"]]
    return {
        "pit": pit_roa,
        "restated_only": roa_table(latest, periods, None),
        "naive": roa_table(latest, None, d),
    }


# -- prices ----------------------------------------------------------------------------------


def load_prices(tickers: list[str], cache_dir: Path) -> pd.DataFrame:
    """Daily adjusted closes, downloaded once and cached as Parquet."""
    key = hashlib.sha256((",".join(sorted(tickers)) + str(PRICE_START) + str(PRICE_END)).encode())
    path = cache_dir / f"adj_close_{key.hexdigest()[:12]}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.download(
        sorted(tickers),
        start=str(PRICE_START),
        end=str(PRICE_END),
        auto_adjust=True,
        progress=False,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes.dropna(how="all")
    cache_dir.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(path)
    return closes


def price_on(prices: pd.DataFrame, ticker: str, d: date) -> float | None:
    """Last close on or before ``d`` (None if the stock was not trading yet)."""
    if ticker not in prices.columns:
        return None
    s = prices[ticker].loc[: pd.Timestamp(d)].dropna()
    if s.empty or (pd.Timestamp(d) - s.index[-1]).days > 7:
        return None
    return float(s.iloc[-1])


# -- backtest ----------------------------------------------------------------------------------


def long_short(
    roa: pd.DataFrame, returns: dict[str, float]
) -> tuple[float, list[str], list[str], int]:
    df = roa[roa["ticker"].isin(returns)].sort_values("roa", ascending=False)
    n = len(df)
    k = n // QUANTILES
    if k == 0:
        return math.nan, [], [], n
    top = df["ticker"].iloc[:k].tolist()
    bottom = df["ticker"].iloc[-k:].tolist()
    ls = sum(returns[t] for t in top) / k - sum(returns[t] for t in bottom) / k
    return ls, top, bottom, n


def backtest(
    pit: PointInTimeDB, latest: pd.DataFrame, prices: pd.DataFrame, month: int, day: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = [date(y, month, day) for y in YEARS]
    tickers = pit.tickers()
    rows = []
    diffs = []
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else date(YEARS[-1] + 1, month, day)
        rets: dict[str, float] = {}
        for t in tickers:
            p0, p1 = price_on(prices, t, d), price_on(prices, t, end)
            if p0 and p1:
                rets[t] = p1 / p0 - 1
        sig = signals(pit, d, latest)
        record: dict[str, object] = {"rebalance": d, "hold_until": end}
        legs = {}
        for name, table in sig.items():
            ls, top, bottom, n = long_short(table, rets)
            record[f"{name}_ls"] = ls
            record[f"{name}_n"] = n
            legs[name] = (set(top), set(bottom))
        record["members_changed"] = len(
            (legs["pit"][0] ^ legs["naive"][0]) | (legs["pit"][1] ^ legs["naive"][1])
        )
        rows.append(record)
        # Which inputs differ between PIT and the typical naive backtest, and why?
        p = sig["pit"].set_index("ticker")
        nv = sig["naive"].set_index("ticker")
        r = sig["restated_only"].set_index("ticker")
        for t in p.index.intersection(nv.index):
            for col in ("net_income", "assets"):
                if p.at[t, "period_end"] != nv.at[t, "period_end"]:
                    diffs.append({"rebalance": d, "ticker": t, "field": col, "reason": "timing"})
                elif t in r.index and not math.isclose(p.at[t, col], r.at[t, col], rel_tol=1e-9):
                    diffs.append({"rebalance": d, "ticker": t, "field": col, "reason": "restated"})
    return pd.DataFrame(rows), pd.DataFrame(
        diffs, columns=["rebalance", "ticker", "field", "reason"]
    )


def main() -> int:
    settings = load_settings(ROOT / "config.yaml")
    with PointInTimeDB(settings.paths.pit_db, read_only=True) as pit:
        latest = pit.as_of("9999-12-31", fields=FIELDS)
        prices = load_prices(pit.tickers(), settings.paths.raw_dir / "yfinance")
        sections = [
            (name, *backtest(pit, latest, prices, month, day))
            for name, (month, day) in SCHEDULES.items()
        ]
    report = render(sections)
    out = settings.paths.reports_dir / "pit_vs_naive.md"
    out.write_text(report)
    print(report)
    print(f"wrote {out}")
    return 0


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


def render_schedule(name: str, res: pd.DataFrame, diff: pd.DataFrame) -> list[str]:
    n = len(res)
    gap = res["naive_ls"] - res["pit_ls"]
    se = gap.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
    total_inputs = int(res["pit_n"].sum() * 2)
    restated = diff[diff["reason"] == "restated"].groupby("ticker").size()
    lines = [
        f"## Rebalance each {name}",
        "",
        "| rebalance | stocks | PIT L/S | naive (restated values only) | naive (typical) "
        "| naive - PIT | quintile members changed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res.to_dict("records"):
        cells = [
            r["rebalance"], r["pit_n"], pct(r["pit_ls"]), pct(r["restated_only_ls"]),
            pct(r["naive_ls"]), pct(r["naive_ls"] - r["pit_ls"]), r["members_changed"],
        ]  # fmt: skip
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += [
        "",
        "| backtest | mean annual long-short |",
        "|---|---|",
        f"| PIT (as_of) | {pct(res['pit_ls'].mean())} |",
        f"| naive, restated values only | {pct(res['restated_only_ls'].mean())} |",
        f"| naive, typical (fiscal-year end + restated values) | {pct(res['naive_ls'].mean())} |",
        "",
        f"Mean naive - PIT gap: **{pct(gap.mean())} per year** (standard error {pct(se)}, "
        f"n = {n} periods); mean absolute yearly gap {gap.abs().mean():.1%}, largest "
        f"{pct(gap.loc[gap.abs().idxmax()])}. Quintile membership differed in "
        f"{int((res['members_changed'] > 0).sum())} of {n} years.",
        "",
        f"Inputs that differ: **{len(diff)} of {total_inputs}** ROA inputs (net income and "
        f"assets, per company and rebalance): **{int((diff['reason'] == 'timing').sum())}** "
        "because the naive backtest used a fiscal year whose 10-K was not filed yet, and "
        f"**{int((diff['reason'] == 'restated').sum())}** because a later filing restated or "
        "recast the value. Restated inputs by company: "
        + (", ".join(f"{t} ({c})" for t, c in restated.items()) or "none")
        + ".",
        "",
    ]
    return lines


def render(sections: list[tuple[str, pd.DataFrame, pd.DataFrame]]) -> str:
    n = len(sections[0][1])
    lines = [
        "# PIT vs naive: look-ahead bias in a simple ROA signal",
        "",
        "> **A demonstration of look-ahead bias, not a trading claim.** 20 large-cap US stocks",
        "> picked in 2026 (survivorship bias), annual rebalancing, equal weights, no costs,",
        f"> **{n} annual periods**, fewer than 30, so no return figure here is statistically",
        "> meaningful. The point is how much the inputs and the answer change.",
        "",
        "Generated by `examples/pit_vs_naive.py`. ROA = net income / total assets (same period",
        "end); long the top quintile and short the bottom quintile (3-4 names per leg), held for",
        "one year. Prices are yfinance adjusted closes.",
        "",
        "The May 1 schedule was chosen before running. Feb 15 is reported as a sensitivity",
        "check: it falls inside 10-K filing season, where using a fiscal year before its 10-K",
        "exists (the timing part of look-ahead) matters most.",
        "",
    ]
    for name, res, diff in sections:
        lines += render_schedule(name, res, diff)
    lines += ["## Reading this", ""]
    for name, res, diff in sections:
        gap = res["naive_ls"] - res["pit_ls"]
        total = int(res["pit_n"].sum() * 2)
        timing = int((diff["reason"] == "timing").sum())
        restated = int((diff["reason"] == "restated").sum())
        changed = int((res["members_changed"] > 0).sum())
        lines.append(
            f"- **{name}:** {len(diff)} of {total} inputs differ ({timing} timing, {restated} "
            f"restated); the quintiles differ in {changed} of {len(res)} years, and the yearly "
            f"long-short return differs by {gap.abs().mean():.1%} on average."
        )
    lines += [
        "- Neither average gap is distinguishable from zero over 15 periods, so this does not",
        "  show that look-ahead makes this signal look better or worse. What it shows is that",
        "  the naive backtest is a different strategy, holding different stocks in many years,",
        "  and that how different depends on when you rebalance relative to filing season.",
        "  The PIT DB removes that dependence instead of hoping it is small.",
        "",
    ]
    lines += [
        "## Limits",
        "",
        "- Survivorship: today's large caps, so they all survived and grew. Both backtests share",
        "  this bias; it does not affect the comparison, but it makes the absolute returns",
        "  meaningless.",
        "- 20 stocks, 3-4 per leg, 15 periods: one company can swing a year's spread.",
        "- yfinance is an unofficial source of adjusted prices; spot-check before relying on it.",
        "- Fiscal years differ (AAPL ends in September, MSFT in June, WMT and NVDA in January),",
        "  which is exactly why the filing date, not the fiscal-year label, has to drive timing.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
