"""Point-in-time (PIT) fundamentals database on DuckDB.

Import this module from other projects to get fundamentals **as they were knowable** on a
date, with no look-ahead::

    from edgar_extract.pit_db import PointInTimeDB

    pit = PointInTimeDB("data/pit.duckdb", read_only=True)
    pit.as_of("2022-03-01", tickers=["AAPL"], fields=["Revenues"])   # what was known then
    pit.history("GE", "Revenues")                       # every version, incl. restatements
    pit.latest_vs_asof("GE", "Revenues", "2023-03-01")                # size of the look-ahead gap

Design:

* **Keyed by date, never by fiscal period alone.** The key is ``(cik, canonical_field,
  period_start, period_end, filing_date, accession)`` plus ``source`` and the text span for
  text rows. ``period_start`` extends the spec's key so a fiscal-year figure and a Q4
  figure ending on the same day do not collide. It is NULL for instants.
* **Append-only.** Every report of a value is its own row. A restated value is a new row
  with its own ``filing_date`` and ``accession``. Rows are never updated or deleted; the
  class exposes no method that could. Re-ingesting the same data inserts nothing, because
  each row's key hash is its primary key.
* **Knowable on D** means ``filing_date <= D`` (EDGAR's filing date, which already moves
  after-hours acceptances to the next business day). ``lag_days`` adds a further lag for
  backtests that trade on D.
* **as_of(D)** returns, per ``(cik, field, period_start, period_end)``, the row with the
  greatest ``filing_date <= D``. Ties go to the later accession, then to XBRL over text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb
import pandas as pd

from edgar_extract.edgar_client import filing_index_url

DateLike = str | date | datetime | pd.Timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik BIGINT PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    name VARCHAR
);
CREATE TABLE IF NOT EXISTS facts (
    row_key VARCHAR PRIMARY KEY,           -- hash of the identifying columns
    cik BIGINT NOT NULL,
    canonical_field VARCHAR NOT NULL,
    period_start DATE,                     -- NULL for instants
    period_end DATE NOT NULL,
    period_type VARCHAR NOT NULL,          -- annual | quarter | instant | other
    filing_date DATE NOT NULL,             -- the date the value became knowable
    accession VARCHAR NOT NULL,
    form VARCHAR,
    value_usd DOUBLE NOT NULL,             -- base units: USD, USD/share or shares (see unit)
    unit VARCHAR NOT NULL,
    source VARCHAR NOT NULL CHECK (source IN ('xbrl', 'text')),
    extractor VARCHAR,                     -- NULL for xbrl rows
    confidence DOUBLE,                     -- NULL when the extractor has no calibrated score
    mismatch_cause VARCHAR,                -- text rows: MATCH or the reconciliation cause
    xbrl_tag VARCHAR,
    lineage_url VARCHAR NOT NULL,          -- filing index (xbrl) or primary document (text)
    lineage_section VARCHAR,
    lineage_span_start INTEGER,
    lineage_span_end INTEGER,
    lineage_sentence VARCHAR,
    inserted_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_lookup ON facts (cik, canonical_field, filing_date);
"""

FACT_COLUMNS = [
    "row_key", "cik", "canonical_field", "period_start", "period_end", "period_type",
    "filing_date", "accession", "form", "value_usd", "unit", "source", "extractor",
    "confidence", "mismatch_cause", "xbrl_tag", "lineage_url", "lineage_section",
    "lineage_span_start", "lineage_span_end", "lineage_sentence", "inserted_at",
]  # fmt: skip
RESULT_COLUMNS = [
    "ticker", "cik", "canonical_field", "period_start", "period_end", "period_type",
    "filing_date", "accession", "form", "value_usd", "unit", "source", "extractor",
    "confidence", "mismatch_cause", "xbrl_tag", "lineage_url", "lineage_section",
    "lineage_span_start", "lineage_span_end", "lineage_sentence",
]  # fmt: skip
KEY_COLUMNS = [
    "cik", "canonical_field", "period_start", "period_end", "filing_date", "accession",
    "source", "extractor", "lineage_section", "lineage_span_start",
]  # fmt: skip


def _to_date(d: DateLike) -> date:
    ts = pd.Timestamp(d)
    if pd.isna(ts):
        raise ValueError(f"invalid date {d!r}")
    return ts.date()


def row_key(row: Mapping[Any, Any]) -> str:
    parts = []
    for col in KEY_COLUMNS:
        v = row.get(col)
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            parts.append("")
        elif isinstance(v, (pd.Timestamp, datetime, date)):
            parts.append(pd.Timestamp(v).date().isoformat())
        elif isinstance(v, float) and v.is_integer():
            parts.append(str(int(v)))
        else:
            parts.append(str(v))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _qualified(col: str) -> str:
    return "c.ticker" if col == "ticker" else f"f.{col}"


class PointInTimeDB:
    """Append-only point-in-time store. See the module docstring for the semantics."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError(
                f"{self.path} not found; build it with `edgar-extract run --stage pit`"
            )
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA)

    # -- context management ------------------------------------------------------------

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> PointInTimeDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- ingestion (insert-only) ---------------------------------------------------------

    def upsert_companies(self, companies: pd.DataFrame) -> None:
        """Ticker/name lookup. Company metadata is not point-in-time data."""
        df = companies[["cik", "ticker", "name"]].drop_duplicates("cik")
        self.con.register("incoming_companies", df)
        self.con.execute(
            "INSERT INTO companies SELECT cik, ticker, name FROM incoming_companies "
            "ON CONFLICT (cik) DO UPDATE SET ticker = excluded.ticker, name = excluded.name"
        )
        self.con.unregister("incoming_companies")

    def _insert(self, rows: pd.DataFrame) -> int:
        if rows.empty:
            return 0
        rows = rows.copy()
        rows["row_key"] = [row_key(r) for r in rows.to_dict("records")]
        rows = rows.drop_duplicates("row_key")
        rows["inserted_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        for col in FACT_COLUMNS:
            if col not in rows.columns:
                rows[col] = None
        before = self.count()
        self.con.register("incoming", rows[FACT_COLUMNS])
        self.con.execute(
            f"INSERT INTO facts ({', '.join(FACT_COLUMNS)}) "
            f"SELECT {', '.join(FACT_COLUMNS)} FROM incoming ON CONFLICT DO NOTHING"
        )
        self.con.unregister("incoming")
        return self.count() - before

    def ingest_xbrl(self, canonical: pd.DataFrame) -> int:
        """Insert canonical XBRL facts (``xbrl.canonicalize`` output). Returns new rows."""
        if canonical.empty:
            return 0
        df = pd.DataFrame(
            {
                "cik": canonical["cik"].astype("int64"),
                "canonical_field": canonical["canonical_field"],
                "period_start": canonical["period_start"],
                "period_end": canonical["period_end"],
                "period_type": canonical["period_type"],
                "filing_date": canonical["filed"],
                "accession": canonical["accession"],
                "form": canonical["form"],
                "value_usd": canonical["value"].astype("float64"),
                "unit": canonical["unit"],
                "source": "xbrl",
                "confidence": 1.0,
                "xbrl_tag": canonical["tag"],
            }
        )
        df = df[df["value_usd"].notna() & df["filing_date"].notna() & df["period_end"].notna()]
        df["lineage_url"] = [
            filing_index_url(c, a) for c, a in zip(df["cik"], df["accession"], strict=True)
        ]
        return self._insert(df)

    def ingest_text(self, reconciliation: pd.DataFrame) -> int:
        """Insert text extractions that were tied to an XBRL period (MATCH or MISMATCH).

        Rows without a resolved period (NO_XBRL_COUNTERPART) cannot be keyed by
        ``period_end`` and are not stored.
        """
        rec = reconciliation[reconciliation["status"].isin(["MATCH", "MISMATCH"])]
        if rec.empty:
            return 0
        conf = rec["role_cue"].astype(str).str.extract(r"^p=([0-9.]+)$")[0].astype(float)
        df = pd.DataFrame(
            {
                "cik": rec["cik"].astype("int64"),
                "canonical_field": rec["canonical_field"],
                "period_start": rec["period_start"],
                "period_end": rec["period_end"],
                "period_type": "annual",
                "filing_date": rec["filing_date"],
                "accession": rec["accession"],
                "form": rec["form"],
                "value_usd": rec["value"].astype("float64"),
                "unit": "USD",
                "source": "text",
                "extractor": rec["extractor"],
                "confidence": conf,
                "mismatch_cause": rec["mismatch_cause"].fillna(rec["status"]),
                "xbrl_tag": rec["xbrl_tag"],
                "lineage_url": rec["primary_doc_url"],
                "lineage_section": rec["section"],
                "lineage_span_start": rec["span_start"].astype("int64"),
                "lineage_span_end": rec["span_end"].astype("int64"),
                "lineage_sentence": rec["sentence"],
            }
        )
        df.loc[df["period_start"].notna() & df["period_end"].notna(), "period_type"] = "annual"
        return self._insert(df[df["period_end"].notna()])

    # -- queries -------------------------------------------------------------------------

    def count(self) -> int:
        row = self.con.execute("SELECT count(*) FROM facts").fetchone()
        return int(row[0]) if row else 0

    def _filters(
        self,
        tickers: Sequence[str] | None,
        fields: Sequence[str] | None,
        period_types: Sequence[str] | None,
        sources: Sequence[str] | None,
        exclude_causes: Sequence[str] | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def among(column: str, values: Sequence[str]) -> None:
            clauses.append(f"{column} IN ({', '.join('?' for _ in values)})")
            params.extend(values)

        if tickers is not None:
            among("c.ticker", [t.upper() for t in tickers])
        if fields is not None:
            among("f.canonical_field", list(fields))
        if period_types is not None:
            among("f.period_type", list(period_types))
        if sources is not None:
            among("f.source", list(sources))
        if exclude_causes:
            clauses.append(
                f"(f.mismatch_cause IS NULL OR f.mismatch_cause NOT IN "
                f"({', '.join('?' for _ in exclude_causes)}))"
            )
            params.extend(exclude_causes)
        return (" AND ".join(clauses) or "TRUE"), params

    def as_of(
        self,
        date: DateLike,
        tickers: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        *,
        period_types: Sequence[str] | None = ("annual", "instant"),
        sources: Sequence[str] | None = ("xbrl",),
        exclude_causes: Sequence[str] | None = ("EXTRACTION_ERROR",),
        lag_days: int = 0,
    ) -> pd.DataFrame:
        """Latest value per (cik, field, period) knowable on ``date``.

        Defaults return annual and balance-sheet XBRL values, the right input for
        screeners. Pass ``sources=("xbrl", "text")`` to include text-extracted values
        (``EXTRACTION_ERROR`` rows are excluded unless ``exclude_causes`` is cleared), and
        ``period_types=None`` for every period type. ``lag_days`` treats a filing as
        knowable only ``lag_days`` after its filing date.
        """
        cutoff = _to_date(date) - timedelta(days=lag_days)
        where, params = self._filters(tickers, fields, period_types, sources, exclude_causes)
        sql = f"""
            WITH known AS (
                SELECT c.ticker, f.*,
                       row_number() OVER (
                           PARTITION BY f.cik, f.canonical_field, f.period_start, f.period_end
                           ORDER BY f.filing_date DESC, f.accession DESC,
                                    CASE f.source WHEN 'xbrl' THEN 0 ELSE 1 END,
                                    f.row_key
                       ) AS rn
                FROM facts f JOIN companies c USING (cik)
                WHERE f.filing_date <= ? AND {where}
            )
            SELECT {", ".join(RESULT_COLUMNS)} FROM known WHERE rn = 1
            ORDER BY ticker, canonical_field, period_end, period_start
        """
        return self.con.execute(sql, [cutoff, *params]).df()

    def history(
        self,
        ticker: str,
        field: str,
        *,
        period_types: Sequence[str] | None = ("annual", "instant"),
        sources: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Every stored version of a field for one company, oldest filing first."""
        where, params = self._filters([ticker], [field], period_types, sources, None)
        sql = f"""
            SELECT {", ".join(_qualified(col) for col in RESULT_COLUMNS)}
            FROM facts f JOIN companies c USING (cik)
            WHERE {where}
            ORDER BY f.period_end, f.period_start, f.filing_date, f.accession, f.source
        """
        return self.con.execute(sql, params).df()

    def latest_vs_asof(
        self,
        ticker: str,
        field: str,
        date: DateLike,
        *,
        period_types: Sequence[str] | None = ("annual", "instant"),
    ) -> pd.DataFrame:
        """Per period known on ``date``: the as-of value next to today's latest value.

        Non-zero ``change`` is the look-ahead gap: what a backtest that used today's
        restated numbers would have seen, but an investor on ``date`` could not.
        """
        then = self.as_of(date, [ticker], [field], period_types=period_types)
        now = self.as_of("9999-12-31", [ticker], [field], period_types=period_types)
        key = ["period_start", "period_end"]
        merged = then.merge(now, on=key, how="left", suffixes=("_asof", "_latest"))
        out = pd.DataFrame(
            {
                "ticker": ticker.upper(),
                "canonical_field": field,
                "period_start": merged["period_start"],
                "period_end": merged["period_end"],
                "asof_value": merged["value_usd_asof"],
                "asof_filing_date": merged["filing_date_asof"],
                "asof_accession": merged["accession_asof"],
                "latest_value": merged["value_usd_latest"],
                "latest_filing_date": merged["filing_date_latest"],
                "latest_accession": merged["accession_latest"],
            }
        )
        out["change"] = out["latest_value"] - out["asof_value"]
        out["rel_change"] = out["change"] / out["asof_value"].abs().where(out["asof_value"] != 0)
        out["restated_since"] = out["latest_accession"] != out["asof_accession"]
        out.loc[out["change"].abs() < 0.5, "restated_since"] = False  # same value re-reported
        return out

    def fiscal_year_ends(
        self, date: DateLike, tickers: Sequence[str] | None = None, *, lag_days: int = 0
    ) -> pd.DataFrame:
        """Fiscal-year ends per company: annual periods reported in a 10-K filed by ``date``.

        Year-long durations also appear in 10-Q trailing-twelve-month figures and in proxy
        pay-versus-performance tables; only 10-K-family filings define fiscal years here.
        """
        cutoff = _to_date(date) - timedelta(days=lag_days)
        where, params = self._filters(tickers, ["NetIncomeLoss"], ("annual",), ("xbrl",), None)
        sql = f"""
            SELECT DISTINCT c.ticker, f.period_end
            FROM facts f JOIN companies c USING (cik)
            WHERE f.filing_date <= ? AND f.form LIKE '10-K%' AND {where}
            ORDER BY c.ticker, f.period_end
        """
        return self.con.execute(sql, [cutoff, *params]).df()

    def restatements(
        self, tickers: Sequence[str] | None = None, fields: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """(company, field, period) whose XBRL value changed between filings, with all values."""
        where, params = self._filters(tickers, fields, ("annual", "instant"), ("xbrl",), None)
        sql = f"""
            SELECT c.ticker, f.canonical_field, f.period_start, f.period_end,
                   count(DISTINCT round(f.value_usd)) AS n_values,
                   min(f.filing_date) AS first_filed, max(f.filing_date) AS last_filed,
                   list(DISTINCT f.value_usd ORDER BY f.value_usd) AS values_usd
            FROM facts f JOIN companies c USING (cik)
            WHERE {where}
            GROUP BY ALL
            HAVING count(DISTINCT round(f.value_usd)) > 1
            ORDER BY c.ticker, f.canonical_field, f.period_end
        """
        return self.con.execute(sql, params).df()

    def fields(self) -> list[str]:
        return [
            r[0]
            for r in self.con.execute(
                "SELECT DISTINCT canonical_field FROM facts ORDER BY 1"
            ).fetchall()
        ]

    def tickers(self) -> list[str]:
        return [
            r[0] for r in self.con.execute("SELECT ticker FROM companies ORDER BY 1").fetchall()
        ]


def build_pit_db(
    path: Path,
    canonical: pd.DataFrame,
    companies: pd.DataFrame,
    reconciliations: Iterable[pd.DataFrame] = (),
) -> dict[str, int]:
    """Create or extend the PIT DB. Safe to rerun: existing rows are never touched."""
    with PointInTimeDB(path) as pit:
        pit.upsert_companies(companies)
        n_xbrl = pit.ingest_xbrl(canonical)
        n_text = sum(pit.ingest_text(r) for r in reconciliations)
        return {"new_xbrl_rows": n_xbrl, "new_text_rows": n_text, "total_rows": pit.count()}
