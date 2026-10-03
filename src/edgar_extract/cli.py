"""Command-line interface: ``edgar-extract run | extract | asof | dashboard``."""

from __future__ import annotations

import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from edgar_extract.config import ConfigError, Settings, load_settings
from edgar_extract.edgar_client import EdgarClient, EdgarError
from edgar_extract.pipeline import run_sections_stage, section_report_path, sections_path
from edgar_extract.skiplog import SkipLog
from edgar_extract.universe import STAGE as UNIVERSE_STAGE
from edgar_extract.universe import build_universe

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
# Wide output when piped (CI logs, files) so accession numbers are never truncated.
console = Console(width=None if sys.stdout.isatty() else 160)

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Path to config.yaml", show_default=True)
]


class Stage(StrEnum):
    universe = "universe"
    sections = "sections"
    extract = "extract"
    xbrl = "xbrl"
    reconcile = "reconcile"
    pit = "pit"
    qa = "qa"
    all = "all"


# Stages land milestone by milestone; until then the CLI says so instead of pretending.
_NOT_YET: dict[Stage, str] = {
    Stage.extract: "M3",
    Stage.xbrl: "M5",
    Stage.reconcile: "M5",
    Stage.pit: "M6",
    Stage.qa: "M7",
}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load(config: Path) -> Settings:
    try:
        return load_settings(config)
    except (ConfigError, ValueError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _not_implemented(what: str, milestone: str) -> typer.Exit:
    console.print(f"[yellow]{what} is not implemented yet (lands in {milestone}).[/yellow]")
    return typer.Exit(2)


def run_universe_stage(settings: Settings, client: EdgarClient) -> pd.DataFrame:
    skiplog = SkipLog(settings.paths.skip_log)
    skiplog.clear_stage(UNIVERSE_STAGE)
    df = build_universe(
        client,
        settings.universe.tickers,
        years=settings.years,
        sic_filter=settings.universe.sic_filter,
        cik_overrides=settings.universe.cik_overrides,
        skiplog=skiplog,
    )
    out = settings.paths.universe_parquet
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _print_universe(df)
    skips = skiplog.for_stage(UNIVERSE_STAGE)
    if skips:
        table = Table(title=f"Skipped in universe stage ({len(skips)})")
        for col in ("ticker", "accession", "reason", "detail"):
            table.add_column(col, no_wrap=col in ("ticker", "accession"))
        for e in skips:
            table.add_row(e.ticker or "", e.accession or "", e.reason, e.detail or "")
        console.print(table)
    console.print(
        f"wrote {len(df)} filings for {df['ticker'].nunique() if len(df) else 0} companies to {out}"
    )
    console.print(
        f"requests: {client.stats.network_fetches} network, {client.stats.cache_hits} cache hits, "
        f"{client.stats.retries} retries; skip log: {settings.paths.skip_log}"
    )
    return df


def _run_sections(settings: Settings, client: EdgarClient) -> None:
    df = run_sections_stage(settings, client)
    table = Table(title="Section isolation")
    for col in ("item", "status", "method", "sections", "median chars"):
        table.add_column(col)
    grouped = df.groupby(["item", "status", "method"])["n_chars"]
    for (item, status, method), n in grouped:
        table.add_row(str(item), str(status), str(method), str(len(n)), f"{n.median():,.0f}")
    console.print(table)
    report = pd.read_csv(section_report_path(settings))
    console.print(
        f"wrote {len(df)} sections from {df['accession'].nunique()} filings to "
        f"{sections_path(settings)}; {int(report['outlier'].sum())} outliers flagged in "
        f"{section_report_path(settings)}"
    )


def _print_universe(df: pd.DataFrame) -> None:
    table = Table(title="10-K universe")
    for col in ("ticker", "fiscal year", "form", "accession", "filing date", "period end"):
        table.add_column(col, no_wrap=True)
    shown = df.sort_values(["ticker", "fiscal_year", "filing_date"]).assign(
        filing_date=lambda d: d["filing_date"].dt.strftime("%Y-%m-%d"),
        report_date=lambda d: d["report_date"].dt.strftime("%Y-%m-%d"),
    )
    cols = ["ticker", "fiscal_year", "form", "accession", "filing_date", "report_date"]
    for values in shown[cols].astype(str).itertuples(index=False):
        table.add_row(*values)
    console.print(table)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="INFO-level logging")] = False,
) -> None:
    """SEC 10-K text-to-numbers extraction with a point-in-time fundamentals database."""
    _setup_logging(verbose)


@app.command()
def run(
    config: ConfigOpt = Path("config.yaml"),
    stage: Annotated[Stage, typer.Option(help="Pipeline stage to run")] = Stage.all,
    offline: Annotated[bool, typer.Option(help="Use only cached SEC responses")] = False,
    refresh: Annotated[
        bool, typer.Option(help="Refetch mutable JSON endpoints (submissions, companyfacts)")
    ] = False,
) -> None:
    """Run the pipeline, or a single stage of it."""
    settings = _load(config)
    try:
        client = EdgarClient.from_settings(settings, offline=offline, refresh=refresh)
        if stage in (Stage.universe, Stage.all):
            run_universe_stage(settings, client)
        if stage in (Stage.sections, Stage.all):
            _run_sections(settings, client)
        if stage is Stage.all:
            console.print(
                "[yellow]Later stages are not implemented yet; "
                "ran the universe and sections stages.[/yellow]"
            )
        elif stage in _NOT_YET:
            raise _not_implemented(f"stage '{stage.value}'", _NOT_YET[stage])
    except (ConfigError, EdgarError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def extract(
    ticker: Annotated[str, typer.Option(help="Ticker to extract")],
    years: Annotated[int, typer.Option(help="Number of fiscal years")] = 5,
    extractor: Annotated[str, typer.Option(help="regex or ner")] = "regex",
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Extract numeric entities from one company's 10-K filings."""
    raise _not_implemented("extract", "M3")


@app.command()
def asof(
    ticker: Annotated[str, typer.Option(help="Ticker")],
    date: Annotated[str, typer.Option(help="As-of date, YYYY-MM-DD")],
    field: Annotated[str, typer.Option(help="Canonical field, e.g. Revenues")],
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Show what was knowable about a field on a given date."""
    raise _not_implemented("asof", "M6")


@app.command()
def dashboard(config: ConfigOpt = Path("config.yaml")) -> None:
    """Launch the Streamlit QA dashboard."""
    raise _not_implemented("dashboard", "M7")


if __name__ == "__main__":
    app()
