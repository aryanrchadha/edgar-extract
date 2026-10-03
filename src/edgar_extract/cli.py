"""Command-line interface: ``edgar-extract``."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from edgar_extract.config import ConfigError, Settings
from edgar_extract.edgar_client import EdgarClient
from edgar_extract.skiplog import SkipLog
from edgar_extract.universe import build_universe

app = typer.Typer(no_args_is_help=True, help="SEC 10-K text extraction and point-in-time DB.")
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml")]


class Stage(StrEnum):
    universe = "universe"
    all = "all"


def _setup(config: Path, offline: bool, verbose: bool) -> tuple[Settings, EdgarClient]:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.load(config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    settings.paths.ensure()
    client = EdgarClient(settings.sec, settings.paths.raw_dir, offline=offline)
    return settings, client


def _not_yet(milestone: str) -> None:
    console.print(f"[yellow]Not implemented yet (lands in {milestone}).[/yellow]")
    raise typer.Exit(2)


@app.command()
def run(
    config: ConfigOpt = Path("config.yaml"),
    stage: Annotated[Stage, typer.Option(help="Pipeline stage to run")] = Stage.all,
    offline: Annotated[bool, typer.Option(help="Use only cached responses")] = False,
    refresh: Annotated[bool, typer.Option(help="Re-fetch JSON endpoints")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the pipeline (or a single stage) for the configured universe."""
    settings, client = _setup(config, offline, verbose)
    skips = SkipLog()
    try:
        uni = build_universe(settings, client, skips, refresh=refresh)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    finally:
        skips.write(settings.paths.interim_dir / "skipped.csv")
    out = settings.paths.interim_dir / "universe.parquet"
    uni.to_parquet(out, index=False)
    _print_universe(uni)
    console.print(
        f"{len(uni)} filings for {uni['ticker'].nunique()} companies -> {out}  "
        f"(network requests: {client.stats['network_requests']}, "
        f"cache hits: {client.stats['cache_hits']}, skips: {len(skips.entries)})"
    )
    if stage is not Stage.universe:
        console.print("[yellow]Later stages are not implemented yet (M2+).[/yellow]")


def _print_universe(uni: pd.DataFrame) -> None:
    table = Table(title="10-K universe")
    for col in ("ticker", "fiscal_year", "form", "accession", "filing_date", "period_end"):
        table.add_column(col)
    for r in uni.itertuples(index=False):
        table.add_row(
            r.ticker,
            str(r.fiscal_year),
            r.form,
            r.accession,
            str(r.filing_date),
            str(r.period_end),
        )
    console.print(table)


@app.command()
def extract(
    ticker: Annotated[str, typer.Option(help="Ticker symbol")],
    years: Annotated[int, typer.Option(help="Number of fiscal years")] = 5,
    extractor: Annotated[str, typer.Option(help="regex or ner")] = "regex",
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Extract numeric entities from one company's 10-K filings."""
    _not_yet("M3")


@app.command()
def asof(
    ticker: Annotated[str, typer.Option(help="Ticker symbol")],
    date: Annotated[str, typer.Option(help="As-of date, YYYY-MM-DD")],
    field: Annotated[str, typer.Option(help="Canonical field, e.g. Revenues")],
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Query the point-in-time database."""
    _not_yet("M6")


@app.command()
def dashboard(config: ConfigOpt = Path("config.yaml")) -> None:
    """Launch the Streamlit QA dashboard."""
    _not_yet("M7")


if __name__ == "__main__":
    app()
