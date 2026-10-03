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
from edgar_extract.pipeline import (
    extractions_path,
    run_extract_stage,
    run_pit_stage,
    run_qa_stage,
    run_reconcile_stage,
    run_sections_stage,
    run_xbrl_stage,
    section_report_path,
    sections_path,
)
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
        if stage in (Stage.extract, Stage.all):
            _print_extractions(run_extract_stage(settings), extractions_path(settings))
        if stage in (Stage.xbrl, Stage.all):
            counts = run_xbrl_stage(settings, client)
            console.print(f"xbrl: {counts}")
        if stage in (Stage.reconcile, Stage.all):
            _print_reconciliation(run_reconcile_stage(settings))
        if stage in (Stage.pit, Stage.all):
            console.print(f"pit: {run_pit_stage(settings)} -> {settings.paths.pit_db}")
        if stage in (Stage.qa, Stage.all):
            console.print(f"qa: wrote {run_qa_stage(settings)}")
    except (ConfigError, EdgarError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


class Extractor(StrEnum):
    regex = "regex"
    ner = "ner"


@app.command()
def extract(
    ticker: Annotated[str, typer.Option(help="Ticker to extract")],
    years: Annotated[int, typer.Option(help="Number of most recent fiscal years")] = 5,
    extractor: Annotated[Extractor, typer.Option(help="Extractor to use")] = Extractor.regex,
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Extract numeric entities from one company's isolated 10-K sections.

    Needs the sections stage to have run (``edgar-extract run --stage sections``).
    """
    settings = _load(config)
    name = f"extractions_{ticker.upper()}_{extractor.value}"
    try:
        df = run_extract_stage(
            settings,
            extractor=extractor.value,
            tickers=[ticker.upper()],
            years=years,
            output_name=name,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if df.empty:
        console.print(
            f"[yellow]no extractions for {ticker.upper()}; is it in the universe?[/yellow]"
        )
        raise typer.Exit(1)
    _print_extractions(df, extractions_path(settings, name))


RECONCILIATION_HINT = "data/interim/reconciliation.parquet"


def _print_reconciliation(df: pd.DataFrame) -> None:
    table = Table(title="Text vs XBRL reconciliation")
    for col in ("status", "cause", "count", "share"):
        table.add_column(col)
    counts = df.groupby(["status", "mismatch_cause"], dropna=False).size().reset_index(name="n")
    for row in counts.to_dict("records"):
        cause = row["mismatch_cause"]
        cause_text = "" if pd.isna(cause) else str(cause)
        n = int(row["n"])
        table.add_row(str(row["status"]), cause_text, str(n), f"{n / len(df):.1%}")
    console.print(table)
    console.print(
        f"{len(df):,} role-bearing extractions reconciled; written to {RECONCILIATION_HINT}"
    )


def _print_extractions(df: pd.DataFrame, path: Path) -> None:
    table = Table(title="Extractions by role and section")
    for col in ("role", "Item 1A", "Item 7", "total"):
        table.add_column(col)
    counts = df.pivot_table(index="role", columns="section", values="span_start", aggfunc="count")
    counts = counts.reindex(columns=["1A", "7"]).fillna(0).astype(int)
    for role, row in counts.iterrows():
        table.add_row(str(role), str(row["1A"]), str(row["7"]), str(row.sum()))
    console.print(table)
    structured = (df["role"] != "OTHER_NUMERIC").mean() if len(df) else 0.0
    console.print(
        f"{len(df):,} numeric spans from {df['accession'].nunique()} filings; "
        f"{structured:.1%} assigned a specific role; written to {path}"
    )


@app.command()
def label(
    prepare: Annotated[
        bool, typer.Option("--prepare", help="Build silver labels and the gold candidate sample")
    ] = False,
    status: Annotated[bool, typer.Option("--status", help="Show gold labeling progress")] = False,
    labeler: Annotated[
        str, typer.Option(help="Your name or initials, stored with each label")
    ] = "",
    target: Annotated[int, typer.Option(help="Number of sentences to aim for")] = 200,
    force: Annotated[bool, typer.Option(help="Resample candidates (only before labeling)")] = False,
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Gold labeling tool: one keypress per candidate number. See labels/LABELING_GUIDE.md."""
    import os

    from edgar_extract.ner.labeling import gold_summary, read_jsonl, run_labeling
    from edgar_extract.pipeline import labels_dir, prepare_labeling

    settings = _load(config)
    folder = labels_dir(settings)
    gold_file = folder / "gold_labels.jsonl"
    if prepare:
        try:
            client = EdgarClient.from_settings(settings)
            info = prepare_labeling(settings, client, force=force)
        except (FileNotFoundError, RuntimeError, EdgarError) as exc:
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(1) from exc
        for key, value in info.items():
            console.print(f"{key}: {value}")
        return
    if status:
        console.print(gold_summary(read_jsonl(gold_file)))
        return
    name = labeler or os.environ.get("USER", "unknown")
    try:
        run_labeling(folder / "gold_candidates.jsonl", gold_file, name, target=target)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print(f"\nstopped; progress is saved in {gold_file}")


ner_app = typer.Typer(help="Train and evaluate the token classifier (needs the `ner` extra).")
app.add_typer(ner_app, name="ner")


@ner_app.command("train")
def ner_train(
    config: ConfigOpt = Path("config.yaml"),
    epochs: Annotated[int | None, typer.Option(help="Override the configured epochs")] = None,
) -> None:
    """Fine-tune DistilBERT on silver + gold-train labels (gold test companies held out)."""
    from edgar_extract.ner.run import run_training

    settings = _load(config)
    try:
        report = run_training(settings, epochs=epochs)
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(report)


@ner_app.command("evaluate")
def ner_evaluate(config: ConfigOpt = Path("config.yaml")) -> None:
    """Score regex and NER on the gold test split; write reports/ner_metrics.md."""
    from edgar_extract.ner.run import run_evaluation

    settings = _load(config)
    try:
        path = run_evaluation(settings)
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(path.read_text())


@app.command()
def asof(
    ticker: Annotated[str, typer.Option(help="Ticker")],
    date: Annotated[str, typer.Option(help="As-of date, YYYY-MM-DD")],
    field: Annotated[str, typer.Option(help="Canonical field, e.g. Revenues")],
    include_text: Annotated[
        bool, typer.Option(help="Also consider text-extracted values (XBRL wins ties)")
    ] = False,
    lag_days: Annotated[int, typer.Option(help="Extra days before a filing counts")] = 0,
    config: ConfigOpt = Path("config.yaml"),
) -> None:
    """Show what was knowable about a field on a date, next to today's latest value."""
    from edgar_extract.pit_db import PointInTimeDB

    settings = _load(config)
    try:
        pit = PointInTimeDB(settings.paths.pit_db, read_only=True)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    with pit:
        if field not in pit.fields():
            console.print(f"[red]unknown field {field!r}.[/red] Fields: {', '.join(pit.fields())}")
            raise typer.Exit(2)
        sources = ("xbrl", "text") if include_text else ("xbrl",)
        known = pit.as_of(date, [ticker], [field], sources=sources, lag_days=lag_days)
        gap = pit.latest_vs_asof(ticker, field, date)
    if known.empty:
        console.print(f"No {field} for {ticker.upper()} was filed on or before {date}.")
        raise typer.Exit(0)
    table = Table(title=f"{ticker.upper()} {field} as knowable on {date}")
    for col in (
        "period",
        "value (as of)",
        "filed",
        "accession",
        "form",
        "source",
        "latest today",
        "change",
    ):
        table.add_column(col, no_wrap=True)
    latest = {(r["period_start"], r["period_end"]): r for r in gap.to_dict("records")}
    for r in known.to_dict("records"):
        start = (
            "" if pd.isna(r["period_start"]) else f"{pd.Timestamp(r['period_start']).date()} to "
        )
        period = f"{start}{pd.Timestamp(r['period_end']).date()}"
        g = latest.get((r["period_start"], r["period_end"]))
        later = "" if g is None or pd.isna(g["latest_value"]) else f"{g['latest_value']:,.0f}"
        change = "" if g is None or not g["restated_since"] else f"{g['rel_change']:+.1%}"
        table.add_row(
            period, f"{r['value_usd']:,.0f}", str(pd.Timestamp(r["filing_date"]).date()),
            str(r["accession"]), str(r["form"]), str(r["source"]), later, change,
        )  # fmt: skip
    console.print(table)
    console.print("[dim]'latest today' is what a backtest using current data would see.[/dim]")


@app.command()
def dashboard(
    config: ConfigOpt = Path("config.yaml"),
    port: Annotated[int, typer.Option(help="Local port")] = 8501,
) -> None:
    """Launch the Streamlit QA dashboard on the built artifacts (needs the `dashboard` extra)."""
    import os
    import subprocess

    settings = _load(config)
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    if importlib_find("streamlit") is None:
        console.print("[red]streamlit is not installed.[/red] Run `uv sync --extra dashboard`.")
        raise typer.Exit(1)
    env = {**os.environ, "EDGAR_EXTRACT_CONFIG": str(config.resolve())}
    # headless: no interactive first-run prompt and no browser launch; open the URL printed.
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--browser.gatherUsageStats", "false", "--server.headless", "true",
        "--server.port", str(port),
    ]  # fmt: skip
    console.print(f"artifacts: {settings.paths.interim_dir}; open http://localhost:{port}")
    raise typer.Exit(subprocess.call(cmd, env=env))


def importlib_find(name: str) -> object | None:
    from importlib.util import find_spec

    return find_spec(name)


if __name__ == "__main__":
    app()
