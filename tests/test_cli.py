from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from edgar_extract import cli
from edgar_extract.edgar_client import EdgarClient
from edgar_extract.skiplog import SkipLog

runner = CliRunner()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "sec": {"contact_email": "me@example.edu"},
                "universe": {"tickers": ["AAPL", "XOM", "NOPE"], "cik_overrides": {"XOM": 34088}},
            }
        )
    )
    return path


@pytest.fixture
def patched_client(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., EdgarClient]
) -> None:
    monkeypatch.setattr(
        EdgarClient, "from_settings", classmethod(lambda cls, s, **kw: make_client())
    )


def test_run_universe_stage(config_path: Path, patched_client: None) -> None:
    result = runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "universe"])
    assert result.exit_code == 0, result.output
    assert "0000320193-25-000079" in result.output
    out = config_path.parent / "data" / "interim" / "universe.parquet"
    df = pd.read_parquet(out)
    assert set(df["ticker"]) == {"AAPL", "XOM"}
    assert len(df) == 10
    skips = SkipLog.read(config_path.parent / "data" / "interim" / "skipped.jsonl")
    assert [(s.ticker, s.stage) for s in skips] == [("NOPE", "universe")]


def test_rerun_does_not_duplicate_skip_log(config_path: Path, patched_client: None) -> None:
    for _ in range(2):
        runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "universe"])
    skips = SkipLog.read(config_path.parent / "data" / "interim" / "skipped.jsonl")
    assert len(skips) == 1


def test_unimplemented_stage_says_so(config_path: Path, patched_client: None) -> None:
    result = runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "pit"])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


def test_bad_config_exits_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2
    assert "config error" in result.output


def test_sections_stage_writes_parquet_report_and_logs_skips(
    config_path: Path, patched_client: None
) -> None:
    result = runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "all"])
    assert result.exit_code == 0, result.output
    interim = config_path.parent / "data" / "interim"
    sections = pd.read_parquet(interim / "sections.parquet")
    # Only AAPL FY2025 and XOM FY2023 documents are recorded fixtures.
    assert set(zip(sections["ticker"], sections["fiscal_year"], strict=True)) == {
        ("AAPL", 2025),
        ("XOM", 2023),
    }
    xom7 = sections[(sections["ticker"] == "XOM") & (sections["item"] == "7")].iloc[0]
    assert xom7["status"] == "ok" and xom7["method"] == "named_section_reference"
    report = pd.read_csv(config_path.parent / "reports" / "section_report.csv")
    assert len(report) == 4 and "outlier" in report.columns
    skips = SkipLog.read(interim / "skipped.jsonl")
    unavailable = [s for s in skips if s.reason == "filing document unavailable"]
    assert len(unavailable) == 8  # 10 filings in the universe, 2 recorded
    assert all(s.stage == "sections" and s.accession for s in unavailable)


def test_extract_command_for_one_ticker(config_path: Path, patched_client: None) -> None:
    runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "universe"])
    runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "sections"])
    result = runner.invoke(
        cli.app, ["extract", "--ticker", "xom", "--years", "5", "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output
    out = config_path.parent / "data" / "interim" / "extractions_XOM_regex.parquet"
    df = pd.read_parquet(out)
    assert set(df["ticker"]) == {"XOM"} and len(df) > 50
    for col in ("accession", "section", "sentence", "span_start", "span_end", "raw_text",
                "value_usd", "unit_scale", "role", "extractor"):  # fmt: skip
        assert col in df.columns


def test_extract_unknown_ticker_fails_cleanly(config_path: Path, patched_client: None) -> None:
    runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "universe"])
    runner.invoke(cli.app, ["run", "--config", str(config_path), "--stage", "sections"])
    result = runner.invoke(cli.app, ["extract", "--ticker", "ZZZ", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "no extractions" in result.output
