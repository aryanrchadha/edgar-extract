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
