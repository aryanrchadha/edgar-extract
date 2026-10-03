from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from edgar_extract.config import (
    ConfigError,
    SecConfig,
    Settings,
    UniverseConfig,
    normalize_ticker,
    sic_matches,
)


def test_user_agent_requires_email() -> None:
    with pytest.raises(ConfigError):
        SecConfig(contact_email="").user_agent()
    with pytest.raises(ConfigError):
        SecConfig(contact_email="not-an-email").user_agent()
    assert SecConfig(contact_email="a@b.com").user_agent().endswith("a@b.com")


def test_rate_limit_must_stay_under_sec_cap() -> None:
    with pytest.raises(ValidationError):
        SecConfig(max_requests_per_second=10)
    assert SecConfig().max_requests_per_second == 8.0


def test_env_email_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_EXTRACT_EMAIL", "env@example.org")
    assert Settings().sec.contact_email == "env@example.org"


def test_ticker_normalization_and_dedup() -> None:
    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert UniverseConfig(tickers=["aapl", "AAPL", "brk.b"]).tickers == ["AAPL", "BRK-B"]


def test_sic_filter() -> None:
    assert sic_matches(6021, ["6000-6999"], [])
    assert not sic_matches(3571, ["6000-6999"], [])
    assert not sic_matches(6021, [], ["6021"])
    assert sic_matches(None, [], [])
    assert not sic_matches(None, ["3571"], [])
    with pytest.raises(ValidationError):
        UniverseConfig(sic_include=["banks"])


def test_load_repo_config_resolves_paths(tmp_path: Path) -> None:
    repo_cfg = Path(__file__).parents[1] / "config.yaml"
    s = Settings.load(repo_cfg, base_dir=tmp_path)
    assert len(s.universe.tickers) == 20
    assert s.paths.raw_dir == tmp_path / "data" / "raw"


def test_missing_config_file() -> None:
    with pytest.raises(ConfigError):
        Settings.load("does-not-exist.yaml")
