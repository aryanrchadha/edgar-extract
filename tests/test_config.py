from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from edgar_extract.config import ConfigError, SicFilter, SicRange, load_settings

REPO_CONFIG = Path(__file__).parents[1] / "config.yaml"


def write_config(tmp_path: Path, **overrides: object) -> Path:
    data: dict[str, object] = {
        "sec": {"contact_email": "me@example.edu"},
        "universe": {"tickers": ["aapl", "MSFT", "AAPL"]},
    }
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_repo_config_loads() -> None:
    settings = load_settings(REPO_CONFIG)
    assert settings.sec.contact_email
    assert "@" in settings.sec.user_agent
    assert settings.sec.max_requests_per_second <= 8
    assert len(settings.universe.tickers) == 20
    assert settings.universe.cik_overrides["XOM"] == 34088
    assert settings.years == 5
    assert settings.paths.raw_dir.is_absolute()


def test_tickers_normalized_and_deduplicated(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path))
    assert settings.universe.tickers == ["AAPL", "MSFT"]


def test_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path))
    assert settings.paths.raw_dir == (tmp_path / "data" / "raw").resolve()
    assert settings.paths.skip_log == settings.paths.interim_dir / "skipped.jsonl"


def test_missing_email_blocks_user_agent(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, sec={}))
    with pytest.raises(ConfigError, match="contact email"):
        _ = settings.sec.user_agent


def test_invalid_email_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_settings(write_config(tmp_path, sec={"contact_email": "not-an-email"}))


@pytest.mark.parametrize("rate", [8.5, 10, 0])
def test_rate_above_8_or_zero_rejected(tmp_path: Path, rate: float) -> None:
    with pytest.raises(ValidationError):
        load_settings(
            write_config(
                tmp_path, sec={"contact_email": "a@b.edu", "max_requests_per_second": rate}
            )
        )


def test_environment_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_EXTRACT_SEC__CONTACT_EMAIL", "env@example.org")
    monkeypatch.setenv("EDGAR_EXTRACT_YEARS", "3")
    settings = load_settings(write_config(tmp_path))
    assert settings.sec.contact_email == "env@example.org"
    assert settings.years == 3


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_settings(write_config(tmp_path, yearz=5))


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("3571", SicRange(3571, 3571)),
        ("6000-6999", SicRange(6000, 6999)),
        (2834, SicRange(2834, 2834)),
    ],
)
def test_sic_range_parse(entry: str | int, expected: SicRange) -> None:
    assert SicRange.parse(entry) == expected


@pytest.mark.parametrize("entry", ["abc", "6999-6000", "1-2-3", ""])
def test_sic_range_parse_rejects(entry: str) -> None:
    with pytest.raises(ValueError):
        SicRange.parse(entry)


def test_sic_filter_rules() -> None:
    f = SicFilter(include=(SicRange(6000, 6999),), exclude=(SicRange(6798, 6798),))
    assert f.reason_rejected(6021) is None
    assert f.reason_rejected(3571) == "SIC 3571 not in sic_include"
    assert f.reason_rejected(6798) == "SIC 6798 in sic_exclude"
    assert f.reason_rejected(None) is not None
    assert SicFilter().reason_rejected(None) is None


def test_sic_filter_from_yaml(tmp_path: Path) -> None:
    settings = load_settings(
        write_config(tmp_path, universe={"tickers": ["X"], "sic_include": ["6000-6999", 3571]})
    )
    assert settings.universe.sic_filter.include == (SicRange(6000, 6999), SicRange(3571, 3571))
