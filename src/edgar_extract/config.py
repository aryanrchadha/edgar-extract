"""Typed configuration loaded from ``config.yaml`` with environment-variable overrides.

Environment variables use the prefix ``EDGAR_EXTRACT_`` and ``__`` as the nesting
delimiter, e.g. ``EDGAR_EXTRACT_SEC__CONTACT_EMAIL=me@example.edu``. They can also sit in
a git-ignored ``.env`` file next to the config, which keeps a personal contact email out of
the committed ``config.yaml``. Precedence: environment, then ``.env``, then the YAML file.

Relative paths in the ``paths`` section are resolved against the directory that contains
the config file, so the CLI behaves the same from any working directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# SEC's published ceiling is 10 req/s; the spec requires staying under 8.
MAX_ALLOWED_RPS = 8.0

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_SIC_ENTRY_RE = re.compile(r"^\s*(\d{1,4})\s*(?:-\s*(\d{1,4})\s*)?$")


class ConfigError(ValueError):
    """Raised when the configuration is missing something the pipeline cannot run without."""


class SecSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str = "edgar-extract research"
    contact_email: str = ""
    max_requests_per_second: float = Field(default=7.0, gt=0, le=MAX_ALLOWED_RPS)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=5, ge=0, le=10)

    @field_validator("contact_email")
    @classmethod
    def _check_email(cls, value: str) -> str:
        value = value.strip()
        if value and not _EMAIL_RE.match(value):
            raise ValueError(f"contact_email {value!r} is not a valid email address")
        return value

    @property
    def user_agent(self) -> str:
        """User-Agent in the form SEC asks for: ``<name> <contact email>``."""
        if not self.contact_email:
            raise ConfigError(
                "sec.contact_email is empty. SEC fair-access policy requires a contact email "
                "in the User-Agent. Set it in config.yaml or EDGAR_EXTRACT_SEC__CONTACT_EMAIL."
            )
        return f"{self.app_name} {self.contact_email}"


@dataclass(frozen=True)
class SicRange:
    """Inclusive SIC code range. A single code is a range of width one."""

    low: int
    high: int

    def contains(self, sic: int) -> bool:
        return self.low <= sic <= self.high

    @classmethod
    def parse(cls, entry: str | int) -> SicRange:
        match = _SIC_ENTRY_RE.match(str(entry))
        if not match:
            raise ValueError(
                f"SIC entry {entry!r} must be a code like '3571' or a range '6000-6999'"
            )
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        if high < low:
            raise ValueError(f"SIC range {entry!r} has high < low")
        return cls(low, high)


@dataclass(frozen=True)
class SicFilter:
    include: tuple[SicRange, ...] = ()
    exclude: tuple[SicRange, ...] = ()

    def is_active(self) -> bool:
        return bool(self.include or self.exclude)

    def reason_rejected(self, sic: int | None) -> str | None:
        """Return why a company with this SIC is filtered out, or ``None`` if it passes."""
        if not self.is_active():
            return None
        if sic is None:
            return "SIC code unavailable while a SIC filter is configured"
        if self.include and not any(r.contains(sic) for r in self.include):
            return f"SIC {sic} not in sic_include"
        if any(r.contains(sic) for r in self.exclude):
            return f"SIC {sic} in sic_exclude"
        return None


class UniverseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(min_length=1)
    cik_overrides: dict[str, int] = Field(default_factory=dict)
    sic_include: list[str] = Field(default_factory=list)
    sic_exclude: list[str] = Field(default_factory=list)

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for ticker in value:
            cleaned = str(ticker).strip().upper()
            if not cleaned:
                raise ValueError("empty ticker in universe.tickers")
            seen.setdefault(cleaned, None)
        return list(seen)

    @field_validator("cik_overrides")
    @classmethod
    def _normalize_overrides(cls, value: dict[str, int]) -> dict[str, int]:
        return {str(k).strip().upper(): int(v) for k, v in value.items()}

    @field_validator("sic_include", "sic_exclude", mode="before")
    @classmethod
    def _stringify_sic(cls, value: Any) -> list[str]:
        if value is None:
            return []
        entries = [str(v) for v in value]
        for entry in entries:
            SicRange.parse(entry)  # validate early
        return entries

    @property
    def sic_filter(self) -> SicFilter:
        return SicFilter(
            include=tuple(SicRange.parse(e) for e in self.sic_include),
            exclude=tuple(SicRange.parse(e) for e in self.sic_exclude),
        )


class PathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    pit_db: Path = Path("data/pit.duckdb")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")

    def resolved(self, base: Path) -> PathSettings:
        def fix(p: Path) -> Path:
            return p if p.is_absolute() else (base / p).resolve()

        return PathSettings(**{name: fix(getattr(self, name)) for name in type(self).model_fields})

    @property
    def skip_log(self) -> Path:
        return self.interim_dir / "skipped.jsonl"

    @property
    def universe_parquet(self) -> Path:
        return self.interim_dir / "universe.parquet"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDGAR_EXTRACT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    sec: SecSettings = Field(default_factory=SecSettings)
    universe: UniverseSettings
    years: int = Field(default=5, ge=1, le=20)
    paths: PathSettings = Field(default_factory=PathSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment beats .env, which beats the YAML file (which arrives as init kwargs).
        return (env_settings, dotenv_settings, init_settings)


def load_settings(path: str | Path) -> Settings:
    """Load ``config.yaml`` (or any YAML path) and resolve relative paths against it."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    settings = Settings(_env_file=config_path.parent / ".env", **raw)
    settings.paths = settings.paths.resolved(config_path.parent)
    return settings
