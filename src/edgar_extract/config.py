"""Configuration: pydantic models loaded from YAML, with environment overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

EMAIL_ENV_VAR = "EDGAR_EXTRACT_EMAIL"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SIC_RE = re.compile(r"^\d{1,4}(-\d{1,4})?$")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid for the requested operation."""


class SecConfig(BaseModel):
    app_name: str = "edgar-extract research"
    contact_email: str = ""
    max_requests_per_second: float = Field(default=8.0, gt=0, le=10)

    @field_validator("max_requests_per_second")
    @classmethod
    def _stay_under_sec_limit(cls, v: float) -> float:
        if v >= 10:
            raise ValueError("SEC fair access caps traffic at 10 req/s; configure < 10")
        return v

    def user_agent(self) -> str:
        """Return the User-Agent header value. Raises ConfigError if no valid email."""
        email = self.contact_email.strip()
        if not _EMAIL_RE.match(email):
            raise ConfigError(
                "SEC requires a contact email in the User-Agent header. Set "
                f"sec.contact_email in config.yaml or the {EMAIL_ENV_VAR} environment variable."
            )
        return f"{self.app_name} {email}"


class UniverseConfig(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    cik_overrides: dict[str, int] = Field(default_factory=dict)
    sic_include: list[str] = Field(default_factory=list)
    sic_exclude: list[str] = Field(default_factory=list)

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for t in v:
            norm = normalize_ticker(t)
            if norm not in seen:
                seen.append(norm)
        return seen

    @field_validator("cik_overrides")
    @classmethod
    def _normalize_override_keys(cls, v: dict[str, int]) -> dict[str, int]:
        return {normalize_ticker(k): int(c) for k, c in v.items()}

    @field_validator("sic_include", "sic_exclude", mode="before")
    @classmethod
    def _validate_sic(cls, v: list[object] | None) -> list[str]:
        out = [str(x).strip() for x in (v or [])]
        for s in out:
            if not _SIC_RE.match(s):
                raise ValueError(f"SIC filter entry {s!r} must be a code or 'lo-hi' range")
        return out


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    pit_db: Path = Path("data/pit.duckdb")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")

    def resolve(self, base: Path) -> PathsConfig:
        """Return a copy with relative paths resolved against ``base``."""
        data = {k: (p if p.is_absolute() else base / p) for k, p in self.model_dump().items()}
        return PathsConfig(**data)

    def ensure(self) -> None:
        for p in (self.data_dir, self.raw_dir, self.interim_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    sec: SecConfig = Field(default_factory=SecConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    years: int = Field(default=5, ge=1, le=30)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @model_validator(mode="after")
    def _env_overrides(self) -> Self:
        env_email = os.environ.get(EMAIL_ENV_VAR, "").strip()
        if env_email:
            self.sec.contact_email = env_email
        return self

    @classmethod
    def load(cls, path: str | Path, base_dir: Path | None = None) -> Settings:
        """Load settings from YAML. Relative paths resolve against ``base_dir`` (default cwd)."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        settings = cls.model_validate(raw)
        settings.paths = settings.paths.resolve(base_dir or Path.cwd())
        return settings


def normalize_ticker(ticker: str) -> str:
    """Upper-case and use SEC's dash convention for share classes (BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def sic_matches(sic: int | None, include: list[str], exclude: list[str]) -> bool:
    """Apply include/exclude SIC filters. Unknown SIC passes only if no include list is set."""

    def hit(code: int, spec: str) -> bool:
        if "-" in spec:
            lo, hi = (int(x) for x in spec.split("-"))
            return lo <= code <= hi
        return code == int(spec)

    if sic is None:
        return not include
    if include and not any(hit(sic, s) for s in include):
        return False
    return not any(hit(sic, s) for s in exclude)
