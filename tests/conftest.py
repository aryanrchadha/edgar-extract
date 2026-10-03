from __future__ import annotations

from pathlib import Path

import pytest

from edgar_extract.config import SecConfig, Settings
from edgar_extract.edgar_client import (
    EdgarClient,
    company_tickers_url,
    submissions_page_url,
    submissions_url,
)

FIXTURES = Path(__file__).parent / "fixtures"
SEC = FIXTURES / "sec"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a real HTTP connection."""

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access attempted in a unit test")

    monkeypatch.setattr("requests.sessions.Session.send", boom)


@pytest.fixture
def sec_config() -> SecConfig:
    return SecConfig(contact_email="tests@example.com")


@pytest.fixture
def offline_client(tmp_path: Path, sec_config: SecConfig) -> EdgarClient:
    """Offline client whose cache is pre-seeded with the synthetic SEC fixtures."""
    client = EdgarClient(sec_config, tmp_path / "raw", offline=True)
    seed = {
        company_tickers_url(): "synthetic_company_tickers.json",
        submissions_url(1111): "synthetic_submissions_alpha.json",
        submissions_url(3333): "synthetic_submissions_bank.json",
        submissions_page_url(
            "CIK0000003333-submissions-001.json"
        ): "synthetic_submissions_bank_page1.json",
    }
    for url, name in seed.items():
        client.cache.put(url, (SEC / name).read_bytes())
    return client


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings.model_validate(
        {"sec": {"contact_email": "tests@example.com"}, "universe": {"tickers": ["AAA", "BNK"]}}
    )
    s.paths = s.paths.resolve(tmp_path)
    return s
