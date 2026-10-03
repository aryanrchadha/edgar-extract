"""Shared test fixtures. Network access is blocked for every test."""

from __future__ import annotations

import gzip
import json
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import requests

from edgar_extract.edgar_client import EdgarClient, RateLimiter

FIXTURES = Path(__file__).parent / "fixtures"
SEC_FIXTURES = FIXTURES / "sec"
TEST_UA = "edgar-extract tests test@example.edu"


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is disabled in unit tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


def make_response(
    url: str, status: int = 200, body: bytes = b"", headers: dict[str, str] | None = None
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = body
    resp.url = url
    resp.headers.update(headers or {})
    return resp


def fixture_bytes(url: str, root: Path = SEC_FIXTURES) -> bytes | None:
    """Recorded body for ``url``: ``<root>/<host>/<path>``, or the same path + ``.gz``."""
    parts = urlsplit(url)
    path = root / parts.netloc / parts.path.lstrip("/")
    if path.is_file():
        return path.read_bytes()
    gz = path.with_name(path.name + ".gz")
    if gz.is_file():
        return gzip.decompress(gz.read_bytes())
    return None


def fixture_filings() -> list[dict[str, Any]]:
    return list(json.loads((FIXTURES / "filings.json").read_text()))


class FixtureSession(requests.Session):
    """A ``requests.Session`` that answers from recorded files or scripted responses.

    URLs map to ``tests/fixtures/sec/<host>/<path>``. Unknown URLs return 404.
    ``script`` maps a URL to a list of responses (or exceptions) served in order before
    falling back to the fixture file.
    """

    def __init__(self, root: Path = SEC_FIXTURES) -> None:
        super().__init__()
        self.root = root
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.script: dict[str, list[requests.Response | Exception]] = {}

    def get(self, url: str | bytes, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        url = url.decode() if isinstance(url, bytes) else url
        self.calls.append((url, dict(self.headers)))
        queue = self.script.get(url)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        body = fixture_bytes(url, self.root)
        if body is not None:
            return make_response(url, 200, body)
        return make_response(url, 404, b"not found")

    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]


@pytest.fixture
def session() -> FixtureSession:
    return FixtureSession()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_client(
    tmp_path: Path, session: FixtureSession, sleeps: list[float]
) -> Iterator[Callable[..., EdgarClient]]:
    def factory(**kwargs: Any) -> EdgarClient:
        kwargs.setdefault("limiter", RateLimiter(8.0, sleep=lambda s: None))
        kwargs.setdefault("session", session)
        kwargs.setdefault("sleep", sleeps.append)
        return EdgarClient(TEST_UA, tmp_path / "raw", **kwargs)

    yield factory


@pytest.fixture
def client(make_client: Callable[..., EdgarClient]) -> EdgarClient:
    return make_client()
