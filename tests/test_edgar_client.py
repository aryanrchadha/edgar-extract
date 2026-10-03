from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import requests

from edgar_extract.config import ConfigError, SecConfig
from edgar_extract.edgar_client import (
    CacheMiss,
    DiskCache,
    EdgarClient,
    EdgarError,
    RateLimiter,
    companyfacts_url,
    filing_document_url,
    submissions_url,
)


def test_url_formats() -> None:
    assert submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert (
        companyfacts_url("320193")
        == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    )
    assert filing_document_url(320193, "0000320193-23-000106", "aapl-20230930.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def test_rate_limiter_spaces_requests() -> None:
    clock = FakeClock()
    lim = RateLimiter(8.0, clock=clock.now, sleep=clock.sleep)
    times = []
    for _ in range(40):
        lim.acquire()
        times.append(clock.t)
    gaps = [b - a for a, b in pairwise(times)]
    assert min(gaps) >= 1 / 8 - 1e-12
    # 40 requests can never fit in under (40-1)/8 seconds.
    assert times[-1] - times[0] >= 39 / 8 - 1e-9
    # In any 1-second window there are at most 8 requests.
    for i, t in enumerate(times):
        assert sum(1 for u in times[i:] if u < t + 1.0) <= 8


def test_disk_cache_roundtrip_and_paths(tmp_path: Path) -> None:
    c = DiskCache(tmp_path)
    url = "https://www.sec.gov/Archives/edgar/data/1/0001/doc.htm"
    assert c.get(url) is None
    c.put(url, b"hello")
    assert c.get(url) == b"hello"
    assert (
        c.path_for(url) == (tmp_path / "www.sec.gov/Archives/edgar/data/1/0001/doc.htm").resolve()
    )
    assert c.path_for("https://www.sec.gov/Archives/x/").name == "index.html"
    with pytest.raises(ValueError):
        c.path_for("https://www.sec.gov/../../etc/passwd")


class FakeResponse:
    def __init__(self, status: int, content: bytes = b"", headers: dict[str, str] | None = None):
        self.status_code = status
        self.content = content
        self.headers = headers or {}


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.headers: dict[str, str] = {"User-Agent": "test tests@example.com"}
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_client(
    tmp_path: Path, responses: list[Any]
) -> tuple[EdgarClient, FakeSession, list[float]]:
    session = FakeSession(responses)
    sleeps: list[float] = []
    client = EdgarClient(
        SecConfig(contact_email="tests@example.com"),
        tmp_path,
        session=session,  # type: ignore[arg-type]
        limiter=RateLimiter(1000),
        sleep=sleeps.append,
        max_retries=3,
    )
    return client, session, sleeps


def test_fetch_caches_and_never_refetches(tmp_path: Path) -> None:
    client, session, _ = make_client(tmp_path, [FakeResponse(200, b'{"a": 1}')])
    url = submissions_url(1)
    assert client.get_json(url) == {"a": 1}
    assert client.get_json(url) == {"a": 1}
    assert len(session.calls) == 1
    assert client.stats == {"cache_hits": 1, "network_requests": 1}


def test_retries_on_429_with_backoff(tmp_path: Path) -> None:
    client, session, sleeps = make_client(
        tmp_path,
        [
            FakeResponse(429),
            requests.ConnectionError("x"),
            FakeResponse(503),
            FakeResponse(200, b"ok"),
        ],
    )
    assert client.get_bytes("https://www.sec.gov/x.htm") == b"ok"
    assert len(session.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_retry_after_header_is_honored(tmp_path: Path) -> None:
    client, _, sleeps = make_client(
        tmp_path, [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, b"ok")]
    )
    client.get_bytes("https://www.sec.gov/y.htm")
    assert sleeps == [7.0]


def test_non_retryable_status_raises(tmp_path: Path) -> None:
    client, session, _ = make_client(tmp_path, [FakeResponse(404)])
    with pytest.raises(EdgarError) as ei:
        client.get_bytes("https://www.sec.gov/missing.htm")
    assert ei.value.status == 404
    assert len(session.calls) == 1


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    client, session, _ = make_client(tmp_path, [FakeResponse(500)] * 4)
    with pytest.raises(EdgarError):
        client.get_bytes("https://www.sec.gov/z.htm")
    assert len(session.calls) == 4


def test_offline_cache_miss(tmp_path: Path) -> None:
    client = EdgarClient(SecConfig(), tmp_path, offline=True)
    with pytest.raises(CacheMiss):
        client.get_bytes("https://www.sec.gov/nothing.htm")


def test_real_session_sets_user_agent_and_refuses_without_email(tmp_path: Path) -> None:
    ok = EdgarClient(SecConfig(contact_email="me@example.com"), tmp_path)
    assert ok.session.headers["User-Agent"] == "edgar-extract research me@example.com"
    bad = EdgarClient(SecConfig(contact_email=""), tmp_path)
    with pytest.raises(ConfigError):
        bad.get_bytes("https://www.sec.gov/x.htm")
