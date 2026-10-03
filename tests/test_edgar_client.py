from __future__ import annotations

import gzip
import math
import threading
import time
from collections.abc import Callable

import pytest
import requests

from edgar_extract import edgar_client as ec
from edgar_extract.edgar_client import (
    COMPANY_TICKERS_URL,
    DiskCache,
    EdgarClient,
    EdgarForbiddenError,
    EdgarHTTPError,
    EdgarNotFoundError,
    OfflineCacheMissError,
    RateLimiter,
)

from .conftest import TEST_UA, FixtureSession, make_response

AAPL_SUB = "https://data.sec.gov/submissions/CIK0000320193.json"
DOC = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


# -- URL helpers -----------------------------------------------------------------------


def test_url_formats() -> None:
    assert ec.format_cik(320193) == "0000320193"
    assert ec.format_cik("CIK0000320193") == "0000320193"
    assert ec.submissions_url(320193) == AAPL_SUB
    assert ec.companyfacts_url("320193") == (
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    )
    assert ec.archive_url(320193, "0000320193-25-000079", "aapl-20250927.htm") == DOC
    assert ec.submissions_page_url("CIK0000072971-submissions-001.json").endswith(
        "/submissions/CIK0000072971-submissions-001.json"
    )


@pytest.mark.parametrize("bad", ["123", "0000320193-25-00007X", ""])
def test_bad_accession_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        ec.accession_nodash(bad)


def test_archive_url_rejects_path_in_document() -> None:
    with pytest.raises(ValueError):
        ec.archive_url(1, "0000320193-25-000079", "../x.htm")


# -- headers, cache, offline -----------------------------------------------------------


def test_user_agent_requires_email(tmp_path) -> None:
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient("no email here", tmp_path)


def test_sends_declared_user_agent(client: EdgarClient, session: FixtureSession) -> None:
    client.submissions(320193)
    ((url, headers),) = session.calls
    assert url == AAPL_SUB
    assert headers["User-Agent"] == TEST_UA
    assert "gzip" in headers["Accept-Encoding"]


def test_second_request_served_from_disk_cache(
    client: EdgarClient, session: FixtureSession
) -> None:
    first = client.submissions(320193)
    second = client.submissions(320193)
    assert first == second
    assert len(session.calls) == 1
    assert client.stats.cache_hits == 1 and client.stats.network_fetches == 1
    cached = client.cache.path_for(AAPL_SUB)
    assert cached.name.endswith(".json.gz") and "data.sec.gov" in cached.parts
    assert gzip.decompress(cached.read_bytes())  # stored compressed
    assert client.cache.fetched_at(AAPL_SUB) is not None


def test_cache_survives_new_client(
    make_client: Callable[..., EdgarClient], session: FixtureSession
) -> None:
    make_client().submissions(320193)
    offline = make_client(offline=True)
    assert offline.submissions(320193)["name"] == "Apple Inc."
    assert len(session.calls) == 1


def test_offline_cache_miss_raises(
    make_client: Callable[..., EdgarClient], session: FixtureSession
) -> None:
    with pytest.raises(OfflineCacheMissError):
        make_client(offline=True).submissions(320193)
    assert session.calls == []


def test_refresh_refetches_json_once_but_never_archives(
    make_client: Callable[..., EdgarClient], session: FixtureSession
) -> None:
    session.script[DOC] = [make_response(DOC, 200, b"<html>doc</html>")]
    first = make_client()
    first.submissions(320193)
    first.get_bytes(DOC)
    refreshing = make_client(refresh=True)
    refreshing.submissions(320193)
    refreshing.submissions(320193)
    refreshing.get_bytes(DOC)
    assert session.urls().count(AAPL_SUB) == 2
    assert session.urls().count(DOC) == 1


def test_disk_cache_paths(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    assert (
        cache.path_for(COMPANY_TICKERS_URL)
        == tmp_path / "www.sec.gov/files/company_tickers.json.gz"
    )
    with_query = cache.path_for("https://www.sec.gov/cgi-bin/browse?a=1")
    assert "__q" in with_query.name
    with pytest.raises(ValueError):
        cache.path_for("https://www.sec.gov/a/../../etc/passwd")
    with pytest.raises(ValueError):
        cache.path_for("relative/path")


# -- error handling and retries ---------------------------------------------------------


def test_404_raises_not_found(client: EdgarClient) -> None:
    with pytest.raises(EdgarNotFoundError):
        client.submissions(999999999)


def test_403_is_not_retried(
    client: EdgarClient, session: FixtureSession, sleeps: list[float]
) -> None:
    session.script[AAPL_SUB] = [make_response(AAPL_SUB, 403)]
    with pytest.raises(EdgarForbiddenError):
        client.submissions(320193)
    assert len(session.calls) == 1 and sleeps == []


def test_429_retried_and_honors_retry_after(
    client: EdgarClient, session: FixtureSession, sleeps: list[float]
) -> None:
    session.script[AAPL_SUB] = [make_response(AAPL_SUB, 429, headers={"Retry-After": "7"})]
    assert client.submissions(320193)["cik"] == "0000320193"
    assert len(session.calls) == 2
    assert sleeps == [7.0]
    assert client.stats.retries == 1


def test_5xx_retries_exhaust(
    make_client: Callable[..., EdgarClient], session: FixtureSession, sleeps: list[float]
) -> None:
    client = make_client(max_retries=2)
    session.script[AAPL_SUB] = [make_response(AAPL_SUB, 503) for _ in range(5)]
    with pytest.raises(EdgarHTTPError) as info:
        client.submissions(320193)
    assert info.value.status == 503
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_connection_error_retried(
    client: EdgarClient, session: FixtureSession, sleeps: list[float]
) -> None:
    session.script[AAPL_SUB] = [requests.ConnectionError("reset")]
    assert client.submissions(320193)["name"] == "Apple Inc."
    assert len(sleeps) == 1


def test_failed_fetch_is_not_cached(client: EdgarClient, session: FixtureSession) -> None:
    session.script[AAPL_SUB] = [make_response(AAPL_SUB, 403)]
    with pytest.raises(EdgarForbiddenError):
        client.submissions(320193)
    assert not client.is_cached(AAPL_SUB)


# -- rate limiting -----------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def max_in_any_window(starts: list[float], width: float = 1.0) -> int:
    starts = sorted(starts)
    best, j = 0, 0
    for i, t in enumerate(starts):
        while starts[j] <= t - width:
            j += 1
        best = max(best, i - j + 1)
    return best


@pytest.mark.parametrize("rate", [8.0, 7.0, 2.5])
def test_rate_limiter_never_exceeds_rate_in_any_second(rate: float) -> None:
    clock = FakeClock()
    limiter = RateLimiter(rate, clock=clock, sleep=clock.sleep)
    starts = []
    for i in range(200):
        if i % 17 == 0:
            clock.now += 0.05  # irregular caller think-time
        limiter.acquire()
        starts.append(clock.now)
    # Evenly spaced starts put at most ceil(rate) in any one-second window; shave 1 us off
    # the window so float accumulation in the spacing cannot add a phantom extra start.
    assert max_in_any_window(starts, width=1.0 - 1e-6) <= math.ceil(rate)


def test_rate_limiter_is_thread_safe() -> None:
    limiter = RateLimiter(8.0)
    starts: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(4):
            limiter.acquire()
            with lock:
                starts.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(starts) == 16
    # Allow 10 ms of scheduler jitter in the measurement.
    assert max_in_any_window(starts, width=1.0 - 0.01) <= 8


def test_rate_limiter_rejects_rates_above_8() -> None:
    with pytest.raises(ValueError):
        RateLimiter(9.0)


def test_global_limiter_is_shared_and_only_slows_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, "_GLOBAL_LIMITER", None)
    a = ec.global_rate_limiter(7.0)
    b = ec.global_rate_limiter(5.0)
    c = ec.global_rate_limiter(8.0)
    assert a is b is c
    assert a.rate == pytest.approx(5.0)


def test_clients_share_global_limiter_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(ec, "_GLOBAL_LIMITER", None)
    one = EdgarClient(TEST_UA, tmp_path, session=FixtureSession())
    two = EdgarClient(TEST_UA, tmp_path, session=FixtureSession())
    assert one.limiter is two.limiter
