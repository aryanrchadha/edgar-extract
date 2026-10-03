"""Rate-limited, disk-cached client for SEC EDGAR.

Endpoints (verified against https://www.sec.gov/search-filings/edgar-application-programming-interfaces):

- ``https://data.sec.gov/submissions/CIK##########.json`` (10-digit zero-padded CIK)
- ``https://data.sec.gov/submissions/<name>.json`` for older-filing pages listed in
  ``filings.files``
- ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``
- ``https://www.sec.gov/files/company_tickers.json`` (ticker -> CIK map)
- ``https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/<document>``

Every network request carries the configured User-Agent and passes through one process-wide
rate limiter. Responses are cached on disk under ``raw_dir/<host>/<path>`` so reruns never
re-download. Archive documents are immutable; JSON endpoints change as companies file, so
they can be refreshed explicitly with ``refresh=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from edgar_extract.config import SecConfig

log = logging.getLogger(__name__)

DATA_HOST = "https://data.sec.gov"
WWW_HOST = "https://www.sec.gov"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class EdgarError(RuntimeError):
    """A request failed after retries, or returned a non-retryable error status."""

    def __init__(self, url: str, status: int | None, message: str) -> None:
        super().__init__(f"{message} [{status}] {url}")
        self.url = url
        self.status = status


class CacheMiss(EdgarError):
    """Raised in offline mode when a URL is not in the disk cache."""

    def __init__(self, url: str) -> None:
        super().__init__(url, None, "not cached and client is offline")


def pad_cik(cik: int | str) -> str:
    return f"{int(cik):010d}"


def accession_nodash(accession: str) -> str:
    return accession.replace("-", "")


def submissions_url(cik: int | str) -> str:
    return f"{DATA_HOST}/submissions/CIK{pad_cik(cik)}.json"


def submissions_page_url(name: str) -> str:
    return f"{DATA_HOST}/submissions/{name}"


def companyfacts_url(cik: int | str) -> str:
    return f"{DATA_HOST}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"


def company_tickers_url() -> str:
    return f"{WWW_HOST}/files/company_tickers.json"


def filing_document_url(cik: int | str, accession: str, document: str) -> str:
    return f"{WWW_HOST}/Archives/edgar/data/{int(cik)}/{accession_nodash(accession)}/{document}"


def filing_index_url(cik: int | str, accession: str) -> str:
    return f"{WWW_HOST}/Archives/edgar/data/{int(cik)}/{accession_nodash(accession)}/"


class RateLimiter:
    """Thread-safe limiter that spaces requests at least ``1/rps`` seconds apart.

    A minimum-interval limiter never bursts, so the instantaneous rate is also capped,
    which is stricter than a token bucket and simpler to verify.
    """

    def __init__(
        self,
        rps: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self.interval = 1.0 / rps
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next_allowed - now
            if wait > 0:
                self._sleep(wait)
                now = self._next_allowed
            self._next_allowed = now + self.interval


class DiskCache:
    """Maps URLs to files under ``root/<host>/<path>``. Writes are atomic."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, url: str) -> Path:
        parts = urlsplit(url)
        path = parts.path.lstrip("/")
        if not path or path.endswith("/"):
            path += "index.html"
        if parts.query:
            digest = hashlib.sha1(parts.query.encode()).hexdigest()[:12]
            path = f"{path}.q{digest}"
        resolved = (self.root / parts.netloc / path).resolve()
        if self.root.resolve() not in resolved.parents:
            raise ValueError(f"refusing to cache outside cache root: {url}")
        return resolved

    def get(self, url: str) -> bytes | None:
        p = self.path_for(url)
        return p.read_bytes() if p.exists() else None

    def put(self, url: str, content: bytes) -> Path:
        p = self.path_for(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return p


_SHARED_LIMITERS: dict[float, RateLimiter] = {}
_SHARED_LOCK = threading.Lock()


def shared_limiter(rps: float) -> RateLimiter:
    """One limiter per process (per rate) so multiple clients cannot exceed the cap together."""
    with _SHARED_LOCK:
        if rps not in _SHARED_LIMITERS:
            _SHARED_LIMITERS[rps] = RateLimiter(rps)
        return _SHARED_LIMITERS[rps]


class EdgarClient:
    def __init__(
        self,
        sec: SecConfig,
        raw_dir: Path,
        *,
        offline: bool = False,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.sec = sec
        self.cache = DiskCache(raw_dir)
        self.offline = offline
        self._session = session
        self.limiter = limiter or shared_limiter(sec.max_requests_per_second)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self.stats = {"cache_hits": 0, "network_requests": 0}

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": self.sec.user_agent(),
                    "Accept-Encoding": "gzip, deflate",
                }
            )
            self._session = s
        return self._session

    def get_bytes(self, url: str, *, refresh: bool = False) -> bytes:
        if not refresh:
            cached = self.cache.get(url)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached
        if self.offline:
            raise CacheMiss(url)
        content = self._fetch(url)
        self.cache.put(url, content)
        return content

    def get_json(self, url: str, *, refresh: bool = False) -> Any:
        return json.loads(self.get_bytes(url, refresh=refresh))

    def _fetch(self, url: str) -> bytes:
        # Building the session validates the User-Agent before any traffic is sent.
        session = self.session
        if "User-Agent" not in session.headers or "@" not in str(session.headers["User-Agent"]):
            raise EdgarError(url, None, "refusing to send request without contact User-Agent")
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            self.stats["network_requests"] += 1
            try:
                resp = session.get(url, timeout=30)
            except requests.RequestException as exc:
                log.warning("request error on %s (attempt %d): %s", url, attempt + 1, exc)
                last_status = None
            else:
                last_status = resp.status_code
                if resp.status_code == 200:
                    log.debug("GET %s 200 (%d bytes)", url, len(resp.content))
                    return resp.content
                if resp.status_code not in RETRY_STATUS:
                    raise EdgarError(url, resp.status_code, "non-retryable HTTP status")
                log.warning("GET %s -> %d (attempt %d)", url, resp.status_code, attempt + 1)
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    self._sleep(float(retry_after))
                    continue
            if attempt < self.max_retries:
                self._sleep(self.backoff_base * (2**attempt))
        raise EdgarError(url, last_status, f"failed after {self.max_retries + 1} attempts")

    # Convenience wrappers -------------------------------------------------

    def company_tickers(self, *, refresh: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = self.get_json(company_tickers_url(), refresh=refresh)
        return result

    def submissions(self, cik: int | str, *, refresh: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = self.get_json(submissions_url(cik), refresh=refresh)
        return result

    def submissions_page(self, name: str) -> dict[str, Any]:
        # Older-filing pages are closed ranges and do not change; never refresh.
        result: dict[str, Any] = self.get_json(submissions_page_url(name))
        return result

    def companyfacts(self, cik: int | str, *, refresh: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = self.get_json(companyfacts_url(cik), refresh=refresh)
        return result

    def filing_document(self, cik: int | str, accession: str, document: str) -> bytes:
        return self.get_bytes(filing_document_url(cik, accession, document))
