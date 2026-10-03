"""Rate-limited, disk-cached client for SEC EDGAR.

Endpoints (checked against sec.gov documentation on 2026-10-03):

* ``https://www.sec.gov/files/company_tickers.json``: ticker -> CIK map
* ``https://data.sec.gov/submissions/CIK##########.json``: filing history; older history
  lives in the files listed under ``filings.files`` (``https://data.sec.gov/submissions/<name>``)
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``: XBRL company facts
* ``https://www.sec.gov/Archives/edgar/data/<cik>/<accession without dashes>/<document>``

Fair access: every request carries ``User-Agent: <app name> <contact email>``, and all
clients in the process share one limiter that keeps traffic under 8 requests/second
(SEC's ceiling is 10). Every response body is cached under ``<cache_dir>/<host>/<path>.gz``
and never fetched again unless ``refresh=True`` is passed for the mutable JSON endpoints.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from edgar_extract.config import MAX_ALLOWED_RPS, Settings

log = logging.getLogger(__name__)

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"
COMPANY_TICKERS_URL = f"{SEC_WWW}/files/company_tickers.json"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 60.0


class EdgarError(RuntimeError):
    """Base class for client errors."""


class OfflineCacheMissError(EdgarError):
    """The client is offline and the URL is not in the cache."""


class EdgarHTTPError(EdgarError):
    def __init__(self, url: str, status: int, message: str = "") -> None:
        super().__init__(f"HTTP {status} for {url}{': ' + message if message else ''}")
        self.url = url
        self.status = status


class EdgarNotFoundError(EdgarHTTPError):
    """HTTP 404. Callers usually log a skip and move on."""


class EdgarForbiddenError(EdgarHTTPError):
    """HTTP 403. SEC returns this for undeclared or throttled clients; never retried."""


def format_cik(cik: int | str) -> str:
    """Zero-pad a CIK to the 10 digits used by data.sec.gov."""
    digits = str(cik).strip().lstrip("CIK").lstrip("0") or "0"
    if not digits.isdigit():
        raise ValueError(f"invalid CIK {cik!r}")
    return digits.zfill(10)


def accession_nodash(accession: str) -> str:
    cleaned = accession.replace("-", "").strip()
    if len(cleaned) != 18 or not cleaned.isdigit():
        raise ValueError(f"invalid accession number {accession!r}")
    return cleaned


def submissions_url(cik: int | str) -> str:
    return f"{SEC_DATA}/submissions/CIK{format_cik(cik)}.json"


def submissions_page_url(name: str) -> str:
    if "/" in name or not name.endswith(".json"):
        raise ValueError(f"unexpected submissions page name {name!r}")
    return f"{SEC_DATA}/submissions/{name}"


def companyfacts_url(cik: int | str) -> str:
    return f"{SEC_DATA}/api/xbrl/companyfacts/CIK{format_cik(cik)}.json"


def archive_url(cik: int | str, accession: str, document: str) -> str:
    """URL of one document inside a filing's archive directory."""
    if not document or "/" in document:
        raise ValueError(f"invalid document name {document!r}")
    return f"{_archive_dir(cik, accession)}/{document}"


def filing_index_url(cik: int | str, accession: str) -> str:
    return f"{_archive_dir(cik, accession)}/{accession}-index.htm"


def _archive_dir(cik: int | str, accession: str) -> str:
    return f"{SEC_WWW}/Archives/edgar/data/{int(format_cik(cik))}/{accession_nodash(accession)}"


class RateLimiter:
    """Thread-safe limiter that spaces requests at least ``1/rate`` seconds apart.

    Even spacing is stricter than a token bucket: with ``rate <= 8`` no one-second window
    can contain more than 8 request starts.
    """

    def __init__(
        self,
        rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < rate <= MAX_ALLOWED_RPS:
            raise ValueError(f"rate must be in (0, {MAX_ALLOWED_RPS}], got {rate}")
        self._interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = float("-inf")

    @property
    def rate(self) -> float:
        return 1.0 / self._interval

    def slow_down_to(self, rate: float) -> None:
        """Lower the rate (never raises it). Used when clients ask for different rates."""
        with self._lock:
            self._interval = max(self._interval, 1.0 / rate)

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next_allowed)
            self._next_allowed = start + self._interval
        wait = start - now
        if wait > 0:
            self._sleep(wait)


_GLOBAL_LIMITER: RateLimiter | None = None
_GLOBAL_LOCK = threading.Lock()


def global_rate_limiter(rate: float) -> RateLimiter:
    """The single process-wide limiter. Asking for a lower rate slows everyone down."""
    global _GLOBAL_LIMITER
    with _GLOBAL_LOCK:
        if _GLOBAL_LIMITER is None:
            _GLOBAL_LIMITER = RateLimiter(rate)
        else:
            _GLOBAL_LIMITER.slow_down_to(rate)
        return _GLOBAL_LIMITER


class DiskCache:
    """Gzip-compressed response bodies stored at ``<root>/<host>/<url path>.gz``.

    A sidecar ``.meta.json`` records the URL and fetch time so a rebuild can say which
    snapshot of the mutable JSON endpoints it used.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, url: str) -> Path:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError(f"not an absolute http(s) URL: {url!r}")
        rel = parts.path.lstrip("/") or "index"
        if ".." in Path(rel).parts:
            raise ValueError(f"refusing path traversal in URL {url!r}")
        if parts.query:
            digest = hashlib.sha256(parts.query.encode()).hexdigest()[:16]
            rel = f"{rel}__q{digest}"
        return self.root / parts.netloc / f"{rel}.gz"

    def get(self, url: str) -> bytes | None:
        path = self.path_for(url)
        if not path.is_file():
            return None
        return gzip.decompress(path.read_bytes())

    def put(self, url: str, body: bytes) -> Path:
        path = self.path_for(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, gzip.compress(body, compresslevel=6))
        meta = {
            "url": url,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "bytes": len(body),
        }
        self._atomic_write(path.with_suffix(".meta.json"), json.dumps(meta).encode())
        return path

    def fetched_at(self, url: str) -> str | None:
        meta = self.path_for(url).with_suffix(".meta.json")
        if not meta.is_file():
            return None
        return str(json.loads(meta.read_text())["fetched_at"])

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


@dataclass
class ClientStats:
    cache_hits: int = 0
    network_fetches: int = 0
    retries: int = 0


def _is_mutable(url: str) -> bool:
    """JSON endpoints that SEC updates in place. Archive documents never change."""
    return url.startswith(SEC_DATA) or url == COMPANY_TICKERS_URL


class EdgarClient:
    """SEC client: declared User-Agent, shared rate limit, disk cache, bounded retries."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        *,
        limiter: RateLimiter | None = None,
        rate: float = 7.0,
        offline: bool = False,
        refresh: bool = False,
        max_retries: int = 5,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC requires a contact email in the User-Agent header")
        self.user_agent = user_agent
        self.cache = DiskCache(cache_dir)
        self.limiter = limiter if limiter is not None else global_rate_limiter(rate)
        self.offline = offline
        self.refresh = refresh
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._sleep = sleep
        self._refreshed: set[str] = set()
        self.stats = ClientStats()

    @classmethod
    def from_settings(
        cls, settings: Settings, *, offline: bool = False, refresh: bool = False
    ) -> EdgarClient:
        return cls(
            settings.sec.user_agent,
            settings.paths.raw_dir,
            rate=settings.sec.max_requests_per_second,
            offline=offline,
            refresh=refresh,
            max_retries=settings.sec.max_retries,
            timeout=settings.sec.timeout_seconds,
        )

    # -- generic access ---------------------------------------------------------------

    def get_bytes(self, url: str) -> bytes:
        force = self.refresh and _is_mutable(url) and url not in self._refreshed
        if not force:
            cached = self.cache.get(url)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached
        if self.offline:
            raise OfflineCacheMissError(f"offline mode and not cached: {url}")
        body = self._fetch(url)
        self.cache.put(url, body)
        self._refreshed.add(url)
        return body

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_bytes(url))

    def is_cached(self, url: str) -> bool:
        return self.cache.path_for(url).is_file()

    # -- SEC endpoints ----------------------------------------------------------------

    def company_tickers(self) -> dict[str, Any]:
        data = self.get_json(COMPANY_TICKERS_URL)
        if not isinstance(data, dict):
            raise EdgarError("company_tickers.json is not a JSON object")
        return data

    def submissions(self, cik: int | str) -> dict[str, Any]:
        data = self.get_json(submissions_url(cik))
        if not isinstance(data, dict) or "filings" not in data:
            raise EdgarError(f"unexpected submissions payload for CIK {cik}")
        return data

    def submissions_page(self, name: str) -> dict[str, Any]:
        data = self.get_json(submissions_page_url(name))
        if not isinstance(data, dict) or "accessionNumber" not in data:
            raise EdgarError(f"unexpected submissions page payload {name}")
        return data

    def company_facts(self, cik: int | str) -> dict[str, Any]:
        data = self.get_json(companyfacts_url(cik))
        if not isinstance(data, dict) or "facts" not in data:
            raise EdgarError(f"unexpected companyfacts payload for CIK {cik}")
        return data

    def filing_document(self, cik: int | str, accession: str, document: str) -> bytes:
        return self.get_bytes(archive_url(cik, accession, document))

    # -- network ----------------------------------------------------------------------

    def _fetch(self, url: str) -> bytes:
        attempt = 0
        while True:
            self.limiter.acquire()
            self.stats.network_fetches += 1
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise EdgarError(
                        f"network error for {url} after {attempt} retries: {exc}"
                    ) from exc
                delay = self._backoff(attempt, None)
                log.warning("network error for %s (%s); retrying in %.1fs", url, exc, delay)
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    raise EdgarNotFoundError(url, 404)
                if resp.status_code == 403:
                    raise EdgarForbiddenError(
                        url, 403, "SEC refused the request; check the User-Agent and request rate"
                    )
                if resp.status_code not in RETRY_STATUSES or attempt >= self.max_retries:
                    raise EdgarHTTPError(url, resp.status_code)
                delay = self._backoff(attempt, resp.headers.get("Retry-After"))
                log.warning("HTTP %s for %s; retrying in %.1fs", resp.status_code, url, delay)
            self.stats.retries += 1
            attempt += 1
            self._sleep(delay)

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        delay: float = min(MAX_BACKOFF_SECONDS, 2.0**attempt)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    delay = max(delay, (when - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError):
                    pass
        return min(delay, MAX_BACKOFF_SECONDS * 5)
