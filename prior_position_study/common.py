"""
Shared Utilities
================
HTTP client with on-disk caching, polite rate limiting and exponential
backoff, plus config loading helpers used by every stage.

Every API response is cached under data/cache/ keyed by a hash of the request
URL. Reruns are therefore free and the harvest is auditable after the fact:
the cache is the raw evidence behind the coded CSV.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"


def load_config(name: str) -> dict:
    """Load a YAML config from config/."""
    with open(BASE_DIR / "config" / name) as f:
        return yaml.safe_load(f)


class CachedSession:
    """
    Rate-limited, retrying, disk-cached GET client.

    Parameters
    ----------
    namespace : str
        Cache subdirectory, e.g. "openalex" or "orcid".
    requests_per_second : float
        Client-side throttle. OpenAlex asks for <= 10/s in the polite pool.
    backoff_seconds : list[int]
        Sleep schedule between retries. Retries cover transport errors and
        429/5xx responses only; 4xx other than 429 are returned to the caller.
    """

    def __init__(
        self,
        namespace: str,
        requests_per_second: float = 5.0,
        max_retries: int = 5,
        backoff_seconds: Optional[list] = None,
        headers: Optional[dict] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.namespace = namespace
        self.min_interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self.max_retries = max_retries
        self.backoff = backoff_seconds or [2, 4, 8, 16, 32]
        self.cache_dir = (cache_dir or CACHE_DIR) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(headers or {})
        self._last_request = 0.0
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    def get_json(self, url: str, use_cache: bool = True) -> Optional[Any]:
        """
        GET a JSON document, returning None if it cannot be retrieved.

        A cached `null` is a negative result (e.g. a 404 for an ORCID with no
        public record) and is honoured, so dead records are not re-requested
        on every run.
        """
        path = self._cache_path(url)
        if use_cache and path.exists():
            self.stats["hits"] += 1
            with open(path) as f:
                return json.load(f).get("body")

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=60)
            except requests.RequestException as exc:
                if attempt == self.max_retries - 1:
                    self.stats["errors"] += 1
                    print(f"  [error] {url}: {exc}")
                    return None
                time.sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
                continue

            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError:
                    self.stats["errors"] += 1
                    return None
                self.stats["misses"] += 1
                with open(path, "w") as f:
                    json.dump({"url": url, "body": body}, f)
                return body

            # 404 is a real answer: cache it so we do not ask again.
            if resp.status_code == 404:
                self.stats["misses"] += 1
                with open(path, "w") as f:
                    json.dump({"url": url, "body": None}, f)
                return None

            if resp.status_code in (429, 500, 502, 503, 504):
                wait = self.backoff[min(attempt, len(self.backoff) - 1)]
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                print(f"  [retry {attempt + 1}/{self.max_retries}] {resp.status_code} -> sleeping {wait}s")
                time.sleep(wait)
                continue

            # 403 from the egress proxy means an organization policy denial,
            # not a transient fault. Fail loudly instead of burning retries.
            if resp.status_code in (403, 407):
                self.stats["errors"] += 1
                raise EgressBlocked(
                    f"{resp.status_code} for {url}. If this came from the agent proxy, "
                    f"the host is blocked by egress policy and retrying will not help."
                )

            self.stats["errors"] += 1
            print(f"  [error] {resp.status_code} for {url}")
            return None

        self.stats["errors"] += 1
        return None


class EgressBlocked(RuntimeError):
    """Raised when the network path to an API is administratively blocked."""


def paginate_openalex(session: CachedSession, url: str, max_results: int = 10000):
    """
    Yield OpenAlex results using cursor pagination.

    The caller passes a fully-formed URL without a cursor parameter.
    """
    cursor = "*"
    fetched = 0
    while cursor and fetched < max_results:
        sep = "&" if "?" in url else "?"
        page = session.get_json(f"{url}{sep}cursor={cursor}")
        if not page:
            return
        results = page.get("results", [])
        if not results:
            return
        for item in results:
            yield item
            fetched += 1
            if fetched >= max_results:
                return
        cursor = page.get("meta", {}).get("next_cursor")


def ensure_dirs():
    for d in (DATA_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list:
    if not Path(path).exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
