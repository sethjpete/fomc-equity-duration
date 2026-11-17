"""Shared HTTP download with retries, backoff and separate connect/read timeouts."""

from __future__ import annotations

import time

import requests

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
}
RETRY_STATUS = {429, 500, 502, 503, 504}


def get(url: str, *, params: dict | None = None, connect_timeout: float = 15,
        read_timeout: float = 120, retries: int = 3, backoff: float = 5,
        session: requests.Session | None = None, sleep=time.sleep) -> requests.Response:
    """GET with up to `retries` extra attempts on timeouts, dropped connections and 429/5xx.

    Waits backoff, 2*backoff, 4*backoff ... between attempts. Other HTTP errors raise at once.
    """
    s = session or requests
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = s.get(url, params=params, headers=HEADERS,
                      timeout=(connect_timeout, read_timeout))
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if isinstance(e, requests.HTTPError) and status not in RETRY_STATUS:
                raise
            last = e
            if attempt < retries:
                wait = backoff * 2**attempt
                print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__} on {url}; "
                      f"waiting {wait:.0f}s")
                sleep(wait)
    raise last  # type: ignore[misc]
