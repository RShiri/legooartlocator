"""Brickognize client — free LEGO part identification from an image.

Brickognize (https://brickognize.com, API https://api.brickognize.com) is a
free, purpose-built model that identifies LEGO parts/minifigs/sets from a
photo or render. We POST a callout crop and get ranked candidate parts back;
the ensemble later keeps only candidates that are in the set inventory.

No API key required. The HTTP transport is injectable so the parser can be
unit-tested offline.

Caching and politeness
-----------------------
A manual scan can call ``predict`` hundreds of times (once per detected
callout crop), and re-running a scan re-identifies every crop from scratch.
To make that cheap and considerate of Brickognize's free service:

* Results are cached (via an injected ``JSONCache``-like object) keyed by
  the SHA-256 content hash of the crop's image bytes, namespaced under
  ``cache_namespace``. Identical crop bytes always hit the cache; failures
  are never cached, so a transient outage doesn't poison future scans.
* Calls are throttled to at most one every ``min_interval`` seconds.
* Transport failures are retried up to ``max_retries`` times with
  exponential backoff (``backoff * 2 ** attempt`` between attempts) before
  giving up and raising ``BrickognizeError``.
* The default HTTP transport reuses a single ``httpx.Client`` for the life
  of the ``BrickognizeClient`` instead of opening a new connection per call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .cache import content_hash

BASE_URL = "https://api.brickognize.com"


@dataclass
class BrickognizeCandidate:
    part_num: str
    name: str
    score: float
    img_url: Optional[str] = None
    category: Optional[str] = None


class BrickognizeError(RuntimeError):
    pass


def parse_predictions(payload: dict) -> List[BrickognizeCandidate]:
    """Parse a Brickognize /predict/ response into ranked candidates.

    The API returns ``{"items": [{"id", "name", "score", "img_url", ...}]}``.
    Parsing is lenient about missing fields. Results are sorted by score desc.
    """
    items = payload.get("items") or []
    candidates: List[BrickognizeCandidate] = []
    for it in items:
        part_num = it.get("id") or it.get("part_num")
        if not part_num:
            continue
        try:
            score = float(it.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        candidates.append(
            BrickognizeCandidate(
                part_num=str(part_num),
                name=it.get("name") or "",
                score=score,
                img_url=it.get("img_url"),
                category=it.get("category"),
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


class BrickognizeClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        transport: Optional[Callable] = None,
        cache: Optional[object] = None,
        cache_namespace: str = "brickognize-v1",
        max_retries: int = 3,
        backoff: float = 0.5,
        min_interval: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        """``transport(image_bytes, filename) -> dict`` may be injected for tests.

        By default an httpx-backed transport POSTs the image as multipart form
        data to ``{base_url}/predict/``, reusing one ``httpx.Client`` for the
        lifetime of this instance.

        ``cache`` is duck-typed to ``JSONCache`` (``.get(namespace, key)`` /
        ``.set(namespace, key, value)``). When provided, results are cached
        under ``cache_namespace`` keyed by the SHA-256 hash of the crop's
        image bytes, so identical crops never hit the network twice. Only
        successful responses are cached.

        ``max_retries``/``backoff`` control retry-with-exponential-backoff on
        transport failures; after the final failure a ``BrickognizeError`` is
        raised. ``min_interval`` throttles consecutive calls to at most one
        per that many seconds. ``sleep``/``clock`` are injectable for tests.
        """
        self.base_url = base_url.rstrip("/")
        self._transport = transport or self._http_transport
        self.cache = cache
        self.cache_namespace = cache_namespace
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last_call: Optional[float] = None
        self._http = None

    def predict(self, image_bytes: bytes, filename: str = "crop.png") -> List[BrickognizeCandidate]:
        key: Optional[str] = None
        if self.cache is not None:
            key = content_hash(image_bytes)
            cached = self.cache.get(self.cache_namespace, key)
            if cached is not None:
                return parse_predictions(cached)

        self._throttle()

        payload = None
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                payload = self._transport(image_bytes, filename)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
                last_exc = exc
                if attempt < self.max_retries - 1:
                    self._sleep(self.backoff * (2**attempt))

        if last_exc is not None:
            raise BrickognizeError(
                f"Brickognize failed after {self.max_retries} attempts: {last_exc}"
            ) from last_exc

        self._last_call = self._clock()
        if self.cache is not None:
            self.cache.set(self.cache_namespace, key, payload)
        return parse_predictions(payload)

    def _throttle(self) -> None:
        if self.min_interval <= 0 or self._last_call is None:
            return
        remaining = self.min_interval - (self._clock() - self._last_call)
        if remaining > 0:
            self._sleep(remaining)

    def _http_transport(self, image_bytes: bytes, filename: str) -> dict:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BrickognizeError("The 'httpx' package is required for Brickognize.") from exc
        if self._http is None:
            self._http = httpx.Client(timeout=30.0)
        url = f"{self.base_url}/predict/"
        files = {"query_image": (filename, image_bytes, "image/png")}
        resp = self._http.post(url, files=files)
        resp.raise_for_status()
        return resp.json()
