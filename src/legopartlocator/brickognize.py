"""Brickognize client — free LEGO part identification from an image.

Brickognize (https://brickognize.com, API https://api.brickognize.com) is a
free, purpose-built model that identifies LEGO parts/minifigs/sets from a
photo or render. We POST a callout crop and get ranked candidate parts back;
the ensemble later keeps only candidates that are in the set inventory.

No API key required. The HTTP transport is injectable so the parser can be
unit-tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

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
    def __init__(self, base_url: str = BASE_URL, transport: Optional[Callable] = None):
        """``transport(image_bytes, filename) -> dict`` may be injected for tests.

        By default an httpx-backed transport POSTs the image as multipart form
        data to ``{base_url}/predict/``.
        """
        self.base_url = base_url.rstrip("/")
        self._transport = transport or self._http_transport

    def predict(self, image_bytes: bytes, filename: str = "crop.png") -> List[BrickognizeCandidate]:
        payload = self._transport(image_bytes, filename)
        return parse_predictions(payload)

    def _http_transport(self, image_bytes: bytes, filename: str) -> dict:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BrickognizeError("The 'httpx' package is required for Brickognize.") from exc
        url = f"{self.base_url}/predict/"
        files = {"query_image": (filename, image_bytes, "image/png")}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, files=files)
            resp.raise_for_status()
            return resp.json()
