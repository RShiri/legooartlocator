"""Rebrickable API client: fetch a set's canonical parts inventory.

Docs: https://rebrickable.com/api/  (free key required)

We only need the parts endpoint:
    GET /api/v3/lego/sets/{set_num}/parts/?page_size=1000
which paginates over inventory lines with part, color, quantity, element_id
and a part image URL. We normalise those into InventoryPart.

Rebrickable expects the set number in "NNNN-1" form (with an inventory
suffix); if the caller passes a bare number we default the "-1" variant.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .models import InventoryPart

BASE_URL = "https://rebrickable.com/api/v3"


class RebrickableError(RuntimeError):
    pass


def normalize_set_num(set_num: str) -> str:
    set_num = str(set_num).strip()
    return set_num if "-" in set_num else f"{set_num}-1"


class RebrickableClient:
    def __init__(self, api_key: Optional[str] = None, client=None, base_url: str = BASE_URL):
        self.api_key = api_key or os.environ.get("REBRICKABLE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._client = client  # injectable httpx.Client for tests

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RebrickableError("The 'httpx' package is required for Rebrickable access.") from exc
        self._client = httpx.Client(timeout=30.0)
        return self._client

    def _headers(self) -> dict:
        if not self.api_key:
            raise RebrickableError(
                "REBRICKABLE_API_KEY is not set. Provide a key or run with --no-rebrickable."
            )
        return {"Authorization": f"key {self.api_key}", "Accept": "application/json"}

    def _get(self, client, url: str, headers: dict, set_num: str):
        """GET ``url``, raising ``RebrickableError`` for every failure mode --
        a 404, any other non-2xx status, or a transport-level error (bad key,
        rate limit, timeout, connection failure) -- so callers (cli.py's
        ``_fetch_inventory``) can catch one exception type and fall back to
        vision-only, instead of a raw httpx exception crashing the scan."""
        try:
            resp = client.get(url, headers=headers)
        except Exception as exc:  # noqa: BLE001 - any transport failure, not just HTTPStatusError
            raise RebrickableError(f"Rebrickable request failed: {exc}") from exc
        if resp.status_code == 404:
            raise RebrickableError(f"Set {set_num} not found on Rebrickable.")
        try:
            resp.raise_for_status()
        except Exception as exc:
            raise RebrickableError(f"Rebrickable request failed: {exc}") from exc
        return resp

    def get_set_info(self, set_num: str) -> dict:
        client = self._get_client()
        url = f"{self.base_url}/lego/sets/{normalize_set_num(set_num)}/"
        resp = self._get(client, url, self._headers(), set_num)
        return resp.json()

    def get_set_parts(
        self, set_num: str, page_size: int = 1000, max_pages: int = 1000
    ) -> List[InventoryPart]:
        """Fetch the full inventory, following pagination.

        ``max_pages`` bounds the ``next``-link loop so a misbehaving API
        response (an unexpected pagination cycle) can't hang forever.
        """
        client = self._get_client()
        url: Optional[str] = (
            f"{self.base_url}/lego/sets/{normalize_set_num(set_num)}/parts/"
            f"?page_size={page_size}"
        )
        parts: List[InventoryPart] = []
        headers = self._headers()
        pages = 0
        while url:
            pages += 1
            if pages > max_pages:
                raise RebrickableError(
                    f"Rebrickable pagination exceeded {max_pages} pages for set {set_num} "
                    "(unexpected 'next' link loop)."
                )
            resp = self._get(client, url, headers, set_num)
            payload = resp.json()
            for row in payload.get("results", []):
                parts.append(_row_to_inventory_part(row))
            url = payload.get("next")
        return parts


def _row_to_inventory_part(row: dict) -> InventoryPart:
    part = row.get("part") or {}
    color = row.get("color") or {}
    try:
        quantity = int(row.get("quantity", 0))
    except (TypeError, ValueError, OverflowError):
        quantity = 0
    return InventoryPart(
        part_num=part.get("part_num", ""),
        name=part.get("name", ""),
        color_id=color.get("id"),
        color_name=color.get("name"),
        quantity=quantity,
        element_id=row.get("element_id"),
        image_url=part.get("part_img_url"),
    )
