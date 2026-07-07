"""Claude vision pass: read a rendered page image into a structured PageExtract.

We force structured output with a tool schema so there is no fragile text
parsing: the model must call ``record_page`` with fields matching PageExtract.

Every call is cached on the SHA-256 of the exact PNG bytes (namespaced by a
prompt version), so re-scanning an unchanged PDF costs nothing.
"""

from __future__ import annotations

import base64
import os
from typing import List, Optional

from .cache import JSONCache, content_hash
from .models import PageCallout, PageExtract
from .pdf_render import RenderedPage

# Bump when the prompt/schema changes so stale cache entries are ignored.
PROMPT_VERSION = "v1"

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

_SYSTEM = (
    "You read a single page image from an official LEGO building-instruction "
    "booklet and report structured facts about it. Be precise and conservative: "
    "only report what is actually visible. Do not invent part ids."
)

_INSTRUCTIONS = (
    "Examine this LEGO instruction page and call record_page.\n"
    "- bag_marker: if the page prominently starts a NEW numbered bag section "
    "(a large standalone numeral, often boxed, sometimes with a bag icon), set "
    "it to that number. Otherwise null. Step numbers are NOT bag markers.\n"
    "- is_parts_list: true only if the page is a full parts-inventory grid (BOM), "
    "not a build step.\n"
    "- printed_page_number: the page number printed on the page, if visible.\n"
    "- callouts: one entry per parts-callout box (the small boxes showing 'Nx' "
    "and a part render). Give quantity, colour as seen, a short shape "
    "description, and any printed design/element id. Do not include the main "
    "assembly art, only the callout boxes."
)

# JSON Schema for the forced tool call. Mirrors PageExtract (minus page_index,
# which the caller owns).
_TOOL = {
    "name": "record_page",
    "description": "Record structured facts extracted from one LEGO instruction page.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bag_marker": {
                "type": ["integer", "null"],
                "description": "Bag number if this page starts a new numbered bag, else null.",
            },
            "is_parts_list": {
                "type": "boolean",
                "description": "True if this is a full BOM/parts-list grid page.",
            },
            "printed_page_number": {
                "type": ["integer", "null"],
                "description": "Page number printed on the page, if any.",
            },
            "callouts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quantity": {"type": "integer", "minimum": 1},
                        "color": {"type": ["string", "null"]},
                        "shape_desc": {"type": ["string", "null"]},
                        "printed_part_id": {"type": ["string", "null"]},
                    },
                    "required": ["quantity"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": ["string", "null"]},
        },
        "required": ["bag_marker", "is_parts_list", "callouts"],
        "additionalProperties": False,
    },
}


class VisionError(RuntimeError):
    pass


def _extract_from_dict(page_index: int, data: dict) -> PageExtract:
    callouts = [
        PageCallout(
            quantity=int(c.get("quantity", 1)),
            color=c.get("color"),
            shape_desc=c.get("shape_desc"),
            printed_part_id=c.get("printed_part_id"),
        )
        for c in (data.get("callouts") or [])
    ]
    return PageExtract(
        page_index=page_index,
        printed_page_number=data.get("printed_page_number"),
        bag_marker=data.get("bag_marker"),
        is_parts_list=bool(data.get("is_parts_list", False)),
        callouts=callouts,
        notes=data.get("notes"),
    )


class VisionExtractor:
    """Wraps the Anthropic client with caching for per-page extraction."""

    def __init__(
        self,
        cache: JSONCache,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client=None,
    ):
        self.cache = cache
        self.model = model
        self._client = client
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise VisionError(
                "ANTHROPIC_API_KEY is not set. Provide a key or pass a client. "
                "(Vision is required unless you supply pre-extracted page JSON.)"
            )
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise VisionError("The 'anthropic' package is required for the vision pass.") from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _cache_key(self, png_bytes: bytes) -> str:
        return content_hash(png_bytes)

    def extract_page(self, page: RenderedPage, use_cache: bool = True) -> PageExtract:
        namespace = f"vision-{PROMPT_VERSION}-{self.model}"
        key = self._cache_key(page.png_bytes)
        if use_cache:
            cached = self.cache.get(namespace, key)
            if cached is not None:
                return _extract_from_dict(page.page_index, cached)

        data = self._call_model(page.png_bytes)
        self.cache.set(namespace, key, data)
        return _extract_from_dict(page.page_index, data)

    def _call_model(self, png_bytes: bytes) -> dict:
        client = self._get_client()
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "record_page"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _INSTRUCTIONS},
                    ],
                }
            ],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_page":
                return dict(block.input)
        raise VisionError("Model did not return the expected record_page tool call.")


def extract_pages(
    pages: List[RenderedPage],
    extractor: VisionExtractor,
    use_cache: bool = True,
    progress=None,
) -> List[PageExtract]:
    """Extract a list of rendered pages, in order. ``progress`` is an optional callback(idx, total)."""
    results: List[PageExtract] = []
    total = len(pages)
    for i, page in enumerate(pages):
        results.append(extractor.extract_page(page, use_cache=use_cache))
        if progress:
            progress(i + 1, total)
    return results
