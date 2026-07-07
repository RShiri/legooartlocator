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

# The detailed pass reads every callout (token-heavy) -> capable model.
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
# The triage pass only classifies each page (cheap) -> fast model.
DEFAULT_TRIAGE_MODEL = os.environ.get("ANTHROPIC_TRIAGE_MODEL", "claude-haiku-4-5-20251001")

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


# --- Triage pass: cheap per-page classification (no part enumeration) ---------

_TRIAGE_SYSTEM = (
    "You quickly classify a single page image from an official LEGO "
    "building-instruction booklet. Answer only what is visible; do not "
    "enumerate individual parts."
)

_TRIAGE_INSTRUCTIONS = (
    "Classify this LEGO instruction page and call triage_page.\n"
    "- bag_marker: the number if the page prominently STARTS a new numbered bag "
    "section (a large standalone numeral, often boxed/with a bag icon), else "
    "null. Step numbers are NOT bag markers.\n"
    "- is_parts_list: true only if the page is a full parts-inventory grid (BOM).\n"
    "- has_callouts: true if the page contains any small parts-callout boxes "
    "(the 'Nx' + part-render boxes) that a detailed pass should read. A BOM page "
    "is NOT callouts. Pure cover/story/completed pages have none.\n"
    "- printed_page_number: the page number printed on the page, if visible."
)

_TRIAGE_TOOL = {
    "name": "triage_page",
    "description": "Quickly classify one LEGO instruction page to decide if it needs detailed reading.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bag_marker": {"type": ["integer", "null"]},
            "is_parts_list": {"type": "boolean"},
            "has_callouts": {"type": "boolean"},
            "printed_page_number": {"type": ["integer", "null"]},
        },
        "required": ["bag_marker", "is_parts_list", "has_callouts"],
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
        triage_model: str = DEFAULT_TRIAGE_MODEL,
        api_key: Optional[str] = None,
        client=None,
    ):
        self.cache = cache
        self.model = model  # detailed pass
        self.triage_model = triage_model  # cheap pass
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

    def triage_page(self, page: RenderedPage, use_cache: bool = True) -> dict:
        """Cheap classification: {bag_marker, is_parts_list, has_callouts, printed_page_number}."""
        namespace = f"triage-{PROMPT_VERSION}-{self.triage_model}"
        key = self._cache_key(page.png_bytes)
        if use_cache:
            cached = self.cache.get(namespace, key)
            if cached is not None:
                return cached
        data = self._call_triage_model(page.png_bytes)
        self.cache.set(namespace, key, data)
        return data

    def _call_triage_model(self, png_bytes: bytes) -> dict:
        client = self._get_client()
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        message = client.messages.create(
            model=self.triage_model,
            max_tokens=512,
            system=_TRIAGE_SYSTEM,
            tools=[_TRIAGE_TOOL],
            tool_choice={"type": "tool", "name": "triage_page"},
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
                        {"type": "text", "text": _TRIAGE_INSTRUCTIONS},
                    ],
                }
            ],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "triage_page":
                return dict(block.input)
        raise VisionError("Model did not return the expected triage_page tool call.")


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


def _merge(page_index: int, triage: dict, detail: Optional[PageExtract]) -> PageExtract:
    """Combine a page's triage classification with its detailed callouts (if any)."""
    if detail is None:
        return PageExtract(
            page_index=page_index,
            printed_page_number=triage.get("printed_page_number"),
            bag_marker=triage.get("bag_marker"),
            is_parts_list=bool(triage.get("is_parts_list", False)),
            callouts=[],
        )
    # Triage saw every page, so trust its bag_marker; fall back to the detail read.
    bag_marker = triage.get("bag_marker")
    if bag_marker is None:
        bag_marker = detail.bag_marker
    return PageExtract(
        page_index=page_index,
        printed_page_number=triage.get("printed_page_number") or detail.printed_page_number,
        bag_marker=bag_marker,
        is_parts_list=bool(triage.get("is_parts_list", False)) or detail.is_parts_list,
        callouts=detail.callouts,
        notes=detail.notes,
    )


def extract_pages_two_pass(
    triage_pages: List[RenderedPage],
    detail_render,
    extractor: VisionExtractor,
    use_cache: bool = True,
    progress=None,
):
    """Two-pass extraction to cut cost.

    Pass 1 (cheap model, low DPI): triage every page for bag marker / BOM /
    whether it has callouts. Pass 2 (capable model, high DPI): read callouts
    only on pages triage flagged. Pages without callouts skip the expensive
    pass entirely.

    ``triage_pages`` are low-DPI RenderedPage objects for all pages.
    ``detail_render(page_index) -> RenderedPage`` renders a single page at high
    DPI on demand. Returns ``(page_extracts, stats)``.
    """
    triage_by_index: dict = {}
    detail_needed: List[int] = []
    for page in sorted(triage_pages, key=lambda p: p.page_index):
        t = extractor.triage_page(page, use_cache=use_cache)
        triage_by_index[page.page_index] = t
        if t.get("has_callouts") and not t.get("is_parts_list"):
            detail_needed.append(page.page_index)

    detail_by_index: dict = {}
    for i, idx in enumerate(detail_needed):
        rendered = detail_render(idx)
        detail_by_index[idx] = extractor.extract_page(rendered, use_cache=use_cache)
        if progress:
            progress(i + 1, len(detail_needed))

    results = [
        _merge(idx, triage_by_index[idx], detail_by_index.get(idx))
        for idx in sorted(triage_by_index)
    ]
    stats = {
        "triaged": len(triage_by_index),
        "detailed": len(detail_needed),
        "skipped": len(triage_by_index) - len(detail_needed),
    }
    return results, stats
