"""Turn per-page callouts into located parts, optionally reconciled to inventory.

Two modes:
  * vision-only: group callout occurrences by a normalised (colour, shape) key.
    Lower confidence, but always available.
  * reconciled: match each callout to a canonical Rebrickable inventory line by
    scoring colour + shape-name overlap + printed id, then validate that the
    quantity we saw across all bags equals the inventory total.

Matching small part renders is inherently fuzzy; the inventory acts as a
constraint (a known, closed set of parts) and count reconciliation surfaces
where the vision read is incomplete or over-counted.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .models import (
    InventoryPart,
    LocatedPart,
    Occurrence,
    PageCallout,
    PageExtract,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Colour words that carry no discriminating value on their own.
_STOP_COLORS = {"color", "colour"}


def _tokens(text: Optional[str]) -> set:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _norm_color(text: Optional[str]) -> str:
    toks = _tokens(text) - _STOP_COLORS
    return " ".join(sorted(toks))


def _vision_key(callout: PageCallout) -> str:
    return f"{_norm_color(callout.color)}|{_norm_color(callout.shape_desc)}"


def _score_match(callout: PageCallout, part: InventoryPart) -> float:
    """Heuristic 0..1 score of how well a callout matches an inventory line."""
    score = 0.0

    # Printed id is the strongest signal when present.
    pid = (callout.printed_part_id or "").strip().lower()
    if pid:
        if part.element_id and pid == part.element_id.strip().lower():
            return 1.0
        if part.part_num and pid == part.part_num.strip().lower():
            return 0.95

    # Colour overlap.
    c_tokens = _tokens(callout.color) - _STOP_COLORS
    p_color_tokens = _tokens(part.color_name) - _STOP_COLORS
    if c_tokens and p_color_tokens:
        overlap = len(c_tokens & p_color_tokens) / len(c_tokens | p_color_tokens)
        score += 0.5 * overlap

    # Shape/name overlap.
    s_tokens = _tokens(callout.shape_desc)
    n_tokens = _tokens(part.name)
    if s_tokens and n_tokens:
        overlap = len(s_tokens & n_tokens) / len(s_tokens | n_tokens)
        score += 0.5 * overlap

    return min(score, 0.9)  # reserve >0.9 for id-based matches


def match_callout_to_inventory(
    callout: PageCallout,
    inventory: List[InventoryPart],
    min_score: float = 0.35,
) -> Tuple[Optional[InventoryPart], float]:
    """Return the best-matching inventory line and its score, or (None, 0)."""
    best: Optional[InventoryPart] = None
    best_score = 0.0
    for part in inventory:
        s = _score_match(callout, part)
        if s > best_score:
            best, best_score = part, s
    if best_score < min_score:
        return None, 0.0
    return best, best_score


def _inv_key(part: InventoryPart) -> str:
    return f"{part.part_num}|{part.color_id if part.color_id is not None else part.color_name}"


def _finalize(part: LocatedPart) -> LocatedPart:
    """Compute derived fields (bags, pages, total_seen, count_matches, confidence)."""
    bags = sorted({o.bag for o in part.occurrences})
    pages = sorted({o.page_index for o in part.occurrences})
    total = sum(o.quantity for o in part.occurrences)
    part.bags = bags
    part.pages = pages
    part.total_seen = total

    if part.reconciled and part.inventory_qty is not None:
        part.count_matches = total == part.inventory_qty
        part.confidence = 0.9 if part.count_matches else 0.6
    else:
        part.count_matches = None
        part.confidence = 0.4
    return part


def reconcile(
    pages: List[PageExtract],
    page_to_bag: Dict[int, int],
    inventory: Optional[List[InventoryPart]] = None,
    min_score: float = 0.35,
) -> Tuple[List[LocatedPart], List[str]]:
    """Build the located-part list from page callouts.

    Returns (parts, warnings). When ``inventory`` is provided, callouts are
    matched to it and quantities are reconciled; otherwise a vision-only
    grouping is produced.
    """
    warnings: List[str] = []
    by_key: Dict[str, LocatedPart] = {}

    for page in sorted(pages, key=lambda p: p.page_index):
        if page.is_parts_list:
            continue  # BOM pages describe the whole set, not a bag
        bag = page_to_bag.get(page.page_index, 0)
        for callout in page.callouts:
            located = _resolve_callout(callout, inventory, by_key, min_score)
            located.occurrences.append(
                Occurrence(bag=bag, page_index=page.page_index, quantity=callout.quantity)
            )

    parts = [_finalize(p) for p in by_key.values()]

    if inventory is not None:
        parts, inv_warnings = _apply_inventory_totals(parts, inventory)
        warnings.extend(inv_warnings)

    parts.sort(key=lambda p: (p.bags[0] if p.bags else 0, p.name or p.key))
    return parts, warnings


def _resolve_callout(
    callout: PageCallout,
    inventory: Optional[List[InventoryPart]],
    by_key: Dict[str, LocatedPart],
    min_score: float,
) -> LocatedPart:
    """Find-or-create the LocatedPart bucket for a callout."""
    if inventory:
        match, score = match_callout_to_inventory(callout, inventory, min_score)
        if match is not None:
            key = _inv_key(match)
            if key not in by_key:
                by_key[key] = LocatedPart(
                    key=key,
                    name=match.name,
                    part_num=match.part_num,
                    color_name=match.color_name,
                    element_id=match.element_id,
                    image_url=match.image_url,
                    inventory_qty=match.quantity,
                    reconciled=True,
                )
            return by_key[key]

    # Vision-only fallback bucket.
    key = f"vision:{_vision_key(callout)}"
    if key not in by_key:
        by_key[key] = LocatedPart(
            key=key,
            name=callout.shape_desc,
            color_name=callout.color,
            reconciled=False,
        )
    return by_key[key]


def _apply_inventory_totals(
    parts: List[LocatedPart], inventory: List[InventoryPart]
) -> Tuple[List[LocatedPart], List[str]]:
    """Recompute count_matches and flag parts whose seen totals don't reconcile."""
    warnings: List[str] = []
    for part in parts:
        if not part.reconciled or part.inventory_qty is None:
            continue
        part.count_matches = part.total_seen == part.inventory_qty
        if not part.count_matches:
            warnings.append(
                f"Count mismatch for {part.name or part.key}: saw {part.total_seen} "
                f"in pages, inventory has {part.inventory_qty}."
            )
    return parts, warnings
