"""Turn sparse per-page bag markers into a dense page -> bag segmentation.

Bag membership is a step function over page order: once a page declares
"bag N", every following page belongs to bag N until the next marker. Pages
before the first marker are attributed to bag 0 (front matter / intro).

We are defensive about noisy vision output:
  * markers must be non-decreasing to be trusted (a lone out-of-order number is
    almost always a misread step number, not a bag divider); such markers are
    dropped and surfaced as warnings,
  * duplicate consecutive markers for the same bag are collapsed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import BagSegment, PageExtract


def _collect_markers(pages: List[PageExtract]) -> List[Tuple[int, int]]:
    """Return [(page_index, bag_number)] for pages that carry a bag marker, in page order."""
    markers = [
        (p.page_index, int(p.bag_marker))
        for p in sorted(pages, key=lambda p: p.page_index)
        if p.bag_marker is not None
    ]
    return markers


def _filter_monotonic(markers: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Keep only markers whose bag number strictly increases; drop the rest as suspect."""
    kept: List[Tuple[int, int]] = []
    warnings: List[str] = []
    last_bag = 0
    for page_index, bag in markers:
        if bag <= last_bag:
            warnings.append(
                f"Ignored suspect bag marker {bag} on page {page_index + 1} "
                f"(not greater than previous bag {last_bag})."
            )
            continue
        kept.append((page_index, bag))
        last_bag = bag
    return kept, warnings


def segment_bags(pages: List[PageExtract], num_pages: int) -> Tuple[List[BagSegment], List[str]]:
    """Compute contiguous bag segments covering pages [0, num_pages).

    Returns (segments, warnings). Segments are sorted by start page and cover
    every page exactly once. Bag 0 covers any intro pages before the first
    marker (omitted if the first marker is on page 0).
    """
    warnings: List[str] = []
    if num_pages <= 0:
        return [], warnings

    raw_markers = _collect_markers(pages)
    markers, mono_warnings = _filter_monotonic(raw_markers)
    warnings.extend(mono_warnings)

    if not markers:
        warnings.append("No bag markers detected; treating the whole PDF as bag 0 (unknown).")
        return [BagSegment(bag=0, start_page=0, end_page=num_pages - 1)], warnings

    segments: List[BagSegment] = []

    # Intro pages before the first marker -> bag 0.
    first_page = markers[0][0]
    if first_page > 0:
        segments.append(BagSegment(bag=0, start_page=0, end_page=first_page - 1))

    for i, (page_index, bag) in enumerate(markers):
        start = page_index
        end = (markers[i + 1][0] - 1) if i + 1 < len(markers) else (num_pages - 1)
        segments.append(BagSegment(bag=bag, start_page=start, end_page=end))

    return segments, warnings


def page_to_bag(segments: List[BagSegment]) -> Dict[int, int]:
    """Flatten segments into a {page_index: bag} lookup."""
    mapping: Dict[int, int] = {}
    for seg in segments:
        for page_index in range(seg.start_page, seg.end_page + 1):
            mapping[page_index] = seg.bag
    return mapping
