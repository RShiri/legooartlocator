"""Shared contract for local page detection.

``vision_local.detect_page`` (OpenCV) produces a ``PageDetection`` per page;
``locate.py`` consumes it, feeding each callout crop to the ensemble identifier.
Keeping the dataclasses here lets the detector and the orchestrator be built
independently against a fixed interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .models import PageCallout, PageExtract


@dataclass
class DetectedCallout:
    """One detected parts-callout: the parsed fields plus the cropped image bytes."""

    callout: PageCallout            # quantity (+ colour/shape if the detector filled them)
    crop_png: bytes                 # PNG bytes of the cropped part render (fed to identifiers)
    bbox: Optional[tuple] = None    # (x, y, w, h) in page-pixel space, for debugging/overlay


@dataclass
class PageDetection:
    """Structured result of locally detecting one page image."""

    page_index: int
    bag_marker: Optional[int] = None
    is_parts_list: bool = False
    printed_page_number: Optional[int] = None
    callouts: List[DetectedCallout] = field(default_factory=list)
    notes: Optional[str] = None
    # Set when the detector finds a plausible tall bag-numeral glyph but cannot
    # (or does not) read it via OCR -- e.g. no Tesseract binary installed. Lets
    # callers (see vision_local.assign_ordinal_bag_numbers) assign the bag number
    # ordinally (1..N in page order) instead of leaving it unknown.
    bag_marker_candidate: bool = False
    bag_marker_bbox: Optional[tuple] = None

    def to_page_extract(self) -> PageExtract:
        """Project into the pipeline's PageExtract (drops crop bytes)."""
        return PageExtract(
            page_index=self.page_index,
            printed_page_number=self.printed_page_number,
            bag_marker=self.bag_marker,
            is_parts_list=self.is_parts_list,
            callouts=[dc.callout for dc in self.callouts],
            notes=self.notes,
        )

    def crops(self) -> List[bytes]:
        """Parallel list of crop PNGs, aligned with ``to_page_extract().callouts``."""
        return [dc.crop_png for dc in self.callouts]
