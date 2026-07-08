"""Pydantic data models shared across the pipeline.

The pipeline flows:
    PDF -> [PageExtract per page] -> [BagSegment list] -> reconcile with
    [InventoryPart list] -> [LocatedPart list] -> ScanResult (JSON/CSV).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PageCallout(BaseModel):
    """A single parts-callout entry read off a build page (e.g. "2x <brick>")."""

    quantity: int = Field(..., ge=1, description="Count shown in the callout (the Nx).")
    color: Optional[str] = Field(None, description="Colour as seen, free text (e.g. 'dark red').")
    shape_desc: Optional[str] = Field(
        None, description="Short shape description (e.g. '2x4 brick', '1x2 plate with clip')."
    )
    printed_part_id: Optional[str] = Field(
        None, description="Design/element id printed in/near the callout, if any."
    )


class PageExtract(BaseModel):
    """Everything the vision pass extracted from one page image."""

    page_index: int = Field(..., ge=0, description="0-based page index within the PDF.")
    printed_page_number: Optional[int] = Field(
        None, description="Page number printed on the page, if detected."
    )
    bag_marker: Optional[int] = Field(
        None, description="Bag number if this page starts a new numbered bag section."
    )
    is_parts_list: bool = Field(
        False, description="True if this page is a full BOM/parts-list grid rather than a build step."
    )
    callouts: List[PageCallout] = Field(default_factory=list)
    notes: Optional[str] = None


class BagSegment(BaseModel):
    """A contiguous run of pages that belong to one numbered bag."""

    bag: int = Field(..., description="Bag number. 0 is used for pre-bag intro pages.")
    start_page: int = Field(..., ge=0, description="First 0-based page index (inclusive).")
    end_page: int = Field(..., ge=0, description="Last 0-based page index (inclusive).")

    @model_validator(mode="after")
    def _check_page_order(self) -> "BagSegment":
        # An inverted segment would make page_to_bag's range(start, end+1)
        # silently return empty, dropping the page from the bag mapping with
        # no warning -- catch it at construction instead.
        if self.end_page < self.start_page:
            raise ValueError(
                f"BagSegment.end_page ({self.end_page}) must be >= start_page ({self.start_page})"
            )
        return self

    def contains(self, page_index: int) -> bool:
        return self.start_page <= page_index <= self.end_page


class InventoryPart(BaseModel):
    """One line of the canonical Rebrickable set inventory."""

    part_num: str
    name: str
    color_id: Optional[int] = None
    color_name: Optional[str] = None
    quantity: int = Field(..., ge=0, description="Total quantity of this part in the whole set.")
    element_id: Optional[str] = None
    image_url: Optional[str] = None


class Occurrence(BaseModel):
    """A place a part was seen: which bag, which page, how many."""

    bag: int
    page_index: int
    quantity: int


class LocatedPart(BaseModel):
    """A part with its resolved location(s) across bags/pages."""

    # Canonical identity (from Rebrickable when reconciled, else best-effort from vision).
    key: str = Field(..., description="Stable identity key (part_num+color or a vision fallback).")
    name: Optional[str] = None
    part_num: Optional[str] = None
    color_name: Optional[str] = None
    element_id: Optional[str] = None
    image_url: Optional[str] = None

    occurrences: List[Occurrence] = Field(default_factory=list)
    bags: List[int] = Field(default_factory=list, description="Sorted unique bag numbers.")
    pages: List[int] = Field(default_factory=list, description="Sorted unique 0-based page indices.")
    total_seen: int = Field(0, description="Sum of quantities seen across occurrences.")
    inventory_qty: Optional[int] = Field(
        None, description="Total quantity per the canonical inventory (if reconciled)."
    )
    reconciled: bool = Field(False, description="True if matched to an inventory line.")
    count_matches: Optional[bool] = Field(
        None, description="True if total_seen == inventory_qty (only when reconciled)."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ScanResult(BaseModel):
    """Top-level output serialised to result.json."""

    set_num: Optional[str] = None
    set_name: Optional[str] = None
    source_pdf: Optional[str] = None
    num_pages: int = 0
    reconciled: bool = False
    bags: List[BagSegment] = Field(default_factory=list)
    parts: List[LocatedPart] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def bag_to_parts(self) -> dict:
        """Reverse index: bag number -> list of part keys located in it."""
        index: dict = {}
        for part in self.parts:
            for bag in part.bags:
                index.setdefault(bag, []).append(part.key)
        return index
