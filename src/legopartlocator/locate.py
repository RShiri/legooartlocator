"""Local end-to-end orchestration: PDF -> detect -> identify -> ScanResult.

This is the free, no-paid-API path. It renders pages, runs the OpenCV
detector (``vision_local``) to find callout crops + bag markers, then runs each
crop through the ensemble identifier (``identify``) — Brickognize + embedding +
colour, constrained to the set inventory — and assembles the same ScanResult
the vision path produces.

``assemble_result`` is the pure, unit-tested core (takes detections + an
identifier, no PDF/rendering needed). ``locate_local`` wraps it with rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .bags import page_to_bag, segment_bags
from .brickognize import BrickognizeClient
from .colors import crop_color_name, rgb_color_scorer
from .detection import PageDetection
from .embedding import EmbeddingBackend, Gallery
from .identify import IdentificationResult, PartIdentifier
from .models import InventoryPart, LocatedPart, Occurrence, ScanResult
from .reconcile import _inv_key


def make_identifier(
    inventory: Sequence[InventoryPart],
    brickognize: Optional[BrickognizeClient] = None,
    gallery: Optional[Gallery] = None,
    backend: Optional[EmbeddingBackend] = None,
    use_color: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> PartIdentifier:
    """Construct a PartIdentifier wired with the RGB colour scorer for the local path."""
    return PartIdentifier(
        inventory,
        brickognize=brickognize,
        gallery=gallery,
        backend=backend,
        weights=weights,
        color_scorer=rgb_color_scorer if use_color else (lambda _c, _p: 0.0),
    )


def assemble_result(
    detections: Sequence[PageDetection],
    inventory: Sequence[InventoryPart],
    identifier: PartIdentifier,
    num_pages: int,
    *,
    use_color: bool = True,
    reconcile_counts: bool = True,
    source_pdf: Optional[str] = None,
    set_num: Optional[str] = None,
    set_name: Optional[str] = None,
) -> ScanResult:
    """Turn page detections into a ScanResult by identifying each callout crop.

    Pure and side-effect free: ``identifier`` is injected, so this is tested with
    synthetic detections and a fake identifier (no OpenCV/torch/network).

    When ``reconcile_counts`` is False (no canonical inventory, e.g. the
    Brickognize-only path), identified parts are kept but not count-validated.
    """
    page_extracts = [d.to_page_extract() for d in detections]
    segments, warnings = segment_bags(page_extracts, num_pages)
    p2b = page_to_bag(segments)

    by_key: Dict[str, LocatedPart] = {}
    conf_acc: Dict[str, List[float]] = {}
    unidentified = 0
    identify_failures = 0
    identify_failure_msgs: List[str] = []

    for det in sorted(detections, key=lambda d: d.page_index):
        if det.is_parts_list:
            continue  # BOM page describes the whole set, not a bag
        bag = p2b.get(det.page_index, 0)
        for dc in det.callouts:
            seen_color = None
            if use_color and dc.crop_png:
                try:
                    seen_color = crop_color_name(dc.crop_png)
                except Exception:
                    seen_color = None
            try:
                result = identifier.identify(dc.crop_png, seen_color=seen_color)
            except Exception as exc:
                identify_failures += 1
                msg = f"identify failed on page {det.page_index + 1}: {exc}"
                if len(identify_failure_msgs) < 3 and msg not in identify_failure_msgs:
                    identify_failure_msgs.append(msg)
                result = IdentificationResult(part=None, confidence=0.0)
            qty = dc.callout.quantity

            if result.part is not None:
                key = _inv_key(result.part)
                lp = by_key.get(key)
                if lp is None:
                    p = result.part
                    lp = LocatedPart(
                        key=key, name=p.name, part_num=p.part_num,
                        color_name=p.color_name, element_id=p.element_id,
                        image_url=p.image_url,
                        inventory_qty=p.quantity if reconcile_counts else None,
                        reconciled=reconcile_counts,
                    )
                    by_key[key] = lp
                    conf_acc[key] = []
                lp.occurrences.append(Occurrence(bag=bag, page_index=det.page_index, quantity=qty))
                conf_acc[key].append(result.confidence)
            else:
                unidentified += 1
                key = f"unknown:{seen_color or '?'}"
                lp = by_key.get(key)
                if lp is None:
                    lp = LocatedPart(
                        key=key, name=f"Unidentified ({seen_color or 'unknown colour'})",
                        color_name=seen_color, reconciled=False,
                    )
                    by_key[key] = lp
                    conf_acc[key] = []
                lp.occurrences.append(Occurrence(bag=bag, page_index=det.page_index, quantity=qty))
                conf_acc[key].append(0.0)

    for key, lp in by_key.items():
        lp.bags = sorted({o.bag for o in lp.occurrences})
        lp.pages = sorted({o.page_index for o in lp.occurrences})
        lp.total_seen = sum(o.quantity for o in lp.occurrences)
        confs = conf_acc.get(key, [])
        lp.confidence = round(sum(confs) / len(confs), 3) if confs else 0.0
        if lp.reconciled and lp.inventory_qty is not None:
            lp.count_matches = lp.total_seen == lp.inventory_qty
            if not lp.count_matches:
                warnings.append(
                    f"Count mismatch for {lp.name or lp.key}: saw {lp.total_seen}, "
                    f"inventory has {lp.inventory_qty}."
                )

    warnings.extend(identify_failure_msgs)
    if identify_failures > 3:
        warnings.append(
            f"{identify_failures} identify call(s) failed in total "
            f"(only the first 3 distinct messages are shown above)."
        )

    if unidentified:
        warnings.append(f"{unidentified} callout(s) could not be identified against the inventory.")

    parts = sorted(by_key.values(), key=lambda p: (p.bags[0] if p.bags else 0, p.name or p.key))
    return ScanResult(
        set_num=set_num, set_name=set_name, source_pdf=source_pdf,
        num_pages=num_pages, reconciled=reconcile_counts, bags=segments, parts=parts, warnings=warnings,
    )


def locate_local(
    pdf_path: str | Path,
    inventory: Sequence[InventoryPart],
    identifier: PartIdentifier,
    *,
    dpi: int = 200,
    page_spec: Optional[str] = None,
    max_pages: Optional[int] = None,
    detector=None,
    use_color: bool = True,
    reconcile_counts: bool = True,
    set_num: Optional[str] = None,
    set_name: Optional[str] = None,
    progress=None,
) -> ScanResult:
    """Render a PDF, detect locally, identify each crop, and assemble the result."""
    from .pdf_render import page_count, parse_page_range, render_pages
    from .vision_local import LocalDetector

    detector = detector or LocalDetector()
    total = page_count(pdf_path)
    indices = parse_page_range(page_spec, total)
    if max_pages is not None:
        indices = indices[:max_pages]

    detections: List[PageDetection] = []
    for i, rendered in enumerate(render_pages(pdf_path, dpi=dpi, page_indices=indices)):
        detections.append(detector.detect_page(rendered.png_bytes, rendered.page_index))
        if progress:
            progress(i + 1, len(indices))

    return assemble_result(
        detections, inventory, identifier, total,
        use_color=use_color, reconcile_counts=reconcile_counts,
        source_pdf=str(pdf_path), set_num=set_num, set_name=set_name,
    )
