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
from typing import Dict, List, Optional, Sequence, Tuple

from .bags import page_to_bag, segment_bags
from .brickognize import BrickognizeClient
from .colors import crop_color_name, rgb_color_scorer
from .detection import PageDetection
from .embedding import EmbeddingBackend, Gallery
from .identify import IdentificationResult, PartIdentifier
from .models import InventoryPart, LocatedPart, Occurrence, ScanResult
from .reconcile import _inv_key

# Capacity-reconcile tunables (see _resolve_assignments).
_EMBED_ONLY_EPS = 1e-9
_MIN_REROUTE_CONFIDENCE = 0.3


def _is_embedding_only(components: Dict[str, float]) -> bool:
    """True when a match was carried by embedding retrieval alone -- Brickognize
    contributed nothing. These are the ones the embedding "attractor" inflates,
    so they're the only matches capacity-reconcile is willing to divert."""
    if not components:
        return False
    return (
        components.get("brickognize", 0.0) <= _EMBED_ONLY_EPS
        and components.get("embedding", 0.0) > _EMBED_ONLY_EPS
    )


def _resolve_assignments(
    pending: List[tuple], capacity_reconcile: bool
) -> List[Tuple[Optional[InventoryPart], float]]:
    """Decide each crop's final (part, confidence).

    Default (``capacity_reconcile`` False): every crop keeps the identifier's
    own winner -- byte-for-byte the previous behaviour.

    With capacity-reconcile: crops are resolved in descending confidence, and a
    part that has already reached its inventory quantity stops accepting further
    *embedding-only* matches (the attractor over-count). Such a crop is diverted
    to its best still-available alternative above ``_MIN_REROUTE_CONFIDENCE``, or
    to unknown if none qualifies. A Brickognize-corroborated winner is never
    diverted, so genuinely repeated parts are preserved.

    ``pending`` items are
    ``(bag, page_index, qty, seen_color, crop_png, IdentificationResult)``.
    """
    if not capacity_reconcile:
        return [(r.part, r.confidence) for (_b, _p, _q, _c, _png, r) in pending]

    order = sorted(range(len(pending)), key=lambda i: pending[i][5].confidence, reverse=True)
    assigned_qty: Dict[str, int] = {}
    out: List[Tuple[Optional[InventoryPart], float]] = [(None, 0.0)] * len(pending)
    for i in order:
        _bag, _page, qty, _color, _png, res = pending[i]
        if res.part is None:
            continue
        # Winner first, then the ranked alternatives as fallback homes.
        candidates = [(res.part, float(res.confidence), _is_embedding_only(res.components))]
        candidates += [(p, float(s), False) for (p, s) in res.alternatives]
        for part, score, emb_only in candidates:
            is_winner = part is res.part
            if not is_winner and score < _MIN_REROUTE_CONFIDENCE:
                continue  # don't reroute into a weak alternative
            key = _inv_key(part)
            cap = part.quantity if part.quantity else None
            saturated = cap is not None and assigned_qty.get(key, 0) >= cap
            # A saturated part only still accepts a corroborated winner; an
            # embedding-only winner or any alternative is turned away.
            if saturated and (emb_only or not is_winner):
                continue
            out[i] = (part, score)
            assigned_qty[key] = assigned_qty.get(key, 0) + qty
            break
    return out


def _dump_crops(
    pending: List[tuple],
    assignments: List[Tuple[Optional[InventoryPart], float]],
    dump_dir,
) -> None:
    """Persist every callout crop with its assignment metadata.

    Writes ``dump_dir/<part_num or 'unknown'>/pNNN_iNNN.png`` per crop plus one
    ``dump_dir/manifest.json`` describing them all — the labelled dataset that
    downstream work needs: embedding-floor calibration (score distributions),
    real-crop self-training (corroborated crops as in-domain labels), and the
    detection-miss vs. identification-miss diagnosis (what's in ``unknown/``).
    """
    import json
    from pathlib import Path as _Path

    out = _Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, ((bag, page_index, qty, seen_color, crop_png, res), (part, confidence)) in enumerate(
        zip(pending, assignments)
    ):
        label = part.part_num if part is not None and part.part_num else "unknown"
        rel = None
        if crop_png:
            sub = out / label
            sub.mkdir(parents=True, exist_ok=True)
            rel = f"{label}/p{page_index:03d}_i{i:03d}.png"
            (out / rel).write_bytes(crop_png)
        entries.append(
            {
                "file": rel,  # None when the crop had no PNG bytes
                "page_index": page_index,  # 0-based
                "bag": bag,
                "quantity": qty,
                "seen_color": seen_color,
                "part_num": part.part_num if part is not None else None,
                "color_name": part.color_name if part is not None else None,
                # The identifier's raw winner, before capacity-reconcile — differs
                # from part_num when the crop was diverted.
                "raw_part_num": res.part.part_num if res.part is not None else None,
                "confidence": float(confidence),
                "components": {k: float(v) for k, v in res.components.items()},
                "alternatives": [
                    [p.part_num, float(s)] for p, s in res.alternatives[:2]
                ],
            }
        )
    (out / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
    capacity_reconcile: bool = True,
    dump_crops_dir=None,
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

    # Phase 1: identify every crop, collecting reads without bucketing yet, so
    # capacity-reconcile (phase 2) can resolve them in confidence order.
    pending: List[tuple] = []  # (bag, page_index, qty, seen_color, crop_png, IdentificationResult)
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
            pending.append((bag, det.page_index, dc.callout.quantity, seen_color, dc.crop_png, result))

    # Phase 2: resolve each crop's winner (optionally capacity/trust-aware),
    # then bucket into LocatedParts exactly as before.
    assignments = _resolve_assignments(pending, capacity_reconcile and reconcile_counts)

    if dump_crops_dir is not None:
        _dump_crops(pending, assignments, dump_crops_dir)

    by_key: Dict[str, LocatedPart] = {}
    conf_acc: Dict[str, List[float]] = {}
    unidentified = 0

    for (bag, page_index, qty, seen_color, _png, _result), (part, confidence) in zip(pending, assignments):
        if part is not None:
            key = _inv_key(part)
            lp = by_key.get(key)
            if lp is None:
                lp = LocatedPart(
                    key=key, name=part.name, part_num=part.part_num,
                    color_name=part.color_name, element_id=part.element_id,
                    image_url=part.image_url,
                    inventory_qty=part.quantity if reconcile_counts else None,
                    reconciled=reconcile_counts,
                )
                by_key[key] = lp
                conf_acc[key] = []
            lp.occurrences.append(Occurrence(bag=bag, page_index=page_index, quantity=qty))
            conf_acc[key].append(confidence)
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
            lp.occurrences.append(Occurrence(bag=bag, page_index=page_index, quantity=qty))
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
                excess = lp.total_seen - lp.inventory_qty
                if 0 < excess < len(lp.occurrences):
                    # Over by less than the number of sightings: the same piece
                    # is simply shown in several build steps — expected for a
                    # location tool, so phrase it as information, not an error.
                    warnings.append(
                        f"{lp.name or lp.key} appears in {len(lp.occurrences)} step(s) "
                        f"(piece count {lp.inventory_qty}) — multi-step reuse, not a miscount."
                    )
                else:
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
    capacity_reconcile: bool = True,
    dump_crops_dir=None,
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

    # Fill in bag numbers that the detector spotted structurally but couldn't
    # OCR (no Tesseract binary, or an unreadable glyph) — real bags are always
    # numbered 1..N in page order, so this needs no digit reading at all.
    from .vision_local import assign_ordinal_bag_numbers

    ordinal_warnings = assign_ordinal_bag_numbers(detections)

    result = assemble_result(
        detections, inventory, identifier, total,
        use_color=use_color, reconcile_counts=reconcile_counts,
        capacity_reconcile=capacity_reconcile, dump_crops_dir=dump_crops_dir,
        source_pdf=str(pdf_path), set_num=set_num, set_name=set_name,
    )
    result.warnings.extend(ordinal_warnings)
    return result
