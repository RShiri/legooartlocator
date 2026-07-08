"""Visual debug/calibration tooling for the local (OpenCV) page detector.

``vision_local.LocalDetector`` is threshold-driven (``DetectConfig``), and its
defaults have never been validated against a real scanned manual. This module
turns a ``PageDetection`` (or a whole PDF) into human-inspectable feedback:

* an annotated overlay image -- red boxes on each detected callout cell
  (labeled with its parsed quantity), a blue box on the bag-marker glyph if
  located, and a one-line summary banner;
* the raw callout-panel threshold mask (the same gray band
  ``DetectConfig.panel_gray_low``/``panel_gray_high`` gates on); and
* a per-page stats dict whose fields map 1:1 onto the ``DetectConfig`` gates
  (``min_cell_area``/``max_cell_area``, ``min_aspect``/``max_aspect``,
  ``bom_cell_count``), so new thresholds can be read straight off real pages.

``run_debug`` drives all three across a rendered PDF and writes them to disk
for eyeballing.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .detection import PageDetection
from .pdf_render import page_count, parse_page_range, render_pages
from .vision_local import OCR, DetectConfig, ImageInput, LocalDetector

# BGR colours (OpenCV convention).
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _as_bgr(image: ImageInput) -> Optional[np.ndarray]:
    """Decode PNG/JPEG bytes to a BGR ndarray, or pass an ndarray through."""
    return LocalDetector._as_bgr(image)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Gray-convert a 2D/BGR/BGRA ndarray, matching ``LocalDetector``'s handling."""
    return LocalDetector._to_gray(img)


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover - only fails on a malformed/empty array
        raise ValueError("cv2.imencode failed to encode PNG")
    return buf.tobytes()


def draw_overlay(image: ImageInput, detection: PageDetection) -> bytes:
    """Draw callout/bag-marker boxes and a summary banner; return PNG bytes.

    Red rectangles mark each non-``None`` ``DetectedCallout.bbox``, labeled
    with its parsed quantity. A blue rectangle marks
    ``getattr(detection, "bag_marker_bbox", None)`` -- accessed via
    ``getattr`` because that field may not exist yet on every
    ``PageDetection`` (it is being added in parallel elsewhere); its absence
    is tolerated, not an error. Works on a copy -- never mutates ``image``.
    """
    img = _as_bgr(image)
    if img is None:
        raise ValueError("draw_overlay: could not decode image")
    canvas = img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    elif canvas.shape[2] == 4:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)

    for dc in detection.callouts:
        if dc.bbox is None:
            continue
        x, y, w, h = dc.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _RED, 2)
        label_y = y - 6 if y - 6 > 12 else y + 16
        cv2.putText(canvas, str(dc.callout.quantity), (x + 2, label_y), _FONT, 0.6, _RED, 2, cv2.LINE_AA)

    bag_bbox = getattr(detection, "bag_marker_bbox", None)
    if bag_bbox is not None:
        x, y, w, h = bag_bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), _BLUE, 2)
        bag_label = str(detection.bag_marker) if detection.bag_marker is not None else "bag?"
        label_y = y - 6 if y - 6 > 12 else y + 16
        cv2.putText(canvas, bag_label, (x + 2, label_y), _FONT, 0.6, _BLUE, 2, cv2.LINE_AA)

    banner = (
        f"page {detection.page_index + 1} | {len(detection.callouts)} callouts | "
        f"BOM={detection.is_parts_list} | bag={detection.bag_marker}"
    )
    (text_w, text_h), baseline = cv2.getTextSize(banner, _FONT, 0.6, 2)
    cv2.rectangle(canvas, (0, 0), (text_w + 12, text_h + baseline + 12), _WHITE, -1)
    cv2.putText(canvas, banner, (6, text_h + 6), _FONT, 0.6, _BLACK, 2, cv2.LINE_AA)

    return _encode_png(canvas)


def panel_mask(image: ImageInput, config: Optional[DetectConfig] = None) -> bytes:
    """Threshold ``image`` on the callout-panel gray band; return the mask as PNG bytes."""
    cfg = config or DetectConfig()
    img = _as_bgr(image)
    if img is None:
        raise ValueError("panel_mask: could not decode image")
    gray = _to_gray(img)
    mask = cv2.inRange(gray, cfg.panel_gray_low, cfg.panel_gray_high)
    return _encode_png(mask)


def _min_median_max(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def page_stats(detection: PageDetection, image_shape: tuple) -> Dict[str, Any]:
    """Per-page stats mapped 1:1 onto ``DetectConfig`` gates.

    ``cell_area_fracs`` (min/median/max of ``(w*h)/(H*W)`` per bbox) reads
    straight against ``min_cell_area``/``max_cell_area``; ``cell_aspects``
    (``w/h``) reads against ``min_aspect``/``max_aspect``; ``n_callouts``
    reads against ``bom_cell_count``. Empty callouts yield ``None`` stats.
    """
    height, width = image_shape[0], image_shape[1]
    page_area = float(height * width)

    quantities = [dc.callout.quantity for dc in detection.callouts]
    area_fracs: List[float] = []
    aspects: List[float] = []
    for dc in detection.callouts:
        if dc.bbox is None:
            continue
        x, y, w, h = dc.bbox
        if page_area > 0:
            area_fracs.append((w * h) / page_area)
        if h > 0:
            aspects.append(w / h)

    return {
        "page_index": detection.page_index,
        "n_callouts": len(detection.callouts),
        "is_parts_list": detection.is_parts_list,
        "bag_marker": detection.bag_marker,
        "quantities": quantities,
        "cell_area_fracs": _min_median_max(area_fracs),
        "cell_aspects": _min_median_max(aspects),
    }


def run_debug(
    pdf_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    dpi: int = 180,
    page_spec: Optional[str] = None,
    config: Optional[DetectConfig] = None,
    ocr: Optional[OCR] = None,
    detector: Optional[LocalDetector] = None,
) -> Dict[str, Any]:
    """Render a PDF's pages, detect, and write overlays/masks/stats to ``out_dir``.

    Writes ``page_{idx:03d}.png`` (annotated overlay) and ``mask_{idx:03d}.png``
    (panel threshold mask) per rendered page, plus ``stats.json`` holding the
    same dict this function returns -- ``{"pdf", "dpi", "pages": [...]}`` --
    so thresholds can be tuned by eye against a real manual.

    One page's failure (a decode error, an unexpected detector exception) is
    recorded as ``{"page_index", "error"}`` in ``pages`` and does not abort the
    rest of the run -- this is exactly the calibration tool meant to survive a
    real, imperfect scanned manual, so one bad page losing every other page's
    output would defeat its purpose. ``stats.json`` is (re)written after every
    page, not just at the end, so progress already made survives even if a
    later page crashes hard enough to abort the process entirely.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    det = detector or LocalDetector(config or DetectConfig(), ocr=ocr)
    indices = parse_page_range(page_spec, page_count(pdf_path))

    pages_stats: List[Dict[str, Any]] = []

    def _write_stats() -> Dict[str, Any]:
        result = {"pdf": str(pdf_path), "dpi": dpi, "pages": pages_stats}
        (out / "stats.json").write_text(json.dumps(result, indent=2))
        return result

    for rendered in render_pages(pdf_path, dpi=dpi, page_indices=indices):
        idx = rendered.page_index
        try:
            img = _as_bgr(rendered.png_bytes)
            if img is None:
                raise ValueError("could not decode rendered page image")
            detection = det.detect_page(rendered.png_bytes, idx)

            (out / f"page_{idx:03d}.png").write_bytes(draw_overlay(img, detection))
            (out / f"mask_{idx:03d}.png").write_bytes(panel_mask(img, det.config))
            pages_stats.append(page_stats(detection, img.shape))
        except Exception as exc:  # noqa: BLE001 - isolate one bad page, keep going
            pages_stats.append({"page_index": idx, "error": str(exc)})
        finally:
            _write_stats()

    return _write_stats()
