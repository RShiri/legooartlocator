"""Tests for the visual debug/calibration overlay tooling (offline, synthetic).

Follows the ``tests/test_vision_local.py`` pattern: synthetic numpy/cv2 pages
and a deterministic ``FakeOCR`` stand-in, so nothing here touches Tesseract or
needs a real scanned manual. ``run_debug`` is additionally exercised against a
tiny two-page PDF generated on the fly with ``fitz`` (PyMuPDF).
"""

from __future__ import annotations

import json

import cv2
import fitz
import numpy as np

from legopartlocator.debug_overlay import draw_overlay, page_stats, panel_mask, run_debug
from legopartlocator.detection import DetectedCallout, PageDetection
from legopartlocator.models import PageCallout
from legopartlocator.vision_local import DetectConfig

GRAY = (220, 220, 220)  # light-gray callout background, same band as test_vision_local


class FakeOCR:
    """Deterministic OCR stand-in that always returns a fixed string."""

    def __init__(self, text: str = ""):
        self.text = text

    def read_text(self, image_bgr: np.ndarray) -> str:
        return self.text


def _blank_page(h: int = 200, w: int = 300) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _callout(quantity: int, bbox) -> DetectedCallout:
    return DetectedCallout(callout=PageCallout(quantity=quantity), crop_png=b"", bbox=bbox)


# --- draw_overlay --------------------------------------------------------------

def test_draw_overlay_draws_boxes_and_round_trips_shape():
    page = _blank_page()
    # Kept clear of the top-left summary banner so its white backing rect
    # can't paint over these edges.
    bbox1 = (10, 70, 50, 30)
    bbox2 = (150, 120, 60, 40)
    detection = PageDetection(
        page_index=2,
        bag_marker=None,
        is_parts_list=False,
        callouts=[_callout(3, bbox1), _callout(5, bbox2)],
    )

    png = draw_overlay(page, detection)
    assert isinstance(png, (bytes, bytearray))
    assert len(png) > 0

    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == page.shape

    # Original array must not be mutated.
    assert np.array_equal(page, _blank_page())

    # Midpoint of each bbox's top edge should be red-ish (B, G low; R high).
    for (x, y, w, h) in (bbox1, bbox2):
        b, g, r = decoded[y, x + w // 2]
        assert int(r) > 180
        assert int(b) < 80 and int(g) < 80


def test_draw_overlay_tolerates_missing_bag_marker_bbox_attribute():
    page = _blank_page()
    # A plain PageDetection -- whether or not ``bag_marker_bbox`` exists on this
    # dataclass (it is being added in parallel elsewhere), draw_overlay must
    # access it via getattr and must not raise either way.
    detection = PageDetection(page_index=0, callouts=[])

    png = draw_overlay(page, detection)
    assert isinstance(png, (bytes, bytearray))
    assert len(png) > 0
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == page.shape


# --- panel_mask ------------------------------------------------------------------

def test_panel_mask_flags_gray_region_only():
    page = _blank_page(h=400, w=400)
    cv2.rectangle(page, (50, 50), (150, 120), GRAY, thickness=-1)

    png = panel_mask(page, DetectConfig())
    mask = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert mask is not None
    assert mask.shape[:2] == (400, 400)

    # Inside the gray rectangle: nonzero.
    assert mask[85, 100] != 0
    # White background: zero.
    assert mask[5, 5] == 0
    assert mask[390, 390] == 0


# --- page_stats --------------------------------------------------------------

def test_page_stats_known_bboxes():
    image_shape = (200, 300, 3)  # H=200, W=300 -> page_area=60000
    bbox1 = (0, 0, 100, 50)  # area_frac = 5000/60000, aspect = 2.0
    bbox2 = (0, 0, 50, 50)  # area_frac = 2500/60000, aspect = 1.0
    detection = PageDetection(
        page_index=5,
        bag_marker=7,
        is_parts_list=False,
        callouts=[_callout(2, bbox1), _callout(4, bbox2)],
    )

    stats = page_stats(detection, image_shape)
    assert stats["page_index"] == 5
    assert stats["n_callouts"] == 2
    assert stats["is_parts_list"] is False
    assert stats["bag_marker"] == 7
    assert stats["quantities"] == [2, 4]

    frac1, frac2 = 5000 / 60000, 2500 / 60000
    assert stats["cell_area_fracs"]["min"] == frac2
    assert stats["cell_area_fracs"]["max"] == frac1
    assert stats["cell_area_fracs"]["median"] == (frac1 + frac2) / 2

    assert stats["cell_aspects"]["min"] == 1.0
    assert stats["cell_aspects"]["max"] == 2.0
    assert stats["cell_aspects"]["median"] == 1.5


def test_page_stats_empty_callouts_gives_none():
    detection = PageDetection(page_index=1, callouts=[])
    stats = page_stats(detection, (200, 300, 3))
    assert stats["n_callouts"] == 0
    assert stats["quantities"] == []
    assert stats["cell_area_fracs"] == {"min": None, "median": None, "max": None}
    assert stats["cell_aspects"] == {"min": None, "median": None, "max": None}


# --- run_debug (real tiny PDF via fitz) -------------------------------------------

def _make_two_page_pdf(path) -> None:
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=612, height=792)
        # A light-gray rect in the same 0.86 band used across this repo's fake pages.
        page.draw_rect(fitz.Rect(40, 40, 140, 110), color=None, fill=(0.86, 0.86, 0.86), fill_opacity=1)
    doc.save(str(path))
    doc.close()


def test_run_debug_writes_overlays_masks_and_stats(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    _make_two_page_pdf(pdf_path)
    out_dir = tmp_path / "debug_out"

    result = run_debug(pdf_path, out_dir, dpi=150, ocr=FakeOCR(text=""))

    for name in ("page_000.png", "mask_000.png", "page_001.png", "mask_001.png", "stats.json"):
        assert (out_dir / name).exists(), name

    assert result["pdf"] == str(pdf_path)
    assert result["dpi"] == 150
    assert len(result["pages"]) == 2
    assert [p["page_index"] for p in result["pages"]] == [0, 1]
    for p in result["pages"]:
        assert p["n_callouts"] == 1  # one gray rect per page

    on_disk = json.loads((out_dir / "stats.json").read_text())
    assert on_disk == result
