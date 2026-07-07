"""Tests for the local OpenCV page detector (offline: fake OCR, no Tesseract).

Synthetic pages are built with numpy/cv2: a white canvas with filled light-gray
rectangles standing in for callout cells, and (optionally) a tall black block
standing in for a bag-start numeral. A ``FakeOCR`` returns controlled strings
keyed off the crop size, so every assertion is deterministic.
"""

from __future__ import annotations

import cv2
import numpy as np

from legopartlocator.detection import DetectedCallout, PageDetection
from legopartlocator.models import PageExtract
from legopartlocator.vision_local import (
    DetectConfig,
    LocalDetector,
    TesseractOCR,
    assign_ordinal_bag_numbers,
    detect_page,
)

PAGE_H, PAGE_W = 1000, 800
GRAY = (220, 220, 220)  # light-gray callout background
BLACK = (0, 0, 0)

# Cells are ~70px tall; the bag numeral block is much taller. FakeOCR splits on
# this height to decide what to "read".
BIG_HEIGHT = 150


class FakeOCR:
    """Deterministic OCR stand-in: reads by crop height, records every call."""

    def __init__(self, cell_text: str = "2x", bag_text: str = "3", big_height: int = BIG_HEIGHT):
        self.cell_text = cell_text
        self.bag_text = bag_text
        self.big_height = big_height
        self.shapes: list = []

    def read_text(self, image_bgr: np.ndarray) -> str:
        self.shapes.append(tuple(image_bgr.shape))
        if image_bgr.shape[0] >= self.big_height:
            return self.bag_text
        return self.cell_text


def _blank_page() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8)


def _draw_cell(page: np.ndarray, x: int, y: int, w: int = 100, h: int = 70) -> None:
    cv2.rectangle(page, (x, y), (x + w, y + h), GRAY, thickness=-1)


def _page_with_cells(n: int) -> np.ndarray:
    """White page with ``n`` well-separated gray callout cells laid out in a grid."""
    page = _blank_page()
    per_row = 5
    for i in range(n):
        row, col = divmod(i, per_row)
        x = 40 + col * 150
        y = 40 + row * 120
        _draw_cell(page, x, y)
    return page


def _draw_bag_numeral(page: np.ndarray, x: int = 300, y: int = 400, w: int = 180, h: int = 260) -> None:
    cv2.rectangle(page, (x, y), (x + w, y + h), BLACK, thickness=-1)


def _draw_two_digit_bag_numeral(
    page: np.ndarray,
    x: int = 300,
    y: int = 400,
    w1: int = 70,
    gap: int = 15,
    w2: int = 70,
    h: int = 260,
) -> tuple:
    """Draw two adjacent tall black blocks standing in for a "12" numeral.

    Returns the expected merged bbox (x, y, w, h) covering both digits.
    """
    cv2.rectangle(page, (x, y), (x + w1, y + h), BLACK, thickness=-1)
    x2 = x + w1 + gap
    cv2.rectangle(page, (x2, y), (x2 + w2, y + h), BLACK, thickness=-1)
    # cv2.rectangle fills both corner pixels inclusive, so each block is
    # actually (w+1) x (h+1) px; the merged bbox spans both blocks' full extent.
    return (x, y, w1 + gap + w2 + 1, h + 1)


# --- callout detection --------------------------------------------------------

def test_detects_exact_number_of_cells():
    page = _page_with_cells(3)
    det = LocalDetector(ocr=FakeOCR())
    result = det.detect_page(page, page_index=4)

    assert isinstance(result, PageDetection)
    assert result.page_index == 4
    assert len(result.callouts) == 3
    assert result.is_parts_list is False


def test_crop_png_is_nonempty_and_decodes():
    page = _page_with_cells(2)
    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=0)

    for dc in result.callouts:
        assert isinstance(dc, DetectedCallout)
        assert isinstance(dc.crop_png, (bytes, bytearray))
        assert len(dc.crop_png) > 0
        decoded = cv2.imdecode(np.frombuffer(dc.crop_png, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.ndim == 3
        # bbox is (x, y, w, h) in page space.
        x, y, w, h = dc.bbox
        assert w > 0 and h > 0


def test_quantity_parsed_from_fake_2x():
    page = _page_with_cells(2)
    result = LocalDetector(ocr=FakeOCR(cell_text="2x")).detect_page(page, page_index=1)
    assert [dc.callout.quantity for dc in result.callouts] == [2, 2]


def test_unreadable_ocr_defaults_quantity_to_one():
    page = _page_with_cells(2)
    result = LocalDetector(ocr=FakeOCR(cell_text="")).detect_page(page, page_index=1)
    assert all(dc.callout.quantity == 1 for dc in result.callouts)
    assert len(result.callouts) == 2


def test_panel_detected_via_border_when_fill_matches_background():
    """Regression: on a page whose background is the same tone as a panel's
    own fill (e.g. 76307's pale-blue booklet), the plain fill-colour detector
    can't isolate the panel -- its interior and the page background merge
    into one giant region and no panel is ever found. A dark border stroke
    around the panel still lets it be found as an enclosed "hole"."""
    page = np.full((PAGE_H, PAGE_W, 3), 220, dtype=np.uint8)  # bg == panel fill tone
    x, y, w, h = 300, 300, 200, 100
    cv2.rectangle(page, (x, y), (x + w, y + h), BLACK, thickness=3)  # border only

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=9)
    assert len(result.callouts) == 1
    bx, by, _, _ = result.callouts[0].bbox
    assert 295 <= bx <= 305 and 295 <= by <= 305


def test_small_irregular_ink_loop_is_not_a_callout_cell():
    """A small closed dark loop elsewhere in the illustration (e.g. a gap
    between studs, an icon's ring) is also a "hole" in the ink mask, but is
    far less rectangular than a real panel -- filtered by contour extent
    (contour area / bbox area), not just size/aspect."""
    page = _blank_page()
    cv2.circle(page, (400, 400), 30, BLACK, thickness=4)  # ring -> circular hole

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=9)
    assert len(result.callouts) == 0


# --- projection + BOM classification -----------------------------------------

def test_to_page_extract_matches_and_flags_bom():
    cfg = DetectConfig()
    page = _page_with_cells(cfg.bom_cell_count)  # exactly the BOM threshold
    result = LocalDetector(config=cfg, ocr=FakeOCR()).detect_page(page, page_index=7)

    assert result.is_parts_list is True
    extract = result.to_page_extract()
    assert isinstance(extract, PageExtract)
    assert len(extract.callouts) == len(result.callouts) == cfg.bom_cell_count
    assert extract.is_parts_list is True
    assert extract.page_index == 7


def test_few_cells_is_not_bom():
    cfg = DetectConfig()
    page = _page_with_cells(cfg.bom_cell_count - 1)
    result = LocalDetector(config=cfg, ocr=FakeOCR()).detect_page(page, page_index=2)
    assert result.is_parts_list is False
    assert len(result.callouts) == cfg.bom_cell_count - 1


# --- bag marker ---------------------------------------------------------------

def test_bag_marker_read_from_big_numeral():
    page = _page_with_cells(2)
    _draw_bag_numeral(page)
    ocr = FakeOCR(bag_text="3")
    result = LocalDetector(ocr=ocr).detect_page(page, page_index=10)

    assert result.bag_marker == 3
    # The tall numeral block must not be mistaken for a gray callout cell.
    assert len(result.callouts) == 2


def test_no_bag_marker_on_page_without_numeral():
    page = _page_with_cells(3)
    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=5)
    assert result.bag_marker is None


def test_two_digit_bag_numeral_merges_into_one_candidate():
    """A two-digit numeral ("12") is two dark contours, but must merge into one bbox."""
    page = _blank_page()
    expected_bbox = _draw_two_digit_bag_numeral(page)

    det = LocalDetector(ocr=FakeOCR())
    gray = det._to_gray(page)
    candidates = det._find_bag_marker_candidates(gray, page)
    assert len(candidates) == 1
    assert candidates[0] == expected_bbox

    result = LocalDetector(ocr=FakeOCR(bag_text="12")).detect_page(page, page_index=6)
    assert result.bag_marker == 12
    assert result.bag_marker_bbox == expected_bbox
    assert result.bag_marker_candidate is False


def test_two_digit_numeral_with_no_ocr_reading_becomes_candidate():
    """No-Tesseract case: OCR reads "" for every crop; still flagged as a candidate."""
    page = _blank_page()
    expected_bbox = _draw_two_digit_bag_numeral(page)

    result = LocalDetector(ocr=FakeOCR(bag_text="")).detect_page(page, page_index=6)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is True
    assert result.bag_marker_bbox == expected_bbox


def test_dark_blob_inside_callout_cell_is_not_bag_marker_candidate():
    """A tall dark render silhouette fully inside a gray cell is not a bag numeral."""
    page = _blank_page()
    cell_x, cell_y, cell_w, cell_h = 300, 300, 200, 400
    cv2.rectangle(page, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), GRAY, thickness=-1)
    blob_x, blob_y, blob_w, blob_h = 350, 320, 60, 200
    cv2.rectangle(page, (blob_x, blob_y), (blob_x + blob_w, blob_y + blob_h), BLACK, thickness=-1)

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=8)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_large_illustration_outline_is_not_bag_marker_candidate():
    """Regression: a thick circular illustration outline (e.g. a "shake the bag
    out" panel) is tall and near-black like a numeral, but far too large in
    area/width to be one. Found on a real instruction page where the whole
    illustration was wrongly flagged as a bag-marker candidate."""
    page = _blank_page()
    cv2.circle(page, (PAGE_W // 2, PAGE_H // 2), 300, BLACK, thickness=15)

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=3)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_dark_saturated_part_render_is_not_bag_marker_candidate():
    """Regression: a dark but *coloured* part render (e.g. maroon Iron Man
    armor) can be just as dark in grayscale as a printed numeral, but real
    ink is low-saturation. Found on a real build-step page (76307 p.13) where
    a dark-red sub-assembly closeup was wrongly flagged as a bag marker."""
    page = _blank_page()
    dark_red_bgr = (20, 20, 120)  # low grayscale value, high HSV saturation
    cv2.rectangle(page, (300, 400), (480, 660), dark_red_bgr, thickness=-1)

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=12)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_thin_column_divider_is_not_bag_marker_candidate():
    """Regression: the hairline rule between two build steps on a two-column
    page spans nearly the full page height and is near-black, but is far too
    thin to be a numeral. Found on a real page (76307 p.11) where it was the
    only surviving bag-marker candidate once colour-render false positives on
    the same page were filtered out."""
    page = _blank_page()
    cv2.line(page, (PAGE_W // 2, 20), (PAGE_W // 2, PAGE_H - 20), BLACK, thickness=2)

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=10)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_hollow_icon_is_not_bag_marker_candidate():
    """Regression: a recurring line-art pictogram (a "rotate the model" icon)
    is tall, near-black, and correctly proportioned, but is a thin-lined
    hollow shape -- real ink digits are solid, thick strokes with far higher
    fill. Found recurring dozens of times through a real 56-page booklet."""
    page = _blank_page()
    cv2.rectangle(page, (300, 400), (420, 520), BLACK, thickness=6)  # hollow square outline

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=19)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_textured_pattern_is_not_bag_marker_candidate():
    """Regression: a busy, high-fill texture (a QR code, tight curly hair on a
    character illustration) can pass the fill-ratio gate but shatters into far
    more separate contours than a real digit ever does. Found on real
    back-matter pages (76307 p.16, p.54, p.56)."""
    page = _blank_page()
    x0, y0 = 300, 400
    cell, step, n = 13, 18, 7  # isolated squares, spaced so none touch (even diagonally)
    for row in range(n):
        for col in range(n):
            x, y = x0 + col * step, y0 + row * step
            cv2.rectangle(page, (x, y), (x + cell, y + cell), BLACK, thickness=-1)

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=15)
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_bag_marker_suppressed_on_parts_list_page():
    """Regression: a cover/contents page (real 76307 booklet page 1) has both
    lots of small graphic elements (logos/badges -- misread as callout cells,
    tripping is_parts_list) AND a dark product-render silhouette sized enough
    to otherwise pass the bag-marker gates. Such a dense page must never also
    report a bag-marker candidate, or ordinal numbering treats page 1 as
    "Bag 1" and every real bag shifts by one."""
    page = _page_with_cells(20)  # trips is_parts_list (default bom_cell_count=12)
    # A moderately large dark blob, sized like the real cover's character
    # render (comfortably inside the bag-marker area/width gates).
    cv2.rectangle(page, (300, 300), (500, 500), BLACK, thickness=-1)

    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=0)
    assert result.is_parts_list is True
    assert result.bag_marker is None
    assert result.bag_marker_candidate is False
    assert result.bag_marker_bbox is None


def test_assign_ordinal_bag_numbers_fills_candidates_in_page_order():
    detections = [
        PageDetection(page_index=0, bag_marker_candidate=True),
        PageDetection(page_index=1),
        PageDetection(page_index=2, bag_marker_candidate=True),
        PageDetection(page_index=3, bag_marker=5),
        PageDetection(page_index=4, bag_marker_candidate=True),
    ]
    warnings = assign_ordinal_bag_numbers(detections)
    assert [d.bag_marker for d in detections] == [1, None, 2, 5, 6]
    assert warnings == []


def test_assign_ordinal_bag_numbers_warns_on_non_monotonic_ocr_read():
    detections = [
        PageDetection(page_index=0, bag_marker=5),
        PageDetection(page_index=1, bag_marker=3),  # contradicts the running sequence
        PageDetection(page_index=2, bag_marker_candidate=True),
    ]
    warnings = assign_ordinal_bag_numbers(detections)
    # Left untouched -- bags.py's own monotonic filter deals with this, not us.
    assert detections[1].bag_marker == 3
    assert len(warnings) == 1
    # Ordinal assignment continues from the last *valid* running number (5), not 3.
    assert detections[2].bag_marker == 6


# --- robustness + convenience API --------------------------------------------

def test_blank_white_page_returns_empty_detection():
    page = _blank_page()
    result = LocalDetector(ocr=FakeOCR()).detect_page(page, page_index=9)
    assert result.page_index == 9
    assert result.callouts == []
    assert result.bag_marker is None
    assert result.is_parts_list is False


def test_accepts_encoded_png_bytes():
    page = _page_with_cells(2)
    ok, buf = cv2.imencode(".png", page)
    assert ok
    result = detect_page(buf.tobytes(), page_index=3, ocr=FakeOCR())
    assert len(result.callouts) == 2


def test_undecodable_bytes_return_empty_detection():
    result = detect_page(b"not-an-image", page_index=0, ocr=FakeOCR())
    assert isinstance(result, PageDetection)
    assert result.callouts == []


def test_default_ocr_is_tesseract_when_none_injected():
    det = LocalDetector()
    assert isinstance(det.ocr, TesseractOCR)
