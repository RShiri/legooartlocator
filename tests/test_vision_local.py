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
