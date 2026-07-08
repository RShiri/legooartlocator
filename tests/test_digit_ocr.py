"""Tests for the dependency-free 'Nx' quantity digit reader (offline, no Tesseract).

The reader is calibrated against real 76307 callout crops; these tests render
synthetic callout cells (with a *different* Hershey font from the reader's own
templates) to guard the localization + parsing + safety logic against
regressions.
"""

import cv2
import numpy as np

from legopartlocator.vision_local import DigitOCR, _parse_quantity


def _cell(label, w=380, h=140, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.1, thick=2):
    """A synthetic callout cell: light background, a dark 'part' blob up top, and
    the quantity label in dark text centered near the bottom baseline."""
    img = np.full((h, w, 3), (230, 220, 205), np.uint8)
    cv2.rectangle(img, (w // 2 - 60, 15), (w // 2 + 60, h // 2 + 10), (70, 70, 70), -1)
    if label:
        (tw, _th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.putText(img, label, ((w - tw) // 2, h - 12), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
    return img


def test_reads_each_single_digit_quantity():
    ocr = DigitOCR()
    for n in range(1, 10):
        assert _parse_quantity(ocr.read_text(_cell(f"{n}x"))) == n


def test_reads_multi_digit_quantities():
    ocr = DigitOCR()
    assert _parse_quantity(ocr.read_text(_cell("12x"))) == 12
    assert _parse_quantity(ocr.read_text(_cell("10x"))) == 10


def test_no_label_cell_returns_empty_and_defaults_to_one():
    ocr = DigitOCR()
    blank = np.full((140, 380, 3), (230, 220, 205), np.uint8)
    cv2.rectangle(blank, (150, 20), (250, 95), (70, 70, 70), -1)  # a part, no label
    assert ocr.read_text(blank) == ""
    assert _parse_quantity(ocr.read_text(blank)) == 1


def test_large_bag_numeral_is_not_read_as_a_quantity():
    """A big single numeral filling the cell (a bag-start marker) must NOT be read
    as an 'Nx' quantity -- it should return "" so ordinal bag numbering is used."""
    ocr = DigitOCR()
    img = np.full((160, 160, 3), 255, np.uint8)
    cv2.putText(img, "3", (35, 130), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 0, 0), 8, cv2.LINE_AA)
    assert ocr.read_text(img) == ""


def test_is_deterministic():
    ocr = DigitOCR()
    img = _cell("4x")
    assert ocr.read_text(img) == ocr.read_text(img)


def test_degenerate_inputs_do_not_raise():
    ocr = DigitOCR()
    assert ocr.read_text(np.zeros((5, 5, 3), np.uint8)) == ""   # too small
    assert ocr.read_text(np.zeros((0, 0), np.uint8)) == ""       # empty
