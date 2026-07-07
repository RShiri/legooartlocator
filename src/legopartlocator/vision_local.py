"""Local (OpenCV) page detector: an offline, no-API alternative to ``vision.py``.

Given a rendered LEGO instruction-page image this locates the small
parts-callout cells (the light-gray rounded rectangles holding a part render
and an "Nx" label), the bag-start numeral, and reads each callout's quantity,
producing a :class:`detection.PageDetection` on the same fixed contract the
Claude vision pass uses.

The pipeline is deliberately simple and threshold-driven:

    threshold the light-gray callout background -> external contours ->
    area/aspect filter -> one cell per contour -> OCR each cell for "Nx".

OCR is injected (see :class:`OCR`) so the detector runs — and is tested —
without the Tesseract binary. All tunables live in :class:`DetectConfig`; the
defaults target the standard LEGO callout style and WILL need tuning against
real rendered pages (DPI, print variations, rounded-corner radius, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable

import cv2
import numpy as np

from .detection import DetectedCallout, PageDetection
from .models import PageCallout

# Accepted image inputs: raw encoded bytes (PNG/JPEG) or a decoded BGR ndarray.
ImageInput = Union[bytes, bytearray, memoryview, np.ndarray]

# Leading integer before an 'x'/'X' in a callout label, e.g. "2x", "12 X".
_QTY_RE = re.compile(r"(\d+)\s*[xX]")


@dataclass
class DetectConfig:
    """Tunable thresholds for :class:`LocalDetector`.

    These defaults are calibrated to the standard LEGO callout style (a light-gray
    rounded panel on a white page, with a small "Nx" label) and WILL need tuning
    against real rendered pages -- render DPI, anti-aliasing, and booklet print
    variations all shift the gray band, cell sizes, and glyph heights.

    Areas are expressed as a *fraction of the page area* so they scale with DPI;
    ``bag_marker_min_height_frac`` is a fraction of page height.
    """

    # Light-gray callout background band on a 0-255 gray image (white page is ~255).
    panel_gray_low: int = 200
    panel_gray_high: int = 240

    # Callout-cell size gate, as a fraction of total page area.
    min_cell_area: float = 0.0005
    max_cell_area: float = 0.2

    # Callout-cell shape gate (bbox width / height).
    min_aspect: float = 0.3
    max_aspect: float = 4.0

    # A page with at least this many callout cells is treated as a BOM/parts list.
    bom_cell_count: int = 12

    # A bag-start numeral is tall relative to the page; its glyph must reach at
    # least this fraction of the page height. ``bag_marker_dark_max`` is the upper
    # gray bound for "ink" pixels (the numeral is near-black).
    bag_marker_min_height_frac: float = 0.12
    bag_marker_dark_max: int = 90


@runtime_checkable
class OCR(Protocol):
    """Reads text from a BGR image crop. Injected so tests need no OCR binary."""

    def read_text(self, image_bgr: np.ndarray) -> str:
        ...


class TesseractOCR:
    """OCR backend using pytesseract.

    Requires the external Tesseract binary to be installed on the system; the
    ``pytesseract`` package alone is not enough. ``pytesseract`` is imported
    lazily so importing this module (and running the tests with a fake OCR)
    never touches it. Uses ``--psm 7`` (treat the image as a single text line),
    which suits the short single-line callout / bag labels.
    """

    def __init__(self, config: str = "--psm 7", lang: str = "eng"):
        self.config = config
        self.lang = lang

    def read_text(self, image_bgr: np.ndarray) -> str:
        try:
            import pytesseract  # lazy: only needed when this backend actually runs
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "TesseractOCR requires the 'pytesseract' package and the Tesseract "
                "binary. Install both, or inject a different OCR backend."
            ) from exc
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_bgr
        return pytesseract.image_to_string(rgb, lang=self.lang, config=self.config).strip()


def _find_contours(mask: np.ndarray) -> list:
    """External contours, tolerant of OpenCV's 2- vs 3-tuple return shapes."""
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found[0] if len(found) == 2 else found[1]


def _parse_quantity(text: str) -> int:
    """Parse the leading integer before an 'x'/'X'. Default 1 if unreadable/<1."""
    if not text:
        return 1
    match = _QTY_RE.search(text)
    if not match:
        return 1
    try:
        qty = int(match.group(1))
    except ValueError:
        return 1
    return qty if qty >= 1 else 1


def _parse_pure_int(text: str) -> Optional[int]:
    """Return the integer value if ``text`` is a bare integer, else ``None``."""
    if not text:
        return None
    token = text.strip()
    if token.isdigit():
        try:
            return int(token)
        except ValueError:
            return None
    return None


class LocalDetector:
    """Detects callout cells, quantities, and the bag numeral on one page image."""

    def __init__(self, config: DetectConfig = DetectConfig(), ocr: Optional[OCR] = None):
        self.config = config
        # Default to Tesseract; tests inject a fake so no binary is needed.
        self.ocr: OCR = ocr if ocr is not None else TesseractOCR()

    # -- public API -------------------------------------------------------

    def detect_page(self, image: ImageInput, page_index: int) -> PageDetection:
        """Detect one page. Never raises on a blank/all-white or undecodable page."""
        img = self._as_bgr(image)
        if img is None or img.size == 0:
            return PageDetection(page_index=page_index)
        height, width = img.shape[:2]
        if height == 0 or width == 0:
            return PageDetection(page_index=page_index)

        gray = self._to_gray(img)
        callouts = self._detect_callouts(gray, img)
        is_parts_list = len(callouts) >= self.config.bom_cell_count
        bag_marker = self._detect_bag_marker(gray, img)

        return PageDetection(
            page_index=page_index,
            bag_marker=bag_marker,
            is_parts_list=is_parts_list,
            printed_page_number=None,
            callouts=callouts,
        )

    # -- internals --------------------------------------------------------

    @staticmethod
    def _as_bgr(image: ImageInput) -> Optional[np.ndarray]:
        """Return a BGR/gray ndarray, decoding encoded bytes; ``None`` if undecodable."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (bytes, bytearray, memoryview)):
            buf = np.frombuffer(bytes(image), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # None on failure
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _cell_boxes(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Bounding boxes of the light-gray callout panels, in reading order."""
        cfg = self.config
        height, width = gray.shape[:2]
        page_area = float(height * width)
        min_area = cfg.min_cell_area * page_area
        max_area = cfg.max_cell_area * page_area

        mask = cv2.inRange(gray, cfg.panel_gray_low, cfg.panel_gray_high)
        boxes: List[Tuple[int, int, int, int]] = []
        for contour in _find_contours(mask):
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            aspect = w / h
            if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
                continue
            boxes.append((x, y, w, h))
        # Deterministic reading order: top-to-bottom, then left-to-right.
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    def _detect_callouts(self, gray: np.ndarray, img: np.ndarray) -> List[DetectedCallout]:
        callouts: List[DetectedCallout] = []
        for (x, y, w, h) in self._cell_boxes(gray):
            crop = img[y : y + h, x : x + w]
            ok, buf = cv2.imencode(".png", crop)
            crop_png = buf.tobytes() if ok else b""
            quantity = _parse_quantity(self.ocr.read_text(crop))
            callout = PageCallout(
                quantity=quantity, color=None, shape_desc=None, printed_part_id=None
            )
            callouts.append(
                DetectedCallout(callout=callout, crop_png=crop_png, bbox=(x, y, w, h))
            )
        return callouts

    def _detect_bag_marker(self, gray: np.ndarray, img: np.ndarray) -> Optional[int]:
        """Return the bag number from a tall standalone numeral, else ``None``.

        Finds near-black ink blobs whose height reaches
        ``bag_marker_min_height_frac`` of the page, OCRs each, and returns the
        tallest one that reads as a bare integer.
        """
        cfg = self.config
        height = gray.shape[0]
        min_h = cfg.bag_marker_min_height_frac * height

        mask = cv2.inRange(gray, 0, cfg.bag_marker_dark_max)
        best_height = -1
        best_value: Optional[int] = None
        for contour in _find_contours(mask):
            x, y, w, h = cv2.boundingRect(contour)
            if h < min_h:
                continue
            crop = img[y : y + h, x : x + w]
            value = _parse_pure_int(self.ocr.read_text(crop))
            if value is not None and h > best_height:
                best_height = h
                best_value = value
        return best_value


def detect_page(
    image: ImageInput,
    page_index: int,
    config: Optional[DetectConfig] = None,
    ocr: Optional[OCR] = None,
) -> PageDetection:
    """Convenience wrapper: detect one page with a one-off :class:`LocalDetector`."""
    detector = LocalDetector(config=config or DetectConfig(), ocr=ocr)
    return detector.detect_page(image, page_index)
