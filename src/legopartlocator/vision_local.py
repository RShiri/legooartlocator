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
from typing import List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import cv2
import numpy as np

from .detection import DetectedCallout, PageDetection
from .models import PageCallout

# Accepted image inputs: raw encoded bytes (PNG/JPEG) or a decoded BGR ndarray.
ImageInput = Union[bytes, bytearray, memoryview, np.ndarray]

# Leading integer before an 'x'/'X' in a callout label, e.g. "2x", "12 X".
_QTY_RE = re.compile(r"(\d+)\s*[xX]")

# -- bag-marker glyph merging / rejection tunables (fixed, not per-config) ----
# Two glyph bboxes merge into one multi-digit numeral candidate when the
# horizontal gap between them is small relative to their height, their
# heights are close (same font size), and they overlap vertically (same row).
_MERGE_GAP_FRAC = 0.6          # max horizontal gap, as a fraction of glyph height
_MERGE_HEIGHT_TOL_FRAC = 0.4   # max relative height difference
_MERGE_MIN_VOVERLAP_FRAC = 0.5  # min vertical overlap, as a fraction of the shorter glyph
# A candidate whose bbox mostly sits inside a callout cell (dark part-render ink,
# not a bag numeral) is rejected once its overlap with any cell exceeds this
# fraction of the candidate's own area.
_CANDIDATE_CELL_OVERLAP_MAX = 0.3


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

    # Alternate panel detector: find the panel by its dark border/outline
    # stroke rather than its fill colour. On a booklet whose page background
    # is itself pale/tinted (not plain white), the panel's fill can sit in the
    # same grayscale band as the surrounding page -- they merge into one giant
    # region and no panel is ever isolated. A real panel is still a rounded
    # rectangle enclosed by a dark stroke though, so it shows up as a "hole"
    # in a dark-ink mask regardless of what shade its interior/background are.
    # Found on a real booklet (76307) with a pale-blue page background where
    # the plain fill-colour approach detected zero real panels across the
    # whole book. Both detectors run and their boxes are merged/deduped, so
    # plain-white-background booklets (where the fill-colour approach already
    # works) are unaffected.
    cell_border_dark_max: int = 120

    # A genuine panel is a (rounded) rectangle, so its contour fills almost
    # all of its own bounding box. Small enclosed loops elsewhere in the
    # illustration (gaps between studs, eyes, any closed line-art shape) also
    # show up as "holes" in the dark-ink mask but are irregular, filling far
    # less of their bbox. Measured ~0.98 for real panels vs. ~0.53-0.55 for
    # illustration noise on a real page.
    cell_border_min_extent: float = 0.85

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
    # gray bound for "ink" pixels (the numeral is near-black). Tried lowering
    # this globally to catch a small circled corner badge on a bigger booklet
    # (76269) but that reintroduced a real regression: 76307's per-step
    # counter numerals (plain solid digits, ~0.06 of page height) then also
    # qualified, since a solid digit is a solid digit regardless of size --
    # nothing else distinguishes "step counter" from "bag marker" for that
    # style. Kept strict; see ``bag_marker_ring_min_height_frac`` for the
    # deliberately more permissive floor that applies only to circled digits,
    # where the ring itself (never used for a step counter) is the signal
    # that licenses a smaller size.
    bag_marker_min_height_frac: float = 0.12
    bag_marker_dark_max: int = 90

    # Upper bounds so large dark artwork (illustration outlines, shaded panels)
    # isn't mistaken for a numeral: a real bag-start glyph is a small graphic
    # element, not a large fraction of the page's area or width. Found by
    # inspecting a real instruction page where a "shake the bag" illustration's
    # thick circle outline (tall, near-black, but ~half the page) was wrongly
    # flagged as a bag-marker candidate.
    bag_marker_max_area_frac: float = 0.10
    bag_marker_max_width_frac: float = 0.35

    # A lower bound on width/height so a page-layout divider (a hairline rule
    # between two build steps on a two-column page -- full page height, only a
    # couple of pixels wide) isn't mistaken for a numeral. Found on a real
    # instruction page where the column-divider line was the only surviving
    # candidate once a colour-render false positive on the same page was fixed.
    bag_marker_min_aspect: float = 0.15

    # A bold printed digit is a solid, thick-stroked glyph: a real font digit
    # fills roughly 45-60% of its own bounding box with ink. Line-art icons and
    # illustrated part renders are mostly hollow/outline and fill well under
    # that -- measured ~0.19 (a recurring "rotate" pictogram) to ~0.36 (a
    # chunky studded-brick closeup) on real false positives from the same
    # instruction booklet, against ~0.46-0.62 for rendered bold digits 0-9.
    bag_marker_min_fill_frac: float = 0.42

    # Some booklets print the bag/booklet marker as a digit inside a circle
    # (a ring), not a solid filled glyph -- measured fill ~0.32 on a real
    # circled "2" (76269), below ``bag_marker_min_fill_frac``. A candidate in
    # this lower band is only accepted if it's also very simple (a ring plus
    # a digit is 2 external contours), since low-fill *and* multi-part shapes
    # are exactly the illustration noise the fill gate exists to reject (e.g.
    # a studded-brick closeup: fill ~0.36, 3 external contours).
    bag_marker_ring_min_fill_frac: float = 0.28
    bag_marker_ring_max_components: int = 2
    # A ring is a separate ink blob from the digit it encloses, so a real
    # circled digit is *at least* 2 disconnected components -- a lone solid
    # digit (no ring) is always exactly 1. Without this floor, an ordinary
    # small step-counter digit (high fill, 1 component) also satisfied the
    # ring branch's fill/max-components checks and was wrongly accepted.
    bag_marker_ring_min_components: int = 2
    # A ring's own bbox spans nearly the whole merged candidate (the digit is
    # nested inside it), so the largest component covers most of the merged
    # area -- measured ~0.87 on a real ring. A two-digit number like "20" also
    # has exactly 2 disconnected blobs (satisfying the component-count check
    # above) but neither digit dominates -- side-by-side, ~0.44-0.46 each on
    # a real merged "20". This is what actually tells the two cases apart.
    bag_marker_ring_min_dominant_frac: float = 0.65

    # The ring branch also gets a lower height floor than
    # ``bag_marker_min_height_frac`` -- a circled digit is never how a
    # per-step counter is printed, so the ring shape itself licenses a
    # smaller marker (measured ~0.07 of page height on 76269's real corner
    # badge) without reopening the door to small solid-digit step counters
    # (76307), which only ever pass the *solid* branch.
    bag_marker_ring_min_height_frac: float = 0.05

    # A real digit (or a merged multi-digit run) is one or two simple strokes,
    # so its near-black mask breaks into only a couple of contours. Busy
    # textures that otherwise pass every other gate -- a QR code, tight curly
    # hair on a character illustration, a studded part close-up -- shatter
    # into dozens of tiny contours instead. Measured 2-3 contours for real
    # rendered digits vs. 28-102 for those real false positives.
    bag_marker_max_components: int = 8

    # A real bag-start numeral is printed in black ink (low HSV saturation)
    # even though it renders "near-black" in grayscale. Dark *coloured* plastic
    # (e.g. a maroon/navy part render) can be just as dark in grayscale but is
    # far more saturated. Found on a real instruction page where a dark-red
    # sub-assembly render (Iron Man armor) was wrongly flagged as a bag-marker
    # candidate on an ordinary build-step page -- mean HSV saturation of its
    # dark pixels was ~157 vs. ~30 for a real printed numeral on the same page.
    bag_marker_max_saturation: int = 90


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


def _should_merge_glyphs(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    """True if two dark-glyph bboxes look like adjacent digits of one numeral.

    Requires: a small horizontal gap relative to glyph height (so digits of the
    same numeral, not unrelated ink elsewhere on the page), similar heights
    (same font size), and substantial vertical overlap (same text row).
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_right, b_right = ax + aw, bx + bw
    gap = max(bx - a_right, ax - b_right, 0)
    max_h = max(ah, bh)
    if max_h <= 0 or gap >= _MERGE_GAP_FRAC * max_h:
        return False

    min_h, hi_h = min(ah, bh), max(ah, bh)
    if hi_h <= 0 or (hi_h - min_h) / hi_h > _MERGE_HEIGHT_TOL_FRAC:
        return False

    top, bottom = max(ay, by), min(ay + ah, by + bh)
    overlap = max(bottom - top, 0)
    if overlap < _MERGE_MIN_VOVERLAP_FRAC * min_h:
        return False

    return True


def _bbox_overlap_fraction(
    candidate: Tuple[int, int, int, int], other: Tuple[int, int, int, int]
) -> float:
    """Fraction of ``candidate``'s area covered by its intersection with ``other``."""
    cx, cy, cw, ch = candidate
    area = cw * ch
    if area <= 0:
        return 0.0
    ox, oy, ow, oh = other
    ix1, iy1 = max(cx, ox), max(cy, oy)
    ix2, iy2 = min(cx + cw, ox + ow), min(cy + ch, oy + oh)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    return inter / area


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
        cell_boxes = self._cell_boxes(gray)
        callouts = self._detect_callouts(img, cell_boxes)
        is_parts_list = len(callouts) >= self.config.bom_cell_count

        # A dense contents/BOM/cover page used to have bag-marker detection
        # skipped outright here, since a large dark product render could pass
        # the (then-only) size gates and get mistaken for "bag N" -- corrupting
        # ordinal numbering, since such a page is usually page 1. That blunt
        # rule turned out to also suppress *real* bag/booklet markers on
        # bigger sets, where the cover is itself busy enough to trip
        # ``is_parts_list`` on illustration noise (e.g. a building render's
        # window panes) while still carrying a real corner marker. The gates
        # added since (saturation, aspect, fill, component-count) reject the
        # original dark-product-render case directly -- confirmed against the
        # real page that motivated the blunt rule -- so it's no longer needed.
        bag_marker, bag_marker_candidate, bag_marker_bbox = self._detect_bag_marker(
            gray, img, cell_boxes
        )

        return PageDetection(
            page_index=page_index,
            bag_marker=bag_marker,
            is_parts_list=is_parts_list,
            printed_page_number=None,
            callouts=callouts,
            bag_marker_candidate=bag_marker_candidate,
            bag_marker_bbox=bag_marker_bbox,
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

    def _size_and_shape_gate(
        self, boxes: List[Tuple[int, int, int, int]], page_area: float
    ) -> List[Tuple[int, int, int, int]]:
        cfg = self.config
        min_area = cfg.min_cell_area * page_area
        max_area = cfg.max_cell_area * page_area
        kept = []
        for x, y, w, h in boxes:
            if h == 0:
                continue
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            aspect = w / h
            if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
                continue
            kept.append((x, y, w, h))
        return kept

    def _cell_boxes_by_fill(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Panels found by their light-gray fill colour (the classic style:
        a plain white page with a distinctly gray panel)."""
        cfg = self.config
        mask = cv2.inRange(gray, cfg.panel_gray_low, cfg.panel_gray_high)
        boxes = [cv2.boundingRect(c) for c in _find_contours(mask)]
        return self._size_and_shape_gate(boxes, float(gray.shape[0] * gray.shape[1]))

    def _cell_boxes_by_border(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Panels found as a "hole" enclosed by a dark border stroke.

        Robust to a page background that shares the panel's fill tone (fill
        colour alone can't separate the two then); relies only on the border
        being darker than the panel/background, which holds regardless of
        what shade either of them is.
        """
        cfg = self.config
        ink_mask = cv2.inRange(gray, 0, cfg.cell_border_dark_max)
        contours, hierarchy = cv2.findContours(ink_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        boxes = []
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # only holes (enclosed by a dark parent)
                continue
            x, y, w, h = cv2.boundingRect(c)
            bbox_area = w * h
            extent = (cv2.contourArea(c) / bbox_area) if bbox_area > 0 else 0.0
            if extent < cfg.cell_border_min_extent:
                continue
            boxes.append((x, y, w, h))
        return self._size_and_shape_gate(boxes, float(gray.shape[0] * gray.shape[1]))

    def _cell_boxes(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Bounding boxes of callout panels, in reading order.

        Merges two independent detectors (fill-colour and border-enclosure;
        see each for why both are needed) and dedupes boxes that clearly
        refer to the same panel.
        """
        boxes = self._cell_boxes_by_fill(gray) + self._cell_boxes_by_border(gray)
        deduped: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            if not any(_bbox_overlap_fraction(box, other) > 0.6 for other in deduped):
                deduped.append(box)
        # Deterministic reading order: top-to-bottom, then left-to-right.
        deduped.sort(key=lambda b: (b[1], b[0]))
        return deduped

    def _detect_callouts(
        self, img: np.ndarray, cell_boxes: List[Tuple[int, int, int, int]]
    ) -> List[DetectedCallout]:
        callouts: List[DetectedCallout] = []
        for (x, y, w, h) in cell_boxes:
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

    def _find_bag_marker_candidates(
        self, gray: np.ndarray, img: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Find candidate bag-numeral bboxes, merging adjacent multi-digit glyphs.

        A bag-start numeral is a tall, near-black glyph (or run of glyphs, for
        numbers >= 10) standing alone on the page. This finds every near-black
        connected component whose bbox height reaches
        ``bag_marker_min_height_frac`` of the page height, then merges
        components that are horizontally adjacent, similarly tall, and
        vertically overlapping (see :func:`_should_merge_glyphs`) so a
        multi-digit number like "12" -- which OCRs and contours as two separate
        components, "1" and "2" -- collapses into a single bbox before OCR is
        ever invoked. Returns merged bboxes sorted by area, largest first.
        """
        cfg = self.config
        height, width = gray.shape[:2]
        # The more permissive of the two height floors, so a ring candidate
        # isn't discarded before it even reaches the branch decision below
        # (which re-applies the stricter, branch-specific floor).
        min_h = min(cfg.bag_marker_min_height_frac, cfg.bag_marker_ring_min_height_frac) * height

        mask = cv2.inRange(gray, 0, cfg.bag_marker_dark_max)
        boxes: List[Tuple[int, int, int, int]] = []
        for contour in _find_contours(mask):
            x, y, w, h = cv2.boundingRect(contour)
            if h < min_h:
                continue
            boxes.append((x, y, w, h))
        if not boxes:
            return []

        n = len(boxes)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                if _should_merge_glyphs(boxes[i], boxes[j]):
                    union(i, j)

        groups: dict = {}
        for i, box in enumerate(boxes):
            groups.setdefault(find(i), []).append(box)

        merged: List[Tuple[int, int, int, int]] = []
        for group in groups.values():
            x0 = min(b[0] for b in group)
            y0 = min(b[1] for b in group)
            x1 = max(b[0] + b[2] for b in group)
            y1 = max(b[1] + b[3] for b in group)
            merged.append((x0, y0, x1 - x0, y1 - y0))

        # Reject shapes too large/wide to be a numeral: illustration outlines and
        # shaded panels are also tall and near-black, but a real bag-start glyph
        # is a small graphic element, not a large fraction of the page.
        page_area = float(height * width)
        max_area = cfg.bag_marker_max_area_frac * page_area
        max_w = cfg.bag_marker_max_width_frac * width
        merged = [
            b
            for b in merged
            if (b[2] * b[3]) <= max_area
            and b[2] <= max_w
            and (b[2] / b[3]) >= cfg.bag_marker_min_aspect
        ]

        # Reject dark but saturated (coloured) shapes: a real printed numeral is
        # black ink (low saturation) even where a dark plastic part render is
        # just as dark in grayscale. Checked only over the pixels that actually
        # tripped the near-black mask, not the whole bbox (which may include
        # lighter background/anti-aliasing).
        saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        kept: List[Tuple[int, int, int, int]] = []
        for b in merged:
            x, y, w, h = b
            region_mask = mask[y : y + h, x : x + w]
            if region_mask.size == 0:
                continue
            fill_frac = float((region_mask > 0).mean())
            n_components = len(_find_contours(region_mask))
            is_solid_digit = (
                fill_frac >= cfg.bag_marker_min_fill_frac
                and h >= cfg.bag_marker_min_height_frac * height
            )
            is_ring_digit = False
            if (
                fill_frac >= cfg.bag_marker_ring_min_fill_frac
                and cfg.bag_marker_ring_min_components
                <= n_components
                <= cfg.bag_marker_ring_max_components
                and h >= cfg.bag_marker_ring_min_height_frac * height
            ):
                component_boxes = [cv2.boundingRect(c) for c in _find_contours(region_mask)]
                dominant_frac = max((cw * ch) / (w * h) for _, _, cw, ch in component_boxes)
                is_ring_digit = dominant_frac >= cfg.bag_marker_ring_min_dominant_frac
            if not (is_solid_digit or is_ring_digit):
                continue
            region_sat = saturation[y : y + h, x : x + w][region_mask > 0]
            if region_sat.size and region_sat.mean() > cfg.bag_marker_max_saturation:
                continue
            if n_components > cfg.bag_marker_max_components:
                continue
            kept.append(b)

        kept.sort(key=lambda b: b[2] * b[3], reverse=True)
        return kept

    def _detect_bag_marker(
        self,
        gray: np.ndarray,
        img: np.ndarray,
        cell_boxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[int], bool, Optional[Tuple[int, int, int, int]]]:
        """Return ``(bag_marker, bag_marker_candidate, bag_marker_bbox)``.

        Gets merged candidate bboxes from :meth:`_find_bag_marker_candidates`
        (multi-digit numerals already collapsed to one bbox each), drops any
        that mostly overlap a detected callout cell -- dark ink rendered inside
        a gray parts-callout panel is not a bag numeral -- then OCRs the
        remaining candidates, largest first, and returns the first bare-integer
        read.

        If no candidate OCRs as a bare integer (this is the expected outcome
        with no Tesseract binary installed: the OCR backend returns ``""`` for
        every crop), ``bag_marker`` stays ``None`` but the single largest
        plausible candidate is still reported via ``bag_marker_candidate`` and
        ``bag_marker_bbox`` so a caller can assign the number structurally
        (see :func:`assign_ordinal_bag_numbers`) instead of losing it.
        """
        candidates = self._find_bag_marker_candidates(gray, img)
        plausible = [
            c
            for c in candidates
            if not any(
                _bbox_overlap_fraction(c, cell) > _CANDIDATE_CELL_OVERLAP_MAX
                for cell in cell_boxes
            )
        ]
        if not plausible:
            return None, False, None

        for bbox in plausible:
            x, y, w, h = bbox
            crop = img[y : y + h, x : x + w]
            value = _parse_pure_int(self.ocr.read_text(crop))
            if value is not None:
                return value, False, bbox

        return None, True, plausible[0]


def detect_page(
    image: ImageInput,
    page_index: int,
    config: Optional[DetectConfig] = None,
    ocr: Optional[OCR] = None,
) -> PageDetection:
    """Convenience wrapper: detect one page with a one-off :class:`LocalDetector`."""
    detector = LocalDetector(config=config or DetectConfig(), ocr=ocr)
    return detector.detect_page(image, page_index)


def assign_ordinal_bag_numbers(detections: Sequence[PageDetection]) -> List[str]:
    """Fill in unread bag numbers structurally, assuming bags are numbered 1..N.

    Real LEGO instruction booklets start each bag's build with a page showing
    that bag's number, and bags are always numbered 1, 2, 3, ... N in page
    order -- there are no gaps and no re-ordering. ``bags.py`` downstream only
    needs the sequence of markers to be monotonically increasing, not that any
    individual number was actually read off the page. That means a page which
    the local detector recognises structurally as a bag-start page (a tall,
    lone numeral-shaped glyph: ``bag_marker_candidate=True``) but could not OCR
    (no Tesseract binary, or an unreadable glyph) can safely be assigned
    "one more than the last known bag number", without reading any digits.

    This walks ``detections`` in page order (the order given -- callers should
    pass them sorted by ``page_index``) mutating each ``PageDetection`` in
    place, tracking ``last_known`` (the last bag number established so far,
    starting at 0):

    - If ``bag_marker`` was already read via OCR: if it is strictly greater
      than ``last_known`` it becomes the new ``last_known``. If it is not
      (OCR misread, or a genuinely out-of-order page), it is left completely
      untouched -- fixing it up is out of scope here, ``bags.py``'s existing
      monotonic filter already handles a bad reading -- but a warning is
      recorded so the discrepancy is visible.
    - Else if ``bag_marker_candidate`` is ``True`` (a plausible bag-start page
      with no readable number): ``bag_marker`` is set to ``last_known + 1``,
      which also becomes the new ``last_known``.
    - Otherwise (no marker, no candidate) the detection is left untouched.

    Returns a list of human-readable warning strings, one per OCR-read marker
    that did not extend the running sequence. Does not raise or drop pages.
    """
    warnings: List[str] = []
    last_known = 0
    for det in detections:
        if det.bag_marker is not None:
            if det.bag_marker > last_known:
                last_known = det.bag_marker
            else:
                warnings.append(
                    f"page {det.page_index}: OCR-read bag_marker={det.bag_marker} does not "
                    f"exceed the running sequence ({last_known}); left as-is for bags.py's "
                    "monotonic filter to handle"
                )
        elif det.bag_marker_candidate:
            last_known += 1
            det.bag_marker = last_known
    return warnings
