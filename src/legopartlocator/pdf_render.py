"""PDF rendering and light text extraction via PyMuPDF (``fitz``).

Responsibilities:
  * render each page to PNG bytes at a chosen DPI (for the vision pass),
  * pull whatever text layer exists (cheap signal; many LEGO BI PDFs are
    image-only, so callers must not rely on text being present),
  * best-effort detect the set number from the cover pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

try:  # PyMuPDF is a hard runtime dependency, but keep import errors friendly.
    import fitz  # type: ignore
except ImportError as exc:  # pragma: no cover - env-dependent
    raise ImportError(
        "PyMuPDF is required for PDF rendering. Install with `pip install pymupdf`."
    ) from exc


# LEGO set numbers are typically 4-7 digits. The 6-8 digit numbers on LEGO
# building-instruction PDFs (e.g. 6559641) are the *booklet* asset id, not the
# set number, so we prefer shorter tokens and validate against context words.
_SET_NUM_RE = re.compile(r"\b(\d{4,7})\b")


@dataclass
class RenderedPage:
    """One rendered page: its 0-based index plus PNG bytes."""

    page_index: int
    png_bytes: bytes
    text: str


def parse_page_range(spec: Optional[str], num_pages: int) -> List[int]:
    """Parse a 1-based, inclusive range spec like "1-10,15,20-22" into 0-based indices.

    ``None`` or empty returns all pages. Out-of-bounds values are clamped/ignored.
    """
    if not spec:
        return list(range(num_pages))
    indices: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(chunk)
        for one_based in range(lo, hi + 1):
            zero_based = one_based - 1
            if 0 <= zero_based < num_pages:
                indices.append(zero_based)
    # Preserve order, drop duplicates.
    seen = set()
    ordered: List[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def open_pdf(pdf_path: str | Path) -> "fitz.Document":
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return fitz.open(path)


def page_count(pdf_path: str | Path) -> int:
    with open_pdf(pdf_path) as doc:
        return doc.page_count


def render_pages(
    pdf_path: str | Path,
    dpi: int = 150,
    page_indices: Optional[Iterable[int]] = None,
) -> Iterator[RenderedPage]:
    """Yield RenderedPage for each requested page (all pages if ``page_indices`` is None).

    Streams one page at a time to keep memory bounded on large manuals.
    """
    with open_pdf(pdf_path) as doc:
        wanted = list(page_indices) if page_indices is not None else list(range(doc.page_count))
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for idx in wanted:
            if idx < 0 or idx >= doc.page_count:
                continue
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png = pix.tobytes("png")
            text = page.get_text("text") or ""
            yield RenderedPage(page_index=idx, png_bytes=png, text=text)


class PageRenderer:
    """Keeps a PDF open so individual pages can be rendered on demand.

    Used by the two-pass flow to render only the pages that need the detailed
    (high-DPI) pass, without reopening the document per page.
    """

    def __init__(self, pdf_path: str | Path, dpi: int = 200):
        self.doc = open_pdf(pdf_path)
        zoom = dpi / 72.0
        self._matrix = fitz.Matrix(zoom, zoom)

    def render(self, page_index: int) -> RenderedPage:
        page = self.doc.load_page(page_index)
        pix = page.get_pixmap(matrix=self._matrix, alpha=False)
        return RenderedPage(
            page_index=page_index,
            png_bytes=pix.tobytes("png"),
            text=page.get_text("text") or "",
        )

    def close(self) -> None:
        self.doc.close()

    def __enter__(self) -> "PageRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def detect_set_number(pdf_path: str | Path, scan_pages: int = 4) -> Optional[str]:
    """Best-effort set-number detection from the first/last pages' text layer.

    Returns the most plausible candidate or None. This is only a convenience;
    the user can always pass --set explicitly.
    """
    candidates: List[Tuple[str, int]] = []  # (number, score)
    with open_pdf(pdf_path) as doc:
        n = doc.page_count
        page_idxs = list(range(min(scan_pages, n))) + list(range(max(0, n - 2), n))
        for idx in dict.fromkeys(page_idxs):  # unique, ordered
            text = doc.load_page(idx).get_text("text") or ""
            for m in _SET_NUM_RE.finditer(text):
                num = m.group(1)
                score = 0
                # Shorter numbers are more likely to be real set numbers.
                score += max(0, 8 - len(num))
                # Context words that often sit next to a set number.
                window = text[max(0, m.start() - 30) : m.end() + 30].lower()
                if any(w in window for w in ("age", "ages", "+", "pcs", "pieces")):
                    score += 3
                candidates.append((num, score))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[0][0]
