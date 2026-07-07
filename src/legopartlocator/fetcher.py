"""Auto-download official building-instruction PDFs from lego.com by set number.

The public page
    https://www.lego.com/{locale}/service/building-instructions/{set}
lists a set's downloadable booklets; their PDF URLs live on the LEGO CDN and are
embedded in the page's JSON (with escaped slashes). We unescape, scrape those
URLs, and download the PDFs so ``lpl scan --set NNNN`` needs no manual download.

Network transports are injected, so the URL parser is unit-tested offline. The
live fetch runs on the user's machine (lego.com is reachable there).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

DEFAULT_LOCALE = "en-gb"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# CDN PDF links embedded in the page, e.g.
#   https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6559641.pdf
_PDF_RE_SPECIFIC = re.compile(
    r"https://[\w.-]+/cdn/product-assets/product\.bi\.[\w.]+/\d+\.pdf"
)
# Fallback: any lego.com URL ending in .pdf.
_PDF_RE_FALLBACK = re.compile(r"https://[\w.-]*lego\.com/\S+?\.pdf")


class FetchError(RuntimeError):
    pass


def page_url(set_num: str, locale: str = DEFAULT_LOCALE) -> str:
    """The building-instructions page URL for a set (strips any -1 inventory suffix)."""
    bare = str(set_num).split("-")[0].strip()
    return f"https://www.lego.com/{locale}/service/building-instructions/{bare}"


def find_pdf_urls(html: str) -> List[str]:
    """Extract CDN PDF URLs from page HTML/JSON, de-duplicated in first-seen order.

    JSON embedded in the page escapes slashes (``https:\\/\\/``), so we unescape
    before matching. The specific CDN pattern is preferred; a generic lego.com
    ``.pdf`` fallback is used only if the specific one finds nothing.
    """
    text = (html or "").replace("\\/", "/")
    for regex in (_PDF_RE_SPECIFIC, _PDF_RE_FALLBACK):
        urls: List[str] = []
        seen = set()
        for match in regex.finditer(text):
            url = match.group(0)
            if url not in seen:
                seen.add(url)
                urls.append(url)
        if urls:
            return urls
    return []


def _default_get_text(url: str) -> str:  # pragma: no cover - network
    import httpx

    resp = httpx.get(
        url, headers={"User-Agent": _UA, "Accept": "text/html"}, timeout=30.0, follow_redirects=True
    )
    resp.raise_for_status()
    return resp.text


def _default_get_bytes(url: str) -> bytes:  # pragma: no cover - network
    import httpx

    resp = httpx.get(url, headers={"User-Agent": _UA}, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


class InstructionFetcher:
    """Fetches a set's instruction page and downloads its PDF booklets."""

    def __init__(
        self,
        get_text: Optional[Callable[[str], str]] = None,
        get_bytes: Optional[Callable[[str], bytes]] = None,
    ):
        self.get_text = get_text or _default_get_text
        self.get_bytes = get_bytes or _default_get_bytes

    def list_pdf_urls(self, set_num: str, locale: str = DEFAULT_LOCALE) -> List[str]:
        return find_pdf_urls(self.get_text(page_url(set_num, locale)))

    def fetch(
        self,
        set_num: str,
        dest_dir: str | Path = ".",
        locale: str = DEFAULT_LOCALE,
        overwrite: bool = False,
    ) -> List[Path]:
        """Download all booklet PDFs for a set; returns their local paths.

        Files are named by their CDN basename and skipped if already present
        (unless ``overwrite``). Raises FetchError if the page lists no PDFs.
        """
        urls = self.list_pdf_urls(set_num, locale)
        if not urls:
            raise FetchError(
                f"No instruction PDFs found for set {set_num} at {page_url(set_num, locale)}. "
                "The set may have no online instructions, the locale may be wrong, or the "
                "page layout changed — download the PDF manually and pass its path instead."
            )
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for url in urls:
            out = dest / url.rsplit("/", 1)[-1]
            if overwrite or not out.exists():
                out.write_bytes(self.get_bytes(url))
            paths.append(out)
        return paths
