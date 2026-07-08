"""Tests for the lego.com PDF auto-fetcher (offline; transports injected)."""

import fitz
import pytest

from legopartlocator.fetcher import FetchError, InstructionFetcher, find_pdf_urls, page_url

_BOOKLET_HTML = "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6559641.pdf"


def _real_pdf_bytes() -> bytes:
    """A minimal, genuinely openable one-page PDF (not just bytes starting
    with the %PDF- magic header) -- needed now that fetch() actually validates
    an existing file opens before skipping its re-download."""
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def test_page_url_strips_inventory_suffix():
    assert page_url("76307-1", "en-us") == "https://www.lego.com/en-us/service/building-instructions/76307"
    assert page_url("76307").endswith("/building-instructions/76307")


def test_find_pdf_urls_unescapes_and_dedupes():
    # LEGO embeds URLs in JSON with escaped slashes; the same URL appears twice.
    html = (
        '{"pdfLocation":"https:\\/\\/www.lego.com\\/cdn\\/product-assets\\/'
        'product.bi.core.pdf\\/6559641.pdf","x":"https:\\/\\/www.lego.com\\/cdn\\/'
        'product-assets\\/product.bi.core.pdf\\/6559641.pdf"}'
    )
    urls = find_pdf_urls(html)
    assert urls == ["https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6559641.pdf"]


def test_find_pdf_urls_multiple_booklets_in_order():
    html = (
        "a https://www.lego.com/cdn/product-assets/product.bi.core.pdf/111.pdf b "
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/222.pdf"
    )
    assert find_pdf_urls(html) == [
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/111.pdf",
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/222.pdf",
    ]


def test_find_pdf_urls_none_when_absent():
    assert find_pdf_urls("<html>no instructions here</html>") == []


def test_fetch_downloads_and_skips_existing(tmp_path):
    downloads = []
    pdf_bytes = _real_pdf_bytes()

    def get_text(url):
        return _BOOKLET_HTML

    def get_bytes(url):
        downloads.append(url)
        return pdf_bytes

    fetcher = InstructionFetcher(get_text=get_text, get_bytes=get_bytes)
    paths = fetcher.fetch("76307", dest_dir=tmp_path)
    assert len(paths) == 1 and paths[0].name == "6559641.pdf"
    assert paths[0].read_bytes() == pdf_bytes

    # Second call must not re-download an existing, valid file.
    fetcher.fetch("76307", dest_dir=tmp_path)
    assert len(downloads) == 1


def test_fetch_redownloads_a_truncated_or_corrupt_existing_file(tmp_path):
    """A file left behind by an interrupted prior download (right name,
    unparseable content) must be re-downloaded, not silently reused forever."""
    downloads = []
    pdf_bytes = _real_pdf_bytes()

    def get_text(url):
        return _BOOKLET_HTML

    def get_bytes(url):
        downloads.append(url)
        return pdf_bytes

    (tmp_path / "6559641.pdf").write_bytes(b"%PDF-1.7 truncated, not a real pdf")

    fetcher = InstructionFetcher(get_text=get_text, get_bytes=get_bytes)
    paths = fetcher.fetch("76307", dest_dir=tmp_path)

    assert len(downloads) == 1
    assert paths[0].read_bytes() == pdf_bytes


def test_fetch_raises_when_no_pdfs(tmp_path):
    fetcher = InstructionFetcher(get_text=lambda u: "<html>nothing</html>", get_bytes=lambda u: b"")
    with pytest.raises(FetchError):
        fetcher.fetch("00000", dest_dir=tmp_path)
