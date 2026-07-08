"""Tests for pdf_render.py's pure page-range parsing and set-number detection.

parse_page_range needs no PDF at all (pure string parsing). detect_set_number
needs a real fitz document, built in-memory here rather than requiring a
checked-in sample PDF.
"""

import fitz
import pytest

from legopartlocator.pdf_render import detect_set_number, parse_page_range


def test_parse_page_range_none_returns_all_pages():
    assert parse_page_range(None, 5) == [0, 1, 2, 3, 4]


def test_parse_page_range_empty_string_returns_all_pages():
    assert parse_page_range("", 5) == [0, 1, 2, 3, 4]


def test_parse_page_range_singles_ranges_and_dedup():
    # 1-based "1-3,2,5" -> 0-based {0,1,2} + 1 (dup, dropped) + 4, order preserved.
    assert parse_page_range("1-3,2,5", num_pages=10) == [0, 1, 2, 4]


def test_parse_page_range_clamps_out_of_bounds():
    assert parse_page_range("1-10", num_pages=3) == [0, 1, 2]


def test_parse_page_range_malformed_range_chunk_raises_value_error():
    with pytest.raises(ValueError, match="not integers"):
        parse_page_range("abc-5", num_pages=10)


def test_parse_page_range_malformed_single_chunk_raises_value_error():
    with pytest.raises(ValueError, match="not an integer"):
        parse_page_range("abc", num_pages=10)


def test_parse_page_range_reversed_range_raises_value_error():
    with pytest.raises(ValueError, match="start > end"):
        parse_page_range("50-10", num_pages=100)


def _make_pdf(tmp_path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_detect_set_number_finds_plausible_candidate(tmp_path):
    # A short number with nearby context words outscores a longer, bare one.
    path = _make_pdf(tmp_path, ["76307 Ages 9+", "unrelated content", "6559641"])
    assert detect_set_number(path) == "76307"


def test_detect_set_number_returns_none_without_candidates(tmp_path):
    path = _make_pdf(tmp_path, ["no numbers here", "nothing at all"])
    assert detect_set_number(path) is None
