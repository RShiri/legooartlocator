"""Tests for two-pass extraction orchestration (offline, no network/PyMuPDF)."""

from types import SimpleNamespace

from legopartlocator.models import PageCallout, PageExtract
from legopartlocator.vision import extract_pages_two_pass


def _render(idx):
    # Duck-typed RenderedPage: only page_index + png_bytes are used.
    return SimpleNamespace(page_index=idx, png_bytes=f"page{idx}".encode(), text="")


class FakeExtractor:
    """Stub extractor: triage from a table, detail returns canned callouts."""

    def __init__(self, triage_by_index, detail_by_index):
        self.triage_by_index = triage_by_index
        self.detail_by_index = detail_by_index
        self.triage_calls = []
        self.detail_calls = []

    def triage_page(self, page, use_cache=True):
        self.triage_calls.append(page.page_index)
        return self.triage_by_index[page.page_index]

    def extract_page(self, page, use_cache=True):
        self.detail_calls.append(page.page_index)
        return self.detail_by_index[page.page_index]


def test_detail_only_runs_on_callout_pages():
    triage = {
        0: {"bag_marker": None, "is_parts_list": False, "has_callouts": False},  # cover
        1: {"bag_marker": 1, "is_parts_list": False, "has_callouts": True},      # bag 1 + parts
        2: {"bag_marker": None, "is_parts_list": False, "has_callouts": True},   # step
        3: {"bag_marker": None, "is_parts_list": True, "has_callouts": False},   # BOM
    }
    detail = {
        1: PageExtract(page_index=1, bag_marker=None,
                       callouts=[PageCallout(quantity=2, color="red", shape_desc="2x4 brick")]),
        2: PageExtract(page_index=2, callouts=[PageCallout(quantity=1, color="black", shape_desc="plate")]),
    }
    ex = FakeExtractor(triage, detail)
    triage_pages = [_render(i) for i in range(4)]

    results, stats = extract_pages_two_pass(triage_pages, _render, ex)

    # Detailed pass ran only on the two callout pages.
    assert sorted(ex.detail_calls) == [1, 2]
    assert stats == {"triaged": 4, "detailed": 2, "skipped": 2}
    assert len(results) == 4


def test_skipped_pages_keep_triage_bag_marker_and_empty_callouts():
    triage = {
        0: {"bag_marker": 1, "is_parts_list": False, "has_callouts": False},
        1: {"bag_marker": None, "is_parts_list": False, "has_callouts": True},
    }
    detail = {1: PageExtract(page_index=1, callouts=[PageCallout(quantity=1, color="blue", shape_desc="1x1")])}
    ex = FakeExtractor(triage, detail)

    results, _ = extract_pages_two_pass([_render(0), _render(1)], _render, ex)

    p0 = next(r for r in results if r.page_index == 0)
    assert p0.bag_marker == 1  # marker preserved from triage
    assert p0.callouts == []   # no detail pass, no callouts
    p1 = next(r for r in results if r.page_index == 1)
    assert p1.bag_marker is None  # triage saw no marker here; step-function fill happens later in bags.py
    assert len(p1.callouts) == 1


def test_bom_page_never_gets_detail_even_if_has_callouts_true():
    # Defensive: a BOM page must be skipped regardless of has_callouts noise.
    triage = {0: {"bag_marker": None, "is_parts_list": True, "has_callouts": True}}
    ex = FakeExtractor(triage, {})
    results, stats = extract_pages_two_pass([_render(0)], _render, ex)
    assert ex.detail_calls == []
    assert stats["detailed"] == 0
    assert results[0].is_parts_list is True
    assert results[0].callouts == []
