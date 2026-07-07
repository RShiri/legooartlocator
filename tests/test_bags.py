"""Tests for bag segmentation (offline, no network)."""

from legopartlocator.bags import page_to_bag, segment_bags
from legopartlocator.models import PageExtract


def _page(idx, bag=None):
    return PageExtract(page_index=idx, bag_marker=bag)


def test_basic_step_function():
    pages = [
        _page(0),  # cover / intro
        _page(1, bag=1),
        _page(2),
        _page(3, bag=2),
        _page(4),
        _page(5),
    ]
    segments, warnings = segment_bags(pages, num_pages=6)
    mapping = page_to_bag(segments)

    assert mapping == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
    assert warnings == []  # clean input, no warnings


def test_no_intro_when_first_page_is_marker():
    pages = [_page(0, bag=1), _page(1), _page(2, bag=2)]
    segments, _ = segment_bags(pages, num_pages=3)
    bags = {s.bag for s in segments}
    assert 0 not in bags  # no bag-0 intro segment
    assert page_to_bag(segments) == {0: 1, 1: 1, 2: 2}


def test_suspect_out_of_order_marker_dropped():
    # A misread "1" appearing after bag 3 must not reset the segmentation.
    pages = [
        _page(0, bag=1),
        _page(1, bag=2),
        _page(2, bag=3),
        _page(3, bag=1),  # suspect
        _page(4, bag=4),
    ]
    segments, warnings = segment_bags(pages, num_pages=5)
    mapping = page_to_bag(segments)

    assert mapping[3] == 3  # stayed in bag 3, not reset to 1
    assert mapping[4] == 4
    assert any("suspect bag marker" in w for w in warnings)


def test_no_markers_falls_back_to_bag_zero():
    pages = [_page(0), _page(1), _page(2)]
    segments, warnings = segment_bags(pages, num_pages=3)
    assert page_to_bag(segments) == {0: 0, 1: 0, 2: 0}
    assert any("No bag markers" in w for w in warnings)


def test_segments_cover_every_page_exactly_once():
    pages = [_page(0), _page(2, bag=1), _page(5, bag=2)]
    segments, _ = segment_bags(pages, num_pages=8)
    covered = []
    for seg in segments:
        covered.extend(range(seg.start_page, seg.end_page + 1))
    assert sorted(covered) == list(range(8))
    assert len(covered) == len(set(covered))  # no overlaps
