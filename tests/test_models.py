"""Tests for the pydantic models (validation rules, not just field presence)."""

import pytest
from pydantic import ValidationError

from legopartlocator.models import BagSegment, PageCallout, PageExtract


def test_bag_segment_accepts_single_page_range():
    seg = BagSegment(bag=1, start_page=5, end_page=5)
    assert seg.contains(5) and not seg.contains(6)


def test_bag_segment_accepts_normal_range():
    seg = BagSegment(bag=1, start_page=2, end_page=8)
    assert seg.contains(2) and seg.contains(8) and not seg.contains(9)


def test_bag_segment_rejects_inverted_range():
    with pytest.raises(ValidationError, match="end_page"):
        BagSegment(bag=1, start_page=10, end_page=5)


def test_bag_segment_rejects_negative_pages():
    with pytest.raises(ValidationError):
        BagSegment(bag=1, start_page=-1, end_page=5)


def test_page_callout_rejects_zero_quantity():
    with pytest.raises(ValidationError):
        PageCallout(quantity=0)


def test_page_callout_accepts_quantity_one():
    assert PageCallout(quantity=1).quantity == 1


def test_page_extract_rejects_negative_page_index():
    with pytest.raises(ValidationError):
        PageExtract(page_index=-1, is_parts_list=False, callouts=[])


def test_page_extract_defaults():
    pe = PageExtract(page_index=0, is_parts_list=False, callouts=[])
    assert pe.bag_marker is None
    assert pe.callouts == []
