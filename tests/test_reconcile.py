"""Tests for callout->inventory reconciliation (offline, no network)."""

from legopartlocator.bags import page_to_bag, segment_bags
from legopartlocator.models import InventoryPart, PageCallout, PageExtract
from legopartlocator.reconcile import match_callout_to_inventory, reconcile


def _inv():
    return [
        InventoryPart(
            part_num="3001",
            name="Brick 2 x 4",
            color_id=5,
            color_name="Red",
            quantity=4,
            element_id="300121",
            image_url="http://img/3001.png",
        ),
        InventoryPart(
            part_num="3020",
            name="Plate 2 x 4",
            color_id=0,
            color_name="Black",
            quantity=2,
            element_id="302026",
            image_url="http://img/3020.png",
        ),
    ]


def test_match_by_printed_element_id():
    callout = PageCallout(quantity=1, printed_part_id="300121")
    match, score = match_callout_to_inventory(callout, _inv())
    assert match is not None and match.part_num == "3001"
    assert score == 1.0


def test_match_by_color_and_shape():
    callout = PageCallout(quantity=2, color="red", shape_desc="2 x 4 brick")
    match, score = match_callout_to_inventory(callout, _inv())
    assert match is not None and match.part_num == "3001"
    assert score > 0.35


def test_reconcile_counts_match_across_bags():
    # Red 2x4 brick: 2 in bag 1, 2 in bag 2 -> total 4 == inventory 4.
    pages = [
        PageExtract(page_index=0, bag_marker=1,
                    callouts=[PageCallout(quantity=2, color="red", shape_desc="2x4 brick")]),
        PageExtract(page_index=1, bag_marker=2,
                    callouts=[PageCallout(quantity=2, color="red", shape_desc="2x4 brick")]),
    ]
    segments, _ = segment_bags(pages, num_pages=2)
    parts, warnings = reconcile(pages, page_to_bag(segments), inventory=_inv())

    brick = next(p for p in parts if p.part_num == "3001")
    assert brick.total_seen == 4
    assert brick.inventory_qty == 4
    assert brick.count_matches is True
    assert brick.bags == [1, 2]
    assert brick.pages == [0, 1]
    assert brick.confidence == 0.9
    assert not any("mismatch" in w for w in warnings)


def test_reconcile_flags_count_mismatch():
    # Only saw 1 of the red brick; inventory says 4 -> mismatch warning.
    pages = [
        PageExtract(page_index=0, bag_marker=1,
                    callouts=[PageCallout(quantity=1, color="red", shape_desc="2x4 brick")]),
    ]
    segments, _ = segment_bags(pages, num_pages=1)
    parts, warnings = reconcile(pages, page_to_bag(segments), inventory=_inv())

    brick = next(p for p in parts if p.part_num == "3001")
    assert brick.count_matches is False
    assert brick.confidence == 0.6
    assert any("mismatch" in w.lower() for w in warnings)


def test_vision_only_mode_groups_by_color_shape():
    pages = [
        PageExtract(page_index=0, bag_marker=1,
                    callouts=[PageCallout(quantity=3, color="blue", shape_desc="1x2 plate")]),
        PageExtract(page_index=1,
                    callouts=[PageCallout(quantity=1, color="blue", shape_desc="1x2 plate")]),
    ]
    segments, _ = segment_bags(pages, num_pages=2)
    parts, warnings = reconcile(pages, page_to_bag(segments), inventory=None)

    assert len(parts) == 1
    p = parts[0]
    assert p.reconciled is False
    assert p.total_seen == 4
    assert p.bags == [1]
    assert p.confidence == 0.4


def test_parts_list_pages_are_ignored():
    pages = [
        PageExtract(page_index=0, bag_marker=1,
                    callouts=[PageCallout(quantity=1, color="red", shape_desc="2x4 brick")]),
        PageExtract(page_index=1, is_parts_list=True,
                    callouts=[PageCallout(quantity=99, color="red", shape_desc="2x4 brick")]),
    ]
    segments, _ = segment_bags(pages, num_pages=2)
    parts, _ = reconcile(pages, page_to_bag(segments), inventory=_inv())
    brick = next(p for p in parts if p.part_num == "3001")
    assert brick.total_seen == 1  # BOM page's 99 excluded
