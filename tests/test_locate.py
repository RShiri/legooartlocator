"""Tests for the local orchestrator's pure core (no OpenCV/torch/network)."""

from legopartlocator.detection import DetectedCallout, PageDetection
from legopartlocator.identify import IdentificationResult
from legopartlocator.locate import assemble_result
from legopartlocator.models import InventoryPart, PageCallout


def _inv():
    return [
        InventoryPart(part_num="3001", name="Brick 2 x 4", color_id=5, color_name="Red", quantity=4),
        InventoryPart(part_num="3020", name="Plate 2 x 4", color_id=0, color_name="Black", quantity=2),
    ]


def _callout(qty, tag):
    # crop_png doubles as a routing tag for the fake identifier.
    return DetectedCallout(callout=PageCallout(quantity=qty), crop_png=tag)


class FakeIdentifier:
    """Maps a crop tag -> (inventory part, confidence); unknown tag -> no match."""

    def __init__(self, inventory, mapping):
        self.inventory = inventory
        self.mapping = mapping  # tag -> (part_index or None, confidence)

    def identify(self, crop_bytes, seen_color=None):
        idx, conf = self.mapping.get(crop_bytes, (None, 0.0))
        part = self.inventory[idx] if idx is not None else None
        return IdentificationResult(part=part, confidence=conf)


class FlakyFakeIdentifier:
    """Like FakeIdentifier, but raises RuntimeError for a chosen set of crop tags
    (simulating a Brickognize HTTP failure / timeout / 429)."""

    def __init__(self, inventory, mapping, failing_tags):
        self.inventory = inventory
        self.mapping = mapping
        self.failing_tags = set(failing_tags)

    def identify(self, crop_bytes, seen_color=None):
        if crop_bytes in self.failing_tags:
            raise RuntimeError(f"boom on {crop_bytes!r}")
        idx, conf = self.mapping.get(crop_bytes, (None, 0.0))
        part = self.inventory[idx] if idx is not None else None
        return IdentificationResult(part=part, confidence=conf)


def test_attributes_parts_to_bags_and_reconciles_counts():
    inv = _inv()
    detections = [
        PageDetection(page_index=0, bag_marker=1, callouts=[_callout(2, b"red"), _callout(1, b"black")]),
        PageDetection(page_index=1, callouts=[_callout(2, b"red")]),          # still bag 1
        PageDetection(page_index=2, bag_marker=2, callouts=[_callout(1, b"black")]),
    ]
    ident = FakeIdentifier(inv, {b"red": (0, 0.9), b"black": (1, 0.8)})

    result = assemble_result(detections, inv, ident, num_pages=3, use_color=False)

    red = next(p for p in result.parts if p.part_num == "3001")
    black = next(p for p in result.parts if p.part_num == "3020")
    assert red.total_seen == 4 and red.inventory_qty == 4 and red.count_matches is True
    assert red.bags == [1] and red.pages == [0, 1]
    assert black.total_seen == 2 and black.bags == [1, 2] and black.count_matches is True
    assert result.reconciled is True
    assert not any("mismatch" in w for w in result.warnings)


def test_confidence_is_averaged_over_occurrences():
    inv = _inv()
    detections = [
        PageDetection(page_index=0, bag_marker=1, callouts=[_callout(2, b"red")]),
        PageDetection(page_index=1, callouts=[_callout(2, b"red")]),
    ]
    ident = FakeIdentifier(inv, {b"red": (0, 0.8)})
    result = assemble_result(detections, inv, ident, num_pages=2, use_color=False)
    red = next(p for p in result.parts if p.part_num == "3001")
    assert red.confidence == 0.8  # mean of [0.8, 0.8]


def test_count_mismatch_is_flagged():
    inv = _inv()
    detections = [PageDetection(page_index=0, bag_marker=1, callouts=[_callout(1, b"red")])]
    ident = FakeIdentifier(inv, {b"red": (0, 0.9)})
    result = assemble_result(detections, inv, ident, num_pages=1, use_color=False)
    red = next(p for p in result.parts if p.part_num == "3001")
    assert red.count_matches is False  # saw 1, inventory 4
    assert any("mismatch" in w.lower() for w in result.warnings)


def test_unidentified_callouts_are_grouped_and_warned():
    inv = _inv()
    detections = [
        PageDetection(page_index=0, bag_marker=1, callouts=[_callout(3, b"mystery"), _callout(1, b"mystery")]),
    ]
    ident = FakeIdentifier(inv, {})  # nothing matches
    result = assemble_result(detections, inv, ident, num_pages=1, use_color=False)
    unknown = [p for p in result.parts if not p.reconciled]
    assert len(unknown) == 1
    assert unknown[0].total_seen == 4
    assert any("could not be identified" in w for w in result.warnings)


def test_unconstrained_mode_keeps_parts_without_count_check():
    inv = _inv()
    detections = [PageDetection(page_index=0, bag_marker=1, callouts=[_callout(1, b"red")])]
    ident = FakeIdentifier(inv, {b"red": (0, 0.9)})
    result = assemble_result(detections, inv, ident, num_pages=1, use_color=False, reconcile_counts=False)
    red = next(p for p in result.parts if p.part_num == "3001")
    assert red.reconciled is False
    assert red.inventory_qty is None
    assert red.count_matches is None            # no count validation
    assert result.reconciled is False
    assert not any("mismatch" in w for w in result.warnings)


def test_identify_failure_lands_in_unidentified_bucket_and_warns():
    inv = _inv()
    detections = [
        PageDetection(page_index=0, bag_marker=1, callouts=[_callout(2, b"red")]),
        PageDetection(page_index=1, callouts=[_callout(3, b"boom")]),
    ]
    ident = FlakyFakeIdentifier(inv, {b"red": (0, 0.9)}, failing_tags={b"boom"})

    result = assemble_result(detections, inv, ident, num_pages=2, use_color=False)

    # The scan completes and still identifies the parts it could.
    red = next(p for p in result.parts if p.part_num == "3001")
    assert red.total_seen == 2

    # The failing callout is bucketed like any other unidentified crop.
    unknown = [p for p in result.parts if not p.reconciled]
    assert len(unknown) == 1
    assert unknown[0].total_seen == 3

    assert any("identify failed on page" in w for w in result.warnings)
    assert any("page 2" in w for w in result.warnings)


def test_identify_failures_are_capped_with_summary_warning():
    inv = _inv()
    # 5 pages, each with one callout whose identify() raises.
    detections = [
        PageDetection(page_index=i, bag_marker=1, callouts=[_callout(1, f"boom{i}".encode())])
        for i in range(5)
    ]
    failing_tags = {f"boom{i}".encode() for i in range(5)}
    ident = FlakyFakeIdentifier(inv, {}, failing_tags=failing_tags)

    result = assemble_result(detections, inv, ident, num_pages=5, use_color=False)

    failure_msgs = [w for w in result.warnings if "identify failed on page" in w]
    assert len(failure_msgs) == 3  # capped at 3 distinct messages

    summary = [w for w in result.warnings if "identify call(s) failed in total" in w]
    assert len(summary) == 1
    assert "5" in summary[0]


def test_parts_list_page_is_skipped():
    inv = _inv()
    detections = [
        PageDetection(page_index=0, bag_marker=1, callouts=[_callout(2, b"red")]),
        PageDetection(page_index=1, is_parts_list=True, callouts=[_callout(99, b"red")]),
    ]
    ident = FakeIdentifier(inv, {b"red": (0, 0.9)})
    result = assemble_result(detections, inv, ident, num_pages=2, use_color=False)
    red = next(p for p in result.parts if p.part_num == "3001")
    assert red.total_seen == 2  # BOM page's 99 excluded
