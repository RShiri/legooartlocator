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


# --- capacity-reconcile (--capacity-reconcile): divert the embedding "attractor" ---

def _cap_inv():
    return [
        InventoryPart(part_num="A1", name="Attractor Part", color_id=1, color_name="Red", quantity=1),
        InventoryPart(part_num="B2", name="Roomy Part", color_id=2, color_name="Blue", quantity=4),
    ]


class PresetIdentifier:
    """Returns a preset IdentificationResult per crop tag (with components and
    alternatives set), so capacity-reconcile's trust/capacity logic can run."""

    def __init__(self, mapping):
        self.mapping = mapping  # tag -> IdentificationResult

    def identify(self, crop_bytes, seen_color=None):
        return self.mapping.get(crop_bytes, IdentificationResult(part=None, confidence=0.0))


def _embed_only_hit(part, alt):
    """A match carried by embedding alone (Brickognize contributed nothing),
    with ``alt`` as its runner-up -- the shape the attractor produces."""
    return IdentificationResult(
        part=part, confidence=0.7,
        components={"embedding": 0.7, "brickognize": 0.0},
        alternatives=[(alt, 0.5)],
    )


def test_capacity_reconcile_diverts_embedding_only_overcount():
    inv = _cap_inv()
    a, b = inv
    ident = PresetIdentifier({
        b"c1": _embed_only_hit(a, b),
        b"c2": _embed_only_hit(a, b),
        b"c3": _embed_only_hit(a, b),
    })
    dets = [PageDetection(page_index=0, bag_marker=1,
                          callouts=[_callout(1, b"c1"), _callout(1, b"c2"), _callout(1, b"c3")])]

    # With reconcile disabled, the attractor A (inventory qty 1) soaks up all three.
    base = assemble_result(dets, inv, ident, num_pages=1, use_color=False, capacity_reconcile=False)
    a_base = next(p for p in base.parts if p.part_num == "A1")
    assert a_base.total_seen == 3 and a_base.count_matches is False

    # Capacity-reconcile: A keeps its one allowed crop; the surplus diverts to B.
    fixed = assemble_result(dets, inv, ident, num_pages=1, use_color=False, capacity_reconcile=True)
    a_fixed = next(p for p in fixed.parts if p.part_num == "A1")
    b_fixed = next(p for p in fixed.parts if p.part_num == "B2")
    assert a_fixed.total_seen == 1 and a_fixed.count_matches is True
    assert b_fixed.total_seen == 2


def test_capacity_reconcile_keeps_corroborated_overcount():
    inv = _cap_inv()
    a, b = inv
    corroborated = IdentificationResult(
        part=a, confidence=0.9,
        components={"embedding": 0.6, "brickognize": 0.8},  # Brickognize agrees
        alternatives=[(b, 0.5)],
    )
    ident = PresetIdentifier({t: corroborated for t in (b"c1", b"c2", b"c3")})
    dets = [PageDetection(page_index=0, bag_marker=1,
                          callouts=[_callout(1, b"c1"), _callout(1, b"c2"), _callout(1, b"c3")])]

    fixed = assemble_result(dets, inv, ident, num_pages=1, use_color=False, capacity_reconcile=True)
    a_fixed = next(p for p in fixed.parts if p.part_num == "A1")
    # A corroborated winner is never diverted, even past its inventory quantity.
    assert a_fixed.total_seen == 3
    assert not any(p.part_num == "B2" for p in fixed.parts)


def test_capacity_reconcile_is_on_by_default():
    inv = _cap_inv()
    a, b = inv
    ident = PresetIdentifier({t: _embed_only_hit(a, b) for t in (b"c1", b"c2", b"c3")})
    dets = [PageDetection(page_index=0, bag_marker=1,
                          callouts=[_callout(1, b"c1"), _callout(1, b"c2"), _callout(1, b"c3")])]
    result = assemble_result(dets, inv, ident, num_pages=1, use_color=False)  # no flag -> default on
    a_res = next(p for p in result.parts if p.part_num == "A1")
    assert a_res.total_seen == 1  # attractor capped by default
    assert any(p.part_num == "B2" for p in result.parts)


def test_dump_crops_writes_files_and_manifest(tmp_path):
    import json

    inv = _inv()
    detections = [
        PageDetection(page_index=0, bag_marker=1, callouts=[_callout(2, b"red"), _callout(1, b"mystery")]),
    ]
    ident = FakeIdentifier(inv, {b"red": (0, 0.9)})  # b"mystery" -> unidentified
    out = tmp_path / "crops"

    assemble_result(detections, inv, ident, num_pages=1, use_color=False, dump_crops_dir=out)

    assert (out / "3001" / "p000_i000.png").read_bytes() == b"red"
    assert (out / "unknown" / "p000_i001.png").read_bytes() == b"mystery"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    red, unknown = manifest
    assert red["part_num"] == "3001" and red["raw_part_num"] == "3001"
    assert red["file"] == "3001/p000_i000.png"
    assert red["page_index"] == 0 and red["bag"] == 1 and red["quantity"] == 2
    assert red["confidence"] == 0.9
    assert isinstance(red["components"], dict) and isinstance(red["alternatives"], list)
    assert unknown["part_num"] is None and unknown["file"] == "unknown/p000_i001.png"


def test_dump_crops_records_raw_winner_when_diverted(tmp_path):
    import json

    inv = _cap_inv()
    a, b = inv
    ident = PresetIdentifier({t: _embed_only_hit(a, b) for t in (b"c1", b"c2")})
    dets = [PageDetection(page_index=0, bag_marker=1, callouts=[_callout(1, b"c1"), _callout(1, b"c2")])]
    out = tmp_path / "crops"

    assemble_result(dets, inv, ident, num_pages=1, use_color=False,
                    capacity_reconcile=True, dump_crops_dir=out)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # One crop stayed on the attractor A1; the diverted one records A1 as its
    # raw winner but B2 as its final assignment.
    finals = sorted(e["part_num"] for e in manifest)
    assert finals == ["A1", "B2"]
    diverted = next(e for e in manifest if e["part_num"] == "B2")
    assert diverted["raw_part_num"] == "A1"
    assert (out / "B2").is_dir()


def test_no_dump_dir_writes_nothing(tmp_path):
    inv = _inv()
    detections = [PageDetection(page_index=0, bag_marker=1, callouts=[_callout(1, b"red")])]
    ident = FakeIdentifier(inv, {b"red": (0, 0.9)})
    assemble_result(detections, inv, ident, num_pages=1, use_color=False)
    assert list(tmp_path.iterdir()) == []  # default: no side effects


def test_capacity_reconcile_can_be_disabled():
    inv = _cap_inv()
    a, b = inv
    ident = PresetIdentifier({b"c1": _embed_only_hit(a, b), b"c2": _embed_only_hit(a, b)})
    dets = [PageDetection(page_index=0, bag_marker=1, callouts=[_callout(1, b"c1"), _callout(1, b"c2")])]
    off = assemble_result(dets, inv, ident, num_pages=1, use_color=False, capacity_reconcile=False)
    a_off = next(p for p in off.parts if p.part_num == "A1")
    assert a_off.total_seen == 2  # both crops stay on the attractor when disabled
    assert not any(p.part_num == "B2" for p in off.parts)
