"""Tests for the ensemble part identifier (pure logic, no network/torch)."""

import numpy as np

from legopartlocator.brickognize import BrickognizeClient
from legopartlocator.embedding import EmbeddingBackend, Gallery
from legopartlocator.identify import (
    BrickognizeOnlyIdentifier,
    PartIdentifier,
    blend_scores,
    color_score,
)
from legopartlocator.models import InventoryPart


def _inventory():
    return [
        InventoryPart(part_num="3001", name="Brick 2 x 4", color_id=5, color_name="Red", quantity=2),
        InventoryPart(part_num="3001", name="Brick 2 x 4", color_id=1, color_name="Blue", quantity=1),
        InventoryPart(part_num="3020", name="Plate 2 x 4", color_id=0, color_name="Black", quantity=3),
    ]


def _fake_brickognize(items):
    return BrickognizeClient(transport=lambda img, name: {"items": items})


class QueryBackend(EmbeddingBackend):
    def __init__(self, vec):
        self.vec = np.asarray([vec], dtype=np.float32)

    def embed_images(self, images):
        return self.vec


# --- blend_scores ---------------------------------------------------------

def test_blend_renormalises_over_present_signals():
    # Only brickognize present -> its weight renormalises to 1.0.
    out = blend_scores({"brickognize": {0: 0.8}}, {"brickognize": 0.5, "color": 0.1})
    assert out[0] == 0.8


def test_blend_combines_two_signals():
    out = blend_scores(
        {"brickognize": {0: 0.9}, "color": {0: 1.0}},
        {"brickognize": 0.5, "embedding": 0.4, "color": 0.1},
    )
    # weights renormalise over {brickognize:0.5, color:0.1} -> total 0.6
    assert abs(out[0] - ((0.5 / 0.6) * 0.9 + (0.1 / 0.6) * 1.0)) < 1e-9


def test_color_score_overlap():
    part = _inventory()[0]  # Red
    assert color_score("red", part) == 1.0
    assert color_score("blue", part) == 0.0


# --- full identify --------------------------------------------------------

def test_color_disambiguates_same_part_across_colors():
    inv = _inventory()
    ident = PartIdentifier(
        inv, brickognize=_fake_brickognize([{"id": "3001", "name": "Brick 2 x 4", "score": 0.9}])
    )
    result = ident.identify(b"crop", seen_color="red")
    assert result.part.part_num == "3001"
    assert result.part.color_name == "Red"  # not Blue
    assert result.confidence > 0.9
    # Runner-up should be the blue 3001.
    assert result.alternatives[0][0].color_name == "Blue"


def test_brickognize_candidates_outside_inventory_are_ignored():
    inv = _inventory()
    ident = PartIdentifier(
        inv,
        brickognize=_fake_brickognize(
            [
                {"id": "99999", "name": "Not in set", "score": 0.99},  # dropped
                {"id": "3020", "name": "Plate 2 x 4", "score": 0.4},
            ]
        ),
    )
    result = ident.identify(b"crop")
    assert result.part.part_num == "3020"  # the out-of-inventory 99999 was ignored


def test_embedding_signal_contributes():
    inv = _inventory()
    # Gallery over the two 3001 colour lines; query aligned with the Blue one.
    gallery = Gallery([inv[0], inv[1]], np.array([[1.0, 0.0], [0.0, 1.0]]))
    ident = PartIdentifier(
        inv,
        gallery=gallery,
        backend=QueryBackend([0.05, 1.0]),  # closest to inv[1] (Blue)
        weights={"embedding": 1.0},
    )
    result = ident.identify(b"crop")
    assert result.part.color_name == "Blue"
    assert result.components["embedding"] > 0.9


def test_no_signals_returns_none():
    ident = PartIdentifier(_inventory())  # no brickognize, no gallery
    result = ident.identify(b"crop")
    assert result.part is None
    assert result.confidence == 0.0


def test_color_only_never_identifies_a_part():
    # No brickognize, no gallery: colour is the only signal available. It must
    # not be enough on its own to name a specific part, even though the
    # inventory contains a Red line that would otherwise "match" perfectly.
    ident = PartIdentifier(_inventory())
    result = ident.identify(b"crop", seen_color="red")
    assert result.part is None
    assert result.confidence == 0.0


def test_color_does_not_rescue_out_of_inventory_brickognize_hit():
    # Brickognize is present but only proposes a part_num outside the
    # inventory; colour agreement alone must not "rescue" a match.
    inv = _inventory()
    ident = PartIdentifier(
        inv,
        brickognize=_fake_brickognize(
            [{"id": "99999", "name": "Not in set", "score": 0.99}]
        ),
    )
    result = ident.identify(b"crop", seen_color="red")
    assert result.part is None


# --- BrickognizeOnlyIdentifier (zero-inventory path) ----------------------

def test_brickognize_only_returns_top_candidate_unconstrained():
    # Part 99999 is NOT in any inventory, but the unconstrained identifier keeps it.
    bk = _fake_brickognize([
        {"id": "99999", "name": "Rare part", "score": 0.7, "img_url": "http://img/9.png"},
        {"id": "3001", "name": "Brick 2 x 4", "score": 0.3},
    ])
    ident = BrickognizeOnlyIdentifier(bk)
    result = ident.identify(b"crop", seen_color="red")
    assert result.part.part_num == "99999"
    assert result.part.name == "Rare part"
    assert result.part.color_name == "red"      # seen colour carried through
    assert result.part.image_url == "http://img/9.png"
    assert result.confidence == 0.7
    assert result.alternatives[0][0].part_num == "3001"


def test_brickognize_only_none_when_no_candidates():
    ident = BrickognizeOnlyIdentifier(_fake_brickognize([]))
    assert ident.identify(b"crop").part is None
