"""Tests for triplet sampling (pure Python + numpy, no torch/network)."""

import numpy as np
import pytest

from legopartlocator.train.sampling import (
    flatten_dataset,
    sample_triplets,
    sample_triplets_semihard,
)


def _dataset():
    return {
        "3001": [b"3001-a", b"3001-b", b"3001-c"],
        "3002": [b"3002-a", b"3002-b"],
        "3003": [b"3003-a"],  # only 1 variant -- ineligible as an anchor
    }


def test_flatten_dataset_is_stable_and_covers_every_crop():
    flat = flatten_dataset(_dataset())
    assert len(flat) == 6
    assert flat == flatten_dataset(_dataset())  # deterministic order


def test_sample_triplets_returns_requested_count():
    triplets = sample_triplets(_dataset(), n=10, seed=0)
    assert len(triplets) == 10


def test_sample_triplets_anchor_and_positive_share_a_part_but_differ():
    flat = flatten_dataset(_dataset())
    for t in sample_triplets(_dataset(), n=25, seed=1):
        anchor_part, _ = flat[t.anchor]
        positive_part, _ = flat[t.positive]
        assert anchor_part == positive_part
        assert t.anchor != t.positive


def test_sample_triplets_negative_is_a_different_part():
    flat = flatten_dataset(_dataset())
    for t in sample_triplets(_dataset(), n=25, seed=2):
        anchor_part, _ = flat[t.anchor]
        negative_part, _ = flat[t.negative]
        assert anchor_part != negative_part


def test_sample_triplets_never_picks_the_single_variant_part_as_anchor():
    flat = flatten_dataset(_dataset())
    for t in sample_triplets(_dataset(), n=25, seed=3):
        anchor_part, _ = flat[t.anchor]
        assert anchor_part != "3003"


def test_sample_triplets_is_deterministic_given_seed():
    a = sample_triplets(_dataset(), n=8, seed=42)
    b = sample_triplets(_dataset(), n=8, seed=42)
    assert a == b


def test_sample_triplets_handles_no_eligible_parts():
    dataset = {"3001": [b"only-one"]}
    assert sample_triplets(dataset, n=5, seed=0) == []


def test_sample_triplets_handles_n_zero():
    assert sample_triplets(_dataset(), n=0, seed=0) == []


def test_sample_triplets_handles_single_part_overall():
    """Only one distinct part in the dataset -- no valid negative exists."""
    dataset = {"3001": [b"a", b"b", b"c"]}
    assert sample_triplets(dataset, n=5, seed=0) == []


# --- semi-hard negative mining (--mining semihard) ---

def _two_part_dataset():
    # flatten order (sorted by part_num): a0=0, a1=1, b0=2, b1=3
    return {"A": [b"a0", b"a1"], "B": [b"b0", b"b1"]}


def test_semihard_selects_the_nearest_different_part_negative():
    dataset = _two_part_dataset()
    flat = flatten_dataset(dataset)
    # Same-part positives are placed FAR apart and the other part's crops sit
    # near every anchor, so for every anchor/positive pair the nearest
    # different-part crop is the mined negative (band choice coincides with it).
    emb = np.array(
        [
            [0.0, 0.0],   # a0
            [0.0, 5.0],   # a1  -- distant positive for a0
            [0.0, 0.1],   # b0
            [0.0, 0.2],   # b1
        ],
        dtype=np.float32,
    )
    triplets = sample_triplets_semihard(dataset, emb, n=20, seed=0, margin=0.3)
    assert triplets
    for t in triplets:
        a_part, _ = flat[t.anchor]
        p_part, _ = flat[t.positive]
        n_part, _ = flat[t.negative]
        assert a_part == p_part and t.anchor != t.positive
        assert n_part != a_part
        neg_idxs = [i for i, (pn, _c) in enumerate(flat) if pn != a_part]
        nearest = neg_idxs[int(np.argmin([np.linalg.norm(emb[i] - emb[t.anchor]) for i in neg_idxs]))]
        assert t.negative == nearest


def test_semihard_prefers_a_negative_inside_the_margin_band():
    dataset = _two_part_dataset()
    flat = flatten_dataset(dataset)
    emb = np.array(
        [
            [0.0, 0.0],    # a0
            [0.0, 0.1],    # a1  (positive; pos_dist 0.1 from a0)
            [0.0, 0.25],   # b0  -- dist 0.25 from a0: inside band (0.1, 0.4)
            [0.0, 0.05],   # b1  -- dist 0.05 from a0: hard (below the positive)
        ],
        dtype=np.float32,
    )
    triplets = sample_triplets_semihard(dataset, emb, n=60, seed=1, margin=0.3)
    exercised = False
    for t in triplets:
        if flat[t.anchor][1] == b"a0" and flat[t.positive][1] == b"a1":
            exercised = True
            # b0 (semi-hard, in band) is preferred over the harder b1.
            assert flat[t.negative][1] == b"b0"
    assert exercised


def test_semihard_is_deterministic_given_seed():
    dataset = _two_part_dataset()
    emb = np.arange(8, dtype=np.float32).reshape(4, 2)
    a = sample_triplets_semihard(dataset, emb, n=8, seed=3)
    b = sample_triplets_semihard(dataset, emb, n=8, seed=3)
    assert a == b


def test_semihard_handles_no_eligible_parts_and_n_zero():
    single = {"A": [b"only"]}
    assert sample_triplets_semihard(single, np.zeros((1, 2), dtype=np.float32), n=5, seed=0) == []
    ds = _two_part_dataset()
    assert sample_triplets_semihard(ds, np.zeros((4, 2), dtype=np.float32), n=0, seed=0) == []


def test_semihard_rejects_misaligned_embeddings():
    ds = _two_part_dataset()  # 4 crops
    with pytest.raises(ValueError):
        sample_triplets_semihard(ds, np.zeros((3, 2), dtype=np.float32), n=5, seed=0)
