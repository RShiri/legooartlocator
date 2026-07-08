"""Tests for triplet sampling (pure Python, no torch/network)."""

from legopartlocator.train.sampling import flatten_dataset, sample_triplets


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
