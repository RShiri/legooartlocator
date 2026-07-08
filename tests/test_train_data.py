"""Tests for the training-data augmentation pipeline (offline: no network, no torch)."""

import cv2
import numpy as np

from legopartlocator.train.data import augment, build_augmented_dataset, train_val_split


def _make_reference_image(w=120, h=100, color=(40, 90, 180)) -> bytes:
    """A simple synthetic "part photo": a filled rectangle on a white background."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), color, thickness=-1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_augment_returns_requested_count():
    ref = _make_reference_image()
    variants = augment(ref, n=5, seed=1)
    assert len(variants) == 5


def test_augment_zero_or_negative_n_returns_empty():
    ref = _make_reference_image()
    assert augment(ref, n=0) == []
    assert augment(ref, n=-3) == []


def test_augment_variants_decode_to_same_shape_as_input():
    ref = _make_reference_image(w=120, h=100)
    ref_img = cv2.imdecode(np.frombuffer(ref, np.uint8), cv2.IMREAD_COLOR)
    for variant in augment(ref, n=4, seed=2):
        decoded = cv2.imdecode(np.frombuffer(variant, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == ref_img.shape


def test_augment_is_deterministic_given_seed():
    ref = _make_reference_image()
    first = augment(ref, n=3, seed=42)
    second = augment(ref, n=3, seed=42)
    assert first == second


def test_augment_different_seeds_produce_different_variants():
    ref = _make_reference_image()
    a = augment(ref, n=1, seed=1)[0]
    b = augment(ref, n=1, seed=2)[0]
    assert a != b


def test_augment_variants_are_not_identical_to_each_other():
    ref = _make_reference_image()
    variants = augment(ref, n=4, seed=7)
    assert len(set(variants)) == len(variants)


def test_build_augmented_dataset_expands_each_part():
    refs = {"3001": _make_reference_image(color=(40, 90, 180)), "3002": _make_reference_image(color=(10, 200, 60))}
    dataset = build_augmented_dataset(refs, variants_per_part=3, seed=0)
    assert set(dataset.keys()) == {"3001", "3002"}
    assert all(len(v) == 3 for v in dataset.values())
    # Different parts' augmented sets shouldn't collide.
    assert set(dataset["3001"]).isdisjoint(dataset["3002"])


def _fake_dataset():
    return {
        "3001": [f"3001-{i}".encode() for i in range(8)],
        "3002": [f"3002-{i}".encode() for i in range(8)],
        "3003": [b"3003-0", b"3003-1"],  # exactly min_train, nothing to hold out
    }


def test_train_val_split_covers_every_crop_exactly_once():
    train, val = train_val_split(_fake_dataset(), val_frac=0.25, seed=0)
    for part_num, all_crops in _fake_dataset().items():
        got = sorted(train.get(part_num, []) + val.get(part_num, []))
        assert got == sorted(all_crops)


def test_train_val_split_keeps_at_least_min_train_variants():
    train, val = train_val_split(_fake_dataset(), val_frac=0.9, seed=0, min_train=2)
    for part_num, crops in train.items():
        assert len(crops) >= 2


def test_train_val_split_skips_val_for_parts_with_too_few_variants():
    train, val = train_val_split(_fake_dataset(), val_frac=0.5, seed=0)
    assert "3003" not in val
    assert sorted(train["3003"]) == sorted(_fake_dataset()["3003"])


def test_train_val_split_train_and_val_are_disjoint_per_part():
    train, val = train_val_split(_fake_dataset(), val_frac=0.25, seed=0)
    for part_num in val:
        assert set(train[part_num]).isdisjoint(val[part_num])


def test_train_val_split_is_deterministic_given_seed():
    a = train_val_split(_fake_dataset(), val_frac=0.25, seed=5)
    b = train_val_split(_fake_dataset(), val_frac=0.25, seed=5)
    assert a == b
