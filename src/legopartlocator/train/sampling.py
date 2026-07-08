"""Triplet sampling: pure index bookkeeping, no torch/images involved.

Kept separate from ``embedding_trainer.py`` so the "which crops go together"
logic is unit-testable without torch installed -- torch is an optional,
lazily-imported extra (see ``embedding_trainer.py`` and ``embedding.py``'s
``ClipBackend``), not part of the base test environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Triplet:
    """Indices into a flattened (part_key, variant_index) crop list."""

    anchor: int
    positive: int
    negative: int


def flatten_dataset(dataset: Dict[str, Sequence[bytes]]) -> List[tuple]:
    """``{part_num: [crop, ...]} -> [(part_num, crop), ...]``, stable order."""
    flat: List[tuple] = []
    for part_num in sorted(dataset):
        for crop in dataset[part_num]:
            flat.append((part_num, crop))
    return flat


def sample_triplets(dataset: Dict[str, Sequence[bytes]], n: int, seed: int = 0) -> List[Triplet]:
    """Sample ``n`` (anchor, positive, negative) triplets as indices into
    ``flatten_dataset(dataset)``.

    Anchor and positive are two different crops of the *same* part (so a
    part needs >= 2 variants to be eligible); negative is a crop of any
    *different* part. Parts with fewer than 2 variants are skipped (no valid
    positive pair exists). Deterministic given ``seed``.
    """
    eligible = [pn for pn, crops in dataset.items() if len(crops) >= 2]
    if len(eligible) < 2 or n <= 0:
        return []

    flat = flatten_dataset(dataset)
    index_of_part: Dict[str, List[int]] = {}
    for i, (part_num, _crop) in enumerate(flat):
        index_of_part.setdefault(part_num, []).append(i)

    # Any part in the whole dataset (not just "eligible") is a valid negative
    # source -- only the anchor/positive pair needs >= 2 variants.
    all_parts_sorted = sorted(dataset)

    rng = random.Random(seed)
    triplets: List[Triplet] = []
    eligible_sorted = sorted(eligible)
    for _ in range(n):
        anchor_part = eligible_sorted[rng.randrange(len(eligible_sorted))]
        anchor_idx, positive_idx = rng.sample(index_of_part[anchor_part], 2)

        other_parts = [p for p in all_parts_sorted if p != anchor_part]
        if not other_parts:
            continue  # only one part in the whole dataset -- no valid negative
        negative_part = other_parts[rng.randrange(len(other_parts))]
        negative_idx = index_of_part[negative_part][rng.randrange(len(index_of_part[negative_part]))]

        triplets.append(Triplet(anchor=anchor_idx, positive=positive_idx, negative=negative_idx))
    return triplets
