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

import numpy as np


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


def sample_triplets_semihard(
    dataset: Dict[str, Sequence[bytes]],
    embeddings: "np.ndarray",
    n: int,
    seed: int = 0,
    margin: float = 0.3,
) -> List[Triplet]:
    """Like :func:`sample_triplets`, but pick each negative by *hardness* using
    the current model's ``embeddings`` instead of uniformly at random.

    ``embeddings`` is an ``(len(flatten_dataset(dataset)), D)`` array whose rows
    line up with ``flatten_dataset(dataset)``. Anchor/positive are chosen exactly
    as in :func:`sample_triplets` (two crops of one part). The negative is then
    chosen from crops of *other* parts: prefer the **semi-hard band** -- farther
    from the anchor than the positive but within ``margin`` of it (a real but
    not collapse-inducing violation) -- picking the nearest such crop; if the
    band is empty, fall back to the hardest (nearest overall) negative.

    Uniform-random negatives go slack once the model separates them (loss -> 0,
    no gradient); mining keeps informative triplets flowing. Deterministic given
    ``seed``. Numpy-only, so it stays unit-testable without torch.
    """
    eligible = [pn for pn, crops in dataset.items() if len(crops) >= 2]
    if len(eligible) < 2 or n <= 0:
        return []

    emb = np.asarray(embeddings, dtype=np.float32)
    flat = flatten_dataset(dataset)
    if emb.ndim != 2 or emb.shape[0] != len(flat):
        raise ValueError(
            f"embeddings must be (n_crops, D) aligned to flatten_dataset "
            f"({len(flat)} crops); got shape {emb.shape}"
        )

    parts_of = [pn for (pn, _crop) in flat]
    index_of_part: Dict[str, List[int]] = {}
    for i, part_num in enumerate(parts_of):
        index_of_part.setdefault(part_num, []).append(i)
    all_indices = np.arange(len(flat))

    rng = random.Random(seed)
    eligible_sorted = sorted(eligible)
    triplets: List[Triplet] = []
    for _ in range(n):
        anchor_part = eligible_sorted[rng.randrange(len(eligible_sorted))]
        anchor_idx, positive_idx = rng.sample(index_of_part[anchor_part], 2)

        neg_mask = np.array([pn != anchor_part for pn in parts_of])
        neg_indices = all_indices[neg_mask]
        if neg_indices.size == 0:
            continue  # only one part in the whole dataset -- no valid negative

        anchor_vec = emb[anchor_idx]
        pos_dist = float(np.linalg.norm(anchor_vec - emb[positive_idx]))
        neg_dists = np.linalg.norm(emb[neg_indices] - anchor_vec, axis=1)

        band = neg_indices[(neg_dists > pos_dist) & (neg_dists < pos_dist + margin)]
        if band.size:
            band_dists = np.linalg.norm(emb[band] - anchor_vec, axis=1)
            negative_idx = int(band[int(np.argmin(band_dists))])
        else:
            negative_idx = int(neg_indices[int(np.argmin(neg_dists))])

        triplets.append(Triplet(anchor=anchor_idx, positive=positive_idx, negative=negative_idx))
    return triplets
