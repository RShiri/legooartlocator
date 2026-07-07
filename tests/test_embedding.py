"""Tests for the nearest-neighbour retrieval math (numpy only, no ML deps)."""

import numpy as np

from legopartlocator.embedding import (
    EmbeddingBackend,
    Gallery,
    build_gallery,
    cosine_similarity,
    download_reference_images,
    l2_normalize,
)
from legopartlocator.models import InventoryPart


def _part(pn, color="Red"):
    return InventoryPart(part_num=pn, name=f"part {pn}", color_name=color, quantity=1)


class FixedBackend(EmbeddingBackend):
    """Returns a preset matrix regardless of input (order must match caller)."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def embed_images(self, images):
        return self.matrix[: len(images)]


def test_cosine_similarity_ranks_aligned_vector_highest():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sims = cosine_similarity(np.array([1.0, 0.0]), gallery)
    assert np.argmax(sims) == 0
    assert sims[0] > sims[2] > sims[1]


def test_l2_normalize_handles_zero_rows():
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.allclose(np.linalg.norm(out[1]), 1.0)
    assert np.allclose(out[0], 0.0)  # zero row stays zero, no NaN


def test_gallery_query_returns_sorted_hits():
    parts = [_part("A"), _part("B")]
    g = Gallery(parts, np.array([[1.0, 0.0], [0.0, 1.0]]))
    hits = g.query(np.array([0.9, 0.1]), top_k=2)
    assert hits[0].part.part_num == "A"
    assert hits[0].score > hits[1].score


def test_build_gallery_drops_parts_without_reference_image():
    parts = [_part("A"), _part("B"), _part("C")]
    ref_images = {"A": b"imgA", "C": b"imgC"}  # B has no image
    backend = FixedBackend([[1.0, 0.0], [0.0, 1.0]])
    g = build_gallery(parts, ref_images, backend)
    assert len(g) == 2
    assert {p.part_num for p in g.parts} == {"A", "C"}


def test_download_reference_images_uses_injected_getter():
    parts = [
        InventoryPart(part_num="A", name="a", quantity=1, image_url="http://img/a.png"),
        InventoryPart(part_num="B", name="b", quantity=1, image_url=None),  # no url -> skipped
        InventoryPart(part_num="C", name="c", quantity=1, image_url="http://img/c.png"),
    ]
    fetched = []

    def fake_get(url):
        fetched.append(url)
        return None if url.endswith("c.png") else b"data"  # C fails to download

    images = download_reference_images(parts, get=fake_get)
    assert set(images) == {"A"}  # B had no url, C failed
    assert "http://img/a.png" in fetched
