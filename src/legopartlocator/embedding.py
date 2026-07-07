"""Embedding retrieval: match a callout crop to the set's inventory by image similarity.

The idea: embed each inventory part's reference image once (a small gallery,
~50-200 vectors because we only consider parts in *this* set), embed the
callout crop, and take the nearest neighbour by cosine similarity. Because a
pretrained encoder is used, no training is required.

This module keeps two layers cleanly separated:
  * the nearest-neighbour math (pure numpy) — fully unit-tested, no ML deps,
  * the image encoder backend (torch + open_clip) — optional, imported lazily,
    only needed to actually turn images into vectors on your machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import InventoryPart


def download_reference_images(
    parts: Sequence[InventoryPart], get: Optional[Callable[[str], Optional[bytes]]] = None
) -> Dict[str, bytes]:
    """Fetch each part's reference image, keyed by part_num.

    ``get(url) -> bytes | None`` is injectable for tests; the default uses httpx.
    Parts without an image URL, or whose fetch fails, are simply omitted.
    """
    if get is None:
        get = _httpx_get
    images: Dict[str, bytes] = {}
    for part in parts:
        if not part.image_url or part.part_num in images:
            continue
        data = get(part.image_url)
        if data:
            images[part.part_num] = data
    return images


def _httpx_get(url: str) -> Optional[bytes]:  # pragma: no cover - network
    import httpx  # type: ignore

    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left as zeros."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity of a query vector (D,) against gallery (N, D) -> (N,)."""
    q = l2_normalize(query)[0]
    g = l2_normalize(gallery)
    return g @ q


@dataclass
class GalleryHit:
    part: InventoryPart
    score: float  # cosine similarity in [-1, 1], typically [0, 1] for images


class Gallery:
    """Inventory reference embeddings for nearest-neighbour part retrieval."""

    def __init__(self, parts: Sequence[InventoryPart], vectors: np.ndarray):
        if len(parts) != len(vectors):
            raise ValueError("parts and vectors must have the same length")
        self.parts = list(parts)
        self.vectors = l2_normalize(vectors) if len(vectors) else np.zeros((0, 0), np.float32)

    def __len__(self) -> int:
        return len(self.parts)

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[GalleryHit]:
        if len(self) == 0:
            return []
        sims = cosine_similarity(vector, self.vectors)
        order = np.argsort(-sims)[:top_k]
        return [GalleryHit(part=self.parts[i], score=float(sims[i])) for i in order]


class EmbeddingBackend:
    """Interface: turn a list of image byte blobs into an (N, D) float array."""

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def build_gallery(
    parts: Sequence[InventoryPart],
    ref_images: Dict[str, bytes],
    backend: EmbeddingBackend,
) -> Gallery:
    """Embed reference images (keyed by part_num) and build a Gallery.

    Parts without a reference image are dropped (they can't be matched by
    embedding, but Brickognize/colour may still identify them).
    """
    usable: List[InventoryPart] = []
    blobs: List[bytes] = []
    for part in parts:
        img = ref_images.get(part.part_num)
        if img is not None:
            usable.append(part)
            blobs.append(img)
    if not usable:
        return Gallery([], np.zeros((0, 0), np.float32))
    vectors = backend.embed_images(blobs)
    return Gallery(usable, vectors)


class ClipBackend(EmbeddingBackend):  # pragma: no cover - requires torch/open_clip
    """open_clip image encoder. Requires the optional ``[ml]`` extra.

    Lazily imports torch/open_clip so the base install stays light. Encodes
    PNG/JPEG bytes into normalised embeddings on CPU or CUDA.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: Optional[str] = None):
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The embedding backend needs the optional ML deps. "
                "Install with: pip install -e \".[ml]\""
            ) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        import io

        from PIL import Image  # type: ignore

        torch = self._torch
        tensors = []
        for blob in images:
            pil = Image.open(io.BytesIO(blob)).convert("RGB")
            tensors.append(self.preprocess(pil))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
        return feats.cpu().float().numpy()
