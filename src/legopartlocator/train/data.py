"""Training-image collection and augmentation.

Two independent pieces:

  * ``collect_training_images`` -- fetch one reference photo per part_num
    (thin wrapper over ``embedding.download_reference_images``, so it shares
    the exact same dedup-by-part_num behaviour the zero-shot embedding path
    already relies on: the model is trained to recognise *shape*, colour
    invariantly, since colour agreement is already a separate signal in
    ``identify.py``'s ensemble).
  * ``augment`` -- pure numpy/cv2 image transforms, no network, no torch.
    Deterministic given a seed so it's unit-testable offline.

Reference photos are glossy, generously-margined catalog images; the crops
this model will actually see at inference are flat, tightly-cropped
instruction-booklet icons. ``augment`` can't close that gap perfectly, but
narrows it: colour jitter (so colour isn't a shortcut), blur/downsample
(mimics a low-res icon render), a flatten pass (posterize + edge-preserving
smoothing, toward a flat-shaded look), and padding/background composition
(mimics variable crop margins).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..embedding import download_reference_images
from ..models import InventoryPart


def collect_training_images(
    parts: Sequence[InventoryPart], get: Optional[Callable[[str], Optional[bytes]]] = None
) -> Dict[str, bytes]:
    """One reference photo per part_num -- thin wrapper over
    ``embedding.download_reference_images`` (kept as its own name/entry point
    in case training-specific caching or filtering is added later without
    disturbing the inference-path helper)."""
    return download_reference_images(parts, get=get)


def _decode(image: bytes) -> np.ndarray:
    buf = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("augment: could not decode image bytes")
    return img


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover - only fails on a malformed/empty array
        raise ValueError("augment: cv2.imencode failed")
    return buf.tobytes()


def _rotate_and_warp(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-25, 25)
    scale = rng.uniform(0.85, 1.05)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # Small random perspective wobble on top of the rotation, via the
    # translation terms -- enough to vary viewpoint without destroying shape.
    matrix[0, 2] += rng.uniform(-0.03, 0.03) * w
    matrix[1, 2] += rng.uniform(-0.03, 0.03) * h
    return cv2.warpAffine(
        img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )


def _color_jitter(img: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-15, 15)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.6, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _blur_and_resample(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    if rng.random() < 0.7:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    factor = rng.uniform(0.3, 0.7)
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _flatten_shading(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Posterize + edge-preserving smoothing: nudges a glossy 3D render
    toward the flat-shaded look of an instruction-booklet icon."""
    smoothed = cv2.edgePreservingFilter(img, flags=cv2.RECURS_FILTER, sigma_s=40, sigma_r=0.3)
    levels = rng.choice([4, 6, 8])
    quantized = (np.round(smoothed.astype(np.float32) / 255 * levels) / levels * 255).astype(np.uint8)
    return quantized


def _pad_on_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    pad_frac = rng.uniform(0.0, 0.25)
    pad = int(min(h, w) * pad_frac)
    if pad == 0:
        return img
    bg = int(rng.uniform(235, 255))
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), bg, dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = img
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)


def _edge_outline(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Paint a black edge outline (Canny -> optional 1px dilate) onto the image.

    Instruction-booklet icons are flat line art with a dark stroke around the
    part; catalog photos have no such outline. Adding one is the single most
    characteristic step toward the icon look."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo = rng.randint(40, 80)
    edges = cv2.Canny(gray, lo, lo * 2)
    if rng.random() < 0.5:
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    out = img.copy()
    out[edges > 0] = (0, 0, 0)
    return out


def augment(image: bytes, n: int, seed: int = 0, style: str = "icon") -> List[bytes]:
    """Return ``n`` deterministic augmented variants of a reference image.

    ``style="icon"`` (default, measured a net win on real scans) narrows the
    domain gap toward flat instruction-booklet icons: it *always* flattens the
    shading (vs. the photo path's coin-flip) and then adds a black edge outline,
    since real icons are flat line art with a dark stroke rather than glossy 3D
    renders. ``style="photo"`` is the original pipeline, unchanged (kept to
    reproduce the earlier catalog-photo baseline)."""
    if n <= 0:
        return []
    if style not in ("photo", "icon"):
        raise ValueError(f"augment: unknown style {style!r} (expected 'photo' or 'icon')")
    img = _decode(image)
    out: List[bytes] = []
    for i in range(n):
        rng = random.Random(seed * 1_000_003 + i)
        variant = img.copy()
        variant = _rotate_and_warp(variant, rng)
        variant = _color_jitter(variant, rng)
        if style == "icon" or rng.random() < 0.5:
            variant = _flatten_shading(variant, rng)
        variant = _pad_on_background(variant, rng)
        variant = _blur_and_resample(variant, rng)
        if style == "icon":
            variant = _edge_outline(variant, rng)
        out.append(_encode(variant))
    return out


def build_augmented_dataset(
    ref_images: Dict[str, bytes], variants_per_part: int = 8, seed: int = 0, style: str = "icon"
) -> Dict[str, List[bytes]]:
    """Expand one reference image per part into ``variants_per_part`` augmented crops."""
    dataset: Dict[str, List[bytes]] = {}
    for i, (part_num, image) in enumerate(sorted(ref_images.items())):
        dataset[part_num] = augment(image, variants_per_part, seed=seed * 1_000_003 + i, style=style)
    return dataset


def load_real_crops(
    manifest_path,
    min_confidence: float = 0.3,
    require_brickognize: bool = True,
) -> Dict[str, List[bytes]]:
    """Load corroborated real callout crops from a ``scan --dump-crops`` manifest
    as ``{part_num: [png bytes, ...]}`` — in-domain training labels.

    Quality gates (why each exists):
      * ``part_num == raw_part_num`` — a capacity-diverted crop's recorded
        component scores describe the identifier's *raw* winner, not the final
        label, so its corroboration can't vouch for the final label.
      * ``require_brickognize`` — an independent model agreed; embedding-only
        self-labels would just teach the model its own mistakes.
      * ``min_confidence`` — trims the weakest tail. Calibrated on real 76307
        data: corroborated blended confidences run ~0.28-0.67 (median 0.43,
        dragged down by often-zero embedding components), so the default is a
        low 0.3, not a "high-confidence" 0.5+.

    Missing/unreadable crop files are skipped silently (the manifest outlives
    its PNGs when a user cleans the dump directory).
    """
    import json
    from pathlib import Path

    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: Dict[str, List[bytes]] = {}
    for e in entries:
        part_num = e.get("part_num")
        rel = e.get("file")
        if not part_num or not rel:
            continue
        if e.get("raw_part_num") != part_num:
            continue
        if float(e.get("confidence", 0.0)) < min_confidence:
            continue
        if require_brickognize and float(e.get("components", {}).get("brickognize", 0.0)) <= 0.0:
            continue
        path = base / rel
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if data:
            out.setdefault(part_num, []).append(data)
    return out


def merge_datasets(
    base: Dict[str, List[bytes]], extra: Dict[str, List[bytes]]
) -> Dict[str, List[bytes]]:
    """Union of two ``{part_num: [crops]}`` datasets; lists concatenate (base
    first). Non-mutating. Parts only in ``extra`` are included — even a
    single-crop part is useful as a triplet negative."""
    merged: Dict[str, List[bytes]] = {pn: list(crops) for pn, crops in base.items()}
    for pn, crops in extra.items():
        merged.setdefault(pn, [])
        merged[pn] = merged[pn] + list(crops)
    return merged


def train_val_split(
    dataset: Dict[str, List[bytes]], val_frac: float = 0.25, seed: int = 0, min_train: int = 2
) -> "tuple[Dict[str, List[bytes]], Dict[str, List[bytes]]]":
    """Split each part's augmented variants into a train and a held-out val set.

    A part-level split (holding out whole parts) doesn't evaluate the thing
    that actually matters for a retrieval model: whether a *different* view
    of an already-seen part still retrieves correctly. So this splits within
    each part's own variant list instead. ``min_train`` variants are always
    kept for training (a part needs >= 2 to form any triplet at all); any
    part with too few variants to also hold out is skipped from validation
    entirely (still fully used for training).
    """
    train: Dict[str, List[bytes]] = {}
    val: Dict[str, List[bytes]] = {}
    for part_num in sorted(dataset):
        variants = list(dataset[part_num])
        rng = random.Random(f"{seed}:{part_num}")
        rng.shuffle(variants)
        n_val = int(len(variants) * val_frac)
        n_val = min(n_val, len(variants) - min_train)
        if n_val <= 0:
            train[part_num] = variants
            continue
        val[part_num] = variants[:n_val]
        train[part_num] = variants[n_val:]
    return train, val
