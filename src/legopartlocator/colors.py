"""LEGO colour utilities: an RGB-based colour signal for part identification.

Gives the identifier a perceptual colour cue to complement name-token overlap:

  * a curated table of common LEGO colours with their RGB values,
  * dominant-colour extraction from a callout crop (ignoring white page
    background and black part outlines),
  * nearest-colour mapping in perceptual (CIELAB) space, and
  * ``rgb_color_scorer`` — a drop-in replacement for the identifier's
    ``color_scorer`` that compares two colour *names* perceptually.

Distances are ``deltaE76`` (Euclidean in CIELAB); the sRGB->XYZ->Lab
conversion is implemented in numpy so the only hard dependency is numpy
(cv2 is used solely to decode crop bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:  # cv2 is only needed to decode raw image bytes
    import cv2
except Exception:  # pragma: no cover - cv2 is a project dependency
    cv2 = None  # type: ignore

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class LegoColor:
    """A named LEGO colour and its reference RGB triple."""

    id: Optional[int]
    name: str
    rgb: RGB


# Curated table of common LEGO colours. RGB values follow the public
# BrickLink / Rebrickable colour references (with the handful of anchor
# values requested by the pipeline spec, e.g. Black as (33, 33, 33) and
# White as (244, 244, 244) rather than pure #000/#FFF).
LEGO_COLORS: List[LegoColor] = [
    LegoColor(0, "Black", (33, 33, 33)),
    LegoColor(15, "White", (244, 244, 244)),
    LegoColor(4, "Red", (201, 26, 9)),
    LegoColor(1, "Blue", (0, 85, 191)),
    LegoColor(14, "Yellow", (242, 205, 55)),
    LegoColor(2, "Green", (35, 120, 65)),
    LegoColor(72, "Dark Bluish Gray", (100, 110, 104)),
    LegoColor(71, "Light Bluish Gray", (160, 165, 169)),
    LegoColor(7, "Light Gray", (155, 161, 157)),
    LegoColor(8, "Dark Gray", (109, 110, 92)),
    LegoColor(19, "Tan", (228, 205, 158)),
    LegoColor(28, "Dark Tan", (149, 138, 115)),
    LegoColor(70, "Reddish Brown", (88, 42, 18)),
    LegoColor(6, "Brown", (88, 57, 39)),
    LegoColor(25, "Orange", (254, 138, 24)),
    LegoColor(484, "Dark Orange", (160, 95, 53)),
    LegoColor(191, "Bright Light Orange", (248, 187, 61)),
    LegoColor(320, "Dark Red", (114, 0, 18)),
    LegoColor(27, "Lime", (165, 202, 24)),
    LegoColor(288, "Dark Green", (0, 69, 26)),
    LegoColor(10, "Bright Green", (75, 159, 74)),
    LegoColor(48, "Sand Green", (160, 188, 172)),
    LegoColor(330, "Olive Green", (155, 154, 90)),
    LegoColor(322, "Medium Azure", (54, 174, 191)),
    LegoColor(321, "Dark Azure", (7, 139, 201)),
    LegoColor(73, "Medium Blue", (90, 147, 219)),
    LegoColor(9, "Light Blue", (180, 210, 227)),
    LegoColor(63, "Dark Blue", (10, 52, 99)),
    LegoColor(379, "Sand Blue", (96, 116, 161)),
    LegoColor(26, "Magenta", (146, 57, 120)),
    LegoColor(22, "Purple", (129, 0, 123)),
    LegoColor(89, "Dark Purple", (63, 54, 145)),
    LegoColor(13, "Pink", (252, 151, 172)),
    LegoColor(353, "Coral", (255, 105, 143)),
    LegoColor(18, "Nougat", (208, 145, 104)),
    LegoColor(150, "Medium Nougat", (170, 125, 85)),
    LegoColor(226, "Bright Light Yellow", (255, 240, 58)),
    LegoColor(47, "Trans-Clear", (252, 252, 252)),
]


# --- colour conversions ---------------------------------------------------

def hex_to_rgb(hex: str) -> RGB:
    """Parse ``"#RRGGBB"`` (or ``"RRGGBB"``) into an ``(r, g, b)`` int tuple."""
    h = hex.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {hex!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# sRGB -> linear-RGB -> XYZ (D65) matrix.
_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_LAB_EPS = (6.0 / 29.0) ** 3
_LAB_KAPPA = 3.0 * (6.0 / 29.0) ** 2


def _srgb_to_lab(rgb) -> np.ndarray:
    """Convert one sRGB triple (0-255) to CIELAB via the standard D65 path."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = (_RGB2XYZ @ lin) / _WHITE_D65
    f = np.where(xyz > _LAB_EPS, np.cbrt(xyz), xyz / _LAB_KAPPA + 4.0 / 29.0)
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def rgb_distance(a, b) -> float:
    """Perceptual distance (deltaE76): Euclidean distance in CIELAB space."""
    return float(np.linalg.norm(_srgb_to_lab(a) - _srgb_to_lab(b)))


def nearest_lego_color(rgb) -> LegoColor:
    """Return the LEGO_COLORS entry perceptually closest to ``rgb``."""
    return min(LEGO_COLORS, key=lambda c: rgb_distance(rgb, c.rgb))


# --- name lookup ----------------------------------------------------------

def _normalize(name: str) -> str:
    """Lowercase and collapse internal whitespace for tolerant matching."""
    return " ".join(name.lower().split())


_NAME_INDEX = {_normalize(c.name): c.rgb for c in LEGO_COLORS}


def name_to_rgb(name: str) -> Optional[RGB]:
    """Look up a LEGO colour by name (case-insensitive, space-normalised)."""
    if not name:
        return None
    return _NAME_INDEX.get(_normalize(name))


# --- dominant colour of a crop -------------------------------------------

def _to_rgb_array(crop) -> np.ndarray:
    """Coerce PNG/JPEG bytes or an ndarray into an ``(H, W, 3)`` RGB uint8 array.

    Bytes are decoded with ``cv2.imdecode`` (BGR); ndarrays are also treated
    as BGR (as produced by cv2). Either way the channel order is flipped to
    RGB so downstream colour maths matches the LEGO_COLORS table.
    """
    if isinstance(crop, (bytes, bytearray, memoryview)):
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("cv2 is required to decode image bytes")
        buf = np.frombuffer(bytes(crop), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, alpha dropped
        if img is None:
            raise ValueError("could not decode image bytes")
    else:
        img = np.asarray(crop)
    if img.ndim == 2:  # grayscale -> replicate channels (order-agnostic)
        return np.stack([img] * 3, axis=-1).astype(np.uint8)
    if img.shape[-1] == 4:  # drop alpha
        img = img[..., :3]
    return np.ascontiguousarray(img[..., ::-1]).astype(np.uint8)  # BGR -> RGB


def dominant_color(crop, ignore_bg: bool = True) -> RGB:
    """Dominant colour of a callout crop as an ``(r, g, b)`` int tuple.

    When ``ignore_bg`` is set, near-white page background (all channels > 235)
    and near-black outline pixels (all channels < 35) are excluded before the
    dominant colour is computed as the mean of the most-populated quantised
    colour bin. If nothing survives the filter, falls back to the overall
    per-channel median.
    """
    pixels = _to_rgb_array(crop).reshape(-1, 3).astype(np.int32)

    subset = pixels
    if ignore_bg:
        not_white = ~np.all(pixels > 235, axis=1)
        not_black = ~np.all(pixels < 35, axis=1)
        subset = pixels[not_white & not_black]

    if subset.shape[0] == 0:  # everything was background -> overall median
        med = np.rint(np.median(pixels, axis=0)).astype(int)
        return (int(med[0]), int(med[1]), int(med[2]))

    # Most-common colour: quantise to 16-level bins, take the fullest bin,
    # then average the true colours inside it for a stable representative.
    bins = subset >> 4
    codes = (bins[:, 0] << 8) | (bins[:, 1] << 4) | bins[:, 2]
    values, counts = np.unique(codes, return_counts=True)
    top = values[int(np.argmax(counts))]
    rep = np.rint(subset[codes == top].mean(axis=0)).astype(int)
    return (int(rep[0]), int(rep[1]), int(rep[2]))


def crop_color_name(crop) -> str:
    """Name of the nearest LEGO colour to a crop's dominant colour."""
    return nearest_lego_color(dominant_color(crop)).name


# --- colour agreement scorer (drop-in for identify.color_score) -----------

# Lab deltaE at which two colours are considered fully unrelated. ~50 keeps
# only near-identical colours scoring high while still crediting close shades.
COLOR_MATCH_THRESHOLD = 50.0


def rgb_color_scorer(
    seen_color: Optional[str], part, threshold: float = COLOR_MATCH_THRESHOLD
) -> float:
    """Perceptual colour agreement in ``[0, 1]`` between a seen colour and a part.

    Signature-compatible with the identifier's ``color_scorer``. Both the
    ``seen_color`` name and ``part.color_name`` are mapped to RGB via the
    LEGO_COLORS table; if either is unknown the score is ``0.0``. Otherwise the
    score is ``max(0, 1 - deltaE / threshold)`` — identical colours score 1.0
    and clearly different colours score 0.0.
    """
    a = name_to_rgb(seen_color) if seen_color else None
    b = name_to_rgb(getattr(part, "color_name", None))
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - rgb_distance(a, b) / threshold)
