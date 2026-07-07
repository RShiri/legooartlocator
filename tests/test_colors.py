"""Tests for the LEGO colour utility (numpy/cv2 only, deterministic)."""

import cv2
import numpy as np

from legopartlocator.colors import (
    LEGO_COLORS,
    crop_color_name,
    dominant_color,
    hex_to_rgb,
    name_to_rgb,
    nearest_lego_color,
    rgb_color_scorer,
    rgb_distance,
)
from legopartlocator.models import InventoryPart


def _red_crop(border: bool = False) -> np.ndarray:
    """60x60 BGR image: white background with a solid red square in the middle."""
    img = np.full((60, 60, 3), 255, dtype=np.uint8)  # white (BGR == RGB here)
    img[15:45, 15:45] = (9, 26, 201)  # red in BGR -> (201, 26, 9) in RGB
    if border:
        cv2.rectangle(img, (15, 15), (44, 44), (0, 0, 0), 3)  # black outline
    return img


def _close(rgb, target, tol: int = 30) -> bool:
    return all(abs(int(c) - int(t)) <= tol for c, t in zip(rgb, target))


# --- nearest colour -------------------------------------------------------

def test_nearest_red():
    assert nearest_lego_color((200, 25, 10)).name == "Red"


def test_nearest_light_bluish_gray():
    assert nearest_lego_color((162, 166, 170)).name == "Light Bluish Gray"


# --- conversions ----------------------------------------------------------

def test_hex_to_rgb():
    assert hex_to_rgb("#C91A09") == (201, 26, 9)
    assert hex_to_rgb("0055BF") == (0, 85, 191)


def test_rgb_distance_identity_is_zero():
    assert rgb_distance((10, 20, 30), (10, 20, 30)) == 0.0
    assert rgb_distance((201, 26, 9), (0, 85, 191)) > 0.0


# --- dominant colour ------------------------------------------------------

def test_dominant_color_ignores_white_background():
    rgb = dominant_color(_red_crop())
    assert _close(rgb, (201, 26, 9))
    # the white background must have been ignored (result is not near-white)
    assert not all(c > 235 for c in rgb)


def test_dominant_color_ignores_black_outline():
    rgb = dominant_color(_red_crop(border=True))
    assert _close(rgb, (201, 26, 9))
    assert not all(c < 35 for c in rgb)


def test_dominant_color_bytes_path():
    ok, buf = cv2.imencode(".png", _red_crop())
    assert ok
    rgb = dominant_color(buf.tobytes())
    assert _close(rgb, (201, 26, 9))


def test_crop_color_name_red_array_and_bytes():
    assert crop_color_name(_red_crop()) == "Red"
    ok, buf = cv2.imencode(".png", _red_crop())
    assert ok
    assert crop_color_name(buf.tobytes()) == "Red"


# --- name lookup ----------------------------------------------------------

def test_name_to_rgb_variants():
    assert name_to_rgb("RED") == (201, 26, 9)
    assert name_to_rgb("light  bluish   gray") == (160, 165, 169)
    assert name_to_rgb("not-a-color") is None


# --- colour scorer --------------------------------------------------------

def test_rgb_color_scorer_matching_color():
    part = InventoryPart(part_num="3001", name="Brick", color_name="Red", quantity=1)
    assert rgb_color_scorer("red", part) > 0.8


def test_rgb_color_scorer_mismatched_color():
    part = InventoryPart(part_num="3001", name="Brick", color_name="Blue", quantity=1)
    assert rgb_color_scorer("red", part) < 0.3


def test_rgb_color_scorer_unmapped_returns_zero():
    part = InventoryPart(part_num="3001", name="Brick", color_name="Red", quantity=1)
    assert rgb_color_scorer("not-a-color", part) == 0.0
    assert rgb_color_scorer(None, part) == 0.0


# --- table sanity ---------------------------------------------------------

def test_table_has_required_colors():
    names = {c.name for c in LEGO_COLORS}
    required = {
        "Black", "White", "Red", "Blue", "Yellow", "Green",
        "Dark Bluish Gray", "Light Bluish Gray", "Tan", "Reddish Brown",
        "Orange", "Dark Red", "Lime", "Dark Green", "Medium Azure",
        "Bright Light Orange", "Dark Tan", "Sand Blue", "Dark Orange",
        "Magenta", "Purple", "Pink", "Trans-Clear",
    }
    assert required <= names
    assert 30 <= len(LEGO_COLORS) <= 45
