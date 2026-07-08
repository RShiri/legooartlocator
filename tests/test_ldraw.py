"""Tests for the minimal LDraw renderer (numpy-only, synthetic library)."""

import numpy as np
import pytest

from legopartlocator.train.ldraw import LDrawLibrary, render_part


@pytest.fixture()
def lib(tmp_path):
    """A tiny LDraw-shaped library: one primitive quad, one part using it."""
    (tmp_path / "p").mkdir()
    (tmp_path / "parts" / "s").mkdir(parents=True)
    # Primitive: a unit quad in the XZ plane plus one edge line.
    (tmp_path / "p" / "Box1.dat").write_text(
        "0 unit quad\n"
        "4 16 0 0 0  1 0 0  1 0 1  0 0 1\n"
        "2 24 0 0 0  1 0 0\n",
        encoding="utf-8",
    )
    # Part: two references to the primitive -- identity, and translated up
    # (referenced with a backslash + different case, like real files do).
    (tmp_path / "parts" / "9999.dat").write_text(
        "0 test part\n"
        "1 16 0 0 0  10 0 0  0 10 0  0 0 10 box1.dat\n"
        "1 16 0 -10 0  10 0 0  0 10 0  0 0 10 BOX1.DAT\n"
        "3 16 0 0 0  10 0 0  0 -10 0\n",
        encoding="utf-8",
    )
    return LDrawLibrary(tmp_path)


def test_load_resolves_subfiles_recursively_and_case_insensitively(lib):
    tris, edges = lib.load("9999.dat")
    # Each Box1 quad -> 2 triangles; two references + one direct triangle = 5.
    assert tris.shape == (5, 3, 3)
    assert edges.shape == (2, 2, 3)
    # The translated reference actually moved its geometry.
    assert tris[:2].mean() != pytest.approx(tris[2:4].mean())


def test_load_scales_by_transform(lib):
    tris, _ = lib.load("9999.dat")
    # The primitive is a unit quad; the reference scales by 10.
    span = tris[:2].reshape(-1, 3).max(axis=0) - tris[:2].reshape(-1, 3).min(axis=0)
    assert span.max() == pytest.approx(10.0)


def test_unknown_reference_is_empty_not_an_error(lib):
    tris, edges = lib.load("no-such-part.dat")
    assert len(tris) == 0 and len(edges) == 0
    assert render_part(lib, "no-such-part.dat") is None


def test_render_produces_flat_shaded_icon_with_edges(lib):
    img = render_part(lib, "9999.dat", size=96, fill_rgb=(200, 40, 40))
    assert img is not None and img.shape == (96, 96, 3)
    flat = img.reshape(-1, 3)
    is_white = np.all(flat == 255, axis=1)
    is_black = np.all(flat <= 20, axis=1)
    is_fill = (flat[:, 0] > 80) & (flat[:, 0] > flat[:, 1] + 30)  # reddish fill
    assert is_white.any()            # background survives
    assert is_fill.any()             # facets got painted
    assert is_black.any()            # edge strokes drawn
    # Icon look: only a handful of distinct colours (quantized flat shading).
    assert len(np.unique(flat[~is_white], axis=0)) < 24


def test_render_is_deterministic(lib):
    a = render_part(lib, "9999.dat", size=64)
    b = render_part(lib, "9999.dat", size=64)
    assert np.array_equal(a, b)


def test_resolve_part_ref_exact_and_print_fallback(lib):
    from legopartlocator.train.ldraw import resolve_part_ref

    assert resolve_part_ref(lib, "9999") == "9999.dat"
    # A decorated (printed) part falls back to its undecorated mould.
    assert resolve_part_ref(lib, "9999pr0001") == "9999.dat"
    assert resolve_part_ref(lib, "12345") is None


def test_render_training_images_builds_dataset(lib, tmp_path):
    from legopartlocator.models import InventoryPart
    from legopartlocator.train.ldraw import render_training_images

    parts = [
        InventoryPart(part_num="9999", name="Test Part", quantity=1, color_name="Red"),
        InventoryPart(part_num="9999", name="Dup line", quantity=1, color_name="Red"),  # deduped
        InventoryPart(part_num="12345", name="No LDraw model", quantity=1, color_name="Blue"),
    ]
    out = render_training_images(parts, lib.root, views=((35.0, -25.0), (-35.0, -25.0)), size=64)
    assert sorted(out) == ["9999"]          # unresolvable part omitted, dup collapsed
    assert len(out["9999"]) == 2            # one image per view
    for blob in out["9999"]:
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"  # real encoded PNGs
