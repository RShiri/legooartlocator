"""Minimal LDraw .dat renderer: flat-shaded + black edges, instruction-icon style.

Prototype for closing the training-data domain gap: instruction booklets show
parts as flat-shaded CAD renders with dark edge strokes, while our training
images are glossy catalog photos. LDraw (ldraw.org's free parts library)
describes every part as facets + edge lines — exactly the icon's own source
style — so rendering from it produces training images in the target domain.

Deliberately small: no perspective, no BFC, no per-facet colours, no
conditional edges — an orthographic z-buffer rasterizer with quantized flat
shading and depth-tested type-2 edge lines. numpy-only (cv2 not required), so
it is unit-testable in the base test environment.

LDraw format essentials handled here (see ldraw.org/article/218.html):
  type 1  subfile reference: ``1 c x y z a b c d e f g h i file`` (4x3
          transform; resolved recursively through parts/, parts/s/, p/)
  type 2  edge line, two vertices
  type 3  triangle
  type 4  quad (split into two triangles)
Everything else (comments/meta/type 5 conditional edges) is ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class LDrawLibrary:
    """Resolves and parses .dat files from an LDraw library directory tree.

    ``root`` is the directory that contains ``parts/`` and ``p/`` (the
    ``ldraw/`` folder inside complete.zip). Parsed files are memoized —
    primitives like studs are referenced thousands of times.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._search = [self.root / "parts", self.root / "p", self.root]
        self._cache: Dict[str, tuple] = {}
        # Case-insensitive filename index (LDraw refs vary in case).
        self._index: Dict[str, Path] = {}
        for base in self._search:
            if not base.is_dir():
                continue
            for f in base.rglob("*.dat"):
                key = str(f.relative_to(base)).replace("\\", "/").lower()
                self._index.setdefault(key, f)
                self._index.setdefault(f.name.lower(), f)

    def resolve(self, ref: str) -> Optional[Path]:
        return self._index.get(ref.replace("\\", "/").lower())

    def load(self, ref: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (triangles (N,3,3), edges (M,2,3)) for a part, fully resolved."""
        key = ref.replace("\\", "/").lower()
        if key in self._cache:
            return self._cache[key]
        path = self.resolve(ref)
        if path is None:
            empty = (np.zeros((0, 3, 3), np.float32), np.zeros((0, 2, 3), np.float32))
            self._cache[key] = empty
            return empty
        tris: List[np.ndarray] = []
        edges: List[np.ndarray] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            kind = fields[0]
            try:
                if kind == "1" and len(fields) >= 15:
                    x, y, z, a, b, c, d, e, f, g, h, i = map(float, fields[2:14])
                    sub_tris, sub_edges = self.load(" ".join(fields[14:]))
                    rot = np.array([[a, b, c], [d, e, f], [g, h, i]], np.float32)
                    off = np.array([x, y, z], np.float32)
                    if len(sub_tris):
                        tris.append(sub_tris @ rot.T + off)
                    if len(sub_edges):
                        edges.append(sub_edges @ rot.T + off)
                elif kind == "2" and len(fields) >= 8:
                    v = np.array(list(map(float, fields[2:8])), np.float32).reshape(2, 3)
                    edges.append(v[None])
                elif kind == "3" and len(fields) >= 11:
                    v = np.array(list(map(float, fields[2:11])), np.float32).reshape(1, 3, 3)
                    tris.append(v)
                elif kind == "4" and len(fields) >= 14:
                    v = np.array(list(map(float, fields[2:14])), np.float32).reshape(4, 3)
                    tris.append(np.stack([v[[0, 1, 2]], v[[0, 2, 3]]]))
            except ValueError:
                continue  # malformed numeric field: skip the line, keep the part
        out = (
            np.concatenate(tris, axis=0) if tris else np.zeros((0, 3, 3), np.float32),
            np.concatenate(edges, axis=0) if edges else np.zeros((0, 2, 3), np.float32),
        )
        self._cache[key] = out
        return out


def _view_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Instruction-style view: yaw about the vertical, then tilt down.
    LDraw's -Y axis points up, so pitch is applied about X after negating Y."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], np.float32
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        np.float32,
    )
    flip = np.diag([1.0, -1.0, 1.0]).astype(np.float32)  # -Y up -> +Y up
    return rx @ ry @ flip


def render_part(
    library: LDrawLibrary,
    ref: str,
    size: int = 224,
    fill_rgb: Tuple[int, int, int] = (180, 30, 30),
    yaw_deg: float = 35.0,
    pitch_deg: float = -25.0,
    shade_levels: int = 3,
    margin_frac: float = 0.12,
) -> Optional[np.ndarray]:
    """Render one part to an (size, size, 3) uint8 RGB icon on white.

    Returns ``None`` when the part resolves to no geometry (unknown ref)."""
    tris, edges = library.load(ref)
    if len(tris) == 0:
        return None

    view = _view_matrix(yaw_deg, pitch_deg)
    vt = tris.reshape(-1, 3) @ view.T
    ve = edges.reshape(-1, 3) @ view.T if len(edges) else np.zeros((0, 3), np.float32)

    # Fit to canvas: x right, y down (image coords), z toward the viewer.
    allv = np.concatenate([vt, ve]) if len(ve) else vt
    lo, hi = allv[:, :2].min(axis=0), allv[:, :2].max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    scale = size * (1 - 2 * margin_frac) / span
    center = (lo + hi) / 2
    def to_px(v):
        out = np.empty_like(v)
        out[:, 0] = (v[:, 0] - center[0]) * scale + size / 2
        out[:, 1] = (v[:, 1] - center[1]) * scale + size / 2
        out[:, 2] = v[:, 2] * scale
        return out

    pt = to_px(vt).reshape(-1, 3, 3)
    pe = to_px(ve).reshape(-1, 2, 3) if len(ve) else ve.reshape(0, 2, 3)

    img = np.full((size, size, 3), 255, np.uint8)
    zbuf = np.full((size, size), -np.inf, np.float32)
    light = np.array([0.35, -0.5, 0.79], np.float32)
    light /= np.linalg.norm(light)
    fill = np.array(fill_rgb, np.float32)

    for tri in pt:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        minx, maxx = int(max(0, min(x0, x1, x2))), int(min(size - 1, max(x0, x1, x2)) + 1)
        miny, maxy = int(max(0, min(y0, y1, y2))), int(min(size - 1, max(y0, y1, y2)) + 1)
        if minx >= maxx or miny >= maxy:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        xs, ys = np.meshgrid(
            np.arange(minx, maxx, dtype=np.float32) + 0.5,
            np.arange(miny, maxy, dtype=np.float32) + 0.5,
        )
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * z0 + w1 * z1 + w2 * z2
        region = zbuf[miny:maxy, minx:maxx]
        visible = inside & (z > region)
        if not visible.any():
            continue
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        shade = abs(float(n @ light) / norm) if norm > 1e-9 else 1.0
        # Quantize to a few flat levels -- the icon look, not smooth 3D shading.
        shade = 0.55 + 0.45 * (round(shade * (shade_levels - 1)) / (shade_levels - 1))
        color = np.clip(fill * shade, 0, 255).astype(np.uint8)
        region[visible] = z[visible]
        img[miny:maxy, minx:maxx][visible] = color

    # Depth-tested edge strokes: sample along each segment, draw where visible.
    eps = max(2.0, 0.01 * size)
    for (p0, p1) in pe:
        steps = int(max(2, np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
        pts = p0[None] + ts * (p1 - p0)[None]
        xs = np.clip(pts[:, 0].round().astype(int), 0, size - 1)
        ys = np.clip(pts[:, 1].round().astype(int), 0, size - 1)
        vis = pts[:, 2] >= zbuf[ys, xs] - eps
        img[ys[vis], xs[vis]] = (0, 0, 0)

    return img


def resolve_part_ref(library: LDrawLibrary, part_num: str) -> Optional[str]:
    """Map a Rebrickable part_num to an LDraw .dat reference.

    Tries the exact number first; a decorated part like ``3068bpr9329`` (print
    suffix) falls back to its undecorated mould (``3068b``) — the print isn't
    in the geometry library, but the mould's shape is still the right icon
    silhouette to train on."""
    import re

    candidates = [part_num]
    m = re.match(r"^(\d+[a-z]*?)(?:pr|pat|px)\w*$", part_num)
    if m:
        candidates.append(m.group(1))
    for cand in candidates:
        if library.resolve(f"{cand}.dat") is not None:
            return f"{cand}.dat"
    return None


# Three instruction-plausible viewpoints (yaw, pitch): the standard from-above
# three-quarter view, its mirror, and a steeper look-down.
DEFAULT_VIEWS: Tuple[Tuple[float, float], ...] = ((35.0, -25.0), (-35.0, -25.0), (20.0, -40.0))


def render_training_images(
    parts: Sequence,
    lib_root: str | Path,
    views: Sequence[Tuple[float, float]] = DEFAULT_VIEWS,
    size: int = 224,
    only_parts: Optional[set] = None,
) -> Dict[str, List[bytes]]:
    """Render icon-style PNG training images for inventory parts with an LDraw
    model: ``{part_num: [png bytes per view]}``.

    ``parts`` is a sequence of ``InventoryPart``-shaped objects (``part_num``,
    ``color_name``). Fill colour comes from the part's colour name via
    ``colors.name_to_rgb`` (mid-gray fallback). Parts with no resolvable .dat
    are simply omitted — the caller merges these renders with catalog-photo
    variants, so such parts keep their photo-only training data.

    ``only_parts``, when given, restricts rendering to that set of part
    numbers — every other part is skipped entirely, keeping its existing
    catalog-photo training data completely untouched. Whole-inventory LDraw
    (the default, ``only_parts=None``) measurably fixed one identification-miss
    part on 76307 but also *regressed* two parts that catalog photos already
    handled correctly — the working theory is that LDraw's flat, textureless
    fill can pull a plain-coloured part's embedding toward other
    similarly-shaped, similarly-flat-coloured parts it wasn't confused with
    before. Targeting only the parts that actually need it (from a
    ``scan --dump-crops`` manifest's unidentified list) is the way to get
    LDraw's proven benefit without that side effect.
    """
    import cv2  # local: keep module importable without OpenCV for the pure paths

    from ..colors import name_to_rgb

    library = LDrawLibrary(lib_root)
    out: Dict[str, List[bytes]] = {}
    for part in parts:
        part_num = part.part_num
        if not part_num or part_num in out:
            continue
        if only_parts is not None and part_num not in only_parts:
            continue
        ref = resolve_part_ref(library, part_num)
        if ref is None:
            continue
        rgb = name_to_rgb(part.color_name or "") or (150, 150, 150)
        images: List[bytes] = []
        for yaw, pitch in views:
            img = render_part(library, ref, size=size, fill_rgb=tuple(rgb), yaw_deg=yaw, pitch_deg=pitch)
            if img is None:
                break
            ok, buf = cv2.imencode(".png", img[:, :, ::-1])  # RGB -> BGR for cv2
            if ok:
                images.append(buf.tobytes())
        if images:
            out[part_num] = images
    return out
