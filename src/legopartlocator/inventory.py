"""Inventory sources: the canonical, closed list of parts a set contains.

Both part identifiers (Brickognize and embedding retrieval) are *constrained*
to this list — the correct answer must be one of these parts — which is what
makes identification tractable. Inventories are free:

  * Rebrickable API (see rebrickable.py) — needs a free key,
  * a local CSV/JSON file exported from BrickLink or Rebrickable — no key,
    handled here so the whole matcher can run with zero accounts.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import List, Optional

from .models import InventoryPart


def _norm_key(s: str) -> str:
    """Normalise a column name: lowercase, drop spaces/underscores/punctuation."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

# Accept several common column spellings from BrickLink / Rebrickable exports.
_COLUMN_ALIASES = {
    "part_num": ["part_num", "part", "partnum", "item_no", "itemid", "number", "blitemno"],
    "name": ["name", "part_name", "item_name", "description"],
    "color_id": ["color_id", "colorid", "color", "bl_color_id"],
    "color_name": ["color_name", "colorname", "colour", "color_text"],
    "quantity": ["quantity", "qty", "count", "minqty"],
    "element_id": ["element_id", "elementid", "element", "lego_element_id"],
    "image_url": ["image_url", "img_url", "part_img_url", "image"],
}


def _pick(row: dict, field: str) -> Optional[str]:
    wanted = {_norm_key(a) for a in _COLUMN_ALIASES[field]}
    for key in row:
        if key is None:
            continue
        if _norm_key(key) in wanted:
            val = row[key]
            if val is not None and str(val).strip() != "":
                return str(val).strip()
    return None


def _to_part(row: dict) -> Optional[InventoryPart]:
    part_num = _pick(row, "part_num")
    if not part_num:
        return None
    qty_raw = _pick(row, "quantity")
    color_id_raw = _pick(row, "color_id")
    try:
        quantity = int(float(qty_raw)) if qty_raw is not None else 0
    except (ValueError, TypeError, OverflowError):
        # OverflowError: a literal "inf"/"Infinity" quantity parses as a float
        # fine but can't become an int. Same degrade-to-0 policy as any other
        # malformed value in this real-world-export-tolerant loader.
        quantity = 0
    color_id = None
    if color_id_raw is not None:
        try:
            color_id = int(color_id_raw)
        except (ValueError, TypeError):
            color_id = None
    return InventoryPart(
        part_num=part_num,
        name=_pick(row, "name") or "",
        color_id=color_id,
        color_name=_pick(row, "color_name"),
        quantity=quantity,
        element_id=_pick(row, "element_id"),
        image_url=_pick(row, "image_url"),
    )


def load_inventory_file(path: str | Path) -> List[InventoryPart]:
    """Load an inventory from a local CSV or JSON file (no API key needed).

    JSON may be either a plain list of inventory dicts, or a raw Rebrickable
    parts payload (an object with a ``results`` list). CSV headers are matched
    case-insensitively against common BrickLink/Rebrickable column names.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Inventory file not found: {p}")

    if p.suffix.lower() == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Inventory file {p} is not valid JSON: {exc}") from exc
        if isinstance(data, dict):
            if "results" not in data:
                raise ValueError(
                    f"Inventory file {p} is a JSON object with no 'results' key — expected "
                    "either a plain list of inventory rows or a Rebrickable-style "
                    "{'results': [...]} payload."
                )
            rows = data["results"]
        else:
            rows = data
        if not isinstance(rows, list):
            raise ValueError(
                f"Inventory file {p}: expected a JSON list of inventory rows, "
                f"got {type(rows).__name__}."
            )
        parts: List[InventoryPart] = []
        for row in rows:
            # Support the nested Rebrickable shape as well as flat dicts.
            if isinstance(row, dict) and "part" in row and isinstance(row["part"], dict):
                from .rebrickable import _row_to_inventory_part

                parts.append(_row_to_inventory_part(row))
            else:
                part = _to_part(row)
                if part:
                    parts.append(part)
        return parts

    # CSV / TSV
    with p.open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        parts = [part for row in reader if (part := _to_part(row)) is not None]
    return parts
