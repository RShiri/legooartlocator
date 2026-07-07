"""Assemble the final ScanResult and serialise it to JSON and CSV."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import List, Optional

from .models import BagSegment, LocatedPart, ScanResult


def build_result(
    parts: List[LocatedPart],
    segments: List[BagSegment],
    num_pages: int,
    set_num: Optional[str] = None,
    set_name: Optional[str] = None,
    source_pdf: Optional[str] = None,
    reconciled: bool = False,
    warnings: Optional[List[str]] = None,
) -> ScanResult:
    return ScanResult(
        set_num=set_num,
        set_name=set_name,
        source_pdf=source_pdf,
        num_pages=num_pages,
        reconciled=reconciled,
        bags=segments,
        parts=parts,
        warnings=warnings or [],
    )


def to_json(result: ScanResult, indent: int = 2) -> str:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=indent)


def to_csv(result: ScanResult) -> str:
    """One row per part, with bags/pages flattened to human-readable strings."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "part_num",
            "name",
            "color",
            "element_id",
            "bags",
            "pages_1based",
            "total_seen",
            "inventory_qty",
            "count_matches",
            "confidence",
            "image_url",
        ]
    )
    for p in result.parts:
        writer.writerow(
            [
                p.part_num or "",
                p.name or "",
                p.color_name or "",
                p.element_id or "",
                " ".join(str(b) for b in p.bags),
                " ".join(str(pg + 1) for pg in p.pages),  # 1-based for humans
                p.total_seen,
                "" if p.inventory_qty is None else p.inventory_qty,
                "" if p.count_matches is None else p.count_matches,
                f"{p.confidence:.2f}",
                p.image_url or "",
            ]
        )
    return buf.getvalue()


def write_outputs(result: ScanResult, out_dir: str | Path) -> dict:
    """Write result.json and result.csv into ``out_dir``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "result.json"
    csv_path = out / "result.csv"
    json_path.write_text(to_json(result), encoding="utf-8")
    csv_path.write_text(to_csv(result), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path)}
