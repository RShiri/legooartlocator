"""Tests for assembling and serialising ScanResult (JSON/CSV)."""

import csv
import io
import json

from legopartlocator.aggregate import build_result, to_csv, to_json, write_outputs
from legopartlocator.models import BagSegment, LocatedPart


def _part(**overrides):
    defaults = dict(
        key="3001|Red",
        name="Brick 2 x 4",
        part_num="3001",
        color_name="Red",
        element_id="300121",
        bags=[1],
        pages=[0, 2],  # 0-based internally
        total_seen=2,
        inventory_qty=2,
        reconciled=True,
        count_matches=True,
        confidence=0.9,
        image_url="http://img/3001.png",
    )
    defaults.update(overrides)
    return LocatedPart(**defaults)


def test_build_result_assembles_scan_result():
    parts = [_part()]
    segments = [BagSegment(bag=1, start_page=0, end_page=2)]
    result = build_result(
        parts=parts, segments=segments, num_pages=3,
        set_num="76307", set_name="Iron Man Mech", source_pdf="a.pdf",
        reconciled=True, warnings=["a warning"],
    )
    assert result.set_num == "76307"
    assert result.set_name == "Iron Man Mech"
    assert result.num_pages == 3
    assert result.reconciled is True
    assert result.parts == parts
    assert result.bags == segments
    assert result.warnings == ["a warning"]


def test_build_result_defaults_warnings_to_empty_list():
    result = build_result(parts=[], segments=[], num_pages=0)
    assert result.warnings == []


def test_to_json_round_trips_through_model_dump():
    result = build_result(parts=[_part()], segments=[], num_pages=3, set_num="76307")
    data = json.loads(to_json(result))
    assert data["set_num"] == "76307"
    assert data["parts"][0]["part_num"] == "3001"
    assert data["parts"][0]["pages"] == [0, 2]  # JSON keeps 0-based


def test_to_csv_header_and_1based_pages():
    result = build_result(parts=[_part()], segments=[], num_pages=3)
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    header, row = rows[0], rows[1]
    assert header[0] == "part_num"
    assert row[0] == "3001"
    assert row[4] == "1"  # bags
    assert row[5] == "1 3"  # pages_1based: [0, 2] -> "1 3"
    assert row[6] == "2"  # total_seen
    assert row[9] == "0.90"  # confidence formatted to 2dp


def test_to_csv_blanks_none_inventory_qty_and_count_matches():
    part = _part(inventory_qty=None, count_matches=None, reconciled=False)
    result = build_result(parts=[part], segments=[], num_pages=3)
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    row = rows[1]
    assert row[7] == ""  # inventory_qty
    assert row[8] == ""  # count_matches


def test_to_csv_blanks_missing_optional_text_fields():
    part = _part(part_num=None, name=None, color_name=None, element_id=None, image_url=None)
    result = build_result(parts=[part], segments=[], num_pages=1)
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    row = rows[1]
    assert row[0] == row[1] == row[2] == row[3] == row[10] == ""


def test_to_csv_no_parts_is_header_only():
    result = build_result(parts=[], segments=[], num_pages=0)
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    assert len(rows) == 1


def test_write_outputs_writes_both_files(tmp_path):
    result = build_result(parts=[_part()], segments=[], num_pages=3, set_num="76307")
    paths = write_outputs(result, tmp_path)

    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    assert paths == {"json": str(json_path), "csv": str(csv_path)}
    assert json.loads(json_path.read_text(encoding="utf-8"))["set_num"] == "76307"
    assert "3001" in csv_path.read_text(encoding="utf-8")


def test_write_outputs_creates_out_dir(tmp_path):
    result = build_result(parts=[], segments=[], num_pages=0)
    out_dir = tmp_path / "nested" / "out"
    write_outputs(result, out_dir)
    assert (out_dir / "result.json").exists()
    assert (out_dir / "result.csv").exists()
