"""Tests for cli.py.

Split into two styles: direct tests of plain helper functions (no Click
machinery needed), and CliRunner-based smoke tests that invoke real commands
end-to-end offline (no network, no torch) -- the latter catch argument-wiring
bugs (wrong param order/threading through the local-engine helpers) that
calling inner functions directly never would.
"""

import json

import click
import fitz
import pytest
from click.testing import CliRunner

from legopartlocator.cli import _load_extracts, main


def test_load_extracts_malformed_json_raises_usage_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(click.UsageError, match="not valid JSON"):
        _load_extracts(str(path))


def test_load_extracts_wrong_top_level_shape_raises_usage_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"page_index": 0}), encoding="utf-8")  # dict, not a list
    with pytest.raises(click.UsageError, match="JSON list"):
        _load_extracts(str(path))


def test_load_extracts_invalid_page_extract_raises_usage_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"page_index": "not-an-int"}]), encoding="utf-8")
    with pytest.raises(click.UsageError, match="invalid page extract"):
        _load_extracts(str(path))


def test_load_extracts_valid_file_parses(tmp_path):
    path = tmp_path / "good.json"
    path.write_text(
        json.dumps([{"page_index": 0, "bag_marker": 1, "is_parts_list": False, "callouts": []}]),
        encoding="utf-8",
    )
    extracts = _load_extracts(str(path))
    assert len(extracts) == 1
    assert extracts[0].page_index == 0


# --- CliRunner smoke tests: real command invocations, fully offline --------------

def _make_synthetic_pdf(path) -> None:
    """A one-page PDF with a light-gray rect -- the same 0.86 band the local
    detector's DetectConfig defaults recognise as a callout panel (see
    test_debug_overlay.py's identical pattern, already proven to detect)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(40, 40, 140, 110), color=None, fill=(0.86, 0.86, 0.86), fill_opacity=1)
    doc.save(str(path))
    doc.close()


def _make_inventory_csv(path) -> None:
    path.write_text("part_num,name,color_name,quantity\n3001,Brick 2 x 4,Red,1\n", encoding="utf-8")


def test_scan_local_engine_runs_offline_end_to_end(tmp_path):
    """Exercises the real `scan` command through Click (not the inner
    functions directly) -- this is what catches argument-wiring bugs, like a
    param silently landing in the wrong position when threaded through
    scan -> _run_local -> locate_local."""
    pdf_path = tmp_path / "tiny.pdf"
    _make_synthetic_pdf(pdf_path)
    inv_path = tmp_path / "inv.csv"
    _make_inventory_csv(inv_path)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(main, [
        "scan", str(pdf_path),
        "--engine", "local",
        "--inventory-file", str(inv_path),
        "--no-brickognize",
        "--out", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    assert (out_dir / "result.json").exists()
    assert (out_dir / "result.csv").exists()
    data = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert data["num_pages"] == 1
    assert data["reconciled"] is True


def test_scan_malformed_pages_spec_is_a_clean_usage_error_not_a_traceback(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    _make_synthetic_pdf(pdf_path)
    inv_path = tmp_path / "inv.csv"
    _make_inventory_csv(inv_path)

    result = CliRunner().invoke(main, [
        "scan", str(pdf_path),
        "--engine", "local",
        "--inventory-file", str(inv_path),
        "--no-brickognize",
        "--pages", "50-10",  # reversed range: caught by the pdf_render.py fix
        "--out", str(tmp_path / "out"),
    ])

    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)  # wrapped into a clean UsageError, not a raw traceback
    assert "start > end" in result.output


def test_scan_extracts_respects_pages_filter(tmp_path):
    """Regression test for the --extracts + --pages/--max-pages wiring fix:
    previously --pages was silently ignored for the --extracts path."""
    extracts_path = tmp_path / "extracts.json"
    extracts_path.write_text(json.dumps([
        {"page_index": 0, "bag_marker": 1, "is_parts_list": False, "callouts": []},
        {"page_index": 1, "bag_marker": None, "is_parts_list": False, "callouts": []},
        {"page_index": 2, "bag_marker": 99, "is_parts_list": False, "callouts": []},
    ]), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(main, [
        "scan", "--extracts", str(extracts_path), "--pages", "1",
        "--no-rebrickable", "--out", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    data = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    # Without the fix, page_index 2's bag_marker=99 would also be processed.
    assert [b["bag"] for b in data["bags"]] == [1]


def test_debug_command_runs_offline_end_to_end(tmp_path):
    pdf_path = tmp_path / "tiny.pdf"
    _make_synthetic_pdf(pdf_path)
    out_dir = tmp_path / "debug_out"

    result = CliRunner().invoke(main, ["debug", str(pdf_path), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "stats.json").exists()
    assert (out_dir / "page_000.png").exists()
