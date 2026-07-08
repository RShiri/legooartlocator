"""Tests for loading inventories from local CSV/JSON (no API key)."""

import json

import pytest

from legopartlocator.inventory import load_inventory_file


def test_load_csv_with_aliased_headers(tmp_path):
    csv_path = tmp_path / "inv.csv"
    csv_path.write_text(
        "Item No,Description,Color,Qty\n"
        "3001,Brick 2 x 4,5,2\n"
        "3020,Plate 2 x 4,0,3\n",
        encoding="utf-8",
    )
    parts = load_inventory_file(csv_path)
    assert len(parts) == 2
    assert parts[0].part_num == "3001"
    assert parts[0].quantity == 2
    assert parts[0].color_id == 5


def test_load_json_flat_list(tmp_path):
    p = tmp_path / "inv.json"
    p.write_text(json.dumps([
        {"part_num": "3001", "name": "Brick 2 x 4", "color_name": "Red", "quantity": 2},
    ]), encoding="utf-8")
    parts = load_inventory_file(p)
    assert parts[0].part_num == "3001"
    assert parts[0].color_name == "Red"


def test_load_rebrickable_nested_json(tmp_path):
    p = tmp_path / "reb.json"
    p.write_text(json.dumps({
        "results": [
            {
                "part": {"part_num": "3001", "name": "Brick 2 x 4", "part_img_url": "http://img/3001.png"},
                "color": {"id": 5, "name": "Red"},
                "quantity": 2,
                "element_id": "300121",
            }
        ]
    }), encoding="utf-8")
    parts = load_inventory_file(p)
    assert parts[0].part_num == "3001"
    assert parts[0].color_name == "Red"
    assert parts[0].element_id == "300121"
    assert parts[0].image_url == "http://img/3001.png"


def test_rows_without_part_num_skipped(tmp_path):
    csv_path = tmp_path / "inv.csv"
    csv_path.write_text("part,name,qty\n,Nameless,1\n3001,Brick,2\n", encoding="utf-8")
    parts = load_inventory_file(csv_path)
    assert len(parts) == 1
    assert parts[0].part_num == "3001"


def test_malformed_quantity_falls_back_to_zero_instead_of_crashing():
    from legopartlocator.inventory import _to_part

    part = _to_part({"part_num": "3001", "name": "Brick", "quantity": "not-a-number"})
    assert part.quantity == 0


def test_infinite_quantity_falls_back_to_zero_not_overflowerror():
    """int(float("Infinity")) raises OverflowError, not ValueError -- must
    still degrade to 0 like any other malformed quantity, not crash."""
    from legopartlocator.inventory import _to_part

    part = _to_part({"part_num": "3001", "name": "Brick", "quantity": "Infinity"})
    assert part.quantity == 0


def test_malformed_color_id_falls_back_to_none():
    from legopartlocator.inventory import _to_part

    part = _to_part({"part_num": "3001", "name": "Brick", "color_id": "not-a-number"})
    assert part.color_id is None


def test_invalid_json_raises_clear_value_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_inventory_file(p)


def test_json_object_without_results_key_raises_clear_error(tmp_path):
    """A dict without a 'results' key used to silently iterate its own keys as
    rows, producing an empty parts list with no error at all."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"part_num": "3001", "quantity": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="results"):
        load_inventory_file(p)


def test_json_non_list_rows_raises_clear_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"results": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON list"):
        load_inventory_file(p)
