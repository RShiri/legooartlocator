"""Tests for loading inventories from local CSV/JSON (no API key)."""

import json

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
