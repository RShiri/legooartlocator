"""Tests for Brickognize response parsing + client (injected transport)."""

from legopartlocator.brickognize import BrickognizeClient, parse_predictions


def test_parse_sorts_by_score_and_skips_idless():
    payload = {
        "items": [
            {"id": "3020", "name": "Plate 2 x 4", "score": 0.4},
            {"name": "no id here", "score": 0.99},  # skipped
            {"id": "3001", "name": "Brick 2 x 4", "score": 0.8},
        ]
    }
    cands = parse_predictions(payload)
    assert [c.part_num for c in cands] == ["3001", "3020"]
    assert cands[0].score == 0.8


def test_parse_is_lenient_about_missing_score():
    cands = parse_predictions({"items": [{"id": "3001", "name": "Brick"}]})
    assert cands[0].score == 0.0


def test_client_uses_injected_transport():
    calls = {}

    def transport(image_bytes, filename):
        calls["filename"] = filename
        calls["bytes"] = image_bytes
        return {"items": [{"id": "3001", "name": "Brick 2 x 4", "score": 0.9}]}

    client = BrickognizeClient(transport=transport)
    cands = client.predict(b"pngdata", filename="crop7.png")
    assert cands[0].part_num == "3001"
    assert calls["filename"] == "crop7.png"
    assert calls["bytes"] == b"pngdata"
