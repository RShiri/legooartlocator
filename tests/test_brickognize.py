"""Tests for Brickognize response parsing + client (injected transport)."""

import pytest

from legopartlocator.brickognize import BrickognizeClient, BrickognizeError, parse_predictions
from legopartlocator.cache import JSONCache


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


def test_retry_then_succeed_records_backoff_sleeps():
    sleeps = []
    attempts = {"n": 0}

    def transport(image_bytes, filename):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return {"items": [{"id": "3001", "name": "Brick", "score": 0.5}]}

    client = BrickognizeClient(
        transport=transport,
        min_interval=0,
        backoff=0.5,
        sleep=sleeps.append,
    )
    cands = client.predict(b"pngdata")
    assert cands[0].part_num == "3001"
    assert attempts["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_retries_exhausted_raises_brickognize_error():
    def transport(image_bytes, filename):
        raise RuntimeError("always fails")

    client = BrickognizeClient(transport=transport, min_interval=0, sleep=lambda s: None)
    with pytest.raises(BrickognizeError) as exc_info:
        client.predict(b"pngdata")
    assert "after 3 attempts" in str(exc_info.value)


def test_cache_hit_skips_transport(tmp_path):
    cache = JSONCache(tmp_path)
    calls = {"n": 0}

    def transport(image_bytes, filename):
        calls["n"] += 1
        return {"items": [{"id": "3001", "name": "Brick", "score": 0.9}]}

    client = BrickognizeClient(transport=transport, cache=cache, min_interval=0)
    first = client.predict(b"samebytes")
    second = client.predict(b"samebytes")
    assert calls["n"] == 1
    assert first == second


def test_failures_are_not_cached(tmp_path):
    cache = JSONCache(tmp_path)

    def failing_transport(image_bytes, filename):
        raise RuntimeError("nope")

    client = BrickognizeClient(
        transport=failing_transport,
        cache=cache,
        min_interval=0,
        max_retries=1,
        sleep=lambda s: None,
    )
    with pytest.raises(BrickognizeError):
        client.predict(b"samebytes")

    def working_transport(image_bytes, filename):
        return {"items": [{"id": "3001", "name": "Brick", "score": 0.9}]}

    client._transport = working_transport
    cands = client.predict(b"samebytes")
    assert cands[0].part_num == "3001"


def test_throttle_sleeps_remaining_interval():
    sleeps = []
    fake_time = {"t": 0.0}

    def clock():
        return fake_time["t"]

    def transport(image_bytes, filename):
        return {"items": [{"id": "3001", "name": "Brick", "score": 0.9}]}

    client = BrickognizeClient(
        transport=transport,
        min_interval=1.0,
        sleep=sleeps.append,
        clock=clock,
    )
    client.predict(b"first")
    assert sleeps == []  # first call: no prior _last_call, no throttle sleep

    fake_time["t"] = 0.2  # only 0.2s elapsed since last call
    client.predict(b"second")
    assert sleeps == [pytest.approx(0.8)]
