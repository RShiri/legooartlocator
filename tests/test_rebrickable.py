"""Tests for RebrickableClient (injected fake HTTP client, no network)."""

import pytest

from legopartlocator.rebrickable import (
    RebrickableClient,
    RebrickableError,
    _row_to_inventory_part,
    normalize_set_num,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error {self.status_code}")

    def json(self):
        return self._json


class _FakeHttpClient:
    """``responder(url) -> _FakeResponse`` may also raise, to simulate a
    transport-level failure (timeout/connection error) rather than a bad
    status code."""

    def __init__(self, responder):
        self._responder = responder

    def get(self, url, headers=None):
        return self._responder(url)


def _client(responder, api_key="k"):
    return RebrickableClient(api_key=api_key, client=_FakeHttpClient(responder))


def test_normalize_set_num_appends_dash_one():
    assert normalize_set_num("76307") == "76307-1"
    assert normalize_set_num("76307-1") == "76307-1"


def test_no_api_key_raises_rebrickable_error(monkeypatch):
    # A real REBRICKABLE_API_KEY may be present in the environment (e.g. via
    # .env, which cli.py's _load_dotenv() -- exercised by test_cli.py's
    # CliRunner invocations -- loads with os.environ.setdefault and leaves
    # set for the rest of the process). api_key=None alone isn't enough to
    # prove "no key" here; the env var must be cleared too.
    monkeypatch.delenv("REBRICKABLE_API_KEY", raising=False)
    client = RebrickableClient(api_key=None, client=_FakeHttpClient(lambda url: _FakeResponse()))
    with pytest.raises(RebrickableError, match="REBRICKABLE_API_KEY"):
        client.get_set_info("76307")


def test_get_set_info_404_raises_rebrickable_error():
    client = _client(lambda url: _FakeResponse(status_code=404))
    with pytest.raises(RebrickableError, match="not found"):
        client.get_set_info("76307")


def test_get_set_info_other_bad_status_is_wrapped_not_raw():
    """A bad key (401), rate limit (429), or outage (5xx) must raise
    RebrickableError -- not a raw exception -- so cli.py's fallback to
    vision-only actually triggers."""
    client = _client(lambda url: _FakeResponse(status_code=401))
    with pytest.raises(RebrickableError):
        client.get_set_info("76307")


def test_get_set_info_transport_failure_is_wrapped():
    def responder(url):
        raise ConnectionError("boom")

    client = _client(responder)
    with pytest.raises(RebrickableError, match="boom"):
        client.get_set_info("76307")


def test_get_set_parts_follows_pagination_and_merges_results():
    page1 = {
        "results": [{"part": {"part_num": "3001", "name": "Brick"}, "quantity": 2}],
        "next": "https://example/page2",
    }
    page2 = {
        "results": [{"part": {"part_num": "3002", "name": "Plate"}, "quantity": 1}],
        "next": None,
    }

    def responder(url):
        return _FakeResponse(json_data=page2 if "page2" in url else page1)

    parts = _client(responder).get_set_parts("76307")
    assert [p.part_num for p in parts] == ["3001", "3002"]


def test_get_set_parts_404_mid_pagination_raises_rebrickable_error():
    def responder(url):
        return _FakeResponse(status_code=404)

    with pytest.raises(RebrickableError, match="not found"):
        _client(responder).get_set_parts("99999")


def test_get_set_parts_pagination_cap_prevents_infinite_loop():
    """A 'next' link that never terminates must not hang forever."""
    def responder(url):
        return _FakeResponse(json_data={"results": [], "next": "https://example/loop"})

    with pytest.raises(RebrickableError, match="exceeded"):
        _client(responder).get_set_parts("76307", max_pages=3)


def test_row_to_inventory_part_bad_quantity_falls_back_to_zero():
    row = {"part": {"part_num": "3001", "name": "Brick"}, "quantity": "not-a-number"}
    part = _row_to_inventory_part(row)
    assert part.quantity == 0


def test_row_to_inventory_part_good_quantity_parses():
    row = {"part": {"part_num": "3001", "name": "Brick"}, "quantity": 4}
    assert _row_to_inventory_part(row).quantity == 4
