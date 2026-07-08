"""Tests for VisionExtractor's API-calling internals (injected fake Anthropic client).

Mirrors test_brickognize.py's style: a fake transport/client that can be made
to fail N times before succeeding, so the retry/backoff logic is exercised
without a real network call or SDK.
"""

import pytest

from legopartlocator.cache import JSONCache
from legopartlocator.pdf_render import RenderedPage
from legopartlocator.vision import VisionError, VisionExtractor


class _Block:
    def __init__(self, type_, name=None, input_=None):
        self.type = type_
        self.name = name
        self.input = input_ or {}


class _Message:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, responder):
        self._responder = responder

    def create(self, **kwargs):
        return self._responder(**kwargs)


class _FakeClient:
    def __init__(self, responder):
        self.messages = _FakeMessages(responder)


def _tool_use(tool_name, data):
    return _Message([_Block("tool_use", name=tool_name, input_=data)])


_EMPTY_PAGE = {"bag_marker": None, "is_parts_list": False, "callouts": []}


def _extractor(client, cache_dir, **kwargs):
    return VisionExtractor(cache=JSONCache(cache_dir), client=client, **kwargs)


def test_call_model_retries_then_succeeds_records_backoff_sleeps(tmp_path):
    sleeps = []
    attempts = {"n": 0}

    def responder(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return _tool_use("record_page", _EMPTY_PAGE)

    extractor = _extractor(_FakeClient(responder), tmp_path, backoff=0.5, sleep=sleeps.append)
    page = RenderedPage(page_index=0, png_bytes=b"png", text="")

    result = extractor.extract_page(page, use_cache=False)

    assert result.page_index == 0
    assert attempts["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_retries_exhausted_raises_vision_error(tmp_path):
    def responder(**kwargs):
        raise RuntimeError("always fails")

    extractor = _extractor(_FakeClient(responder), tmp_path, max_retries=2, sleep=lambda s: None)
    page = RenderedPage(page_index=0, png_bytes=b"png", text="")

    with pytest.raises(VisionError) as exc_info:
        extractor.extract_page(page, use_cache=False)
    assert "2 attempts" in str(exc_info.value)


def test_successful_first_attempt_never_sleeps(tmp_path):
    sleeps = []
    extractor = _extractor(
        _FakeClient(lambda **kw: _tool_use("record_page", _EMPTY_PAGE)),
        tmp_path,
        sleep=sleeps.append,
    )
    page = RenderedPage(page_index=0, png_bytes=b"png", text="")

    extractor.extract_page(page, use_cache=False)

    assert sleeps == []


def test_cache_hit_never_calls_the_client(tmp_path):
    calls = {"n": 0}

    def responder(**kwargs):
        calls["n"] += 1
        return _tool_use("record_page", _EMPTY_PAGE)

    cache = JSONCache(tmp_path)
    extractor = VisionExtractor(cache=cache, client=_FakeClient(responder))
    page = RenderedPage(page_index=0, png_bytes=b"png", text="")

    extractor.extract_page(page, use_cache=True)
    extractor.extract_page(page, use_cache=True)

    assert calls["n"] == 1  # second call was a cache hit


def test_missing_tool_use_block_raises_vision_error_without_retrying(tmp_path):
    """A response with no matching tool_use block is a schema problem, not a
    transient failure -- it must not be retried max_retries times."""
    attempts = {"n": 0}

    def responder(**kwargs):
        attempts["n"] += 1
        return _Message([])  # no tool_use block at all

    extractor = _extractor(_FakeClient(responder), tmp_path, sleep=lambda s: None)
    page = RenderedPage(page_index=0, png_bytes=b"png", text="")

    with pytest.raises(VisionError, match="did not return the expected"):
        extractor.extract_page(page, use_cache=False)
    assert attempts["n"] == 1


def test_triage_page_also_retries(tmp_path):
    sleeps = []
    attempts = {"n": 0}

    def responder(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("boom")
        return _tool_use("triage_page", {"bag_marker": None, "is_parts_list": False, "has_callouts": False})

    extractor = _extractor(_FakeClient(responder), tmp_path, backoff=0.1, sleep=sleeps.append)
    page = RenderedPage(page_index=0, png_bytes=b"png", text="")

    result = extractor.triage_page(page, use_cache=False)

    assert result["has_callouts"] is False
    assert sleeps == [0.1]
