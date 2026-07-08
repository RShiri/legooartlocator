"""Tests for the namespaced on-disk JSON cache."""

from legopartlocator.cache import JSONCache, content_hash


def test_content_hash_is_deterministic_and_sensitive_to_input():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


def test_get_missing_key_returns_none(tmp_path):
    cache = JSONCache(tmp_path)
    assert cache.get("ns", "missing") is None


def test_set_then_get_round_trips(tmp_path):
    cache = JSONCache(tmp_path)
    cache.set("ns", "key1", {"a": 1, "b": [1, 2, 3]})
    assert cache.get("ns", "key1") == {"a": 1, "b": [1, 2, 3]}


def test_namespaces_are_isolated(tmp_path):
    cache = JSONCache(tmp_path)
    cache.set("ns1", "key", "value1")
    cache.set("ns2", "key", "value2")
    assert cache.get("ns1", "key") == "value1"
    assert cache.get("ns2", "key") == "value2"


def test_corrupt_json_on_disk_is_treated_as_a_cache_miss(tmp_path):
    cache = JSONCache(tmp_path)
    path = cache._path("ns", "key")
    path.write_text("{not valid json", encoding="utf-8")
    assert cache.get("ns", "key") is None


def test_root_directory_is_created_on_construction(tmp_path):
    root = tmp_path / "does" / "not" / "exist" / "yet"
    JSONCache(root)
    assert root.is_dir()


def test_set_overwrites_existing_value(tmp_path):
    cache = JSONCache(tmp_path)
    cache.set("ns", "key", "first")
    cache.set("ns", "key", "second")
    assert cache.get("ns", "key") == "second"
