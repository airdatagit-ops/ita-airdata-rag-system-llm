"""Tests for the cache backend."""

import time
import threading

import pytest

from search.cache import InMemoryCache, make_cache_key


class TestMakeCacheKey:
    def test_deterministic(self):
        assert make_cache_key("a", "b") == make_cache_key("a", "b")

    def test_different_inputs_differ(self):
        assert make_cache_key("a", "b") != make_cache_key("a", "c")

    def test_returns_hex_string(self):
        key = make_cache_key("x")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestInMemoryCacheBasic:
    def test_get_returns_none_on_miss(self):
        cache = InMemoryCache()
        assert cache.get("missing") is None

    def test_set_and_get(self):
        cache = InMemoryCache()
        cache.set("k", [1, 2, 3])
        assert cache.get("k") == [1, 2, 3]

    def test_overwrite_existing_key(self):
        cache = InMemoryCache()
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"

    def test_delete_existing_key(self):
        cache = InMemoryCache()
        cache.set("k", "v")
        cache.delete("k")
        assert cache.get("k") is None

    def test_delete_missing_key_is_noop(self):
        cache = InMemoryCache()
        cache.delete("nope")

    def test_clear(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestTTL:
    def test_entry_expires_after_ttl(self):
        cache = InMemoryCache()
        cache.set("k", "v", ttl=1)
        assert cache.get("k") == "v"
        time.sleep(1.05)
        assert cache.get("k") is None

    def test_default_ttl_applied(self):
        cache = InMemoryCache(default_ttl=1)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(1.05)
        assert cache.get("k") is None

    def test_per_entry_ttl_overrides_default(self):
        cache = InMemoryCache(default_ttl=0)
        cache.set("k", "v", ttl=3600)
        assert cache.get("k") == "v"

    def test_no_ttl_means_no_expiry(self):
        cache = InMemoryCache()
        cache.set("k", "v")
        assert cache.get("k") == "v"


class TestLRUEviction:
    def test_evicts_oldest_when_full(self):
        cache = InMemoryCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_order(self):
        cache = InMemoryCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")
        cache.set("c", 3)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3


class TestStats:
    def test_tracks_hits_and_misses(self):
        cache = InMemoryCache()
        cache.set("k", "v")
        cache.get("k")
        cache.get("k")
        cache.get("missing")

        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["size"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_clear_resets_stats(self):
        cache = InMemoryCache()
        cache.set("k", "v")
        cache.get("k")
        cache.clear()
        s = cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["size"] == 0

    def test_empty_cache_zero_hit_rate(self):
        s = InMemoryCache().stats()
        assert s["hit_rate"] == 0.0


class TestThreadSafety:
    def test_concurrent_writes(self):
        cache = InMemoryCache(maxsize=500)
        errors = []

        def writer(start: int):
            try:
                for i in range(100):
                    cache.set(f"key-{start + i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n * 100,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.stats()["size"] <= 500
