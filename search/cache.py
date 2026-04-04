"""Extensible cache backend for the RAG pipeline.

Provides an abstract ``CacheBackend`` and a thread-safe in-memory
implementation with LRU eviction and per-entry TTL.  The interface is
intentionally minimal so that a Redis (or other) backend can be swapped
in without touching consumers.
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional


class CacheBackend(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Return cached value or ``None`` on miss / expiry."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store *value* under *key* with optional TTL in seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* if present (no-op otherwise)."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all entries."""

    @abstractmethod
    def stats(self) -> dict:
        """Return cache statistics (hits, misses, size, …)."""


class InMemoryCache(CacheBackend):
    """Thread-safe in-memory LRU cache with per-entry TTL.

    Parameters
    ----------
    maxsize:
        Maximum number of entries before LRU eviction.
    default_ttl:
        Default time-to-live in seconds (``None`` = no expiry).
    """

    def __init__(
        self,
        maxsize: int = 1024,
        default_ttl: Optional[int] = None,
    ):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, tuple[Any, Optional[float]]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None

            value, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._data[key]
                self._misses += 1
                return None

            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (time.monotonic() + effective_ttl) if effective_ttl else None

        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, expires_at)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


def make_cache_key(*parts: str) -> str:
    """Build a deterministic SHA-256 cache key from string parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()
