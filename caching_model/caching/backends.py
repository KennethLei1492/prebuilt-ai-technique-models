"""Exact-match cache backends.

All backends share :class:`CacheBackend`, a synchronous, thread-safe
interface over :class:`CacheEntry` records that carry ``created_at``,
``ttl_s``, ``tags``, ``hits``, ``version``, ``size_bytes`` and the measured
``recompute_cost_s`` used by XFetch early refresh.

* :class:`InMemoryBackend` - LRU (default; Belady 1966, vLLM, SGLang), LFU or
  FIFO eviction bounded by **both** ``max_entries`` and ``max_bytes``
  (size-aware eviction: GPTCache's entry-count eviction is a documented OOM
  pitfall).
* :class:`SQLiteBackend` - durable, thread-safe, TTL and tag index.
* :class:`FileBackend` - one JSON file per entry; for simple persistence.
* :class:`RedisBackend` - lazy ``redis`` import, same interface, native TTL.

The clock is injectable so TTL behaviour is unit-testable without sleeping.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import CacheError, canonical_json

from .config import EvictionPolicy

Clock = Callable[[], float]


class CacheEntry(BaseModel):
    """One stored value plus the metadata every tier needs."""

    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any
    namespace: str = ""
    created_at: float
    last_access_at: float
    ttl_s: Optional[float] = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    hits: int = 0
    version: int = 1
    size_bytes: int = 0
    recompute_cost_s: float = Field(default=0.0, ge=0.0, description="Measured latency of producing the value (XFetch delta)")
    negative: bool = Field(default=False, description="Entry records a deterministic failure (negative caching)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def expires_at(self) -> Optional[float]:
        return None if self.ttl_s is None else self.created_at + self.ttl_s

    def is_expired(self, now: float) -> bool:
        exp = self.expires_at()
        return exp is not None and now >= exp

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.created_at)


class BackendStats(BaseModel):
    """Counters a backend maintains."""

    model_config = ConfigDict(extra="forbid")

    entries: int = 0
    bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0
    writes: int = 0


def value_size(value: Any) -> int:
    """Approximate serialised size of a value in bytes."""
    try:
        return len(canonical_json(value).encode("utf-8"))
    except Exception:  # noqa: BLE001 - size is advisory
        return len(str(value).encode("utf-8"))


class CacheBackend(ABC):
    """Synchronous, thread-safe exact-match store."""

    name: str = "backend"

    def __init__(self, *, clock: Clock = time.time) -> None:
        self.clock = clock
        self.stats = BackendStats()
        self._lock = threading.RLock()

    # ------------------------------------------------------------ abstract
    @abstractmethod
    def _load(self, key: str) -> Optional[CacheEntry]: ...

    @abstractmethod
    def _store(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    def _remove(self, key: str) -> bool: ...

    @abstractmethod
    def _keys_for_tag(self, tag: str) -> list[str]: ...

    @abstractmethod
    def keys(self) -> list[str]:
        """All stored keys (including expired ones not yet purged)."""

    @abstractmethod
    def clear(self) -> None: ...

    # -------------------------------------------------------------- public
    def get_entry(self, key: str, *, touch: bool = True) -> Optional[CacheEntry]:
        """Return the live entry for ``key`` or ``None`` (expired entries are removed)."""
        with self._lock:
            entry = self._load(key)
            now = self.clock()
            if entry is None:
                self.stats.misses += 1
                return None
            if entry.is_expired(now):
                self._remove(key)
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            if touch:
                entry.hits += 1
                entry.last_access_at = now
                self._store(entry)
            self.stats.hits += 1
            return entry

    def peek(self, key: str) -> Optional[CacheEntry]:
        """Like :meth:`get_entry` but without touching the entry or counting a hit/miss.

        Expired entries are still removed (and counted as expirations). Used by
        :meth:`caching.CachingModel.execute` to report ``cached_entry_exists``
        without skewing :class:`BackendStats`.
        """
        with self._lock:
            entry = self._load(key)
            if entry is None:
                return None
            if entry.is_expired(self.clock()):
                self._remove(key)
                self.stats.expirations += 1
                return None
            return entry

    def get(self, key: str) -> Optional[Any]:
        entry = self.get_entry(key)
        return None if entry is None else entry.value

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_s: Optional[float] = None,
        tags: Sequence[str] = (),
        namespace: str = "",
        version: int = 1,
        recompute_cost_s: float = 0.0,
        negative: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Optional[CacheEntry]:
        """Store ``value`` under ``key``.

        ``ttl_s=None`` means no expiry. ``ttl_s <= 0`` means "do not retain":
        nothing is written, any existing entry for ``key`` is dropped and
        ``None`` is returned (a caller or an LLM plan saying ``ttl_s=0`` must
        never crash the hot path).
        """
        if ttl_s is not None and ttl_s <= 0:
            with self._lock:
                self._remove(key)
            return None
        now = self.clock()
        entry = CacheEntry(
            key=key,
            value=value,
            namespace=namespace,
            created_at=now,
            last_access_at=now,
            ttl_s=ttl_s,
            tags=list(dict.fromkeys(tags)),
            version=version,
            size_bytes=value_size(value),
            recompute_cost_s=recompute_cost_s,
            negative=negative,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._store(entry)
            self.stats.writes += 1
        return entry

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove(key)

    def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int:
        """Remove entries by tag and/or key; returns the number removed."""
        removed = 0
        with self._lock:
            targets: set[str] = set(keys)
            for tag in tags:
                targets.update(self._keys_for_tag(tag))
            for k in targets:
                if self._remove(k):
                    removed += 1
            self.stats.invalidations += removed
        return removed

    def purge_expired(self) -> int:
        now = self.clock()
        removed = 0
        with self._lock:
            for k in list(self.keys()):
                e = self._load(k)
                if e is not None and e.is_expired(now):
                    self._remove(k)
                    removed += 1
            self.stats.expirations += removed
        return removed

    def __len__(self) -> int:
        return len(self.keys())

    def snapshot_stats(self) -> BackendStats:
        s = self.stats.model_copy()
        s.entries = len(self)
        return s


# --------------------------------------------------------------------------
# In-memory
# --------------------------------------------------------------------------


class InMemoryBackend(CacheBackend):
    """LRU / LFU / FIFO in-process store with entry and byte limits."""

    name = "memory"

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        max_bytes: int = 256 * 1024 * 1024,
        policy: EvictionPolicy = "lru",
        clock: Clock = time.time,
    ) -> None:
        super().__init__(clock=clock)
        if max_entries < 1 or max_bytes < 1:
            raise CacheError("max_entries and max_bytes must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.policy: EvictionPolicy = policy
        self._data: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._tags: dict[str, set[str]] = {}
        self._bytes = 0

    # abstract impl ---------------------------------------------------------
    def _load(self, key: str) -> Optional[CacheEntry]:
        e = self._data.get(key)
        if e is not None and self.policy == "lru":
            self._data.move_to_end(key)
        return e

    def _store(self, entry: CacheEntry) -> None:
        old = self._data.get(entry.key)
        if old is not None:
            self._bytes -= old.size_bytes
            for t in old.tags:
                self._tags.get(t, set()).discard(entry.key)
        self._data[entry.key] = entry
        if self.policy == "lru":
            self._data.move_to_end(entry.key)
        self._bytes += entry.size_bytes
        for t in entry.tags:
            self._tags.setdefault(t, set()).add(entry.key)
        self._evict()

    def _remove(self, key: str) -> bool:
        e = self._data.pop(key, None)
        if e is None:
            return False
        self._bytes -= e.size_bytes
        for t in e.tags:
            s = self._tags.get(t)
            if s is not None:
                s.discard(key)
                if not s:
                    self._tags.pop(t, None)
        return True

    def _keys_for_tag(self, tag: str) -> list[str]:
        return list(self._tags.get(tag, ()))

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._tags.clear()
            self._bytes = 0

    # eviction --------------------------------------------------------------
    def _victim(self) -> Optional[str]:
        if not self._data:
            return None
        if self.policy == "lru":
            return next(iter(self._data))  # least recently used is first
        if self.policy == "fifo":
            return min(self._data.values(), key=lambda e: (e.created_at, e.key)).key
        # lfu: fewest hits, then least recently accessed
        return min(self._data.values(), key=lambda e: (e.hits, e.last_access_at, e.key)).key

    def _evict(self) -> None:
        while self._data and (len(self._data) > self.max_entries or self._bytes > self.max_bytes):
            victim = self._victim()
            if victim is None:
                break
            self._remove(victim)
            self.stats.evictions += 1

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def snapshot_stats(self) -> BackendStats:
        s = super().snapshot_stats()
        s.bytes = self._bytes
        return s


# --------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT '',
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_access_at REAL NOT NULL,
    ttl_s REAL,
    expires_at REAL,
    tags TEXT NOT NULL DEFAULT '[]',
    hits INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    recompute_cost_s REAL NOT NULL DEFAULT 0,
    negative INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS tags (tag TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY (tag, key));
CREATE INDEX IF NOT EXISTS idx_entries_expires ON entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_entries_access ON entries(last_access_at);
"""


class SQLiteBackend(CacheBackend):
    """Durable store; ``path=':memory:'`` for tests. LRU eviction by ``max_entries``."""

    name = "sqlite"

    def __init__(self, path: str | Path = ":memory:", *, max_entries: int = 100_000, clock: Clock = time.time) -> None:
        super().__init__(clock=clock)
        self.path = str(path)
        self.max_entries = max_entries
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _row_to_entry(self, row: tuple[Any, ...]) -> CacheEntry:
        (key, namespace, value, created_at, last_access_at, ttl_s, _exp, tags, hits, version, size_bytes, rc, negative, metadata) = row
        return CacheEntry(
            key=key,
            namespace=namespace,
            value=json.loads(value),
            created_at=created_at,
            last_access_at=last_access_at,
            ttl_s=ttl_s,
            tags=json.loads(tags),
            hits=hits,
            version=version,
            size_bytes=size_bytes,
            recompute_cost_s=rc,
            negative=bool(negative),
            metadata=json.loads(metadata),
        )

    def _load(self, key: str) -> Optional[CacheEntry]:
        cur = self._conn.execute("SELECT * FROM entries WHERE key = ?", (key,))
        row = cur.fetchone()
        return None if row is None else self._row_to_entry(row)

    def _store(self, entry: CacheEntry) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry.key,
                entry.namespace,
                json.dumps(entry.value, ensure_ascii=False, default=str),
                entry.created_at,
                entry.last_access_at,
                entry.ttl_s,
                entry.expires_at(),
                json.dumps(entry.tags),
                entry.hits,
                entry.version,
                entry.size_bytes,
                entry.recompute_cost_s,
                int(entry.negative),
                json.dumps(entry.metadata, default=str),
            ),
        )
        self._conn.execute("DELETE FROM tags WHERE key = ?", (entry.key,))
        self._conn.executemany("INSERT OR IGNORE INTO tags VALUES (?, ?)", [(t, entry.key) for t in entry.tags])
        self._conn.commit()
        self._evict()

    def _remove(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
        self._conn.execute("DELETE FROM tags WHERE key = ?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def _keys_for_tag(self, tag: str) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT key FROM tags WHERE tag = ?", (tag,))]

    def keys(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT key FROM entries")]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.execute("DELETE FROM tags")
            self._conn.commit()

    def _evict(self) -> None:
        n = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        if n <= self.max_entries:
            return
        excess = n - self.max_entries
        victims = [r[0] for r in self._conn.execute("SELECT key FROM entries ORDER BY last_access_at ASC LIMIT ?", (excess,))]
        for k in victims:
            self._remove(k)
            self.stats.evictions += 1

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# File
# --------------------------------------------------------------------------


class FileBackend(CacheBackend):
    """One JSON document per entry under ``directory``; tag index rebuilt on start."""

    name = "file"

    def __init__(self, directory: str | Path, *, clock: Clock = time.time) -> None:
        super().__init__(clock=clock)
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._tags: dict[str, set[str]] = {}
        for p in self.dir.glob("*.json"):
            try:
                e = CacheEntry.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - corrupt files are ignored
                continue
            for t in e.tags:
                self._tags.setdefault(t, set()).add(e.key)

    def _path(self, key: str) -> Path:
        return self.dir / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:40] + ".json")

    def _load(self, key: str) -> Optional[CacheEntry]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return CacheEntry.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise CacheError(f"corrupt cache file {p}: {exc}") from exc

    def _store(self, entry: CacheEntry) -> None:
        path = self._path(entry.key)
        if path.exists():  # re-tagging: forget the tags the previous version carried (mirrors InMemoryBackend)
            try:
                old = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt predecessor has no tags to forget
                old = None
            if old is not None:
                for t in old.tags:
                    s = self._tags.get(t)
                    if s is not None:
                        s.discard(entry.key)
                        if not s:
                            self._tags.pop(t, None)
        path.write_text(entry.model_dump_json(), encoding="utf-8")
        for t in entry.tags:
            self._tags.setdefault(t, set()).add(entry.key)

    def _remove(self, key: str) -> bool:
        p = self._path(key)
        if not p.exists():
            return False
        try:
            e = CacheEntry.model_validate_json(p.read_text(encoding="utf-8"))
            for t in e.tags:
                self._tags.get(t, set()).discard(key)
        except Exception:  # noqa: BLE001
            pass
        p.unlink()
        return True

    def _keys_for_tag(self, tag: str) -> list[str]:
        return list(self._tags.get(tag, ()))

    def keys(self) -> list[str]:
        out: list[str] = []
        for p in self.dir.glob("*.json"):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8"))["key"])
            except Exception:  # noqa: BLE001
                continue
        return out

    def clear(self) -> None:
        with self._lock:
            for p in self.dir.glob("*.json"):
                p.unlink()
            self._tags.clear()


# --------------------------------------------------------------------------
# Redis (lazy import)
# --------------------------------------------------------------------------


class RedisBackend(CacheBackend):
    """Redis-backed store with native TTL and tag sets. Requires ``pip install redis``."""

    name = "redis"

    def __init__(self, url: str = "redis://localhost:6379/0", *, prefix: str = "caching", client: Any = None, clock: Clock = time.time) -> None:
        super().__init__(clock=clock)
        if client is None:
            try:
                import redis  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise CacheError("the 'redis' package is required for RedisBackend (pip install redis)") from exc
            client = redis.Redis.from_url(url)
        self._r = client
        self.prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self.prefix}:e:{key}"

    def _t(self, tag: str) -> str:
        return f"{self.prefix}:t:{tag}"

    def _load(self, key: str) -> Optional[CacheEntry]:
        raw = self._r.get(self._k(key))
        if raw is None:
            return None
        return CacheEntry.model_validate_json(raw)

    def _store(self, entry: CacheEntry) -> None:
        ex = None if entry.ttl_s is None else max(1, int(entry.ttl_s + 1))
        self._r.set(self._k(entry.key), entry.model_dump_json(), ex=ex)
        for t in entry.tags:
            self._r.sadd(self._t(t), entry.key)

    def _remove(self, key: str) -> bool:
        raw = self._r.get(self._k(key))
        if raw is not None:
            try:
                for t in CacheEntry.model_validate_json(raw).tags:
                    self._r.srem(self._t(t), key)
            except Exception:  # noqa: BLE001
                pass
        return bool(self._r.delete(self._k(key)))

    def _keys_for_tag(self, tag: str) -> list[str]:
        return [k.decode() if isinstance(k, bytes) else k for k in self._r.smembers(self._t(tag))]

    def keys(self) -> list[str]:
        plen = len(self._k(""))
        return [(k.decode() if isinstance(k, bytes) else k)[plen:] for k in self._r.scan_iter(match=self._k("*"))]

    def clear(self) -> None:
        for k in self._r.scan_iter(match=f"{self.prefix}:*"):
            self._r.delete(k)


def make_backend(
    kind: str,
    *,
    max_entries: int = 10_000,
    max_bytes: int = 256 * 1024 * 1024,
    policy: EvictionPolicy = "lru",
    sqlite_path: str = ":memory:",
    file_dir: str = ".cache/caching",
    redis_url: str = "redis://localhost:6379/0",
    key_prefix: str = "caching",
    clock: Clock = time.time,
) -> CacheBackend:
    """Factory used by :class:`caching.CachingModel` from :class:`caching.CachingConfig`."""
    if kind == "memory":
        return InMemoryBackend(max_entries=max_entries, max_bytes=max_bytes, policy=policy, clock=clock)
    if kind == "sqlite":
        return SQLiteBackend(sqlite_path, max_entries=max_entries, clock=clock)
    if kind == "file":
        return FileBackend(file_dir, clock=clock)
    if kind == "redis":
        return RedisBackend(redis_url, prefix=key_prefix, clock=clock)
    raise CacheError(f"unknown cache backend '{kind}'")


__all__ = [
    "BackendStats",
    "CacheBackend",
    "CacheEntry",
    "FileBackend",
    "InMemoryBackend",
    "RedisBackend",
    "SQLiteBackend",
    "make_backend",
    "value_size",
]
