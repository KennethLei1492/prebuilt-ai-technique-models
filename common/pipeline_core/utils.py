"""Small, dependency-free helpers shared across components.

Everything here is deterministic. Determinism matters because several
components (caching, prompt fingerprinting, prompt-prefix caching) rely on
byte-stable serialisation: the same logical object must always render to the
same bytes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to JSON with sorted keys and no insignificant whitespace.

    Used for cache keys and prompt fingerprints. Non-JSON types fall back to
    ``str()`` so that pydantic models, enums and paths serialise predictably.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "value"):
        return o.value
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def stable_hash(*parts: Any, length: int = 32) -> str:
    """SHA-256 over the canonical JSON of ``parts``; truncated hex digest."""
    h = hashlib.sha256()
    for p in parts:
        h.update(canonical_json(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token for English prose).

    Use ``client.messages.count_tokens`` for billing-grade numbers; this is for
    budgeting, chunk sizing and cache-threshold decisions where a 15% error is
    acceptable.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*")


def tokenize_words(text: str, lowercase: bool = True) -> list[str]:
    """Simple word tokenizer used by lexical components and heuristics."""
    toks = _WORD_RE.findall(text)
    return [t.lower() for t in toks] if lowercase else toks


def normalize_text(text: str) -> str:
    """Whitespace- and case-normalised text for exact-match cache keys."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    *,
    unit: str = "chars",
) -> list[str]:
    """Split text into overlapping windows measured in characters or words.

    Sentence-aware splitting lives in the retrieval component; this is the
    primitive it falls back to.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    if unit == "words":
        words = text.split()
        if not words:
            return []
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("size must be positive")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from synchronous code, even inside a running event loop.

    Inside a running loop (notebooks, async frameworks) a fresh loop is run in a
    worker thread so callers never hit "asyncio.run() cannot be called from a
    running event loop".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            result["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


async def gather_limited(
    coros: Iterable[Awaitable[T]],
    limit: int,
    *,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    """``asyncio.gather`` with a concurrency ceiling."""
    sem = asyncio.Semaphore(max(1, limit))

    async def _wrap(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    return await asyncio.gather(*(_wrap(c) for c in coros), return_exceptions=return_exceptions)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_json_loads(text: str) -> Any:
    """Parse JSON, tolerating code fences and leading/trailing prose."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        candidates = [i for i in (s.find("{"), s.find("[")) if i >= 0]
        if not candidates:
            raise
        start = min(candidates)
        end = max(s.rfind("}"), s.rfind("]"))
        return json.loads(s[start : end + 1])


def dedupe_preserve_order(items: Iterable[T], key: Callable[[T], Any] | None = None) -> list[T]:
    seen: set[Any] = set()
    out: list[T] = []
    for it in items:
        k = key(it) if key else it
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out
