"""Semantic (similarity-based) response cache with safety guards.

Design, following the dossier:

* **Pluggable embedder** - the :class:`Embedder` protocol mirrors the shape of
  ``hybrid_retrieval.dense.Embedder`` but is defined locally so this package
  has no dependency on another model. :class:`HashingEmbedder` is the
  deterministic offline baseline (character n-gram and word feature hashing,
  L2-normalised); production deployments inject a real embedder.
* **Conservative threshold** - cosine >= 0.95 by default for general-purpose
  embeddings (GPTCache example distance 0.2; Zhu et al. 2024 show general
  embeddings predict "same answer" at chance level on hard pairs).
* **Exact-match guards** - identifiers, part numbers, versions, quantities and
  numbers extracted from both texts must be identical (``5 mm`` vs ``50 mm``
  must never collide).
* **Context chain** - multi-turn queries carry a hash of the preceding turns
  (MeanCache) so follow-ups never match standalone entries.
* **Per-entry thresholds under an error budget** - a vCache-style online rule
  raises an entry's threshold after an incorrect hit and relaxes it slowly
  while the observed error rate stays under the budget.
* **Namespace isolation** - entries are partitioned by namespace (model,
  template version, corpus snapshot, tenant); matching never crosses them.

When semantic caching is unsafe
-------------------------------
Time-sensitive questions ("current", "today", "latest"), personalised
requests ("my account"), numeric-precision or identifier-bearing engineering
queries, and diversity-requiring generation (brainstorming) must not be
served from similar-but-different entries. :mod:`caching.plan` classifies
these and disables the tier; the guards here are the second line of defence.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .config import KeyNormalization
from .keys import extract_guards, normalize_prompt_text

Clock = Callable[[], float]
_WORD = re.compile(r"[A-Za-z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps texts to equal-length float vectors."""

    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0.0 when either is all zeros)."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class HashingEmbedder:
    """Deterministic feature-hashing embedder (offline baseline).

    Features: lower-cased word unigrams, word bigrams and character 3-5-grams
    with word-boundary padding, hashed with BLAKE2b into ``dimension`` buckets
    with a sign bit; the result is L2-normalised so cosine is well defined.
    Good enough to distinguish near-duplicates from unrelated text; **not** a
    substitute for a domain-tuned model in production (Zhu et al. 2024).
    """

    def __init__(self, dimension: int = 256, *, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        self.dimension = dimension
        self.ngram_range = ngram_range

    def _bucket(self, feature: str) -> tuple[int, float]:
        d = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(d[:4], "big") % self.dimension
        sign = 1.0 if d[4] & 1 else -1.0
        return idx, sign

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        lo, hi = self.ngram_range
        for w in words:
            padded = f" {w} "
            for n in range(lo, hi + 1):
                for i in range(0, max(0, len(padded) - n + 1)):
                    feats.append(f"c{n}:{padded[i : i + n]}")
        return feats

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for f in self._features(text):
            idx, sign = self._bucket(f)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class CallableEmbedder:
    """Adapter for any ``fn(texts) -> vectors`` callable (e.g. a Voyage client)."""

    def __init__(self, fn: Callable[[Sequence[str]], list[list[float]]], dimension: int) -> None:
        self._fn = fn
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._fn(texts)


class SemanticEntry(BaseModel):
    """A cached (text, value) pair with its embedding and learned threshold."""

    model_config = ConfigDict(extra="forbid")

    key: str
    namespace: str
    text: str
    embedding: list[float]
    value: Any
    guards: list[str] = Field(default_factory=list)
    context_key: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: float
    ttl_s: Optional[float] = None
    hits: int = 0
    threshold: float = Field(ge=0.0, le=1.0, description="Per-entry similarity threshold (vCache-style)")
    correct_hits: int = 0
    incorrect_hits: int = 0
    exact_key: Optional[str] = Field(default=None, description="Exact-tier key of the stored response, if any")
    embedder_id: str = Field(default="", description="Embedder that produced ``embedding``; entries from another embedder never match")
    version: int = Field(default=1, description="Namespace version the entry was written under (see caching.VersionRegistry)")

    def is_expired(self, now: float) -> bool:
        return self.ttl_s is not None and now >= self.created_at + self.ttl_s

    def error_rate(self) -> float:
        n = self.correct_hits + self.incorrect_hits
        return self.incorrect_hits / n if n else 0.0


class SemanticHit(BaseModel):
    """Result of a successful lookup."""

    model_config = ConfigDict(extra="forbid")

    entry: SemanticEntry
    similarity: float
    borderline: bool = Field(description="Similarity within the borderline band above the threshold; verify before serving")
    age_s: float


class SemanticMetrics(BaseModel):
    """Counters for hit quality and false-hit risk."""

    model_config = ConfigDict(extra="forbid")

    lookups: int = 0
    hits: int = 0
    borderline_hits: int = 0
    misses: int = 0
    guard_rejections: int = 0
    context_rejections: int = 0
    threshold_rejections: int = 0
    embedder_rejections: int = 0
    feedback_correct: int = 0
    feedback_incorrect: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def observed_false_hit_rate(self) -> float:
        n = self.feedback_correct + self.feedback_incorrect
        return self.feedback_incorrect / n if n else 0.0


class SemanticCache:
    """Per-namespace semantic cache with guards, context chains and adaptive thresholds."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.95,
        top_k: int = 5,
        borderline_band: float = 0.02,
        error_budget: float = 0.02,
        identifier_guards: bool = True,
        max_entries: int = 5_000,
        threshold_step: float = 0.005,
        min_threshold: float = 0.80,
        normalization: KeyNormalization | None = None,
        clock: Clock = time.time,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.top_k = top_k
        self.borderline_band = borderline_band
        self.error_budget = error_budget
        self.identifier_guards = identifier_guards
        self.max_entries = max_entries
        self.threshold_step = threshold_step
        self.min_threshold = max(0.0, min(min_threshold, threshold))
        self.normalization = normalization or KeyNormalization()
        self.clock = clock
        self.metrics = SemanticMetrics()
        self._ns: dict[str, dict[str, SemanticEntry]] = {}

    # ------------------------------------------------------------- helpers
    @property
    def embedder_id(self) -> str:
        return f"{type(self.embedder).__name__}:{getattr(self.embedder, 'dimension', '?')}"

    def _prepare(self, text: str) -> tuple[str, list[float], frozenset[str]]:
        norm = normalize_prompt_text(text, self.normalization)
        return norm, self.embedder.embed([norm])[0], extract_guards(norm)

    def __len__(self) -> int:
        return sum(len(v) for v in self._ns.values())

    def namespaces(self) -> list[str]:
        return sorted(self._ns)

    # ------------------------------------------------------------- lookup
    def lookup(self, text: str, *, namespace: str = "", context_key: str = "") -> Optional[SemanticHit]:
        """Return the best admissible entry for ``text`` or ``None``.

        Admission order: namespace -> not expired -> context chain -> guard
        equality -> top-k by cosine -> per-entry threshold.
        """
        self.metrics.lookups += 1
        bucket = self._ns.get(namespace)
        if not bucket:
            self.metrics.misses += 1
            return None
        norm, vec, guards = self._prepare(text)
        now = self.clock()
        scored: list[tuple[float, SemanticEntry]] = []
        for e in list(bucket.values()):
            if e.is_expired(now):
                bucket.pop(e.key, None)
                continue
            if (e.embedder_id and e.embedder_id != self.embedder_id) or len(e.embedding) != len(vec):
                # written by a different embedder (swap or dimension change): a miss, never a ValueError
                self.metrics.embedder_rejections += 1
                continue
            if e.context_key != context_key:
                self.metrics.context_rejections += 1
                continue
            if self.identifier_guards and frozenset(e.guards) != guards:
                self.metrics.guard_rejections += 1
                continue
            scored.append((cosine(vec, e.embedding), e))
        scored.sort(key=lambda t: -t[0])
        for sim, e in scored[: self.top_k]:
            if sim >= e.threshold:
                borderline = sim < e.threshold + self.borderline_band
                self.metrics.hits += 1
                if borderline:
                    self.metrics.borderline_hits += 1
                return SemanticHit(entry=e, similarity=sim, borderline=borderline, age_s=now - e.created_at)
            self.metrics.threshold_rejections += 1
        self.metrics.misses += 1
        return None

    def confirm_hit(self, entry_key: str) -> bool:
        """Count a served hit on ``entry_key`` (eviction rank).

        :meth:`lookup` does not count hits itself because the middleware may still
        reject the candidate (staleness, verifier); only hits that were actually
        served should raise an entry's SCALM-style value rank.
        """
        for bucket in self._ns.values():
            e = bucket.get(entry_key)
            if e is not None:
                e.hits += 1
                return True
        return False

    # ---------------------------------------------------------------- add
    def add(
        self,
        text: str,
        value: Any,
        *,
        namespace: str = "",
        context_key: str = "",
        tags: Sequence[str] = (),
        ttl_s: Optional[float] = None,
        threshold: Optional[float] = None,
        exact_key: Optional[str] = None,
        version: int = 1,
    ) -> SemanticEntry:
        norm, vec, guards = self._prepare(text)
        key = hashlib.sha256(f"{namespace}|{context_key}|{norm}".encode("utf-8")).hexdigest()[:32]
        entry = SemanticEntry(
            key=key,
            namespace=namespace,
            text=norm,
            embedding=vec,
            value=value,
            guards=sorted(guards),
            context_key=context_key,
            tags=list(dict.fromkeys(tags)),
            created_at=self.clock(),
            ttl_s=ttl_s,
            threshold=self.threshold if threshold is None else threshold,
            exact_key=exact_key,
            embedder_id=self.embedder_id,
            version=version,
        )
        bucket = self._ns.setdefault(namespace, {})
        bucket[key] = entry
        self._evict()
        return entry

    def _evict(self) -> None:
        total = len(self)
        if total <= self.max_entries:
            return
        # value-aware: evict entries that served the fewest hits, oldest first (SCALM-style ranking proxy)
        all_entries = [e for b in self._ns.values() for e in b.values()]
        all_entries.sort(key=lambda e: (e.hits, e.created_at))
        for e in all_entries[: total - self.max_entries]:
            self._ns[e.namespace].pop(e.key, None)
            self.metrics.evictions += 1

    # ----------------------------------------------------------- feedback
    def record_feedback(self, entry_key: str, *, similarity: float, correct: bool) -> Optional[float]:
        """Update the entry's threshold from an observed hit outcome (vCache rule).

        Incorrect hit: raise the threshold just above the offending similarity.
        Correct hit: once at least five outcomes are observed and the error
        rate is under the budget, relax the threshold by one step (never below
        ``min_threshold`` or the global threshold minus 0.1).
        Returns the new threshold or ``None`` if the entry is unknown.
        """
        for bucket in self._ns.values():
            e = bucket.get(entry_key)
            if e is None:
                continue
            if correct:
                e.correct_hits += 1
                self.metrics.feedback_correct += 1
                n = e.correct_hits + e.incorrect_hits
                if n >= 5 and e.error_rate() <= self.error_budget:
                    floor = max(self.min_threshold, self.threshold - 0.1)
                    e.threshold = max(floor, e.threshold - self.threshold_step)
            else:
                e.incorrect_hits += 1
                self.metrics.feedback_incorrect += 1
                e.threshold = min(0.999, max(e.threshold, similarity + self.threshold_step))
            return e.threshold
        return None

    # -------------------------------------------------------- invalidation
    def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = (), namespace: str | None = None) -> int:
        removed = 0
        tagset = set(tags)
        keyset = set(keys)
        for ns, bucket in list(self._ns.items()):
            if namespace is not None and ns != namespace:
                continue
            for k, e in list(bucket.items()):
                if k in keyset or (e.exact_key and e.exact_key in keyset) or (tagset and tagset.intersection(e.tags)):
                    bucket.pop(k, None)
                    removed += 1
            if not bucket:
                self._ns.pop(ns, None)
        return removed

    def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._ns.clear()
        else:
            self._ns.pop(namespace, None)

    def false_hit_risk(self) -> dict[str, float]:
        """Summary used by evaluation: borderline share and observed error rate."""
        m = self.metrics
        return {
            "hit_rate": m.hit_rate,
            "borderline_share": (m.borderline_hits / m.hits) if m.hits else 0.0,
            "observed_false_hit_rate": m.observed_false_hit_rate,
            "guard_rejections": float(m.guard_rejections),
            "error_budget": self.error_budget,
        }


__all__ = [
    "CallableEmbedder",
    "Embedder",
    "HashingEmbedder",
    "SemanticCache",
    "SemanticEntry",
    "SemanticHit",
    "SemanticMetrics",
    "cosine",
]
