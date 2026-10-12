"""Tag-based and version-based invalidation plus dependency fingerprints.

Rules from the dossier's "dependency-based invalidation" technique:

* never mutate entries in place - every determinant of an output lives in the
  key namespace, so a change naturally misses;
* for fine-grained dependencies each entry carries tags (``doc:<id>``,
  ``template:<name>@<version>``, ``model:<id>``) and a re-index or document
  update purges by tag;
* a :class:`DependencyFingerprint` over the retrieved evidence (ids and
  content hashes) is folded into the exact key so a RAG answer misses as soon
  as any source changes, and its tags let ingestion purge it proactively;
* a :class:`VersionRegistry` bumps a namespace version for coarse
  invalidation (many documents changed, verifier upgraded): the bump purges
  every entry tagged ``ns:<namespace>`` from both tiers, and wrapped clients
  sharing the registry reject entries stamped with an older version;
* every invalidation is logged with its cause for cost attribution.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, stable_hash

from .backends import CacheBackend
from .keys import tagged_keys
from .semantic import SemanticCache


class DependencyFingerprint(BaseModel):
    """Hash of the documents an answer was grounded on."""

    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] = Field(default_factory=list)
    content_hashes: dict[str, str] = Field(default_factory=dict, description="document id -> content hash")
    corpus_version: Optional[str] = None
    fingerprint: str

    @classmethod
    def from_evidence(cls, evidence: Sequence[Evidence], *, corpus_version: str | None = None) -> "DependencyFingerprint":
        ids = sorted({e.id for e in evidence})
        hashes = {e.id: stable_hash(e.content, length=16) for e in evidence}
        fp = stable_hash(corpus_version, [(i, hashes[i]) for i in ids], length=24)
        return cls(document_ids=ids, content_hashes={i: hashes[i] for i in ids}, corpus_version=corpus_version, fingerprint=fp)

    @classmethod
    def from_documents(cls, documents: dict[str, str], *, corpus_version: str | None = None) -> "DependencyFingerprint":
        ev = [Evidence(id=k, content=v) for k, v in documents.items()]
        return cls.from_evidence(ev, corpus_version=corpus_version)

    @classmethod
    def empty(cls) -> "DependencyFingerprint":
        return cls(fingerprint=stable_hash(None, [], length=24))

    def tags(self) -> list[str]:
        return tagged_keys(document_ids=self.document_ids)

    def changed_since(self, other: "DependencyFingerprint") -> list[str]:
        """Ids whose content differs or that appeared/disappeared relative to ``other``."""
        ids = set(self.document_ids) | set(other.document_ids)
        return sorted(i for i in ids if self.content_hashes.get(i) != other.content_hashes.get(i))


class InvalidationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: float
    cause: str
    tags: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    removed_exact: int = 0
    removed_semantic: int = 0


class VersionRegistry:
    """Monotonic version per namespace; entries store the version they were written under."""

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}

    def current(self, namespace: str) -> int:
        return self._versions.get(namespace, 1)

    def bump(self, namespace: str) -> int:
        self._versions[namespace] = self.current(namespace) + 1
        return self._versions[namespace]

    def is_current(self, namespace: str, version: int) -> bool:
        return version == self.current(namespace)


class InvalidationManager:
    """Coordinates tag/version invalidation across the exact and semantic tiers."""

    def __init__(
        self,
        backend: CacheBackend,
        semantic: SemanticCache | None = None,
        *,
        versions: VersionRegistry | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.backend = backend
        self.semantic = semantic
        self.versions = versions or VersionRegistry()
        self.clock = clock
        self.log: list[InvalidationEvent] = []

    def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = (), cause: str = "manual") -> InvalidationEvent:
        removed_exact = self.backend.invalidate(tags=tags, keys=keys)
        removed_sem = self.semantic.invalidate(tags=tags, keys=keys) if self.semantic is not None else 0
        ev = InvalidationEvent(at=self.clock(), cause=cause, tags=list(tags), keys=list(keys), removed_exact=removed_exact, removed_semantic=removed_sem)
        self.log.append(ev)
        return ev

    def invalidate_documents(self, document_ids: Iterable[str], *, cause: str = "document_updated") -> InvalidationEvent:
        return self.invalidate(tags=tagged_keys(document_ids=document_ids), cause=cause)

    def invalidate_template(self, name: str, version: str | None = None, *, cause: str = "template_changed") -> InvalidationEvent:
        return self.invalidate(tags=tagged_keys(prompt_template=name, prompt_version=version), cause=cause)

    def invalidate_model(self, model: str, *, cause: str = "model_changed") -> InvalidationEvent:
        return self.invalidate(tags=tagged_keys(model=model), cause=cause)

    def bump_version(self, namespace: str, *, cause: str = "version_bump") -> int:
        """Coarse invalidation of a whole namespace (re-index, verifier upgrade).

        Bumps the namespace version, purges every exact-tier entry tagged
        ``ns:<namespace>`` (wrapped clients tag every entry they write) and
        drops the namespace's semantic entries. Clients that share this
        registry (``CachingModel.wrap``) also stamp new entries with the
        current version and reject entries written under an older one.
        """
        v = self.versions.bump(namespace)
        tag = tagged_keys(namespace=namespace)
        removed_exact = self.backend.invalidate(tags=tag)
        removed_sem = 0
        if self.semantic is not None:
            before = len(self.semantic)
            self.semantic.clear(namespace)
            removed_sem = before - len(self.semantic)
        self.log.append(InvalidationEvent(at=self.clock(), cause=cause, tags=tag, removed_exact=removed_exact, removed_semantic=removed_sem))
        return v

    def on_dependency_change(self, before: DependencyFingerprint, after: DependencyFingerprint) -> InvalidationEvent | None:
        """Purge entries grounded on documents whose content changed between two fingerprints."""
        changed = after.changed_since(before)
        if not changed:
            return None
        return self.invalidate_documents(changed, cause="dependency_changed")

    def causes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ev in self.log:
            out[ev.cause] = out.get(ev.cause, 0) + ev.removed_exact + ev.removed_semantic
        return out


__all__ = ["DependencyFingerprint", "InvalidationEvent", "InvalidationManager", "VersionRegistry"]
