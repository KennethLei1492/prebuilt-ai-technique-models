"""Indexing: a minimal built-in BM25 retriever plus RAPTOR-style summaries.

The package must work in isolation, so :class:`IndexBuilder` turns
``request.inputs["documents"]`` into chunks with a situating prefix
(Anthropic Contextual Retrieval), indexes them with Okapi BM25 (k1=1.2,
b=0.75) and exposes them through :class:`BuiltinRetriever`, which satisfies
the ``pipeline_core.protocols.Retriever`` protocol. When the hybrid retrieval
model is injected the built-in index is only used for documents passed inline
with the request and both result lists are fused.

Hierarchical summaries (:meth:`IndexBuilder.build_hierarchy`) follow RAPTOR
(Sarthi et al. 2024) with the dossier's engineering adaptation: clusters are
seeded by document structure (one summary node per document, one root over
all documents) instead of UMAP + GMM, and summary nodes are stored in the
same index with a ``level`` tag so collapsed-tree retrieval works.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, estimate_tokens, gather_limited, stable_hash, tokenize_words

from .config import PipelineRAGConfig

Summarizer = Callable[[str, str], Awaitable[str]]
"""``await summarizer(title, text) -> summary`` supplied by the model (LLM) or a heuristic."""

Prefixer = Callable[[str, str, str], Awaitable[str]]
"""``await prefixer(title, document_head, chunk) -> prefix`` for LLM contextual prefixes."""


class SourceDocument(BaseModel):
    """A normalised input document."""

    model_config = ConfigDict(extra="allow")

    id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.source or self.metadata.get("title") or self.id


class Chunk(BaseModel):
    """One indexed unit: a passage of a document or a summary node."""

    model_config = ConfigDict(extra="allow")

    id: str
    doc_id: str
    text: str
    source: str = ""
    prefix: str = Field(default="", description="Situating context prepended for indexing (Contextual Retrieval)")
    level: int = Field(default=0, description="0 = leaf passage, 1 = document summary, 2 = corpus summary")
    span: Optional[tuple[int, int]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def indexed_text(self) -> str:
        return f"{self.prefix}\n{self.text}" if self.prefix else self.text

    def to_evidence(self, score: float, rank: int, retriever: str = "sparse") -> Evidence:
        return Evidence(
            id=self.id,
            content=self.text,
            source=self.source or self.doc_id,
            score=score,
            rank=rank,
            retriever=retriever,
            span=self.span,
            metadata={"doc_id": self.doc_id, "level": self.level, "prefix": self.prefix, **self.metadata},
        )


def normalize_documents(raw: Any) -> list[SourceDocument]:
    """Accept strings, dicts (``text``/``content``, ``id``, ``source``, ``metadata``), Evidence or SourceDocument."""
    if raw is None:
        return []
    items: Iterable[Any]
    if isinstance(raw, dict):
        items = [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in raw.items()]
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = [raw]
    docs: list[SourceDocument] = []
    for i, item in enumerate(items, 1):
        if isinstance(item, SourceDocument):
            docs.append(item)
        elif isinstance(item, Evidence):
            docs.append(SourceDocument(id=item.id, text=item.content, source=item.source, metadata=dict(item.metadata)))
        elif isinstance(item, str):
            docs.append(SourceDocument(id=f"doc-{i}", text=item))
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            docs.append(
                SourceDocument(
                    id=str(item.get("id") or f"doc-{i}"),
                    text=str(text),
                    source=str(item.get("source") or item.get("title") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        else:
            text = getattr(item, "text", None) or getattr(item, "content", None) or str(item)
            docs.append(SourceDocument(id=str(getattr(item, "id", f"doc-{i}")), text=str(text), source=str(getattr(item, "source", ""))))
    seen: dict[str, int] = {}
    for d in docs:  # make ids unique without changing the first occurrence
        if d.id in seen:
            seen[d.id] += 1
            d.id = f"{d.id}-{seen[d.id]}"
        else:
            seen[d.id] = 0
    return [d for d in docs if d.text.strip()]


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])|\n{2,}")


def chunk_document(doc: SourceDocument, chunk_size_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """Sentence-aware windows of roughly ``chunk_size_tokens`` with ``overlap_tokens`` of overlap.

    Sentences are packed greedily; when a single sentence exceeds the window it
    is split on word boundaries. Spans are character offsets into ``doc.text``.
    """
    text = doc.text
    if not text.strip():
        return []
    if estimate_tokens(text) <= chunk_size_tokens:
        return [Chunk(id=doc.id, doc_id=doc.id, text=text.strip(), source=doc.source, span=(0, len(text)), metadata=dict(doc.metadata))]

    # sentence boundaries with offsets
    pieces: list[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        end = m.start()
        if end > pos:
            pieces.append((pos, end))
        pos = m.end()
    if pos < len(text):
        pieces.append((pos, len(text)))
    # split oversize sentences on words
    refined: list[tuple[int, int]] = []
    for s, e in pieces:
        if estimate_tokens(text[s:e]) <= chunk_size_tokens:
            refined.append((s, e))
            continue
        start = s
        for wm in re.finditer(r"\S+\s*", text[s:e]):
            if estimate_tokens(text[start : s + wm.end()]) >= chunk_size_tokens:
                refined.append((start, s + wm.end()))
                start = s + wm.end()
        if start < e:
            refined.append((start, e))

    chunks: list[Chunk] = []
    i = 0
    n = 0
    while i < len(refined):
        j = i
        while j < len(refined) and estimate_tokens(text[refined[i][0] : refined[j][1]]) <= chunk_size_tokens:
            j += 1
        j = max(j, i + 1)
        start, end = refined[i][0], refined[j - 1][1]
        n += 1
        chunks.append(Chunk(id=f"{doc.id}#{n}", doc_id=doc.id, text=text[start:end].strip(), source=doc.source, span=(start, end), metadata=dict(doc.metadata)))
        if j >= len(refined):
            break
        # step back for overlap
        k = j
        while k > i + 1 and estimate_tokens(text[refined[k - 1][0] : end]) < overlap_tokens:
            k -= 1
        i = max(k, i + 1)
    return chunks


def heuristic_prefix(doc: SourceDocument, index: int, total: int) -> str:
    """Zero-cost situating prefix: title and position (first sentence as topic hint)."""
    head = re.split(r"(?<=[.!?])\s", doc.text.strip(), maxsplit=1)[0][:160]
    part = f"part {index} of {total}" if total > 1 else "complete document"
    return f"From {doc.title} ({part}). Topic: {head}"


class BM25Index:
    """Okapi BM25 over tokenised chunks. Pure Python, deterministic."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._len: dict[str, int] = {}
        self._avgdl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return tokenize_words(text)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[Chunk]) -> None:
        for ch in chunks:
            if ch.id in self.chunks:
                self._remove(ch.id)
            toks = self.tokenize(ch.indexed_text())
            tf = Counter(toks)
            self.chunks[ch.id] = ch
            self._tf[ch.id] = tf
            self._len[ch.id] = len(toks)
            for t in tf:
                self._df[t] += 1
        self._avgdl = (sum(self._len.values()) / len(self._len)) if self._len else 0.0

    def _remove(self, chunk_id: str) -> None:
        tf = self._tf.pop(chunk_id, None)
        if tf:
            for t in tf:
                self._df[t] -= 1
                if self._df[t] <= 0:
                    del self._df[t]
        self.chunks.pop(chunk_id, None)
        self._len.pop(chunk_id, None)

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], chunk_id: str) -> float:
        tf = self._tf[chunk_id]
        dl = self._len[chunk_id]
        s = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            s += self.idf(t) * f * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 10, *, filters: Optional[dict[str, Any]] = None) -> list[tuple[str, float]]:
        """Top-k ``(chunk_id, score)`` pairs; ties broken by chunk id for determinism."""
        q = self.tokenize(query)
        if not q or not self.chunks:
            return []
        scored: list[tuple[str, float]] = []
        for cid, ch in self.chunks.items():
            if filters and not _matches(ch, filters):
                continue
            s = self.score(q, cid)
            if s > 0:
                scored.append((cid, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def _matches(chunk: Chunk, filters: dict[str, Any]) -> bool:
    for key, want in filters.items():
        have = chunk.metadata.get(key, getattr(chunk, key, None))
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class BuiltinRetriever:
    """Minimal ``Retriever`` over a :class:`BM25Index` (sparse only; dense retrieval lives in hybrid_retrieval)."""

    name = "builtin_bm25"

    def __init__(self, index: Optional[BM25Index] = None) -> None:
        self.index = index or BM25Index()
        self.documents: dict[str, SourceDocument] = {}
        self.llm_prefixed: bool = False
        """Set by the model once :meth:`IndexBuilder.apply_llm_prefixes` has run over this index."""

    def __len__(self) -> int:
        return len(self.index)

    @property
    def version(self) -> str:
        """Stable fingerprint of the indexed content (for cache keys and traces)."""
        return stable_hash(sorted((cid, ch.indexed_text()) for cid, ch in self.index.chunks.items()), length=16)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.index.add(chunks)

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return sorted((c for c in self.index.chunks.values() if c.doc_id == doc_id and c.level == 0), key=lambda c: c.id)

    def leaf_chunks(self) -> list[Chunk]:
        return [c for c in self.index.chunks.values() if c.level == 0]

    def summary_chunks(self) -> list[Chunk]:
        return [c for c in self.index.chunks.values() if c.level > 0]

    def total_tokens(self) -> int:
        return sum(estimate_tokens(c.text) for c in self.leaf_chunks())

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        hits = self.index.search(query, k=k, filters=filters)
        return [self.index.chunks[cid].to_evidence(score, rank, self.name) for rank, (cid, score) in enumerate(hits, 1)]

    def retrieve_sync(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        hits = self.index.search(query, k=k, filters=filters)
        return [self.index.chunks[cid].to_evidence(score, rank, self.name) for rank, (cid, score) in enumerate(hits, 1)]

    def all_evidence(self) -> list[Evidence]:
        return [c.to_evidence(0.0, i, self.name) for i, c in enumerate(sorted(self.leaf_chunks(), key=lambda c: c.id), 1)]


class IndexBuilder:
    """Build a :class:`BuiltinRetriever` from raw documents; optionally add RAPTOR-style summaries."""

    def __init__(self, config: Optional[PipelineRAGConfig] = None) -> None:
        self.config = config or PipelineRAGConfig()

    def build(self, documents: Any) -> BuiltinRetriever:
        """Chunk, prefix (heuristically) and index ``documents``. Synchronous and LLM-free."""
        retriever = BuiltinRetriever()
        self.add(retriever, documents)
        return retriever

    def add(self, retriever: BuiltinRetriever, documents: Any) -> list[Chunk]:
        """Chunk, prefix and add ``documents`` to ``retriever``; returns the chunks added.

        Re-adding a document with the same id and text replaces it in place. A
        *different* document whose id is already taken (typically the
        auto-generated ``doc-N`` of plain strings added in a later call) gets a
        fresh ``doc-N-k`` id instead of silently overwriting the first one.
        """
        docs = normalize_documents(documents)
        all_chunks: list[Chunk] = []
        for doc in docs:
            existing = retriever.documents.get(doc.id)
            if existing is not None and existing.text != doc.text:
                base, n = doc.id, 1
                while f"{base}-{n}" in retriever.documents:
                    n += 1
                doc.id = f"{base}-{n}"
            retriever.documents[doc.id] = doc
            chunks = chunk_document(doc, self.config.chunk_size_tokens, self.config.chunk_overlap_tokens)
            if self.config.contextual_prefix != "none":
                for i, ch in enumerate(chunks, 1):
                    ch.prefix = heuristic_prefix(doc, i, len(chunks))
            all_chunks.extend(chunks)
        retriever.add_chunks(all_chunks)
        if all_chunks:
            retriever.llm_prefixed = False  # new chunks still carry heuristic prefixes
        return all_chunks

    async def apply_llm_prefixes(self, retriever: BuiltinRetriever, prefixer: Prefixer, *, max_chunks: int = 200, concurrency: int = 8) -> int:
        """Replace heuristic prefixes with LLM-generated ones (Contextual Retrieval). Returns chunks updated.

        Chunks that already carry an LLM prefix are skipped, so calling this
        again after more documents were added only pays for the new chunks.
        A prefixer failure keeps that chunk's heuristic prefix. Sets
        ``retriever.llm_prefixed`` once every leaf chunk was attempted.
        """
        pending = [c for c in sorted(retriever.leaf_chunks(), key=lambda c: c.id) if not c.metadata.get("llm_prefix")]
        todo = pending[:max_chunks]

        async def one(ch: Chunk) -> Optional[Chunk]:
            doc = retriever.documents.get(ch.doc_id)
            head = doc.text[:1200] if doc else ""
            prefix = await prefixer(doc.title if doc else ch.doc_id, head, ch.text)
            if not prefix or not prefix.strip():
                return None
            return ch.model_copy(update={"prefix": prefix.strip(), "metadata": {**ch.metadata, "llm_prefix": True}})

        results = await gather_limited([one(c) for c in todo], limit=concurrency)
        updated = [r for r in results if isinstance(r, Chunk)]
        if updated:
            retriever.add_chunks(updated)
        retriever.llm_prefixed = len(pending) <= max_chunks
        return len(updated)

    async def build_hierarchy(self, retriever: BuiltinRetriever, summarizer: Optional[Summarizer] = None, *, max_documents: int = 50) -> list[Chunk]:
        """Add level-1 (per document) and level-2 (corpus) summary nodes; returns the nodes added.

        Without a ``summarizer`` a lead-sentence heuristic is used so the
        collapsed-tree search still has something to match.
        """
        existing = {c.id for c in retriever.summary_chunks()}
        nodes: list[Chunk] = []
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for ch in retriever.leaf_chunks():
            by_doc[ch.doc_id].append(ch)
        for doc_id in sorted(by_doc)[:max_documents]:
            node_id = f"{doc_id}#summary"
            if node_id in existing:
                continue
            doc = retriever.documents.get(doc_id)
            title = doc.title if doc else doc_id
            text = "\n".join(c.text for c in sorted(by_doc[doc_id], key=lambda c: c.id))
            summary = await _summarize(summarizer, title, text)
            nodes.append(Chunk(id=node_id, doc_id=doc_id, text=summary, source=title, level=1, prefix=f"Summary of {title}", metadata={"summary_of": doc_id}))
        if len(by_doc) > 1 and "corpus#summary" not in existing:
            joined = "\n\n".join(n.text for n in nodes) or "\n\n".join(c.text for c in retriever.leaf_chunks()[:20])
            summary = await _summarize(summarizer, "the whole corpus", joined)
            nodes.append(Chunk(id="corpus#summary", doc_id="corpus", text=summary, source="corpus", level=2, prefix="Summary of all documents", metadata={"summary_of": sorted(by_doc)}))
        retriever.add_chunks(nodes)
        return nodes


async def _summarize(summarizer: Optional[Summarizer], title: str, text: str) -> str:
    if summarizer is not None:
        try:
            out = await summarizer(title, text)
            if out and out.strip():
                return out.strip()
        except Exception:  # noqa: BLE001 - fall back to the heuristic summary
            pass
    return lead_summary(title, text)


def lead_summary(title: str, text: str, sentences: int = 4) -> str:
    """Heuristic summary: the first ``sentences`` sentences plus any lines with numbers and units."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    lead = parts[:sentences]
    numeric = [p for p in parts[sentences:] if re.search(r"\d", p)][:2]
    body = " ".join(lead + numeric)
    return f"{title}: {body}" if body else title


__all__ = [
    "BM25Index",
    "BuiltinRetriever",
    "Chunk",
    "IndexBuilder",
    "Prefixer",
    "SourceDocument",
    "Summarizer",
    "chunk_document",
    "heuristic_prefix",
    "lead_summary",
    "normalize_documents",
]
