"""Documents, chunks, chunking strategies and the document store.

The store is the shared substrate for both retrieval legs: it owns the
documents and their chunks, answers metadata filters (which are pushed into
every channel as an allow-list, never applied after fusion), and persists to
JSON so an index can be rebuilt deterministically.

Chunking follows the dossier's evidence: recursive boundary-aware splitting at
200-400 tokens with zero overlap was the robust choice (Smith 2024; Qu et al.
2024), and chunks lose meaning without their heading, so the title and section
path are kept both as metadata and, optionally, as a contextual header on the
indexed text (Anthropic 2024).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, estimate_tokens

CHARS_PER_TOKEN = 4

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^```")


class Document(BaseModel):
    """A source document before chunking."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict, description="title, source, type, revision, tags, ... used for filters and headers")

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or self.metadata.get("source") or self.id)

    @classmethod
    def coerce(cls, item: Any, *, index: int = 0) -> "Document":
        """Accept a Document, a dict, an :class:`Evidence` or a plain string."""
        if isinstance(item, Document):
            return item
        if isinstance(item, Evidence):
            meta = dict(item.metadata)
            if item.source:
                meta.setdefault("source", item.source)
            return cls(id=item.id, text=item.content, metadata=meta)
        if isinstance(item, str):
            return cls(id=f"doc-{index + 1}", text=item)
        if isinstance(item, dict):
            data = dict(item)
            text = data.pop("text", None) or data.pop("content", None)
            if text is None:
                raise ValueError(f"document {index} has no 'text' or 'content'")
            doc_id = str(data.pop("id", None) or f"doc-{index + 1}")
            meta = dict(data.pop("metadata", {}) or {})
            meta.update(data)  # remaining top-level keys become metadata
            return cls(id=doc_id, text=str(text), metadata=meta)
        raise TypeError(f"unsupported document type: {type(item)!r}")


class Chunk(BaseModel):
    """A retrievable unit derived from a document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    span: tuple[int, int] = Field(default=(0, 0), description="Character offsets inside the source document")
    ordinal: int = 0
    section_path: list[str] = Field(default_factory=list)

    def header(self) -> str:
        """Contextual header: title and section path."""
        parts: list[str] = []
        title = str(self.metadata.get("title") or self.metadata.get("source") or "")
        first = self.section_path[0].lower() if self.section_path else ""
        if title and not (first and (first.startswith(title.lower()) or title.lower().startswith(first))):
            parts.append(title)
        parts.extend(self.section_path)
        return " > ".join(parts)

    def index_text(self, *, prepend_header: bool = True) -> str:
        """Text that is indexed (header + body) as opposed to the display text."""
        hdr = self.header() if prepend_header else ""
        return f"{hdr}\n{self.text}" if hdr else self.text

    def to_evidence(self, *, score: float = 0.0, rank: int | None = None, retriever: str = "") -> Evidence:
        meta = dict(self.metadata)
        meta.update({"chunk_id": self.id, "doc_id": self.doc_id, "ordinal": self.ordinal, "section_path": list(self.section_path)})
        return Evidence(
            id=self.id,
            content=self.text,
            source=str(self.metadata.get("source") or self.metadata.get("title") or self.doc_id),
            score=score,
            rank=rank,
            retriever=retriever,
            metadata=meta,
            span=self.span,
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries and newlines; never returns empty strings."""
    return [s for s in (p.strip() for p in _SENTENCE_RE.split(text)) if s]


def _pack(units: Sequence[str], budget_chars: int, overlap_chars: int, joiner: str = " ") -> list[str]:
    """Greedily pack units into windows of at most ``budget_chars`` with approximate overlap."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        if len(u) > budget_chars:  # oversize unit: flush and hard-split it
            if cur:
                chunks.append(joiner.join(cur))
                cur, cur_len = [], 0
            step = max(1, budget_chars - overlap_chars)
            for i in range(0, len(u), step):
                chunks.append(u[i : i + budget_chars])
                if i + budget_chars >= len(u):
                    break
            continue
        add = len(u) + (len(joiner) if cur else 0)
        if cur and cur_len + add > budget_chars:
            chunks.append(joiner.join(cur))
            if overlap_chars > 0:  # carry trailing units as overlap
                carried: list[str] = []
                carried_len = 0
                for prev in reversed(cur):
                    if carried_len + len(prev) > overlap_chars:
                        break
                    carried.insert(0, prev)
                    carried_len += len(prev) + len(joiner)
                cur, cur_len = carried, sum(len(c) for c in carried) + len(joiner) * max(0, len(carried) - 1)
            else:
                cur, cur_len = [], 0
        cur.append(u)
        cur_len += len(u) + (len(joiner) if len(cur) > 1 else 0)
    if cur:
        chunks.append(joiner.join(cur))
    return chunks


def _locate(text: str, piece: str, start: int) -> tuple[int, int]:
    """Best-effort character span of ``piece`` in ``text`` at or after ``start``."""
    probe = piece[:40]
    i = text.find(probe, start) if probe else -1
    if i < 0:
        i = text.find(probe) if probe else -1
    if i < 0:
        return (start, min(len(text), start + len(piece)))
    return (i, min(len(text), i + len(piece)))


class Chunker:
    """Splits documents into :class:`Chunk` objects.

    Strategies
    ----------
    ``fixed``        character windows of ``size`` tokens with ``overlap`` tokens.
    ``sentence``     sentence-aware greedy packing up to the budget.
    ``structure``    recursive: markdown headings / numbered clauses / fenced code
                     blocks define sections; sections are packed by paragraph then
                     sentence; the section path is recorded.
    ``token_budget`` recursive paragraph -> sentence -> word packing that never
                     exceeds the token budget (estimate: 4 chars/token).
    """

    def __init__(self, strategy: str = "structure", size_tokens: int = 400, overlap_tokens: int = 0, *, token_estimator: Callable[[str], int] = estimate_tokens) -> None:
        if overlap_tokens >= size_tokens:
            raise ValueError("overlap must be smaller than chunk size")
        self.strategy = strategy
        self.size_tokens = size_tokens
        self.overlap_tokens = overlap_tokens
        self.estimate = token_estimator

    @property
    def budget_chars(self) -> int:
        return self.size_tokens * CHARS_PER_TOKEN

    @property
    def overlap_chars(self) -> int:
        return self.overlap_tokens * CHARS_PER_TOKEN

    # ----------------------------------------------------------------- api
    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []
        if self.strategy == "fixed":
            pieces = [(p, []) for p in self._fixed(text)]
        elif self.strategy == "sentence":
            pieces = [(p, []) for p in _pack(split_sentences(text), self.budget_chars, self.overlap_chars)]
        elif self.strategy == "token_budget":
            pieces = [(p, []) for p in self._token_budget(text)]
        elif self.strategy == "structure":
            pieces = self._structure(text)
        else:
            raise ValueError(f"unknown chunk strategy {self.strategy!r}")
        out: list[Chunk] = []
        cursor = 0
        for i, (piece, path) in enumerate(pieces):
            span = _locate(text, piece, cursor)
            cursor = span[0] + 1
            meta = dict(doc.metadata)
            meta.setdefault("title", doc.title)
            out.append(Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=piece, metadata=meta, span=span, ordinal=i, section_path=path))
        return out

    def chunk_many(self, docs: Iterable[Document]) -> list[Chunk]:
        return [c for d in docs for c in self.chunk(d)]

    # ----------------------------------------------------------- strategies
    def _fixed(self, text: str) -> list[str]:
        step = max(1, self.budget_chars - self.overlap_chars)
        out = []
        for i in range(0, len(text), step):
            piece = text[i : i + self.budget_chars].strip()
            if piece:
                out.append(piece)
            if i + self.budget_chars >= len(text):
                break
        return out

    def _token_budget(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: list[str] = []
        for p in paragraphs:
            if self.estimate(p) <= self.size_tokens:
                units.append(p)
                continue
            for s in split_sentences(p):
                if self.estimate(s) <= self.size_tokens:
                    units.append(s)
                else:  # word-level fallback
                    words = s.split()
                    cur: list[str] = []
                    for w in words:
                        cand = " ".join(cur + [w])
                        if cur and self.estimate(cand) > self.size_tokens:
                            units.append(" ".join(cur))
                            cur = [w]
                        else:
                            cur.append(w)
                    if cur:
                        units.append(" ".join(cur))
        return _pack(units, self.budget_chars, self.overlap_chars, joiner="\n")

    def _structure(self, text: str) -> list[tuple[str, list[str]]]:
        sections: list[tuple[list[str], list[str]]] = []  # (path, lines)
        path: list[str] = []
        levels: list[int] = []
        cur_lines: list[str] = []
        in_fence = False

        def flush() -> None:
            nonlocal cur_lines
            if any(l.strip() for l in cur_lines):
                sections.append((list(path), cur_lines))
            cur_lines = []

        for line in text.splitlines():
            if _FENCE_RE.match(line.strip()):
                in_fence = not in_fence
                cur_lines.append(line)
                continue
            if in_fence:
                cur_lines.append(line)
                continue
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                while levels and levels[-1] >= level:
                    levels.pop()
                    path.pop()
                levels.append(level)
                path.append(m.group(2).strip())
                continue
            c = _CLAUSE_RE.match(line.strip())
            if c and len(c.group(1).split(".")) <= 3 and len(c.group(2)) < 120 and (not cur_lines or not cur_lines[-1].strip()):
                # numbered clause preceded by a blank line: start a new section under the same heading path
                flush()
                cur_lines.append(f"{c.group(1)} {c.group(2)}")
                continue
            cur_lines.append(line)
        flush()

        out: list[tuple[str, list[str]]] = []
        for sec_path, lines in sections:
            body = "\n".join(lines).strip()
            if not body:
                continue
            if "```" in body:
                blocks = self._split_keep_fences(body)
            else:
                blocks = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            units: list[str] = []
            for b in blocks:
                if len(b) <= self.budget_chars or b.startswith("```"):
                    units.append(b)
                else:
                    units.extend(split_sentences(b))
            for piece in _pack(units, self.budget_chars, self.overlap_chars, joiner="\n"):
                out.append((piece, list(sec_path)))
        return out

    @staticmethod
    def _split_keep_fences(body: str) -> list[str]:
        blocks: list[str] = []
        buf: list[str] = []
        in_fence = False
        for line in body.splitlines():
            if line.strip().startswith("```"):
                if in_fence:
                    buf.append(line)
                    blocks.append("\n".join(buf))
                    buf = []
                    in_fence = False
                else:
                    if any(l.strip() for l in buf):
                        blocks.extend(p.strip() for p in re.split(r"\n\s*\n", "\n".join(buf)) if p.strip())
                    buf = [line]
                    in_fence = True
                continue
            buf.append(line)
        if buf:
            if in_fence:
                blocks.append("\n".join(buf))
            else:
                blocks.extend(p.strip() for p in re.split(r"\n\s*\n", "\n".join(buf)) if p.strip())
        return blocks


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def match_filter(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Evaluate a filter specification against chunk metadata.

    Supported forms: ``{"field": value}`` (equality, or any-of when ``value`` is a
    list), ``{"field": {"$eq"|"$ne"|"$in"|"$nin"|"$contains"|"$gte"|"$lte"|"$gt"|"$lt"|"$exists": x}}``.
    Comparisons are string-based except the numeric operators.
    """
    if not filters:
        return True
    for field, cond in filters.items():
        present = field in metadata
        val = metadata.get(field)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$exists":
                    if bool(target) != present:
                        return False
                    continue
                if not present:
                    return False
                if op == "$eq" and str(val) != str(target):
                    return False
                if op == "$ne" and str(val) == str(target):
                    return False
                if op == "$in" and str(val) not in {str(t) for t in (target if isinstance(target, (list, tuple, set)) else [target])}:
                    return False
                if op == "$nin" and str(val) in {str(t) for t in (target if isinstance(target, (list, tuple, set)) else [target])}:
                    return False
                if op == "$contains":
                    hay = [str(x).lower() for x in val] if isinstance(val, (list, tuple, set)) else [str(val).lower()]
                    if not any(str(target).lower() in h for h in hay):
                        return False
                if op in ("$gte", "$lte", "$gt", "$lt"):
                    a, b = _num(val), _num(target)
                    if a is None or b is None:
                        return False
                    if op == "$gte" and not a >= b:
                        return False
                    if op == "$lte" and not a <= b:
                        return False
                    if op == "$gt" and not a > b:
                        return False
                    if op == "$lt" and not a < b:
                        return False
            continue
        if not present:
            return False
        if isinstance(cond, (list, tuple, set)):
            if str(val) not in {str(c) for c in cond}:
                return False
        elif isinstance(val, (list, tuple, set)):
            if str(cond) not in {str(v) for v in val}:
                return False
        elif str(val) != str(cond):
            return False
    return True


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DocumentStore:
    """In-memory document and chunk store with metadata filters and JSON persistence."""

    def __init__(self, chunker: Chunker | None = None) -> None:
        self.chunker = chunker or Chunker()
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self._doc_chunks: dict[str, list[str]] = {}
        self.version: int = 0

    # ----------------------------------------------------------------- crud
    def add(self, docs: Iterable[Any]) -> list[Chunk]:
        """Add (or replace by id) documents; returns the new chunks."""
        return self.upsert(docs)[0]

    def upsert(self, docs: Iterable[Any]) -> tuple[list[Chunk], list[str]]:
        """Add (or replace by id) documents; returns ``(new_chunks, removed_chunk_ids)``.

        The removed ids are exactly the chunks dropped by replacing an existing
        document, so callers that mirror the store in other indexes (BM25, dense)
        remove precisely those and can never diverge from the store.
        """
        new: list[Chunk] = []
        removed: list[str] = []
        for item in docs:
            doc = Document.coerce(item, index=self._next_index())
            if doc.id in self.documents:
                removed.extend(self.remove([doc.id]))
            self.documents[doc.id] = doc
            chunks = self.chunker.chunk(doc)
            self._doc_chunks[doc.id] = [c.id for c in chunks]
            for c in chunks:
                self.chunks[c.id] = c
            new.extend(chunks)
        self.version += 1
        return new, removed

    def _next_index(self) -> int:
        """0-based index whose ``doc-{index+1}`` id is free (used for plain-string documents)."""
        n = len(self.documents)
        while f"doc-{n + 1}" in self.documents:
            n += 1
        return n

    def remove(self, doc_ids: Iterable[str]) -> list[str]:
        """Remove documents; returns removed chunk ids."""
        removed: list[str] = []
        for d in doc_ids:
            self.documents.pop(d, None)
            for cid in self._doc_chunks.pop(d, []):
                self.chunks.pop(cid, None)
                removed.append(cid)
        self.version += 1
        return removed

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def __len__(self) -> int:
        return len(self.chunks)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self.chunks

    # -------------------------------------------------------------- filters
    def filter(self, filters: dict[str, Any] | None) -> set[str] | None:
        """Chunk ids whose metadata matches ``filters``; ``None`` means no restriction."""
        if not filters:
            return None
        return {cid for cid, c in self.chunks.items() if match_filter(c.metadata, filters)}

    def metadata_fields(self) -> dict[str, list[str]]:
        """Field -> sample values, used to describe the corpus to the planner."""
        fields: dict[str, set[str]] = {}
        for c in self.chunks.values():
            for k, v in c.metadata.items():
                if k in ("title",) or isinstance(v, (dict,)):
                    continue
                vals = fields.setdefault(k, set())
                if len(vals) < 8:
                    for item in (v if isinstance(v, (list, tuple)) else [v]):
                        vals.add(str(item))
        return {k: sorted(v) for k, v in sorted(fields.items())}

    def total_tokens(self) -> int:
        return sum(estimate_tokens(d.text) for d in self.documents.values())

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "chunker": {"strategy": self.chunker.strategy, "size_tokens": self.chunker.size_tokens, "overlap_tokens": self.chunker.overlap_tokens},
            "documents": [d.model_dump() for d in self.documents.values()],
            "chunks": [c.model_dump() for c in self.chunks.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentStore":
        ck = data.get("chunker", {})
        store = cls(Chunker(ck.get("strategy", "structure"), ck.get("size_tokens", 400), ck.get("overlap_tokens", 0)))
        for d in data.get("documents", []):
            doc = Document.model_validate(d)
            store.documents[doc.id] = doc
            store._doc_chunks[doc.id] = []
        for c in data.get("chunks", []):
            ch = Chunk.model_validate(c)
            store.chunks[ch.id] = ch
            store._doc_chunks.setdefault(ch.doc_id, []).append(ch.id)
        store.version = int(data.get("version", 0))
        return store

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DocumentStore":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
