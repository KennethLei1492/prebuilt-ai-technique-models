"""Dense retrieval: embedder protocol, offline baseline embedder and a brute-force index.

The dense leg exists for vocabulary mismatch and paraphrase (Karpukhin et al.
2020) and is always fused with the lexical leg because single-vector models
underperform BM25 out of domain (Thakur et al. 2021). Production deployments
plug in Voyage or a sentence-transformers model; tests and demos use the
deterministic :class:`HashingEmbedder`, which is a *lexical-overlap* baseline
and is documented as such.

Embedding models trained with asymmetric prefixes (E5 ``query:``/``passage:``,
BGE query instructions) degrade without them (Wang et al. 2022; Xiao et al.
2024). The :class:`Embedder` protocol therefore requires only the spec's
``embed(texts)`` and ``dimension``; ``embed_queries(texts)`` is an *optional*
hook that the built-in embedders implement and :meth:`DenseIndex.embed_query`
uses when present, falling back to ``embed`` otherwise, so a minimal
third-party embedder works unchanged.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

try:  # numpy is optional; pure Python fallback below
    import numpy as _np
except Exception:  # pragma: no cover - exercised only without numpy
    _np = None

Vector = list[float]
ScoredId = tuple[str, float]
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


@runtime_checkable
class Embedder(Protocol):
    """Structural interface for embedding backends (ARCHITECTURE 4.1: ``embed`` + ``dimension``).

    Optional hook, not part of the protocol: ``embed_queries(texts) -> list[Vector]``
    for models with asymmetric query/document prefixes. :func:`embed_queries` applies
    it when the embedder defines it and uses ``embed`` otherwise.
    """

    dimension: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def embed_queries(embedder: Embedder, texts: Sequence[str]) -> list[Vector]:
    """Query-side embedding: ``embedder.embed_queries`` when available, else ``embedder.embed``."""
    fn = getattr(embedder, "embed_queries", None)
    return list(fn(texts)) if callable(fn) else list(embedder.embed(texts))


def l2_normalize(v: Sequence[float]) -> Vector:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else list(v)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def mean_vector(vectors: Sequence[Sequence[float]]) -> Vector:
    """Element-wise mean (used by HyDE to average hypothetical-document embeddings)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vectors) for x in out]


class HashingEmbedder:
    """Deterministic character-n-gram feature-hashing embedder (offline baseline).

    Features are lowercase word tokens plus character 3- and 4-grams of each
    word (with boundary markers); each feature is hashed with BLAKE2b to a
    bucket and a sign, weighted by ``log(1 + count)`` and L2-normalised so the
    dot product is a cosine similarity. It captures lexical overlap and
    misspellings, not semantics; it is documented as the offline baseline that
    production embedders replace.
    """

    def __init__(self, dimension: int = 512, *, ngrams: Sequence[int] = (3, 4), query_prefix: str = "", document_prefix: str = "") -> None:
        self.dimension = dimension
        self.ngrams = tuple(ngrams)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _features(self, text: str) -> dict[str, int]:
        feats: dict[str, int] = {}
        for w in _WORD_RE.findall(text.lower()):
            feats[f"w:{w}"] = feats.get(f"w:{w}", 0) + 1
            padded = f"^{w}$"
            for n in self.ngrams:
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    g = f"g{n}:{padded[i : i + n]}"
                    feats[g] = feats.get(g, 0) + 1
        return feats

    def _vector(self, text: str) -> Vector:
        v = [0.0] * self.dimension
        for feat, count in self._features(text).items():
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dimension
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign * math.log1p(count)
        return l2_normalize(v)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(self.document_prefix + t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(self.query_prefix + t) for t in texts]

    def describe(self) -> dict[str, Any]:
        return {"embedder": "hashing", "dimension": self.dimension, "ngrams": list(self.ngrams)}


class CallableEmbedder:
    """Wrap any ``fn(texts) -> vectors`` (and optionally a query function) as an :class:`Embedder`."""

    def __init__(self, fn: Callable[[Sequence[str]], Sequence[Sequence[float]]], dimension: int, *, query_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None, name: str = "callable") -> None:
        self._fn = fn
        self._qfn = query_fn or fn
        self.dimension = dimension
        self.name = name

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [list(map(float, v)) for v in self._fn(list(texts))]

    def embed_queries(self, texts: Sequence[str]) -> list[Vector]:
        return [list(map(float, v)) for v in self._qfn(list(texts))]

    def describe(self) -> dict[str, Any]:
        return {"embedder": self.name, "dimension": self.dimension}


class VoyageEmbedder:
    """Voyage AI embeddings (lazy import of ``voyageai``; asymmetric ``input_type``)."""

    def __init__(self, model: str = "voyage-3", *, dimension: int = 1024, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            try:
                import voyageai  # noqa: WPS433
            except ImportError as exc:  # pragma: no cover
                raise ImportError("voyageai is not installed (pip install voyageai)") from exc
            client = voyageai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.dimension = dimension

    def _embed(self, texts: Sequence[str], input_type: str) -> list[Vector]:
        if not texts:
            return []
        res = self._client.embed(list(texts), model=self.model, input_type=input_type)
        return [list(map(float, v)) for v in res.embeddings]

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts, "query")

    def describe(self) -> dict[str, Any]:
        return {"embedder": "voyage", "model": self.model, "dimension": self.dimension}


class SentenceTransformerEmbedder:
    """sentence-transformers embeddings (lazy import) with optional E5/BGE-style prefixes."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", *, query_prefix: str = "", document_prefix: str = "", model: Any = None) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: WPS433
            except ImportError as exc:  # pragma: no cover
                raise ImportError("sentence-transformers is not installed") from exc
            model = SentenceTransformer(model_name)
        self._model = model
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        self.dimension = int(dim) if dim else 768

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vecs = self._model.encode([self.document_prefix + t for t in texts], normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]

    def embed_queries(self, texts: Sequence[str]) -> list[Vector]:
        vecs = self._model.encode([self.query_prefix + t for t in texts], normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]

    def describe(self) -> dict[str, Any]:
        return {"embedder": "sentence_transformers", "model": self.model_name, "dimension": self.dimension}


@runtime_checkable
class ANNBackend(Protocol):
    """Optional approximate-nearest-neighbour hook (HNSW etc.)."""

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None: ...

    def remove(self, ids: Sequence[str]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[ScoredId]: ...


class DenseIndex:
    """Brute-force cosine index (numpy when available, pure Python otherwise) with an ANN hook.

    Filters are applied as an allow-list *during* scoring (pre-filtering), never
    after taking the top-k, because post-filtering collapses recall for
    selective predicates (Patel et al. 2024).
    """

    def __init__(self, embedder: Embedder, *, ann: ANNBackend | None = None) -> None:
        self.embedder = embedder
        self.ann = ann
        self._ids: list[str] = []
        self._vectors: list[Vector] = []
        self._pos: dict[str, int] = {}
        self._matrix: Any = None

    # ------------------------------------------------------------- building
    def add(self, items: Sequence[tuple[str, str]]) -> None:
        if not items:
            return
        ids = [i for i, _ in items]
        existing = [i for i in ids if i in self._pos]
        if existing:
            self.remove(existing)
        vecs = self.embedder.embed([t for _, t in items])
        for i, v in zip(ids, vecs):
            self._pos[i] = len(self._ids)
            self._ids.append(i)
            self._vectors.append(l2_normalize(v))
        self._matrix = None
        if self.ann is not None:
            self.ann.add(ids, vecs)

    def remove(self, ids: Sequence[str]) -> None:
        drop = {i for i in ids if i in self._pos}
        if not drop:
            return
        keep = [(i, v) for i, v in zip(self._ids, self._vectors) if i not in drop]
        self._ids = [i for i, _ in keep]
        self._vectors = [v for _, v in keep]
        self._pos = {i: n for n, i in enumerate(self._ids)}
        self._matrix = None
        if self.ann is not None:
            self.ann.remove(list(drop))

    def __len__(self) -> int:
        return len(self._ids)

    def vector(self, doc_id: str) -> Optional[Vector]:
        p = self._pos.get(doc_id)
        return self._vectors[p] if p is not None else None

    # -------------------------------------------------------------- search
    def embed_query(self, query: str) -> Vector:
        """Embed one query (asymmetric ``embed_queries`` when the embedder has it, else ``embed``)."""
        return l2_normalize(embed_queries(self.embedder, [query])[0])

    def search_vector(self, vector: Sequence[float], k: int = 10, *, allowed_ids: set[str] | None = None) -> list[ScoredId]:
        if not self._ids:
            return []
        if self.ann is not None and allowed_ids is None:
            return self.ann.search(vector, k)[:k]
        q = l2_normalize(vector)
        if _np is not None:
            if self._matrix is None:
                self._matrix = _np.asarray(self._vectors, dtype=float)
            sims = self._matrix @ _np.asarray(q, dtype=float)
            pairs = [(self._ids[i], float(sims[i])) for i in range(len(self._ids)) if allowed_ids is None or self._ids[i] in allowed_ids]
        else:  # pragma: no cover - pure Python fallback
            pairs = [(i, cosine(q, v)) for i, v in zip(self._ids, self._vectors) if allowed_ids is None or i in allowed_ids]
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        return pairs[:k]

    def search(self, query: str, k: int = 10, *, allowed_ids: set[str] | None = None) -> list[ScoredId]:
        return self.search_vector(self.embed_query(query), k, allowed_ids=allowed_ids)

    def describe(self) -> dict[str, Any]:
        d = {"docs": len(self), "backend": "numpy" if _np is not None else "python", "ann": self.ann is not None}
        if hasattr(self.embedder, "describe"):
            d["embedder"] = self.embedder.describe()
        return d
