"""Lexical retrieval: Okapi BM25 / BM25+ with an identifier-aware tokenizer.

BM25 stays a first-class channel in every hybrid stack because it is cheap,
needs no GPU and matches exact identifiers (part numbers, error codes,
requirement IDs) that embeddings blur (Robertson & Zaragoza 2009; Thakur et al.
2021). The implementation is pure Python so the package has no required
dependency; :class:`RankBM25Index` adapts the optional ``rank_bm25`` package.

Scoring (Okapi)::

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(q, d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

BM25+ adds ``delta`` to the saturated term-frequency component for every
matching term so long documents are not under-scored.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Optional, Sequence

ScoredId = tuple[str, float]

_IDENT_RE = re.compile(r"[A-Za-z0-9_]+(?:[-._:/][A-Za-z0-9_]+)*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SUB_SPLIT_RE = re.compile(r"[-._:/]+")
_ALNUM_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")

DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    "a an the and or of to in on for with by at from as is are was were be been being this that these those it its "
    "into about over under between which who whom whose what when where why how do does did done can could should would "
    "will shall may might must not no nor so than too very we you they he she i me my our your their them us if then else "
    "there here any all each every some such only own same other more most less least also just".split()
)


def looks_like_identifier(token: str) -> bool:
    """Heuristic: contains digits and letters, separators, or camelCase boundaries."""
    if any(ch in token for ch in "-_./:"):
        return True
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    if has_digit and has_alpha:
        return True
    return bool(_CAMEL_RE.search(token))


def whole_tokens(text: str, *, lowercase: bool = True) -> list[str]:
    """Whole word/identifier tokens without sub-token expansion (``ERR-1203`` stays one token)."""
    toks = _IDENT_RE.findall(text)
    return [t.lower() for t in toks] if lowercase else toks


class Tokenizer:
    """Lowercasing tokenizer that preserves identifiers and emits their sub-tokens.

    ``ERR-1203`` -> ``["err-1203", "err", "1203"]``; ``getUserName`` ->
    ``["getusername", "get", "user", "name"]``. A ``stemmer`` callable (for
    example ``nltk.stem.PorterStemmer().stem``) is applied to plain words and
    sub-tokens, never to whole identifiers.
    """

    def __init__(self, *, lowercase: bool = True, preserve_identifiers: bool = True, stemmer: Callable[[str], str] | None = None, stopwords: Iterable[str] | None = None) -> None:
        self.lowercase = lowercase
        self.preserve_identifiers = preserve_identifiers
        self.stemmer = stemmer
        self.stopwords = frozenset(stopwords) if stopwords is not None else frozenset()

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in _IDENT_RE.findall(text):
            whole = raw.lower() if self.lowercase else raw
            if self.preserve_identifiers and looks_like_identifier(raw):
                out.append(whole)
                subs = [s for part in _SUB_SPLIT_RE.split(raw) for s in _CAMEL_RE.split(part) if s]
                subs = [s for sub in subs for s in _ALNUM_BOUNDARY_RE.split(sub) if s]
                for s in subs:
                    tok = s.lower() if self.lowercase else s
                    if tok != whole:
                        out.append(self._stem(tok))
            else:
                tok = whole
                if tok in self.stopwords:
                    continue
                out.append(self._stem(tok))
        return out

    def _stem(self, tok: str) -> str:
        return self.stemmer(tok) if self.stemmer and tok.isalpha() else tok

    def content_tokens(self, text: str) -> set[str]:
        """Tokens minus default stopwords; used by drift guards and confidence."""
        return {t for t in self(text) if t not in DEFAULT_STOPWORDS and len(t) > 1}


class BM25Index:
    """Pure-Python inverted index scored with Okapi BM25 or BM25+."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, variant: str = "okapi", delta: float = 1.0, tokenizer: Tokenizer | None = None) -> None:
        if variant not in ("okapi", "plus"):
            raise ValueError("variant must be 'okapi' or 'plus'")
        self.k1 = k1
        self.b = b
        self.variant = variant
        self.delta = delta
        self.tokenizer = tokenizer or Tokenizer()
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._total_len = 0

    # ------------------------------------------------------------- building
    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_len:
            self.remove([doc_id])
        toks = self.tokenizer(text)
        self._doc_len[doc_id] = len(toks)
        self._total_len += len(toks)
        for t in toks:
            self._postings.setdefault(t, {})
            self._postings[t][doc_id] = self._postings[t].get(doc_id, 0) + 1

    def add_many(self, items: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def remove(self, doc_ids: Iterable[str]) -> None:
        for d in doc_ids:
            n = self._doc_len.pop(d, None)
            if n is None:
                continue
            self._total_len -= n
            for t in list(self._postings):
                self._postings[t].pop(d, None)
                if not self._postings[t]:
                    del self._postings[t]

    def __len__(self) -> int:
        return len(self._doc_len)

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._doc_len) if self._doc_len else 0.0

    # -------------------------------------------------------------- scoring
    def idf(self, term: str) -> float:
        df = len(self._postings.get(term, {}))
        n = len(self._doc_len)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def score_all(self, query: str, *, allowed_ids: set[str] | None = None) -> dict[str, float]:
        """BM25 score for every document sharing a term with ``query``."""
        scores: dict[str, float] = {}
        avgdl = self.avgdl or 1.0
        qterms = self.tokenizer(query)
        seen: set[str] = set()
        for t in qterms:
            if t in seen:
                continue
            seen.add(t)
            postings = self._postings.get(t)
            if not postings:
                continue
            idf = self.idf(t)
            for d, tf in postings.items():
                if allowed_ids is not None and d not in allowed_ids:
                    continue
                dl = self._doc_len[d]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                comp = tf * (self.k1 + 1.0) / denom
                if self.variant == "plus":
                    comp += self.delta
                scores[d] = scores.get(d, 0.0) + idf * comp
        return scores

    def search(self, query: str, k: int = 10, *, allowed_ids: set[str] | None = None) -> list[ScoredId]:
        """Top-k ``(doc_id, score)`` sorted by score desc, id asc (deterministic)."""
        scores = self.score_all(query, allowed_ids=allowed_ids)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def describe(self) -> dict[str, object]:
        return {"backend": "python", "variant": self.variant, "k1": self.k1, "b": self.b, "docs": len(self), "terms": len(self._postings)}


class RankBM25Index:
    """Adapter over the optional ``rank_bm25`` package (same interface as :class:`BM25Index`).

    ``rank_bm25`` rebuilds its statistics on every change, so this adapter is
    intended for static corpora. It raises ``ImportError`` lazily when the
    package is missing.
    """

    def __init__(self, *, k1: float = 1.2, b: float = 0.75, variant: str = "okapi", delta: float = 1.0, tokenizer: Tokenizer | None = None) -> None:
        try:
            import rank_bm25  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise ImportError("rank_bm25 is not installed; use BM25Index or pip install rank-bm25") from exc
        self._mod = rank_bm25
        self.k1, self.b, self.variant, self.delta = k1, b, variant, delta
        self.tokenizer = tokenizer or Tokenizer()
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._model = None

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._ids:
            self.remove([doc_id])
        self._ids.append(doc_id)
        self._tokens.append(self.tokenizer(text))
        self._model = None

    def add_many(self, items: Iterable[tuple[str, str]]) -> None:
        for d, t in items:
            self.add(d, t)

    def remove(self, doc_ids: Iterable[str]) -> None:
        drop = set(doc_ids)
        keep = [(i, t) for i, t in zip(self._ids, self._tokens) if i not in drop]
        self._ids = [i for i, _ in keep]
        self._tokens = [t for _, t in keep]
        self._model = None

    def __len__(self) -> int:
        return len(self._ids)

    def _ensure(self) -> None:
        if self._model is None and self._ids:
            if self.variant == "plus":
                self._model = self._mod.BM25Plus(self._tokens, k1=self.k1, b=self.b, delta=self.delta)
            else:
                self._model = self._mod.BM25Okapi(self._tokens, k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 10, *, allowed_ids: set[str] | None = None) -> list[ScoredId]:
        self._ensure()
        if self._model is None:
            return []
        scores = self._model.get_scores(self.tokenizer(query))
        pairs = [(i, float(s)) for i, s in zip(self._ids, scores) if s > 0 and (allowed_ids is None or i in allowed_ids)]
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        return pairs[:k]

    def describe(self) -> dict[str, object]:
        return {"backend": "rank_bm25", "variant": self.variant, "k1": self.k1, "b": self.b, "docs": len(self)}


def make_bm25(backend: str = "python", **kw: object) -> BM25Index | RankBM25Index:
    """Factory used by the retriever; ``backend='rank_bm25'`` falls back to python when unavailable."""
    if backend == "rank_bm25":
        try:
            return RankBM25Index(**kw)  # type: ignore[arg-type]
        except ImportError:
            pass
    return BM25Index(**kw)  # type: ignore[arg-type]
