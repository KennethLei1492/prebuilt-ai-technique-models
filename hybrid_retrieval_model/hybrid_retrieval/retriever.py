"""The hybrid retriever: channels -> fusion -> rerank, executing a :class:`RetrievalPlan`.

:class:`HybridRetriever` owns a :class:`~hybrid_retrieval.index.DocumentStore`,
a BM25 index over the chunks' *indexed text* (contextual header + body) and a
:class:`~hybrid_retrieval.dense.DenseIndex`. Given a plan it

1. resolves metadata filters to an allow-list that is pushed into *every*
   channel (never applied after fusion; Patel et al. 2024),
2. applies the plan's query transforms through budgeted LLM calls: multi-query
   and HyDE widen both legs, Query2doc rewrites the lexical query, and the free
   keyword extraction adds a *sparse-only* variant (it never touches the dense leg),
3. runs the sparse and dense legs concurrently for every query variant,
4. fuses the per-variant channel lists (unweighted RRF with ``k=60`` by default,
   or a weighted convex combination / DBSF with the plan's weights) and then the
   variants with RRF (Cormack et al. 2009; Bruch et al. 2023),
5. optionally reranks the top ``rerank_depth`` candidates, and
6. returns ``Evidence`` objects whose ``retriever`` field is
   ``sparse | dense | fused | reranked`` and whose metadata carries per-leg
   scores and ranks, plus a confidence estimate from pure functions.

Budgets never fail a search (spec 3.1.4 / 3.2): before every transform and
rerank window the remaining calls, cost and time are checked, exhausted budgets
skip the step with a warning, and a ``BudgetExceededError`` raised mid-way keeps
the fused candidates already computed. ``reranked`` is reported only when the
reranker actually changed the order (``reranker.applied``), so the confidence
bonus is never granted for a rerank that did nothing.

It implements the ``pipeline_core.protocols.Retriever`` protocol directly
(:meth:`retrieve`), so other models can consume it without the component wrapper.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import ENGINEERING_RIGOR, BaseLLMClient, DomainProfile, Evidence, Usage, get_profile, run_sync
from pipeline_core.component import RunContext
from pipeline_core.errors import BudgetExceededError
from pipeline_core.protocols import Executor
from pipeline_core.utils import clamp

from .config import HybridRetrievalConfig
from .dense import ANNBackend, DenseIndex, Embedder, HashingEmbedder, Vector, l2_normalize, mean_vector
from .fusion import ScoredId, channel_overlap, fuse, reciprocal_rank_fusion
from .index import Chunk, Chunker, Document, DocumentStore
from .plan import HeuristicPlanner, RetrievalPlan, budget_exhausted, estimate_llm_calls
from .query_transform import HyDE, KeywordExtractor, MultiQueryExpander, Query2Doc
from .rerank import NoopReranker, Reranker, make_reranker
from .sparse import DEFAULT_STOPWORDS, Tokenizer, make_bm25

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ChannelStats(BaseModel):
    """Per-channel diagnostics (dossier observability recommendation 12)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    queries: int = Field(description="Query variants executed on this channel")
    candidates: int = Field(description="Distinct chunk ids returned across variants")
    top_score: float = Field(default=0.0, description="Best raw score of the original query")


class RetrievalResult(BaseModel):
    """Everything :meth:`HybridRetriever.search` knows about one retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str
    evidence: list[Evidence] = Field(default_factory=list)
    channels: list[ChannelStats] = Field(default_factory=list)
    overlap_at_k: float = Field(default=0.0, description="|top-k sparse & top-k dense| / k for the original query")
    coverage: float = Field(default=0.0, description="Share of the query's content tokens present in the top-3 results")
    reranked: bool = Field(default=False, description="True only when a reranker actually reordered the candidates")
    rerank_method: str = "none"
    transforms_applied: list[str] = Field(default_factory=list)
    n_queries: int = Field(default=1, description="Distinct query strings executed on any channel (variants plus sparse-only keyword queries)")
    filtered_chunks: Optional[int] = Field(default=None, description="Chunks admitted by the filters (None = no filter)")
    confidence: float = 0.0
    confidence_basis: str = ""
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage, description="LLM usage incurred outside a RunContext (direct API)")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without an LLM)
# ---------------------------------------------------------------------------


def query_coverage(query: str, texts: Sequence[str], tokenizer: Tokenizer | None = None) -> float:
    """Fraction of the query's content tokens that occur in the union of ``texts``.

    A cheap lexical grounding signal: 1.0 means every informative query token
    appears somewhere in the top results; 0.0 means the results share no
    vocabulary with the query (a strong hint that the evidence is weak).
    """
    tok = tokenizer or Tokenizer()
    q = tok.content_tokens(query)
    if not q:
        return 1.0 if texts else 0.0
    found: set[str] = set()
    for t in texts:
        found |= tok.content_tokens(t)
    return len(q & found) / len(q)


def retrieval_confidence(
    *,
    coverage: float,
    overlap: float,
    n_results: int,
    k: int,
    reranked: bool,
    both_channels: bool,
) -> tuple[float, str]:
    """Combine retrieval diagnostics into a 0-1 confidence and a human-readable basis.

    ``0.15 + 0.45*coverage + 0.20*overlap (or 0.10 flat for a single channel)
    + 0.10*fill + 0.10 if reranked``, clamped to [0, 1]. Coverage dominates
    because lexical grounding of the query terms is the signal downstream
    generators can act on; channel agreement is the complementarity/drift
    diagnostic recommended by the dossier; reranking adds a bounded bonus
    because it is the largest quality lever in the cascade (Thakur et al. 2021).
    """
    if n_results <= 0 or k <= 0:
        return 0.0, "no candidates matched the query (or the filters admitted none)"
    fill = min(1.0, n_results / k)
    agreement = 0.20 * clamp(overlap, 0.0, 1.0) if both_channels else 0.10
    score = 0.15 + 0.45 * clamp(coverage, 0.0, 1.0) + agreement + 0.10 * fill + (0.10 if reranked else 0.0)
    score = round(clamp(score, 0.0, 1.0), 4)
    basis = (
        f"query-term coverage in top-3 = {coverage:.2f}; "
        + (f"sparse/dense overlap@{k} = {overlap:.2f}; " if both_channels else "single channel; ")
        + f"{n_results}/{k} results; "
        + ("reranked" if reranked else "fused order")
    )
    return score, basis


def merge_leg_metadata(
    chunk: Chunk,
    *,
    fused_score: float,
    rank: int,
    retriever: str,
    sparse: dict[str, tuple[int, float]],
    dense: dict[str, tuple[int, float]],
    variants_hit: int,
) -> Evidence:
    """Build the :class:`Evidence` for a fused candidate with per-leg scores and ranks in metadata."""
    ev = chunk.to_evidence(score=round(fused_score, 6), rank=rank, retriever=retriever)
    meta = ev.metadata
    s = sparse.get(chunk.id)
    d = dense.get(chunk.id)
    meta["fused_score"] = round(fused_score, 6)
    meta["sparse_rank"] = s[0] if s else None
    meta["sparse_score"] = round(s[1], 6) if s else None
    meta["dense_rank"] = d[0] if d else None
    meta["dense_score"] = round(d[1], 6) if d else None
    meta["channels"] = [c for c, hit in (("sparse", s is not None), ("dense", d is not None)) if hit]
    meta["variants_hit"] = variants_hit
    return ev


def _rank_map(ranking: Sequence[ScoredId]) -> dict[str, tuple[int, float]]:
    return {doc_id: (i + 1, score) for i, (doc_id, score) in enumerate(ranking)}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Sparse + dense retrieval with fusion and optional reranking over a :class:`DocumentStore`.

    Parameters
    ----------
    config:
        :class:`HybridRetrievalConfig` (defaults are literature-backed).
    embedder:
        Any :class:`Embedder`; defaults to the offline :class:`HashingEmbedder`.
    llm:
        Client used for query transforms and LLM rerankers when no
        ``RunContext`` is supplied (direct API). Optional.
    executor:
        Optional ``Executor`` (parallelization model) for pointwise reranking fan-out.
    cross_encoder:
        Optional cross-encoder object with ``predict(pairs)`` for ``rerank="cross_encoder"``.
    profile:
        Domain profile used by :meth:`retrieve` (the protocol API) to plan.
    """

    def __init__(
        self,
        config: HybridRetrievalConfig | None = None,
        *,
        embedder: Embedder | None = None,
        llm: BaseLLMClient | None = None,
        executor: Executor | None = None,
        cross_encoder: Any = None,
        profile: DomainProfile | str | None = None,
        store: DocumentStore | None = None,
        ann: ANNBackend | None = None,
    ) -> None:
        self.config = config or HybridRetrievalConfig()
        cfg = self.config
        self.llm = llm
        self.executor = executor
        self.cross_encoder = cross_encoder
        self.profile: DomainProfile = profile if isinstance(profile, DomainProfile) else get_profile(profile)
        self.tokenizer = Tokenizer(preserve_identifiers=cfg.preserve_identifiers, stopwords=DEFAULT_STOPWORDS if cfg.remove_stopwords else None)
        self.store = store or DocumentStore(Chunker(cfg.chunk_strategy, cfg.chunk_size_tokens, cfg.chunk_overlap_tokens))
        self.embedder: Embedder = embedder or HashingEmbedder(cfg.embedding_dimension)
        self.sparse = make_bm25(cfg.bm25_backend, k1=cfg.bm25_k1, b=cfg.bm25_b, variant=cfg.bm25_variant, delta=cfg.bm25_delta, tokenizer=self.tokenizer)
        self.dense = DenseIndex(self.embedder, ann=ann)
        self.planner = HeuristicPlanner(cfg)
        self.usage = Usage()
        if len(self.store):
            self.rebuild()

    # ------------------------------------------------------------ indexing
    def _index_chunks(self, chunks: Sequence[Chunk]) -> None:
        items = [(c.id, c.index_text(prepend_header=self.config.prepend_headers)) for c in chunks]
        self.sparse.add_many(items)
        self.dense.add(items)

    def add_documents(self, documents: Iterable[Any]) -> int:
        """Add or replace documents (dicts, strings, :class:`Document` or :class:`Evidence`); returns new chunk count."""
        docs = list(documents)
        if not docs:
            return 0
        # the store decides which documents are replaced (ids are assigned there), so the
        # chunks it dropped are exactly the ones to drop from both channels
        chunks, removed = self.store.upsert(docs)
        if removed:
            self.sparse.remove(removed)
            self.dense.remove(removed)
        self._index_chunks(chunks)
        return len(chunks)

    def index(self, documents: Iterable[Any]) -> int:
        """Replace the whole index with ``documents``; returns the chunk count."""
        self.clear()
        return self.add_documents(documents)

    def remove_documents(self, doc_ids: Iterable[str]) -> int:
        """Remove documents by id from the store and both channels; returns removed chunk count."""
        removed = self.store.remove(doc_ids)
        if removed:
            self.sparse.remove(removed)
            self.dense.remove(removed)
        return len(removed)

    def clear(self) -> None:
        self.store = DocumentStore(self.store.chunker)
        self.sparse = make_bm25(self.config.bm25_backend, k1=self.config.bm25_k1, b=self.config.bm25_b, variant=self.config.bm25_variant, delta=self.config.bm25_delta, tokenizer=self.tokenizer)
        self.dense = DenseIndex(self.embedder, ann=self.dense.ann)

    def rebuild(self) -> None:
        """Rebuild both channels from the store (after :meth:`load`)."""
        self.sparse = make_bm25(self.config.bm25_backend, k1=self.config.bm25_k1, b=self.config.bm25_b, variant=self.config.bm25_variant, delta=self.config.bm25_delta, tokenizer=self.tokenizer)
        self.dense = DenseIndex(self.embedder, ann=self.dense.ann)
        self._index_chunks(list(self.store.chunks.values()))

    @property
    def n_documents(self) -> int:
        return len(self.store.documents)

    @property
    def n_chunks(self) -> int:
        return len(self.store)

    def save(self, path: str | Path) -> Path:
        """Persist the document store (chunks included) as JSON; channels are rebuilt on load."""
        return self.store.save(path)

    @classmethod
    def load(cls, path: str | Path, **kw: Any) -> "HybridRetriever":
        """Load a store saved with :meth:`save` and rebuild both channels."""
        store = DocumentStore.load(path)
        return cls(store=store, **kw)

    def describe(self) -> dict[str, Any]:
        return {
            "documents": self.n_documents,
            "chunks": self.n_chunks,
            "sparse": self.sparse.describe(),
            "dense": self.dense.describe(),
            "chunker": {"strategy": self.store.chunker.strategy, "size_tokens": self.store.chunker.size_tokens, "overlap_tokens": self.store.chunker.overlap_tokens},
            "profile": self.profile.name,
        }

    # -------------------------------------------------------------- search
    def _extra_system(self, profile: DomainProfile | None) -> list[str]:
        """Stable system blocks appended to every transform/rerank prompt: profile text, plus ENGINEERING_RIGOR for high-risk domains (spec 3.3)."""
        prof = profile or self.profile
        blocks = [prof.system_text()]
        if prof.risk_level in ("high", "safety_critical"):
            blocks.append(ENGINEERING_RIGOR)
        return blocks

    def _hyde_vector(self, query: str, passages: Sequence[str]) -> Vector:
        """Mean of the query embedding and the hypothetical-passage embeddings (Gao et al. 2023); runs in a worker thread."""
        qvec = self.dense.embed_query(query)
        pvecs = [l2_normalize(v) for v in self.embedder.embed(list(passages))]
        return l2_normalize(mean_vector([qvec, *pvecs]))

    async def _apply_transforms(
        self,
        plan: RetrievalPlan,
        ctx: RunContext | None,
        profile: DomainProfile | None,
        warnings: list[str],
    ) -> tuple[list[str], list[str], dict[str, str], Optional[Vector], list[str]]:
        """Return ``(queries, sparse_only_queries, sparse_query_overrides, hyde_vector, transforms_applied)``.

        ``keywords`` feeds only the lexical leg (``sparse_only_queries``). LLM
        transforms are skipped with a warning when the context's budget is exhausted
        and a ``BudgetExceededError`` ends the transform phase without discarding
        what was gathered; nothing raises out of here (spec 3.1.4 / 3.2).
        """
        cfg = self.config
        queries = plan.all_queries()
        sparse_only: list[str] = []
        sparse_override: dict[str, str] = {}
        hyde_vec: Optional[Vector] = None
        applied: list[str] = []
        can_llm = ctx is not None or self.llm is not None
        extra_system = self._extra_system(profile)
        common = dict(min_overlap=cfg.expansion_min_overlap, extra_system=extra_system)

        def warn(msg: str) -> None:
            warnings.append(msg)
            if ctx is not None:
                ctx.warn(msg)

        for t in plan.query_transforms:
            if t == "keywords":
                terms = await KeywordExtractor(self.llm, use_llm=False).extract(plan.query, ctx)
                kw_query = " ".join(terms)
                if kw_query and kw_query.lower() != plan.query.lower() and plan.use_sparse:
                    sparse_only.append(kw_query)
                    applied.append("keywords")
                continue
            if not can_llm:
                warn(f"transform '{t}' skipped: no LLM available")
                continue
            reason = budget_exhausted(ctx)
            if reason:
                warn(f"transform '{t}' skipped: {reason}")
                continue
            try:
                if t == "multi_query":
                    ex = MultiQueryExpander(self.llm, n=cfg.multi_query_n, **common)
                    extra = await ex.expand(plan.query, ctx)
                    self.usage = self.usage + ex.usage
                    warnings.extend(ex.warnings)
                    if extra:
                        queries.extend(q for q in extra if q.lower() not in {x.lower() for x in queries})
                        applied.append("multi_query")
                elif t == "hyde" and plan.use_dense:
                    hy = HyDE(self.llm, n=cfg.hyde_n, **common)
                    passages = await hy.generate(plan.query, ctx)
                    self.usage = self.usage + hy.usage
                    warnings.extend(hy.warnings)
                    if passages:
                        hyde_vec = await asyncio.to_thread(self._hyde_vector, plan.query, passages)
                        applied.append("hyde")
                elif t == "query2doc" and plan.use_sparse:
                    q2d = Query2Doc(self.llm, repeat=cfg.query2doc_repeat, max_tokens=cfg.query2doc_max_tokens, **common)
                    expanded = await q2d.expand(plan.query, ctx)
                    self.usage = self.usage + q2d.usage
                    warnings.extend(q2d.warnings)
                    if expanded != plan.query:
                        sparse_override[plan.query] = expanded
                        applied.append("query2doc")
            except BudgetExceededError as exc:  # the transform spent the last of the budget: keep what we have
                warn(f"budget exhausted during transform '{t}' ({exc}); continuing with the queries gathered so far")
                break
        return queries, sparse_only, sparse_override, hyde_vec, applied

    async def _run_legs(
        self,
        plan: RetrievalPlan,
        queries: Sequence[str],
        sparse_only: Sequence[str],
        sparse_override: dict[str, str],
        hyde_vec: Optional[Vector],
        allowed: set[str] | None,
    ) -> tuple[list[list[ScoredId]], list[list[ScoredId]]]:
        """Run sparse and dense legs for every variant concurrently; ``sparse_only`` variants skip the dense leg."""
        all_queries = list(queries) + list(sparse_only)
        n_full = len(queries)
        tasks: list[Any] = []
        kinds: list[tuple[str, int]] = []
        for i, q in enumerate(all_queries):
            if plan.use_sparse:
                sq = sparse_override.get(q, q)
                tasks.append(asyncio.to_thread(self.sparse.search, sq, plan.k_sparse, allowed_ids=allowed))
                kinds.append(("sparse", i))
            if plan.use_dense and i < n_full:
                if i == 0 and hyde_vec is not None:
                    tasks.append(asyncio.to_thread(self.dense.search_vector, hyde_vec, plan.k_dense, allowed_ids=allowed))
                else:
                    tasks.append(asyncio.to_thread(self.dense.search, q, plan.k_dense, allowed_ids=allowed))
                kinds.append(("dense", i))
        results = await asyncio.gather(*tasks) if tasks else []
        sparse_lists: list[list[ScoredId]] = [[] for _ in all_queries]
        dense_lists: list[list[ScoredId]] = [[] for _ in all_queries]
        for (kind, i), res in zip(kinds, results):
            (sparse_lists if kind == "sparse" else dense_lists)[i] = list(res)
        return sparse_lists, dense_lists

    @staticmethod
    def fusion_weights(plan: RetrievalPlan) -> Optional[list[float]]:
        """Channel weights handed to :func:`fuse`: the plan's weights for score-based fusion (``convex``, ``dbsf``), ``None`` for rank-based ``rrf``/``interleave`` (unweighted, Cormack et al. 2009)."""
        return [plan.sparse_weight, plan.dense_weight] if plan.fusion in ("convex", "dbsf") else None

    def _fuse(self, plan: RetrievalPlan, sparse_lists: Sequence[Sequence[ScoredId]], dense_lists: Sequence[Sequence[ScoredId]]) -> list[ScoredId]:
        cfg = self.config
        per_variant: list[list[ScoredId]] = []
        for s, d in zip(sparse_lists, dense_lists):
            if plan.use_sparse and plan.use_dense:
                fused = fuse(
                    plan.fusion,
                    [list(s), list(d)],
                    weights=self.fusion_weights(plan),
                    rrf_k=cfg.rrf_k,
                    normalization=cfg.normalization,
                    ranges=[None, cfg.dense_score_range],
                )
            elif plan.use_sparse:
                fused = list(s)
            else:
                fused = list(d)
            if fused:
                per_variant.append(fused)
        if not per_variant:
            return []
        if len(per_variant) == 1:
            return per_variant[0]
        return reciprocal_rank_fusion(per_variant, cfg.rrf_k)

    async def search(self, plan: RetrievalPlan, *, ctx: RunContext | None = None, profile: DomainProfile | None = None) -> RetrievalResult:
        """Execute ``plan`` and return a :class:`RetrievalResult` (never raises for empty results or exhausted budgets)."""
        cfg = self.config
        warnings: list[str] = []
        usage_before = self.usage
        if self.n_chunks == 0:
            return RetrievalResult(query=plan.query, warnings=["index is empty"], confidence=0.0, confidence_basis="no documents indexed")

        allowed = self.store.filter(plan.filter_dict())
        if allowed is not None and not allowed:
            msg = f"filters {plan.filter_dict()} matched no chunks"
            warnings.append(msg)
            if ctx is not None:
                ctx.warn(msg)
            return RetrievalResult(query=plan.query, filtered_chunks=0, warnings=warnings, confidence=0.0, confidence_basis="filters admitted no chunks")

        queries, sparse_only, sparse_override, hyde_vec, applied = await self._apply_transforms(plan, ctx, profile, warnings)
        sparse_lists, dense_lists = await self._run_legs(plan, queries, sparse_only, sparse_override, hyde_vec, allowed)
        fused = self._fuse(plan, sparse_lists, dense_lists)
        n_queries = len(queries) + len(sparse_only)

        sparse0 = _rank_map(sparse_lists[0]) if plan.use_sparse else {}
        dense0 = _rank_map(dense_lists[0]) if plan.use_dense else {}
        hits: dict[str, int] = {}
        for lst in (*sparse_lists, *dense_lists):
            for doc_id, _ in lst:
                hits[doc_id] = hits.get(doc_id, 0) + 1

        channel_name = "fused" if plan.use_sparse and plan.use_dense else ("sparse" if plan.use_sparse else "dense")
        depth = plan.rerank_depth if plan.rerank != "none" else plan.k_final
        candidates: list[Evidence] = []
        for rank, (chunk_id, score) in enumerate(fused[:depth], 1):
            chunk = self.store.get(chunk_id)
            if chunk is None:  # pragma: no cover - index and store are kept in sync
                continue
            candidates.append(merge_leg_metadata(chunk, fused_score=score, rank=rank, retriever=channel_name, sparse=sparse0, dense=dense0, variants_hit=hits.get(chunk_id, 0)))

        reranked = False
        method = "none"
        if plan.rerank != "none" and candidates:
            needs_llm = plan.rerank in ("listwise", "pointwise")
            if needs_llm and ctx is None and self.llm is None:
                msg = f"rerank '{plan.rerank}' skipped: no LLM available in direct mode"
                warnings.append(msg)
            elif plan.rerank == "cross_encoder" and self.cross_encoder is None:
                msg = "rerank 'cross_encoder' skipped: no cross-encoder configured"
                warnings.append(msg)
                if ctx is not None:
                    ctx.warn(msg)
            else:
                reranker: Reranker = make_reranker(
                    plan.rerank,
                    self.llm,
                    window=cfg.rerank_window,
                    stride=cfg.rerank_stride,
                    passage_chars=cfg.rerank_passage_chars,
                    concurrency=cfg.pointwise_concurrency,
                    executor=self.executor,
                    extra_system=self._extra_system(profile),
                    cross_encoder=self.cross_encoder,
                )
                if not isinstance(reranker, NoopReranker):
                    try:
                        candidates = await reranker.rerank(plan.query, candidates, k=plan.k_final, ctx=ctx)
                    except BudgetExceededError as exc:  # keep the fused candidates computed so far (spec 3.1.4)
                        msg = f"budget exhausted during '{plan.rerank}' rerank ({exc}); returning fused order"
                        warnings.append(msg)
                        if ctx is not None:
                            ctx.warn(msg)
                    self.usage = self.usage + getattr(reranker, "usage", Usage())
                    warnings.extend(getattr(reranker, "warnings", []))
                    reranked = bool(getattr(reranker, "applied", True))  # no bonus for a rerank that did nothing
                    method = reranker.name if reranked else "none"

        final = candidates[: plan.k_final]
        for i, ev in enumerate(final, 1):
            ev.rank = i

        k = plan.k_final
        overlap = channel_overlap(sparse_lists[0], dense_lists[0], k) if plan.use_sparse and plan.use_dense else 0.0
        coverage = query_coverage(plan.query, [f"{e.source}\n{e.content}" for e in final[:3]], self.tokenizer)
        confidence, basis = retrieval_confidence(coverage=coverage, overlap=overlap, n_results=len(final), k=k, reranked=reranked, both_channels=plan.use_sparse and plan.use_dense)
        if final and confidence < cfg.weak_evidence_confidence:
            msg = f"weak evidence: confidence {confidence:.2f} below {cfg.weak_evidence_confidence:.2f}; downstream generators should consider abstaining"
            warnings.append(msg)
            if ctx is not None:
                ctx.warn(msg)
        if not final:
            msg = "no candidates matched the query"
            warnings.append(msg)
            if ctx is not None:
                ctx.warn(msg)

        channels: list[ChannelStats] = []
        if plan.use_sparse:
            channels.append(ChannelStats(name="sparse", queries=n_queries, candidates=len({d for lst in sparse_lists for d, _ in lst}), top_score=round(sparse_lists[0][0][1], 6) if sparse_lists[0] else 0.0))
        if plan.use_dense:
            channels.append(ChannelStats(name="dense", queries=len(queries), candidates=len({d for lst in dense_lists for d, _ in lst}), top_score=round(dense_lists[0][0][1], 6) if dense_lists[0] else 0.0))

        delta = Usage(
            input_tokens=self.usage.input_tokens - usage_before.input_tokens,
            output_tokens=self.usage.output_tokens - usage_before.output_tokens,
            llm_calls=self.usage.llm_calls - usage_before.llm_calls,
            cost_usd=self.usage.cost_usd - usage_before.cost_usd,
        )
        return RetrievalResult(
            query=plan.query,
            evidence=final,
            channels=channels,
            overlap_at_k=round(overlap, 4),
            coverage=round(coverage, 4),
            reranked=reranked,
            rerank_method=method,
            transforms_applied=applied,
            n_queries=n_queries,
            filtered_chunks=len(allowed) if allowed is not None else None,
            confidence=confidence,
            confidence_basis=basis,
            warnings=warnings,
            usage=delta,
        )

    # ------------------------------------------------------- protocol API
    def plan_for(self, query: str, *, k: int = 10, filters: dict[str, Any] | None = None, profile: DomainProfile | None = None) -> RetrievalPlan:
        """Heuristic plan for the direct API: free transforms only; reranking only when ``direct_rerank`` and an LLM exist."""
        plan = self.planner.plan(query, profile=profile or self.profile, mandatory_filters=filters, k_final=k, remaining_calls=None)
        plan.query_transforms = [t for t in plan.query_transforms if t == "keywords"]
        if not (self.config.direct_rerank and (self.llm is not None or plan.rerank == "cross_encoder" and self.cross_encoder is not None)):
            plan.rerank = "none"
        plan.expected_llm_calls = estimate_llm_calls(plan, self.config)
        return plan

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        """``pipeline_core.protocols.Retriever`` implementation: plan heuristically and search."""
        plan = self.plan_for(query, k=k, filters=filters)
        result = await self.search(plan)
        return result.evidence

    def retrieve_sync(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        """Synchronous wrapper around :meth:`retrieve`."""
        return run_sync(self.retrieve(query, k=k, filters=filters))
