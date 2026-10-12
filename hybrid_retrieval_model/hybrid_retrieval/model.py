"""``HybridRetrievalModel`` - the pipeline component wrapping :class:`HybridRetriever`.

Life cycle (``pipeline_core.PipelineComponent``):

``plan()``
    Reads the task, indexes any documents passed in ``request.inputs``, and
    produces a :class:`RetrievalPlan`. Whenever ``ctx.offline`` is true (any
    ``MockLLM``, ``ScriptedLLM`` included; spec 3.1.2), ``constraints.deterministic``
    is set, or the budget is already exhausted, the :class:`HeuristicPlanner`
    decides from query features and the domain profile; online, one structured
    LLM call (``PLANNER_PROMPT`` + ``PlannerOutput`` schema) proposes the plan and
    Python clamps it. A planner call that fails or blows the budget falls back
    to the heuristic plan with a warning. Either way :func:`apply_budget` fits
    the plan to the remaining call budget. An explicit plan in ``inputs["plan"]`` wins.

``execute()``
    Runs the retriever, derives confidence from pure diagnostics, compares it
    with the domain's ``min_confidence_to_accept`` and the request's
    ``min_confidence`` (below either -> ``NEEDS_REVIEW``), and renders a
    readable ``output_text``. Budget exhaustion inside transforms or reranking
    degrades to the fused order with warnings; it never discards evidence.

Direct API (what other models consume through the ``Retriever`` protocol):
``index(documents)``, ``add_documents(documents)``, ``remove_documents(ids)``,
``await retrieve(query, k=10, filters=None)``, ``retrieve_sync(...)``,
``save_index(path)`` / ``load_index(path)``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    ENGINEERING_RIGOR,
    BaseLLMClient,
    ComponentSpec,
    Evidence,
    LLMRequest,
    Message,
    PipelineComponent,
    RunContext,
    Status,
    TaskRequest,
    TaskResult,
    Tracer,
    register,
    run_sync,
)
from pipeline_core.errors import BudgetExceededError, LLMError, StructuredOutputError
from pipeline_core.protocols import Executor

from .config import HybridRetrievalConfig
from .dense import Embedder
from .plan import HeuristicPlanner, PlannerOutput, QueryFeatures, RetrievalPlan, analyze_query, apply_budget, budget_exhausted
from .prompts import PLANNER_PROMPT, schema_of
from .retriever import ChannelStats, HybridRetriever, RetrievalResult

log = logging.getLogger(__name__)


class ResultRow(BaseModel):
    """Compact view of one retrieved passage for ``result.output``."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    id: str
    doc_id: str
    source: str
    score: float
    retriever: str
    channels: list[str] = Field(default_factory=list)
    section: str = ""
    snippet: str


class RetrievalOutput(BaseModel):
    """Typed ``TaskResult.output`` of the hybrid retrieval model."""

    model_config = ConfigDict(extra="forbid")

    query: str
    plan_summary: str
    n_results: int
    n_documents: int
    n_chunks: int
    filtered_chunks: Optional[int] = None
    channels: list[ChannelStats] = Field(default_factory=list)
    overlap_at_k: float = 0.0
    coverage: float = 0.0
    reranked: bool = False
    rerank_method: str = "none"
    transforms_applied: list[str] = Field(default_factory=list)
    n_queries: int = 1
    results: list[ResultRow] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_basis: str = ""


@register
class HybridRetrievalModel(PipelineComponent):
    """Hybrid (BM25 + dense) retrieval with fusion and reranking, planned per query.

    Parameters
    ----------
    llm:
        Any ``BaseLLMClient``; ``None`` -> ``make_llm()`` (MockLLM without credentials).
    config:
        :class:`HybridRetrievalConfig`, a dict of its fields, or ``None`` for defaults;
        keyword overrides (``k_final=5``) are applied on top.
    embedder, executor, cross_encoder:
        Injected backends (see :class:`HybridRetriever`).
    domain:
        Default domain profile for the direct API (``retrieve``); ``run()`` uses ``request.domain``.
    """

    spec = ComponentSpec(
        name="hybrid_retrieval",
        version="1.0.0",
        description="Lexical (BM25) + dense retrieval with RRF/convex fusion, LLM or cross-encoder reranking, query transforms and metadata filters.",
        capabilities=["retrieval", "bm25", "dense_retrieval", "rank_fusion", "reranking", "query_expansion", "metadata_filtering", "chunking"],
        consumes=["inputs.documents (list[str|dict|Document|Evidence]) optional", "inputs.query optional (defaults to task)", "inputs.k", "inputs.filters", "inputs.plan", "context (Evidence) indexed when the store is empty"],
        produces=["RetrievalOutput", "evidence: list[Evidence] with retriever in {sparse,dense,fused,reranked}"],
        composes_with=["pipeline_rag", "answer_verification", "task_decomposition", "parallelization", "caching"],
    )

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        config: HybridRetrievalConfig | dict[str, Any] | None = None,
        embedder: Embedder | None = None,
        executor: Executor | None = None,
        cross_encoder: Any = None,
        domain: str = "general",
        tracer: Tracer | None = None,
        raise_on_error: bool = False,
        **overrides: Any,
    ) -> None:
        if isinstance(config, HybridRetrievalConfig):
            cfg = config.model_copy(update=overrides) if overrides else config
        else:
            cfg = HybridRetrievalConfig(**{**(config or {}), **overrides})
        super().__init__(llm, tracer=tracer, raise_on_error=raise_on_error, **cfg.model_dump())
        self.cfg = cfg
        self.domain = domain
        self.retriever = HybridRetriever(cfg, embedder=embedder, llm=self.llm, executor=executor, cross_encoder=cross_encoder, profile=domain)
        self.planner = HeuristicPlanner(cfg)

    # ------------------------------------------------------------ direct API
    @property
    def store(self):
        """The underlying :class:`DocumentStore`."""
        return self.retriever.store

    def index(self, documents: Iterable[Any]) -> int:
        """Replace the index with ``documents`` (dicts with id/text/metadata, strings, Documents or Evidence); returns chunk count."""
        return self.retriever.index(documents)

    def add_documents(self, documents: Iterable[Any]) -> int:
        """Add or replace documents by id; returns the number of new chunks."""
        return self.retriever.add_documents(documents)

    def remove_documents(self, doc_ids: Iterable[str]) -> int:
        """Remove documents by id; returns removed chunk count."""
        return self.retriever.remove_documents(doc_ids)

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        """``Retriever`` protocol: heuristic plan (domain = ``self.domain``), no LLM calls unless ``direct_rerank``."""
        return await self.retriever.retrieve(query, k=k, filters=filters)

    def retrieve_sync(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        """Synchronous wrapper of :meth:`retrieve`."""
        return run_sync(self.retrieve(query, k=k, filters=filters))

    def save_index(self, path: str | Path) -> Path:
        """Persist documents and chunks to JSON (channels are rebuilt on load)."""
        return self.retriever.save(path)

    def load_index(self, path: str | Path) -> int:
        """Load a JSON index saved with :meth:`save_index`; returns chunk count."""
        from .index import DocumentStore

        self.retriever.store = DocumentStore.load(path)
        self.retriever.rebuild()
        return self.retriever.n_chunks

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["index"] = self.retriever.describe()
        return d

    # ------------------------------------------------------------- planning
    def _ingest(self, request: TaskRequest, ctx: RunContext) -> int:
        """Index ``inputs['documents']`` (or upstream context when the store is empty). Idempotent per request."""
        if ctx.scratch.get("ingested"):
            return 0
        ctx.scratch["ingested"] = True
        docs = request.inputs.get("documents")
        added = 0
        if docs:
            with ctx.step("index") as rec:
                added = self.add_documents(docs if isinstance(docs, (list, tuple)) else [docs])
                rec.summary = f"indexed {added} chunks from {len(docs) if isinstance(docs, (list, tuple)) else 1} documents"
        elif self.retriever.n_chunks == 0 and request.context:
            with ctx.step("index") as rec:
                added = self.add_documents(request.context)
                rec.summary = f"indexed {added} chunks from {len(request.context)} context evidence items"
        return added

    def _query_of(self, request: TaskRequest) -> str:
        q = request.inputs.get("query")
        return str(q).strip() if q else request.task.strip()

    def _corpus_description(self) -> str:
        fields = self.retriever.store.metadata_fields()
        if not fields:
            return f"documents={self.retriever.n_documents} chunks={self.retriever.n_chunks}; fields: none"
        rendered = "; ".join(f"{k}=[{', '.join(v)}]" for k, v in fields.items())
        return f"documents={self.retriever.n_documents} chunks={self.retriever.n_chunks}; fields: {rendered}"

    def _base_sparse_weight(self, ctx: RunContext) -> float:
        hint = ctx.profile.retrieval_hints.get("sparse_weight")
        return float(hint) if isinstance(hint, (int, float)) else self.cfg.default_sparse_weight

    async def _llm_plan(self, request: TaskRequest, ctx: RunContext, query: str, features: QueryFeatures, mandatory: dict[str, Any]) -> RetrievalPlan:
        remaining = ctx.budget.remaining_calls()
        extra_system = [ctx.profile.system_text()]
        if ctx.profile.risk_level in ("high", "safety_critical"):
            extra_system.append(ENGINEERING_RIGOR)
        rendered = PLANNER_PROMPT.render(
            extra_system=extra_system,
            task=request.task,
            query=query,
            features=features.render(),
            corpus=self._corpus_description(),
            budget=f"llm_calls={'unlimited' if remaining is None else max(0, remaining - 1)}",
            defaults=f"sparse_weight={self._base_sparse_weight(ctx):.2f}; k_final={request.inputs.get('k') or self.cfg.k_final}",
        )
        req = LLMRequest(
            system_stable=rendered.system_blocks,
            messages=[Message(role="user", content=rendered.user)],
            output_schema=schema_of(PlannerOutput),
            effort="high",
            max_tokens=1500,
            tags=["plan", "hybrid_retrieval"],
        )
        resp = await ctx.llm_call(req, step="plan.llm")
        out = PlannerOutput.model_validate(resp.parsed)
        return RetrievalPlan.from_planner_output(
            out,
            query=query,
            config=self.cfg,
            features=features,
            mandatory_filters=mandatory,
            corpus_fields=self.retriever.store.metadata_fields() or None,
        )

    async def plan(self, request: TaskRequest, ctx: RunContext) -> RetrievalPlan:
        """Decide channels, weights, fusion, transforms, reranking and k for this request."""
        self._ingest(request, ctx)
        query = self._query_of(request)
        k = request.inputs.get("k")
        k_final = int(k) if k else None
        mandatory = dict(request.inputs.get("filters") or {})
        features = analyze_query(query)

        explicit = request.inputs.get("plan")
        if explicit:
            data = dict(explicit) if isinstance(explicit, dict) else explicit.model_dump()
            data.update({"query": query, "source": "explicit", "features": features.model_dump()})
            if mandatory:
                data["mandatory_filters"] = {**data.get("mandatory_filters", {}), **mandatory}
            if k_final:
                data["k_final"] = k_final
            plan = RetrievalPlan.model_validate(data)
            return apply_budget(plan, self.cfg, remaining_calls=ctx.budget.remaining_calls(), deterministic=request.constraints.deterministic)

        # Spec 3.1.2: the heuristic planner whenever ctx.offline (every MockLLM, ScriptedLLM included),
        # when the request is deterministic, or when no budget is left for the planning call.
        if ctx.offline or request.constraints.deterministic or budget_exhausted(ctx):
            return self.planner.plan(query, profile=ctx.profile, constraints=request.constraints, mandatory_filters=mandatory, remaining_calls=ctx.budget.remaining_calls(), k_final=k_final)

        try:
            plan = await self._llm_plan(request, ctx, query, features, mandatory)
        except BudgetExceededError as exc:  # the planning call itself exhausted cost/latency: degrade, do not fail
            ctx.warn(f"LLM planner skipped ({exc}); using heuristic plan without LLM steps")
            return self.planner.plan(query, profile=ctx.profile, constraints=request.constraints, mandatory_filters=mandatory, remaining_calls=0, k_final=k_final)
        except (LLMError, StructuredOutputError, ValueError) as exc:
            ctx.warn(f"LLM planner failed ({type(exc).__name__}: {str(exc)[:120]}); using heuristic plan")
            return self.planner.plan(query, profile=ctx.profile, constraints=request.constraints, mandatory_filters=mandatory, remaining_calls=ctx.budget.remaining_calls(), k_final=k_final)
        if k_final:
            plan.k_final = k_final
            plan.rerank_depth = max(plan.rerank_depth, k_final)
        return apply_budget(plan, self.cfg, remaining_calls=ctx.budget.remaining_calls(), deterministic=False)

    # ------------------------------------------------------------ execution
    async def execute(self, request: TaskRequest, plan: RetrievalPlan, ctx: RunContext) -> TaskResult:
        """Run the plan, assess confidence and render the result."""
        self._ingest(request, ctx)
        if self.retriever.n_chunks == 0:
            return self.new_result(
                request,
                status=Status.FAILED,
                output=None,
                output_text="No documents are indexed. Pass inputs['documents'] or call HybridRetrievalModel.index(documents) first.",
                confidence=0.0,
                confidence_basis="empty index",
                errors=["RetrievalError: no documents indexed"],
            )

        with ctx.step("retrieve", plan=plan.describe()) as rec:
            result: RetrievalResult = await self.retriever.search(plan, ctx=ctx, profile=ctx.profile)
            rec.summary = f"{len(result.evidence)} results; transforms={result.transforms_applied}; rerank={result.rerank_method}"

        with ctx.step("assess") as rec:
            threshold = max(ctx.profile.min_confidence_to_accept, request.constraints.min_confidence)
            if not result.evidence:
                status = Status.NEEDS_REVIEW
            elif result.confidence < threshold:
                status = Status.NEEDS_REVIEW
                ctx.warn(f"confidence {result.confidence:.2f} below acceptance threshold {threshold:.2f} ({ctx.profile.name}); flagged for review")
            else:
                status = Status.SUCCESS
            rec.summary = f"confidence={result.confidence:.2f} threshold={threshold:.2f} status={status.value}"

        output = self._output(plan, result)
        return self.new_result(
            request,
            status=status,
            output=output,
            output_text=self._render(plan, result, output, threshold),
            evidence=result.evidence,
            confidence=result.confidence,
            confidence_basis=result.confidence_basis,
            warnings=list(result.warnings),
            metadata={"plan": plan.model_dump(mode="json"), "index": self.retriever.describe(), "threshold": threshold},
        )

    # ------------------------------------------------------------- rendering
    def _output(self, plan: RetrievalPlan, result: RetrievalResult) -> RetrievalOutput:
        rows = [
            ResultRow(
                rank=ev.rank or i,
                id=ev.id,
                doc_id=str(ev.metadata.get("doc_id", ev.id)),
                source=ev.source,
                score=round(ev.score, 6),
                retriever=ev.retriever,
                channels=list(ev.metadata.get("channels", [])),
                section=" > ".join(ev.metadata.get("section_path", [])),
                snippet=ev.short(200),
            )
            for i, ev in enumerate(result.evidence, 1)
        ]
        return RetrievalOutput(
            query=plan.query,
            plan_summary=plan.describe(),
            n_results=len(result.evidence),
            n_documents=self.retriever.n_documents,
            n_chunks=self.retriever.n_chunks,
            filtered_chunks=result.filtered_chunks,
            channels=result.channels,
            overlap_at_k=result.overlap_at_k,
            coverage=result.coverage,
            reranked=result.reranked,
            rerank_method=result.rerank_method,
            transforms_applied=result.transforms_applied,
            n_queries=result.n_queries,
            results=rows,
            confidence=result.confidence,
            confidence_basis=result.confidence_basis,
        )

    def _render(self, plan: RetrievalPlan, result: RetrievalResult, output: RetrievalOutput, threshold: float) -> str:
        lines = [f'Hybrid retrieval for: "{plan.query}"', f"Plan: {plan.describe()}"]
        if plan.rationale:
            lines.append(f"Rationale: {plan.rationale}")
        if plan.degradations:
            lines.append(f"Degradations: {'; '.join(plan.degradations)}")
        corpus = f"Corpus: {output.n_documents} documents / {output.n_chunks} chunks"
        if output.filtered_chunks is not None:
            corpus += f"; filters admitted {output.filtered_chunks} chunks"
        lines.append(corpus)
        ch = ", ".join(f"{c.name}: {c.candidates} candidates over {c.queries} query variant(s), top raw score {c.top_score:.3f}" for c in output.channels)
        if ch:
            lines.append(f"Channels: {ch}")
        lines.append(f"Transforms applied: {output.transforms_applied or 'none'}; rerank: {output.rerank_method}")
        n_show = min(len(output.results), self.cfg.max_documents_in_output_text)
        lines.append(f"Results ({len(output.results)}, showing {n_show}):")
        for row in output.results[:n_show]:
            where = f"{row.source}" + (f" > {row.section}" if row.section else "")
            lines.append(f"  {row.rank:>2}. [{row.id}] score={row.score:.4f} via {row.retriever}({'+'.join(row.channels) or '-'})  {where}")
            lines.append(f"      {row.snippet}")
        if not output.results:
            lines.append("  (none)")
        lines.append(f"Confidence: {output.confidence:.2f} (threshold {threshold:.2f}) - {output.confidence_basis}")
        unique_warnings = list(dict.fromkeys(result.warnings))  # reranker/transform warnings are recorded on ctx too
        if unique_warnings:
            lines.append("Warnings: " + " | ".join(unique_warnings))
        return "\n".join(lines)
