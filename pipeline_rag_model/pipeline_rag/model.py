"""``PipelineRAGModel``: the end-to-end retrieval-augmented generation component.

Life cycle (``pipeline_core.PipelineComponent``):

``plan()``
    Reads the task and decides the :class:`~pipeline_rag.plan.RAGPlan`:
    route, query transforms, k, grading, compression, hops, verification and
    citation style. Online this is one structured LLM call seeded with the
    heuristic suggestion; offline (``ctx.offline``), under
    ``constraints.deterministic`` or when the call budget is nearly exhausted
    the :class:`~pipeline_rag.plan.HeuristicPlanner` decides alone.

``execute()``
    index -> transform -> retrieve (fan-out + RRF) -> grade (CRAG) ->
    iterate (IRCoT/FLARE, multi-hop routes) -> compress + order -> generate
    with citations -> reflect/verify -> confidence -> status, with Self-Route
    escalation to the next richer route when the cheap plan abstains.

Dependencies are injected and optional: ``retriever`` (hybrid retrieval
model), ``verifier`` (answer verification model), ``executor``
(parallelization model), ``cache`` (caching model). Without them the
built-in BM25 index, the Self-RAG reflection judge, ``gather_limited`` and
no cache are used, so the package works in isolation.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    BudgetExceededError,
    ComponentSpec,
    Constraints,
    DomainProfile,
    Evidence,
    Executor,
    PipelineComponent,
    ResponseCache,
    Retriever,
    RunContext,
    Status,
    TaskRequest,
    TaskResult,
    Usage,
    Verdict,
    Verifier,
    estimate_tokens,
    gather_limited,
    get_profile,
    is_offline_client,
    register,
    run_sync,
    sanitize_schema,
    stable_hash,
)

from .budget import calls_left, degraded_stages, note_degraded
from .config import PipelineRAGConfig
from .context import ContextCompressor, ContextGrader, ContextOrderer, DocGrade, GradeDecision, deduplicate
from .generator import Citation, GenerationOutput, GroundedGenerator, Statement, citation_coverage, cited_ids, render_answer, split_statements
from .indexing import BuiltinRetriever, IndexBuilder, normalize_documents
from .iterate import HopRecord, MultiHopLoop, ReflectionFlags, ReflectionJudge
from .plan import HeuristicPlanner, RAGPlan
from .prompts import CONTEXT_PREFIX_PROMPT, PLAN_PROMPT, SUMMARY_PROMPT, make_request
from .router import QueryRouter, Route, RouteDecision, TaskFeatures, extract_task_features, next_route
from .transform import QueryTransformer, QueryVariant, glossary_expansion, keyword_query, reciprocal_rank_fusion, step_back_query

_MULTI = ("multi_step", "agentic")


# ----------------------------------------------------------------- outputs


class SummaryOut(BaseModel):
    """Structured output of the RAPTOR-style summary call."""

    model_config = ConfigDict(extra="forbid")

    summary: str


class ContextPrefixOut(BaseModel):
    """Structured output of the Contextual Retrieval prefix call."""

    model_config = ConfigDict(extra="forbid")

    context: str


class ConfidenceSignals(BaseModel):
    """The four confidence signals and their weighted combination."""

    model_config = ConfigDict(extra="forbid")

    self_reported: Optional[float] = Field(default=None, description="Generator's verbalised confidence")
    citation_coverage: Optional[float] = Field(default=None, description="Fraction of statements with a resolving citation (ALCE)")
    support: Optional[float] = Field(default=None, description="Statement support fraction (Self-RAG IsSup / verifier score)")
    evidence_quality: Optional[float] = Field(default=None, description="Mean CRAG grade of the cited units mapped to [0, 1]")
    combined: float = 0.0
    threshold: float = 0.0
    basis: str = ""


class RAGOutput(BaseModel):
    """Typed ``TaskResult.output`` of the pipeline."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str = Field(default="", description="Answer with [doc-id] markers")
    answer_text: str = Field(default="", description="Answer rendered in the plan's citation style")
    citations: list[Citation] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str = ""
    route: Route = "single_step"
    routes_tried: list[str] = Field(default_factory=list, description="Self-Route escalation path, cheapest first")
    query_variants: list[QueryVariant] = Field(default_factory=list)
    grade_decision: Optional[GradeDecision] = None
    grades: list[DocGrade] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list, description="Ids of the evidence shown to the generator, in context order")
    cited_ids: list[str] = Field(default_factory=list)
    statements: list[Statement] = Field(default_factory=list)
    hops: list[HopRecord] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    reflection: Optional[ReflectionFlags] = None
    verdict: Optional[dict[str, Any]] = None
    confidence: ConfidenceSignals = Field(default_factory=ConfidenceSignals)
    full_corpus_used: bool = False
    compressed_units: int = 0
    context_tokens: int = 0


def aggregate_confidence(
    signals: dict[str, Optional[float]],
    weights: dict[str, float],
    *,
    abstained: bool = False,
) -> tuple[float, str]:
    """Weighted mean of the available signals (weights renormalised over those present).

    Missing signals (``None``) are skipped rather than counted as zero so a
    no-retrieval answer is judged on the signals that apply to it. An
    abstention caps the result at 0.25: an honest "insufficient evidence" is
    not a confident answer, but it is not a wrong one either.
    """
    present = {k: v for k, v in signals.items() if v is not None and k in weights and weights[k] > 0}
    if not present:
        return (0.0, "no confidence signals available")
    total_w = sum(weights[k] for k in present)
    combined = sum(weights[k] * max(0.0, min(1.0, float(v))) for k, v in present.items()) / total_w
    parts = [f"{k.replace('_', ' ')} {v:.2f} (w {weights[k] / total_w:.2f})" for k, v in present.items()]
    basis = ", ".join(parts) + f" -> {combined:.2f}"
    if abstained:
        capped = min(combined, 0.25)
        basis = "abstained (insufficient evidence); " + basis + (f"; capped at {capped:.2f}" if capped < combined else "")
        combined = capped
    return (round(combined, 4), basis)


class _Attempt(BaseModel):
    """Everything one pipeline pass produced (one per Self-Route escalation)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    plan: RAGPlan
    variants: list[QueryVariant] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    grade_decision: Optional[GradeDecision] = None
    grades: list[DocGrade] = Field(default_factory=list)
    hops: list[HopRecord] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    generation: GenerationOutput = Field(default_factory=GenerationOutput)
    statements: list[Statement] = Field(default_factory=list)
    reflection: Optional[ReflectionFlags] = None
    verdict: Optional[Verdict] = None
    full_corpus_used: bool = False
    compressed_units: int = 0
    context_tokens: int = 0


def _without_summaries(retr: BuiltinRetriever) -> Any:
    """A fresh BM25 index holding only ``retr``'s leaf chunks (summary nodes are rebuilt on demand)."""
    from .indexing import BM25Index

    idx = BM25Index(retr.index.k1, retr.index.b)
    idx.add(retr.leaf_chunks())
    return idx


# ------------------------------------------------------------------- model


@register
class PipelineRAGModel(PipelineComponent):
    """Route, transform, retrieve, grade, compress, generate with citations, verify, iterate."""

    spec = ComponentSpec(
        name="pipeline_rag",
        version="1.0.0",
        description=(
            "End-to-end retrieval-augmented generation: Adaptive-RAG routing, query transformation with RRF fusion, "
            "CRAG evidence grading, RECOMP-style compression with lost-in-the-middle ordering, citation-grounded generation, "
            "Self-RAG reflection, IRCoT/FLARE multi-hop iteration and Self-Route escalation."
        ),
        capabilities=[
            "question answering over documents",
            "citation-grounded generation",
            "multi-hop retrieval",
            "corpus-level summarisation (RAPTOR-style)",
            "abstention on insufficient evidence",
        ],
        consumes=["request.task (question)", "inputs.documents (list[str|dict]|dict)", "inputs.question (optional)", "context (upstream Evidence)"],
        produces=["RAGOutput (answer, citations, unsupported_claims, route, hops, reflection, confidence signals)"],
        composes_with=["hybrid_retrieval", "answer_verification", "parallelization", "caching", "task_decomposition"],
    )

    def __init__(
        self,
        llm: Any = None,
        *,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
        executor: Optional[Executor] = None,
        cache: Optional[ResponseCache] = None,
        fallback_retriever: Optional[Retriever] = None,
        config: Optional[PipelineRAGConfig] = None,
        tracer: Any = None,
        raise_on_error: bool = False,
        **overrides: Any,
    ) -> None:
        """Create the model.

        ``config`` (or keyword ``overrides`` of :class:`PipelineRAGConfig`
        fields) tunes every stage. The protocol objects are optional; see the
        module docstring for what replaces them when absent.
        """
        base = config.model_dump() if config is not None else {}
        base.update(overrides)
        self.cfg = PipelineRAGConfig(**base)
        super().__init__(llm, tracer=tracer, raise_on_error=raise_on_error, **self.cfg.model_dump())
        self.retriever = retriever
        self.verifier = verifier
        self.executor = executor
        self.cache = cache
        self.fallback_retriever = fallback_retriever
        self.index_builder = IndexBuilder(self.cfg)
        self.builtin = BuiltinRetriever()
        self.router = QueryRouter(self.cfg)
        self.planner = HeuristicPlanner(self.cfg)
        self.transformer = QueryTransformer(self.cfg)
        self.grader = ContextGrader(self.cfg)
        self.compressor = ContextCompressor(self.cfg)
        self.orderer = ContextOrderer()
        self.generator = GroundedGenerator(self.cfg)
        self.loop = MultiHopLoop(self.cfg)
        self.judge = ReflectionJudge(self.cfg)
        self.direct_usage = Usage()
        self._inline_indexes: "OrderedDict[str, BuiltinRetriever]" = OrderedDict()
        self.retrieval_cache_epoch = 0
        """Folded into retrieval cache keys of retrievers without a ``version``; bumped by :meth:`invalidate_retrieval_cache`."""
        self._cache_warned = False

    # ============================================================ life cycle

    async def plan(self, request: TaskRequest, ctx: RunContext) -> RAGPlan:
        """Decide the execution plan: heuristics offline, one structured LLM call online."""
        cfg = self.cfg
        features = extract_task_features(
            request.task,
            request.inputs,
            context=request.context,
            has_retriever=self.retriever is not None or len(self.builtin) > 0,
            profile=ctx.profile,
        )
        ctx.scratch["features"] = features
        heuristic = self.planner.plan_from_features(features, ctx.profile)
        if not self._llm_allowed(ctx) or request.constraints.deterministic:
            return heuristic
        if self._calls_left(ctx) < 2:
            ctx.warn("call budget too small for an LLM plan; using the heuristic planner")
            return heuristic
        try:
            req = make_request(
                PLAN_PROMPT,
                ctx.profile,
                schema=sanitize_schema(RAGPlan.model_json_schema()),
                effort=cfg.planner_effort,
                task=request.task,
                features=features.model_dump_json(),
                corpus_summary=self._corpus_summary(request, features),
                heuristic=heuristic.describe(),
            )
            resp = await ctx.llm_call(req, step="plan_llm")
            plan = RAGPlan.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - planning must never fail the run
            ctx.warn(f"LLM planning failed ({type(exc).__name__}); using the heuristic plan")
            note_degraded(ctx, "plan")
            return heuristic
        plan = plan.model_copy(update={"planner": "llm"})
        return self._enforce_domain(plan, ctx.profile, features).normalized(cfg)

    async def execute(self, request: TaskRequest, plan: RAGPlan, ctx: RunContext) -> TaskResult:
        """Carry the plan out, escalating the route on abstention (Self-Route)."""
        cfg = self.cfg
        features: TaskFeatures = ctx.scratch.get("features") or extract_task_features(
            request.task, request.inputs, context=request.context, has_retriever=self.retriever is not None or len(self.builtin) > 0, profile=ctx.profile
        )
        question = self._question(request)
        use_llm = self._llm_allowed(ctx)
        # constraints.deterministic: every advisory stage (transforms, grading, hops, reflection, summaries,
        # contextual prefixes) runs its heuristic twin; generation (once per Self-Route pass) is the only LLM call
        aux_llm = use_llm and not request.constraints.deterministic
        k_override = request.inputs.get("k")
        if isinstance(k_override, int) and k_override > 0:
            plan = plan.model_copy(update={"k": k_override})

        with ctx.step("index") as st:
            sources = await self._sources(request, plan, ctx, aux_llm)
            st.summary = f"{len(sources)} retrieval source(s)"
            st.details.update(sources=[name for name, _ in sources])

        routes_tried: list[str] = []
        current = plan
        attempt: Optional[_Attempt] = None
        for escalation in range(cfg.max_escalations + 1):
            routes_tried.append(current.route)
            suffix = "" if escalation == 0 else f"_{escalation + 1}"
            if escalation > 0 and current.use_hierarchy:
                # an escalated plan (e.g. into 'agentic') needs the summary nodes the first plan did not build
                with ctx.step("index" + suffix) as st:
                    built = await self._ensure_hierarchy(sources, ctx, aux_llm)
                    st.summary = f"{built} summary node(s) added for the escalated '{current.route}' plan"
            attempt = await self._run_pipeline(request, current, ctx, question=question, features=features, use_llm=use_llm, aux_llm=aux_llm, sources=sources, suffix=suffix)
            if not attempt.generation.abstained or not current.escalate_on_abstention:
                break
            nxt = next_route(current.route)
            if nxt is None or not features.has_corpus or ctx.deadline_exceeded():
                break
            if escalation >= cfg.max_escalations:
                break
            ctx.warn(f"plan '{current.route}' abstained; escalating to '{nxt}' (Self-Route)")
            current = current.model_copy(update={"route": nxt, "rationale": f"escalated from {current.route} after abstention; {current.rationale}", "max_hops": cfg.max_hops if nxt in _MULTI else 1}).normalized(cfg)
        assert attempt is not None
        return self._finish(request, current, attempt, ctx, question=question, routes_tried=routes_tried)

    # ========================================================== direct API

    async def answer(
        self,
        question: str,
        documents: Any = None,
        *,
        domain: str = "general",
        context: Sequence[Evidence] = (),
        constraints: Optional[Constraints] = None,
        k: Optional[int] = None,
        **inputs: Any,
    ) -> TaskResult:
        """Answer ``question`` from ``documents`` (and/or the indexed corpus) with citations.

        This is the direct API other models and orchestrators call; it wraps
        :meth:`run` so the full plan/trace/budget machinery applies.
        """
        payload: dict[str, Any] = dict(inputs)
        if documents is not None:
            payload["documents"] = documents
        if k is not None:
            payload["k"] = k
        req = TaskRequest(task=question, inputs=payload, context=list(context), domain=domain, constraints=constraints or Constraints())
        return await self.run(req)

    def answer_sync(self, question: str, documents: Any = None, **kw: Any) -> TaskResult:
        """Synchronous :meth:`answer`."""
        return run_sync(self.answer(question, documents, **kw))

    def index(self, documents: Any) -> int:
        """Chunk, prefix and index documents into the persistent built-in BM25 index; returns chunks added.

        Summary nodes built earlier are dropped so the next hierarchical plan
        rebuilds them over the enlarged corpus.
        """
        added = self.index_builder.add(self.builtin, documents)
        if added and self.builtin.summary_chunks():
            self.builtin.index = _without_summaries(self.builtin)
        return len(added)

    add_documents = index

    def clear_index(self) -> None:
        """Drop the persistent built-in index and the per-request index cache."""
        self.builtin = BuiltinRetriever()
        self._inline_indexes.clear()
        self.retrieval_cache_epoch += 1

    async def invalidate_retrieval_cache(self) -> int:
        """Forget cached retrieval results (call after re-indexing an injected retriever that has no ``version``).

        Bumps :attr:`retrieval_cache_epoch` (so stale keys are never read
        again) and asks the injected cache to drop entries tagged
        ``pipeline_rag``; returns the number the cache reports as removed.
        """
        self.retrieval_cache_epoch += 1
        if self.cache is None:
            return 0
        try:
            return int(await self.cache.invalidate(tags=["pipeline_rag"]) or 0)
        except Exception:  # noqa: BLE001 - the epoch bump already guarantees fresh keys
            return 0

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        """Fused retrieval over every configured source for one literal query (no LLM).

        Makes the model itself usable as a ``Retriever``.
        """
        sources = self._static_sources()
        if not sources:
            return []
        variants = [QueryVariant(kind="literal", text=query)]
        return await self._retrieve_fused(variants, k=k, sources=sources, provided=(), filters=filters, ctx=None, max_parallelism=8)

    def route(self, task: str, inputs: Optional[dict[str, Any]] = None, *, domain: str = "general", context: Sequence[Evidence] = ()) -> RouteDecision:
        """Heuristic Adaptive-RAG routing of a task (no LLM)."""
        profile = get_profile(domain)
        features = extract_task_features(task, inputs or {}, context=context, has_retriever=self.retriever is not None or len(self.builtin) > 0, profile=profile)
        return self.router.classify(task, inputs, features=features)

    async def route_llm(self, task: str, inputs: Optional[dict[str, Any]] = None, *, domain: str = "general", context: Sequence[Evidence] = ()) -> RouteDecision:
        """Adaptive-RAG routing with one low-effort structured LLM call (ROUTE_PROMPT), outside ``run()``.

        Falls back to :meth:`route` (heuristics) when the client is offline or
        ``llm_mode`` is ``never``, and on any LLM failure. Usage is added to
        ``self.direct_usage``.
        """
        from pipeline_core import Tracer

        heuristic = self.route(task, inputs, domain=domain, context=context)
        if self.cfg.llm_mode == "never" or (self.cfg.llm_mode == "auto" and is_offline_client(self.llm)):
            return heuristic
        profile = get_profile(domain)
        features = extract_task_features(task, inputs or {}, context=context, has_retriever=self.retriever is not None or len(self.builtin) > 0, profile=profile)
        ctx = RunContext(TaskRequest(task=task, inputs=dict(inputs or {}), context=list(context), domain=domain), Tracer(), self.llm, self.spec.name)
        decision = await self.router.classify_llm(task, features, ctx)
        self.direct_usage = self.direct_usage + ctx.usage
        return decision

    def plan_for(self, task: str, inputs: Optional[dict[str, Any]] = None, *, domain: str = "general", context: Sequence[Evidence] = ()) -> RAGPlan:
        """Heuristic plan preview for a task (what ``plan()`` returns offline)."""
        profile = get_profile(domain)
        features = extract_task_features(task, inputs or {}, context=context, has_retriever=self.retriever is not None or len(self.builtin) > 0, profile=profile)
        return self.planner.plan_from_features(features, profile)

    async def generate(
        self,
        question: str,
        evidence: Sequence[Evidence],
        *,
        domain: str = "general",
        answer_type: str = "factual",
        citation_style: str = "inline",
    ) -> GenerationOutput:
        """Grounded generation over caller-supplied evidence, outside ``run()``.

        Uses ``self.llm.complete`` directly and accounts usage in
        ``self.direct_usage``; heuristic (extractive) when the client is a
        mock or ``llm_mode`` is ``never``.
        """
        offline = is_offline_client(self.llm)  # a MockLLM, or a cache/recorder wrapping one
        if self.cfg.llm_mode == "never" or (self.cfg.llm_mode == "auto" and offline):
            return self.generator.generate_heuristic(question, evidence, answer_type=answer_type)
        from .generator import citation_instruction
        from .prompts import GENERATE_PROMPT, NO_RETRIEVAL_PROMPT
        from pipeline_core import render_evidence

        profile = get_profile(domain)
        schema = sanitize_schema(GenerationOutput.model_json_schema())
        if evidence:
            req = make_request(
                GENERATE_PROMPT,
                profile,
                schema=schema,
                effort=self.cfg.generation_effort,
                documents=render_evidence(evidence, max_chars_each=self.cfg.max_evidence_chars),
                question=question,
                answer_type=answer_type,
                citation_instruction=citation_instruction(citation_style, self.cfg.require_citations),  # type: ignore[arg-type]
                max_tokens=8000,
            )
        else:
            req = make_request(NO_RETRIEVAL_PROMPT, profile, schema=schema, effort=self.cfg.generation_effort, question=question, answer_type=answer_type, max_tokens=8000)
        resp = await self.llm.complete(req)
        self.direct_usage = self.direct_usage + resp.usage
        try:
            return GenerationOutput.model_validate(resp.parsed)
        except Exception:  # noqa: BLE001 - degrade to the extractive answer
            return self.generator.generate_heuristic(question, evidence, answer_type=answer_type)

    @property
    def solver(self):  # type: ignore[no-untyped-def]
        """``SubTaskSolver``-shaped callable for the task decomposition model (``solver=rag.solver``)."""
        return self.run

    # ======================================================= plan helpers

    def _llm_allowed(self, ctx: RunContext) -> bool:
        mode = self.cfg.llm_mode
        if mode == "never":
            return False
        if mode == "always":
            return True
        return not ctx.offline

    @staticmethod
    def _calls_left(ctx: RunContext) -> int:
        """LLM calls still affordable (0 once the latency or cost budget is spent); see :func:`budget.calls_left`."""
        return calls_left(ctx)

    @staticmethod
    def _question(request: TaskRequest) -> str:
        q = request.inputs.get("question")
        return str(q).strip() if isinstance(q, str) and q.strip() else request.task

    def _corpus_summary(self, request: TaskRequest, features: TaskFeatures) -> str:
        parts = [f"{features.document_count} inline document(s) (~{features.corpus_tokens} tokens)"]
        if len(self.builtin):
            parts.append(f"persistent built-in index with {len(self.builtin)} chunks (~{self.builtin.total_tokens()} tokens)")
        parts.append("external retriever injected" if self.retriever is not None else "no external retriever")
        parts.append(f"{len(request.context)} upstream evidence unit(s)")
        return "; ".join(parts)

    @staticmethod
    def _enforce_domain(plan: RAGPlan, profile: DomainProfile, features: TaskFeatures) -> RAGPlan:
        """Domain rules that an LLM plan may not override (high / safety-critical risk)."""
        data = plan.model_dump()
        if profile.risk_level in ("high", "safety_critical"):
            data.update(grade_context=data["route"] != "no_retrieval", verify=True, abstain_when_insufficient=True)
            if data["route"] != "no_retrieval":
                data["citation_style"] = "inline"
        if features.has_identifiers:
            data["query_transforms"] = [t for t in data["query_transforms"] if t not in ("hyde", "step_back")] or ["rewrite"]
        if not features.has_corpus:
            data["route"] = "no_retrieval"
        return RAGPlan(**data)

    # ==================================================== retrieval sources

    def _static_sources(self) -> list[tuple[str, Any]]:
        sources: list[tuple[str, Any]] = []
        if self.retriever is not None:
            sources.append(("injected", self.retriever))
        if len(self.builtin):
            sources.append(("builtin", self.builtin))
        return sources

    async def _sources(self, request: TaskRequest, plan: RAGPlan, ctx: RunContext, use_llm: bool) -> list[tuple[str, Any]]:
        sources = self._static_sources()
        docs = request.inputs.get("documents")
        if docs:
            inline = self._inline_index(docs)
            sources.append(("inline", inline))
        if self.cfg.contextual_prefix == "llm":
            await self._ensure_llm_prefixes(sources, ctx, use_llm)
        if plan.use_hierarchy:
            await self._ensure_hierarchy(sources, ctx, use_llm)
        return sources

    async def _ensure_hierarchy(self, sources: list[tuple[str, Any]], ctx: RunContext, use_llm: bool) -> int:
        """Build RAPTOR-style summary nodes on every built-in source that has none yet (idempotent); returns nodes added."""
        added = 0
        for name, retr in sources:
            if isinstance(retr, BuiltinRetriever) and not retr.summary_chunks():
                summarizer = self._summarizer(ctx) if use_llm and self._calls_left(ctx) > 3 else None
                nodes = await self.index_builder.build_hierarchy(retr, summarizer)
                added += len(nodes)
                ctx.tracer.event("hierarchy_built", source=name, nodes=len(nodes))
        return added

    async def _ensure_llm_prefixes(self, sources: list[tuple[str, Any]], ctx: RunContext, use_llm: bool) -> None:
        """``contextual_prefix='llm'``: replace heuristic chunk prefixes with CONTEXT_PREFIX_PROMPT output once per index.

        One sub-call per chunk (at most ``llm_prefix_max_chunks`` per index),
        each guarded so the last calls of the budget stay reserved for
        generation; offline, under ``deterministic`` or without budget the
        heuristic prefixes are kept.
        """
        pending = [(name, r) for name, r in sources if isinstance(r, BuiltinRetriever) and not r.llm_prefixed and len(r)]
        if not pending or not use_llm:
            return
        if self._calls_left(ctx) <= 3:
            ctx.warn("call budget too small for LLM contextual prefixes; keeping heuristic prefixes")
            return
        prefixer = self._prefixer(ctx)
        for name, retr in pending:
            n = await self.index_builder.apply_llm_prefixes(retr, prefixer, max_chunks=self.cfg.llm_prefix_max_chunks, concurrency=ctx.request.constraints.max_parallelism)
            ctx.tracer.event("llm_prefixes_applied", source=name, chunks=n)

    def _prefixer(self, ctx: RunContext):  # type: ignore[no-untyped-def]
        cfg = self.cfg

        async def prefix(title: str, document_head: str, chunk: str) -> str:
            if self._calls_left(ctx) <= 2:
                raise RuntimeError("call budget reserved for generation")
            req = make_request(
                CONTEXT_PREFIX_PROMPT,
                ctx.profile,
                schema=sanitize_schema(ContextPrefixOut.model_json_schema()),
                effort=cfg.grading_effort,
                model=cfg.sub_call_model,
                title=title,
                document_head=document_head,
                chunk=chunk[: cfg.max_evidence_chars],
            )
            resp = await ctx.llm_call(req, step="context_prefix")
            return ContextPrefixOut.model_validate(resp.parsed).context

        return prefix

    def _inline_index(self, docs: Any) -> BuiltinRetriever:
        key = stable_hash(docs, self.cfg.chunk_size_tokens, self.cfg.chunk_overlap_tokens, self.cfg.contextual_prefix)
        cached = self._inline_indexes.get(key)
        if cached is not None:
            self._inline_indexes.move_to_end(key)
            return cached
        retr = self.index_builder.build(docs)
        self._inline_indexes[key] = retr
        while len(self._inline_indexes) > 8:
            self._inline_indexes.popitem(last=False)
        return retr

    def _summarizer(self, ctx: RunContext):  # type: ignore[no-untyped-def]
        cfg = self.cfg

        async def summarize(title: str, text: str) -> str:
            if self._calls_left(ctx) <= 2:
                raise RuntimeError("call budget reserved for generation")
            req = make_request(
                SUMMARY_PROMPT,
                ctx.profile,
                schema=sanitize_schema(SummaryOut.model_json_schema()),
                effort=cfg.grading_effort,
                model=cfg.sub_call_model,
                title=title,
                text=text[: cfg.max_evidence_chars * 8],
            )
            resp = await ctx.llm_call(req, step="summarize")
            return SummaryOut.model_validate(resp.parsed).summary

        return summarize

    # ====================================================== one pipeline pass

    async def _run_pipeline(
        self,
        request: TaskRequest,
        plan: RAGPlan,
        ctx: RunContext,
        *,
        question: str,
        features: TaskFeatures,
        use_llm: bool,
        aux_llm: Optional[bool] = None,
        sources: list[tuple[str, Any]],
        suffix: str = "",
    ) -> _Attempt:
        cfg = self.cfg
        aux = use_llm if aux_llm is None else aux_llm  # advisory stages (off under constraints.deterministic)
        att = _Attempt(plan=plan)
        provided = list(request.context)
        multi_hop = plan.route in _MULTI
        max_par = request.constraints.max_parallelism

        if plan.route != "no_retrieval":
            # ---------------------------------------------------- transform
            with ctx.step("transform" + suffix) as st:
                att.variants = await self.transformer.transform(
                    question,
                    plan.query_transforms,
                    ctx=ctx,
                    use_llm=aux and self._calls_left(ctx) >= 3,
                    features=features,
                    retriever=self.retriever,
                )
                gloss = glossary_expansion(question, ctx.profile.glossary)
                if gloss and all(gloss.lower() != v.text.lower() for v in att.variants):
                    att.variants.append(QueryVariant(kind="glossary", text=gloss, weight=0.8))
                st.summary = f"{len(att.variants)} query variant(s)"
                st.details.update(variants=[v.model_dump() for v in att.variants])

            # ----------------------------------------------------- retrieve
            with ctx.step("retrieve" + suffix) as st:
                gate = self._full_corpus_evidence(sources)
                if gate is not None:
                    att.evidence = gate
                    att.full_corpus_used = True
                    st.summary = f"full corpus ({len(gate)} units) under the {cfg.full_corpus_gate_tokens}-token gate"
                else:
                    att.evidence = await self._retrieve_fused(
                        att.variants, k=plan.k, sources=sources, provided=provided, filters=None, ctx=ctx, max_parallelism=max_par
                    )
                    st.summary = f"{len(att.evidence)} evidence units after RRF (k={plan.k})"
                st.details.update(evidence_ids=[e.id for e in att.evidence])

            # -------------------------------------------------------- grade
            if plan.grade_context:
                with ctx.step("grade" + suffix) as st:
                    if att.evidence:
                        kept, decision, grades = await self.grader.grade(question, att.evidence, ctx=ctx, use_llm=aux and self._calls_left(ctx) >= 2)
                    else:  # nothing retrieved at all: the strongest INCORRECT case, so CRAG's fallback must still run
                        kept, decision, grades = [], "incorrect", []
                    rounds = 0
                    while decision == "incorrect" and rounds < cfg.max_fallback_rounds and (sources or self.fallback_retriever is not None):
                        rounds += 1
                        ctx.warn(f"CRAG grade 'incorrect' on round {rounds}; running fallback retrieval")
                        fb_variants = [QueryVariant(kind="rewrite", text=keyword_query(question)), QueryVariant(kind="step_back", text=step_back_query(question), weight=0.6)]
                        fb_sources = [("fallback", self.fallback_retriever)] if self.fallback_retriever is not None else sources
                        extra = await self._retrieve_fused(fb_variants, k=plan.k * 2, sources=fb_sources, provided=(), filters=None, ctx=ctx, max_parallelism=max_par)
                        pool = deduplicate([*att.evidence, *extra])
                        if pool:
                            kept, decision, grades = await self.grader.grade(question, pool, ctx=ctx, use_llm=aux and self._calls_left(ctx) >= 2)
                    att.grade_decision, att.grades = decision, grades
                    att.evidence = kept
                    st.summary = f"decision={decision}, kept {len(kept)}"
                    st.details.update(decision=decision, fallback_rounds=rounds)

            # ------------------------------------------------------ iterate
            if multi_hop and plan.max_hops > 1:
                with ctx.step("iterate" + suffix) as st:

                    async def hop_retrieve(q: str, k: int) -> list[Evidence]:
                        hits = await self._retrieve_fused([QueryVariant(kind="hop", text=q)], k=k, sources=sources, provided=(), filters=None, ctx=ctx, max_parallelism=max_par)
                        if plan.grade_context:  # knowledge-strip refinement on new units, lexical (no extra call)
                            graded = self.grader.grade_heuristic(question, hits)
                            by_id = {g.doc_id: g for g in graded}
                            out = []
                            for ev in hits:
                                g = by_id[ev.id]
                                if g.label == "incorrect":
                                    continue
                                meta = dict(ev.metadata)
                                meta.update(grade=g.label, grade_score=round(g.score, 4), grade_reason=g.reason)
                                out.append(ev.model_copy(update={"metadata": meta}))
                            return out
                        return hits

                    att.evidence, att.hops, att.reasoning = await self.loop.run(
                        question, att.evidence, hop_retrieve, max_hops=plan.max_hops, ctx=ctx, use_llm=aux
                    )
                    st.summary = f"{len(att.hops)} hop(s), {len(att.evidence)} units"
                    st.details.update(hops=[h.model_dump() for h in att.hops])

            # ----------------------------------------------- compress/order
            with ctx.step("compress" + suffix) as st:
                before = len(att.evidence)
                if plan.compress and att.evidence:
                    att.evidence = self.compressor.compress(att.evidence, question, token_budget=plan.context_token_budget, multi_hop=multi_hop)
                att.evidence = self.orderer.order(att.evidence)
                att.compressed_units = sum(1 for e in att.evidence if e.metadata.get("compressed"))
                att.context_tokens = sum(estimate_tokens(e.content) for e in att.evidence)
                st.summary = f"{before} -> {len(att.evidence)} units, ~{att.context_tokens} tokens, {att.compressed_units} compressed"
        elif provided:
            att.evidence = self.orderer.order(deduplicate(provided))
            att.context_tokens = sum(estimate_tokens(e.content) for e in att.evidence)

        # ------------------------------------------------------- generate
        with ctx.step("generate" + suffix) as st:
            if use_llm and self._calls_left(ctx) >= 1:
                att.generation = await self.generator.generate_llm(
                    question,
                    att.evidence,
                    ctx,
                    answer_type=plan.expected_answer_type,
                    citation_style=plan.citation_style,
                    require_citations=cfg.require_citations,
                )
            else:
                if use_llm:
                    ctx.warn("call budget exhausted before generation; using the extractive answer")
                    note_degraded(ctx, "generate")
                att.generation = self.generator.generate_heuristic(question, att.evidence, answer_type=plan.expected_answer_type, multi_hop=multi_hop)
            if att.generation.abstained and not plan.abstain_when_insufficient and att.generation.answer.strip():
                att.generation.abstained = False  # the plan asked for a best-effort answer
            att.statements = split_statements(att.generation.answer, [e.id for e in att.evidence])
            st.summary = "abstained" if att.generation.abstained else f"{len(att.statements)} statement(s), {len(att.generation.citations)} citation(s)"

        # --------------------------------------------------------- verify
        if plan.verify and not att.generation.abstained and att.statements:
            with ctx.step("verify" + suffix) as st:
                if self.verifier is not None:
                    try:
                        verdict = await self._verify_external(request, question, att, ctx)
                        att.verdict = verdict
                        att.reflection = self.judge.judge_heuristic(att.statements, att.evidence, question=question).model_copy(
                            update={"support_fraction": float(verdict.score), "is_supported": "fully" if verdict.passed else "partially", "method": "verifier"}
                        )
                        st.summary = f"verifier {verdict.method}: passed={verdict.passed} score={verdict.score:.2f}"
                    except Exception as exc:  # noqa: BLE001 - verifier failure is a review condition, not a crash
                        ctx.warn(f"injected verifier failed ({type(exc).__name__}); using built-in reflection")
                        note_degraded(ctx, "verify")
                        att.reflection = self.judge.judge_heuristic(att.statements, att.evidence, question=question)
                else:
                    effort = self._verify_effort(ctx.profile)
                    if aux and self._calls_left(ctx) >= 1 and att.evidence:
                        att.reflection = await self.judge.judge_llm(question, att.generation.answer, att.statements, att.evidence, ctx, effort=effort)
                    else:
                        att.reflection = self.judge.judge_heuristic(att.statements, att.evidence, question=question)
                    st.summary = f"reflection ({att.reflection.method}): supported={att.reflection.is_supported} useful={att.reflection.is_useful}"
        return att

    async def _verify_external(self, request: TaskRequest, question: str, att: _Attempt, ctx: RunContext) -> Verdict:
        """Run the injected verifier and charge its usage to this run (ARCHITECTURE 3.1.3 / 5).

        A :class:`PipelineComponent` verifier (the answer verification model)
        is invoked through its ``run()`` with a child request carrying the
        remaining call / cost / latency budget; its ``result.usage`` is
        charged with ``ctx.charge`` and the verdict comes from
        ``result.output.to_verdict()`` (or a ``Verdict`` output). A plain
        ``Verifier`` protocol object is called through ``verify()``; it may
        report its spend as ``verdict.details["usage"]`` (a ``Usage`` or dict).
        """
        answer = att.generation.answer
        if isinstance(self.verifier, PipelineComponent):
            child = request.derive(question, inputs={"answer": answer}, context=list(att.evidence), constraints=self._child_constraints(request, ctx))
            res = await self.verifier.run(child)
            self._charge_nested(ctx, res.usage, what=f"verifier {self.verifier.spec.name}")
            if res.status == Status.FAILED:
                raise RuntimeError("; ".join(res.errors[:2]) or "verifier run failed")
            out = res.output
            if hasattr(out, "to_verdict"):
                return out.to_verdict()
            if isinstance(out, Verdict):
                return out
            if isinstance(out, dict) and {"method", "passed", "score"} <= set(out):
                return Verdict.model_validate(out)
            return Verdict(
                method=self.verifier.spec.name,
                passed=res.status == Status.SUCCESS,
                score=float(res.confidence),
                issues=list(res.warnings)[:20],
                details={"status": res.status.value},
            )
        verdict = await self.verifier.verify(answer, task=question, evidence=att.evidence)  # type: ignore[union-attr]
        usage = verdict.details.get("usage") if isinstance(verdict.details, dict) else None
        if isinstance(usage, dict):
            usage = Usage(**usage)
        if isinstance(usage, Usage):
            self._charge_nested(ctx, usage, what="verifier")
        return verdict

    @staticmethod
    def _child_constraints(request: TaskRequest, ctx: RunContext) -> Constraints:
        """The parent's constraints narrowed to what is left of its budget, for a nested ``run()``."""
        update: dict[str, Any] = {}
        rem_calls = ctx.budget.remaining_calls()
        if rem_calls is not None:
            update["max_llm_calls"] = rem_calls
        rem_cost = ctx.budget.remaining_cost()
        if rem_cost is not None:
            update["max_cost_usd"] = rem_cost
        rem_time = ctx.remaining_time()
        if rem_time is not None:
            update["max_latency_s"] = max(rem_time, 1e-3)
        return request.constraints.model_copy(update=update)

    @staticmethod
    def _charge_nested(ctx: RunContext, usage: Usage, *, what: str) -> None:
        """``ctx.charge`` a nested component's usage; overspend is recorded (the work is already done), not raised."""
        try:
            ctx.charge(usage)
        except BudgetExceededError as exc:
            ctx.warn(f"{what} spend exceeded the run budget ({exc})")

    def _verify_effort(self, profile: DomainProfile) -> str:
        if profile.risk_level == "safety_critical":
            return self.cfg.verification_effort_safety_critical
        if profile.risk_level == "high":
            return self.cfg.generation_effort
        return self.cfg.grading_effort

    def _full_corpus_evidence(self, sources: list[tuple[str, Any]]) -> Optional[list[Evidence]]:
        gate = self.cfg.full_corpus_gate_tokens
        if gate <= 0 or not sources:
            return None
        if any(not isinstance(r, BuiltinRetriever) for _, r in sources):
            return None  # an external retriever's corpus size is unknown
        total = sum(r.total_tokens() for _, r in sources)
        if total == 0 or total > gate:
            return None
        out: list[Evidence] = []
        for _, r in sources:
            out.extend(r.all_evidence())
        return deduplicate(out)

    async def _retrieve_fused(
        self,
        variants: Sequence[QueryVariant],
        *,
        k: int,
        sources: list[tuple[str, Any]],
        provided: Sequence[Evidence],
        filters: Optional[dict[str, Any]],
        ctx: Optional[RunContext],
        max_parallelism: int,
    ) -> list[Evidence]:
        """Fan every variant out to every source, fuse with RRF, return the top-k as Evidence."""
        cfg = self.cfg
        if ctx is not None and ctx.deadline_exceeded() and len(variants) > 1:
            # architecture 3.2: respect the deadline before a fan-out; keep only the first (literal) variant
            ctx.warn("latency budget exhausted; retrieving with the first query variant only")
            note_degraded(ctx, "retrieve")
            variants = list(variants[:1])
        jobs = [(name, retr, v) for name, retr in sources for v in variants]
        pool: dict[str, Evidence] = {}
        owner: dict[str, tuple[str, str]] = {}  # pooled id -> (source name, content) of its first occurrence
        collided: set[str] = set()

        def pooled_id(name: str, ev: Evidence) -> Evidence:
            """Same id from another source with different text (e.g. two auto-generated ``doc-1``): namespace it."""
            first = owner.setdefault(ev.id, (name, ev.content))
            if first[0] == name or first[1] == ev.content:
                return ev
            new_id = f"{name}:{ev.id}"
            if ev.id not in collided:
                collided.add(ev.id)
                if ctx is not None:
                    ctx.warn(f"evidence id {ev.id!r} returned by '{first[0]}' and '{name}' with different text; the '{name}' unit is cited as {new_id!r}")
            owner.setdefault(new_id, (name, ev.content))
            return ev.model_copy(update={"id": new_id})

        async def one(job: tuple[str, Any, QueryVariant]) -> tuple[str, float, list[Evidence]]:
            name, retr, variant = job
            try:
                hits = await self._cached_retrieve(name, retr, variant.text, k=cfg.candidates_per_variant, filters=filters, ctx=ctx)
            except Exception as exc:  # noqa: BLE001 - one leg failing must not sink the run
                if ctx is not None:
                    ctx.warn(f"retrieval leg {name}/{variant.kind} failed ({type(exc).__name__})")
                    note_degraded(ctx, f"retrieval leg {name}")
                hits = []
            return name, variant.weight, hits

        results: list[Any]
        if jobs and self.executor is not None:
            try:
                results = await self.executor.map(one, jobs, concurrency=max_parallelism)
            except Exception as exc:  # noqa: BLE001 - fall back to the local gather
                if ctx is not None:
                    ctx.warn(f"executor failed ({type(exc).__name__}); retrieving with local gather")
                    note_degraded(ctx, "executor")
                results = await gather_limited([one(j) for j in jobs], limit=max_parallelism)
        elif jobs:
            results = await gather_limited([one(j) for j in jobs], limit=max_parallelism)
        else:
            results = []

        rankings: list[list[tuple[str, float]]] = []
        weights: list[float] = []
        legs: dict[str, list[str]] = {}
        for res in results:
            if not isinstance(res, tuple) or len(res) != 3:
                continue
            name, weight, hits = res
            ranking: list[tuple[str, float]] = []
            for ev in hits:
                ev = pooled_id(name, ev)
                ranking.append((ev.id, ev.score))
                cur = pool.get(ev.id)
                if cur is None or ev.score > cur.score:
                    pool[ev.id] = ev
                legs.setdefault(ev.id, []).append(name)
            if ranking:
                rankings.append(ranking)
                weights.append(weight)
        if provided:
            ranking = []
            for ev in provided:
                ev = pooled_id("provided", ev)
                ranking.append((ev.id, ev.score))
                pool.setdefault(ev.id, ev.model_copy(update={"retriever": ev.retriever or "provided"}))
                legs.setdefault(ev.id, []).append("provided")
            rankings.append(ranking)
            weights.append(1.0)
        if not rankings:
            return []
        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k, weights=weights)
        out: list[Evidence] = []
        for rank, (doc_id, score) in enumerate(fused[:k], 1):
            ev = pool[doc_id]
            meta = dict(ev.metadata)
            meta.update(fused_score=round(score, 6), legs=sorted(set(legs.get(doc_id, []))), leg_score=ev.score, leg_rank=ev.rank)
            out.append(ev.model_copy(update={"score": score, "rank": rank, "retriever": "fused", "metadata": meta}))
        return deduplicate(out)

    async def _cached_retrieve(self, name: str, retr: Any, query: str, *, k: int, filters: Optional[dict[str, Any]], ctx: Optional[RunContext] = None) -> list[Evidence]:
        if self.cache is None:
            return await retr.retrieve(query, k=k, filters=filters)
        # a retriever without a content ``version`` is keyed by its identity and the model's cache epoch, so
        # re-indexing it only needs invalidate_retrieval_cache() instead of serving stale hits forever
        version = getattr(retr, "version", None) or f"unversioned:{type(retr).__name__}:{id(retr)}:{self.retrieval_cache_epoch}"
        key = stable_hash("pipeline_rag.retrieve", name, version, query, k, filters)
        try:
            hit = await self.cache.get(key)
        except Exception as exc:  # noqa: BLE001 - cache errors are misses
            self._warn_cache(ctx, exc)
            hit = None
        if isinstance(hit, list):
            try:
                return [Evidence.model_validate(h) for h in hit]
            except Exception:  # noqa: BLE001 - corrupt entry: treat as a miss
                pass
        hits = await retr.retrieve(query, k=k, filters=filters)
        try:
            await self.cache.set(key, [h.model_dump(mode="json") for h in hits], tags=["pipeline_rag", f"index:{version}", f"source:{name}"])
        except Exception as exc:  # noqa: BLE001 - a failed store is only a lost optimisation
            self._warn_cache(ctx, exc)
        return hits

    def _warn_cache(self, ctx: Optional[RunContext], exc: Exception) -> None:
        """Surface the first cache failure (per model) instead of hiding a misconfigured cache."""
        if ctx is not None and not self._cache_warned:
            self._cache_warned = True
            ctx.warn(f"retrieval cache error ({type(exc).__name__}: {str(exc)[:80]}); retrieving without the cache")

    # ============================================================ finishing

    def _finish(self, request: TaskRequest, plan: RAGPlan, att: _Attempt, ctx: RunContext, *, question: str, routes_tried: list[str]) -> TaskResult:
        cfg = self.cfg
        gen = att.generation
        profile = ctx.profile
        high_risk = profile.risk_level in ("high", "safety_critical")
        statements = att.statements
        evidence_ids = [e.id for e in att.evidence]
        cited = cited_ids(gen, statements)
        grade_by_id = {e.id: e.metadata.get("grade_score") for e in att.evidence}

        # ---- signals
        self_reported: Optional[float] = gen.confidence if (gen.answer or gen.abstained) else None
        coverage: Optional[float] = None
        if plan.route != "no_retrieval" and statements:
            coverage = citation_coverage(statements) if cfg.require_citations else None
        support: Optional[float] = None
        if att.reflection is not None:
            support = att.reflection.support_fraction
        quality: Optional[float] = None
        graded = [grade_by_id[c] for c in cited if isinstance(grade_by_id.get(c), (int, float))]
        if not graded:
            graded = [v for v in grade_by_id.values() if isinstance(v, (int, float))]
        if graded:
            quality = sum((float(g) + 1.0) / 2.0 for g in graded) / len(graded)
        elif att.evidence and plan.route != "no_retrieval":
            quality = 0.5  # retrieved but ungraded: neutral prior
        weights = cfg.confidence_weights()
        combined, basis = aggregate_confidence(
            {"self_reported": self_reported, "citation_coverage": coverage, "support": support, "evidence_quality": quality},
            weights,
            abstained=gen.abstained,
        )
        threshold = max(profile.min_confidence_to_accept, request.constraints.min_confidence)
        basis = f"{basis}; accept threshold {threshold:.2f} ({profile.name})"

        # ---- status
        status = Status.SUCCESS
        reasons: list[str] = []
        if gen.abstained:
            status = Status.NEEDS_REVIEW
            reasons.append("abstained: " + (gen.abstention_reason or "insufficient evidence"))
        if combined < threshold:
            status = Status.NEEDS_REVIEW
            reasons.append(f"confidence {combined:.2f} below threshold {threshold:.2f}")
        if att.verdict is not None and not att.verdict.passed:
            status = Status.NEEDS_REVIEW
            reasons.append(f"verifier {att.verdict.method} failed: " + "; ".join(att.verdict.issues[:3]))
        if att.reflection is not None and not att.reflection.is_relevant and not gen.abstained and plan.route != "no_retrieval":
            status = Status.NEEDS_REVIEW
            reasons.append(f"reflection ({att.reflection.method}) IsRel=false: the evidence does not address the question; best-effort answer")
        if high_risk and not gen.abstained and (gen.unsupported_claims or (att.reflection is not None and att.reflection.unsupported_statements)):
            status = Status.NEEDS_REVIEW
            reasons.append(f"{profile.risk_level} domain with unsupported statements")
        if high_risk and plan.route != "no_retrieval" and not gen.abstained and coverage is not None and coverage < cfg.min_citation_coverage:
            status = Status.NEEDS_REVIEW
            reasons.append(f"citation coverage {coverage:.2f} below {cfg.min_citation_coverage:.2f}")
        degraded = degraded_stages(ctx)
        if status == Status.SUCCESS and degraded:
            status = Status.PARTIAL
            reasons.append("stages degraded to their fallback: " + ", ".join(degraded))

        signals = ConfidenceSignals(
            self_reported=self_reported,
            citation_coverage=coverage,
            support=support,
            evidence_quality=quality,
            combined=combined,
            threshold=threshold,
            basis=basis,
        )
        answer_text = render_answer(gen, att.evidence, plan.citation_style)
        output = RAGOutput(
            question=question,
            answer=gen.answer,
            answer_text=answer_text,
            citations=gen.citations,
            unsupported_claims=list(gen.unsupported_claims),
            abstained=gen.abstained,
            abstention_reason=gen.abstention_reason,
            route=plan.route,
            routes_tried=routes_tried,
            query_variants=att.variants,
            grade_decision=att.grade_decision,
            grades=att.grades,
            evidence_ids=evidence_ids,
            cited_ids=cited,
            statements=statements,
            hops=att.hops,
            reasoning=att.reasoning,
            reflection=att.reflection,
            verdict=att.verdict.model_dump(mode="json") if att.verdict is not None else None,
            confidence=signals,
            full_corpus_used=att.full_corpus_used,
            compressed_units=att.compressed_units,
            context_tokens=att.context_tokens,
        )
        result = self.new_result(
            request,
            status=status,
            output=output,
            output_text=answer_text,
            evidence=list(att.evidence),
            confidence=combined,
            confidence_basis=basis,
            warnings=list(reasons),
            metadata={"plan": plan.model_dump(mode="json"), "routes_tried": routes_tried, "review_reasons": reasons, "degraded_stages": degraded},
        )
        return result


__all__ = ["ConfidenceSignals", "ContextPrefixOut", "PipelineRAGModel", "RAGOutput", "SummaryOut", "aggregate_confidence"]
