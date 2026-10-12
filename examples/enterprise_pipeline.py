"""Enterprise pipeline: the six models wired into one orchestrated, traced, budgeted run.

Wiring (docs/ARCHITECTURE.md section 5, docs/COMPOSITION_PATTERNS.md section 0)::

    TaskRequest
      -> index          HybridRetrievalModel indexes request documents (Retriever)
      -> cache lookup   CachingModel.cache (ResponseCache) serves repeated requests as CACHED
      -> route          TaskDecompositionModel.plan_for(): simple -> RAG, complex -> decomposition
      -> solve          PipelineRAGModel(retriever, verifier, executor, cache)            [simple]
                        TaskDecompositionModel(solver=RAG, executor, verifier) -> waves  [complex]
      -> verify_final   AnswerVerificationModel gates the synthesised answer (Verifier)
      -> store          the result is cached, tagged with the documents it cites
    One CachedLLMClient (CachingModel.wrap) is the ``llm`` of every component and one
    Tracer is shared, so ``tracer.tree()`` shows the whole run and ``result.usage`` is
    the rolled-up cost.

Run offline (deterministic MockLLM, heuristic stages, no network)::

    python examples/enterprise_pipeline.py --offline
    python examples/enterprise_pipeline.py --offline --task "What is the NPSH margin?" --json

With ``ANTHROPIC_API_KEY`` set and no ``--offline`` flag the same code runs against Claude.

The file is self-contained: it sets ``sys.path`` so it runs from the repository root
without ``PYTHONPATH``. Where a package API made the wiring awkward, a small local
adapter absorbs the friction; each adapter's docstring names the friction so it can be
fed back to the package owners.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence, TypeVar

ROOT = Path(__file__).resolve().parents[1]
for _folder in (
    "common",
    "hybrid_retrieval_model",
    "task_decomposition_model",
    "answer_verification_model",
    "parallelization_model",
    "caching_model",
    "pipeline_rag_model",
):
    _p = str(ROOT / _folder)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from pipeline_core import (  # noqa: E402
    BaseLLMClient,
    Evidence,
    Executor,
    MockLLM,
    ResponseCache,
    Retriever,
    Status,
    StepRecord,
    TaskRequest,
    TaskResult,
    Tracer,
    Usage,
    Verdict,
    Verifier,
    get_profile,
    make_llm,
    normalize_text,
    run_sync,
    stable_hash,
)

from answer_verification import AnswerVerificationModel  # noqa: E402
from caching import CachedLLMClient, CachingModel  # noqa: E402
from hybrid_retrieval import HybridRetrievalModel  # noqa: E402
from parallelization import ParallelizationModel  # noqa: E402
from parallelization.executor import ParallelExecutor  # noqa: E402
from pipeline_rag import PipelineRAGModel  # noqa: E402
from task_decomposition import DecompositionPlan, TaskDecompositionModel  # noqa: E402

T = TypeVar("T")
R = TypeVar("R")

COMPONENT = "enterprise_pipeline"
DEFAULT_DOMAIN = "mechanical_engineering"
Route = Literal["rag", "decompose"]

# ---------------------------------------------------------------------------
# Built-in engineering corpus: pump station PS-7 (specs, calculation note, test
# logs, standards clauses, an incident report, an FMEA, a maintenance procedure)
# ---------------------------------------------------------------------------

CORPUS: list[dict[str, Any]] = [
    {
        "id": "SPEC-PS7-HYD",
        "text": (
            "Pump Station PS-7 Hydraulic Specification, Rev C, section 3 Hydraulic sizing. "
            "The rated duty point is 320 m3/h at 42 m total head. The hydraulic sizing shall place the duty point "
            "between 80 percent and 110 percent of the best efficiency point (BEP) flow. Rated speed 1480 rpm. "
            "The system curve is derived in calculation note CALC-PS7-01. Minimum continuous stable flow is 110 m3/h."
        ),
        "metadata": {"type": "specification", "title": "PS-7 Hydraulic Specification", "revision": "C"},
    },
    {
        "id": "SPEC-PS7-MECH",
        "text": (
            "Pump Station PS-7 Mechanical Specification, Rev C, section 4 Pressure casing. "
            "The pump casing shall be ASTM A216 WCB carbon steel. The specification requires a corrosion allowance of 3 mm "
            "on all wetted casing surfaces, consistent with API 610 clause 6.3.1. The impeller shall be duplex stainless steel "
            "ASTM A890 Grade 4A. Casing design pressure 16 bar at 80 degC."
        ),
        "metadata": {"type": "specification", "title": "PS-7 Mechanical Specification", "revision": "C"},
    },
    {
        "id": "SPEC-PS7-ELEC",
        "text": (
            "Pump Station PS-7 Electrical Specification, Rev B, section 5 Drivers. "
            "Each pump motor rating is 75 kW, 400 V, 50 Hz, 4-pole, 1480 rpm, IE3 efficiency class, service factor 1.15. "
            "The motor rating shall exceed the pump shaft power at the end of curve (run-out) by at least 10 percent. "
            "Motor protection: thermal overload and vibration trip at 7.1 mm/s RMS."
        ),
        "metadata": {"type": "specification", "title": "PS-7 Electrical Specification", "revision": "B"},
    },
    {
        "id": "CALC-PS7-01",
        "text": (
            "Calculation note CALC-PS7-01 System curve and hydraulic sizing, Rev 2. "
            "Static head 31.0 m; friction losses at 320 m3/h are 10.6 m (DN300 pipeline, 840 m, C=120), giving a system head of 41.6 m "
            "against the specified 42 m duty point head. The selected pump curve gives 43.1 m at 320 m3/h, a head margin of 1.1 m (2.6 percent). "
            "BEP flow of the selected impeller is 350 m3/h, so the duty point sits at 91 percent of BEP, inside the 80-110 percent band. "
            "Run-out flow 400 m3/h at 36 m head requires 61.0 kW shaft power; rated motor 75 kW gives a 23 percent margin."
        ),
        "metadata": {"type": "calculation note", "title": "CALC-PS7-01 System curve", "revision": "2"},
    },
    {
        "id": "FAT-P101A",
        "text": (
            "Factory acceptance test log, pump P-101A, test TL-2026-014. Measured flow 318 m3/h at 41.6 m head; "
            "shaft power 58.2 kW; pump efficiency 79 percent; measured NPSH required 2.6 m at the duty point. "
            "Vibration 2.1 mm/s RMS at the bearing housings. Result: accepted, within ISO 9906 grade 2B tolerances."
        ),
        "metadata": {"type": "test report", "title": "FAT log P-101A", "revision": "1"},
    },
    {
        "id": "FAT-P101B",
        "text": (
            "Factory acceptance test log, pump P-101B, test TL-2026-015. Measured flow 322 m3/h at 42.3 m head; "
            "shaft power 59.0 kW at the duty point and 61.4 kW at run-out; motor current 112 A at run-out against a 135 A rating. "
            "Vibration 2.6 mm/s RMS. Result: accepted; the motor rating of 75 kW exceeds the run-out shaft power by 22 percent."
        ),
        "metadata": {"type": "test report", "title": "FAT log P-101B", "revision": "1"},
    },
    {
        "id": "STD-API610",
        "text": (
            "API 610 (ISO 13709) selected clauses. Clause 6.3.1: pressure casings shall have a minimum corrosion allowance of 3 mm. "
            "Clause 6.1.9: the NPSH margin, NPSH available minus NPSH required, shall be at least 1 m at the rated duty point. "
            "Clause 6.1.11: pumps shall operate between 80 percent and 110 percent of BEP flow at rated conditions. "
            "Clause 7.1.4: the motor rating shall be at least 110 percent of the maximum power on the pump curve when the rated power is above 55 kW."
        ),
        "metadata": {"type": "standard", "title": "API 610 clauses", "revision": "12th ed."},
    },
    {
        "id": "STD-ISO10816",
        "text": (
            "ISO 10816-3 vibration evaluation for pumps above 15 kW with rigid foundations. Zone A/B boundary 2.3 mm/s RMS, "
            "zone B/C boundary 4.5 mm/s RMS, zone C/D boundary 7.1 mm/s RMS. Operation in zone D is normally considered to cause damage; "
            "the trip setpoint shall be set at the C/D boundary."
        ),
        "metadata": {"type": "standard", "title": "ISO 10816-3 vibration zones", "revision": "2009"},
    },
    {
        "id": "NPSH-PS7",
        "text": (
            "PS-7 suction conditions, data sheet DS-PS7-02. NPSH available at the low wet-well level is 4.1 m. "
            "Measured NPSH required is 2.6 m, so the NPSH margin is 1.5 m, above the API 610 minimum of 1 m. "
            "At the low-low level alarm the NPSH available falls to 3.0 m and the margin reduces to 0.4 m."
        ),
        "metadata": {"type": "data sheet", "title": "PS-7 suction conditions", "revision": "2"},
    },
    {
        "id": "INC-2026-03",
        "text": (
            "Trip report INC-2026-03, pump P-101B. On 14 March 2026 P-101B tripped on high vibration (7.8 mm/s RMS) after cavitation "
            "at a low suction level. Root cause: the suction strainer was 70 percent blocked, lowering NPSH available below NPSH required. "
            "Open risks from the trip report: the strainer inspection interval is not defined in MP-14; the low-level alarm setpoint "
            "was never verified against the NPSH margin; the vibration trip setpoint of 7.1 mm/s allowed sustained operation in ISO 10816-3 zone C."
        ),
        "metadata": {"type": "incident report", "title": "Trip report INC-2026-03", "revision": "1"},
    },
    {
        "id": "FMEA-PS7",
        "text": (
            "FMEA PS-7 pumps, Rev A. Failure mode: cavitation from low suction level, severity 8, occurrence 4, detection 5, RPN 160; "
            "recommended action: interlock the low-low level alarm with a pump stop. Failure mode: casing wall loss beyond the 3 mm corrosion allowance, "
            "severity 9, occurrence 2, detection 6, RPN 108; recommended action: ultrasonic thickness survey every 4 years."
        ),
        "metadata": {"type": "FMEA", "title": "FMEA PS-7 pumps", "revision": "A"},
    },
    {
        "id": "MP-14",
        "text": (
            "Maintenance procedure MP-14 Rev D, suction strainer inspection. Inspect and clean the suction strainers of P-101A and P-101B; "
            "the inspection interval is to be defined from the differential pressure trend (open action from INC-2026-03). "
            "Record the differential pressure across the strainer before and after cleaning."
        ),
        "metadata": {"type": "maintenance procedure", "title": "MP-14 strainer inspection", "revision": "D"},
    },
]

DEMO_TASKS: list[dict[str, str]] = [
    {
        "label": "simple factual question",
        "task": "What corrosion allowance does the specification require for the pump casing?",
    },
    {
        "label": "complex multi-part design review",
        "task": (
            "Review the pump station PS-7 design: check the hydraulic sizing against the duty point, "
            "verify the motor rating against the measured shaft power, check the NPSH margin against API 610, "
            "and list the open risks from the P-101B trip report."
        ),
    },
]


# ---------------------------------------------------------------------------
# Local adapters (friction absorbed here, reported in the run's api_issues)
# ---------------------------------------------------------------------------


class OfflineAwareCachedLLMClient(CachedLLMClient, MockLLM):
    """A ``CachedLLMClient`` that still counts as a ``MockLLM``.

    Friction: ``RunContext.offline`` is ``isinstance(llm, MockLLM)``. Wrapping the offline
    mock in ``CachingModel.wrap()`` hides that, so every component believes it is online and
    drives the mock through its LLM paths (schema examples such as "mock The answer")
    instead of its heuristic offline paths. Inheriting from both keeps the shared cache in
    front of the mock while ``ctx.offline`` stays true. ``CachedLLMClient.__init__`` reaches
    ``MockLLM.__init__`` through the MRO with the model id in the ``responder`` slot, so the
    mock attributes are restored afterwards.
    """

    name = "cached-mock"

    def __init__(self, inner: MockLLM, **kwargs: Any) -> None:
        CachedLLMClient.__init__(self, inner, **kwargs)
        self.model = inner.model
        self.responder = None
        self.calls = inner.calls  # share the recorded-request list with the wrapped mock
        self.latency_s = 0.0
        self.simulate_cost_as = None


def make_shared_client(caching: CachingModel, inner: BaseLLMClient, *, profile: str | None = None) -> CachedLLMClient:
    """Wrap ``inner`` with the caching model's tiers; offline mocks keep their ``MockLLM`` identity."""
    if isinstance(inner, MockLLM):
        client: CachedLLMClient = OfflineAwareCachedLLMClient(
            inner,
            backend=caching.backend,
            semantic=caching.semantic,
            config=caching.config,
            planner=caching.planner,
            profile=profile,
            clock=caching.clock,
        )
        caching.clients.append(client)  # so CachingModel.metrics()/savings_report() see it
        return client
    return caching.wrap(inner, profile=profile)


_TIER: contextvars.ContextVar[int] = contextvars.ContextVar("enterprise_pipeline_executor_tier", default=0)


class ReentrantExecutor:
    """``Executor`` facade over one ``ParallelExecutor`` that is safe to nest and shows up in the trace.

    Friction: ``ParallelExecutor.map`` takes Bulkhead slots from one ``default`` pool. The
    decomposition scheduler holds a slot per sub-task while each sub-task's RAG run (retrieval
    fan-out) and verifier (sample fan-out) wait for slots from the same pool, which deadlocks
    once the outer wave fills the pool. This facade routes each nesting level to its own
    bulkhead resource (``tier0``, ``tier1`` ...) through ``map_outcomes(resource=...)``, so the
    rate limiter, retry budget, breaker and stats stay shared while nested fan-outs cannot
    starve each other. It also opens a span, which the direct API does not.
    """

    def __init__(self, inner: ParallelExecutor, tracer: Tracer) -> None:
        self.inner = inner
        self.tracer = tracer

    @property
    def stats(self):  # type: ignore[no-untyped-def]
        return self.inner.stats

    def snapshot(self) -> dict[str, Any]:
        return self.inner.snapshot()

    async def map(self, fn: Callable[[T], Awaitable[R]], items: Sequence[T], *, concurrency: Optional[int] = None) -> list[Any]:
        tier = _TIER.get()
        token = _TIER.set(tier + 1)
        try:
            with self.tracer.span("parallelization.map", "parallelization", items=len(items), tier=tier, concurrency=concurrency) as sp:
                outcomes = await self.inner.map_outcomes(fn, items, concurrency=concurrency, resource=f"tier{tier}")
                sp.attributes["failed"] = sum(1 for o in outcomes if not o.ok)
                return [o.value if o.ok else o.exception for o in outcomes]
        finally:
            _TIER.reset(token)


class TracedRetriever:
    """``Retriever`` facade over ``HybridRetrievalModel`` that traces the direct API and exposes an index version.

    Friction: ``HybridRetrievalModel.retrieve()`` (the protocol API) opens no span under the
    shared tracer, and the model has no ``version`` attribute although ``PipelineRAGModel``
    keys its retrieval cache on ``getattr(retriever, "version", "unversioned")``; without one a
    re-indexed corpus would serve stale cached retrievals.
    """

    def __init__(self, model: HybridRetrievalModel, tracer: Tracer) -> None:
        self.model = model
        self.tracer = tracer

    @property
    def version(self) -> str:
        return f"v{self.model.store.version}"

    def index(self, documents: Any) -> int:
        return self.model.index(documents)

    def add_documents(self, documents: Any) -> int:
        return self.model.add_documents(documents)

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        with self.tracer.span("hybrid_retrieval.retrieve", "hybrid_retrieval", k=k, index_version=self.version) as sp:
            hits = await self.model.retrieve(query, k=k, filters=filters)
            sp.attributes["results"] = len(hits)
            return hits


_CITATION_RE = re.compile(r"\s*\[[^\[\]\s]{1,80}\]")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s.*$")
_ACTIVE_DOMAIN: contextvars.ContextVar[str] = contextvars.ContextVar("enterprise_pipeline_domain", default=DEFAULT_DOMAIN)


class GatedVerifier:
    """``Verifier`` facade over ``AnswerVerificationModel`` that carries usage and cleans the candidate.

    Friction: (1) ``AnswerVerificationModel.verify()`` returns a ``Verdict`` without the
    run's ``Usage`` although ``pipeline_rag`` and ``task_decomposition`` both look for
    ``verdict.details["usage"]`` to roll it up, so verifier cost would vanish from the parent
    result; (2) the offline grounding heuristic treats ``[doc-id]`` citation markers and
    markdown headings as claim identifiers that no passage contains, so cited answers are
    reported as contradicted. The facade runs the full component (so its spans join the shared
    trace), strips markers and headings from the candidate, and attaches usage and status.
    """

    def __init__(self, model: AnswerVerificationModel) -> None:
        self.model = model

    @staticmethod
    def clean(answer: str) -> str:
        text = _HEADING_RE.sub("", answer)
        text = _CITATION_RE.sub("", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict:
        request = TaskRequest(task=task, inputs={"answer": self.clean(answer) or answer}, context=list(evidence), domain=_ACTIVE_DOMAIN.get())
        result = await self.model.run(request)
        output = result.output
        if hasattr(output, "to_verdict"):
            verdict: Verdict = output.to_verdict()
        else:
            verdict = Verdict(method="answer_verification", passed=False, score=0.0, issues=list(result.errors) or ["verification run failed"])
        verdict.details["usage"] = result.usage.model_dump()
        verdict.details["component_status"] = result.status.value
        verdict.details["request_id"] = result.request_id
        return verdict


class RAGSubTaskSolver:
    """``SubTaskSolver`` that answers leaves with RAG and lets the verifier own ``verify`` nodes.

    The decomposition model appends an independent ``verify`` node for high-risk profiles.
    Routed through RAG that node has nothing to retrieve ("insufficient evidence"); routed to
    the ``Verifier`` it checks the combined upstream results against evidence retrieved for
    the root task, which is what the node asks for. Every other node is one ``PipelineRAGModel``
    run whose ``usage`` the decomposition model charges to its own budget.
    """

    def __init__(self, rag: PipelineRAGModel, retriever: Retriever, verifier: Verifier, *, k: int = 8) -> None:
        self.rag = rag
        self.retriever = retriever
        self.verifier = verifier
        self.k = k

    async def __call__(self, request: TaskRequest) -> TaskResult:
        prior = [e for e in request.context if e.id.startswith("result:")]
        if request.metadata.get("tool_hint") == "verify" and prior:
            return await self._verify_node(request, prior)
        return await self.rag.run(request)

    async def _verify_node(self, request: TaskRequest, prior: list[Evidence]) -> TaskResult:
        root = str(request.metadata.get("root_task") or request.task)
        combined = "\n\n".join(e.content for e in prior)
        evidence = list(await self.retriever.retrieve(root, k=self.k))
        verdict = await self.verifier.verify(combined, task=root, evidence=evidence)
        raw_usage = verdict.details.get("usage") if isinstance(verdict.details, dict) else None
        usage = Usage.model_validate(raw_usage) if isinstance(raw_usage, dict) else Usage()
        lines = [f"Verification of the combined sub-task results: {'PASS' if verdict.passed else 'FAIL'} ({verdict.method}, score {verdict.score:.2f})."]
        lines.extend(f"- {issue}" for issue in verdict.issues[:6])
        cited = set(verdict.evidence_ids)
        return TaskResult(
            request_id=request.request_id,
            component="answer_verification",
            status=Status.SUCCESS if verdict.passed else Status.NEEDS_REVIEW,
            output=verdict,
            output_text="\n".join(lines),
            evidence=[e for e in evidence if e.id in cited] or evidence[:5],
            confidence=verdict.score,
            confidence_basis=str(verdict.details.get("confidence_basis") or f"{verdict.method} score"),
            usage=usage,
            metadata={"verify_node": True, "checked_results": [e.id for e in prior]},
        )


def _llm_leaf_usage(tracer: Tracer, root: Any, start_index: int) -> Usage:
    """Sum the usage of every LLM-call span (``gen_ai.operation.name == "chat"``) below ``root`` recorded since ``start_index``."""
    recent = tracer.spans[start_index:]
    children: dict[str, list[Any]] = {}
    for s in recent:
        if s.parent_id is not None:
            children.setdefault(s.parent_id, []).append(s)
    total = Usage()
    stack = [root.span_id]
    while stack:
        for child in children.get(stack.pop(), []):
            if child.attributes.get("gen_ai.operation.name") == "chat":
                total = total + child.usage
            stack.append(child.span_id)
    return total


def _reconcile_usage(reported: Usage, traced: Usage) -> Usage:
    """Field-wise maximum of a component's reported usage and the usage its spans recorded (latency stays reported)."""
    fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "llm_calls", "cache_hits", "cost_usd")
    merged = reported.model_copy()
    for f in fields:
        setattr(merged, f, max(getattr(reported, f), getattr(traced, f)))
    return merged


# ---------------------------------------------------------------------------
# Router, output model, pipeline
# ---------------------------------------------------------------------------


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route
    strategy: str
    complexity: str
    estimated_steps: int
    rationale: str


class ComplexityRouter:
    """Adaptive routing on the decomposition planner's complexity estimate (Adaptive-RAG / COMPOSITION_PATTERNS section 2)."""

    def __init__(self, decomposition: TaskDecompositionModel) -> None:
        self.decomposition = decomposition

    async def decide(self, request: TaskRequest) -> RouteDecision:
        forced = request.metadata.get("route")
        plan: DecompositionPlan = await self.decomposition.plan_for(request)
        gate = self.decomposition.config.decompose_min_steps
        complex_task = plan.strategy != "direct" and plan.estimated_steps >= gate
        if forced in ("rag", "decompose"):
            route: Route = forced  # type: ignore[assignment]
            why = f"route forced by request.metadata['route']={forced!r}"
        elif complex_task:
            route = "decompose"
            why = f"~{plan.estimated_steps} steps >= gate {gate} and strategy {plan.strategy}: decompose into a {plan.strategy} of sub-tasks"
        else:
            route = "rag"
            why = f"~{plan.estimated_steps} step(s), strategy {plan.strategy}: answer directly from the corpus"
        return RouteDecision(route=route, strategy=plan.strategy, complexity=plan.complexity_estimate, estimated_steps=plan.estimated_steps, rationale=f"{why}; planner: {plan.rationale}")


class EnterpriseOutput(BaseModel):
    """Typed ``TaskResult.output`` of the pipeline (JSON-safe so it survives the result cache)."""

    model_config = ConfigDict(extra="forbid")

    task: str
    domain: str
    route: RouteDecision
    component: str = Field(description="Component that produced the answer: pipeline_rag or task_decomposition")
    component_status: str
    component_confidence: float
    component_confidence_basis: str
    component_request_id: str
    verdict: Optional[dict[str, Any]] = None
    verdict_source: str = Field(default="none", description="component (reused from the solver) | pipeline (run by the final gate) | none")
    threshold: float
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sub_tasks: list[dict[str, Any]] = Field(default_factory=list)
    stages: list[StepRecord] = Field(default_factory=list)
    corpus_version: str
    cache_key: str
    cache_hit: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    component_usage: Usage = Field(default_factory=Usage)


class EnterprisePipeline:
    """All six models behind one ``run(request)``.

    Parameters
    ----------
    llm:
        The inner client; ``None`` -> ``make_llm()`` (a ``MockLLM`` without credentials).
    tracer:
        Shared by every component; created when omitted.
    domain:
        Default ``DomainProfile`` for requests and for the direct retriever/verifier APIs.
    result_cache_ttl_s:
        TTL of whole-result entries in the shared cache (``None`` disables result caching).
    *_config:
        Per-package configuration overrides (dicts of the package config fields).
    """

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        tracer: Tracer | None = None,
        domain: str = DEFAULT_DOMAIN,
        result_cache_ttl_s: Optional[float] = 3600.0,
        caching_config: dict[str, Any] | None = None,
        retrieval_config: dict[str, Any] | None = None,
        verification_config: dict[str, Any] | None = None,
        parallel_config: dict[str, Any] | None = None,
        rag_config: dict[str, Any] | None = None,
        decomposition_config: dict[str, Any] | None = None,
    ) -> None:
        self.tracer = tracer or Tracer(name=COMPONENT)
        self.domain = domain
        self.result_cache_ttl_s = result_cache_ttl_s
        self.inner_llm = llm if llm is not None else make_llm()
        self.offline = isinstance(self.inner_llm, MockLLM)

        # 1. cache first: one CachedLLMClient shared by every component
        self.caching = CachingModel(self.inner_llm, tracer=self.tracer, config=caching_config)
        self.llm: CachedLLMClient = make_shared_client(self.caching, self.inner_llm, profile=domain)
        self.cache: ResponseCache = self.caching.cache

        # 2. retrieval, verification, execution
        self.retrieval = HybridRetrievalModel(self.llm, tracer=self.tracer, domain=domain, config=retrieval_config)
        self.retriever: Retriever = TracedRetriever(self.retrieval, self.tracer)
        self.verification = AnswerVerificationModel(self.llm, tracer=self.tracer, config=verification_config)
        self.verifier: Verifier = GatedVerifier(self.verification)
        self.parallel = ParallelizationModel(self.llm, tracer=self.tracer, **(parallel_config or {}))
        self.executor: Executor = ReentrantExecutor(self.parallel.executor, self.tracer)

        # 3. RAG with injected protocol objects; decomposition with RAG as the solver
        self.rag = PipelineRAGModel(
            self.llm, retriever=self.retriever, verifier=self.verifier, executor=self.executor, cache=self.cache, tracer=self.tracer, **(rag_config or {})
        )
        self.solver = RAGSubTaskSolver(self.rag, self.retriever, self.verifier)
        self.decomposition = TaskDecompositionModel(
            self.llm, solver=self.solver, executor=self.executor, verifier=self.verifier, tracer=self.tracer, config=decomposition_config
        )
        self.router = ComplexityRouter(self.decomposition)

        self.components: dict[str, Any] = {
            "caching": self.caching,
            "hybrid_retrieval": self.retrieval,
            "answer_verification": self.verification,
            "parallelization": self.parallel,
            "pipeline_rag": self.rag,
            "task_decomposition": self.decomposition,
        }
        self._doc_hashes: dict[str, str] = {}
        self.corpus_version: str = "empty"
        self.result_cache_hits = 0

    # ------------------------------------------------------------ indexing
    def index(self, documents: Sequence[Any]) -> int:
        """Add or replace documents in the hybrid index; invalidate cached answers grounded on changed documents."""
        docs = list(documents)
        changed: list[str] = []
        for i, d in enumerate(docs, 1):
            if isinstance(d, dict):
                doc_id, text = str(d.get("id") or f"doc-{i}"), str(d.get("text") or d.get("content") or "")
            elif isinstance(d, Evidence):
                doc_id, text = d.id, d.content
            else:
                doc_id, text = f"doc-{i}", str(d)
            h = stable_hash(text, length=16)
            if self._doc_hashes.get(doc_id) != h:
                changed.append(doc_id)
            self._doc_hashes[doc_id] = h
        with self.tracer.span("hybrid_retrieval.index", "hybrid_retrieval", documents=len(docs)) as sp:
            chunks = self.retrieval.add_documents(docs)
            sp.attributes["chunks"] = chunks
        if changed and self.corpus_version != "empty":
            event = self.caching.invalidate_documents(changed, cause="document_updated")
            self.tracer.event("cache_invalidated", documents=len(changed), removed_exact=event.removed_exact, removed_semantic=event.removed_semantic)
        self.corpus_version = stable_hash(sorted(self._doc_hashes.items()), length=12)
        return chunks

    # ----------------------------------------------------------------- run
    async def run(self, request: TaskRequest | str, **request_kwargs: Any) -> TaskResult:
        if isinstance(request, str):
            request_kwargs.setdefault("domain", self.domain)
            request = TaskRequest(task=request, **request_kwargs)
        stages: list[StepRecord] = []
        warnings: list[str] = []
        domain_token = _ACTIVE_DOMAIN.set(request.domain)
        t_start = time.perf_counter()
        try:
            with self.tracer.span(f"{COMPONENT}.run", COMPONENT, request_id=request.request_id, domain=request.domain) as root_span:
                # stage: index -------------------------------------------------------------
                documents = request.inputs.get("documents")
                child_inputs = {k: v for k, v in request.inputs.items() if k != "documents"}
                if documents:
                    with self._stage(stages, "index") as rec:
                        n = self.index(documents if isinstance(documents, (list, tuple)) else [documents])
                        rec.summary = f"indexed {n} chunk(s); corpus version {self.corpus_version}"
                child = request.derive(request.task, inputs=child_inputs, metadata={**request.metadata, "corpus_version": self.corpus_version})
                child = child.model_copy(update={"request_id": request.request_id})

                # stage: cache lookup ------------------------------------------------------
                key = self.result_cache_key(request)
                if self.result_cache_ttl_s is not None and not request.metadata.get("no_cache"):
                    with self._stage(stages, "cache_lookup", key=key[:12]) as rec:
                        with self.tracer.span("caching.lookup", "caching", key=key[:12]) as sp:
                            hit = await self.cache.get(key)
                            sp.attributes["hit"] = hit is not None
                            if hit is not None:
                                rec.usage = Usage(cache_hits=1)
                                self.tracer.record_usage(rec.usage, span=sp)
                        rec.summary = "hit" if hit is not None else "miss"
                    if isinstance(hit, dict):
                        self.result_cache_hits += 1
                        result = self._cached_result(hit, request, stages, key)
                        root_span.attributes["status"] = result.status.value
                        return result

                # stage: route -------------------------------------------------------------
                with self._stage(stages, "route") as rec:
                    decision = await self.router.decide(child)
                    rec.summary = f"{decision.route} ({decision.complexity}, ~{decision.estimated_steps} steps, strategy {decision.strategy})"
                    rec.details.update(decision.model_dump())

                # stage: solve -------------------------------------------------------------
                with self._stage(stages, "solve", route=decision.route) as rec:
                    component = self.rag if decision.route == "rag" else self.decomposition
                    inner = await component.run(child)
                    rec.usage = inner.usage
                    rec.status = inner.status if inner.status != Status.CACHED else Status.SUCCESS
                    rec.summary = f"{inner.component}: {inner.status.value}, confidence {inner.confidence:.2f}, {inner.usage.llm_calls} LLM call(s)"
                warnings.extend(inner.warnings)

                # stage: final verification gate --------------------------------------------
                profile = get_profile(request.domain)
                threshold = max(profile.min_confidence_to_accept, request.constraints.min_confidence)
                verdict, verdict_source, verify_usage = await self._final_verdict(stages, child, inner)
                status, confidence, basis, reasons = self._gate(inner, verdict, threshold)

                # stage: store -------------------------------------------------------------
                output = self._output(request, decision, inner, verdict, verdict_source, threshold, stages, key, reasons)
                total_usage = Usage()
                for s in stages:
                    total_usage = total_usage + s.usage
                total_usage.latency_s = time.perf_counter() - t_start
                result = TaskResult(
                    request_id=request.request_id,
                    component=COMPONENT,
                    status=status,
                    output=output,
                    output_text=inner.output_text or inner.confidence_basis,
                    evidence=list(inner.evidence),
                    confidence=confidence,
                    confidence_basis=basis,
                    steps=stages + list(inner.steps),
                    usage=total_usage,
                    warnings=list(dict.fromkeys(warnings + reasons)),
                    errors=list(inner.errors),
                    metadata={
                        "route": decision.route,
                        "plan": decision.model_dump(),
                        "component_plan": inner.metadata.get("plan"),
                        "threshold": threshold,
                        "cache_key": key,
                        "corpus_version": self.corpus_version,
                        "verify_usage": verify_usage.model_dump(),
                    },
                    trace_id=self.tracer.trace_id,
                )
                if self.result_cache_ttl_s is not None and status != Status.FAILED:
                    with self._stage(stages, "store") as rec:
                        tags = ["enterprise_pipeline", f"domain:{request.domain}", f"corpus:{self.corpus_version}"] + [f"doc:{d}" for d in self._cited_doc_ids(inner.evidence)]
                        with self.tracer.span("caching.store", "caching", tags=len(tags)):
                            await self.cache.set(key, result.model_dump(mode="json"), ttl_s=self.result_cache_ttl_s, tags=tags)
                        rec.summary = f"stored under {key[:12]} with {len(tags)} tag(s)"
                    output.stages = list(stages)
                    result.steps = stages + list(inner.steps)
                root_span.attributes["status"] = status.value
                root_span.attributes["confidence"] = confidence
                return result
        finally:
            _ACTIVE_DOMAIN.reset(domain_token)

    def run_sync(self, request: TaskRequest | str, **request_kwargs: Any) -> TaskResult:
        return run_sync(self.run(request, **request_kwargs))

    # -------------------------------------------------------------- helpers
    def result_cache_key(self, request: TaskRequest) -> str:
        inputs = {k: v for k, v in request.inputs.items() if k != "documents"}
        return stable_hash(COMPONENT, normalize_text(request.task), request.domain, self.corpus_version, request.constraints.model_dump(), inputs, self.inner_llm.model)

    class _StageCtx:
        def __init__(self, pipeline: "EnterprisePipeline", stages: list[StepRecord], name: str, details: dict[str, Any]) -> None:
            self.rec = StepRecord(name=name, component=COMPONENT, details=details)
            self.pipeline = pipeline
            self.stages = stages
            self.t0 = 0.0

        def __enter__(self) -> StepRecord:
            self.t0 = time.perf_counter()
            self.start_index = len(self.pipeline.tracer.spans)
            self.span_cm = self.pipeline.tracer.span(self.rec.name, COMPONENT)
            self.span = self.span_cm.__enter__()
            return self.rec

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            if exc is not None:
                self.rec.status = Status.FAILED
                self.rec.summary = self.rec.summary or f"{type(exc).__name__}: {exc}"
            self.rec.duration_s = time.perf_counter() - self.t0
            self.span_cm.__exit__(exc_type, exc, tb)
            # Friction: PipelineComponent.run merges result.usage with ctx.usage through _max_usage(), which
            # compares only llm_calls and cost_usd. A run whose every LLM call was a cache hit therefore keeps the
            # empty result.usage, loses its cache_hits, and records the negative difference onto its own span.
            # The individual LLM-call spans under this stage are still correct, so reconcile against their sum.
            leaves = _llm_leaf_usage(self.pipeline.tracer, self.span, self.start_index)
            self.rec.usage = _reconcile_usage(self.rec.usage, leaves)
            self.stages.append(self.rec)

    def _stage(self, stages: list[StepRecord], name: str, **details: Any) -> "EnterprisePipeline._StageCtx":
        return EnterprisePipeline._StageCtx(self, stages, name, dict(details))

    async def _final_verdict(self, stages: list[StepRecord], request: TaskRequest, inner: TaskResult) -> tuple[Optional[Verdict], str, Usage]:
        """Reuse the solver's verdict on the synthesised answer when it has one; otherwise run the verifier now."""
        with self._stage(stages, "verify_final") as rec:
            existing = getattr(inner.output, "verdict", None)
            if isinstance(existing, Verdict):
                rec.summary = f"reused {existing.method} verdict from {inner.component}: {'passed' if existing.passed else 'failed'} score={existing.score:.2f}"
                return existing, "component", Usage()
            if isinstance(existing, dict) and existing:
                try:
                    v = Verdict.model_validate(existing)
                    rec.summary = f"reused {v.method} verdict from {inner.component}: {'passed' if v.passed else 'failed'} score={v.score:.2f}"
                    return v, "component", Usage()
                except Exception:  # noqa: BLE001 - fall through to a fresh verification
                    pass
            if inner.status == Status.FAILED or not inner.output_text.strip():
                rec.status = Status.SKIPPED
                rec.summary = "skipped: nothing to verify"
                return None, "none", Usage()
            try:
                v = await self.verifier.verify(inner.output_text, task=request.task, evidence=inner.evidence)
            except Exception as exc:  # noqa: BLE001 - a broken verifier is a review condition, not a crash
                rec.status = Status.FAILED
                rec.summary = f"verifier raised {type(exc).__name__}"
                return None, "none", Usage()
            raw = v.details.get("usage") if isinstance(v.details, dict) else None
            usage = Usage.model_validate(raw) if isinstance(raw, dict) else Usage()
            rec.usage = usage
            rec.summary = f"{v.method}: {'passed' if v.passed else 'failed'} score={v.score:.2f}"
            return v, "pipeline", usage

    @staticmethod
    def _gate(inner: TaskResult, verdict: Optional[Verdict], threshold: float) -> tuple[Status, float, str, list[str]]:
        reasons: list[str] = []
        confidence = inner.confidence
        parts = [f"{inner.component} confidence {inner.confidence:.2f}"]
        if verdict is not None:
            confidence = min(confidence, verdict.score)
            parts.append(f"{verdict.method} {'passed' if verdict.passed else 'failed'} (score {verdict.score:.2f})")
            if not verdict.passed:
                reasons.append("final verification failed: " + "; ".join(verdict.issues[:3]))
        else:
            parts.append("no verifier verdict")
            if inner.status != Status.FAILED:
                reasons.append("answer not externally verified")
        parts.append(f"accept threshold {threshold:.2f}")
        if inner.status == Status.FAILED:
            status = Status.FAILED
            reasons.append(f"{inner.component} failed: " + "; ".join(inner.errors[:2]))
        elif reasons or confidence < threshold or inner.status == Status.NEEDS_REVIEW:
            status = Status.NEEDS_REVIEW
            if confidence < threshold:
                reasons.append(f"confidence {confidence:.2f} below accept threshold {threshold:.2f}")
            if inner.status == Status.NEEDS_REVIEW:
                reasons.append(f"{inner.component} asked for review")
        elif inner.status == Status.PARTIAL:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS
        return status, round(max(0.0, min(1.0, confidence)), 4), "; ".join(parts), reasons

    @staticmethod
    def _cited_doc_ids(evidence: Sequence[Evidence]) -> list[str]:
        out: list[str] = []
        for e in evidence:
            doc_id = str(e.metadata.get("doc_id") or e.id.split("#")[0])
            if doc_id not in out and not doc_id.startswith("result:"):
                out.append(doc_id)
        return out

    def _output(
        self,
        request: TaskRequest,
        decision: RouteDecision,
        inner: TaskResult,
        verdict: Optional[Verdict],
        verdict_source: str,
        threshold: float,
        stages: list[StepRecord],
        key: str,
        reasons: list[str],
    ) -> EnterpriseOutput:
        citations = [
            {"id": e.id, "source": e.source or str(e.metadata.get("title") or ""), "score": round(e.score, 4), "retriever": e.retriever, "snippet": e.short(120)}
            for e in inner.evidence
            if not e.id.startswith("result:")
        ]
        sub_tasks: list[dict[str, Any]] = []
        results = getattr(inner.output, "results", None)
        graph = getattr(inner.output, "graph", None)
        if isinstance(results, dict):
            for tid, r in results.items():
                desc = ""
                if graph is not None:
                    try:
                        desc = graph.get(tid).description
                    except Exception:  # noqa: BLE001
                        desc = ""
                sub_tasks.append({"id": tid, "description": desc, "status": r.status.value, "confidence": round(r.confidence, 3), "attempts": r.attempts, "component": r.metadata.get("component", ""), "error": r.error})
        return EnterpriseOutput(
            task=request.task,
            domain=request.domain,
            route=decision,
            component=inner.component,
            component_status=inner.status.value,
            component_confidence=inner.confidence,
            component_confidence_basis=inner.confidence_basis,
            component_request_id=inner.request_id,
            verdict=verdict.model_dump(mode="json") if verdict is not None else None,
            verdict_source=verdict_source,
            threshold=threshold,
            citations=citations,
            sub_tasks=sub_tasks,
            stages=list(stages),
            corpus_version=self.corpus_version,
            cache_key=key,
            review_reasons=reasons,
            component_usage=inner.usage,
        )

    def _cached_result(self, hit: dict[str, Any], request: TaskRequest, stages: list[StepRecord], key: str) -> TaskResult:
        cached = TaskResult.model_validate(hit)
        output = EnterpriseOutput.model_validate(cached.output) if isinstance(cached.output, dict) else cached.output
        if isinstance(output, EnterpriseOutput):
            output.cache_hit = True
            output.stages = list(stages)
        usage = Usage(cache_hits=1)
        return TaskResult(
            request_id=request.request_id,
            component=COMPONENT,
            status=Status.CACHED,
            output=output,
            output_text=cached.output_text,
            evidence=list(cached.evidence),
            confidence=cached.confidence,
            confidence_basis=f"served from the shared cache (original: {cached.confidence_basis})",
            steps=list(stages),
            usage=usage,
            warnings=list(cached.warnings),
            metadata={**cached.metadata, "cache_key": key, "cached_from_request": cached.request_id, "original_status": cached.status.value},
            trace_id=self.tracer.trace_id,
        )

    # ------------------------------------------------------------ reporting
    def cache_stats(self) -> dict[str, Any]:
        m = self.llm.metrics
        backend = self.caching.backend.snapshot_stats()
        return {
            "llm_requests": m.requests,
            "llm_exact_hits": m.exact_hits,
            "llm_semantic_hits": m.semantic_hits,
            "llm_coalesced": m.coalesced,
            "llm_misses": m.misses,
            "llm_cost_saved_usd": round(m.cost_saved_usd, 6),
            "result_cache_hits": self.result_cache_hits,
            "backend_entries": backend.entries,
            "backend_hits": backend.hits,
            "backend_misses": backend.misses,
            "backend_invalidations": backend.invalidations,
        }

    def report(self, result: TaskResult, *, tree: bool = True) -> str:
        out = result.output
        lines = ["=" * 100, f"TASK   : {result.output.task if isinstance(out, EnterpriseOutput) else ''}", f"STATUS : {result.status.value.upper()}   confidence={result.confidence:.2f}   {result.confidence_basis}"]
        if isinstance(out, EnterpriseOutput):
            lines.append(f"ROUTE  : {out.route.route} ({out.route.complexity}, ~{out.route.estimated_steps} steps, strategy {out.route.strategy}) -> {out.component}")
            lines.append(f"         {out.route.rationale}")
            lines.append("STAGES :")
            for s in out.stages:
                lines.append(f"  - {s.name:<13} {s.status.value:<12} {s.duration_s * 1000:7.1f} ms  calls={s.usage.llm_calls} hits={s.usage.cache_hits}  {s.summary}")
            if out.sub_tasks:
                lines.append("SUB-TASKS:")
                for st in out.sub_tasks:
                    lines.append(f"  - [{st['id']}] {st['status']:<12} conf={st['confidence']:.2f} via {st['component'] or '-'}  {st['description'][:80]}" + (f"  error={st['error']}" if st["error"] else ""))
            if out.verdict:
                v = out.verdict
                lines.append(f"VERDICT: {v.get('method')} {'PASS' if v.get('passed') else 'FAIL'} score={float(v.get('score', 0.0)):.2f} (source: {out.verdict_source}; threshold {out.threshold:.2f})")
                for issue in list(v.get("issues") or [])[:4]:
                    lines.append(f"  - {issue}")
            if out.review_reasons:
                lines.append("REVIEW : " + " | ".join(out.review_reasons))
            lines.append(f"CITATIONS ({len(out.citations)}):")
            for c in out.citations[:8]:
                lines.append(f"  - [{c['id']}] {c['source']} score={c['score']:.3f} via {c['retriever']}: {c['snippet']}")
        lines.append("ANSWER :")
        lines.extend("  " + ln for ln in (result.output_text or "(empty)").splitlines()[:40])
        u = result.usage
        lines.append(f"USAGE  : llm_calls={u.llm_calls} tokens(in/out/cache_read)={u.input_tokens}/{u.output_tokens}/{u.cache_read_input_tokens} cost=${u.cost_usd:.4f} cache_hits={u.cache_hits} latency={u.latency_s:.2f}s")
        cs = self.cache_stats()
        lines.append(f"CACHE  : llm requests={cs['llm_requests']} exact_hits={cs['llm_exact_hits']} semantic_hits={cs['llm_semantic_hits']} result_hits={cs['result_cache_hits']} backend hits/misses/entries={cs['backend_hits']}/{cs['backend_misses']}/{cs['backend_entries']}")
        ex = self.parallel.executor.stats
        lines.append(f"EXEC   : submitted={ex.submitted} succeeded={ex.succeeded} failed={ex.failed} retries={ex.retries} breaker={self.parallel.executor.snapshot().get('breaker', {}).get('state', '-')}")
        if tree:
            lines.append("TRACE  :")
            lines.extend("  " + ln for ln in self.tracer.tree().splitlines())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def build_pipeline(*, offline: bool = False, domain: str = DEFAULT_DOMAIN, tracer: Tracer | None = None, **kwargs: Any) -> EnterprisePipeline:
    """Construct the pipeline with ``make_llm()`` (``offline=True`` forces the deterministic MockLLM)."""
    llm = make_llm(offline=True) if offline else make_llm()
    return EnterprisePipeline(llm, tracer=tracer, domain=domain, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the six-model enterprise pipeline over the built-in pump station corpus.")
    ap.add_argument("--offline", action="store_true", help="Force the deterministic MockLLM (heuristic stages, no network)")
    ap.add_argument("--domain", default=DEFAULT_DOMAIN, help="DomainProfile name (default: mechanical_engineering)")
    ap.add_argument("--task", action="append", default=[], help="Run this task instead of the two demo tasks (repeatable)")
    ap.add_argument("--json", action="store_true", help="Print each result as JSON instead of the report")
    ap.add_argument("--no-trace", action="store_true", help="Omit the span tree from the reports")
    args = ap.parse_args(argv)

    pipeline = build_pipeline(offline=args.offline, domain=args.domain)
    print(f"client: {pipeline.llm.describe()['client']} over {pipeline.inner_llm.describe()['client']} ({pipeline.inner_llm.model}); offline={pipeline.offline}; trace={pipeline.tracer.trace_id}")
    tasks = [{"label": f"task {i + 1}", "task": t} for i, t in enumerate(args.task)] or DEMO_TASKS
    results: list[TaskResult] = []
    for i, item in enumerate(tasks):
        request = TaskRequest(task=item["task"], inputs={"documents": CORPUS} if i == 0 else {}, domain=args.domain)
        result = pipeline.run_sync(request)
        results.append(result)
        print(f"\n### {item['label']}")
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=1, default=str)[:20000])
        else:
            print(pipeline.report(result, tree=False))

    # repeat the first task: served from the shared cache without recomputation
    repeat = pipeline.run_sync(TaskRequest(task=tasks[0]["task"], domain=args.domain))
    results.append(repeat)
    print(f"\n### repeat of {tasks[0]['label']} (cache demonstration)")
    print(f"status={repeat.status.value} cache_hits={repeat.usage.cache_hits} stages={[s.name + ':' + s.summary for s in repeat.steps]}")
    print(f"cache: {json.dumps(pipeline.cache_stats())}")
    savings = pipeline.caching.savings_report()
    print(f"caching model savings report: hits={savings['hits']} hit_rate={savings['hit_rate']} cost_saved_usd={savings['cost_saved_usd']:.6f}")
    if not args.no_trace:
        print("\n### span tree (one Tracer shared by every component)")
        print(pipeline.tracer.tree())
    summary = pipeline.tracer.summary()
    print(f"\ntrace summary: spans={summary['spans']} errors={summary['errors']} usage={summary['usage']}")
    print("components in trace:", ", ".join(sorted({s.component for s in pipeline.tracer.spans if s.component})))
    return 0 if all(r.status != Status.FAILED for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
