"""Sub-task solvers: how one node of the graph gets answered.

Two implementations share one call shape so the scheduler does not care:

* :class:`LLMSubTaskSolver` - the default. One ``SOLVE_TEMPLATE`` (or
  ``PLAN_AND_SOLVE_TEMPLATE``) call through ``ctx.llm_call`` with only what the
  node needs (ReWOO compact context): its description, inputs, the outputs of
  its dependencies and the evidence. An injected :class:`Retriever` adds
  Self-Ask-style per-sub-question retrieval; an injected :class:`Verifier`
  supplies the external signal that may fail a node under ``strict``
  verification (Huang et al.: never fail a node on self-judgement alone).
* :class:`ProtocolSolver` - adapts any ``SubTaskSolver`` callable (typically
  another component's ``run``) and charges its usage to the run context.

When a :class:`Verifier` is injected, :class:`VerifierNodeSolver` wraps either
solver and takes over *verification nodes* (``tool_hint`` ``verify`` with at
least one dependency, e.g. the terminal ``verify`` node high-risk profiles
append): the verifier checks the combined upstream outputs against the upstream
evidence, and the node's status/confidence come from the :class:`Verdict`. A
solver such as RAG has nothing to retrieve for "verify the results above", and
a verdict is not content of the answer, so synthesis records it instead of
composing it (:func:`is_verification_result`).

Offline (MockLLM) answers carry no epistemic confidence, so a structural
estimate is used and later capped by ``offline_confidence_cap``.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from pipeline_core import (
    BudgetExceededError,
    Evidence,
    Retriever,
    RunContext,
    Status,
    SubTaskSolver,
    TaskRequest,
    TaskResult,
    Usage,
    Verdict,
    Verifier,
    dedupe_preserve_order,
    validate_structured,
)

from .config import TaskDecompositionConfig
from .graph import SubTask, TaskGraph
from .plan import DecompositionPlan
from .prompting import build_request, render_documents, render_inputs
from .prompts import PLAN_AND_SOLVE_TEMPLATE, SOLVE_TEMPLATE
from .scheduler import SubTaskResult
from .schemas import SubTaskAnswer, schema_for
from .strategies import stop_reason

_OFFLINE_BASE = {"low": 0.8, "medium": 0.75, "high": 0.65}
VERIFICATION_HINTS = frozenset({"verify", "verification"})
"""``SubTask.tool_hint`` values that mark a node as verification of its dependencies' outputs."""


def check_budget_before(task_id: str, ctx: RunContext) -> None:
    """Raise :class:`BudgetExceededError` when no call may start for ``task_id`` (deadline, call cap or cost cap).

    ``ctx.llm_call`` only checks the deadline and the call count *before* a call and discovers a cost
    overrun *after* paying for it; this gate is applied before every node (LLM or nested component) so a
    cost-capped run stops spending once the cap is reached (architecture rule 3.2).
    """
    c = ctx.request.constraints
    reason = stop_reason(ctx)
    if reason is None:
        return
    if reason == "deadline":
        raise BudgetExceededError(f"latency budget {c.max_latency_s}s exhausted before sub-task {task_id}")
    rem = ctx.budget.remaining_calls()
    if rem is not None and rem <= 0:
        raise BudgetExceededError(f"LLM call budget of {c.max_llm_calls} exhausted before sub-task {task_id}")
    raise BudgetExceededError(f"cost budget of ${c.max_cost_usd:.4f} exhausted before sub-task {task_id}")


def charge_nested_usage(ctx: RunContext, usage: Usage, *, what: str) -> None:
    """Charge work a nested component or verifier already performed.

    The work is done and its result is valid, so an overshoot of ``max_llm_calls``/``max_cost_usd`` is
    recorded (``ctx.usage`` is updated before the tracker raises) and reported as a warning instead of
    discarding the result; the scheduler's stop check halts further dispatch.
    """
    try:
        ctx.charge(usage)
    except BudgetExceededError as exc:
        ctx.warn(f"budget exceeded after {what}: {exc}")


def offline_confidence(task: SubTask, *, gaps: int = 0, partial: bool = False) -> float:
    """Structural confidence for MockLLM answers: by node complexity, minus gaps; partial answers sit at 0.4."""
    if partial:
        return 0.4
    return max(0.0, _OFFLINE_BASE.get(task.complexity, 0.7) - 0.1 * min(gaps, 3))


def evidence_from_inputs(inputs: dict[str, Any]) -> list[Evidence]:
    """Turn ``inputs["documents"]`` (strings or dicts with ``content``/``text``) into :class:`Evidence`."""
    docs = inputs.get("documents") or inputs.get("document") or []
    if isinstance(docs, (str, dict)):
        docs = [docs]
    out: list[Evidence] = []
    for i, d in enumerate(docs, 1):
        if isinstance(d, Evidence):
            out.append(d)
        elif isinstance(d, str):
            out.append(Evidence(id=f"doc-{i}", content=d, source="inputs.documents", retriever="provided"))
        elif isinstance(d, dict):
            content = str(d.get("content") or d.get("text") or "")
            if content:
                out.append(Evidence(id=str(d.get("id") or f"doc-{i}"), content=content, source=str(d.get("source") or "inputs.documents"), retriever="provided", metadata={k: v for k, v in d.items() if k not in ("id", "content", "text", "source")}))
    return out


def render_prior(prior: dict[str, SubTaskResult], graph: TaskGraph, *, max_chars: int) -> str:
    """Render dependency outputs as ``<result>`` blocks (compact per-node context)."""
    if not prior:
        return "(none)"
    parts = []
    for tid, res in prior.items():
        try:
            desc = graph.get(tid).description
        except KeyError:
            desc = tid
        body = res.output_text if len(res.output_text) <= max_chars else res.output_text[: max_chars - 3] + "..."
        parts.append(f'<result id="{tid}" status="{res.status.value}" confidence="{res.confidence:.2f}">\n{desc}\n---\n{body}\n</result>')
    return "\n".join(parts)


def map_status(status: Status) -> Status:
    """Collapse a nested component's status onto the sub-task result vocabulary."""
    if status in (Status.SUCCESS, Status.CACHED):
        return Status.SUCCESS
    if status in (Status.PARTIAL, Status.NEEDS_REVIEW):
        return Status.PARTIAL
    return Status.FAILED


class NodeSolver(Protocol):
    """Call shape shared by the solvers; the model binds the extra keyword arguments per run."""

    async def solve(self, task: SubTask, prior: dict[str, SubTaskResult], *, graph: TaskGraph, request: TaskRequest, ctx: RunContext, plan: DecompositionPlan, depth: int = 0) -> SubTaskResult: ...


class LLMSubTaskSolver:
    """Default leaf solver: one structured LLM call per node, optional retrieval and verification."""

    def __init__(self, config: Optional[TaskDecompositionConfig] = None, *, retriever: Optional[Retriever] = None, verifier: Optional[Verifier] = None) -> None:
        self.config = config or TaskDecompositionConfig()
        self.retriever = retriever
        self.verifier = verifier

    async def gather_evidence(self, task: SubTask, request: TaskRequest, ctx: RunContext) -> list[Evidence]:
        """Request context + ``inputs.documents`` + per-sub-question retrieval when a retriever is injected."""
        evidence: list[Evidence] = list(request.context) + evidence_from_inputs(request.inputs)
        if self.retriever is not None and request.constraints.allow_external_tools and task.tool_hint in ("retrieve", "llm", "aggregate"):
            try:
                hits = await self.retriever.retrieve(task.description, k=self.config.retrieve_k)
                evidence = list(hits) + evidence
            except Exception as exc:  # noqa: BLE001 - retrieval failure degrades to provided context
                ctx.warn(f"retrieval failed for {task.id} ({type(exc).__name__}); continuing with provided context")
        return dedupe_preserve_order(evidence, key=lambda e: e.id)[: self.config.max_context_docs]

    async def solve(self, task: SubTask, prior: dict[str, SubTaskResult], *, graph: TaskGraph, request: TaskRequest, ctx: RunContext, plan: DecompositionPlan, depth: int = 0) -> SubTaskResult:
        cfg = self.config
        check_budget_before(task.id, ctx)  # cost cap included; ctx.llm_call alone would pay for one more call first
        evidence = await self.gather_evidence(task, request, ctx)
        whole_task = len(graph.tasks) == 1 and depth == 0 and task.tool_hint in ("llm", "plan_and_solve") and graph.strategy in ("direct", "plan_and_solve", "recursive")
        if task.tool_hint == "plan_and_solve" or (whole_task and graph.strategy == "plan_and_solve"):
            req = build_request(
                PLAN_AND_SOLVE_TEMPLATE,
                profile=ctx.profile,
                values={"task": task.description, "inputs": render_inputs({**request.inputs, **task.inputs}), "documents": render_documents(evidence, max_items=cfg.max_context_docs)},
                schema=schema_for(SubTaskAnswer),
                effort=cfg.planner_effort,
                tags=["solve", task.id],
            )
        else:
            effort = cfg.planner_effort if whole_task else cfg.solver_effort
            if task.tool_hint == "verify" and ctx.profile.risk_level == "safety_critical" and not graph.dependents(task.id):
                # rule 3.2: max effort only for *final* verification (a terminal verify node) in safety-critical domains
                effort = cfg.safety_critical_verification_effort
            req = build_request(
                SOLVE_TEMPLATE,
                profile=ctx.profile,
                values={
                    "task_id": task.id,
                    "description": task.description,
                    "expected_output": task.expected_output or "(unspecified)",
                    "criteria": "\n".join(f"{i}. {c}" for i, c in enumerate(task.verification_criteria, 1)) or "(none)",
                    "inputs": render_inputs(task.inputs),
                    "prior_results": render_prior(prior, graph, max_chars=cfg.max_subtask_output_chars),
                    "documents": render_documents(evidence, max_items=cfg.max_context_docs),
                    "root_task": graph.root_task,
                },
                schema=schema_for(SubTaskAnswer),
                effort=effort,
                model=None if whole_task else cfg.solver_model,
                tags=["solve", task.id, task.tool_hint],
            )
        resp = await ctx.llm_call(req, step=f"solve.{task.id}")
        answer = validate_structured(resp.parsed, SubTaskAnswer)
        status = {"completed": Status.SUCCESS, "partial": Status.PARTIAL, "failed": Status.FAILED}[answer.status]
        known = {e.id for e in evidence}
        cited = [i for i in answer.evidence_ids if i in known] or ([e.id for e in evidence] if evidence else [])
        confidence = answer.confidence
        if ctx.offline:
            confidence = offline_confidence(task, gaps=len(answer.gaps), partial=status == Status.PARTIAL)
        result = SubTaskResult(
            task_id=task.id,
            status=status,
            output_text=answer.answer.strip(),
            confidence=confidence,
            criteria_met=list(answer.criteria_met),
            evidence_ids=cited,
            evidence=[e for e in evidence if e.id in set(cited)],
            gaps=list(answer.gaps),
            error=None if status != Status.FAILED else ("; ".join(answer.gaps) or "solver reported failure"),
            usage=resp.usage,
            depth=depth,
            metadata={"tool_hint": task.tool_hint, "model": resp.model},
        )
        if self.verifier is not None and plan.verification == "strict" and result.ok and result.output_text:
            result = await self.verify_node(result, task, evidence, ctx)
        return result

    async def verify_node(self, result: SubTaskResult, task: SubTask, evidence: Sequence[Evidence], ctx: RunContext) -> SubTaskResult:
        """External per-node verification (strict mode): a failed verdict fails the node so the scheduler retries / the replanner acts."""
        try:
            verdict: Verdict = await self.verifier.verify(result.output_text, task=task.description, evidence=evidence)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - a broken verifier must not fail the run
            ctx.warn(f"verifier raised for {task.id} ({type(exc).__name__}); node left unverified")
            result.gaps.append("verifier unavailable")
            return result
        charge_verdict_usage(verdict, ctx, what=f"verifier verdict for {task.id}")  # never fails a verified node
        result.metadata["verdict"] = verdict.model_dump(mode="json")
        result.confidence = min(result.confidence, verdict.score)
        if not verdict.passed:
            result.status = Status.FAILED
            result.error = "verifier: " + ("; ".join(verdict.issues) or "verification failed")
        return result


def charge_verdict_usage(verdict: Verdict, ctx: RunContext, *, what: str = "verifier verdict") -> None:
    """Roll up usage a verifier reports in ``verdict.details["usage"]`` (verifiers are protocol objects, not nested runs).

    A budget overshoot is warned about through :func:`charge_nested_usage`, never raised: the verdict
    already exists and must not be turned into a failure. Malformed usage is ignored.
    """
    raw = verdict.details.get("usage") if isinstance(verdict.details, dict) else None
    if isinstance(raw, dict):
        try:
            usage = Usage.model_validate(raw)
        except Exception:  # noqa: BLE001 - malformed usage is ignored
            return
        charge_nested_usage(ctx, usage, what=what)


def is_verification_node(task: SubTask) -> bool:
    """True for a node that verifies upstream results: a verification ``tool_hint`` *and* at least one dependency.

    A dependency-free node whose text starts with "verify"/"check" ("verify the motor rating against the
    measured shaft power") is content work that needs retrieval, so it stays with the solver.
    """
    return task.tool_hint.strip().lower() in VERIFICATION_HINTS and bool(task.depends_on)


def is_verification_result(result: SubTaskResult) -> bool:
    """True when ``result`` was produced by :class:`VerifierNodeSolver` (a verdict, not answer content)."""
    return result.metadata.get("verification_node") is True


def node_verdict(result: SubTaskResult) -> Optional[Verdict]:
    """The :class:`Verdict` recorded on a verification-node result, if the verifier ran."""
    raw = result.metadata.get("verdict") if is_verification_result(result) else None
    if isinstance(raw, Verdict):
        return raw
    if isinstance(raw, dict):
        try:
            return Verdict.model_validate(raw)
        except Exception:  # noqa: BLE001 - malformed metadata is treated as "no verdict"
            return None
    return None


class VerifierNodeSolver:
    """Route verification nodes to the injected :class:`Verifier`; delegate every other node to ``inner``.

    The verifier receives the combined outputs of the node's completed dependencies as the answer, the
    root task as the task, and the evidence those dependencies gathered (plus the request's own context
    and ``inputs.documents``) as evidence. Sub-task outputs themselves (``result:<id>``) are never passed
    as evidence, so the results are not checked against themselves. The verifier's usage
    (``verdict.details["usage"]``) is charged to the run. A passing verdict gives ``SUCCESS``; a failing
    one gives ``NEEDS_REVIEW`` with a ``verifier:`` error rather than ``FAILED``: retrying or replanning
    the check cannot repair upstream content, so the run is flagged for review instead.
    """

    def __init__(self, inner: NodeSolver, verifier: Verifier, config: Optional[TaskDecompositionConfig] = None) -> None:
        self.inner = inner
        self.verifier = verifier
        self.config = config or TaskDecompositionConfig()

    async def solve(self, task: SubTask, prior: dict[str, SubTaskResult], *, graph: TaskGraph, request: TaskRequest, ctx: RunContext, plan: DecompositionPlan, depth: int = 0) -> SubTaskResult:
        upstream = {tid: r for tid, r in prior.items() if r.ok and r.output_text.strip() and not is_verification_result(r)}
        if not (is_verification_node(task) and upstream):
            return await self.inner.solve(task, prior, graph=graph, request=request, ctx=ctx, plan=plan, depth=depth)
        return await self.verify_upstream(task, upstream, graph=graph, request=request, ctx=ctx, depth=depth)

    def verifier_name(self, verdict: Optional[Verdict] = None) -> str:
        """Who answered the node: the verifier's ``spec.name``, else the verdict's ``method``, else the class name.

        Verifiers are protocol objects, often wrapped (gates, tracers), so the verdict's method is the
        portable fallback; :class:`ProtocolSolver` records the nested component under the same key.
        """
        spec = getattr(self.verifier, "spec", None)
        name = getattr(spec, "name", "") or (verdict.method if verdict is not None else "")
        return str(name or type(self.verifier).__name__)

    @staticmethod
    def upstream_evidence(upstream: dict[str, SubTaskResult], request: TaskRequest) -> list[Evidence]:
        """Evidence the upstream sub-tasks used, then the request context; sub-task outputs are excluded."""
        pooled: list[Evidence] = [e for r in upstream.values() for e in r.evidence]
        pooled += list(request.context) + evidence_from_inputs(request.inputs)
        return dedupe_preserve_order([e for e in pooled if not e.id.startswith("result:")], key=lambda e: e.id)

    async def verify_upstream(self, task: SubTask, upstream: dict[str, SubTaskResult], *, graph: TaskGraph, request: TaskRequest, ctx: RunContext, depth: int = 0) -> SubTaskResult:
        """Verify the combined upstream outputs and express the :class:`Verdict` as this node's result."""
        check_budget_before(task.id, ctx)  # BudgetExceededError becomes a budget stop in the scheduler
        combined = "\n\n".join(r.output_text.strip() for r in upstream.values())
        evidence = self.upstream_evidence(upstream, request)
        root = graph.root_task or request.task
        base_meta: dict[str, Any] = {
            "verification_node": True,
            "component": self.verifier_name(),
            "tool_hint": task.tool_hint,
            "checked_results": list(upstream),
        }
        try:
            verdict: Verdict = await self.verifier.verify(combined, task=root, evidence=evidence)
        except Exception as exc:  # noqa: BLE001 - a broken verifier must not fail the run
            ctx.warn(f"verifier raised for verification node {task.id} ({type(exc).__name__}); results left unverified")
            return SubTaskResult(
                task_id=task.id,
                status=Status.NEEDS_REVIEW,
                confidence=0.0,
                gaps=["verifier unavailable"],
                error=f"verifier unavailable ({type(exc).__name__})",
                depth=depth,
                metadata={**base_meta, "verdict": None},
            )
        charge_verdict_usage(verdict, ctx, what=f"verifier verdict for verification node {task.id}")
        raw_usage = verdict.details.get("usage") if isinstance(verdict.details, dict) else None
        try:
            usage = Usage.model_validate(raw_usage) if isinstance(raw_usage, dict) else Usage()
        except Exception:  # noqa: BLE001 - malformed usage is ignored (and was not charged)
            usage = Usage()
        known = {e.id for e in evidence}
        cited = [i for i in verdict.evidence_ids if i in known] or [e.id for e in evidence]
        issues = [str(i) for i in verdict.issues]
        summary = f"{verdict.method}: {'passed' if verdict.passed else 'failed'} (score {verdict.score:.2f}) on {', '.join(upstream)}"
        return SubTaskResult(
            task_id=task.id,
            status=Status.SUCCESS if verdict.passed else Status.NEEDS_REVIEW,
            output_text=summary + "".join(f"\n- {i}" for i in issues[:6]),
            confidence=max(0.0, min(1.0, verdict.score)),
            criteria_met=[verdict.passed] * len(task.verification_criteria),
            evidence_ids=cited,
            evidence=[e for e in evidence if e.id in set(cited)],
            gaps=[] if verdict.passed else issues[:5],
            error=None if verdict.passed else "verifier: " + ("; ".join(issues[:3]) or "verification failed"),
            usage=usage,
            depth=depth,
            metadata={**base_meta, "component": self.verifier_name(verdict), "verdict": verdict.model_dump(mode="json")},
        )


class ProtocolSolver:
    """Adapt a ``SubTaskSolver`` callable (``TaskRequest -> TaskResult``) to the node-solver shape."""

    def __init__(self, solver: SubTaskSolver, config: Optional[TaskDecompositionConfig] = None) -> None:
        self.solver = solver
        self.config = config or TaskDecompositionConfig()

    def child_request(self, task: SubTask, prior: dict[str, SubTaskResult], *, graph: TaskGraph, request: TaskRequest) -> TaskRequest:
        """Derive the child request: dependency outputs travel as ``Evidence`` so any component can consume them."""
        prior_evidence = [
            Evidence(id=f"result:{tid}", content=r.output_text, source=f"subtask:{tid}", score=r.confidence, retriever="provided", metadata={"status": r.status.value})
            for tid, r in prior.items()
            if r.ok and r.output_text
        ]
        return request.derive(
            task.description,
            inputs={**request.inputs, **task.inputs},
            context=list(request.context) + prior_evidence,
            metadata={
                **request.metadata,
                "subtask_id": task.id,
                "root_task": graph.root_task,
                "expected_output": task.expected_output,
                "verification_criteria": list(task.verification_criteria),
                "tool_hint": task.tool_hint,
            },
        )

    async def solve(self, task: SubTask, prior: dict[str, SubTaskResult], *, graph: TaskGraph, request: TaskRequest, ctx: RunContext, plan: DecompositionPlan, depth: int = 0) -> SubTaskResult:
        # Check the budget (calls, cost, deadline) *before* the nested call, not only when charging afterwards,
        # otherwise a nested component can spend past the limit (section 3.2: check before fan-outs).
        check_budget_before(task.id, ctx)
        child = self.child_request(task, prior, graph=graph, request=request)
        result: TaskResult = await self.solver(child)
        # The nested run has already happened: an overshoot (a solver spending more than the one reserved call)
        # is charged and warned about, and the completed result is kept; the scheduler stops further dispatch.
        charge_nested_usage(ctx, result.usage, what=f"nested sub-task {task.id} ({result.component or 'solver'})")
        status = map_status(result.status)
        text = result.output_text or (str(result.output) if result.output is not None else "")
        return SubTaskResult(
            task_id=task.id,
            status=status,
            output_text=text,
            confidence=result.confidence,
            evidence_ids=[e.id for e in result.evidence],
            evidence=list(result.evidence),
            gaps=list(result.warnings)[:5] + (["nested component asked for review"] if result.status == Status.NEEDS_REVIEW else []),
            error=("; ".join(result.errors) or f"nested component status {result.status.value}") if status == Status.FAILED else None,
            usage=result.usage,
            depth=depth,
            metadata={"component": result.component, "nested_request_id": result.request_id, "nested_status": result.status.value},
        )


__all__ = [
    "LLMSubTaskSolver",
    "NodeSolver",
    "ProtocolSolver",
    "VERIFICATION_HINTS",
    "VerifierNodeSolver",
    "charge_nested_usage",
    "charge_verdict_usage",
    "check_budget_before",
    "evidence_from_inputs",
    "is_verification_node",
    "is_verification_result",
    "map_status",
    "node_verdict",
    "offline_confidence",
    "render_prior",
]
