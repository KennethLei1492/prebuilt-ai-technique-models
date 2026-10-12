"""``ParallelizationModel``: plan how a prompt should be parallelised, then run it.

``plan()`` routes the task to one of eight execution patterns (sequential,
sectioning, skeleton-of-thought, voting, map-reduce, branch-solve-merge,
ensemble, speculative) with one structured LLM call online and the
:class:`~parallelization.plan.HeuristicPlanner` offline; ``execute()`` runs
the pattern on the resilient :class:`~parallelization.executor.ParallelExecutor`
and returns a :class:`TaskResult` whose ``output`` is a
:class:`ParallelizationOutput`.

The model also exposes the executor itself (``model.executor``) and the
patterns as a direct API so other components can fan out through the same
rate limits, retry budget and circuit breaker.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    ENGINEERING_RIGOR,
    BaseLLMClient,
    ComponentSpec,
    DomainProfile,
    Evidence,
    LLMError,
    LLMRequest,
    Message,
    PipelineComponent,
    PlanValidationError,
    RunContext,
    Status,
    StructuredOutputError,
    TaskRequest,
    TaskResult,
    Tracer,
    Usage,
    get_profile,
    register,
    render_evidence,
    run_sync,
    sanitize_schema,
    xml,
)

from .config import ParallelizationConfig
from .executor import ExecutorStats, ParallelExecutor
from .patterns import (
    NO_CONTEXT,
    BranchRecord,
    PatternResult,
    PatternRunner,
    ToolCall,
    ToolResult,
    generic_map_reduce,
    parallel_tool_calls,
    run_branch_solve_merge,
    run_ensemble,
    run_pattern,
    run_sectioning,
    run_skeleton,
    run_voting,
    speculative,
)
from .plan import HeuristicPlanner, ParallelPlan, Partition, PlannerOutput, extract_features, input_items, input_text
from .prompts import PLANNER

T = TypeVar("T")
R = TypeVar("R")


class ParallelizationOutput(BaseModel):
    """Typed ``result.output`` of the parallelization model."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    answer: str
    aggregator: str
    plan_summary: str
    branches: list[BranchRecord] = Field(default_factory=list)
    failed_branches: list[str] = Field(default_factory=list)
    quorum_met: bool = True
    llm_calls: int = 0
    expected_calls: int = 0
    expected_speedup: float = 1.0
    measured_speedup: float = 1.0
    total_work_s: float = 0.0
    wall_s: float = 0.0
    cache_warmups: int = 0
    cache_waits: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


def build_context(request: TaskRequest, *, include_inputs: bool = True, max_chars_each: int = 4000) -> str:
    """Render upstream evidence and text-like inputs for the user turn.

    Evidence is rendered as ``<documents>`` so branches can cite ``[doc-id]``;
    text inputs (``text``, ``document``, ``documents`` ...) are rendered in an
    ``<inputs>`` block. Map-reduce passes ``include_inputs=False`` because it
    chunks the inputs itself.
    """
    parts: list[str] = []
    if request.context:
        parts.append(render_evidence(request.context, max_chars_each=max_chars_each))
    if include_inputs:
        text = input_text(request)
        if text and not request.context:
            parts.append(xml("inputs", text))
        elif text:
            # evidence already rendered; add only inputs that are not evidence
            extra = [k for k in ("text", "document", "content", "log", "logs", "transcript", "documents") if k in request.inputs]
            if extra:
                parts.append(xml("inputs", text))
    other = {k: v for k, v in request.inputs.items() if k not in ("text", "document", "content", "log", "logs", "transcript", "documents", "items")}
    if other:
        parts.append(xml("parameters", "\n".join(f"{k}: {v}" for k, v in sorted(other.items()))))
    return "\n".join(parts) if parts else NO_CONTEXT


def run_stats(stats: ExecutorStats) -> dict[str, Any]:
    """Serialise a run's own executor counters for ``result.metadata['executor']``.

    The counters come from the per-run :class:`ExecutorStats` the runner
    passes to every executor call, so they stay correct when several runs
    share one executor concurrently.
    """
    out = stats.model_dump()
    out["rate_limit_wait_s"] = round(stats.rate_limit_wait_s, 6)
    return out


def rate_limit_mismatch(plan: ParallelPlan, executor: ParallelExecutor) -> str | None:
    """Warning text when the executor enforces different limits than the plan reports."""
    configured = plan.rate_limits.model_dump()
    enforced = executor.rate_limiter.limits() if executor.rate_limiter is not None else {k: None for k in configured}
    if configured == enforced:
        return None
    return f"plan.rate_limits {configured} differ from the limits the shared executor enforces {enforced}; the executor's limits apply"


@register
class ParallelizationModel(PipelineComponent):
    """Prompt-level parallel reasoning plus a production-grade parallel executor.

    Parameters
    ----------
    llm:
        Any :class:`BaseLLMClient`; defaults to :func:`pipeline_core.make_llm`
        (a ``MockLLM`` without credentials).
    config:
        :class:`ParallelizationConfig`; keyword overrides are applied on top.
    executor:
        Share an existing :class:`ParallelExecutor` (and therefore its rate
        limiter, retry budget and breaker) across components; by default one
        is built from ``config``.
    """

    spec = ComponentSpec(
        name="parallelization",
        version="1.0.0",
        description="Executes LLM and tool work concurrently and safely: bounded concurrency, rate limiting, retries, circuit breaking, failure policies, and the parallel reasoning patterns (sectioning, skeleton-of-thought, voting, map-reduce, branch-solve-merge, ensemble, speculative).",
        capabilities=["parallel-execution", "fan-out", "voting", "map-reduce", "skeleton-of-thought", "branch-solve-merge", "ensemble", "speculative-execution", "rate-limiting", "retry", "circuit-breaker"],
        consumes=["task", "inputs.items (list for fan-out)", "inputs.text|document|documents (long input for map-reduce)", "context (Evidence)"],
        produces=["ParallelizationOutput(pattern, answer, branches, failed_branches, speedup metrics)"],
        composes_with=["task_decomposition", "hybrid_retrieval", "answer_verification", "pipeline_rag", "caching"],
    )

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        config: ParallelizationConfig | None = None,
        executor: ParallelExecutor | None = None,
        tracer: Tracer | None = None,
        raise_on_error: bool = False,
        **overrides: Any,
    ) -> None:
        super().__init__(llm, tracer=tracer, raise_on_error=raise_on_error)
        base = config or ParallelizationConfig()
        self.config: ParallelizationConfig = base.model_copy(update=overrides) if overrides else base  # type: ignore[assignment]
        if overrides:
            self.config = ParallelizationConfig.model_validate(self.config.model_dump())
        self.planner = HeuristicPlanner(self.config)
        self.executor: ParallelExecutor = executor or ParallelExecutor.from_config(self.config)

    # ------------------------------------------------------------------ plan
    async def plan(self, request: TaskRequest, ctx: RunContext) -> ParallelPlan:
        """Decide the execution pattern.

        Heuristics are used when ``ctx.offline``, when
        ``constraints.deterministic`` is set, when ``planner_mode`` is
        ``heuristic``, or when fewer than two LLM calls remain. Otherwise one
        structured call fills :class:`PlannerOutput`; invalid output falls back
        to the heuristic decision with a warning. Every decision is finalised
        against domain policy, budgets and configuration.
        """
        profile = ctx.profile
        remaining = ctx.budget.remaining_calls()
        mode = self.config.planner_mode
        use_llm = mode == "llm" or (mode == "auto" and not ctx.offline and not request.constraints.deterministic)
        if remaining is not None and remaining < 2:
            use_llm = False
        if not use_llm:
            return self._surface(self.planner.plan(request, profile, remaining_calls=remaining), ctx)
        features = extract_features(request, profile)
        items = input_items(request)
        inputs_summary = f"keys={sorted(request.inputs)}; items={len(items)}; input_tokens~{features.input_tokens}; evidence={len(request.context)}"
        if items:
            inputs_summary += "\nitems: " + "; ".join(str(it)[:80] for it in items[:16])
        limits = (
            f"max_partitions={self.config.max_partitions}; max_samples={self.config.voting_max_samples}; "
            f"max_parallelism={request.constraints.max_parallelism}; remaining_llm_calls={remaining if remaining is not None else 'unbounded'}; "
            f"domain_risk={profile.risk_level}"
        )
        rendered = PLANNER.render(task=request.task, inputs_summary=inputs_summary, features=features.summary(), limits=limits)
        stable = [*rendered.system_blocks, profile.system_text()]
        if profile.risk_level in ("high", "safety_critical"):
            stable.append(ENGINEERING_RIGOR)
        req = LLMRequest(
            system_stable=stable,
            messages=[Message(role="user", content=rendered.user)],
            output_schema=sanitize_schema(PlannerOutput.model_json_schema()),
            effort=self.config.planner_effort,
            tags=[PLANNER.name],
        )
        try:
            resp = await ctx.llm_call(req, step="plan.llm")
            decision = self.planner.validate_llm_output(resp.parsed)
            planner = "llm"
        except (PlanValidationError, StructuredOutputError, LLMError) as exc:
            ctx.warn(f"LLM planner output unusable ({type(exc).__name__}: {exc}); using heuristic plan")
            decision = self.planner.choose(features, request, profile)
            planner = "heuristic"
        return self._surface(self.planner.finalize(decision, request, profile, remaining_calls=ctx.budget.remaining_calls(), planner=planner), ctx)

    @staticmethod
    def _surface(plan: ParallelPlan, ctx: RunContext) -> ParallelPlan:
        """Report planning degradations (dropped or folded partitions) as run warnings."""
        for w in plan.warnings:
            ctx.warn(w)
        return plan

    # --------------------------------------------------------------- execute
    async def execute(self, request: TaskRequest, plan: ParallelPlan, ctx: RunContext) -> TaskResult:
        """Run the planned pattern and grade the outcome."""
        runner = PatternRunner.from_ctx(ctx, self.config, self.executor, cache_aware=plan.cache_aware)
        context = build_context(request, include_inputs=plan.pattern != "map_reduce")
        text = input_text(request) if plan.pattern == "map_reduce" else ""
        mismatch = rate_limit_mismatch(plan, self.executor)
        if mismatch:
            ctx.warn(mismatch)
        pr: PatternResult = await run_pattern(runner, plan, request.task, context=context, text=text)
        exec_delta = run_stats(runner.stats)

        threshold = max(ctx.profile.min_confidence_to_accept, request.constraints.min_confidence)
        errors: list[str] = []
        if not pr.quorum_met:
            status = Status.NEEDS_REVIEW
            errors.append(f"failure policy '{plan.failure_policy}' not satisfied: failed branches {pr.failed_branches}")
        elif not pr.answer.strip():
            status = Status.FAILED
            errors.append("no answer produced")
        elif pr.confidence < threshold:
            status = Status.NEEDS_REVIEW
        elif pr.partial:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS
        basis = pr.confidence_basis
        if status == Status.NEEDS_REVIEW and pr.quorum_met and pr.answer.strip():
            basis += f"; below acceptance threshold {threshold:.2f} ({'domain' if ctx.profile.min_confidence_to_accept >= request.constraints.min_confidence else 'request'})"

        output = ParallelizationOutput(
            pattern=pr.pattern,
            answer=pr.answer,
            aggregator=pr.aggregator,
            plan_summary=plan.summary(),
            branches=pr.branches,
            failed_branches=pr.failed_branches,
            quorum_met=pr.quorum_met,
            llm_calls=pr.llm_calls,
            expected_calls=plan.expected_calls,
            expected_speedup=plan.expected_speedup,
            measured_speedup=pr.measured_speedup,
            total_work_s=pr.total_work_s,
            wall_s=pr.wall_s,
            cache_warmups=runner.gate.warmups,
            cache_waits=runner.gate.waits,
            details=pr.details,
        )
        evidence = [
            Evidence(id=f"branch:{b.id}", content=b.text, source=f"parallelization:{pr.pattern}:{b.stage}", score=label_conf(b.confidence_label), retriever="provided", metadata={"stage": b.stage, "latency_s": b.latency_s})
            for b in pr.branches
            if b.ok and b.text and b.stage not in ("merge", "aggregate", "reduce", "usc", "coherence", "verify")
        ]
        return self.new_result(
            request,
            status=status,
            output=output,
            output_text=pr.answer,
            evidence=evidence,
            confidence=pr.confidence,
            confidence_basis=basis,
            warnings=list(pr.warnings),
            errors=errors,
            metadata={"plan": plan.model_dump(mode="json"), "executor": exec_delta, "cache_gate": runner.gate.snapshot(), "threshold": threshold},
        )

    # ------------------------------------------------------------ direct API
    async def map(self, fn: Callable[[T], Awaitable[R]], items: Sequence[T], *, concurrency: Optional[int] = None) -> list[Any]:
        """``Executor`` protocol: run ``fn`` over ``items`` through the shared executor."""
        return await self.executor.map(fn, items, concurrency=concurrency)

    def heuristic_plan(self, request: TaskRequest | str, *, domain: str = "general", **request_kwargs: Any) -> ParallelPlan:
        """Return the heuristic plan for a task without calling the LLM (inspection, CLI ``--plan-only``)."""
        if isinstance(request, str):
            request = TaskRequest(task=request, domain=domain, **request_kwargs)
        return self.planner.plan(request, get_profile(request.domain), remaining_calls=request.constraints.max_llm_calls)

    def _runner(self, domain: str | DomainProfile) -> PatternRunner:
        profile = domain if isinstance(domain, DomainProfile) else get_profile(domain)
        return PatternRunner.from_llm(self.llm, self.config, self.executor, profile, cache_aware=self.config.cache_aware_fan_out)

    async def skeleton_of_thought(self, prompt: str, *, context: str = NO_CONTEXT, domain: str = "general") -> PatternResult:
        """Skeleton -> parallel expansion -> merge. Usage is reported in ``result.usage``."""
        return await run_skeleton(self._runner(domain), prompt, context=context)

    async def branch_solve_merge(self, prompt: str, *, context: str = NO_CONTEXT, domain: str = "general") -> PatternResult:
        """Branch-Solve-Merge over model-proposed criteria or constraints."""
        return await run_branch_solve_merge(self._runner(domain), prompt, context=context)

    async def vote(self, prompt: str, *, samples: int | None = None, max_samples: int | None = None, free_form: bool = False, context: str = NO_CONTEXT, domain: str = "general") -> PatternResult:
        """Sampling-and-voting with early stopping (USC selection when ``free_form``)."""
        profile = get_profile(domain)
        s = samples or self.config.voting_initial(profile.risk_level)
        return await run_voting(self._runner(profile), prompt, context=context, samples=s, max_samples=max(s, max_samples or self.config.voting_max_samples), aggregator="usc" if free_form else "code")

    async def ensemble(self, prompt: str, *, proposers: int | None = None, layers: int | None = None, context: str = NO_CONTEXT, domain: str = "general") -> PatternResult:
        """Mixture-of-Agents (homogeneous proposers + critical aggregator)."""
        return await run_ensemble(self._runner(domain), prompt, context=context, proposers=proposers or self.config.ensemble_proposers, layers=layers or self.config.ensemble_layers)

    mixture_of_agents = ensemble

    async def sectioning(self, task: str, partitions: Sequence[Partition | str], *, context: str = NO_CONTEXT, domain: str = "general", aggregator: str = "llm") -> PatternResult:
        """Solve independent sub-tasks (strings or :class:`Partition` with dependencies) as a DAG and merge."""
        parts = [p if isinstance(p, Partition) else Partition(id=f"p{i + 1}", task=p) for i, p in enumerate(partitions)]
        return await run_sectioning(self._runner(domain), task, parts, context=context, aggregator=aggregator)

    async def map_reduce(self, items: Sequence[str], map_prompt: str, reduce_prompt: str, *, fanin: int | None = None, domain: str = "general", system: Sequence[str] = ()) -> PatternResult:
        """Prompt-driven map-reduce: ``{item}`` in ``map_prompt``, ``{records}`` in ``reduce_prompt``."""
        return await generic_map_reduce(self._runner(domain), items, map_prompt, reduce_prompt, fanin=fanin, system=system)

    async def speculative(self, fast: Callable[[], Awaitable[T]], slow: Callable[[], Awaitable[T]], verifier: Callable[[T], Awaitable[bool]]) -> tuple[T, str]:
        """Race a cheap path against an expensive one; see :func:`parallelization.patterns.speculative`."""
        return await speculative(fast, slow, verifier)

    async def parallel_tool_calls(self, calls: Sequence[ToolCall | dict[str, Any]], tools: dict[str, Callable[..., Any]], *, ordered_tools: Sequence[str] | None = None) -> list[ToolResult]:
        """Execute a model turn's tool calls concurrently (ordered tools sequentially)."""
        parsed = [c if isinstance(c, ToolCall) else ToolCall.model_validate(c) for c in calls]
        return await parallel_tool_calls(self.executor, parsed, tools, ordered_tools=ordered_tools if ordered_tools is not None else self.config.ordered_tools)

    # ------------------------------------------------------------- utilities
    def describe(self) -> dict[str, Any]:
        return {**self.spec.model_dump(), "llm": self.llm.describe(), "config": self.config.model_dump(), "executor": self.executor.snapshot()}

    def run_pattern_sync(self, coro: Awaitable[T]) -> T:
        """Run a direct-API coroutine from synchronous code."""
        return run_sync(coro)


def label_conf(label: str | None) -> float:
    from .aggregation import label_to_confidence

    return label_to_confidence(label, default=0.5)


__all__ = ["ParallelizationModel", "ParallelizationOutput", "build_context", "rate_limit_mismatch", "run_stats"]
