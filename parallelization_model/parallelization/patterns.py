"""Parallel reasoning patterns from the literature, built on :class:`ParallelExecutor`.

Every pattern is an ``async`` function over a :class:`PatternRunner`, which
abstracts *how* the LLM is called (``ctx.llm_call`` inside a component run,
``llm.complete`` with self-accounting in the direct API) and adds the three
cross-cutting behaviours every fan-out needs:

* **budget guard** - a branch is never started when the remaining call budget
  (minus calls already in flight) is zero, so fan-outs degrade to partial
  results instead of raising out of ``execute()``;
* **warm-then-burst prompt caching** (:class:`PrefixWarmGate`) - the first
  request on a cacheable shared prefix runs alone and siblings are released
  once its response has arrived, because a cache entry only exists after the
  first response begins (Anthropic prompt-caching documentation);
* **resilience** - every call goes through the executor's rate limiter,
  retry policy, circuit breaker and bulkhead.

Patterns implemented (dossier technique -> function):

================  ===============================================================
``sequential``    one careful call (the SoT-R fallback for step-dependent work)
``sectioning``    LLMCompiler-style DAG of independent sub-tasks, LLM/concat merge
``skeleton``      Skeleton-of-Thought: skeleton -> parallel expansion -> concat
``voting``        self-consistency / Agent Forest sampling with early stopping; USC
``map_reduce``    LLM x MapReduce structured records, confidence-aware reduce
``bsm``           Branch-Solve-Merge
``ensemble``      Mixture-of-Agents (Self-MoA homogeneous proposers) + aggregator
``speculative``   race a fast model against the default model, verify the fast one
================  ===============================================================

plus :func:`parallel_tool_calls` (execute a model turn's tool calls
concurrently, ordered tools sequentially) and :func:`cache_aware_fan_out`.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import time
from contextlib import asynccontextmanager, nullcontext
from typing import Any, AsyncIterator, Awaitable, Callable, ContextManager, Mapping, Optional, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    ENGINEERING_RIGOR,
    MIN_CACHEABLE_TOKENS,
    BaseLLMClient,
    BudgetExceededError,
    DomainProfile,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    PromptTemplate,
    RunContext,
    StructuredOutputError,
    Usage,
    chunk_text,
    estimate_tokens,
    sanitize_schema,
    validate_structured,
    xml,
)

from .aggregation import (
    combine_confidence,
    concat_in_order,
    confidence_from_vote,
    label_to_confidence,
    majority_vote,
    missing_branch_notice,
    success_fraction,
    weighted_confidence,
)
from .config import ParallelizationConfig
from .executor import ExecutorStats, FailurePolicy, ParallelExecutionError, ParallelExecutor, TaskOutcome
from .plan import ParallelPlan, Partition, truncate_partitions
from .prompts import (
    AGGREGATE,
    BRANCH,
    COHERENCE,
    EXPAND,
    MAP,
    MERGE,
    PROPOSE,
    PROPOSER_PERSPECTIVES,
    REDUCE,
    SAMPLE,
    SAMPLE_FRAMINGS,
    SEQUENTIAL,
    SKELETON,
    SOLVE,
    USC_SELECT,
    VERIFY_FAST,
    AggregateOutput,
    AnswerOutput,
    BranchOutput,
    BranchSolution,
    CoherenceOutput,
    ExpansionOutput,
    MapRecord,
    MergeOutput,
    ProposerOutput,
    ReduceOutput,
    SampleOutput,
    SkeletonOutput,
    USCOutput,
    VerifyOutput,
    system_blocks,
)

T = TypeVar("T")
LLMCall = Callable[..., Awaitable[LLMResponse]]
NO_CONTEXT = "(no additional context)"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


class BranchRecord(BaseModel):
    """What one branch of a fan-out produced (kept in ``result.output`` for traceability)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    stage: str = Field(default="branch", description="Pattern stage: skeleton, expand, sample, map, reduce, solve, merge, propose, aggregate, ...")
    ok: bool = True
    text: str = ""
    confidence_label: Optional[str] = None
    latency_s: float = 0.0
    attempts: int = 1
    retries: int = 0
    error: Optional[str] = None
    hedged: bool = False
    cache_read_tokens: int = 0


class PatternResult(BaseModel):
    """Outcome of one parallel pattern: the answer plus everything needed to judge it."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    answer: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_basis: str = ""
    aggregator: str = "code"
    branches: list[BranchRecord] = Field(default_factory=list)
    failed_branches: list[str] = Field(default_factory=list)
    quorum_met: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    total_work_s: float = Field(default=0.0, description="Sum of all call latencies (what a sequential run would have taken)")
    wall_s: float = Field(default=0.0, description="Wall-clock of the pattern")
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage, description="Filled only by the direct API (no RunContext accounting)")

    @property
    def measured_speedup(self) -> float:
        """``total_work_s / wall_s`` (1.0 when nothing was measured)."""
        if self.wall_s <= 0 or self.total_work_s <= 0:
            return 1.0
        return round(max(1.0, self.total_work_s / self.wall_s), 3)

    @property
    def partial(self) -> bool:
        return bool(self.failed_branches)


# --------------------------------------------------------------------------
# Warm-then-burst gate
# --------------------------------------------------------------------------


class PrefixWarmGate:
    """Admission control implementing Anthropic's cache-aware fan-out guidance.

    A prompt-cache entry becomes available only after the first response on a
    prefix begins, so a naive burst pays the cache write N times. The gate lets
    the first request on each cacheable prefix fingerprint run alone; siblings
    wait for its completion (success or failure) and then burst. Prefixes below
    the model's minimum cacheable size are not gated: nothing would cache.

    Fan-outs enter the gate through the executor's ``admit`` hook, i.e.
    *before* the rate limiter, bulkhead slot and per-attempt timeout, so a
    sibling's wait for the warm-up never consumes its timeout and is reported
    as ``TaskOutcome.wait_s`` rather than call latency.
    """

    def __init__(self, *, enabled: bool = True, min_prefix_tokens: int | None = None) -> None:
        self.enabled = enabled
        self.min_prefix_tokens = min_prefix_tokens
        self._events: dict[str, asyncio.Event] = {}
        self.warmups = 0
        self.waits = 0

    def threshold(self, model: str | None) -> int:
        if self.min_prefix_tokens is not None:
            return self.min_prefix_tokens
        return MIN_CACHEABLE_TOKENS.get(model or "", 512)

    def cacheable(self, request: LLMRequest, model: str | None) -> bool:
        return self.enabled and request.cache and request.prefix_tokens() >= self.threshold(request.model or model)

    @asynccontextmanager
    async def admit(self, request: LLMRequest, model: str | None) -> AsyncIterator[bool]:
        """Yield ``True`` for the warm-up request of a prefix, ``False`` otherwise."""
        if not self.cacheable(request, model):
            yield False
            return
        fp = request.prefix_fingerprint(request.model or model)
        ev = self._events.get(fp)
        if ev is None:
            ev = asyncio.Event()
            self._events[fp] = ev
            self.warmups += 1
            try:
                yield True
            finally:
                ev.set()
            return
        if not ev.is_set():
            self.waits += 1
            await ev.wait()
        yield False

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "warmups": self.warmups, "waits": self.waits}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class PatternRunner:
    """Executes LLM calls for the patterns with budget, cache and resilience behaviour.

    Build one per run with :meth:`from_ctx` (inside ``plan``/``execute``) or
    :meth:`from_llm` (direct API; usage accumulates in :attr:`usage`).
    """

    def __init__(
        self,
        *,
        call: LLMCall,
        config: ParallelizationConfig,
        executor: ParallelExecutor,
        profile: DomainProfile,
        model: str | None = None,
        remaining_calls: Callable[[], Optional[int]] = lambda: None,
        remaining_time: Callable[[], Optional[float]] = lambda: None,
        warn: Callable[[str], None] | None = None,
        step: Callable[[str], ContextManager[Any]] | None = None,
        cache_aware: bool = True,
        min_prefix_tokens: int | None = None,
    ) -> None:
        self._call = call
        self.config = config
        self.executor = executor
        self.profile = profile
        self.model = model
        self.remaining_calls = remaining_calls
        self.remaining_time = remaining_time
        self._warn = warn
        self._step = step
        self.gate = PrefixWarmGate(enabled=cache_aware, min_prefix_tokens=min_prefix_tokens)
        self.records: list[BranchRecord] = []
        self.warnings: list[str] = []
        self.usage = Usage()
        self.calls = 0
        self.total_work_s = 0.0
        self._inflight = 0
        self.stats = ExecutorStats()
        """Executor counters attributable to this runner only (correct on a shared executor)."""

    # ----------------------------------------------------------- factories
    @classmethod
    def from_ctx(cls, ctx: RunContext, config: ParallelizationConfig, executor: ParallelExecutor, *, cache_aware: bool = True) -> "PatternRunner":
        """Runner whose calls go through ``ctx.llm_call`` (budget + tracing)."""
        return cls(
            call=ctx.llm_call,
            config=config,
            executor=executor,
            profile=ctx.profile,
            model=getattr(ctx.llm, "model", None),
            remaining_calls=ctx.budget.remaining_calls,
            remaining_time=ctx.remaining_time,
            warn=ctx.warn,
            step=ctx.step,
            cache_aware=cache_aware and not ctx.request.constraints.deterministic,
        )

    @classmethod
    def from_llm(cls, llm: BaseLLMClient, config: ParallelizationConfig, executor: ParallelExecutor, profile: DomainProfile, *, cache_aware: bool = True) -> "PatternRunner":
        """Runner for the direct API: calls ``llm.complete`` and accounts usage itself."""

        async def call(request: LLMRequest, *, step: str | None = None) -> LLMResponse:
            return await llm.complete(request)

        return cls(call=call, config=config, executor=executor, profile=profile, model=getattr(llm, "model", None), cache_aware=cache_aware)

    # ------------------------------------------------------------ helpers
    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self._warn is not None:
            self._warn(message)

    def step(self, name: str) -> ContextManager[Any]:
        return self._step(name) if self._step is not None else nullcontext()

    def budget_cap(self, n: int, *, reserve: int = 0) -> int:
        """Largest fan-out width <= ``n`` that leaves ``reserve`` calls for later stages."""
        rem = self.remaining_calls()
        if rem is None:
            return n
        capped = max(0, min(n, rem - self._inflight - reserve))
        if capped < n:
            self.warn(f"budget: fan-out reduced from {n} to {capped} to respect max_llm_calls")
        return capped

    def system(self, template: PromptTemplate, **values: Any) -> list[str]:
        """Stable system blocks: template rules, then the domain profile, then rigor for risky domains."""
        blocks = system_blocks(template, **values)
        blocks.append(self.profile.system_text())
        if self.profile.risk_level in ("high", "safety_critical"):
            blocks.append(ENGINEERING_RIGOR)
        return blocks

    def request(
        self,
        template: PromptTemplate,
        schema: type[BaseModel],
        *,
        effort: Any = None,
        model: str | None = None,
        system_values: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
        **values: Any,
    ) -> LLMRequest:
        """Render ``template`` into an :class:`LLMRequest` with structured output."""
        rendered = template.render(**{k: str(v) for k, v in values.items()})
        return LLMRequest(
            system_stable=self.system(template, **(system_values or {})),
            messages=[Message(role="user", content=rendered.user)],
            output_schema=sanitize_schema(schema.model_json_schema()),
            effort=effort,
            model=model,
            tags=[template.name, *tags],
        )

    # --------------------------------------------------------------- calls
    def admit(self, item: tuple[str, LLMRequest]) -> Any:
        """Executor ``admit`` hook for ``(key, request)`` items: the warm-then-burst gate."""
        return self.gate.admit(item[1], self.model)

    def exec_kwargs(self, *, admit: Any = None) -> dict[str, Any]:
        """Keyword arguments every executor call of this runner passes (per-run stats, gate)."""
        kw: dict[str, Any] = {"stats": self.stats}
        if admit is not None:
            kw["admit"] = admit
        return kw

    async def call(self, request: LLMRequest, *, step: str | None = None, gated: bool = True) -> LLMResponse:
        """One LLM call through the budget guard and (when ``gated``) the warm-then-burst gate.

        Fan-outs pass ``gated=False`` because the executor already admitted the
        request through :meth:`admit` before its timeout started.
        """
        rem = self.remaining_calls()
        if rem is not None and rem - self._inflight <= 0:
            raise BudgetExceededError("LLM call budget exhausted before branch was started")
        t_left = self.remaining_time()
        if t_left is not None and t_left <= 0:
            raise BudgetExceededError("latency budget exhausted before branch was started")
        self._inflight += 1
        try:
            if gated:
                async with self.gate.admit(request, self.model):
                    resp = await self._call(request, step=step)
            else:
                resp = await self._call(request, step=step)
        finally:
            self._inflight -= 1
        self.calls += 1
        self.total_work_s += resp.latency_s or resp.usage.latency_s
        self.usage = self.usage + resp.usage
        return resp

    async def structured(self, request: LLMRequest, schema: type[T], *, step: str | None = None, gated: bool = True) -> tuple[T, LLMResponse]:
        """Call and validate the parsed output into ``schema``."""
        resp = await self.call(request, step=step, gated=gated)
        parsed = validate_structured(resp.parsed, schema)
        return parsed, resp

    async def fan_out(
        self,
        keyed_requests: Sequence[tuple[str, LLMRequest]],
        schema: type[BaseModel],
        *,
        stage: str,
        policy: FailurePolicy | None = None,
        concurrency: int | None = None,
    ) -> tuple[list[TaskOutcome], bool]:
        """Run requests concurrently through the executor; returns (outcomes, policy_satisfied).

        Outcome values are the validated pydantic objects. Never raises for
        branch failures; an unmet ``fail_fast``/``quorum`` policy is reported
        through the boolean.
        """
        if not keyed_requests:
            return [], True
        keys = [k for k, _ in keyed_requests]

        async def one(item: tuple[str, LLMRequest]) -> BaseModel:
            key, req = item
            parsed, _ = await self.structured(req, schema, step=f"{stage}:{key}", gated=False)
            return parsed

        est_in = estimate_tokens("\n".join(keyed_requests[0][1].system_stable) + keyed_requests[0][1].last_user_text())
        ok = True
        try:
            outs = await self.executor.map_outcomes(
                one,
                list(keyed_requests),
                keys=keys,
                concurrency=concurrency,
                failure_policy=policy or FailurePolicy.best_effort(),
                estimated_input_tokens=est_in,
                resource="llm",
                **self.exec_kwargs(admit=self.admit),
            )
        except ParallelExecutionError as exc:
            outs = exc.outcomes
            ok = False
            self.warn(str(exc))
        self._record(outs, stage)
        return outs, ok

    def _record(self, outs: Sequence[TaskOutcome], stage: str) -> None:
        for o in outs:
            self.records.append(
                BranchRecord(
                    id=o.key,
                    stage=stage,
                    ok=o.ok,
                    text=_answer_text(o.value) if o.ok else "",
                    confidence_label=getattr(o.value, "confidence", None) if o.ok else None,
                    latency_s=round(o.latency_s, 6),
                    attempts=o.attempts,
                    retries=o.retries,
                    error=o.error,
                    hedged=o.hedged,
                )
            )

    def record_single(self, key: str, stage: str, parsed: Any, resp: LLMResponse) -> None:
        self.records.append(
            BranchRecord(
                id=key,
                stage=stage,
                ok=True,
                text=_answer_text(parsed),
                confidence_label=getattr(parsed, "confidence", None),
                latency_s=round(resp.latency_s or resp.usage.latency_s, 6),
                cache_read_tokens=resp.usage.cache_read_input_tokens,
            )
        )

    def result(self, pattern: str, *, started: float, **kw: Any) -> PatternResult:
        """Assemble a :class:`PatternResult` with the runner's bookkeeping attached."""
        return PatternResult(
            pattern=pattern,
            branches=list(self.records),
            llm_calls=self.calls,
            total_work_s=round(self.total_work_s, 6),
            wall_s=round(time.perf_counter() - started, 6),
            warnings=list(self.warnings),
            usage=self.usage,
            **kw,
        )


def _answer_text(parsed: Any) -> str:
    for attr in ("answer", "text", "final_answer"):
        v = getattr(parsed, attr, None)
        if isinstance(v, str):
            return v
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()
    return "" if parsed is None else str(parsed)


def _failed_ids(outs: Sequence[TaskOutcome]) -> list[str]:
    return [o.key for o in outs if not o.ok]


def _label_conf(outs: Sequence[TaskOutcome]) -> float:
    pairs = [(label_to_confidence(getattr(o.value, "confidence", None)), 1.0) for o in outs if o.ok]
    return weighted_confidence(pairs) if pairs else 0.0


def _policy(plan: ParallelPlan | None) -> FailurePolicy:
    if plan is None:
        return FailurePolicy.best_effort()
    return FailurePolicy(mode=plan.failure_policy, quorum=plan.quorum)


def _quorum_failure(runner: PatternRunner, pattern: str, started: float, outs: Sequence[TaskOutcome], aggregator: str) -> PatternResult:
    failed = _failed_ids(outs)
    notice = missing_branch_notice(failed)
    return runner.result(
        pattern,
        started=started,
        answer=f"Failure policy not satisfied; no answer was assembled. {notice}".strip(),
        confidence=0.0,
        confidence_basis=f"{len(outs) - len(failed)}/{len(outs)} branches succeeded; the failure policy forbids aggregating partial results",
        aggregator=aggregator,
        failed_branches=failed,
        quorum_met=False,
    )


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------


async def run_sequential(runner: PatternRunner, task: str, context: str = NO_CONTEXT) -> PatternResult:
    """One careful call: the no-parallelism path for step-dependent tasks."""
    started = time.perf_counter()
    cfg = runner.config
    with runner.step("answer"):
        req = runner.request(SEQUENTIAL, AnswerOutput, effort=cfg.aggregator_effort, task=task, context=context or NO_CONTEXT)
        parsed, resp = await runner.structured(req, AnswerOutput, step="answer")
    runner.record_single("answer", "answer", parsed, resp)
    conf = label_to_confidence(parsed.confidence)
    return runner.result(
        "sequential",
        started=started,
        answer=parsed.answer,
        confidence=conf,
        confidence_basis=f"single sequential call; verbalised confidence '{parsed.confidence}'",
        aggregator="code",
    )


async def run_sectioning(
    runner: PatternRunner,
    task: str,
    partitions: Sequence[Partition],
    *,
    context: str = NO_CONTEXT,
    aggregator: str = "llm",
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
) -> PatternResult:
    """Independent sub-tasks as a DAG (LLMCompiler); merged by an LLM call or by ordered concatenation."""
    started = time.perf_counter()
    cfg = runner.config
    policy = policy or FailurePolicy.best_effort()
    parts = {p.id: p for p in partitions}
    reserve = 1 if aggregator == "llm" else 0
    cap = runner.budget_cap(len(partitions), reserve=reserve)
    if cap < len(partitions):
        # keep a dependency-closed subset so no kept node waits on a dropped one
        kept, dropped = truncate_partitions(list(partitions), cap)
        runner.warn(f"budget: partitions {dropped} not run")
        partitions = kept
    graph = {p.id: list(p.depends_on) for p in partitions}

    def build(node: str, deps: dict[str, Any]) -> LLMRequest:
        sub = parts[node].task
        if deps:
            sub += "\n<dependencies>\n" + "\n".join(xml("dependency", _answer_text(v), id=k) for k, v in deps.items()) + "\n</dependencies>"
        return runner.request(SOLVE, BranchSolution, effort=cfg.branch_effort, task=task, context=context or NO_CONTEXT, branch_id=node, branch=sub)

    async def solve(node: str, deps: dict[str, Any]) -> BranchSolution:
        parsed, _ = await runner.structured(build(node, deps), BranchSolution, step=f"solve:{node}", gated=False)
        return parsed

    def admit(node: str, deps: dict[str, Any]) -> Any:
        return runner.gate.admit(build(node, deps), runner.model)

    quorum_met = True
    with runner.step("solve"):
        try:
            results = await runner.executor.dag(graph, solve, failure_policy=policy, concurrency=concurrency, resource="llm", **runner.exec_kwargs(admit=admit))
            outs = [results[k] for k in graph]
        except ParallelExecutionError as exc:
            outs = list(exc.outcomes)
            quorum_met = False
            runner.warn(str(exc))
    runner._record(outs, "solve")
    if not quorum_met or not any(o.ok for o in outs):
        return _quorum_failure(runner, "sectioning", started, outs, aggregator)

    failed = _failed_ids(outs)
    structural = success_fraction([o.ok for o in outs])
    if aggregator == "llm":
        solutions = "\n".join(
            xml(
                "solution",
                f"sub_task: {parts[o.key].task}\nanswer: {o.value.answer}\nevidence: {'; '.join(o.value.evidence) or 'none'}\nrationale: {o.value.rationale}\nconfidence: {o.value.confidence}",
                id=o.key,
            )
            for o in outs
            if o.ok
        )
        with runner.step("merge"):
            try:
                req = runner.request(MERGE, MergeOutput, effort=cfg.aggregator_effort, task=task, solutions=solutions, notice=missing_branch_notice(failed))
                merged, resp = await runner.structured(req, MergeOutput, step="merge")
                runner.record_single("merge", "merge", merged, resp)
                answer = merged.answer
                if merged.unresolved:
                    answer += "\n\nUnresolved: " + "; ".join(merged.unresolved)
                verbal = label_to_confidence(merged.confidence)
                basis = f"{len(outs) - len(failed)}/{len(outs)} partitions solved; LLM merge confidence '{merged.confidence}'"
            except (BudgetExceededError, LLMError) as exc:
                runner.warn(f"merge call unavailable ({type(exc).__name__}); falling back to ordered concatenation")
                aggregator = "concat"
    if aggregator != "llm":
        ordered = [(i, f"[{o.key}] {parts[o.key].task}\n{o.value.answer}") for i, o in enumerate(outs) if o.ok]
        answer = concat_in_order(ordered)
        notice = missing_branch_notice(failed)
        if notice:
            answer += "\n\n" + notice
        verbal = _label_conf(outs)
        basis = f"{len(outs) - len(failed)}/{len(outs)} partitions solved; mean verbalised confidence {verbal:.2f}; deterministic concatenation"
    conf = combine_confidence(structural, verbal)
    return runner.result(
        "sectioning",
        started=started,
        answer=answer,
        confidence=conf,
        confidence_basis=basis,
        aggregator=aggregator,
        failed_branches=failed,
        details={"graph": graph},
    )


async def run_skeleton(
    runner: PatternRunner,
    task: str,
    *,
    context: str = NO_CONTEXT,
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
) -> PatternResult:
    """Skeleton-of-Thought: skeleton -> parallel point expansion (shared prefix) -> ordered concat."""
    started = time.perf_counter()
    cfg = runner.config
    with runner.step("skeleton"):
        req = runner.request(
            SKELETON,
            SkeletonOutput,
            effort=cfg.branch_effort,
            system_values={"min_points": cfg.skeleton_min_points, "max_points": cfg.skeleton_max_points},
            task=task,
            context=context or NO_CONTEXT,
        )
        skel, resp = await runner.structured(req, SkeletonOutput, step="skeleton")
    runner.record_single("skeleton", "skeleton", skel, resp)
    points = [p.strip() for p in skel.points if p.strip()][: cfg.skeleton_max_points]
    n = runner.budget_cap(len(points), reserve=1 if cfg.coherence_pass else 0)
    points = points[:n]
    skeleton_text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
    keyed = [
        (
            f"point-{i + 1}",
            runner.request(
                EXPAND,
                ExpansionOutput,
                effort=cfg.branch_effort,
                system_values={"max_words": cfg.expansion_max_words},
                task=task,
                context=context or NO_CONTEXT,
                skeleton=skeleton_text,
                index=i + 1,
                point=p,
            ),
        )
        for i, p in enumerate(points)
    ]
    with runner.step("expand"):
        outs, ok = await runner.fan_out(keyed, ExpansionOutput, stage="expand", policy=policy, concurrency=concurrency)
    if not ok:
        return _quorum_failure(runner, "skeleton_of_thought", started, outs, "concat")
    failed = _failed_ids(outs)
    parts = [(o.index, f"{points[o.index]}\n{o.value.text}") for o in outs if o.ok]
    answer = concat_in_order(parts)
    notice = missing_branch_notice(failed)
    if notice:
        answer += "\n\n" + notice
    aggregator = "concat"
    if cfg.coherence_pass and answer:
        with runner.step("coherence"):
            try:
                req = runner.request(COHERENCE, CoherenceOutput, effort=cfg.map_effort, task=task, draft=answer)
                coh, resp = await runner.structured(req, CoherenceOutput, step="coherence")
                runner.record_single("coherence", "coherence", coh, resp)
                answer = coh.text
                aggregator = "concat+coherence"
            except (BudgetExceededError, LLMError) as exc:
                runner.warn(f"coherence pass skipped ({type(exc).__name__})")
    structural = success_fraction([o.ok for o in outs]) if outs else 0.0
    conf = combine_confidence(structural, 0.7)
    return runner.result(
        "skeleton_of_thought",
        started=started,
        answer=answer,
        confidence=conf,
        confidence_basis=f"skeleton of {len(points)} points, {len(outs) - len(failed)}/{len(outs)} expanded; no verbalised confidence (prior 0.7)",
        aggregator=aggregator,
        failed_branches=failed,
        details={"points": points},
    )


async def run_voting(
    runner: PatternRunner,
    task: str,
    *,
    context: str = NO_CONTEXT,
    samples: int = 5,
    max_samples: int = 10,
    aggregator: str = "code",
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
    margin: int | None = None,
    leader_share: float | None = None,
) -> PatternResult:
    """Sampling-and-voting with early stopping; USC selection for free-form answers.

    Diversity comes from varied reasoning framings (no sampling knobs). The
    first batch has ``samples`` members; while the vote is not decisive
    (leader margin < ``margin`` and share < ``leader_share``) further batches
    are issued up to ``max_samples``.
    """
    started = time.perf_counter()
    cfg = runner.config
    margin = cfg.early_stop_margin if margin is None else margin
    leader_share = cfg.early_stop_leader_share if leader_share is None else leader_share
    usc = aggregator == "usc"
    reserve = 1 if usc else 0
    all_outs: list[TaskOutcome] = []
    rounds = 0
    issued = 0
    vote = majority_vote([])
    quorum_met = True
    while issued < max_samples:
        batch = min(samples if rounds == 0 else max(1, samples // 2), max_samples - issued)
        batch = runner.budget_cap(batch, reserve=reserve)
        if batch <= 0:
            break
        keyed = [
            (f"sample-{issued + i + 1}", runner.request(SAMPLE, SampleOutput, effort=cfg.branch_effort, task=task, context=context or NO_CONTEXT, framing=SAMPLE_FRAMINGS[(issued + i) % len(SAMPLE_FRAMINGS)]))
            for i in range(batch)
        ]
        with runner.step(f"sample-round-{rounds + 1}"):
            outs, ok = await runner.fan_out(keyed, SampleOutput, stage="sample", policy=policy, concurrency=concurrency)
        all_outs.extend(outs)
        issued += batch
        rounds += 1
        if not ok:
            quorum_met = False
            break
        answers = [o.value.final_answer for o in all_outs if o.ok]
        vote = majority_vote(answers, margin=margin, leader_share=leader_share)
        if usc:
            break  # free-form answers cannot be tallied; USC selects among the initial batch (5-8 candidates)
        if vote.decisive and len(answers) >= samples:
            break
    if not quorum_met or not any(o.ok for o in all_outs):
        return _quorum_failure(runner, "voting", started, all_outs, aggregator)
    failed = _failed_ids(all_outs)
    ok_outs = [o for o in all_outs if o.ok]
    details: dict[str, Any] = {"rounds": rounds, "issued": issued, "vote": vote.model_dump()}
    if usc:
        cands = ok_outs[: cfg.usc_max_candidates]
        rendered = "\n".join(xml("candidate", o.value.final_answer, index=i + 1) for i, o in enumerate(cands))
        with runner.step("usc_select"):
            try:
                req = runner.request(USC_SELECT, USCOutput, effort=cfg.aggregator_effort, task=task, candidates=rendered)
                sel, resp = await runner.structured(req, USCOutput, step="usc_select")
                runner.record_single("usc", "usc", sel, resp)
                idx = sel.selected_index - 1 if 0 <= sel.selected_index - 1 < len(cands) else 0
                answer = cands[idx].value.final_answer
                verbal = label_to_confidence(sel.confidence)
                structural = max(vote.share, success_fraction([o.ok for o in all_outs]) * 0.5)
                conf = combine_confidence(structural, verbal)
                basis = f"universal self-consistency selected candidate {idx + 1} of {len(cands)} (confidence '{sel.confidence}'); literal agreement {vote.share:.2f}"
                details["selected_index"] = idx + 1
                details["justification"] = sel.justification
            except (BudgetExceededError, LLMError) as exc:
                runner.warn(f"USC selection unavailable ({type(exc).__name__}); falling back to majority vote")
                usc = False
                aggregator = "code"
    if not usc:
        answer = vote.winner_raw
        conf = confidence_from_vote(vote)
        basis = f"majority vote: {vote.counts.get(vote.winner, 0)}/{vote.n} agree on '{vote.winner}' (margin {vote.margin}, entropy {vote.entropy:.2f}, early_stop={vote.decisive})"
    if failed:
        basis += f"; {len(failed)} sample(s) failed"
    return runner.result(
        "voting",
        started=started,
        answer=answer,
        confidence=conf,
        confidence_basis=basis,
        aggregator=aggregator,
        failed_branches=failed,
        details=details,
    )


def chunk_input(text: str, config: ParallelizationConfig) -> list[str]:
    """Split a long input into overlapping character windows sized from ``chunk_tokens``."""
    size = max(100, config.chunk_tokens * 4)
    overlap = int(size * config.chunk_overlap_ratio)
    return chunk_text(text, chunk_size=size, overlap=overlap, unit="chars")


def global_summary(text: str, *, max_chars: int = 600) -> str:
    """Cheap deterministic global context shared by every mapper (mitigates inter-chunk dependency loss)."""
    head = text.strip().replace("\r", "")
    if len(head) <= max_chars:
        return head
    return head[: max_chars - 3].rstrip() + "..."


def render_records(records: Sequence[tuple[str, MapRecord | ReduceOutput]]) -> str:
    """Render structured map/reduce records for the reducer prompt."""
    parts = []
    for key, rec in records:
        if isinstance(rec, MapRecord):
            body = f"answer: {rec.answer}\nconfidence: {rec.confidence}\nrationale: {rec.rationale}\nevidence: {'; '.join(rec.extracted_evidence) or 'none'}"
        else:
            body = f"answer: {rec.answer}\nconfidence: {rec.confidence}\nconflicts: {'; '.join(rec.conflicts) or 'none'}"
        parts.append(xml("record", body, id=key))
    return "\n".join(parts)


def is_irrelevant(record: MapRecord) -> bool:
    return record.answer.strip().lower().startswith("no relevant information")


async def run_map_reduce(
    runner: PatternRunner,
    task: str,
    text: str,
    *,
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
    fan_in: int | None = None,
) -> PatternResult:
    """LLM x MapReduce: structured map records over chunks, hierarchical confidence-aware reduce."""
    started = time.perf_counter()
    cfg = runner.config
    fan_in = max(2, fan_in or cfg.reduce_fan_in)
    chunks = chunk_input(text, cfg)
    if not chunks:
        return await run_sequential(runner, task, NO_CONTEXT)
    levels = 0
    level = len(chunks)
    while level > 1:
        level = math.ceil(level / fan_in)
        levels += 1
    n = runner.budget_cap(len(chunks), reserve=max(1, levels))
    chunks = chunks[:n]
    if not chunks:
        return runner.result("map_reduce", started=started, answer="", confidence=0.0, confidence_basis="no budget for map phase", aggregator="llm", quorum_met=False)
    summary = global_summary(text)
    keyed = [
        (f"chunk-{i + 1}", runner.request(MAP, MapRecord, effort=cfg.map_effort, task=task, summary=summary, index=i + 1, total=len(chunks), chunk=c))
        for i, c in enumerate(chunks)
    ]
    with runner.step("map"):
        outs, ok = await runner.fan_out(keyed, MapRecord, stage="map", policy=policy, concurrency=concurrency)
    if not ok or not any(o.ok for o in outs):
        return _quorum_failure(runner, "map_reduce", started, outs, "llm")
    failed = _failed_ids(outs)
    records: list[tuple[str, MapRecord | ReduceOutput]] = [(o.key, o.value) for o in outs if o.ok]
    relevant = [(k, r) for k, r in records if not (isinstance(r, MapRecord) and is_irrelevant(r))]
    dropped = len(records) - len(relevant)
    records = relevant or records[:1]
    level_no = 0
    final: ReduceOutput | None = None
    while True:
        level_no += 1
        groups = [records[i : i + fan_in] for i in range(0, len(records), fan_in)]
        keyed_r = [
            (f"reduce-{level_no}-{g + 1}", runner.request(REDUCE, ReduceOutput, effort=cfg.aggregator_effort if len(groups) == 1 else cfg.map_effort, task=task, records=render_records(group)))
            for g, group in enumerate(groups)
        ]
        with runner.step(f"reduce-level-{level_no}"):
            if len(keyed_r) == 1:
                try:
                    red, resp = await runner.structured(keyed_r[0][1], ReduceOutput, step=keyed_r[0][0])
                    runner.record_single(keyed_r[0][0], "reduce", red, resp)
                    final = red
                except (BudgetExceededError, LLMError) as exc:
                    runner.warn(f"final reduce unavailable ({type(exc).__name__}); concatenating map answers")
                break
            r_outs, r_ok = await runner.fan_out(keyed_r, ReduceOutput, stage="reduce", policy=FailurePolicy.best_effort(), concurrency=concurrency)
        next_records: list[tuple[str, MapRecord | ReduceOutput]] = [(o.key, o.value) for o in r_outs if o.ok]
        if not next_records:
            runner.warn("all intermediate reducers failed; concatenating map answers")
            break
        if len(next_records) < len(r_outs):
            failed.extend(_failed_ids(r_outs))
        records = next_records
    structural = success_fraction([o.ok for o in outs])
    if final is not None:
        answer = final.answer
        if final.conflicts:
            answer += "\n\nConflicts resolved: " + "; ".join(final.conflicts)
        verbal = label_to_confidence(final.confidence)
        basis = f"{len(outs) - len(_failed_ids(outs))}/{len(outs)} chunks mapped ({dropped} irrelevant dropped), {level_no} reduce level(s); reducer confidence '{final.confidence}'"
        aggregator = "llm"
    else:
        answer = concat_in_order([(i, f"[{k}] {r.answer}") for i, (k, r) in enumerate(records)])
        verbal = weighted_confidence([(label_to_confidence(r.confidence), 1.0) for _, r in records])
        basis = f"{len(outs) - len(_failed_ids(outs))}/{len(outs)} chunks mapped; reducer unavailable, deterministic concatenation"
        aggregator = "concat"
    notice = missing_branch_notice(_failed_ids(outs))
    if notice:
        answer += "\n\n" + notice
    return runner.result(
        "map_reduce",
        started=started,
        answer=answer,
        confidence=combine_confidence(structural, verbal),
        confidence_basis=basis,
        aggregator=aggregator,
        failed_branches=list(dict.fromkeys(failed)),
        details={"chunks": len(chunks), "dropped_irrelevant": dropped, "reduce_levels": level_no, "fan_in": fan_in},
    )


async def run_branch_solve_merge(
    runner: PatternRunner,
    task: str,
    *,
    context: str = NO_CONTEXT,
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
) -> PatternResult:
    """Branch-Solve-Merge: the model proposes criteria/constraints, each is solved concurrently, then merged."""
    started = time.perf_counter()
    cfg = runner.config
    with runner.step("branch"):
        req = runner.request(
            BRANCH,
            BranchOutput,
            effort=cfg.aggregator_effort,
            system_values={"min_branches": cfg.bsm_min_branches, "max_branches": cfg.bsm_max_branches},
            task=task,
            context=context or NO_CONTEXT,
        )
        br, resp = await runner.structured(req, BranchOutput, step="branch")
    runner.record_single("branch", "branch", br, resp)
    branches = [b.strip() for b in br.branches if b.strip()][: cfg.bsm_max_branches]
    n = runner.budget_cap(len(branches), reserve=1)
    branches = branches[:n]
    keyed = [
        (f"b{i + 1}", runner.request(SOLVE, BranchSolution, effort=cfg.branch_effort, task=task, context=context or NO_CONTEXT, branch_id=f"b{i + 1}", branch=b))
        for i, b in enumerate(branches)
    ]
    with runner.step("solve"):
        outs, ok = await runner.fan_out(keyed, BranchSolution, stage="solve", policy=policy, concurrency=concurrency)
    if not ok or not any(o.ok for o in outs):
        return _quorum_failure(runner, "branch_solve_merge", started, outs, "llm")
    failed = _failed_ids(outs)
    solutions = "\n".join(
        xml("solution", f"branch: {branches[o.index]}\nanswer: {o.value.answer}\nevidence: {'; '.join(o.value.evidence) or 'none'}\nrationale: {o.value.rationale}\nconfidence: {o.value.confidence}", id=o.key)
        for o in outs
        if o.ok
    )
    structural = success_fraction([o.ok for o in outs])
    aggregator = "llm"
    with runner.step("merge"):
        try:
            req = runner.request(MERGE, MergeOutput, effort=cfg.aggregator_effort, task=task, solutions=solutions, notice=missing_branch_notice(failed))
            merged, resp = await runner.structured(req, MergeOutput, step="merge")
            runner.record_single("merge", "merge", merged, resp)
            answer = merged.answer
            if merged.unresolved:
                answer += "\n\nUnresolved: " + "; ".join(merged.unresolved)
            verbal = label_to_confidence(merged.confidence)
            basis = f"{len(branches)} branches, {len(outs) - len(failed)} solved; merge confidence '{merged.confidence}'"
        except (BudgetExceededError, LLMError) as exc:
            runner.warn(f"merge call unavailable ({type(exc).__name__}); falling back to ordered concatenation")
            aggregator = "concat"
            answer = concat_in_order([(o.index, f"{branches[o.index]}\n{o.value.answer}") for o in outs if o.ok])
            verbal = _label_conf(outs)
            basis = f"{len(outs) - len(failed)}/{len(outs)} branches solved; deterministic concatenation"
    return runner.result(
        "branch_solve_merge",
        started=started,
        answer=answer,
        confidence=combine_confidence(structural, verbal),
        confidence_basis=basis,
        aggregator=aggregator,
        failed_branches=failed,
        details={"branches": branches},
    )


async def run_ensemble(
    runner: PatternRunner,
    task: str,
    *,
    context: str = NO_CONTEXT,
    proposers: int = 3,
    layers: int = 2,
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
) -> PatternResult:
    """Mixture-of-Agents (MoA-Lite): layers of homogeneous proposers, then a critical aggregator.

    Proposers in a layer run concurrently and see every proposal from the
    previous layer; perspectives diversify them without mixing models (Self-MoA).
    """
    started = time.perf_counter()
    cfg = runner.config
    previous = "(none - first layer)"
    last_outs: list[TaskOutcome] = []
    failed: list[str] = []
    total = 0
    for layer in range(1, layers + 1):
        n = runner.budget_cap(proposers, reserve=1)
        if n <= 0:
            runner.warn(f"ensemble: layer {layer} skipped (no budget)")
            break
        models = [] if cfg.homogeneous_ensemble else list(cfg.ensemble_proposer_models)
        keyed = [
            (
                f"L{layer}-p{i + 1}",
                runner.request(
                    PROPOSE,
                    ProposerOutput,
                    effort=cfg.branch_effort,
                    model=models[i % len(models)] if models else None,
                    task=task,
                    context=context or NO_CONTEXT,
                    previous=previous,
                    perspective=PROPOSER_PERSPECTIVES[i % len(PROPOSER_PERSPECTIVES)],
                ),
            )
            for i in range(n)
        ]
        with runner.step(f"propose-layer-{layer}"):
            outs, ok = await runner.fan_out(keyed, ProposerOutput, stage="propose", policy=policy, concurrency=concurrency)
        total += len(outs)
        failed.extend(_failed_ids(outs))
        if not ok:
            return _quorum_failure(runner, "ensemble", started, outs, "llm")
        good = [o for o in outs if o.ok]
        if not good:
            if last_outs:
                break
            return _quorum_failure(runner, "ensemble", started, outs, "llm")
        last_outs = good
        previous = "\n".join(xml("proposal", o.value.answer, id=o.key) for o in good)
    if not last_outs:
        return runner.result("ensemble", started=started, answer="", confidence=0.0, confidence_basis="no proposals produced", aggregator="llm", quorum_met=False)
    proposals = "\n".join(xml("proposal", o.value.answer, id=o.key) for o in last_outs)
    structural = (total - len(failed)) / total if total else 0.0
    aggregator = "llm"
    with runner.step("aggregate"):
        try:
            req = runner.request(AGGREGATE, AggregateOutput, effort=cfg.aggregator_effort, task=task, context=context or NO_CONTEXT, proposals=proposals)
            agg, resp = await runner.structured(req, AggregateOutput, step="aggregate")
            runner.record_single("aggregate", "aggregate", agg, resp)
            answer = agg.answer
            verbal = label_to_confidence(agg.confidence)
            basis = f"{layers}-layer ensemble, {total - len(failed)}/{total} proposals; aggregator confidence '{agg.confidence}', {len(agg.corrections)} correction(s)"
            details: dict[str, Any] = {"corrections": agg.corrections, "layers": layers, "proposers": proposers}
        except (BudgetExceededError, LLMError) as exc:
            runner.warn(f"aggregator unavailable ({type(exc).__name__}); returning the first final-layer proposal")
            aggregator = "code"
            answer = last_outs[0].value.answer
            verbal = 0.6
            basis = f"{total - len(failed)}/{total} proposals; aggregator unavailable, first final-layer proposal returned"
            details = {"layers": layers, "proposers": proposers}
    return runner.result(
        "ensemble",
        started=started,
        answer=answer,
        confidence=combine_confidence(structural, verbal),
        confidence_basis=basis,
        aggregator=aggregator,
        failed_branches=failed,
        details=details,
    )


async def speculative(
    fast: Callable[[], Awaitable[T]],
    slow: Callable[[], Awaitable[T]],
    verifier: Callable[[T], Awaitable[bool]],
) -> tuple[T, str]:
    """Race a cheap path against an expensive one.

    Both start immediately. When the fast result arrives it is checked by
    ``verifier``; if accepted the slow path is cancelled and ``(value,
    "fast")`` is returned, otherwise the slow result is awaited and returned
    as ``(value, "slow")``. The fast path failing is treated as a rejection.
    """
    slow_task = asyncio.ensure_future(slow())
    accepted = False
    fast_value: Any = None
    try:
        fast_value = await fast()
        accepted = bool(await verifier(fast_value))
    except asyncio.CancelledError:
        slow_task.cancel()
        raise
    except Exception:  # noqa: BLE001 - a failing fast path simply loses the race
        accepted = False
    if accepted:
        if not slow_task.done():
            slow_task.cancel()
        with contextlib.suppress(BaseException):
            await slow_task
        return fast_value, "fast"
    return await slow_task, "slow"


async def run_speculative(runner: PatternRunner, task: str, *, context: str = NO_CONTEXT) -> PatternResult:
    """Speculative execution: fast model answer verified cheaply, default model as the fallback."""
    started = time.perf_counter()
    cfg = runner.config
    verdict: dict[str, Any] = {}

    async def fast() -> AnswerOutput:
        req = runner.request(SEQUENTIAL, AnswerOutput, effort="low", model=cfg.fast_model, task=task, context=context or NO_CONTEXT, tags=["fast"])
        parsed, resp = await runner.structured(req, AnswerOutput, step="fast")
        runner.record_single("fast", "fast", parsed, resp)
        return parsed

    async def slow() -> AnswerOutput:
        req = runner.request(SEQUENTIAL, AnswerOutput, effort=cfg.aggregator_effort, task=task, context=context or NO_CONTEXT, tags=["slow"])
        parsed, resp = await runner.structured(req, AnswerOutput, step="slow")
        runner.record_single("slow", "slow", parsed, resp)
        return parsed

    async def verify(candidate: AnswerOutput) -> bool:
        req = runner.request(VERIFY_FAST, VerifyOutput, effort="low", task=task, answer=candidate.answer)
        parsed, resp = await runner.structured(req, VerifyOutput, step="verify")
        runner.record_single("verify", "verify", parsed, resp)
        verdict.update(parsed.model_dump())
        return parsed.acceptable

    with runner.step("speculate"):
        try:
            value, source = await speculative(fast, slow, verify)
        except (BudgetExceededError, LLMError, StructuredOutputError) as exc:
            return runner.result("speculative", started=started, answer="", confidence=0.0, confidence_basis=f"both paths failed: {exc}", aggregator="code", quorum_met=False, failed_branches=["fast", "slow"])
    conf = label_to_confidence(value.confidence)
    if source == "fast":
        basis = f"fast-model answer accepted by verifier; verbalised confidence '{value.confidence}'"
    else:
        basis = f"fast answer rejected ({'; '.join(verdict.get('issues', [])) or 'verifier declined or fast path failed'}); default-model answer with confidence '{value.confidence}'"
    return runner.result(
        "speculative",
        started=started,
        answer=value.answer,
        confidence=conf,
        confidence_basis=basis,
        aggregator="code",
        details={"source": source, "verdict": verdict},
    )


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


async def cache_aware_fan_out(
    runner: PatternRunner,
    requests: Sequence[LLMRequest],
    *,
    schema: type[BaseModel] | None = None,
    policy: FailurePolicy | None = None,
    concurrency: int | None = None,
    keys: Sequence[str] | None = None,
) -> list[TaskOutcome]:
    """Send ``requests`` concurrently with warm-then-burst on shared cacheable prefixes.

    The runner's :class:`PrefixWarmGate` holds siblings until the first
    request on each cacheable prefix has returned, then releases the burst;
    the hold happens at executor admission, before rate limiting and the
    per-attempt timeout.
    Outcome values are :class:`LLMResponse` objects (or validated ``schema``
    instances when a schema is given).
    """
    keys = list(keys) if keys is not None else [f"req-{i + 1}" for i in range(len(requests))]

    async def one(item: tuple[str, LLMRequest]) -> Any:
        key, req = item
        if schema is not None:
            parsed, _ = await runner.structured(req, schema, step=f"fan-out:{key}", gated=False)
            return parsed
        return await runner.call(req, step=f"fan-out:{key}", gated=False)

    try:
        return await runner.executor.map_outcomes(one, list(zip(keys, requests)), keys=keys, concurrency=concurrency, failure_policy=policy or FailurePolicy.best_effort(), resource="llm", **runner.exec_kwargs(admit=runner.admit))
    except ParallelExecutionError as exc:
        runner.warn(str(exc))
        return list(exc.outcomes)


class ToolCall(BaseModel):
    """One tool call emitted by a model turn."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of one tool call, returned to the model in the same turn."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    ok: bool
    result: Any = None
    error: Optional[str] = None
    latency_s: float = 0.0
    attempts: int = 1


async def parallel_tool_calls(
    executor: ParallelExecutor,
    calls: Sequence[ToolCall],
    tools: Mapping[str, Callable[..., Any]],
    *,
    ordered_tools: Sequence[str] = (),
    concurrency: int | None = None,
    idempotent: bool = True,
) -> list[ToolResult]:
    """Execute a model turn's tool calls concurrently, keyed by ``call_id``.

    Tools named in ``ordered_tools`` (side effects with ordering) run
    sequentially, in emission order, after the concurrent batch, under their
    own bulkhead resource, and are attempted exactly once: no retry, no hedge
    and no memoisation (a lost acknowledgement must not re-send a write, and a
    ``call_id`` reused by a later turn must not return an earlier result).
    Unknown tool names fail their call (never retried, never counted against
    the circuit breaker). Concurrent tools are retried on transient errors
    only when ``idempotent`` is True; pass ``idempotent=False`` when any
    non-ordered tool has side effects.
    Results are returned in the original order so they can be handed back to
    the model in one turn.
    """
    ordered = set(ordered_tools)

    async def run(call: ToolCall) -> Any:
        fn = tools.get(call.name)
        if fn is None:
            raise KeyError(f"unknown tool '{call.name}'")
        out = fn(**call.args)
        if asyncio.iscoroutine(out) or isinstance(out, asyncio.Future):
            out = await out
        return out

    concurrent = [c for c in calls if c.name not in ordered]
    sequential = [c for c in calls if c.name in ordered]
    outcomes: dict[str, TaskOutcome] = {}
    if concurrent:
        try:
            outs = await executor.map_outcomes(run, concurrent, keys=[c.call_id for c in concurrent], concurrency=concurrency, failure_policy=FailurePolicy.best_effort(), resource="tools", idempotent=idempotent)
        except ParallelExecutionError as exc:  # pragma: no cover - best_effort never raises
            outs = list(exc.outcomes)
        for o in outs:
            outcomes[o.key] = o
    for c in sequential:
        outcomes[c.call_id] = await executor.submit(run, c, key=c.call_id, resource=f"ordered:{c.name}", idempotent=False)
    results: list[ToolResult] = []
    for c in calls:
        o = outcomes[c.call_id]
        results.append(ToolResult(call_id=c.call_id, name=c.name, ok=o.ok, result=o.value if o.ok else None, error=o.error, latency_s=o.latency_s, attempts=o.attempts))
    return results


async def generic_map_reduce(
    runner: PatternRunner,
    items: Sequence[str],
    map_prompt: str,
    reduce_prompt: str,
    *,
    fanin: int | None = None,
    system: Sequence[str] = (),
    concurrency: int | None = None,
) -> PatternResult:
    """Prompt-driven map-reduce over arbitrary items (direct API).

    ``map_prompt`` must contain ``{item}``; ``reduce_prompt`` must contain
    ``{records}``. Both are user-turn text; ``system`` blocks are shared by
    every call (cache-friendly). Reduction is hierarchical with ``fanin``
    records per reducer and the final answer is the last reducer's text.
    """
    started = time.perf_counter()
    cfg = runner.config
    fanin = max(2, fanin or cfg.reduce_fan_in)
    stable = [*system, runner.profile.system_text()]
    keyed = [(f"item-{i + 1}", LLMRequest(system_stable=stable, messages=[Message(role="user", content=map_prompt.replace("{item}", str(it)))], effort=cfg.map_effort, tags=["map"])) for i, it in enumerate(items)]

    async def one(item: tuple[str, LLMRequest]) -> str:
        key, req = item
        return (await runner.call(req, step=f"map:{key}", gated=False)).text

    try:
        outs = await runner.executor.map_outcomes(one, keyed, keys=[k for k, _ in keyed], concurrency=concurrency, failure_policy=FailurePolicy.best_effort(), resource="llm", **runner.exec_kwargs(admit=runner.admit))
    except ParallelExecutionError as exc:  # pragma: no cover
        outs = list(exc.outcomes)
    runner._record(outs, "map")
    records = [(o.key, o.value) for o in outs if o.ok]
    failed = _failed_ids(outs)
    if not records:
        return runner.result("map_reduce", started=started, answer="", confidence=0.0, confidence_basis="every map call failed", aggregator="llm", failed_branches=failed, quorum_met=False)
    level = 0
    while True:
        level += 1
        groups = [records[i : i + fanin] for i in range(0, len(records), fanin)]
        reqs = [(f"reduce-{level}-{g + 1}", LLMRequest(system_stable=stable, messages=[Message(role="user", content=reduce_prompt.replace("{records}", "\n".join(xml("record", r, id=k) for k, r in group)))], effort=cfg.aggregator_effort, tags=["reduce"])) for g, group in enumerate(groups)]
        if len(reqs) == 1:
            resp = await runner.call(reqs[0][1], step=reqs[0][0])
            runner.records.append(BranchRecord(id=reqs[0][0], stage="reduce", text=resp.text, latency_s=resp.latency_s))
            answer = resp.text
            break
        r_outs = await runner.executor.map_outcomes(one, reqs, keys=[k for k, _ in reqs], concurrency=concurrency, failure_policy=FailurePolicy.best_effort(), resource="llm", **runner.exec_kwargs(admit=runner.admit))
        runner._record(r_outs, "reduce")
        records = [(o.key, o.value) for o in r_outs if o.ok] or records[:1]
    structural = success_fraction([o.ok for o in outs])
    return runner.result(
        "map_reduce",
        started=started,
        answer=answer,
        confidence=combine_confidence(structural, 0.7),
        confidence_basis=f"{len(outs) - len(failed)}/{len(outs)} items mapped, {level} reduce level(s); free-text reduce (prior 0.7)",
        aggregator="llm",
        failed_branches=failed,
        details={"fan_in": fanin, "reduce_levels": level},
    )


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


async def run_pattern(runner: PatternRunner, plan: ParallelPlan, task: str, *, context: str = NO_CONTEXT, text: str = "") -> PatternResult:
    """Execute ``plan`` for ``task``. ``text`` is the long input used by map-reduce."""
    policy = _policy(plan)
    width = plan.concurrency
    if plan.pattern == "sequential":
        return await run_sequential(runner, task, context)
    if plan.pattern == "sectioning":
        return await run_sectioning(runner, task, plan.partitions, context=context, aggregator=plan.aggregator, policy=policy, concurrency=width)
    if plan.pattern == "skeleton_of_thought":
        return await run_skeleton(runner, task, context=context, policy=policy, concurrency=width)
    if plan.pattern == "voting":
        return await run_voting(runner, task, context=context, samples=plan.samples, max_samples=plan.max_samples, aggregator=plan.aggregator, policy=policy, concurrency=width)
    if plan.pattern == "map_reduce":
        return await run_map_reduce(runner, task, text or context, policy=policy, concurrency=width)
    if plan.pattern == "branch_solve_merge":
        return await run_branch_solve_merge(runner, task, context=context, policy=policy, concurrency=width)
    if plan.pattern == "ensemble":
        return await run_ensemble(runner, task, context=context, proposers=plan.samples, layers=plan.layers, policy=policy, concurrency=width)
    if plan.pattern == "speculative":
        return await run_speculative(runner, task, context=context)
    raise ValueError(f"unknown pattern {plan.pattern!r}")


# --------------------------------------------------------------------------
# ctx-first convenience API (spec signatures)
# --------------------------------------------------------------------------


def _runner_for(ctx: RunContext, config: ParallelizationConfig | None, executor: ParallelExecutor | None) -> PatternRunner:
    cfg = config or ParallelizationConfig()
    return PatternRunner.from_ctx(ctx, cfg, executor or ParallelExecutor.from_config(cfg), cache_aware=cfg.cache_aware_fan_out)


async def skeleton_of_thought(ctx: RunContext, prompt: str, *, context: str = NO_CONTEXT, config: ParallelizationConfig | None = None, executor: ParallelExecutor | None = None) -> PatternResult:
    """Skeleton -> parallel expansion -> merge, inside a component run."""
    return await run_skeleton(_runner_for(ctx, config, executor), prompt, context=context)


async def branch_solve_merge(ctx: RunContext, prompt: str, *, context: str = NO_CONTEXT, config: ParallelizationConfig | None = None, executor: ParallelExecutor | None = None) -> PatternResult:
    """Branch-Solve-Merge inside a component run."""
    return await run_branch_solve_merge(_runner_for(ctx, config, executor), prompt, context=context)


async def map_reduce(ctx: RunContext, items: Sequence[str], map_prompt: str, reduce_prompt: str, *, fanin: int | None = None, config: ParallelizationConfig | None = None, executor: ParallelExecutor | None = None) -> PatternResult:
    """Prompt-driven map-reduce over ``items`` inside a component run."""
    return await generic_map_reduce(_runner_for(ctx, config, executor), items, map_prompt, reduce_prompt, fanin=fanin)


async def ensemble(ctx: RunContext, prompt: str, *, proposers: int | None = None, layers: int | None = None, context: str = NO_CONTEXT, config: ParallelizationConfig | None = None, executor: ParallelExecutor | None = None) -> PatternResult:
    """Mixture-of-Agents ensemble inside a component run."""
    cfg = config or ParallelizationConfig()
    return await run_ensemble(_runner_for(ctx, cfg, executor), prompt, context=context, proposers=proposers or cfg.ensemble_proposers, layers=layers or cfg.ensemble_layers)


mixture_of_agents = ensemble


async def voting(ctx: RunContext, prompt: str, *, samples: int | None = None, max_samples: int | None = None, free_form: bool = False, context: str = NO_CONTEXT, config: ParallelizationConfig | None = None, executor: ParallelExecutor | None = None) -> PatternResult:
    """Sampling-and-voting (or USC when ``free_form``) inside a component run."""
    cfg = config or ParallelizationConfig()
    s = samples or cfg.voting_initial(ctx.profile.risk_level)
    return await run_voting(_runner_for(ctx, cfg, executor), prompt, context=context, samples=s, max_samples=max(s, max_samples or cfg.voting_max_samples), aggregator="usc" if free_form else "code")


__all__ = [
    "BranchRecord",
    "NO_CONTEXT",
    "PatternResult",
    "PatternRunner",
    "PrefixWarmGate",
    "ToolCall",
    "ToolResult",
    "branch_solve_merge",
    "cache_aware_fan_out",
    "chunk_input",
    "ensemble",
    "generic_map_reduce",
    "global_summary",
    "map_reduce",
    "mixture_of_agents",
    "parallel_tool_calls",
    "render_records",
    "run_branch_solve_merge",
    "run_ensemble",
    "run_map_reduce",
    "run_pattern",
    "run_sectioning",
    "run_sequential",
    "run_skeleton",
    "run_speculative",
    "run_voting",
    "skeleton_of_thought",
    "speculative",
    "voting",
]
