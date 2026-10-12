"""Wave scheduler: runs a :class:`TaskGraph` with bounded concurrency, retries and budgets.

The scheduler is pure code (no LLM), as the dossier recommends
(LLMCompiler's task-fetching unit, ReWOO's worker): it computes topological
waves, dispatches every ready node through an :class:`Executor`, substitutes
``$id`` placeholders with finished outputs, retries failed nodes with
jittered exponential backoff, skips nodes whose dependencies failed, and stops
dispatching when the budget or deadline callback says so.

Clock, sleep and random source are injectable so retry timing is unit-tested
without waiting.
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Any, Awaitable, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import BudgetExceededError, Evidence, Executor, LLMRefusalError, PlanValidationError, Status, StepRecord, Usage, gather_limited

from .graph import SubTask, TaskGraph, substitute_placeholders

SolveFn = Callable[[SubTask, dict[str, "SubTaskResult"]], Awaitable["SubTaskResult"]]

_NOT_STARTED_RE = re.compile(r"^not started:\s*(\w+)")


def not_started_reason(result: "SubTaskResult") -> Optional[str]:
    """``"budget"``/``"deadline"`` when ``result`` was skipped because dispatch stopped, else ``None``."""
    if result.status != Status.SKIPPED:
        return None
    m = _NOT_STARTED_RE.match(result.error or "")
    return m.group(1) if m else None


class SubTaskResult(BaseModel):
    """Outcome of one sub-task (possibly after retries or recursive expansion)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Status = Status.SUCCESS
    output_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    criteria_met: list[bool] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list, description="Evidence gathered for this node (retrieved or provided) so the run can surface it")
    gaps: list[str] = Field(default_factory=list)
    attempts: int = 0
    error: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    duration_s: float = 0.0
    depth: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (Status.SUCCESS, Status.PARTIAL, Status.CACHED)


class RetryPolicy(BaseModel):
    """Exponential backoff with full jitter (Google SRE guidance: never indefinite, cap attempts)."""

    model_config = ConfigDict(extra="forbid")

    base_delay_s: float = Field(default=0.5, ge=0.0)
    factor: float = Field(default=2.0, ge=1.0)
    max_delay_s: float = Field(default=8.0, ge=0.0)
    jitter: bool = True

    def delay(self, attempt: int, rng: Optional[random.Random] = None) -> float:
        """Delay before retry number ``attempt`` (1-based)."""
        cap = min(self.max_delay_s, self.base_delay_s * (self.factor ** max(0, attempt - 1)))
        if not self.jitter:
            return cap
        r = rng or random
        return r.uniform(0.0, cap)

    @staticmethod
    def is_retryable(exc: BaseException) -> bool:
        """Budget exhaustion, refusals and plan defects are not retried; transient failures are."""
        return not isinstance(exc, (BudgetExceededError, LLMRefusalError, PlanValidationError, asyncio.CancelledError))


class BoundedExecutor:
    """Default :class:`Executor`: ``asyncio.gather`` with a concurrency ceiling."""

    def __init__(self, concurrency: int = 8) -> None:
        self.concurrency = max(1, concurrency)

    async def map(self, fn: Callable[[Any], Awaitable[Any]], items: Sequence[Any], *, concurrency: Optional[int] = None) -> list[Any]:
        limit = concurrency or self.concurrency
        return await gather_limited((fn(it) for it in items), limit, return_exceptions=False)


class ScheduleReport(BaseModel):
    """What happened during one scheduling pass."""

    model_config = ConfigDict(extra="forbid")

    results: dict[str, SubTaskResult] = Field(default_factory=dict)
    waves: list[list[str]] = Field(default_factory=list)
    waves_executed: int = 0
    retries: int = 0
    stopped_reason: Optional[str] = Field(default=None, description="budget | deadline | None when the graph ran to completion")
    max_concurrency_observed: int = 0
    steps: list[StepRecord] = Field(default_factory=list, description="One StepRecord per sub-task (name subtask.<id>) for the parent TaskResult")

    @property
    def failed_ids(self) -> list[str]:
        return [tid for tid, r in self.results.items() if r.status == Status.FAILED]

    @property
    def skipped_ids(self) -> list[str]:
        return [tid for tid, r in self.results.items() if r.status == Status.SKIPPED]

    @property
    def completed_ids(self) -> list[str]:
        return [tid for tid, r in self.results.items() if r.ok]

    def all_ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results.values())


class Scheduler:
    """Executes a graph wave by wave through an :class:`Executor`."""

    def __init__(
        self,
        *,
        executor: Optional[Executor] = None,
        retry: Optional[RetryPolicy] = None,
        parallelism: int = 8,
        max_output_chars: int = 4000,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Optional[random.Random] = None,
        component: str = "task_decomposition",
    ) -> None:
        self.executor: Executor = executor or BoundedExecutor(parallelism)
        self.retry = retry or RetryPolicy()
        self.parallelism = max(1, parallelism)
        self.max_output_chars = max_output_chars
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random(0)
        self.component = component

    # ------------------------------------------------------------------ run
    async def run(
        self,
        graph: TaskGraph,
        solve: SolveFn,
        *,
        completed: Optional[dict[str, SubTaskResult]] = None,
        should_stop: Optional[Callable[[], Optional[str]]] = None,
        remaining_calls: Optional[Callable[[], Optional[int]]] = None,
        on_result: Optional[Callable[[SubTaskResult], None]] = None,
    ) -> ScheduleReport:
        """Run every node of ``graph`` not already in ``completed``.

        ``should_stop()`` is consulted before each dispatch; a non-empty string
        (``"budget"``, ``"deadline"``) stops dispatch and marks the remaining
        nodes ``SKIPPED`` with that reason (``not started: <reason>``); nodes
        downstream of one that was never started inherit ``not started:
        <reason> (upstream ...)`` rather than ``dependency failed``, so a pure
        budget stop is never reported as a dependency failure. ``remaining_calls()`` (the LLM call
        budget left, ``None`` for unlimited) bounds how many nodes of a wave
        are dispatched at once: one call is reserved per node, so a fan-out
        never starts more work than the budget can pay for (architecture rule
        3.2: check remaining calls before fan-outs, degrade rather than fail).
        """
        waves = graph.topological_waves()
        report = ScheduleReport(waves=waves)
        results: dict[str, SubTaskResult] = dict(completed or {})
        outputs: dict[str, str] = {tid: self._truncate(r.output_text) for tid, r in results.items() if r.ok}
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def _one(task: SubTask) -> SubTaskResult:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                prior = {d: results[d] for d in task.depends_on if d in results}
                resolved = task.model_copy(
                    update={
                        "inputs": substitute_placeholders(task.inputs, outputs),
                        "description": substitute_placeholders(task.description, outputs),
                    }
                )
                return await self._solve_with_retries(resolved, prior, solve, report, stop_reason=should_stop)
            finally:
                async with lock:
                    in_flight -= 1

        def _stop_reason() -> Optional[str]:
            reason = should_stop() if should_stop else None
            if reason:
                return reason
            rem = remaining_calls() if remaining_calls else None
            return "budget" if rem is not None and rem <= 0 else None

        for wave in waves:
            pending = [tid for tid in wave if tid not in results]
            if not pending:
                report.waves_executed += 1
                continue
            runnable: list[SubTask] = []
            for tid in pending:
                task = graph.get(tid)
                bad = [d for d in task.depends_on if d in results and not results[d].ok]
                if bad:
                    upstream_stops = [not_started_reason(results[d]) for d in bad]
                    if all(upstream_stops):
                        # nothing upstream failed; dispatch stopped (budget/deadline) before the dependency could start
                        error = f"not started: {upstream_stops[0]} (upstream {', '.join(bad)})"
                    else:
                        error = f"dependency failed: {', '.join(d for d, r in zip(bad, upstream_stops) if r is None)}"
                    results[tid] = SubTaskResult(task_id=tid, status=Status.SKIPPED, error=error)
                else:
                    runnable.append(task)

            while runnable:
                reason = _stop_reason()
                if reason:
                    for task in runnable:
                        results[task.id] = SubTaskResult(task_id=task.id, status=Status.SKIPPED, error=f"not started: {reason}")
                    report.stopped_reason = reason
                    break
                batch = runnable
                rem = remaining_calls() if remaining_calls else None
                if rem is not None and rem < len(runnable):
                    batch = runnable[:rem]  # reserve one call per node; re-check the budget before the rest
                runnable = runnable[len(batch):]
                batch_results = await self.executor.map(_one, batch, concurrency=self.parallelism)
                for res in batch_results:
                    results[res.task_id] = res
                    if res.ok:
                        outputs[res.task_id] = self._truncate(res.output_text)
                    if on_result:
                        on_result(res)
            report.waves_executed += 1

        report.results = results
        report.max_concurrency_observed = peak
        report.steps = [self.step_record(r) for r in results.values()]
        return report

    def step_record(self, res: SubTaskResult) -> StepRecord:
        """Render one sub-task outcome as a :class:`StepRecord` for the parent ``TaskResult``."""
        return StepRecord(
            name=f"subtask.{res.task_id}",
            component=self.component,
            status=res.status,
            duration_s=res.duration_s,
            summary=res.error or res.output_text[:120],
            usage=res.usage,
            details={"attempts": res.attempts, "confidence": res.confidence, "depth": res.depth},
        )

    # -------------------------------------------------------------- helpers
    async def _solve_with_retries(
        self,
        task: SubTask,
        prior: dict[str, SubTaskResult],
        solve: SolveFn,
        report: ScheduleReport,
        *,
        stop_reason: Optional[Callable[[], Optional[str]]] = None,
    ) -> SubTaskResult:
        attempts_allowed = task.max_retries + 1
        last_error = ""
        t0 = self.clock()
        for attempt in range(1, attempts_allowed + 1):
            try:
                res = await solve(task, prior)
                res.attempts = attempt
                res.duration_s = self.clock() - t0
                if res.status == Status.FAILED and attempt < attempts_allowed:
                    last_error = res.error or "solver reported failure"
                    report.retries += 1
                    await self.sleep(self.retry.delay(attempt, self.rng))
                    continue
                return res
            except BudgetExceededError as exc:
                # the callback knows whether the deadline or the call/cost budget tripped; "budget" is the fallback
                report.stopped_reason = report.stopped_reason or (stop_reason() if stop_reason else None) or "budget"
                return SubTaskResult(task_id=task.id, status=Status.FAILED, error=f"budget: {exc}", attempts=attempt, duration_s=self.clock() - t0)
            except Exception as exc:  # noqa: BLE001 - failures become results, never exceptions
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts_allowed and self.retry.is_retryable(exc):
                    report.retries += 1
                    await self.sleep(self.retry.delay(attempt, self.rng))
                    continue
                return SubTaskResult(task_id=task.id, status=Status.FAILED, error=last_error, attempts=attempt, duration_s=self.clock() - t0)
        return SubTaskResult(task_id=task.id, status=Status.FAILED, error=last_error or "exhausted retries", attempts=attempts_allowed, duration_s=self.clock() - t0)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars - 3] + "..."
