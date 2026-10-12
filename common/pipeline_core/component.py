"""Base class every model implements.

A component's life cycle is **plan -> execute**:

``plan(request, ctx)``
    Read the natural-language task and decide *how* to carry it out: which
    strategy, how many samples, which retrievers, what parallelism. This is
    the "reasoning" step; it normally costs one structured LLM call, and must
    degrade to heuristics when the LLM is unavailable (``ctx.offline``).

``execute(request, plan, ctx)``
    Carry the plan out and return a :class:`TaskResult`.

``run(request)`` wraps both with tracing, budget enforcement, timing and
error containment, so callers always receive a result (status FAILED on
error) unless ``raise_on_error=True``.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, ClassVar, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import BudgetTracker, Status, StepRecord, TaskRequest, TaskResult, Usage
from .domain import DomainProfile, get_profile
from .errors import BudgetExceededError, PipelineError
from .llm import BaseLLMClient, LLMRequest, LLMResponse, is_offline_client
from .telemetry import Tracer

log = logging.getLogger(__name__)


class ComponentSpec(BaseModel):
    """Static description used by registries, routers and documentation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0.0"
    description: str
    capabilities: list[str] = Field(default_factory=list)
    consumes: list[str] = Field(default_factory=list, description="What the component expects in request.inputs/context")
    produces: list[str] = Field(default_factory=list, description="Shape of result.output")
    composes_with: list[str] = Field(default_factory=list, description="Names of components it is designed to chain with")


class RunContext:
    """Per-run state shared by plan and execute. Not reusable across runs."""

    def __init__(self, request: TaskRequest, tracer: Tracer, llm: BaseLLMClient, component: str) -> None:
        self.request = request
        self.tracer = tracer
        self.llm = llm
        self.component = component
        self.budget = BudgetTracker(constraints=request.constraints)
        self.started = time.perf_counter()
        self.usage = Usage()
        self.steps: list[StepRecord] = []
        self.warnings: list[str] = []
        self.profile: DomainProfile = get_profile(request.domain)
        self.offline: bool = is_offline_client(llm)
        self.scratch: dict[str, Any] = {}

    # ----------------------------------------------------------------- time
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def remaining_time(self) -> Optional[float]:
        lim = self.request.constraints.max_latency_s
        return None if lim is None else max(0.0, lim - self.elapsed())

    def deadline_exceeded(self) -> bool:
        rem = self.remaining_time()
        return rem is not None and rem <= 0.0

    # ------------------------------------------------------------------ llm
    async def llm_call(self, request: LLMRequest, *, step: str | None = None) -> LLMResponse:
        """Invoke the LLM with budget enforcement and tracing."""
        if self.deadline_exceeded():
            raise BudgetExceededError(f"latency budget {self.request.constraints.max_latency_s}s exhausted")
        remaining = self.budget.remaining_calls()
        if remaining is not None and remaining <= 0:
            raise BudgetExceededError(f"LLM call budget of {self.request.constraints.max_llm_calls} exhausted")
        # Propagate request-level policy so client middleware (e.g. the caching model's
        # CachedLLMClient) can apply per-domain and deterministic rules. Explicit values win.
        request.metadata.setdefault("domain", self.request.domain)
        if self.request.constraints.deterministic:
            request.metadata.setdefault("deterministic", True)
        with self.tracer.span(step or "llm", self.component, **{"gen_ai.operation.name": "chat", "tags": request.tags}) as sp:
            resp = await self.llm.complete(request)
            sp.attributes["gen_ai.request.model"] = resp.model
            sp.attributes["stop_reason"] = resp.stop_reason
            sp.attributes["cached"] = resp.cached
            self.tracer.record_usage(resp.usage, model=resp.model, span=sp)
        self.usage = self.usage + resp.usage
        self.budget.charge(resp.usage)
        return resp

    def charge(self, usage: Usage) -> None:
        """Account for work done by a nested component."""
        self.usage = self.usage + usage
        self.budget.charge(usage)

    # ---------------------------------------------------------------- steps
    @contextmanager
    def step(self, name: str, **details: Any) -> Iterator[StepRecord]:
        rec = StepRecord(name=name, component=self.component, details=dict(details))
        before = self.usage
        t0 = time.perf_counter()
        with self.tracer.span(name, self.component):
            try:
                yield rec
            except Exception as exc:
                rec.status = Status.FAILED
                rec.summary = rec.summary or f"{type(exc).__name__}: {exc}"
                raise
            finally:
                rec.duration_s = time.perf_counter() - t0
                rec.usage = _usage_delta(self.usage, before)
                self.steps.append(rec)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.tracer.event("warning", message=message, component=self.component)
        log.warning("[%s] %s", self.component, message)


def _usage_delta(after: Usage, before: Usage) -> Usage:
    return Usage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        cache_read_input_tokens=after.cache_read_input_tokens - before.cache_read_input_tokens,
        cache_creation_input_tokens=after.cache_creation_input_tokens - before.cache_creation_input_tokens,
        llm_calls=after.llm_calls - before.llm_calls,
        cache_hits=after.cache_hits - before.cache_hits,
        cost_usd=after.cost_usd - before.cost_usd,
        latency_s=after.latency_s - before.latency_s,
    )


class PipelineComponent(ABC):
    """Abstract base for the six models (and any future one)."""

    spec: ClassVar[ComponentSpec]

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        tracer: Tracer | None = None,
        raise_on_error: bool = False,
        **config: Any,
    ) -> None:
        if llm is None:
            from .config import make_llm

            llm = make_llm()
        self.llm = llm
        self.tracer = tracer or Tracer(name=self.spec.name)
        self.raise_on_error = raise_on_error
        self.config: dict[str, Any] = config

    # ------------------------------------------------------------- abstract
    @abstractmethod
    async def plan(self, request: TaskRequest, ctx: RunContext) -> Any:
        """Decide how to execute ``request``. Must work offline (heuristics) too."""

    @abstractmethod
    async def execute(self, request: TaskRequest, plan: Any, ctx: RunContext) -> TaskResult:
        """Carry out ``plan``."""

    # --------------------------------------------------------------- public
    async def run(self, request: TaskRequest | str, **request_kwargs: Any) -> TaskResult:
        if isinstance(request, str):
            request = TaskRequest(task=request, **request_kwargs)
        ctx = RunContext(request, self.tracer, self.llm, self.spec.name)
        with self.tracer.span(f"{self.spec.name}.run", self.spec.name, request_id=request.request_id, domain=request.domain) as span:
            try:
                with ctx.step("plan"):
                    plan = await self.plan(request, ctx)
                result = await self.execute(request, plan, ctx)
                if plan is not None and "plan" not in result.metadata:
                    result.metadata["plan"] = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else plan
            except BudgetExceededError as exc:
                result = self.failed_result(request, exc, status=Status.PARTIAL if ctx.steps else Status.FAILED)
            except PipelineError as exc:
                if self.raise_on_error:
                    raise
                result = self.failed_result(request, exc)
            except Exception as exc:  # noqa: BLE001 - containment is the contract
                if self.raise_on_error:
                    raise
                log.exception("[%s] unexpected error", self.spec.name)
                result = self.failed_result(request, exc)
            # merge bookkeeping
            known = {id(s) for s in result.steps}
            for s in ctx.steps:
                if id(s) not in known:
                    result.steps.append(s)
            result.usage = _max_usage(result.usage, ctx.usage)
            result.usage.latency_s = ctx.elapsed()
            result.warnings = list(dict.fromkeys(result.warnings + ctx.warnings))
            result.trace_id = self.tracer.trace_id
            result.component = result.component or self.spec.name
            span.attributes["status"] = result.status.value
            span.attributes["confidence"] = result.confidence
            # nested llm spans already recorded usage; only record cache-hit/cost adjustments made outside llm_call
            extra = _usage_delta(result.usage, ctx.usage)
            if extra.llm_calls or extra.cost_usd or extra.cache_hits:
                self.tracer.record_usage(extra, span=span)
        return result

    def run_sync(self, request: TaskRequest | str, **request_kwargs: Any) -> TaskResult:
        from .utils import run_sync

        return run_sync(self.run(request, **request_kwargs))

    async def __call__(self, request: TaskRequest | str, **request_kwargs: Any) -> TaskResult:
        return await self.run(request, **request_kwargs)

    # -------------------------------------------------------------- helpers
    def failed_result(self, request: TaskRequest, error: BaseException, *, status: Status = Status.FAILED) -> TaskResult:
        return TaskResult(
            request_id=request.request_id,
            component=self.spec.name,
            status=status,
            output=None,
            output_text="",
            confidence=0.0,
            confidence_basis="run failed",
            errors=[f"{type(error).__name__}: {error}"],
        )

    def new_result(self, request: TaskRequest, **kw: Any) -> TaskResult:
        return TaskResult(request_id=request.request_id, component=self.spec.name, **kw)

    def describe(self) -> dict[str, Any]:
        return {**self.spec.model_dump(), "llm": self.llm.describe(), "config": self.config}


def _max_usage(a: Usage, b: Usage) -> Usage:
    """Components may have already copied ctx usage into the result; avoid double counting."""
    chosen = a if (a.llm_calls >= b.llm_calls and a.cost_usd >= b.cost_usd) else b
    if chosen.cache_hits < max(a.cache_hits, b.cache_hits):
        chosen = chosen.model_copy(update={"cache_hits": max(a.cache_hits, b.cache_hits)})
    return chosen
