"""Data contracts exchanged between pipeline components.

Every component consumes a :class:`TaskRequest` and produces a
:class:`TaskResult`. Keeping these two types stable is what lets the six
models be chained, nested or run side by side without adapters.

Design rules
------------
* Everything is a pydantic model so it validates at the boundary, serialises
  to JSON for tracing and caching, and documents itself.
* ``Usage`` is additive (``a + b``) so orchestrators can roll up cost across
  nested components without bespoke bookkeeping.
* ``Evidence`` is the single currency for "something retrieved or cited";
  retrieval produces it, RAG consumes it, verification checks against it.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .utils import now_iso


class Status(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"          # some sub-steps failed but output is usable
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"  # verification could not reach required confidence
    CACHED = "cached"            # served from a cache without recomputation
    SKIPPED = "skipped"


class Constraints(BaseModel):
    """Operational limits a request must respect. All optional."""

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: Optional[float] = Field(default=None, ge=0)
    max_latency_s: Optional[float] = Field(default=None, gt=0)
    max_llm_calls: Optional[int] = Field(default=None, ge=0)
    max_parallelism: int = Field(default=8, ge=1, le=256)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    allow_external_tools: bool = True
    allow_code_execution: bool = False
    deterministic: bool = Field(
        default=False,
        description="Prefer cached and heuristic paths over fresh sampling (reproducible runs).",
    )


class Evidence(BaseModel):
    """A retrieved or cited unit of information."""

    model_config = ConfigDict(extra="allow")

    id: str
    content: str
    source: str = Field(default="", description="URI, path, title or identifier of the origin document")
    score: float = Field(default=0.0, description="Retriever- or fusion-assigned relevance score (higher is better)")
    rank: Optional[int] = None
    retriever: str = Field(default="", description="sparse | dense | fused | reranked | web | provided | ...")
    metadata: dict[str, Any] = Field(default_factory=dict)
    span: Optional[tuple[int, int]] = Field(default=None, description="Character offsets inside the source document")

    def short(self, n: int = 160) -> str:
        c = self.content.replace("\n", " ")
        return c if len(c) <= n else c[: n - 3] + "..."


class TaskRequest(BaseModel):
    """A natural-language task plus structured inputs, context and limits."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., min_length=1, description="What to do, in natural language")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Structured inputs: documents, data, code, parameters")
    context: list[Evidence] = Field(default_factory=list, description="Evidence already gathered upstream")
    constraints: Constraints = Field(default_factory=Constraints)
    domain: str = Field(default="general", description="Name of a registered DomainProfile")
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)

    def derive(self, task: str, **overrides: Any) -> "TaskRequest":
        """Create a child request that inherits constraints, domain and metadata."""
        data = {
            "task": task,
            "inputs": dict(self.inputs),
            "context": list(self.context),
            "constraints": self.constraints.model_copy(),
            "domain": self.domain,
            "parent_id": self.request_id,
            "metadata": dict(self.metadata),
        }
        data.update(overrides)
        return TaskRequest(**data)


class Usage(BaseModel):
    """Token, call, cost and latency accounting. Additive."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            cache_hits=self.cache_hits + other.cache_hits,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
        )

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def cache_hit_ratio(self) -> float:
        total = self.total_prompt_tokens
        return self.cache_read_input_tokens / total if total else 0.0


class StepRecord(BaseModel):
    """One logical step inside a component run (plan, retrieve, verify, ...)."""

    model_config = ConfigDict(extra="allow")

    name: str
    component: str
    status: Status = Status.SUCCESS
    started_at: str = Field(default_factory=now_iso)
    duration_s: float = 0.0
    summary: str = ""
    usage: Usage = Field(default_factory=Usage)
    details: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """What every component returns."""

    model_config = ConfigDict(extra="allow")

    request_id: str
    component: str
    status: Status = Status.SUCCESS
    output: Any = Field(default=None, description="Component-specific structured output")
    output_text: str = Field(default="", description="Human-readable rendering of output")
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_basis: str = Field(default="", description="How confidence was derived (e.g. 'self-consistency 4/5 agree')")
    steps: list[StepRecord] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None
    finished_at: str = Field(default_factory=now_iso)

    @property
    def ok(self) -> bool:
        return self.status in (Status.SUCCESS, Status.CACHED, Status.PARTIAL)

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)
        self.usage = self.usage + step.usage

    def as_evidence(self, source: str | None = None) -> Evidence:
        """Expose this result to a downstream component as evidence."""
        return Evidence(
            id=f"result:{self.request_id}",
            content=self.output_text or str(self.output),
            source=source or f"{self.component}:{self.request_id}",
            score=self.confidence,
            retriever="provided",
            metadata={"status": self.status.value, "component": self.component},
        )

    def summary(self) -> str:
        return (
            f"[{self.component}] status={self.status.value} confidence={self.confidence:.2f} "
            f"calls={self.usage.llm_calls} cost=${self.usage.cost_usd:.4f} "
            f"tokens(in/out/cached)={self.usage.input_tokens}/{self.usage.output_tokens}/{self.usage.cache_read_input_tokens}"
        )


class BudgetTracker(BaseModel):
    """Mutable accumulator checked against :class:`Constraints` during a run."""

    model_config = ConfigDict(extra="forbid")

    constraints: Constraints
    spent: Usage = Field(default_factory=Usage)

    def charge(self, usage: Usage) -> None:
        from .errors import BudgetExceededError  # local import avoids a cycle at import time

        self.spent = self.spent + usage
        c = self.constraints
        if c.max_cost_usd is not None and self.spent.cost_usd > c.max_cost_usd:
            raise BudgetExceededError(f"cost ${self.spent.cost_usd:.4f} exceeds max ${c.max_cost_usd:.4f}")
        if c.max_llm_calls is not None and self.spent.llm_calls > c.max_llm_calls:
            raise BudgetExceededError(f"{self.spent.llm_calls} LLM calls exceed max {c.max_llm_calls}")

    def remaining_calls(self) -> Optional[int]:
        if self.constraints.max_llm_calls is None:
            return None
        return max(0, self.constraints.max_llm_calls - self.spent.llm_calls)

    def remaining_cost(self) -> Optional[float]:
        if self.constraints.max_cost_usd is None:
            return None
        return max(0.0, self.constraints.max_cost_usd - self.spent.cost_usd)
