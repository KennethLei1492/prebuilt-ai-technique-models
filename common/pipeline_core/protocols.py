"""Structural interfaces that let the six models call each other without importing each other.

Each model package depends only on ``pipeline_core``. When a model needs a
capability another model provides (RAG needs a retriever, decomposition needs
an executor, everything benefits from a cache) it accepts an object satisfying
one of these protocols. The orchestrator wires concrete instances together;
unit tests inject small fakes.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Evidence, TaskRequest, TaskResult

T = TypeVar("T")
R = TypeVar("R")


class Verdict(BaseModel):
    """Outcome of one verification method."""

    model_config = ConfigDict(extra="allow")

    method: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="Calibrated probability the answer is acceptable")
    issues: list[str] = Field(default_factory=list)
    corrected_answer: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence that supports (or contradicts) the answer")
    details: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Provided by the hybrid retrieval model; consumed by pipeline RAG and verification."""

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]: ...


@runtime_checkable
class Verifier(Protocol):
    """Provided by the answer verification model; consumed by RAG and decomposition."""

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict: ...


@runtime_checkable
class Executor(Protocol):
    """Provided by the parallelization model; consumed by decomposition, retrieval fan-out and verification sampling."""

    async def map(
        self,
        fn: Callable[[T], Awaitable[R]],
        items: Sequence[T],
        *,
        concurrency: Optional[int] = None,
    ) -> list[Any]: ...


@runtime_checkable
class ResponseCache(Protocol):
    """Provided by the caching model; consumed by every LLM-calling component through ``CachedLLMClient``."""

    async def get(self, key: str) -> Optional[Any]: ...

    async def set(self, key: str, value: Any, *, ttl_s: Optional[float] = None, tags: Sequence[str] = ()) -> None: ...

    async def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int: ...


SubTaskSolver = Callable[[TaskRequest], Awaitable[TaskResult]]
"""Callable that solves one sub-task; the decomposition model accepts any of these (often another component's ``run``)."""
