"""Shared fixtures: fakes for the protocol objects and deterministic model factories."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pytest

from pipeline_core import Evidence, MockLLM, TaskRequest, TaskResult, Usage, Verdict

from task_decomposition import TaskDecompositionModel

PUMP_TASK = "For each of the three pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status."
BRIDGE_TASK = "Review the bridge deck design: verify load combinations, check shear capacity, check deflection limits, and list the code clauses used."
TRIVIAL_TASK = "Convert 25 degC to kelvin."
DEBUG_TASK = "Debug why the nightly ETL job fails intermittently and propose a fix with a regression test."
EXPLORE_TASK = "Compare three alternative gearbox layouts for the stated envelope and select the best trade-off between mass, efficiency and cost."
CHAIN_TASK = "First compute the design load, then size the beam section, then check deflection against the limit, and finally report the demand/capacity ratio with the code clause used."


class FakeRetriever:
    def __init__(self, docs: Optional[Sequence[Evidence]] = None) -> None:
        self.docs = list(docs or [Evidence(id="doc-x", content="Datasheet: duty point 120 m3/h at 32 m head.", source="datasheet", score=0.9, retriever="sparse")])
        self.queries: list[str] = []

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        self.queries.append(query)
        return self.docs[:k]


class FakeVerifier:
    def __init__(self, *, passed: bool = True, score: float = 0.9, fail_when: Optional[str] = None) -> None:
        self.passed = passed
        self.score = score
        self.fail_when = fail_when
        self.calls: list[tuple[str, str]] = []
        self.evidence: list[list[str]] = []  # evidence ids passed with each call, aligned with ``calls``

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict:
        self.calls.append((answer, task))
        self.evidence.append([e.id for e in evidence])
        failed = (not self.passed) or (self.fail_when is not None and self.fail_when.lower() in task.lower())
        return Verdict(method="fake", passed=not failed, score=0.2 if failed else self.score, issues=["fake issue"] if failed else [], details={"usage": Usage(llm_calls=1).model_dump()})


class FakeExecutor:
    def __init__(self) -> None:
        self.concurrencies: list[Optional[int]] = []

    async def map(self, fn, items, *, concurrency: Optional[int] = None):  # type: ignore[no-untyped-def]
        self.concurrencies.append(concurrency)
        out = []
        for it in items:
            out.append(await fn(it))
        return out


class FakeSolver:
    """A ``SubTaskSolver`` standing in for another component's ``run``."""

    def __init__(self) -> None:
        self.requests: list[TaskRequest] = []

    async def __call__(self, request: TaskRequest) -> TaskResult:
        self.requests.append(request)
        return TaskResult(request_id=request.request_id, component="fake_rag", output_text=f"solved: {request.task}", confidence=0.9, usage=Usage(llm_calls=1, cost_usd=0.01), evidence=list(request.context)[:1])


@pytest.fixture
def fast_model():
    """Model factory with zero retry delay so failure paths stay fast."""

    def _make(llm: Any = None, **kw: Any) -> TaskDecompositionModel:
        kw.setdefault("retry_base_delay_s", 0.0)
        kw.setdefault("retry_max_delay_s", 0.0)
        return TaskDecompositionModel(llm or MockLLM(), **kw)

    return _make
