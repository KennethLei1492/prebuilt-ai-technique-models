"""Scheduler and retry policy with an injected clock, sleep and rng (no waiting)."""
from __future__ import annotations

import asyncio
import random

import pytest

from pipeline_core import BudgetExceededError, LLMRefusalError, PlanValidationError, Status

from task_decomposition import RetryPolicy, Scheduler, SubTask, SubTaskResult, TaskGraph, failure_classes


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        self.t += 0.25
        return self.t


def _graph() -> TaskGraph:
    return TaskGraph(
        root_task="r",
        tasks=[
            SubTask(id="t1", description="a"),
            SubTask(id="t2", description="b"),
            SubTask(id="t3", description="c uses $t1.output", depends_on=["t1", "t2"], inputs={"x": "$t1.output"}),
            SubTask(id="t4", description="d", depends_on=["t3"]),
        ],
    )


def test_retry_policy_backoff_is_capped_and_jittered_deterministically():
    pol = RetryPolicy(base_delay_s=0.5, factor=2.0, max_delay_s=3.0, jitter=False)
    assert [pol.delay(a) for a in (1, 2, 3, 4, 5)] == [0.5, 1.0, 2.0, 3.0, 3.0]
    jit = RetryPolicy(base_delay_s=1.0, factor=2.0, max_delay_s=8.0)
    rng = random.Random(7)
    d = [jit.delay(a, rng) for a in (1, 2, 3)]
    assert all(0.0 <= x <= cap for x, cap in zip(d, (1.0, 2.0, 4.0)))
    assert d == [RetryPolicy(base_delay_s=1.0, factor=2.0, max_delay_s=8.0).delay(a, random.Random(7)) for a in (1,)] + d[1:]
    assert RetryPolicy.is_retryable(RuntimeError("x"))
    assert not RetryPolicy.is_retryable(BudgetExceededError("b"))
    assert not RetryPolicy.is_retryable(LLMRefusalError("r"))
    assert not RetryPolicy.is_retryable(PlanValidationError("p"))


async def test_scheduler_substitutes_outputs_and_records_waves():
    sched = Scheduler(parallelism=4, clock=FakeClock(), sleep=_no_sleep)
    seen: dict[str, SubTask] = {}

    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        seen[task.id] = task
        return SubTaskResult(task_id=task.id, output_text=f"out-{task.id}", confidence=0.9)

    report = await sched.run(_graph(), solve)
    assert report.all_ok() and report.waves == [["t1", "t2"], ["t3"], ["t4"]]
    assert seen["t3"].inputs == {"x": "out-t1"} and "out-t1" in seen["t3"].description
    assert report.results["t3"].duration_s > 0 and report.results["t3"].attempts == 1
    assert report.max_concurrency_observed >= 1 and report.retries == 0


async def test_retry_then_success_uses_injected_sleep():
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    attempts = {"t1": 0}

    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        attempts[task.id] = attempts.get(task.id, 0) + 1
        if task.id == "t1" and attempts["t1"] == 1:
            raise RuntimeError("transient")
        return SubTaskResult(task_id=task.id, output_text="ok", confidence=0.8)

    sched = Scheduler(parallelism=2, retry=RetryPolicy(base_delay_s=0.5, jitter=False), clock=FakeClock(), sleep=fake_sleep)
    g = TaskGraph(root_task="r", tasks=[SubTask(id="t1", description="a", max_retries=1)])
    report = await sched.run(g, solve)
    assert report.all_ok() and report.retries == 1 and sleeps == [0.5]
    assert report.results["t1"].attempts == 2


async def test_failure_skips_dependents_and_budget_is_not_retried():
    calls = {"n": 0}

    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        calls["n"] += 1
        if task.id == "t1":
            raise BudgetExceededError("LLM call budget exhausted")
        return SubTaskResult(task_id=task.id, output_text="ok", confidence=0.8)

    sched = Scheduler(parallelism=4, clock=FakeClock(), sleep=_no_sleep)
    report = await sched.run(_graph(), solve)
    r1 = report.results["t1"]
    assert r1.status == Status.FAILED and r1.error.startswith("budget") and r1.attempts == 1
    assert report.results["t3"].status == Status.SKIPPED and "dependency failed: t1" in report.results["t3"].error
    assert report.results["t4"].status == Status.SKIPPED
    assert report.failed_ids == ["t1"] and sorted(report.skipped_ids) == ["t3", "t4"] and report.completed_ids == ["t2"]
    assert calls["n"] == 2  # t1 once (not retried), t2 once


async def test_should_stop_marks_remaining_not_started_and_resumes_from_completed():
    stop = {"flag": False}

    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        return SubTaskResult(task_id=task.id, output_text="ok", confidence=0.8)

    sched = Scheduler(parallelism=4, clock=FakeClock(), sleep=_no_sleep)
    first = await sched.run(_graph(), solve, should_stop=lambda: "budget" if stop["flag"] else None, on_result=lambda r: stop.__setitem__("flag", True))
    assert first.stopped_reason == "budget"
    assert first.results["t3"].status == Status.SKIPPED and first.results["t3"].error == "not started: budget"
    completed = {tid: r for tid, r in first.results.items() if r.ok}
    second = await sched.run(_graph(), solve, completed=completed)
    assert second.all_ok() and second.waves_executed == 3


async def test_concurrency_ceiling_is_respected():
    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        await asyncio.sleep(0.001)
        return SubTaskResult(task_id=task.id, output_text="ok", confidence=0.8)

    wide = TaskGraph(root_task="r", tasks=[SubTask(id=f"t{i}", description=f"n{i}") for i in range(6)])
    sched = Scheduler(parallelism=2, clock=FakeClock(), sleep=_no_sleep)
    report = await sched.run(wide, solve)
    assert report.all_ok() and 1 <= report.max_concurrency_observed <= 2


async def test_long_outputs_are_truncated_for_dependents():
    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        return SubTaskResult(task_id=task.id, output_text="x" * 500 if task.id == "t1" else task.inputs.get("x", ""), confidence=0.8)

    g = TaskGraph(root_task="r", tasks=[SubTask(id="t1", description="a"), SubTask(id="t2", description="b", depends_on=["t1"], inputs={"x": "$t1.output"})])
    sched = Scheduler(parallelism=1, max_output_chars=50, clock=FakeClock(), sleep=_no_sleep)
    report = await sched.run(g, solve)
    assert len(report.results["t2"].output_text) == 50 and report.results["t2"].output_text.endswith("...")


async def _no_sleep(_: float) -> None:
    return None


async def test_dependents_of_not_started_nodes_inherit_the_stop_reason():
    """Regression: a node downstream of one that never started (budget/deadline) is a budget stop, not a dependency failure."""
    spent = {"n": 0}

    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        spent["n"] += 1
        if task.id == "x":
            raise RuntimeError("real failure")
        return SubTaskResult(task_id=task.id, output_text="ok", confidence=0.8)

    g = TaskGraph(root_task="r", tasks=[SubTask(id="a", description="a"), SubTask(id="b", description="b"), SubTask(id="c", description="c", depends_on=["b"])])
    sched = Scheduler(parallelism=4, clock=FakeClock(), sleep=_no_sleep)
    report = await sched.run(g, solve, remaining_calls=lambda: 1 - spent["n"])
    assert report.results["a"].ok and report.results["b"].error == "not started: budget"
    assert report.results["c"].status == Status.SKIPPED and report.results["c"].error == "not started: budget (upstream b)"
    assert failure_classes(report.results) == {"budget_or_deadline": 2}
    # a real upstream failure still reads "dependency failed", and names only the failed dependency
    spent["n"] = 0
    g2 = TaskGraph(root_task="r", tasks=[SubTask(id="x", description="x", max_retries=0), SubTask(id="b", description="b"), SubTask(id="c", description="c", depends_on=["x", "b"])])
    report2 = await Scheduler(parallelism=1, clock=FakeClock(), sleep=_no_sleep).run(g2, solve, remaining_calls=lambda: 1 - spent["n"])
    assert report2.results["c"].error == "dependency failed: x"
    assert failure_classes(report2.results) == {"solver_failed": 1, "budget_or_deadline": 1, "skipped_dependency": 1}


async def test_remaining_calls_reserves_budget_before_fanout():
    """Section 3.2: a fan-out never dispatches more nodes than the call budget can pay for."""
    spent = {"n": 0}

    async def solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
        spent["n"] += 1
        return SubTaskResult(task_id=task.id, output_text="ok", confidence=0.8)

    wide = TaskGraph(root_task="r", tasks=[SubTask(id=f"t{i}", description=f"n{i}") for i in range(5)])
    sched = Scheduler(parallelism=8, clock=FakeClock(), sleep=_no_sleep)
    report = await sched.run(wide, solve, remaining_calls=lambda: 2 - spent["n"])
    assert spent["n"] == 2 and report.stopped_reason == "budget"
    assert sorted(report.completed_ids) == ["t0", "t1"]
    assert all(report.results[f"t{i}"].error == "not started: budget" for i in (2, 3, 4))
    assert len(report.steps) == 5 and {s.name for s in report.steps} == {f"subtask.t{i}" for i in range(5)}
    assert report.steps[0].component == "task_decomposition" and report.steps[0].status == Status.SUCCESS
    # unlimited budget: the whole wave is dispatched at once
    spent["n"] = 0
    full = await Scheduler(parallelism=8, clock=FakeClock(), sleep=_no_sleep, component="x").run(wide, solve, remaining_calls=lambda: None)
    assert full.all_ok() and spent["n"] == 5 and full.steps[0].component == "x"
