"""Regression tests for the adversarial-review findings (one per critical/major fix, plus cheap minors)."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from pipeline_core import LLMRequest, LLMResponse, MockLLM, ScriptedLLM, Status, TaskRequest, get_profile

from parallelization import ParallelizationModel
from parallelization.cli import main
from parallelization.config import ParallelizationConfig
from parallelization.evaluation import metrics_from_result
from parallelization.executor import (
    MAX_LATENCY_SAMPLES,
    NESTED_SEP,
    CircuitBreaker,
    ExecutorStats,
    FailurePolicy,
    ManualClock,
    ParallelExecutionError,
    ParallelExecutor,
    RetryPolicy,
    TransientError,
)
from parallelization.patterns import PatternRunner, ToolCall, cache_aware_fan_out, parallel_tool_calls
from parallelization.plan import HeuristicPlanner, Partition, PlannerOutput, detect_enumerations, truncate_partitions

SOLUTION = {"answer": "margin is adequate", "evidence": ["datasheet"], "rationale": "numbers", "confidence": "high"}
MERGE = {"answer": "Both margins are adequate.", "unresolved": [], "confidence": "high"}


def manual_executor(**kw: Any) -> ParallelExecutor:
    return ParallelExecutor(clock=ManualClock(), **kw)


# ------------------------------------------------- critical: side effects


async def test_non_idempotent_work_is_attempted_exactly_once_on_transient_error():
    ex = manual_executor(retry=RetryPolicy(max_attempts=3, base_delay_s=0.0))
    writes: list[str] = []

    async def write(record: str) -> str:
        writes.append(record)  # the write commits ...
        raise TransientError("ack lost")  # ... and the acknowledgement is lost

    out = await ex.submit(write, "A", idempotent=False)
    assert not out.ok and out.attempts == 1 and out.retries == 0
    assert writes == ["A"] and ex.stats.retries == 0
    # the same failure on idempotent work is still retried
    out2 = await ex.submit(write, "B")
    assert out2.attempts == 3 and writes.count("B") == 3


async def test_ordered_tool_runs_once_and_is_not_memoised_across_turns():
    ex = manual_executor(retry=RetryPolicy(max_attempts=3, base_delay_s=0.0))
    writes: list[str] = []
    flaky = {"first": True}

    async def plm_write(record: str) -> str:
        writes.append(record)
        if flaky["first"]:
            flaky["first"] = False
            raise TransientError("timeout after commit")
        return f"written:{record}"

    tools = {"plm_write": plm_write}
    r0 = await parallel_tool_calls(ex, [ToolCall(call_id="1", name="plm_write", args={"record": "A"})], tools, ordered_tools=["plm_write"])
    assert not r0[0].ok and writes == ["A"]  # executed once, failure surfaced, never re-sent
    r1 = await parallel_tool_calls(ex, [ToolCall(call_id="1", name="plm_write", args={"record": "B"})], tools, ordered_tools=["plm_write"])
    r2 = await parallel_tool_calls(ex, [ToolCall(call_id="1", name="plm_write", args={"record": "C"})], tools, ordered_tools=["plm_write"])
    assert r1[0].result == "written:B" and r2[0].result == "written:C"  # a reused call_id never returns an earlier result
    assert writes == ["A", "B", "C"] and ex.stats.idempotent_hits == 0


async def test_non_idempotent_submit_is_never_memoised_even_with_a_key():
    ex = manual_executor()
    calls = 0

    async def work() -> int:
        nonlocal calls
        calls += 1
        return calls

    a = await ex.submit(work, idempotency_key="k", idempotent=False)
    b = await ex.submit(work, idempotency_key="k", idempotent=False)
    assert (a.value, b.value) == (1, 2) and not b.cached_idempotent


# ------------------------------------------------- major: breaker hygiene


async def test_caller_errors_do_not_open_the_shared_breaker():
    clock = ManualClock()
    breaker = CircuitBreaker(failure_threshold=2, window=4, min_calls=2, clock=clock, name="llm")
    ex = ParallelExecutor(breaker=breaker, clock=clock, retry=RetryPolicy(base_delay_s=0.0))
    calls = [ToolCall(call_id=str(i), name="no_such_tool") for i in range(5)]
    results = await parallel_tool_calls(ex, calls, {})
    assert all(not r.ok and "unknown tool" in (r.error or "") for r in results)

    async def bad() -> None:
        raise ValueError("caller bug")

    for _ in range(3):
        assert not (await ex.submit(bad, resource="llm")).ok

    async def healthy() -> str:
        return "ok"

    out = await ex.submit(healthy, resource="llm")
    assert out.ok and breaker.state == "closed" and breaker.trips == 0 and ex.stats.breaker_rejections == 0

    # positive control: dependency-class failures still trip it, and the trip is counted
    async def flaky() -> None:
        raise TransientError("503")

    ex.retry = RetryPolicy(max_attempts=1)
    await ex.submit(flaky)
    await ex.submit(flaky)
    assert breaker.state == "open" and ex.stats.breaker_trips == 1


def test_late_failures_while_open_do_not_retrip_or_extend_the_window():
    clock = ManualClock()
    b = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0, clock=clock)
    for _ in range(5):
        b.record_failure()
    assert b.state == "open" and b.trips == 1
    clock.advance(20.0)
    for _ in range(6):
        b.record_failure()  # in-flight calls failing after the trip
    assert b.trips == 1
    clock.advance(10.0)
    assert b.state == "half_open"


def test_small_breaker_window_keeps_the_failure_rate_rule():
    cfg = ParallelizationConfig(breaker_window=5, breaker_failure_threshold=100)
    ex = ParallelExecutor.from_config(cfg, clock=ManualClock())
    assert ex.breaker is not None and ex.breaker.min_calls == 5
    for _ in range(5):
        ex.breaker.record_failure()
    assert ex.breaker.state == "open"


# ------------------------------------------------- major: gate vs timeout


class SlowLLM(MockLLM):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(self.delay)
        return await super()._complete(request)


async def test_gate_wait_does_not_consume_the_per_attempt_timeout():
    cfg = ParallelizationConfig()
    # timeout ~2x one call: before the fix siblings timed out while waiting for the warm-up
    ex = ParallelExecutor(per_task_timeout_s=0.08, retry=RetryPolicy(max_attempts=1), clock=ManualClock())
    runner = PatternRunner(call=lambda req, step=None: SlowLLM(0.05).complete(req), config=cfg, executor=ex, profile=get_profile("general"), min_prefix_tokens=1)
    reqs = [LLMRequest.simple(f"q{i}", system=["shared stable prefix"]) for i in range(4)]
    outs = await cache_aware_fan_out(runner, reqs)
    assert all(o.ok for o in outs), [o.error for o in outs]
    assert ex.stats.timeouts == 0 and runner.gate.warmups == 1 and runner.gate.waits == 3
    waited = [o for o in outs if o.wait_s > 0.03]
    assert len(waited) == 3 and all(o.latency_s < 0.08 for o in outs)  # gate wait reported separately
    assert runner.stats.submitted == 4  # per-run counters


# ------------------------------------------------- major: plan truncation


def test_token_cap_batches_surplus_items_instead_of_dropping_them():
    model = ParallelizationModel(MockLLM())
    tickets = [f"ticket-{i}" for i in range(1, 17)]
    plan = model.heuristic_plan(TaskRequest(task="Classify each ticket.", inputs={"items": tickets}))
    assert plan.pattern == "sectioning" and len(plan.partitions) == 15
    text = "\n".join(p.task for p in plan.partitions)
    assert all(t in text for t in tickets)  # nothing silently dropped
    assert "batched" in plan.rationale and plan.expected_calls == 16


def test_truncate_partitions_keeps_a_dependency_closed_subset():
    parts = [Partition(id="summary", task="s", depends_on=["a", "b"]), Partition(id="a", task="a"), Partition(id="b", task="b")]
    kept, dropped = truncate_partitions(parts, 2)
    assert [p.id for p in kept] == ["a", "b"] and dropped == ["summary"]


async def test_budget_truncation_of_llm_plan_degrades_instead_of_failing():
    plan_json = {
        "pattern": "sectioning",
        "partitions": [
            {"id": "summary", "task": "Summarise both margins.", "depends_on": ["a", "b"]},
            {"id": "a", "task": "Assess the thermal margin.", "depends_on": []},
            {"id": "b", "task": "Assess the structural margin.", "depends_on": []},
        ],
        "aggregator": "llm",
        "rationale": "two assessments and a summary",
    }
    llm = ScriptedLLM([plan_json, SOLUTION, SOLUTION, MERGE])
    model = ParallelizationModel(llm, planner_mode="llm", executor=ParallelExecutor.from_config(ParallelizationConfig(), clock=ManualClock()))
    from pipeline_core import Constraints

    result = await model.run(TaskRequest(task="Assess the enclosure.", constraints=Constraints(max_llm_calls=4)))
    assert result.status != Status.FAILED and result.output_text == "Both margins are adequate."
    assert [p["id"] for p in result.metadata["plan"]["partitions"]] == ["a", "b"]
    assert any("summary" in w and "dropped" in w for w in result.warnings)


# ------------------------------------------------- minor: hedging, DAG, domain floor


async def test_hedge_returns_first_success_when_primary_fails_late():
    ex = ParallelExecutor(hedge_after_s=0.005, hedge_fraction=1.0, retry=RetryPolicy(max_attempts=1))
    calls = 0

    async def work() -> str:
        nonlocal calls
        calls += 1
        me = calls
        if me == 1:
            await asyncio.sleep(0.02)
            raise TransientError("primary failed after the hedge was sent")
        await asyncio.sleep(0.04)
        return "hedge"

    out = await ex.submit(work)
    assert out.ok and out.value == "hedge" and ex.stats.hedge_wins == 1


async def test_dag_fail_fast_cancels_in_wave_siblings():
    ex = manual_executor()
    never = asyncio.Event()
    started: list[str] = []

    async def fn(node: str, deps: dict) -> str:
        started.append(node)
        if node == "a":
            raise ValueError("a failed")
        await never.wait()  # would hang forever without cancellation
        return node

    with pytest.raises(ParallelExecutionError) as info:
        await asyncio.wait_for(ex.dag({"a": [], "b": [], "c": [], "d": ["a"]}, fn, failure_policy=FailurePolicy.fail_fast()), 2.0)
    by_key = {o.key: o for o in info.value.outcomes}
    assert by_key["b"].cancelled and by_key["c"].cancelled and by_key["d"].cancelled and "d" not in started


def test_llm_planned_vote_respects_the_high_risk_sample_floor():
    planner = HeuristicPlanner(ParallelizationConfig())
    req = TaskRequest(task="Compute the bolt shear load.", domain="civil_structural_engineering")
    plan = planner.finalize(PlannerOutput(pattern="voting", samples=5, verifiability="exact_match", rationale="x"), req, get_profile(req.domain), planner="llm")
    assert plan.samples == 10 and plan.max_samples == 10


def test_enumeration_trailing_clause_is_not_part_of_the_last_item():
    assert detect_enumerations("Compare steel, aluminium and titanium for a lightweight bracket.") == ["steel", "aluminium", "titanium"]


def test_partition_has_no_unused_kind_field():
    assert "kind" not in Partition.model_fields


# ------------------------------------------------- minor: metrics


def test_amdahl_bound_uses_the_plans_serial_fraction():
    model = ParallelizationModel(MockLLM(), executor=ParallelExecutor.from_config(ParallelizationConfig(), clock=ManualClock()))
    res = model.run_sync(TaskRequest(task="Justify the trade-off between bolt preload and gasket creep for the flange of the pressure vessel inspected last week.", domain="mechanical_engineering"))
    m = metrics_from_result(res)
    assert m.pattern == "ensemble" and m.expected_calls == 7
    assert m.amdahl_bound == pytest.approx(7 / 3, rel=1e-6)
    assert "breaker_trips" in res.metadata["executor"] and m.submitted == res.metadata["executor"]["submitted"]


async def test_per_run_stats_are_correct_on_a_shared_executor():
    ex = ParallelExecutor.from_config(ParallelizationConfig(), clock=ManualClock())
    model = ParallelizationModel(MockLLM(), executor=ex)
    a, b = await asyncio.gather(
        model.run("Compare steel, aluminium and titanium for a bracket."),
        model.run("Compare copper and brass for a busbar."),
    )
    assert a.metadata["executor"]["submitted"] == 3 and b.metadata["executor"]["submitted"] == 2
    assert ex.stats.submitted == 5


def test_latency_samples_are_bounded():
    s = ExecutorStats()
    for i in range(MAX_LATENCY_SAMPLES + 5):
        s.record_latency(float(i))
    assert len(s.latencies_s) == MAX_LATENCY_SAMPLES and s.latencies_s[-1] == MAX_LATENCY_SAMPLES + 4


async def test_rate_limit_mismatch_is_warned():
    model = ParallelizationModel(MockLLM(), config=ParallelizationConfig(requests_per_minute=100), executor=ParallelExecutor(clock=ManualClock()))
    result = await model.run("Compare copper and brass for a busbar.")
    assert any("plan.rate_limits" in w for w in result.warnings)


# ------------------------------------------------- coordinator: re-entrant executor


async def test_nested_map_on_the_same_executor_does_not_deadlock():
    ex = ParallelExecutor(max_concurrency=2, clock=ManualClock())

    async def inner(x: int) -> int:
        await asyncio.sleep(0)
        return x * 10

    async def outer(i: int) -> list[Any]:
        return await ex.map(inner, [i, i + 1, i + 2])

    results = await asyncio.wait_for(ex.map(outer, [0, 1, 2, 3]), 2.0)
    assert results == [[0, 10, 20], [10, 20, 30], [20, 30, 40], [30, 40, 50]]
    snap = ex.bulkhead.snapshot()
    assert snap["max_inflight"]["default"] <= 2 and snap["max_inflight"][f"default{NESTED_SEP}1"] <= 2
    assert ex.stats.submitted == 16 and ex.nesting_depth() == 0


async def test_nested_dag_and_gather_inside_a_fan_out_complete():
    ex = ParallelExecutor(max_concurrency=1, clock=ManualClock())

    async def leaf() -> str:
        return "leaf"

    async def node(n: str, deps: dict) -> str:
        outs = await ex.gather(leaf, leaf)
        return n + str(len(outs))

    async def outer(i: int) -> dict:
        res = await ex.dag({"a": [], "b": ["a"]}, node)
        return {k: v.value for k, v in res.items()}

    results = await asyncio.wait_for(ex.map(outer, [0, 1]), 2.0)
    assert results == [{"a": "a2", "b": "b2"}] * 2


# ------------------------------------------------- tests hygiene: CLI does not mutate env


def test_cli_offline_does_not_mutate_process_environment(capsys, monkeypatch):
    monkeypatch.delenv("PIPELINE_OFFLINE", raising=False)
    assert main(["--offline", "--plan-only", "Compare copper and brass"]) == 0
    capsys.readouterr()
    assert "PIPELINE_OFFLINE" not in os.environ
