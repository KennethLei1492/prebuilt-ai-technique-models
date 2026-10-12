"""Pattern machinery: warm-then-burst gate, speculative race, parallel tool calls, direct API."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pipeline_core import LLMRequest, LLMResponse, MockLLM, get_profile

from parallelization.config import ParallelizationConfig
from parallelization.executor import ManualClock, ParallelExecutor, RetryPolicy
from parallelization.patterns import PatternRunner, PrefixWarmGate, ToolCall, cache_aware_fan_out, parallel_tool_calls, speculative
from parallelization.prompts import AnswerOutput


class TrackingLLM(MockLLM):
    """MockLLM that records the in-flight count seen by every call."""

    def __init__(self, delay: float = 0.01, **kw: Any) -> None:
        super().__init__(**kw)
        self.delay = delay
        self.inflight = 0
        self.seen: list[int] = []
        self.order: list[str] = []

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        self.inflight += 1
        self.seen.append(self.inflight)
        self.order.append(f"start:{request.last_user_text()}")
        try:
            await asyncio.sleep(self.delay)
            return await super()._complete(request)
        finally:
            self.inflight -= 1
            self.order.append(f"end:{request.last_user_text()}")


LONG_PREFIX = ["You are a careful reviewer. " * 300]  # ~2k tokens: above every model's cacheable minimum


def make_runner(llm: MockLLM, *, cache_aware: bool = True) -> PatternRunner:
    cfg = ParallelizationConfig()
    ex = ParallelExecutor.from_config(cfg, clock=ManualClock())
    return PatternRunner.from_llm(llm, cfg, ex, get_profile("general"), cache_aware=cache_aware)


async def test_warm_then_burst_runs_first_request_alone_then_bursts():
    llm = TrackingLLM()
    runner = make_runner(llm)
    reqs = [LLMRequest.simple(f"q{i}", system=LONG_PREFIX) for i in range(5)]
    outs = await cache_aware_fan_out(runner, reqs)
    assert all(o.ok for o in outs)
    assert llm.seen[0] == 1  # warm-up ran alone
    assert llm.order[1] == "end:q0" or llm.order[0] == "start:q0"
    assert llm.order.index("end:q0") < min(llm.order.index(f"start:q{i}") for i in range(1, 5))
    assert max(llm.seen[1:]) > 1  # the remaining four burst concurrently
    assert runner.gate.warmups == 1 and runner.gate.waits == 4


async def test_gate_is_inert_for_short_prefixes_and_when_disabled():
    llm = TrackingLLM()
    runner = make_runner(llm)
    outs = await cache_aware_fan_out(runner, [LLMRequest.simple(f"q{i}", system=["short"]) for i in range(4)])
    assert all(o.ok for o in outs) and runner.gate.warmups == 0 and max(llm.seen) > 1
    llm2 = TrackingLLM()
    runner2 = make_runner(llm2, cache_aware=False)
    await cache_aware_fan_out(runner2, [LLMRequest.simple(f"q{i}", system=LONG_PREFIX) for i in range(4)])
    assert runner2.gate.warmups == 0 and max(llm2.seen) > 1


async def test_gate_releases_siblings_even_if_warmup_fails():
    gate = PrefixWarmGate(min_prefix_tokens=1)
    req = LLMRequest.simple("x", system=["sys"])
    with pytest.raises(RuntimeError):
        async with gate.admit(req, "m"):
            raise RuntimeError("warm-up failed")
    async with gate.admit(req, "m") as first:
        assert first is False  # sibling proceeds immediately


async def test_speculative_accepts_fast_and_cancels_slow():
    slow_cancelled = False

    async def fast() -> str:
        await asyncio.sleep(0.01)  # let the slow path start so cancellation is observable
        return "fast"

    async def slow() -> str:
        nonlocal slow_cancelled
        try:
            await asyncio.sleep(1)
            return "slow"
        except asyncio.CancelledError:
            slow_cancelled = True
            raise

    value, source = await speculative(fast, slow, lambda v: _ok(True))
    assert (value, source) == ("fast", "fast") and slow_cancelled


async def test_speculative_falls_back_to_slow_when_rejected_or_fast_fails():
    async def fast_bad() -> str:
        raise RuntimeError("fast path exploded")

    async def slow() -> str:
        return "slow"

    assert await speculative(fast_bad, slow, lambda v: _ok(True)) == ("slow", "slow")

    async def fast() -> str:
        return "fast"

    assert await speculative(fast, slow, lambda v: _ok(False)) == ("slow", "slow")


async def _ok(flag: bool) -> bool:
    return flag


async def test_parallel_tool_calls_runs_concurrently_and_orders_side_effects():
    ex = ParallelExecutor(retry=RetryPolicy(max_attempts=1), clock=ManualClock())
    writes: list[str] = []

    async def add(a: int, b: int) -> int:
        await asyncio.sleep(0.005)
        return a + b

    def mul(a: int, b: int) -> int:
        return a * b

    async def write(line: str) -> str:
        writes.append(line)
        return "ok"

    calls = [
        ToolCall(call_id="c1", name="write", args={"line": "first"}),
        ToolCall(call_id="c2", name="add", args={"a": 1, "b": 2}),
        ToolCall(call_id="c3", name="mul", args={"a": 3, "b": 4}),
        ToolCall(call_id="c4", name="write", args={"line": "second"}),
        ToolCall(call_id="c5", name="nope", args={}),
    ]
    results = await parallel_tool_calls(ex, calls, {"add": add, "mul": mul, "write": write}, ordered_tools=["write"])
    assert [r.call_id for r in results] == ["c1", "c2", "c3", "c4", "c5"]
    assert results[1].result == 3 and results[2].result == 12
    assert writes == ["first", "second"]  # ordered tools preserve emission order
    assert not results[4].ok and "unknown tool" in (results[4].error or "")
    assert ex.bulkhead.limit_for("ordered:write") >= 1


async def test_runner_budget_guard_blocks_calls_without_remaining_budget():
    llm = MockLLM()
    cfg = ParallelizationConfig()
    ex = ParallelExecutor.from_config(cfg, clock=ManualClock())
    remaining = {"n": 1}
    runner = PatternRunner(call=lambda req, step=None: llm.complete(req), config=cfg, executor=ex, profile=get_profile("general"), remaining_calls=lambda: remaining["n"])
    assert runner.budget_cap(5, reserve=0) == 1
    await runner.call(LLMRequest.simple("one"))
    remaining["n"] = 0
    from pipeline_core import BudgetExceededError

    with pytest.raises(BudgetExceededError):
        await runner.call(LLMRequest.simple("two"))
    assert runner.calls == 1 and runner.usage.llm_calls == 1


async def test_runner_system_blocks_include_profile_and_rigor_for_high_risk():
    llm = MockLLM()
    cfg = ParallelizationConfig()
    ex = ParallelExecutor.from_config(cfg, clock=ManualClock())
    from parallelization.prompts import SEQUENTIAL

    low = PatternRunner.from_llm(llm, cfg, ex, get_profile("general"))
    high = PatternRunner.from_llm(llm, cfg, ex, get_profile("civil_structural_engineering"))
    req_low = low.request(SEQUENTIAL, AnswerOutput, task="t", context="c")
    req_high = high.request(SEQUENTIAL, AnswerOutput, task="t", context="c")
    assert "Domain: general" in "\n".join(req_low.system_stable) and "engineering rigor" not in "\n".join(req_low.system_stable).lower()
    assert "Domain: civil_structural_engineering" in "\n".join(req_high.system_stable) and "engineering rigor" in "\n".join(req_high.system_stable).lower()
    assert req_high.output_schema is not None and req_high.output_schema["additionalProperties"] is False
    assert req_low.system_stable == low.request(SEQUENTIAL, AnswerOutput, task="other", context="x").system_stable  # byte-identical prefix across branches
