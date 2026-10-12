"""Executor kernel: token bucket, retry/backoff with an injectable clock, breaker, bulkhead, policies, DAG waves."""
from __future__ import annotations

import asyncio
import random

import pytest

from pipeline_core import BudgetExceededError, LLMRefusalError

from parallelization.executor import (
    Bulkhead,
    CircuitBreaker,
    FailurePolicy,
    ManualClock,
    ParallelExecutionError,
    ParallelExecutor,
    RateLimiter,
    RetryBudget,
    RetryPolicy,
    TokenBucket,
    TransientError,
)


# ----------------------------------------------------------- token bucket


def test_token_bucket_refills_with_injected_clock():
    clock = ManualClock()
    bucket = TokenBucket(rate_per_s=2.0, capacity=4.0, clock=clock)
    assert bucket.try_acquire(4.0)
    assert not bucket.try_acquire(1.0)
    assert bucket.time_until(1.0) == pytest.approx(0.5)
    clock.advance(1.0)
    assert bucket.available() == pytest.approx(2.0)
    assert bucket.try_acquire(2.0)


async def test_token_bucket_acquire_waits_on_clock_not_wall_time():
    clock = ManualClock()
    bucket = TokenBucket(rate_per_s=1.0, capacity=1.0, clock=clock, initial=0.0)
    waited = await bucket.acquire(1.0)
    assert waited == pytest.approx(1.0)
    assert clock.sleeps == [pytest.approx(1.0)]


async def test_rate_limiter_enforces_rpm_and_tokens():
    clock = ManualClock()
    rl = RateLimiter(requests_per_minute=60, input_tokens_per_minute=6000, burst_s=1.0, clock=clock)
    assert rl.enabled
    await rl.acquire(input_tokens=100)  # bucket starts full: 1 request, 100 tokens -> no wait
    waited = await rl.acquire(input_tokens=100)  # request bucket empty -> wait 1s
    assert waited == pytest.approx(1.0)
    rl.feedback(remaining_requests=0)
    assert rl.requests is not None and rl.requests.available() == 0.0


# ------------------------------------------------------------------ retry


def test_retry_policy_full_jitter_is_capped_and_honours_retry_after():
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0, jitter="full")
    rng = random.Random(1)
    for attempt in range(6):
        d = policy.delay(attempt, rng)
        assert 0.0 <= d <= min(4.0, 0.5 * 2**attempt)
    assert policy.delay(0, rng, retry_after=7.0) >= 7.0
    assert RetryPolicy(jitter="none", base_delay_s=1.0, max_delay_s=20.0).delay(3) == 8.0


def test_retry_policy_classifies_errors():
    policy = RetryPolicy()
    assert policy.is_retryable(TransientError("429"))
    assert policy.is_retryable(TimeoutError())
    assert not policy.is_retryable(LLMRefusalError("refused"))  # LLMError subclass, but never_retry wins
    assert not policy.is_retryable(ValueError("bad"))
    assert not policy.is_retryable(BudgetExceededError("budget"))


def test_retry_budget_caps_retries_to_fraction_of_requests():
    clock = ManualClock()
    budget = RetryBudget(ratio=0.1, min_retries=1, window_s=60, clock=clock)
    for _ in range(30):
        budget.record_request()
    assert budget.allowance() == 3
    assert [budget.try_consume() for _ in range(4)] == [True, True, True, False]
    assert budget.denied == 1
    clock.advance(61)
    assert budget.requests == 0 and budget.allowance() == 1


async def test_executor_retries_transient_errors_with_backoff_on_manual_clock():
    clock = ManualClock()
    ex = ParallelExecutor(retry=RetryPolicy(max_attempts=3, base_delay_s=0.5, max_delay_s=20), clock=clock, rng=random.Random(0))
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TransientError("503", retry_after_s=2.0)
        return "ok"

    out = await ex.submit(flaky)
    assert out.ok and out.value == "ok"
    assert out.attempts == 3 and out.retries == 2
    assert len(clock.sleeps) == 2 and all(s >= 2.0 for s in clock.sleeps)  # retry-after is a floor
    assert ex.stats.retries == 2 and ex.stats.succeeded == 1


async def test_executor_does_not_retry_non_retryable_or_beyond_budget():
    clock = ManualClock()
    ex = ParallelExecutor(retry=RetryPolicy(max_attempts=3), retry_budget=RetryBudget(ratio=0.1, min_retries=0, clock=clock), clock=clock)

    async def bad() -> None:
        raise ValueError("permanent")

    out = await ex.submit(bad)
    assert not out.ok and out.attempts == 1 and out.error_type == "ValueError"

    async def transient() -> None:
        raise TransientError("429")

    out = await ex.submit(transient)
    assert not out.ok and out.attempts == 1  # budget allowance is 0 -> no retry
    assert ex.retry_budget.denied == 1


# ------------------------------------------------------------- breaker


def test_circuit_breaker_closed_open_half_open_cycle():
    clock = ManualClock()
    br = CircuitBreaker(failure_threshold=2, recovery_timeout_s=30, clock=clock)
    assert br.state == "closed" and br.allow()
    br.record_failure()
    br.record_failure()
    assert br.state == "open" and br.trips == 1
    assert not br.allow() and br.rejected == 1
    clock.advance(30)
    assert br.state == "half_open"
    assert br.allow()  # one probe
    assert not br.allow()  # second caller rejected while probe in flight
    br.record_success()
    assert br.state == "closed"
    br.record_failure()
    assert br.state == "closed"  # one failure is below threshold


def test_circuit_breaker_opens_on_failure_rate_window():
    br = CircuitBreaker(failure_threshold=100, failure_rate=0.5, window=10, min_calls=10, clock=ManualClock())
    for ok in [True, False] * 5:
        (br.record_success if ok else br.record_failure)()
    assert br.state == "open"


async def test_executor_fails_fast_while_breaker_open():
    clock = ManualClock()
    br = CircuitBreaker(failure_threshold=1, recovery_timeout_s=10, clock=clock)
    ex = ParallelExecutor(breaker=br, retry=RetryPolicy(max_attempts=1), clock=clock)

    async def boom() -> None:
        raise TransientError("x")

    await ex.submit(boom)
    out = await ex.submit(boom)
    assert out.error_type == "CircuitOpenError" and ex.stats.breaker_rejections == 1


# ------------------------------------------------------------- bulkhead


async def test_bulkhead_limits_concurrency_per_resource():
    bh = Bulkhead({"slow": 2}, default_limit=8)
    peak = 0
    inflight = 0

    async def work() -> None:
        nonlocal peak, inflight
        async with bh.slot("slow"):
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0.005)
            inflight -= 1

    await asyncio.gather(*(work() for _ in range(6)))
    assert peak == 2
    assert bh.snapshot()["max_inflight"]["slow"] == 2
    assert bh.snapshot()["waits"]["slow"] >= 1


# ----------------------------------------------------------- policies


def test_failure_policy_required_counts():
    assert FailurePolicy.fail_fast().required(5) == 5
    assert FailurePolicy.best_effort().required(5) == 1
    assert FailurePolicy.quorum_of(0.8).required(5) == 4
    assert FailurePolicy.quorum_of(3).required(5) == 3
    assert FailurePolicy.quorum_of(0.8).satisfied(4, 5) and not FailurePolicy.quorum_of(0.8).satisfied(3, 5)


async def test_map_best_effort_returns_exceptions_in_order():
    ex = ParallelExecutor(retry=RetryPolicy(max_attempts=1), clock=ManualClock())

    async def f(x: int) -> int:
        if x == 2:
            raise ValueError("two")
        return x * 10

    vals = await ex.map(f, [1, 2, 3])
    assert vals[0] == 10 and vals[2] == 30 and isinstance(vals[1], ValueError)


async def test_map_fail_fast_cancels_siblings_and_raises():
    ex = ParallelExecutor(failure_policy=FailurePolicy.fail_fast(), retry=RetryPolicy(max_attempts=1), max_concurrency=1, clock=ManualClock())
    ran: list[int] = []

    async def f(x: int) -> int:
        ran.append(x)
        if x == 1:
            raise ValueError("first fails")
        return x

    with pytest.raises(ParallelExecutionError) as ei:
        await ex.map(f, [1, 2, 3])
    outs = ei.value.outcomes
    assert len(outs) == 3 and not outs[0].ok
    assert ran == [1]  # concurrency 1 + stop flag: later items never run
    assert all(o.cancelled for o in outs[1:])


async def test_quorum_policy():
    ex = ParallelExecutor(retry=RetryPolicy(max_attempts=1), clock=ManualClock())

    async def f(x: int) -> int:
        if x % 5 == 0:
            raise TransientError("fail")
        return x

    outs = await ex.map_outcomes(f, list(range(1, 6)), failure_policy=FailurePolicy.quorum_of(0.8))
    assert sum(o.ok for o in outs) == 4
    with pytest.raises(ParallelExecutionError):
        await ex.map_outcomes(f, [5, 10, 15, 4], failure_policy=FailurePolicy.quorum_of(0.8))


# ------------------------------------------------------------ dag / misc


async def test_dag_executes_in_waves_and_skips_dependents_of_failures():
    ex = ParallelExecutor(retry=RetryPolicy(max_attempts=1), clock=ManualClock())
    order: list[str] = []

    async def fn(node: str, deps: dict) -> str:
        order.append(node)
        if node == "a":
            raise ValueError("a failed")
        return node + "".join(sorted(deps.values()))

    results = await ex.dag({"a": [], "b": ["a"], "c": [], "d": ["c"]}, fn)
    assert not results["a"].ok and not results["b"].ok and "dependency failed" in (results["b"].error or "")
    assert results["c"].ok and results["d"].ok and results["d"].value == "dc"
    assert order.index("c") < order.index("d") and "b" not in order


async def test_per_task_timeout_counts_and_fails():
    ex = ParallelExecutor(per_task_timeout_s=0.02, retry=RetryPolicy(max_attempts=1))

    async def slow() -> None:
        await asyncio.Event().wait()  # never completes; the 20 ms timeout cancels it

    out = await ex.submit(slow)
    assert not out.ok and out.error_type == "TimeoutError" and ex.stats.timeouts == 1


async def test_hedged_request_wins_when_primary_is_slow():
    ex = ParallelExecutor(hedge_after_s=0.01, hedge_fraction=1.0, retry=RetryPolicy(max_attempts=1))
    calls = 0

    async def work() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()  # primary never answers; cancelled once the hedge wins
        return f"call-{calls}"

    out = await ex.submit(work)
    assert out.ok and out.hedged and out.value == "call-2"
    assert ex.stats.hedges == 1 and ex.stats.hedge_wins == 1


async def test_idempotency_key_memoises_and_progress_callback_fires():
    events: list[str] = []
    ex = ParallelExecutor(progress=lambda ev, out, done, total: events.append(ev), clock=ManualClock())
    calls = 0

    async def work(x: int) -> int:
        nonlocal calls
        calls += 1
        return x

    a = await ex.submit(work, 1, idempotency_key="k")
    b = await ex.submit(work, 1, idempotency_key="k")
    assert a.ok and b.ok and b.cached_idempotent and calls == 1
    assert ex.stats.idempotent_hits == 1 and events == ["done"]


async def test_deadline_propagation_blocks_new_attempts():
    ex = ParallelExecutor(remaining_time=lambda: 0.0, clock=ManualClock())

    async def work() -> int:
        return 1

    out = await ex.submit(work)
    assert not out.ok and out.error_type == "BudgetExceededError" and ex.stats.deadline_misses == 1


async def test_spawn_join_and_gather():
    ex = ParallelExecutor(clock=ManualClock())

    async def work(x: int) -> int:
        return x + 1

    outs = await ex.join([ex.spawn(work, 1), ex.spawn(work, 2)])
    assert [o.value for o in outs] == [2, 3]
    g = await ex.gather(lambda: work(5), lambda: work(6))
    assert [o.value for o in g] == [6, 7]
    snap = ex.snapshot()
    assert snap["submitted"] == 4 and snap["bulkhead"]["limits"]
