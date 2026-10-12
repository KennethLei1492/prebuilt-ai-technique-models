"""Resilient asynchronous executor: the substrate every parallel pattern runs on.

Implements the ``Executor`` protocol from ``pipeline_core`` (``map``) plus
``gather``, ``submit``, ``spawn``/``join`` and ``dag`` (wave scheduling), on
top of the reliability kernel the literature prescribes:

* :class:`TokenBucket` / :class:`RateLimiter` - client-side RPM / ITPM / OTPM
  admission mirroring the provider's token-bucket enforcement.
* :class:`RetryPolicy` - capped exponential backoff with *full jitter*
  (Brooker 2015), at most 3 attempts (Google SRE), retry only on retryable
  error classes, ``retry_after_s`` honoured as a floor.
* :class:`RetryBudget` - retries capped at 10% of recent requests (SRE) so a
  fan-out that fails together cannot storm the provider.
* :class:`CircuitBreaker` - closed / open / half-open (Nygard 2018).
* :class:`Bulkhead` - per-resource concurrency pools. Re-entrant: work that
  calls ``map``/``gather``/``dag``/``submit`` on the same executor from inside
  a running task (a RAG fan-out inside a decomposition wave) is routed to a
  per-nesting-level pool (``<resource>@nested<k>``, tracked with a
  ``contextvars.ContextVar``) so it never waits on slots its parent holds;
  the rate limiter, retry budget, breaker and stats stay shared.
* :class:`FailurePolicy` - ``fail_fast`` | ``best_effort`` | ``quorum``.
* Hedged requests after a delay for idempotent work (Dean & Barroso 2013),
  idempotency keys with single-flight de-duplication, deadline propagation
  and cancellation of siblings on fail-fast.

Side effects: work submitted with ``idempotent=False`` is attempted exactly
once (no retry, no hedge, no memoisation) because a lost acknowledgement
after a committed write must not be re-sent. The breaker counts only
dependency-class failures (the retryable error classes), never caller errors
such as ``ValueError``/``KeyError``/``BudgetExceededError``.

The clock is injectable (:class:`ManualClock`) so backoff, bucket refill and
breaker recovery are unit-tested without real sleeps. Per-attempt timeouts
and hedge delays use the event loop (``asyncio.wait``) and therefore real
time; keep them small in tests.
"""
from __future__ import annotations

import asyncio
import contextvars
import math
import random
import time
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, Callable, Literal, Mapping, Optional, Protocol, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import BudgetExceededError, PipelineError

from .dag import Graph, normalize_graph, topological_waves, validate_dag

T = TypeVar("T")
R = TypeVar("R")

ProgressCallback = Callable[[str, "TaskOutcome | None", int, int], None]
AdmitFn = Callable[..., AsyncContextManager[Any]]

MAX_LATENCY_SAMPLES = 10_000
"""Latency samples kept per :class:`ExecutorStats` (older samples are dropped)."""

_NESTING: contextvars.ContextVar[tuple[tuple[int, int], ...]] = contextvars.ContextVar("parallel_executor_nesting", default=())
"""Per-task nesting depth for each executor (``id(executor) -> depth``), immutable for safe context copies."""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TransientError(PipelineError):
    """Raise from a task to signal a retryable failure (e.g. 429/5xx from a tool).

    ``retry_after_s`` (when known) becomes the backoff floor.
    """

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class CircuitOpenError(PipelineError):
    """The circuit breaker is open; the call was not attempted (fail fast)."""


class ParallelExecutionError(PipelineError):
    """A fan-out violated its failure policy. ``outcomes`` carries every branch result."""

    def __init__(self, message: str, outcomes: Sequence["TaskOutcome"]) -> None:
        super().__init__(message)
        self.outcomes = list(outcomes)


# --------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------


class Clock(Protocol):
    """Time source used for backoff, rate limiting and breaker recovery."""

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    """Production clock backed by ``time.monotonic`` and ``asyncio.sleep``."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class ManualClock:
    """Deterministic clock for tests: ``sleep`` advances time instantly and records the request."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += max(0.0, seconds)
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        self._now += max(0.0, seconds)


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """Capped exponential backoff with jitter and error classification.

    ``delay(attempt)`` for ``jitter="full"`` is
    ``uniform(0, min(max_delay_s, base_delay_s * 2**attempt))`` (Brooker's
    full jitter, which does the least total work). ``retry_after`` is a
    floor, never a ceiling, so a provider's hint is always honoured.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=20)
    base_delay_s: float = Field(default=0.5, ge=0.0)
    max_delay_s: float = Field(default=20.0, ge=0.0)
    jitter: Literal["full", "equal", "none"] = "full"
    retry_on: tuple[str, ...] = Field(
        default=("TransientError", "LLMError", "TimeoutError", "ConnectionError", "OSError"),
        description="Exception class names (any class in the MRO) that are retryable",
    )
    never_retry_on: tuple[str, ...] = Field(
        default=(
            "LLMRefusalError",
            "LLMTruncatedError",
            "StructuredOutputError",
            "BudgetExceededError",
            "PlanValidationError",
            "ConfigurationError",
            "CircuitOpenError",
            "ValueError",
            "TypeError",
            "KeyError",
        ),
        description="Exception class names that are never retried (they win over retry_on)",
    )

    def is_retryable(self, exc: BaseException) -> bool:
        names = {c.__name__ for c in type(exc).__mro__}
        if names & set(self.never_retry_on):
            return False
        return bool(names & set(self.retry_on))

    def delay(self, attempt: int, rng: random.Random | None = None, *, retry_after: float | None = None) -> float:
        """Backoff before retry number ``attempt`` (0-based)."""
        rng = rng or random
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** max(0, attempt)))
        if self.jitter == "full":
            d = rng.uniform(0.0, cap)
        elif self.jitter == "equal":
            d = cap / 2 + rng.uniform(0.0, cap / 2)
        else:
            d = cap
        if retry_after is not None:
            d = max(d, float(retry_after))
        return d


class RetryBudget:
    """Sliding-window retry budget: retries may not exceed ``ratio`` of requests.

    A floor of ``min_retries`` keeps single-request runs able to retry at
    all; the SRE 10% rule is about aggregate amplification, which the floor
    does not materially change.
    """

    def __init__(self, *, ratio: float = 0.10, min_retries: int = 3, window_s: float = 120.0, clock: Clock | None = None) -> None:
        self.ratio = ratio
        self.min_retries = min_retries
        self.window_s = window_s
        self.clock = clock or MonotonicClock()
        self._requests: deque[float] = deque()
        self._retries: deque[float] = deque()
        self.denied = 0

    def _prune(self) -> None:
        cutoff = self.clock.now() - self.window_s
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        while self._retries and self._retries[0] < cutoff:
            self._retries.popleft()

    def record_request(self) -> None:
        self._requests.append(self.clock.now())

    @property
    def requests(self) -> int:
        self._prune()
        return len(self._requests)

    @property
    def retries(self) -> int:
        self._prune()
        return len(self._retries)

    def allowance(self) -> int:
        return max(self.min_retries, math.floor(self.ratio * self.requests))

    def try_consume(self) -> bool:
        """Reserve one retry; False when the budget is exhausted."""
        self._prune()
        if len(self._retries) >= self.allowance():
            self.denied += 1
            return False
        self._retries.append(self.clock.now())
        return True


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class TokenBucket:
    """Continuous-refill token bucket (the provider's enforcement model)."""

    def __init__(self, rate_per_s: float, capacity: float, *, clock: Clock | None = None, initial: float | None = None) -> None:
        if rate_per_s <= 0 or capacity <= 0:
            raise ValueError("rate_per_s and capacity must be positive")
        self.rate = float(rate_per_s)
        self.capacity = float(capacity)
        self.clock = clock or MonotonicClock()
        self._tokens = float(capacity if initial is None else initial)
        self._last = self.clock.now()
        self.total_wait_s = 0.0

    def _refill(self) -> None:
        now = self.clock.now()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def available(self) -> float:
        self._refill()
        return self._tokens

    def time_until(self, n: float) -> float:
        """Seconds until ``n`` tokens are available (0 when they already are)."""
        self._refill()
        if self._tokens >= n:
            return 0.0
        return (n - self._tokens) / self.rate

    def try_acquire(self, n: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    async def acquire(self, n: float = 1.0) -> float:
        """Wait until ``n`` tokens are available, take them, return the time waited."""
        waited = 0.0
        n = min(n, self.capacity)  # a request larger than the bucket would never pass
        while not self.try_acquire(n):
            wait = self.time_until(n)
            waited += wait
            await self.clock.sleep(wait)
        self.total_wait_s += waited
        return waited

    def set_remaining(self, remaining: float) -> None:
        """Adopt the provider's view of remaining capacity (from rate-limit headers)."""
        self._refill()
        self._tokens = min(self.capacity, max(0.0, float(remaining)))


class RateLimiter:
    """RPM / ITPM / OTPM buckets behind one ``acquire`` call. Unset limits are not enforced."""

    def __init__(
        self,
        *,
        requests_per_minute: int | None = None,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
        burst_s: float = 5.0,
        clock: Clock | None = None,
    ) -> None:
        self.clock = clock or MonotonicClock()
        self.requests_per_minute = requests_per_minute or None
        self.input_tokens_per_minute = input_tokens_per_minute or None
        self.output_tokens_per_minute = output_tokens_per_minute or None
        self.requests = self._bucket(requests_per_minute, burst_s)
        self.input_tokens = self._bucket(input_tokens_per_minute, burst_s)
        self.output_tokens = self._bucket(output_tokens_per_minute, burst_s)
        self.total_wait_s = 0.0

    def _bucket(self, per_minute: int | None, burst_s: float) -> TokenBucket | None:
        if not per_minute:
            return None
        rate = per_minute / 60.0
        return TokenBucket(rate, max(1.0, rate * burst_s), clock=self.clock)

    @property
    def enabled(self) -> bool:
        return any(b is not None for b in (self.requests, self.input_tokens, self.output_tokens))

    def limits(self) -> dict[str, int | None]:
        """The per-minute limits this limiter enforces (None = not enforced)."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "input_tokens_per_minute": self.input_tokens_per_minute,
            "output_tokens_per_minute": self.output_tokens_per_minute,
        }

    async def acquire(self, *, input_tokens: int = 0, output_tokens: int = 0) -> float:
        waited = 0.0
        if self.requests is not None:
            waited += await self.requests.acquire(1.0)
        if self.input_tokens is not None and input_tokens > 0:
            waited += await self.input_tokens.acquire(float(input_tokens))
        if self.output_tokens is not None and output_tokens > 0:
            waited += await self.output_tokens.acquire(float(output_tokens))
        self.total_wait_s += waited
        return waited

    def feedback(self, *, remaining_requests: float | None = None, remaining_input_tokens: float | None = None, remaining_output_tokens: float | None = None) -> None:
        """Refresh buckets from ``anthropic-ratelimit-*-remaining`` style headers."""
        if self.requests is not None and remaining_requests is not None:
            self.requests.set_remaining(remaining_requests)
        if self.input_tokens is not None and remaining_input_tokens is not None:
            self.input_tokens.set_remaining(remaining_input_tokens)
        if self.output_tokens is not None and remaining_output_tokens is not None:
            self.output_tokens.set_remaining(remaining_output_tokens)


# --------------------------------------------------------------------------
# Circuit breaker and bulkhead
# --------------------------------------------------------------------------

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Closed / open / half-open breaker (Nygard).

    Opens after ``failure_threshold`` consecutive failures *or* when the
    failure rate over the last ``window`` calls reaches ``failure_rate``
    (only once the window holds at least ``min_calls`` samples). While open,
    ``allow()`` is False until ``recovery_timeout_s`` has elapsed; then one
    probe is admitted (half-open). A probe success closes the breaker, a
    probe failure re-opens it.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        failure_rate: float = 0.5,
        window: int = 20,
        min_calls: int = 10,
        recovery_timeout_s: float = 30.0,
        clock: Clock | None = None,
        name: str = "default",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.failure_rate = failure_rate
        self.window = window
        self.min_calls = min_calls
        self.recovery_timeout_s = recovery_timeout_s
        self.clock = clock or MonotonicClock()
        self.name = name
        self._state: BreakerState = "closed"
        self._opened_at = 0.0
        self._consecutive = 0
        self._recent: deque[bool] = deque(maxlen=window)
        self._probe_inflight = False
        self.trips = 0
        self.rejected = 0

    @property
    def state(self) -> BreakerState:
        if self._state == "open" and self.clock.now() - self._opened_at >= self.recovery_timeout_s:
            self._state = "half_open"
            self._probe_inflight = False
        return self._state

    def allow(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open" and not self._probe_inflight:
            self._probe_inflight = True
            return True
        self.rejected += 1
        return False

    def record_success(self) -> None:
        self._consecutive = 0
        self._recent.append(True)
        if self._state == "half_open":
            self._state = "closed"
            self._probe_inflight = False
            self._recent.clear()

    def record_failure(self) -> None:
        if self._state == "open" and self.clock.now() - self._opened_at < self.recovery_timeout_s:
            # late failures from calls that were in flight when the breaker tripped must not
            # re-trip it (that would extend the open window and inflate ``trips``)
            self._recent.append(False)
            return
        self._consecutive += 1
        self._recent.append(False)
        if self.state == "half_open":
            self._trip()
            return
        if self._consecutive >= self.failure_threshold:
            self._trip()
            return
        if len(self._recent) >= self.min_calls:
            rate = sum(1 for ok in self._recent if not ok) / len(self._recent)
            if rate >= self.failure_rate:
                self._trip()

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = self.clock.now()
        self._probe_inflight = False
        self._consecutive = 0
        self.trips += 1

    def reset(self) -> None:
        self._state = "closed"
        self._consecutive = 0
        self._recent.clear()
        self._probe_inflight = False

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "trips": self.trips, "rejected": self.rejected}


NESTED_SEP = "@nested"
"""Suffix separator of the bulkhead pool used by nested fan-outs (``llm@nested1``)."""


class Bulkhead:
    """Per-resource concurrency pools so one slow dependency cannot starve the rest.

    A pool named ``<resource>@nested<k>`` (used by fan-outs nested *k* levels
    deep inside work already holding a slot) inherits ``<resource>``'s limit.
    """

    def __init__(self, limits: Mapping[str, int] | None = None, *, default_limit: int = 8) -> None:
        self.limits = dict(limits or {})
        self.default_limit = default_limit
        self._sems: dict[str, asyncio.Semaphore] = {}
        self.waits: dict[str, int] = {}
        self.max_inflight: dict[str, int] = {}
        self._inflight: dict[str, int] = {}

    def limit_for(self, resource: str) -> int:
        if resource in self.limits:
            return max(1, self.limits[resource])
        base = resource.split(NESTED_SEP, 1)[0]
        return max(1, self.limits.get(base, self.default_limit))

    def _sem(self, resource: str) -> asyncio.Semaphore:
        sem = self._sems.get(resource)
        if sem is None:
            sem = asyncio.Semaphore(self.limit_for(resource))
            self._sems[resource] = sem
        return sem

    @asynccontextmanager
    async def slot(self, resource: str = "default") -> AsyncIterator[None]:
        sem = self._sem(resource)
        if sem.locked():
            self.waits[resource] = self.waits.get(resource, 0) + 1
        async with sem:
            self._inflight[resource] = self._inflight.get(resource, 0) + 1
            self.max_inflight[resource] = max(self.max_inflight.get(resource, 0), self._inflight[resource])
            try:
                yield
            finally:
                self._inflight[resource] -= 1

    def snapshot(self) -> dict[str, Any]:
        return {"limits": {r: self.limit_for(r) for r in set(self.limits) | set(self._sems)}, "waits": dict(self.waits), "max_inflight": dict(self.max_inflight)}


# --------------------------------------------------------------------------
# Failure policy and outcomes
# --------------------------------------------------------------------------


class FailurePolicy(BaseModel):
    """What a fan-out does when branches fail.

    * ``fail_fast`` - cancel siblings on the first failure and raise.
    * ``best_effort`` - run everything, return successes and failures.
    * ``quorum`` - run everything; raise unless at least ``quorum`` succeeded
      (a fraction in (0, 1] or an absolute count when > 1).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["fail_fast", "best_effort", "quorum"] = "best_effort"
    quorum: float = Field(default=0.8, gt=0.0)

    @classmethod
    def fail_fast(cls) -> "FailurePolicy":
        return cls(mode="fail_fast")

    @classmethod
    def best_effort(cls) -> "FailurePolicy":
        return cls(mode="best_effort")

    @classmethod
    def quorum_of(cls, n: float) -> "FailurePolicy":
        return cls(mode="quorum", quorum=n)

    def required(self, total: int) -> int:
        if self.mode == "fail_fast":
            return total
        if self.mode == "best_effort":
            return 1 if total else 0
        if self.quorum > 1:
            return min(total, int(math.ceil(self.quorum)))
        return min(total, int(math.ceil(self.quorum * total)))

    def satisfied(self, ok: int, total: int) -> bool:
        if total == 0:
            return True
        return ok >= self.required(total)


class TaskOutcome(BaseModel):
    """Result of one unit of work run through the executor."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    key: str
    index: int = 0
    ok: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attempts: int = 0
    retries: int = 0
    latency_s: float = 0.0
    wait_s: float = Field(default=0.0, description="Admission wait (cache gate, rate limiter) before the attempt started; excluded from latency_s")
    hedged: bool = False
    cancelled: bool = False
    cached_idempotent: bool = False
    exception: Optional[BaseException] = Field(default=None, exclude=True, repr=False)

    @classmethod
    def failure(cls, key: str, exc: BaseException, *, index: int = 0, attempts: int = 0, retries: int = 0, latency_s: float = 0.0, wait_s: float = 0.0, cancelled: bool = False) -> "TaskOutcome":
        return cls(
            key=key,
            index=index,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            attempts=attempts,
            retries=retries,
            latency_s=latency_s,
            wait_s=wait_s,
            cancelled=cancelled,
            exception=exc,
        )


class ExecutorStats(BaseModel):
    """Counters exposed for evaluation and tracing.

    The executor keeps one lifetime instance (``executor.stats``); callers that
    need per-run attribution pass their own instance as ``submit(...,
    stats=run_stats)`` and every counter is incremented on both.
    ``latencies_s`` is bounded to :data:`MAX_LATENCY_SAMPLES`.
    """

    model_config = ConfigDict(extra="forbid")

    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    timeouts: int = 0
    cancelled: int = 0
    hedges: int = 0
    hedge_wins: int = 0
    rate_limit_wait_s: float = 0.0
    breaker_rejections: int = 0
    breaker_trips: int = Field(default=0, description="Times a failure recorded through this executor opened the breaker")
    idempotent_hits: int = 0
    deadline_misses: int = 0
    latencies_s: list[float] = Field(default_factory=list)

    def record_latency(self, latency_s: float) -> None:
        self.latencies_s.append(latency_s)
        if len(self.latencies_s) > MAX_LATENCY_SAMPLES:
            del self.latencies_s[: len(self.latencies_s) - MAX_LATENCY_SAMPLES]


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------


class ParallelExecutor:
    """Bounded, rate-limited, retrying, breaker-protected async executor.

    Parameters mirror :class:`~parallelization.config.ParallelizationConfig`
    but the executor is usable standalone (see :meth:`from_config`).
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 8,
        rate_limiter: RateLimiter | None = None,
        retry: RetryPolicy | None = None,
        retry_budget: RetryBudget | None = None,
        breaker: CircuitBreaker | None = None,
        bulkhead: Bulkhead | None = None,
        failure_policy: FailurePolicy | None = None,
        per_task_timeout_s: float | None = None,
        hedge_after_s: float | None = None,
        hedge_fraction: float = 0.05,
        clock: Clock | None = None,
        rng: random.Random | None = None,
        progress: ProgressCallback | None = None,
        remaining_time: Callable[[], float | None] | None = None,
        name: str = "parallel-executor",
        nested_bulkheads: bool = True,
    ) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.nested_bulkheads = nested_bulkheads
        self.clock: Clock = clock or MonotonicClock()
        self.rate_limiter = rate_limiter
        self.retry = retry if retry is not None else RetryPolicy()
        self.retry_budget = retry_budget or RetryBudget(clock=self.clock)
        self.breaker = breaker
        self.bulkhead = bulkhead or Bulkhead(default_limit=self.max_concurrency)
        self.failure_policy = failure_policy or FailurePolicy()
        self.per_task_timeout_s = per_task_timeout_s
        self.hedge_after_s = hedge_after_s
        self.hedge_fraction = hedge_fraction
        self.rng = rng or random.Random()
        self.progress = progress
        self.remaining_time = remaining_time
        self.name = name
        self.stats = ExecutorStats()
        self._memo: dict[str, TaskOutcome] = {}
        self._inflight: dict[str, asyncio.Future[TaskOutcome]] = {}
        self._counter = 0

    # ------------------------------------------------------------- factory
    @classmethod
    def from_config(cls, config: Any, *, clock: Clock | None = None, remaining_time: Callable[[], float | None] | None = None, max_concurrency: int | None = None) -> "ParallelExecutor":
        """Build from a :class:`ParallelizationConfig` (duck-typed to avoid an import cycle)."""
        clock = clock or MonotonicClock()
        limiter = RateLimiter(
            requests_per_minute=config.requests_per_minute,
            input_tokens_per_minute=config.input_tokens_per_minute,
            output_tokens_per_minute=config.output_tokens_per_minute,
            burst_s=config.rate_limit_burst_s,
            clock=clock,
        )
        return cls(
            max_concurrency=max_concurrency or config.max_concurrency,
            rate_limiter=limiter if limiter.enabled else None,
            retry=RetryPolicy(max_attempts=config.retry_max_attempts, base_delay_s=config.retry_base_delay_s, max_delay_s=config.retry_max_delay_s),
            retry_budget=RetryBudget(ratio=config.retry_budget_ratio, min_retries=config.retry_budget_min, window_s=config.retry_budget_window_s, clock=clock),
            breaker=CircuitBreaker(
                failure_threshold=config.breaker_failure_threshold,
                failure_rate=config.breaker_failure_rate,
                window=config.breaker_window,
                min_calls=min(10, config.breaker_window),
                recovery_timeout_s=config.breaker_recovery_s,
                clock=clock,
                name="llm",
            ),
            failure_policy=FailurePolicy(mode="best_effort", quorum=config.quorum_fraction),
            per_task_timeout_s=config.per_task_timeout_s,
            hedge_after_s=config.hedge_after_s,
            hedge_fraction=config.hedge_fraction,
            clock=clock,
            remaining_time=remaining_time,
        )

    # --------------------------------------------------------------- stats
    def reset_stats(self) -> None:
        self.stats = ExecutorStats()
        self._memo.clear()

    def snapshot(self) -> dict[str, Any]:
        """Serialisable view of counters, breaker and bulkhead state."""
        d = self.stats.model_dump()
        d["breaker"] = self.breaker.snapshot() if self.breaker else None
        d["bulkhead"] = self.bulkhead.snapshot()
        d["retry_budget_denied"] = self.retry_budget.denied
        d["rate_limited"] = bool(self.rate_limiter and self.rate_limiter.enabled)
        return d

    def _emit(self, event: str, outcome: TaskOutcome | None, done: int, total: int) -> None:
        if self.progress is not None:
            try:
                self.progress(event, outcome, done, total)
            except Exception:  # noqa: BLE001 - callbacks must not break execution
                pass

    def _inc(self, name: str, n: float = 1, sink: ExecutorStats | None = None) -> None:
        """Increment a counter on the lifetime stats and on the caller's per-run sink."""
        setattr(self.stats, name, getattr(self.stats, name) + n)
        if sink is not None:
            setattr(sink, name, getattr(sink, name) + n)

    def _latency(self, latency_s: float, sink: ExecutorStats | None) -> None:
        self.stats.record_latency(latency_s)
        if sink is not None:
            sink.record_latency(latency_s)

    # ------------------------------------------------------------ nesting
    def nesting_depth(self) -> int:
        """How many slots of this executor the current task is already running inside (0 at top level)."""
        for ex_id, depth in _NESTING.get():
            if ex_id == id(self):
                return depth
        return 0

    def _resource_for(self, resource: str) -> str:
        """Route nested work to its own bulkhead pool so it cannot wait on slots its parent holds."""
        depth = self.nesting_depth()
        if depth == 0 or not self.nested_bulkheads:
            return resource
        return f"{resource}{NESTED_SEP}{depth}"

    def _enter_nesting(self) -> contextvars.Token[tuple[tuple[int, int], ...]]:
        depth = self.nesting_depth() + 1
        rest = tuple((k, v) for k, v in _NESTING.get() if k != id(self))
        return _NESTING.set(rest + ((id(self), depth),))

    # -------------------------------------------------------------- submit
    async def submit(
        self,
        fn: Callable[..., Awaitable[R]],
        *args: Any,
        key: str | None = None,
        index: int = 0,
        idempotency_key: str | None = None,
        idempotent: bool = True,
        resource: str = "default",
        timeout_s: float | None = None,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        criticality: Literal["primary", "sheddable"] = "primary",
        admit: AdmitFn | None = None,
        stats: ExecutorStats | None = None,
        **kwargs: Any,
    ) -> TaskOutcome:
        """Run ``fn(*args, **kwargs)`` once through the whole resilience stack.

        Never raises for task failures: the returned :class:`TaskOutcome`
        carries ``ok=False`` and the exception.

        * ``idempotent=False`` marks side-effecting work: it is attempted once
          (no retry, no hedge) and never memoised, even with an
          ``idempotency_key``.
        * ``idempotency_key`` memoises successful idempotent outcomes and
          de-duplicates concurrent identical work (single flight).
        * ``admit(*args, **kwargs)`` returns an async context manager entered
          **before** the rate limiter, bulkhead and per-attempt timeout (the
          warm-then-burst cache gate uses it); its wait is reported in
          ``TaskOutcome.wait_s`` and excluded from ``latency_s``.
        * ``stats`` is a per-run :class:`ExecutorStats` that receives the same
          increments as ``executor.stats`` (per-run attribution on a shared
          executor).
        """
        self._counter += 1
        key = key or f"task-{self._counter}"
        if idempotency_key is not None:
            hit = self._memo.get(idempotency_key)
            if hit is not None and hit.ok:
                self._inc("idempotent_hits", sink=stats)
                return hit.model_copy(update={"key": key, "index": index, "cached_idempotent": True})
            fut = self._inflight.get(idempotency_key)
            if fut is not None:
                out = await asyncio.shield(fut)
                self._inc("idempotent_hits", sink=stats)
                return out.model_copy(update={"key": key, "index": index, "cached_idempotent": True})
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._inflight[idempotency_key] = fut
            try:
                out = await self._execute(fn, args, kwargs, key, index, idempotent, resource, timeout_s, estimated_input_tokens, estimated_output_tokens, criticality, admit, stats)
                if out.ok and idempotent:
                    self._memo[idempotency_key] = out
                if not fut.done():
                    fut.set_result(out)
                return out
            except BaseException as exc:  # pragma: no cover - cancellation path
                if not fut.done():
                    fut.set_exception(exc)
                raise
            finally:
                self._inflight.pop(idempotency_key, None)
        return await self._execute(fn, args, kwargs, key, index, idempotent, resource, timeout_s, estimated_input_tokens, estimated_output_tokens, criticality, admit, stats)

    async def _execute(
        self,
        fn: Callable[..., Awaitable[R]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        key: str,
        index: int,
        idempotent: bool,
        resource: str,
        timeout_s: float | None,
        est_in: int,
        est_out: int,
        criticality: str,
        admit: AdmitFn | None = None,
        sink: ExecutorStats | None = None,
    ) -> TaskOutcome:
        self._inc("submitted", sink=sink)
        self.retry_budget.record_request()
        wall0 = time.perf_counter()
        wait_s = 0.0
        attempts = 0
        retries = 0
        last_exc: BaseException | None = None
        while True:
            attempts += 1
            # deadline propagation
            remaining = self.remaining_time() if self.remaining_time else None
            if remaining is not None and remaining <= 0.0:
                self._inc("deadline_misses", sink=sink)
                last_exc = BudgetExceededError("deadline exhausted before attempt")
                break
            if self.breaker is not None and not self.breaker.allow():
                self._inc("breaker_rejections", sink=sink)
                last_exc = CircuitOpenError(f"circuit '{self.breaker.name}' is open")
                break
            hedged = False
            try:
                async with AsyncExitStack() as stack:
                    # admission (cache gate, then rate limiter) happens before the timeout clock starts
                    if admit is not None:
                        t0 = time.perf_counter()
                        await stack.enter_async_context(admit(*args, **kwargs))
                        wait_s += time.perf_counter() - t0
                    if self.rate_limiter is not None:
                        t0 = time.perf_counter()
                        waited = await self.rate_limiter.acquire(input_tokens=est_in, output_tokens=est_out)
                        wait_s += time.perf_counter() - t0
                        self._inc("rate_limit_wait_s", waited, sink=sink)
                    remaining = self.remaining_time() if self.remaining_time else None
                    if remaining is not None and remaining <= 0.0:
                        self._inc("deadline_misses", sink=sink)
                        raise BudgetExceededError("deadline exhausted while waiting for admission")
                    attempt_timeout = timeout_s if timeout_s is not None else self.per_task_timeout_s
                    if remaining is not None:
                        attempt_timeout = remaining if attempt_timeout is None else min(attempt_timeout, remaining)
                    async with self.bulkhead.slot(self._resource_for(resource)):
                        # work started inside this slot is one nesting level deeper: its own
                        # submit/map/gather/dag calls use a separate pool (no self-deadlock)
                        token = self._enter_nesting()
                        try:
                            value, hedged = await self._attempt(fn, args, kwargs, attempt_timeout, idempotent and criticality == "primary", sink)
                        finally:
                            _NESTING.reset(token)
                if self.breaker is not None:
                    self.breaker.record_success()
                latency = max(0.0, time.perf_counter() - wall0 - wait_s)
                self._inc("succeeded", sink=sink)
                self._latency(latency, sink)
                out = TaskOutcome(key=key, index=index, ok=True, value=value, attempts=attempts, retries=retries, latency_s=latency, wait_s=wait_s, hedged=hedged)
                self._emit("done", out, self.stats.succeeded + self.stats.failed, self.stats.submitted)
                return out
            except asyncio.CancelledError:
                self._inc("cancelled", sink=sink)
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._inc("timeouts", sink=sink)
                retryable = self.retry.is_retryable(exc)
                if self.breaker is not None and retryable:
                    # only dependency-class failures count against the breaker; caller errors
                    # (ValueError, KeyError, budget, schema) must not open it
                    trips_before = self.breaker.trips
                    self.breaker.record_failure()
                    if self.breaker.trips > trips_before:
                        self._inc("breaker_trips", sink=sink)
                if attempts >= self.retry.max_attempts or not idempotent or not retryable:
                    break
                if not self.retry_budget.try_consume():
                    break
                retries += 1
                self._inc("retries", sink=sink)
                delay = self.retry.delay(attempts - 1, self.rng, retry_after=getattr(exc, "retry_after_s", None))
                self._emit("retry", None, self.stats.succeeded + self.stats.failed, self.stats.submitted)
                await self.clock.sleep(delay)
        latency = max(0.0, time.perf_counter() - wall0 - wait_s)
        self._inc("failed", sink=sink)
        self._latency(latency, sink)
        assert last_exc is not None
        out = TaskOutcome.failure(key, last_exc, index=index, attempts=attempts, retries=retries, latency_s=latency, wait_s=wait_s)
        self._emit("failed", out, self.stats.succeeded + self.stats.failed, self.stats.submitted)
        return out

    def _hedge_allowed(self) -> bool:
        if self.hedge_after_s is None:
            return False
        cap = max(1, math.ceil(self.hedge_fraction * max(1, self.stats.submitted)))
        return self.stats.hedges < cap

    async def _attempt(
        self,
        fn: Callable[..., Awaitable[R]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout_s: float | None,
        may_hedge: bool,
        sink: ExecutorStats | None = None,
    ) -> tuple[R, bool]:
        """One attempt with optional per-attempt timeout and hedged duplicate.

        Once a hedge is issued the attempt returns the **first successful**
        response (Dean & Barroso): a primary that fails after the hedge was
        sent does not discard the duplicate's work. The attempt fails only
        when every copy has failed or the timeout elapsed.
        """
        if not (may_hedge and self._hedge_allowed()):
            if timeout_s is None:
                return await fn(*args, **kwargs), False
            return await asyncio.wait_for(fn(*args, **kwargs), timeout_s), False

        primary = asyncio.ensure_future(fn(*args, **kwargs))
        hedge_wait = self.hedge_after_s if timeout_s is None else min(self.hedge_after_s or 0.0, timeout_s)
        done, _pending = await asyncio.wait({primary}, timeout=hedge_wait)
        if done:
            return primary.result(), False
        self._inc("hedges", sink=sink)
        duplicate = asyncio.ensure_future(fn(*args, **kwargs))
        deadline = None if timeout_s is None else time.perf_counter() + max(0.0, timeout_s - (hedge_wait or 0.0))
        pending: set[asyncio.Future[Any]] = {primary, duplicate}
        errors: list[BaseException] = []
        while pending:
            timeout = None if deadline is None else max(0.0, deadline - time.perf_counter())
            done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                for p in pending:
                    p.cancel()
                raise asyncio.TimeoutError(f"attempt exceeded {timeout_s}s including hedge")
            for fut in done:
                if fut.cancelled():
                    continue
                exc = fut.exception()
                if exc is None:
                    for p in pending:
                        p.cancel()
                    if fut is duplicate:
                        self._inc("hedge_wins", sink=sink)
                    return fut.result(), True
                errors.append(exc)
        raise errors[-1]

    # ----------------------------------------------------------------- map
    async def map_outcomes(
        self,
        fn: Callable[[T], Awaitable[R]],
        items: Sequence[T],
        *,
        concurrency: int | None = None,
        keys: Sequence[str] | None = None,
        failure_policy: FailurePolicy | None = None,
        **submit_kwargs: Any,
    ) -> list[TaskOutcome]:
        """Run ``fn`` over ``items`` concurrently; return one outcome per item in order.

        Raises :class:`ParallelExecutionError` (with all outcomes attached)
        when the failure policy is violated.
        """
        policy = failure_policy or self.failure_policy
        width = max(1, min(concurrency or self.max_concurrency, self.max_concurrency))
        sem = asyncio.Semaphore(width)
        total = len(items)
        outcomes: list[TaskOutcome | None] = [None] * total
        stop = asyncio.Event()
        tasks: list[asyncio.Task[None]] = []
        started: set[int] = set()
        sink: ExecutorStats | None = submit_kwargs.get("stats")

        async def run_one(i: int, item: T) -> None:
            async with sem:
                if stop.is_set():
                    outcomes[i] = TaskOutcome.failure(_key(i), asyncio.CancelledError("cancelled by fail_fast"), index=i, cancelled=True)
                    self._inc("cancelled", sink=sink)
                    return
                started.add(i)
                out = await self.submit(fn, item, key=_key(i), index=i, **submit_kwargs)
                outcomes[i] = out
                if not out.ok and policy.mode == "fail_fast" and not stop.is_set():
                    stop.set()
                    for t in tasks:
                        if t is not asyncio.current_task():
                            t.cancel()

        def _key(i: int) -> str:
            return keys[i] if keys is not None else f"item-{i}"

        tasks = [asyncio.create_task(run_one(i, it)) for i, it in enumerate(items)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if outcomes[i] is None:
                exc = r if isinstance(r, BaseException) else asyncio.CancelledError("cancelled")
                outcomes[i] = TaskOutcome.failure(_key(i), exc, index=i, cancelled=isinstance(exc, asyncio.CancelledError))
                if isinstance(exc, asyncio.CancelledError) and i not in started:
                    self._inc("cancelled", sink=sink)  # in-flight cancellations were already counted by _execute
        final = [o for o in outcomes if o is not None]
        ok = sum(1 for o in final if o.ok)
        if policy.mode == "fail_fast" and ok < total:
            raise ParallelExecutionError(f"fail_fast: {total - ok} of {total} branches failed", final)
        if policy.mode == "quorum" and not policy.satisfied(ok, total):
            raise ParallelExecutionError(f"quorum not met: {ok}/{total} succeeded, {policy.required(total)} required", final)
        return final

    async def map(self, fn: Callable[[T], Awaitable[R]], items: Sequence[T], *, concurrency: int | None = None) -> list[Any]:
        """``Executor`` protocol: values in input order; failed items hold the exception object.

        Uses the executor's failure policy; ``fail_fast`` and unmet ``quorum``
        raise :class:`ParallelExecutionError`.
        """
        outs = await self.map_outcomes(fn, items, concurrency=concurrency)
        return [o.value if o.ok else o.exception for o in outs]

    async def gather(self, *fns: Callable[[], Awaitable[Any]], failure_policy: FailurePolicy | None = None, concurrency: int | None = None) -> list[TaskOutcome]:
        """Run zero-argument coroutine factories concurrently."""

        async def call(f: Callable[[], Awaitable[Any]]) -> Any:
            return await f()

        return await self.map_outcomes(call, list(fns), failure_policy=failure_policy, concurrency=concurrency)

    # ---------------------------------------------------------- spawn/join
    def spawn(self, fn: Callable[..., Awaitable[R]], *args: Any, **submit_kwargs: Any) -> "asyncio.Task[TaskOutcome]":
        """Start a unit of work immediately (orchestrator-workers / APR style)."""
        return asyncio.create_task(self.submit(fn, *args, **submit_kwargs))

    async def join(self, handles: Sequence["asyncio.Task[TaskOutcome]"], *, failure_policy: FailurePolicy | None = None) -> list[TaskOutcome]:
        """Wait for spawned work; apply a failure policy to the set."""
        policy = failure_policy or self.failure_policy
        outs = list(await asyncio.gather(*handles))
        ok = sum(1 for o in outs if o.ok)
        if policy.mode == "fail_fast" and ok < len(outs):
            raise ParallelExecutionError(f"fail_fast: {len(outs) - ok} of {len(outs)} spawned tasks failed", outs)
        if policy.mode == "quorum" and not policy.satisfied(ok, len(outs)):
            raise ParallelExecutionError("quorum not met", outs)
        return outs

    # ----------------------------------------------------------------- dag
    async def dag(
        self,
        graph: Graph,
        fn: Callable[[str, dict[str, Any]], Awaitable[Any]],
        *,
        failure_policy: FailurePolicy | None = None,
        concurrency: int | None = None,
        **submit_kwargs: Any,
    ) -> dict[str, TaskOutcome]:
        """Execute a dependency DAG in topological waves.

        ``fn(node_id, dependency_values)`` receives the resolved outputs of the
        node's dependencies (LLMCompiler placeholder binding at dispatch time).
        Nodes whose dependencies failed are skipped with ``ok=False``. Under
        ``fail_fast`` the first failure cancels its in-wave siblings and no
        further wave is scheduled (cancellation propagation). ``admit`` /
        ``stats`` and other ``submit`` keywords pass through; ``admit`` is
        called as ``admit(node_id, dependency_values)``.
        """
        g = normalize_graph(graph)
        validate_dag(g)
        policy = failure_policy or self.failure_policy
        results: dict[str, TaskOutcome] = {}
        total = len(g)
        admit: AdmitFn | None = submit_kwargs.pop("admit", None)

        def _abort() -> ParallelExecutionError:
            for node in g:
                if node not in results:
                    results[node] = TaskOutcome.failure(node, asyncio.CancelledError("skipped by fail_fast"), cancelled=True)
            return ParallelExecutionError("fail_fast: a DAG node failed", list(results.values()))

        for wave in topological_waves(g):
            runnable: list[str] = []
            for node in wave:
                failed_deps = [d for d in g[node] if not results[d].ok]
                if failed_deps:
                    results[node] = TaskOutcome.failure(node, PipelineError(f"dependency failed: {', '.join(failed_deps)}"), index=len(results))
                else:
                    runnable.append(node)
            if runnable:

                def deps_of(node: str) -> dict[str, Any]:
                    return {d: results[d].value for d in g[node]}

                async def call(node: str) -> Any:
                    return await fn(node, deps_of(node))

                wave_kwargs = dict(submit_kwargs)
                if admit is not None:
                    wave_kwargs["admit"] = lambda node: admit(node, deps_of(node))
                wave_policy = FailurePolicy.fail_fast() if policy.mode == "fail_fast" else FailurePolicy.best_effort()
                try:
                    outs = await self.map_outcomes(call, runnable, keys=runnable, concurrency=concurrency, failure_policy=wave_policy, **wave_kwargs)
                except ParallelExecutionError as exc:
                    for o in exc.outcomes:
                        results[o.key] = o
                    raise _abort() from None
                for o in outs:
                    results[o.key] = o
            ok_so_far = sum(1 for o in results.values() if o.ok)
            if policy.mode == "fail_fast" and ok_so_far < len(results):
                raise _abort()
        ok = sum(1 for o in results.values() if o.ok)
        if policy.mode == "quorum" and not policy.satisfied(ok, total):
            raise ParallelExecutionError(f"quorum not met: {ok}/{total} nodes succeeded", list(results.values()))
        return results


__all__ = [
    "Bulkhead",
    "CircuitBreaker",
    "CircuitOpenError",
    "Clock",
    "ExecutorStats",
    "FailurePolicy",
    "ManualClock",
    "MonotonicClock",
    "NESTED_SEP",
    "ParallelExecutionError",
    "ParallelExecutor",
    "RateLimiter",
    "RetryBudget",
    "RetryPolicy",
    "TaskOutcome",
    "TokenBucket",
    "TransientError",
]
