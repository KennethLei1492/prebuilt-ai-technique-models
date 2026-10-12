"""Speedup, cost/latency trade-off and concurrency recommendations.

Amdahl (1967): ``S = 1 / ((1 - p) + p / s)``; as ``s -> inf`` the bound is
``1 / (1 - p)``. In an LLM pipeline the serial part is the planner call plus
the aggregator call plus any dependent chain, so these are what bound the
wall-clock gain regardless of fan-out width. The estimator computes the
bound, the critical-path speedup of a concrete plan under a concurrency
limit, a cost-latency table for candidate widths, and a recommended width
from rate limits (Little's law).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .dag import critical_path, total_work, topological_waves


def amdahl_speedup(parallel_fraction: float, workers: float) -> float:
    """Speedup with fraction ``p`` parallelisable across ``workers`` units."""
    p = min(max(parallel_fraction, 0.0), 1.0)
    s = max(float(workers), 1.0)
    return 1.0 / ((1.0 - p) + p / s)


def amdahl_bound(serial_fraction: float) -> float:
    """Upper bound ``1 / (1 - p)``; infinite when nothing is serial."""
    f = min(max(serial_fraction, 0.0), 1.0)
    return math.inf if f == 0.0 else 1.0 / f


def parallel_efficiency(speedup: float, width: int) -> float:
    """Speedup divided by fan-out width (1.0 = perfect scaling)."""
    return speedup / max(1, width)


def waves_for(n_tasks: int, concurrency: int) -> int:
    """Number of sequential waves needed to run ``n_tasks`` at ``concurrency``."""
    if n_tasks <= 0:
        return 0
    return math.ceil(n_tasks / max(1, concurrency))


class SpeedupEstimate(BaseModel):
    """Result of :func:`estimate_graph` / :func:`estimate_fan_out`."""

    model_config = ConfigDict(extra="forbid")

    total_work: float = Field(description="Sum of all node latencies (sequential wall-clock)")
    critical_path: float = Field(description="Longest dependency chain under the concurrency limit")
    speedup: float = Field(description="total_work / critical_path")
    serial_fraction: float = Field(description="Share of total work that is inherently serial")
    amdahl_bound: float = Field(description="1 / serial_fraction (inf when nothing is serial)")
    efficiency: float = Field(description="speedup / width")
    width: int
    waves: int
    near_bound: bool = Field(description="True when speedup >= 80% of the Amdahl bound; widening fan-out is pointless")


def estimate_fan_out(
    parallel_tasks: int,
    *,
    serial_calls: int = 1,
    concurrency: int = 8,
    task_latency: float = 1.0,
    serial_latency: float = 1.0,
) -> SpeedupEstimate:
    """Estimate for the common shape: ``serial_calls`` serial steps plus one fan-out.

    Latencies are relative units (a call = 1.0 by default) so the estimate
    works before anything has been measured.
    """
    n = max(0, parallel_tasks)
    width = max(1, min(concurrency, n) if n else 1)
    waves = waves_for(n, width)
    work = serial_calls * serial_latency + n * task_latency
    crit = serial_calls * serial_latency + waves * task_latency
    speedup = work / crit if crit > 0 else 1.0
    serial_fraction = (serial_calls * serial_latency) / work if work > 0 else 1.0
    bound = amdahl_bound(serial_fraction)
    return SpeedupEstimate(
        total_work=work,
        critical_path=crit,
        speedup=round(speedup, 4),
        serial_fraction=round(serial_fraction, 4),
        amdahl_bound=bound if math.isfinite(bound) else 1e9,
        efficiency=round(parallel_efficiency(speedup, width), 4),
        width=width,
        waves=waves,
        near_bound=math.isfinite(bound) and speedup >= 0.8 * bound,
    )


def estimate_graph(
    graph: Mapping[str, Sequence[str]],
    durations: Mapping[str, float] | None = None,
    *,
    concurrency: int = 8,
) -> SpeedupEstimate:
    """Critical-path estimate for an arbitrary DAG with a concurrency ceiling.

    Each wave is widened to ``ceil(len(wave) / concurrency)`` sub-waves of its
    longest node, which is an upper bound on the scheduler's wall-clock.
    """
    if not graph:
        return estimate_fan_out(0, serial_calls=0, concurrency=concurrency)
    work = total_work(graph, durations)
    unconstrained, _path = critical_path(graph, durations)
    waves = topological_waves(graph)
    constrained = 0.0
    for wave in waves:
        longest = max((float(durations.get(n, 1.0)) if durations else 1.0) for n in wave)
        constrained += waves_for(len(wave), concurrency) * longest
    crit = max(unconstrained, constrained)
    speedup = work / crit if crit > 0 else 1.0
    width = max(1, min(concurrency, max(len(w) for w in waves)))
    serial = sum((float(durations.get(w[0], 1.0)) if durations else 1.0) for w in waves if len(w) == 1)
    serial_fraction = serial / work if work > 0 else 1.0
    bound = amdahl_bound(serial_fraction)
    return SpeedupEstimate(
        total_work=work,
        critical_path=crit,
        speedup=round(speedup, 4),
        serial_fraction=round(serial_fraction, 4),
        amdahl_bound=bound if math.isfinite(bound) else 1e9,
        efficiency=round(parallel_efficiency(speedup, width), 4),
        width=width,
        waves=len(waves),
        near_bound=math.isfinite(bound) and speedup >= 0.8 * bound,
    )


class TradeoffRow(BaseModel):
    """One candidate concurrency level in :func:`tradeoff_table`."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int
    latency_s: float
    cost_usd: float
    speedup: float
    efficiency: float
    waves: int


def tradeoff_table(
    n_tasks: int,
    *,
    task_latency_s: float,
    task_cost_usd: float,
    serial_latency_s: float = 0.0,
    serial_cost_usd: float = 0.0,
    candidates: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> list[TradeoffRow]:
    """Latency and cost for each candidate width. Cost is flat in width; latency falls in waves."""
    rows: list[TradeoffRow] = []
    seq_latency = serial_latency_s + n_tasks * task_latency_s
    cost = serial_cost_usd + n_tasks * task_cost_usd
    for c in candidates:
        if c < 1:
            continue
        waves = waves_for(n_tasks, c)
        latency = serial_latency_s + waves * task_latency_s
        speedup = seq_latency / latency if latency > 0 else 1.0
        rows.append(
            TradeoffRow(
                concurrency=c,
                latency_s=round(latency, 4),
                cost_usd=round(cost, 6),
                speedup=round(speedup, 4),
                efficiency=round(parallel_efficiency(speedup, min(c, max(1, n_tasks))), 4),
                waves=waves,
            )
        )
    return rows


def recommended_concurrency(
    *,
    requests_per_minute: Optional[int] = None,
    input_tokens_per_minute: Optional[int] = None,
    est_input_tokens_per_request: int = 1000,
    task_latency_s: float = 5.0,
    hard_cap: int = 8,
    n_tasks: Optional[int] = None,
) -> int:
    """Width that keeps a steady fan-out inside the rate limits (Little's law).

    Concurrency ~ throughput x latency. With 60 RPM and 5 s per call, more
    than 5 in flight only queues on the provider's token bucket.
    """
    limit = hard_cap
    if requests_per_minute:
        limit = min(limit, max(1, math.floor(requests_per_minute / 60.0 * task_latency_s)))
    if input_tokens_per_minute:
        rps_by_tokens = input_tokens_per_minute / 60.0 / max(1, est_input_tokens_per_request)
        limit = min(limit, max(1, math.floor(rps_by_tokens * task_latency_s)))
    if n_tasks is not None:
        limit = min(limit, max(1, n_tasks))
    return max(1, limit)


def token_multiplier(parallel_calls: int, baseline_calls: int = 1) -> float:
    """How many single-call baselines a fan-out spends (Anthropic: expect 4-15x for agents)."""
    return parallel_calls / max(1, baseline_calls)


def within_token_cap(parallel_calls: int, cap: float, baseline_calls: int = 1) -> bool:
    """True when the fan-out respects the configured token-multiplier cap."""
    return token_multiplier(parallel_calls, baseline_calls) <= cap
