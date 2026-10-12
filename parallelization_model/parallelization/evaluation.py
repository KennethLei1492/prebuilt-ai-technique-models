"""Metrics and an offline evaluation harness for the parallelization model.

The dossier's evaluation metrics (speedup, parallel efficiency, critical path
versus the Amdahl bound, p50/p95 latency, retry rate, breaker trips, cache
hit rate, partial-failure rate, token multiplier) are computed from a
:class:`TaskResult` produced by :class:`ParallelizationModel` and from the
executor counters it attaches to ``result.metadata["executor"]``. Everything
here is pure Python so it can run in CI on ``MockLLM`` results.
"""
from __future__ import annotations

import math
import time
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Constraints, Status, TaskRequest, TaskResult, run_sync

from .estimator import amdahl_bound, parallel_efficiency


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in [0, 100]); 0.0 for an empty sequence."""
    if not values:
        return 0.0
    xs = sorted(values)
    q = min(max(q, 0.0), 100.0)
    k = max(0, min(len(xs) - 1, math.ceil(q / 100.0 * len(xs)) - 1))
    return float(xs[k])


class RunMetrics(BaseModel):
    """Metrics for one ``run()`` of the parallelization model."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = ""
    status: str = ""
    confidence: float = 0.0
    llm_calls: int = 0
    expected_calls: int = 0
    expected_speedup: float = 1.0
    measured_speedup: float = Field(default=1.0, description="sum of call latencies / wall-clock of the pattern")
    utilisation: float = Field(default=0.0, description="total_work / (wall x concurrency) - share of the fan-out capacity actually busy")
    serial_fraction: float = Field(default=1.0, description="Share of the plan's expected calls on the serial critical path (plan.serial_fraction)")
    amdahl_bound: float = Field(default=1.0, description="1 / serial_fraction from the plan's call structure (1e9 when nothing is serial)")
    efficiency: float = Field(default=0.0, description="measured speedup / plan concurrency")
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    wall_s: float = 0.0
    total_work_s: float = 0.0
    branches: int = 0
    failed_branches: int = 0
    partial_failure_rate: float = 0.0
    submitted: int = Field(default=0, description="Units of work submitted to the executor by this run")
    retries: int = 0
    retry_rate: float = Field(default=0.0, description="retries / submitted (target < 10%)")
    timeouts: int = 0
    hedges: int = 0
    breaker_rejections: int = 0
    breaker_trips: int = Field(default=0, description="Times this run's failures opened the circuit breaker")
    rate_limit_wait_s: float = 0.0
    cache_hit_ratio: float = 0.0
    cache_warmups: int = 0
    cache_waits: int = 0
    token_multiplier: float = Field(default=1.0, description="LLM calls relative to a single-call baseline")
    cost_usd: float = 0.0


def metrics_from_result(result: TaskResult) -> RunMetrics:
    """Compute :class:`RunMetrics` from a parallelization :class:`TaskResult`."""
    out = result.output
    data: dict[str, Any] = out.model_dump() if hasattr(out, "model_dump") else (out if isinstance(out, dict) else {})
    plan = result.metadata.get("plan", {}) or {}
    ex = result.metadata.get("executor", {}) or {}
    branches = data.get("branches", []) or []
    lat = [float(b.get("latency_s", 0.0)) for b in branches if b.get("ok")] or list(ex.get("latencies_s", []))
    submitted = int(ex.get("submitted", 0) or 0)
    retries = int(ex.get("retries", 0) or 0)
    n_branches = len(branches)
    failed = len(data.get("failed_branches", []) or [])
    concurrency = int(plan.get("concurrency", 1) or 1)
    wall = float(data.get("wall_s", 0.0) or 0.0)
    work = float(data.get("total_work_s", 0.0) or 0.0)
    expected_calls = int(data.get("expected_calls", plan.get("expected_calls", 1)) or 1)
    measured = float(data.get("measured_speedup", 1.0) or 1.0)
    if "serial_fraction" in plan:
        serial_fraction = float(plan["serial_fraction"])
    else:  # plans serialised before serial_fraction existed: assume one serial call
        serial_fraction = 1.0 / expected_calls if expected_calls > 0 else 1.0
    bound = amdahl_bound(serial_fraction) if serial_fraction > 0 else math.inf
    return RunMetrics(
        pattern=str(data.get("pattern", plan.get("pattern", ""))),
        status=result.status.value,
        confidence=result.confidence,
        llm_calls=result.usage.llm_calls,
        expected_calls=expected_calls,
        expected_speedup=float(data.get("expected_speedup", plan.get("expected_speedup", 1.0)) or 1.0),
        measured_speedup=measured,
        utilisation=round(work / (wall * concurrency), 4) if wall > 0 and concurrency > 0 else 0.0,
        serial_fraction=round(serial_fraction, 6),
        amdahl_bound=bound if math.isfinite(bound) else 1e9,
        efficiency=round(parallel_efficiency(measured, concurrency), 4),
        p50_latency_s=round(percentile(lat, 50), 6),
        p95_latency_s=round(percentile(lat, 95), 6),
        wall_s=wall,
        total_work_s=work,
        branches=n_branches,
        failed_branches=failed,
        partial_failure_rate=round(failed / n_branches, 4) if n_branches else 0.0,
        submitted=submitted,
        retries=retries,
        retry_rate=round(retries / submitted, 4) if submitted else 0.0,
        timeouts=int(ex.get("timeouts", 0) or 0),
        hedges=int(ex.get("hedges", 0) or 0),
        breaker_rejections=int(ex.get("breaker_rejections", 0) or 0),
        breaker_trips=int(ex.get("breaker_trips", 0) or 0),
        rate_limit_wait_s=float(ex.get("rate_limit_wait_s", 0.0) or 0.0),
        cache_hit_ratio=round(result.usage.cache_hit_ratio, 4),
        cache_warmups=int(data.get("cache_warmups", 0) or 0),
        cache_waits=int(data.get("cache_waits", 0) or 0),
        token_multiplier=float(result.usage.llm_calls),
        cost_usd=result.usage.cost_usd,
    )


class EvaluationReport(BaseModel):
    """Aggregate over several runs."""

    model_config = ConfigDict(extra="forbid")

    runs: int = 0
    by_pattern: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    mean_confidence: float = 0.0
    mean_llm_calls: float = 0.0
    mean_measured_speedup: float = 0.0
    mean_expected_speedup: float = 0.0
    p50_wall_s: float = 0.0
    p95_wall_s: float = 0.0
    retry_rate: float = Field(default=0.0, description="total retries / total submitted (same denominator as RunMetrics.retry_rate)")
    partial_failure_rate: float = 0.0
    breaker_rejections: int = 0
    breaker_trips: int = 0
    cache_hit_ratio: float = 0.0
    total_cost_usd: float = 0.0
    elapsed_s: float = 0.0
    per_run: list[RunMetrics] = Field(default_factory=list)

    def table(self) -> str:
        """Compact text table for CLIs and READMEs."""
        lines = [f"runs={self.runs} elapsed={self.elapsed_s:.2f}s cost=${self.total_cost_usd:.4f}"]
        lines.append(f"patterns: {', '.join(f'{k}={v}' for k, v in sorted(self.by_pattern.items()))}")
        lines.append(f"status:   {', '.join(f'{k}={v}' for k, v in sorted(self.by_status.items()))}")
        lines.append(
            f"confidence={self.mean_confidence:.2f} calls/run={self.mean_llm_calls:.1f} speedup(measured/expected)={self.mean_measured_speedup:.2f}/{self.mean_expected_speedup:.2f} "
            f"wall p50/p95={self.p50_wall_s:.3f}/{self.p95_wall_s:.3f}s retry_rate={self.retry_rate:.3f} breaker_trips={self.breaker_trips} partial_failure={self.partial_failure_rate:.3f} cache_hit={self.cache_hit_ratio:.2f}"
        )
        return "\n".join(lines)


def aggregate(metrics: Sequence[RunMetrics], *, elapsed_s: float = 0.0) -> EvaluationReport:
    """Roll individual :class:`RunMetrics` up into an :class:`EvaluationReport`."""
    rep = EvaluationReport(runs=len(metrics), elapsed_s=round(elapsed_s, 4), per_run=list(metrics))
    if not metrics:
        return rep
    n = len(metrics)
    for m in metrics:
        rep.by_pattern[m.pattern] = rep.by_pattern.get(m.pattern, 0) + 1
        rep.by_status[m.status] = rep.by_status.get(m.status, 0) + 1
    rep.mean_confidence = round(sum(m.confidence for m in metrics) / n, 4)
    rep.mean_llm_calls = round(sum(m.llm_calls for m in metrics) / n, 3)
    rep.mean_measured_speedup = round(sum(m.measured_speedup for m in metrics) / n, 3)
    rep.mean_expected_speedup = round(sum(m.expected_speedup for m in metrics) / n, 3)
    walls = [m.wall_s for m in metrics]
    rep.p50_wall_s = round(percentile(walls, 50), 6)
    rep.p95_wall_s = round(percentile(walls, 95), 6)
    total_branches = sum(m.branches for m in metrics)
    rep.partial_failure_rate = round(sum(m.failed_branches for m in metrics) / total_branches, 4) if total_branches else 0.0
    submitted = sum(m.submitted for m in metrics)
    rep.retry_rate = round(sum(m.retries for m in metrics) / submitted, 4) if submitted else 0.0
    rep.breaker_rejections = sum(m.breaker_rejections for m in metrics)
    rep.breaker_trips = sum(m.breaker_trips for m in metrics)
    rep.cache_hit_ratio = round(sum(m.cache_hit_ratio for m in metrics) / n, 4)
    rep.total_cost_usd = round(sum(m.cost_usd for m in metrics), 6)
    return rep


def synthetic_tasks(domain: str = "general", *, ensemble_domain: str = "mechanical_engineering") -> list[TaskRequest]:
    """A small engineering-flavoured task set that exercises all eight patterns offline.

    The ensemble task always runs in a high-risk domain (``ensemble_domain``)
    because the router reserves layered ensembles for high-value free-form work
    in high-risk domains; the speculative task carries a tight latency budget.
    """
    long_log = "\n".join(f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z worker-{i % 7} ERROR timeout contacting inventory-service after 30s (attempt {i % 3 + 1})" for i in range(400))
    return [
        TaskRequest(task="Compare steel, aluminium and titanium for a lightweight bracket.", domain=domain),
        TaskRequest(task="What are the main failure modes of a bolted flange joint?", domain=domain),
        TaskRequest(task="Calculate the bending stress for a 2 m beam with a 10 kN central load and section modulus 200 cm3.", domain=domain),
        TaskRequest(task="Review this design note against the requirements.", inputs={"text": "The bracket shall carry 5 kN with a factor of safety of 2. Material: 6061-T6 aluminium."}, domain=domain),
        TaskRequest(task="Summarise the failure clusters in these logs.", inputs={"text": long_log}, domain=domain),
        TaskRequest(task="Derive the deflection formula for a simply supported beam step by step.", domain=domain),
        TaskRequest(task="Classify each ticket as bug, feature or question.", inputs={"items": ["Crash on save", "Add dark mode", "How do I export?"]}, domain=domain),
        TaskRequest(task="Justify the trade-off between bolt preload and gasket creep for the flange of the pressure vessel inspected last week.", domain=ensemble_domain),
        TaskRequest(task="Name the SI unit of torque.", domain="general", constraints=Constraints(max_latency_s=20)),
    ]


async def evaluate_async(model: Any, tasks: Iterable[TaskRequest]) -> EvaluationReport:
    """Run ``model`` over ``tasks`` and aggregate metrics."""
    t0 = time.perf_counter()
    metrics: list[RunMetrics] = []
    for req in tasks:
        res: TaskResult = await model.run(req)
        metrics.append(metrics_from_result(res))
    return aggregate(metrics, elapsed_s=time.perf_counter() - t0)


def evaluate(model: Any, tasks: Iterable[TaskRequest] | None = None, *, domain: str = "general") -> EvaluationReport:
    """Synchronous evaluation harness; defaults to :func:`synthetic_tasks`."""
    return run_sync(evaluate_async(model, list(tasks) if tasks is not None else synthetic_tasks(domain)))


def status_counts(results: Iterable[TaskResult]) -> dict[str, int]:
    """Count results by :class:`Status`."""
    counts: dict[str, int] = {}
    for r in results:
        key = r.status.value if isinstance(r.status, Status) else str(r.status)
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = ["EvaluationReport", "RunMetrics", "aggregate", "evaluate", "evaluate_async", "metrics_from_result", "percentile", "status_counts", "synthetic_tasks"]
