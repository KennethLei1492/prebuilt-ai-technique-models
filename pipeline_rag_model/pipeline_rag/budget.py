"""One budget view shared by every stage, plus the record of degraded stages.

Architecture rule 3.2 asks every component to respect ``ctx.remaining_time()``
and ``ctx.budget.remaining_calls()`` before fan-outs and to degrade rather
than fail. The model, the query transformer, the multi-hop loop and the
judges used to carry private copies of that check which drifted apart (one
ignored the deadline, two ignored the cost budget); they all call
:func:`calls_left` now.

:func:`note_degraded` records which stage fell back to its heuristic twin
(LLM failure, budget exhaustion, a failing retrieval leg or executor) in
``ctx.scratch["degraded"]``; ``_finish`` turns a non-empty list into
``Status.PARTIAL`` with an explicit reason instead of pattern-matching warning
text.
"""
from __future__ import annotations

from typing import Optional

from pipeline_core import RunContext

UNLIMITED_CALLS = 10_000
"""Stand-in for "no call limit" so comparisons such as ``calls_left(ctx) >= 2`` stay simple."""


def calls_left(ctx: Optional[RunContext]) -> int:
    """LLM calls still affordable for this run.

    Zero once the latency budget (``max_latency_s``) or the cost budget
    (``max_cost_usd``) is spent, otherwise the remaining call budget
    (``max_llm_calls``) or :data:`UNLIMITED_CALLS` when none is set.
    """
    if ctx is None:
        return UNLIMITED_CALLS
    if ctx.deadline_exceeded():
        return 0
    cost_left = ctx.budget.remaining_cost()
    if cost_left is not None and cost_left <= 0.0:
        return 0
    rem = ctx.budget.remaining_calls()
    return UNLIMITED_CALLS if rem is None else rem


def note_degraded(ctx: Optional[RunContext], stage: str) -> None:
    """Record that ``stage`` did not run as planned and fell back (idempotent per stage)."""
    if ctx is None:
        return
    stages: list[str] = ctx.scratch.setdefault("degraded", [])
    if stage not in stages:
        stages.append(stage)


def degraded_stages(ctx: Optional[RunContext]) -> list[str]:
    """Stages recorded by :func:`note_degraded`, in order of occurrence."""
    if ctx is None:
        return []
    return list(ctx.scratch.get("degraded", []))


__all__ = ["UNLIMITED_CALLS", "calls_left", "degraded_stages", "note_degraded"]
