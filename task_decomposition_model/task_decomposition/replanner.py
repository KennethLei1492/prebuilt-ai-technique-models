"""Reflexion-style replanning, gated on external signals and bounded.

After a scheduling pass, failed nodes are either *replannable* (solver error,
failed verification, invalid output) or not (budget or deadline exhaustion,
refusals). For replannable failures the :class:`Replanner` writes a short
structured :class:`Reflection` (root cause, what to change) into a bounded
FIFO :class:`ReflectionMemory` and proposes replacement sub-tasks for the
failed region only; completed nodes keep their outputs. The model loops at
most ``max_replans`` times.

Design constraints from the literature:

* Reflexion (Shinn et al.): episodic memory capped at 1-3 entries; trials
  bounded.
* Huang et al.: a failure is only ever declared by an external signal (solver
  status, verifier verdict, schema validation), never by the planner
  re-reading its own output.
* LLMCompiler / LangChain planning agents: the joiner/replanner is a bounded
  LLM step; the scheduler itself never calls a model.

``apply_revision`` and ``ReflectionMemory`` are pure and unit-tested.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import BudgetExceededError, LLMError, PlanValidationError, RunContext, StructuredOutputError, TaskRequest, validate_structured

from .config import TaskDecompositionConfig
from .graph import PLACEHOLDER_RE, SubTask, TaskGraph
from .plan import DecompositionPlan
from .prompting import build_request
from .prompts import REPLAN_TEMPLATE
from .scheduler import ScheduleReport, SubTaskResult
from .schemas import Reflection, Revision, SubTaskDraft, schema_for
from .strategies import can_call, heuristic_segments, use_llm_path

NON_REPLANNABLE_PREFIXES = ("budget", "not started", "refusal", "LLMRefusalError")


class ReflectionMemory(BaseModel):
    """Bounded FIFO of reflections (Reflexion episodic memory, Omega = 1-3)."""

    model_config = ConfigDict(extra="forbid")

    cap: int = Field(default=3, ge=1)
    entries: list[Reflection] = Field(default_factory=list)

    def add(self, reflection: Reflection) -> None:
        q = deque(self.entries, maxlen=self.cap)
        q.append(reflection)
        self.entries = list(q)

    def render(self) -> str:
        if not self.entries:
            return "(no prior reflections)"
        return "\n".join(f"- root cause: {r.root_cause} | change: {r.change} | affected: {', '.join(r.affected_task_ids) or '-'}" for r in self.entries)


def replannable_failures(report: ScheduleReport) -> list[str]:
    """Failed node ids whose failure came from the work itself, not from budget/deadline/refusal."""
    out: list[str] = []
    for tid in report.failed_ids:
        err = report.results[tid].error or ""
        if any(err.lower().startswith(p.lower()) for p in NON_REPLANNABLE_PREFIXES):
            continue
        out.append(tid)
    return out


def apply_revision(graph: TaskGraph, failed_ids: Sequence[str], replacements: Sequence[SubTask]) -> TaskGraph:
    """Replace ``failed_ids`` with ``replacements`` and re-point dependents to the last replacement.

    Replacement ids must be fresh. Dependents of a removed node (direct or via
    placeholders) now depend on the final replacement node, whose output
    stands in for the removed one. Raises :class:`PlanValidationError` when the
    revised graph is invalid.
    """
    failed = set(failed_ids)
    if not replacements:
        raise PlanValidationError("revision has no replacement sub-tasks")
    existing = {t.id for t in graph.tasks if t.id not in failed}
    for r in replacements:
        if r.id in existing or r.id in failed:
            raise PlanValidationError(f"replacement id {r.id!r} collides with an existing sub-task")
        existing.add(r.id)
    stand_in = replacements[-1].id
    mapping = {fid: stand_in for fid in failed}
    kept: list[SubTask] = []
    for t in graph.tasks:
        if t.id in failed:
            continue
        deps: list[str] = []
        for d in t.depends_on:
            d2 = mapping.get(d, d)
            if d2 not in deps:
                deps.append(d2)
        inputs = _remap_placeholders(t.inputs, mapping)
        desc = _remap_text(t.description, mapping)
        kept.append(t.model_copy(update={"depends_on": deps, "inputs": inputs, "description": desc}))
    new_tasks = kept + [r.model_copy(deep=True) for r in replacements]
    revised = graph.with_tasks(new_tasks).infer_dependencies()
    revised.metadata = {**graph.metadata, "revisions": int(graph.metadata.get("revisions", 0)) + 1}
    revised.ensure_valid()
    return revised


def _remap_text(text: str, mapping: dict[str, str]) -> str:
    """Re-point every ``$id`` / ``${id}`` / ``$id.output`` placeholder in ``text`` through ``mapping``.

    Uses the shared :data:`PLACEHOLDER_RE` so a reference followed by any punctuation (``$t2,``, ``($t2)``,
    ``$t2;``) is rewritten too; otherwise the dependent keeps pointing at the removed node and
    ``substitute_placeholders`` leaves the literal ``$t2`` in its prompt.
    """

    def _sub(m: "re.Match[str]") -> str:
        ref = m.group(1)
        if ref not in mapping:
            return m.group(0)
        return m.group(0).replace(ref, mapping[ref], 1)

    return PLACEHOLDER_RE.sub(_sub, text)


def _remap_placeholders(value, mapping: dict[str, str]):  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        return _remap_text(value, mapping)
    if isinstance(value, dict):
        return {k: _remap_placeholders(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap_placeholders(v, mapping) for v in value]
    return value


class Replanner:
    """Reflect on externally observed failures and revise the failed region of the graph."""

    def __init__(self, config: Optional[TaskDecompositionConfig] = None) -> None:
        self.config = config or TaskDecompositionConfig()

    # ------------------------------------------------------------ heuristic
    def heuristic_revision(self, graph: TaskGraph, failed_ids: Sequence[str], results: dict[str, SubTaskResult], *, round_no: int) -> Revision:
        """Model-free revision: split each failed node into two shallower halves (Faith and Fate: bound per-step depth); give up on repeats."""
        drafts: list[SubTaskDraft] = []
        causes: list[str] = []
        for fid in failed_ids:
            node = graph.get(fid)
            err = results[fid].error or "failed"
            causes.append(f"{fid}: {err[:120]}")
            if "narrowed from" in node.description:
                continue  # already split once; a second split would only re-run the same work (ADaPT depth bound)
            _, segs = heuristic_segments(node.description, max_segments=2)
            if len(segs) < 2:
                words = node.description.split()
                if len(words) < 6:
                    continue
                half = len(words) // 2
                segs = [" ".join(words[:half]), " ".join(words[half:])]
            a_id, b_id = f"{fid}r{round_no}a", f"{fid}r{round_no}b"
            drafts.append(SubTaskDraft(id=a_id, description=f"{segs[0]} (narrowed from {fid})", depends_on=list(node.depends_on), expected_output=node.expected_output, verification_criteria=node.verification_criteria[:2], tool_hint=node.tool_hint if node.tool_hint in ("llm", "retrieve", "compute", "verify", "aggregate") else "llm", complexity="low"))
            drafts.append(SubTaskDraft(id=b_id, description=f"{segs[1]} (narrowed from {fid}; builds on the first half)", depends_on=[a_id], inputs=[], expected_output=node.expected_output, verification_criteria=node.verification_criteria[:2], tool_hint="aggregate", complexity="low"))
        reflection = Reflection(root_cause="; ".join(causes) or "unknown", change="split the failed sub-task into two shallower steps", affected_task_ids=list(failed_ids))
        return Revision(reflection=reflection, replacement_tasks=drafts, give_up=not drafts)

    # ------------------------------------------------------------------ llm
    async def revise(
        self,
        *,
        graph: TaskGraph,
        report: ScheduleReport,
        request: TaskRequest,
        ctx: RunContext,
        plan: DecompositionPlan,
        memory: ReflectionMemory,
        round_no: int,
    ) -> Optional[Revision]:
        """Return a :class:`Revision` for the replannable failures, or ``None`` when nothing can be done."""
        failed = replannable_failures(report)
        if not failed:
            return None
        if use_llm_path(ctx, self.config) and can_call(ctx):
            completed = "\n".join(f"- {tid}: {r.output_text[:200]}" for tid, r in report.results.items() if r.ok) or "(none)"
            failures = "\n".join(f"- {tid} ({graph.get(tid).description[:120]}): {report.results[tid].error}" for tid in failed)
            req = build_request(
                REPLAN_TEMPLATE,
                profile=ctx.profile,
                values={
                    "task": graph.root_task,
                    "graph": graph.to_ascii(),
                    "completed": completed,
                    "failures": failures,
                    "memory": memory.render(),
                    "bounds": f"max replacement sub-tasks={plan.max_breadth}; existing ids={', '.join(graph.ids())}; use fresh ids such as r{round_no}_1",
                },
                schema=schema_for(Revision),
                effort=self.config.planner_effort,
                tags=["replan", f"round{round_no}"],
            )
            try:
                resp = await ctx.llm_call(req, step=f"replan.{round_no}.llm")
                revision = validate_structured(resp.parsed, Revision)
                revision.reflection.affected_task_ids = revision.reflection.affected_task_ids or list(failed)
                return revision
            except (BudgetExceededError, StructuredOutputError, LLMError) as exc:
                ctx.warn(f"replanner LLM call failed ({type(exc).__name__}); using heuristic revision")
        return self.heuristic_revision(graph, failed, report.results, round_no=round_no)

    def materialise(self, graph: TaskGraph, revision: Revision, failed_ids: Sequence[str]) -> TaskGraph:
        """Apply a revision to the graph (validated)."""
        replacements = [d.to_subtask(max_retries=self.config.max_retries_per_task) for d in revision.replacement_tasks if d.description.strip()]
        return apply_revision(graph, failed_ids, replacements)


__all__ = ["NON_REPLANNABLE_PREFIXES", "ReflectionMemory", "Replanner", "apply_revision", "replannable_failures"]
