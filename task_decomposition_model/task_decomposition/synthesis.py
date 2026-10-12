"""Synthesis: compose sub-task results into the final answer with traceability.

Two paths share the same output type:

* **LLM** - one ``SYNTHESIZE_TEMPLATE`` call (effort ``high``) that re-asks the
  original task with every sub-answer in context, which is the Least-to-Most
  rule for forcing composition and the ReWOO/LLMCompiler "solver"/"joiner".
* **Heuristic** - deterministic concatenation with per-sub-task traceability,
  used offline and when the budget cannot afford another call.

Verification-node results (a :class:`Verdict` produced by an injected
verifier, see :class:`~task_decomposition.solvers.VerifierNodeSolver`) are not
content: neither path composes their text into the answer or counts them in
confidence/coverage. A failed or unavailable verification is listed under
``unresolved``; the verdict itself stays on the sub-task result and in
``DecompositionOutput.node_verdicts``.

Confidence aggregation is a pure function. It treats a composed answer as no
stronger than its weakest essential part and applies a composition discount,
because the compositionality gap (Self-Ask) means correct sub-answers do not
guarantee a correct composition.
"""
from __future__ import annotations

from typing import Optional, Sequence

from pipeline_core import RunContext, Status, TaskRequest, validate_structured

from .config import TaskDecompositionConfig
from .graph import TaskGraph
from .prompting import build_request, render_documents
from .prompts import SYNTHESIZE_TEMPLATE
from .scheduler import SubTaskResult
from .schemas import Contribution, SynthesisOutput, schema_for
from .solvers import is_verification_result


def aggregate_confidence(
    confidences: Sequence[float],
    *,
    composition_discount: float = 0.10,
    coverage: float = 1.0,
    weakest_allowance: float = 0.2,
) -> float:
    """Combine sub-answer confidences into one number in [0, 1].

    ``base = min(mean, weakest + weakest_allowance)`` keeps a chain close to
    its weakest link while not collapsing to the minimum; the composition
    discount applies once when more than one part is composed; ``coverage``
    (fraction of planned sub-tasks that completed) scales the result.
    """
    vals = [max(0.0, min(1.0, c)) for c in confidences]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    base = min(mean, min(vals) + weakest_allowance)
    if len(vals) > 1:
        base *= 1.0 - max(0.0, min(1.0, composition_discount))
    return max(0.0, min(1.0, base * max(0.0, min(1.0, coverage))))


def claims_text(answer: str) -> str:
    """The assertions of a composed answer, without Markdown heading lines.

    Heuristic composition titles the answer with the task and each section with ``## <id>: <sub-task
    description>``. Those lines restate instructions ("Check the corrosion allowance"); a claim-level
    verifier would grade them as unsupported or contradicted claims (the ``t1`` id reads as a number), so
    the final verifier checks the claims only. Falls back to ``answer`` when nothing else remains.
    """
    kept = [line for line in answer.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(kept).strip()
    return text or answer.strip()


def coverage_ratio(results: dict[str, SubTaskResult], graph: TaskGraph) -> float:
    """Fraction of content sub-tasks that completed (verification-node verdicts are not content)."""
    content = [t for t in graph.tasks if not (t.id in results and is_verification_result(results[t.id]))]
    if not content:
        return 0.0
    ok = sum(1 for t in content if t.id in results and results[t.id].ok)
    return ok / len(content)


def verification_notes(tid: str, res: SubTaskResult) -> list[str]:
    """Open items for a verification node that did not pass (empty when it passed)."""
    if res.status == Status.SUCCESS:
        return []
    head = f"{tid} ({res.status.value}): {res.error or 'verification did not pass'}"
    return [head] + [f"{tid}: {g}" for g in res.gaps if g not in head]


class Synthesizer:
    """Compose sub-task results into a :class:`SynthesisOutput`."""

    def __init__(self, config: Optional[TaskDecompositionConfig] = None) -> None:
        self.config = config or TaskDecompositionConfig()

    # ------------------------------------------------------------- heuristic
    def heuristic(self, task: str, graph: TaskGraph, results: dict[str, SubTaskResult]) -> SynthesisOutput:
        """Deterministic composition: ordered sections per sub-task plus an open-items list."""
        sections: list[str] = []
        trace: list[Contribution] = []
        unresolved: list[str] = []
        confs: list[float] = []
        order = [tid for wave in graph.topological_waves() for tid in wave] if graph.tasks else list(results)
        for tid in order:
            res = results.get(tid)
            if res is None:
                continue
            if is_verification_result(res):
                unresolved.extend(verification_notes(tid, res))  # a verdict, not content
                continue
            try:
                desc = graph.get(tid).description
            except KeyError:
                desc = tid
            if res.ok and res.output_text.strip():
                sections.append(f"## {tid}: {desc}\n{res.output_text.strip()}")
                trace.append(Contribution(task_id=tid, contribution=desc))
                confs.append(res.confidence)
            else:
                unresolved.append(f"{tid} ({res.status.value}): {res.error or 'no output'}")
            unresolved.extend(f"{tid}: {g}" for g in res.gaps)
        if len(graph.tasks) == 1 and sections:
            answer = results[graph.tasks[0].id].output_text.strip()
        else:
            answer = f"# {task.strip()}\n\n" + "\n\n".join(sections) if sections else ""
        conf = aggregate_confidence(confs, composition_discount=self.config.composition_discount, coverage=coverage_ratio(results, graph))
        return SynthesisOutput(answer=answer, confidence=conf, traceability=trace, unresolved=unresolved)

    # ------------------------------------------------------------------- llm
    async def synthesize(
        self,
        *,
        task: str,
        graph: TaskGraph,
        results: dict[str, SubTaskResult],
        request: TaskRequest,
        ctx: RunContext,
        use_llm: bool,
    ) -> SynthesisOutput:
        """LLM composition when ``use_llm``; falls back to :meth:`heuristic` on any error."""
        if not use_llm:
            return self.heuristic(task, graph, results)
        if len(graph.tasks) == 1 and graph.tasks[0].id in results and results[graph.tasks[0].id].ok and not self.config.always_append_original_task:
            return self.heuristic(task, graph, results)
        if len(graph.tasks) == 1 and graph.tasks[0].id in results and results[graph.tasks[0].id].ok and graph.strategy in ("direct", "plan_and_solve"):
            # A single call already answered the original task; re-asking adds cost without composition value.
            return self.heuristic(task, graph, results)
        rendered: list[str] = []
        for tid in [t for wave in graph.topological_waves() for t in wave]:
            res = results.get(tid)
            if res is None or is_verification_result(res):
                continue
            desc = graph.get(tid).description
            body = res.output_text.strip() if res.ok else f"({res.status.value}) {res.error or 'no output'}"
            rendered.append(f'<result id="{tid}" status="{res.status.value}" confidence="{res.confidence:.2f}">\n{desc}\n---\n{body}\n</result>')
        req = build_request(
            SYNTHESIZE_TEMPLATE,
            profile=ctx.profile,
            values={
                "task": task,
                "results": "\n".join(rendered) or "(no results)",
                "documents": render_documents(request.context, max_items=self.config.max_context_docs),
            },
            schema=schema_for(SynthesisOutput),
            effort=self.config.synthesis_effort,
            tags=["synthesize"],
        )
        try:
            resp = await ctx.llm_call(req, step="synthesize.llm")
            out = validate_structured(resp.parsed, SynthesisOutput)
        except Exception as exc:  # noqa: BLE001 - degrade to heuristic composition
            ctx.warn(f"synthesis LLM call failed ({type(exc).__name__}); using heuristic composition")
            return self.heuristic(task, graph, results)
        confs = [r.confidence for r in results.values() if r.ok and not is_verification_result(r)]
        structural = aggregate_confidence(confs, composition_discount=self.config.composition_discount, coverage=coverage_ratio(results, graph))
        out.confidence = min(out.confidence, structural) if confs else min(out.confidence, 0.3)
        if not out.answer.strip():
            fallback = self.heuristic(task, graph, results)
            out.answer = fallback.answer
        failed = [tid for tid, r in results.items() if r.status in (Status.FAILED, Status.SKIPPED)]
        for tid in failed:
            note = f"{tid} did not complete"
            if note not in out.unresolved:
                out.unresolved.append(note)
        for tid, r in results.items():
            if is_verification_result(r):
                out.unresolved.extend(n for n in verification_notes(tid, r) if n not in out.unresolved)
        out.traceability = [c for c in out.traceability if not (c.task_id in results and is_verification_result(results[c.task_id]))]
        return out
