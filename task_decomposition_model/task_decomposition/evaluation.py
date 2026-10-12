"""Plan-quality metrics and an offline evaluation harness.

The planning survey (Huang et al., 2024) warns that final-success-only
evaluation hides decomposition failures, so this module scores plans
*step-wise* (architecture recommendation 11):

* ``coverage``                 keyword coverage of the task by the sub-tasks (model-free judge);
  ``llm_coverage`` is the LLM-judge variant (requirements listed, then mapped to sub-task ids)
* ``dependency_correctness``   placeholders consistent with ``depends_on``, acyclic, no dangling ids
* ``parallel_speedup``         total nodes / critical path (LLMCompiler latency ratio)
* ``plan validity``            structural defects before any repair (HuggingGPT planning F1 ~50%)
* ``compositionality_gap``     sub-tasks all correct but composition wrong (Self-Ask)
* ``failure classes``          AgentBench-style budget / format / verification / solver buckets
* cost: LLM calls, tokens and USD per task

``evaluate()`` runs a list of cases through a model offline (MockLLM) and
reports the aggregates; ``python -m task_decomposition.eval_cli`` prints them
(the entry point lives in :mod:`task_decomposition.eval_cli`, which the package
``__init__`` does not import, so ``runpy`` executes it once and warning-free).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import BaseLLMClient, BudgetExceededError, DomainProfile, LLMError, RunContext, Status, StructuredOutputError, TaskRequest, TaskResult, Usage, get_profile, run_sync, validate_structured

from .graph import TaskGraph, find_placeholders
from .prompting import build_request
from .prompts import COVERAGE_JUDGE_TEMPLATE
from .scheduler import SubTaskResult
from .schemas import CoverageJudgement, schema_for
from .strategies import content_words

CoverageJudge = Literal["keyword", "llm"]


class PlanQualityReport(BaseModel):
    """Step-wise quality metrics for one :class:`TaskGraph`."""

    model_config = ConfigDict(extra="forbid")

    n_tasks: int = 0
    depth: int = 0
    breadth: int = 0
    critical_path: int = 0
    parallel_speedup: float = 1.0
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    dependency_correctness: float = Field(default=1.0, ge=0.0, le=1.0)
    valid: bool = True
    issues: list[str] = Field(default_factory=list)
    with_criteria: float = Field(default=0.0, ge=0.0, le=1.0, description="Share of nodes carrying at least one verification criterion")
    estimated_llm_calls: int = 0

    def score(self) -> float:
        """Single 0-1 summary: coverage and dependency correctness weighted, validity as a gate."""
        base = 0.5 * self.coverage + 0.3 * self.dependency_correctness + 0.2 * self.with_criteria
        return round(base if self.valid else base * 0.5, 3)


def coverage(graph: TaskGraph, task: Optional[str] = None) -> float:
    """Fraction of the task's content words that appear in some sub-task description (keyword judge)."""
    words = content_words(task or graph.root_task)
    if not words:
        return 1.0
    covered = content_words(" ".join(f"{t.description} {t.expected_output}" for t in graph.tasks))
    return len(words & covered) / len(words)


async def judge_coverage(
    graph: TaskGraph,
    task: Optional[str] = None,
    *,
    llm: Optional[BaseLLMClient] = None,
    ctx: Optional[RunContext] = None,
    profile: Optional[DomainProfile] = None,
    effort: str = "low",
) -> tuple[float, Usage]:
    """LLM-judge coverage with its cost: ``(coverage, usage of the judge call)``.

    Lists the task's atomic requirements with one cheap structured call (``effort`` low by default) and
    returns the fraction addressed by some sub-task. With a :class:`RunContext` the call goes through
    ``ctx.llm_call`` (traced, budgeted and charged, architecture rule 3.1.3); without one, ``llm.complete``
    is used and the returned ``usage`` lets the caller account for it. Falls back to the keyword judge
    (zero usage) when the model returns no requirements (for example the offline ``MockLLM``) or the call
    fails, so the metric is always defined.
    """
    task = task or graph.root_task
    client = llm or (ctx.llm if ctx is not None else None)
    if client is None:
        raise ValueError("judge_coverage needs an LLM client: pass llm=... or ctx=...")
    req = build_request(
        COVERAGE_JUDGE_TEMPLATE,
        profile=profile or (ctx.profile if ctx is not None else get_profile("general")),
        values={"task": task, "graph": graph.to_ascii()},
        schema=schema_for(CoverageJudgement),
        effort=effort,  # type: ignore[arg-type]
        tags=["coverage_judge"],
    )
    try:
        resp = await (ctx.llm_call(req, step="coverage_judge.llm") if ctx is not None else client.complete(req))
        judgement = validate_structured(resp.parsed, CoverageJudgement)
    except (LLMError, StructuredOutputError, BudgetExceededError):
        return coverage(graph, task), Usage()
    if not judgement.requirements:
        return coverage(graph, task), resp.usage
    return round(judgement.coverage(set(graph.ids())), 3), resp.usage


async def llm_coverage(
    graph: TaskGraph,
    task: Optional[str] = None,
    *,
    llm: Optional[BaseLLMClient] = None,
    ctx: Optional[RunContext] = None,
    profile: Optional[DomainProfile] = None,
    effort: str = "low",
) -> float:
    """LLM-judge coverage as a number; see :func:`judge_coverage` for the variant that also returns the judge's usage."""
    value, _ = await judge_coverage(graph, task, llm=llm, ctx=ctx, profile=profile, effort=effort)
    return value


def dependency_correctness(graph: TaskGraph) -> float:
    """1.0 when every placeholder is backed by a dependency, no dependency dangles and the graph is acyclic.

    Each defect (dangling dependency, placeholder without edge, self-edge) costs one point out of the
    number of edges plus placeholders; a cycle scores 0.
    """
    if not graph.tasks:
        return 0.0
    known = set(graph.ids())
    checks = 0
    bad = 0
    for t in graph.tasks:
        for d in t.depends_on:
            checks += 1
            if d not in known or d == t.id:
                bad += 1
        for ref in find_placeholders(t.inputs) + find_placeholders(t.description):
            if ref == t.id or ref not in known:
                continue
            checks += 1
            if ref not in t.depends_on:
                bad += 1
    if any("cycle" in i for i in graph.validate_structure(detect_duplicates=False)):
        return 0.0
    return 1.0 if checks == 0 else max(0.0, 1.0 - bad / checks)


def parallel_speedup(graph: TaskGraph) -> float:
    """Total nodes divided by critical-path length: latency ratio of sequential vs wave execution."""
    return graph.parallel_speedup_estimate() if graph.tasks and not graph.validate_structure(detect_duplicates=False) else 1.0


def evaluate_plan(graph: TaskGraph, task: Optional[str] = None, *, max_depth: Optional[int] = None, max_breadth: Optional[int] = None, max_nodes: Optional[int] = None) -> PlanQualityReport:
    """Compute every plan metric without a model call."""
    issues = graph.validate_structure(max_depth=max_depth, max_breadth=max_breadth, max_nodes=max_nodes)
    structural_ok = not any(("cycle" in i or "unknown" in i or "no sub-tasks" in i or "duplicate sub-task ids" in i) for i in issues)
    depth = graph.depth() if structural_ok else 0
    breadth = graph.breadth() if structural_ok else 0
    cp = graph.critical_path_length() if structural_ok else 0
    return PlanQualityReport(
        n_tasks=len(graph.tasks),
        depth=depth,
        breadth=breadth,
        critical_path=cp,
        parallel_speedup=round(len(graph.tasks) / cp, 3) if cp else 1.0,
        coverage=round(coverage(graph, task), 3),
        dependency_correctness=round(dependency_correctness(graph), 3),
        valid=not issues,
        issues=issues,
        with_criteria=round(sum(1 for t in graph.tasks if t.verification_criteria) / len(graph.tasks), 3) if graph.tasks else 0.0,
        estimated_llm_calls=len(graph.tasks) + (1 if len(graph.tasks) > 1 else 0),
    )


def compositionality_gap(outcomes: Iterable[tuple[bool, bool]]) -> float:
    """Self-Ask gap: among cases whose sub-tasks all succeeded, the fraction whose final answer was wrong."""
    rows = list(outcomes)
    eligible = [final_ok for subs_ok, final_ok in rows if subs_ok]
    if not eligible:
        return 0.0
    return sum(1 for ok in eligible if not ok) / len(eligible)


def failure_classes(results: dict[str, SubTaskResult]) -> dict[str, int]:
    """Bucket non-ok sub-task results into AgentBench-style failure classes."""
    counts: Counter[str] = Counter()
    for r in results.values():
        if r.ok:
            continue
        err = (r.error or "").lower()
        if r.status == Status.SKIPPED:
            counts["skipped_dependency" if "dependency" in err else "budget_or_deadline"] += 1
        elif err.startswith("budget"):
            counts["budget_or_deadline"] += 1
        elif err.startswith("verifier"):
            counts["verification_failed"] += 1
        elif "structuredoutput" in err or "not valid json" in err or "validation" in err:
            counts["invalid_format"] += 1
        elif "refus" in err:
            counts["refusal"] += 1
        else:
            counts["solver_failed"] += 1
    return dict(counts)


def subtask_success_rate(results: dict[str, SubTaskResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results.values() if r.ok) / len(results)


# ----------------------------------------------------------------- harness


class EvalCase(BaseModel):
    """One evaluation case: a task, its domain and keywords a correct answer must mention (keyword judge)."""

    model_config = ConfigDict(extra="forbid")

    task: str
    domain: str = "general"
    expected_keywords: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_strategy: Optional[str] = None


class CaseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    domain: str
    status: str
    strategy: str = ""
    strategy_match: Optional[bool] = None
    confidence: float = 0.0
    final_ok: bool = False
    subtasks_ok: bool = False
    plan: PlanQualityReport = Field(default_factory=PlanQualityReport)
    llm_calls: int = Field(default=0, description="Calls made by the run plus the coverage judge call when judge='llm'")
    cost_usd: float = Field(default=0.0, description="Cost of the run plus the judge call when judge='llm'")
    latency_s: float = 0.0
    judge_calls: int = Field(default=0, description="LLM calls spent by the coverage judge (0 for the keyword judge)")
    replans: int = 0
    failure_classes: dict[str, int] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: int = 0
    success_rate: float = 0.0
    needs_review_rate: float = 0.0
    keyword_accuracy: float = 0.0
    plan_validity_rate: float = 0.0
    strategy_accuracy: Optional[float] = None
    mean_coverage: float = 0.0
    mean_parallel_speedup: float = 1.0
    compositionality_gap: float = 0.0
    mean_llm_calls: float = 0.0
    total_cost_usd: float = 0.0
    mean_latency_s: float = 0.0
    strategies: dict[str, int] = Field(default_factory=dict)
    failure_classes: dict[str, int] = Field(default_factory=dict)
    outcomes: list[CaseOutcome] = Field(default_factory=list)

    def table(self) -> str:
        lines = [
            f"cases={self.cases} success={self.success_rate:.2f} needs_review={self.needs_review_rate:.2f} keyword_acc={self.keyword_accuracy:.2f}",
            f"plan_validity={self.plan_validity_rate:.2f} coverage={self.mean_coverage:.2f} speedup={self.mean_parallel_speedup:.2f} comp_gap={self.compositionality_gap:.2f}",
            f"calls/task={self.mean_llm_calls:.1f} cost=${self.total_cost_usd:.4f} latency/task={self.mean_latency_s:.2f}s",
            f"strategies={self.strategies} failures={self.failure_classes}" + (f" strategy_acc={self.strategy_accuracy:.2f}" if self.strategy_accuracy is not None else ""),
        ]
        return "\n".join(lines)


def keyword_judge(text: str, keywords: Sequence[str]) -> bool:
    """True when every expected keyword appears in the text (case-insensitive); vacuously true with no keywords."""
    low = text.lower()
    return all(k.lower() in low for k in keywords)


def sample_cases() -> list[EvalCase]:
    """A small synthetic engineering suite exercising every strategy family."""
    return [
        EvalCase(task="Convert 25 degC to kelvin.", expected_strategy="direct"),
        EvalCase(task="Summarise the main risks in the attached design note and recommend mitigations.", domain="software_engineering", inputs={"documents": ["Design note: the service retries failed writes indefinitely and has no circuit breaker."]}, expected_strategy="plan_and_solve"),
        EvalCase(
            task="For each of the three pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status.",
            domain="mechanical_engineering",
            expected_keywords=["duty point"],
            expected_strategy="dag",
        ),
        EvalCase(
            task="First compute the design load, then size the beam section, then check deflection against the limit, and finally report the demand/capacity ratio with the code clause used.",
            domain="civil_structural_engineering",
            expected_strategy="least_to_most",
        ),
        EvalCase(task="Debug why the nightly ETL job fails intermittently and propose a fix with a regression test.", domain="data_engineering", expected_strategy="recursive"),
        EvalCase(task="Compare three alternative gearbox layouts for the stated envelope and select the best trade-off between mass, efficiency and cost.", domain="mechanical_engineering", expected_strategy="tree_of_thoughts"),
        EvalCase(
            task="Review the pump station design: verify hydraulic sizing against the design flow, verify the motor rating against the duty point, check the relief valve setting, and list open risks.",
            domain="chemical_process_engineering",
            expected_strategy="dag",
        ),
    ]


async def evaluate(model: Any, cases: Sequence[EvalCase], *, judge: CoverageJudge = "keyword") -> EvaluationSummary:
    """Run every case through ``model.run`` and aggregate step-wise and end-to-end metrics.

    ``judge`` selects the coverage metric: ``"keyword"`` (model-free, default) or ``"llm"``
    (:func:`judge_coverage` through ``model.llm``; one extra low-effort call per case whose usage is
    added to the case's ``llm_calls``/``cost_usd`` and reported in ``judge_calls``, so
    ``EvaluationSummary.total_cost_usd`` is the true cost of a judged run).
    """
    outcomes: list[CaseOutcome] = []
    for case in cases:
        req = TaskRequest(task=case.task, domain=case.domain, inputs=dict(case.inputs))
        result: TaskResult = await model.run(req)
        out = result.output
        plan_report = PlanQualityReport()
        strategy = ""
        subtasks_ok = False
        replans = 0
        fclasses: dict[str, int] = {}
        judge_usage = Usage()
        if out is not None and hasattr(out, "graph"):
            plan_report = out.metrics if getattr(out, "metrics", None) is not None else evaluate_plan(out.graph, case.task)
            if judge == "llm" and getattr(model, "llm", None) is not None:
                judged, judge_usage = await judge_coverage(out.graph, case.task, llm=model.llm, profile=get_profile(case.domain))
                plan_report = plan_report.model_copy(update={"coverage": judged})
            strategy = out.plan.strategy
            subtasks_ok = bool(out.results) and all(r.ok for r in out.results.values())
            replans = out.replans
            fclasses = failure_classes(out.results)
        elif isinstance(result.metadata.get("plan"), dict):
            strategy = str(result.metadata["plan"].get("strategy", ""))
        final_ok = result.ok and keyword_judge(result.output_text, case.expected_keywords)
        outcomes.append(
            CaseOutcome(
                task=case.task,
                domain=case.domain,
                status=result.status.value,
                strategy=strategy,
                strategy_match=(strategy == case.expected_strategy) if case.expected_strategy else None,
                confidence=result.confidence,
                final_ok=final_ok,
                subtasks_ok=subtasks_ok,
                plan=PlanQualityReport.model_validate(plan_report.model_dump()),  # tolerate a report class from a second import of this module
                llm_calls=result.usage.llm_calls + judge_usage.llm_calls,
                cost_usd=result.usage.cost_usd + judge_usage.cost_usd,
                latency_s=result.usage.latency_s,
                judge_calls=judge_usage.llm_calls,
                replans=replans,
                failure_classes=fclasses,
            )
        )
    n = len(outcomes)
    if n == 0:
        return EvaluationSummary()
    strategies = Counter(o.strategy for o in outcomes)
    failures: Counter[str] = Counter()
    for o in outcomes:
        failures.update(o.failure_classes)
    matched = [o.strategy_match for o in outcomes if o.strategy_match is not None]
    return EvaluationSummary(
        cases=n,
        success_rate=sum(1 for o in outcomes if o.status in ("success", "partial", "cached")) / n,
        needs_review_rate=sum(1 for o in outcomes if o.status == "needs_review") / n,
        keyword_accuracy=sum(1 for o in outcomes if o.final_ok) / n,
        plan_validity_rate=sum(1 for o in outcomes if o.plan.valid) / n,
        strategy_accuracy=(sum(1 for m in matched if m) / len(matched)) if matched else None,
        mean_coverage=sum(o.plan.coverage for o in outcomes) / n,
        mean_parallel_speedup=sum(o.plan.parallel_speedup for o in outcomes) / n,
        compositionality_gap=compositionality_gap((o.subtasks_ok, o.final_ok) for o in outcomes),
        mean_llm_calls=sum(o.llm_calls for o in outcomes) / n,
        total_cost_usd=sum(o.cost_usd for o in outcomes),
        mean_latency_s=sum(o.latency_s for o in outcomes) / n,
        strategies=dict(strategies),
        failure_classes=dict(failures),
        outcomes=outcomes,
    )


def evaluate_sync(model: Any, cases: Sequence[EvalCase], *, judge: CoverageJudge = "keyword") -> EvaluationSummary:
    return run_sync(evaluate(model, cases, judge=judge))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Harness CLI; the implementation is :func:`task_decomposition.eval_cli.main` (kept here for backward compatibility).

    Run it as ``python -m task_decomposition.eval_cli``: that module is not imported by the package
    ``__init__``, so Python does not warn about a double import.
    """
    from .eval_cli import main as _main  # lazy: eval_cli imports this module

    return _main(argv)


__all__ = [
    "CaseOutcome",
    "EvalCase",
    "EvaluationSummary",
    "PlanQualityReport",
    "compositionality_gap",
    "coverage",
    "dependency_correctness",
    "evaluate",
    "evaluate_plan",
    "evaluate_sync",
    "failure_classes",
    "judge_coverage",
    "keyword_judge",
    "llm_coverage",
    "main",
    "parallel_speedup",
    "sample_cases",
    "subtask_success_rate",
]
