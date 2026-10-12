"""``TaskDecompositionModel``: plan -> decompose -> schedule -> replan -> synthesise -> verify.

Life cycle (``PipelineComponent``):

``plan()``     reads the task and returns a :class:`DecompositionPlan` - one
               structured LLM call online, :class:`HeuristicPlanner` offline or
               under ``constraints.deterministic``; the result is normalised to
               the configured bounds and the profile's risk floor.
``execute()``  builds a :class:`TaskGraph` with the chosen strategy, runs it
               wave by wave through the :class:`Scheduler` (bounded concurrency,
               retries, budget stops), replans failed regions at most
               ``max_replans`` times on external signals, composes the answer
               with the :class:`Synthesizer`, optionally verifies it with an
               injected :class:`Verifier`, and sets status/confidence.

With an injected :class:`Verifier`, verification nodes (``tool_hint="verify"``
with dependencies, e.g. the terminal ``verify`` node of high-risk profiles) are
answered by the verifier on the combined upstream outputs and upstream evidence
(:class:`VerifierNodeSolver`), never by the solver; their verdicts are recorded
in ``DecompositionOutput.node_verdicts`` and bound confidence/status, but are
not composed into the answer.

Dependencies are protocol objects (``solver``, ``retriever``, ``verifier``,
``executor``); without them the model is self-contained.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    BaseLLMClient,
    BudgetExceededError,
    ComponentSpec,
    Constraints,
    Evidence,
    Executor,
    LLMError,
    PipelineComponent,
    PlanValidationError,
    Retriever,
    RunContext,
    Status,
    StructuredOutputError,
    SubTaskSolver,
    TaskRequest,
    TaskResult,
    Tracer,
    Usage,
    Verdict,
    Verifier,
    register,
    run_sync,
    validate_structured,
)

from .config import TaskDecompositionConfig
from .evaluation import PlanQualityReport, evaluate_plan, failure_classes, subtask_success_rate
from .graph import SubTask, TaskGraph
from .plan import DecompositionPlan, HeuristicPlanner, TaskFeatures, extract_features, features_text, normalize_plan
from .prompting import build_request
from .prompts import PLAN_TEMPLATE
from .replanner import ReflectionMemory, Replanner, replannable_failures
from .scheduler import RetryPolicy, Scheduler, ScheduleReport, SubTaskResult
from .schemas import Reflection, SynthesisOutput, schema_for
from .solvers import (
    LLMSubTaskSolver,
    NodeSolver,
    ProtocolSolver,
    VerifierNodeSolver,
    charge_verdict_usage,
    evidence_from_inputs,
    is_verification_result,
    node_verdict,
)
from .strategies import RecursiveDecomposer, Strategy, can_call, get_strategy, stop_reason, use_llm_path
from .synthesis import Synthesizer, claims_text


class DecompositionOutput(BaseModel):
    """Typed ``TaskResult.output`` of the model."""

    model_config = ConfigDict(extra="forbid")

    plan: DecompositionPlan
    graph: TaskGraph
    results: dict[str, SubTaskResult] = Field(default_factory=dict)
    synthesis: SynthesisOutput
    metrics: PlanQualityReport = Field(default_factory=PlanQualityReport)
    waves: list[list[str]] = Field(default_factory=list)
    replans: int = 0
    reflections: list[Reflection] = Field(default_factory=list)
    expansions: list[dict[str, Any]] = Field(default_factory=list, description="Recursive expansions performed (task id, depth, children)")
    verdict: Optional[Verdict] = None
    node_verdicts: dict[str, Verdict] = Field(default_factory=dict, description="Verdicts of verification nodes answered by the injected verifier (task id -> verdict)")
    subtask_success_rate: float = 0.0
    failure_classes: dict[str, int] = Field(default_factory=dict)

    def render(self) -> str:
        """Human-readable summary: plan, graph, per-node status, open items."""
        lines = ["PLAN", "  " + self.plan.describe().replace("\n", "\n  "), "", "GRAPH", "  " + self.graph.to_ascii().replace("\n", "\n  "), "", "RESULTS"]
        for tid, r in self.results.items():
            flag = "ok" if r.ok else r.status.value
            extra = f" error={r.error}" if r.error else ""
            v = self.node_verdicts.get(tid)
            if v is not None:
                extra += f" verdict={v.method}:{'passed' if v.passed else 'failed'}"
            lines.append(f"  [{tid}] {flag} confidence={r.confidence:.2f} attempts={r.attempts}{extra}")
        if self.replans:
            lines.append(f"  replans={self.replans}; reflections={[r.change for r in self.reflections]}")
        if self.expansions:
            lines.append(f"  recursive expansions={len(self.expansions)}")
        if self.synthesis.unresolved:
            lines.append("")
            lines.append("UNRESOLVED")
            lines.extend(f"  - {u}" for u in self.synthesis.unresolved)
        lines.append("")
        lines.append(f"METRICS coverage={self.metrics.coverage:.2f} deps={self.metrics.dependency_correctness:.2f} speedup={self.metrics.parallel_speedup:.2f} valid={self.metrics.valid}")
        return "\n".join(lines)


SolveClosure = Callable[[SubTask, dict[str, SubTaskResult]], Awaitable[SubTaskResult]]


@register
class TaskDecompositionModel(PipelineComponent):
    """Turn a complex task into a validated plan, execute it, replan on failure and synthesise the answer."""

    spec = ComponentSpec(
        name="task_decomposition",
        version="1.0.0",
        description="Decomposes a complex task into a validated sequence or DAG of sub-tasks, schedules them with bounded concurrency, solves them through a pluggable solver, replans on external failure signals and synthesises a traceable answer.",
        capabilities=["planning", "decomposition", "dag_scheduling", "recursive_decomposition", "tree_search", "replanning", "synthesis"],
        consumes=["task (natural language)", "inputs.documents (optional)", "context (Evidence)", "metadata.strategy (optional override)", "metadata.graph (optional pre-approved TaskGraph)"],
        produces=["DecompositionOutput(plan, graph, results, synthesis, metrics, verdict)"],
        composes_with=["pipeline_rag", "hybrid_retrieval", "answer_verification", "parallelization", "caching"],
    )

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        config: TaskDecompositionConfig | dict[str, Any] | None = None,
        solver: Optional[SubTaskSolver] = None,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
        executor: Optional[Executor] = None,
        tracer: Tracer | None = None,
        raise_on_error: bool = False,
        **overrides: Any,
    ) -> None:
        base = config if isinstance(config, TaskDecompositionConfig) else TaskDecompositionConfig(**(config or {}))
        cfg = TaskDecompositionConfig(**{**base.model_dump(), **overrides}) if overrides else base
        super().__init__(llm, tracer=tracer, raise_on_error=raise_on_error)
        self.config: TaskDecompositionConfig = cfg  # type: ignore[assignment]
        self.solver = solver
        self.retriever = retriever
        self.verifier = verifier
        self.executor = executor
        self.planner = HeuristicPlanner(cfg)
        self.synthesizer = Synthesizer(cfg)
        self.replanner = Replanner(cfg)

    def describe(self) -> dict[str, Any]:
        return {**self.spec.model_dump(), "llm": self.llm.describe(), "config": self.config.model_dump(), "solver": "protocol" if self.solver else "llm", "retriever": self.retriever is not None, "verifier": self.verifier is not None, "executor": self.executor is not None}

    # ------------------------------------------------------------------ plan
    async def plan(self, request: TaskRequest, ctx: RunContext) -> DecompositionPlan:
        """Decide the decomposition strategy and its bounds (one structured call online, heuristics offline)."""
        cfg = self.config
        features = extract_features(request.task, inputs=request.inputs, context=request.context, profile=ctx.profile)
        ctx.scratch["features"] = features
        override = request.metadata.get("strategy")
        plan: Optional[DecompositionPlan] = None
        if override is None and use_llm_path(ctx, cfg) and can_call(ctx):
            try:
                plan = await self._llm_plan(request, ctx, features)
            except (BudgetExceededError, StructuredOutputError, LLMError) as exc:
                ctx.warn(f"LLM planning failed ({type(exc).__name__}); using heuristic planner")
        if plan is None:
            plan = self.planner.plan(request.task, profile=ctx.profile, constraints=request.constraints, inputs=request.inputs, context=request.context, features=features, strategy_override=override)
            plan.rationale = f"[heuristic] {plan.rationale}"
        else:
            plan.rationale = f"[llm] {plan.rationale}"
        return normalize_plan(plan, cfg, request.constraints, ctx.profile)

    async def _llm_plan(self, request: TaskRequest, ctx: RunContext, features: TaskFeatures) -> DecompositionPlan:
        cfg = self.config
        bounds = (
            f"max_depth<={max(cfg.max_depth, cfg.deep_domain_depth)} breadth={cfg.min_subtasks}-{cfg.max_subtasks} "
            f"parallelism<={min(cfg.max_parallelism, request.constraints.max_parallelism)} search_enabled={str(cfg.enable_search).lower()} "
            f"risk_level={ctx.profile.risk_level} decomposition_gate={cfg.decompose_min_steps} steps"
        )
        req = build_request(
            PLAN_TEMPLATE,
            profile=ctx.profile,
            values={"task": request.task, "features": features_text(features), "bounds": bounds},
            schema=schema_for(DecompositionPlan),
            effort=cfg.planner_effort,
            tags=["plan"],
        )
        resp = await ctx.llm_call(req, step="plan.llm")
        return validate_structured(resp.parsed, DecompositionPlan)

    # --------------------------------------------------------------- execute
    async def execute(self, request: TaskRequest, plan: DecompositionPlan, ctx: RunContext) -> TaskResult:
        """Decompose, schedule, replan, synthesise, verify; never raises for expected conditions."""
        cfg = self.config
        features: TaskFeatures = ctx.scratch.get("features") or extract_features(request.task, inputs=request.inputs, context=request.context, profile=ctx.profile)
        strategy = get_strategy(plan.strategy, cfg)

        # 1. decompose -------------------------------------------------------
        with ctx.step("decompose", strategy=plan.strategy) as st:
            graph = await self._build_graph(request, plan, ctx, strategy, features)
            st.summary = f"{len(graph.tasks)} sub-task(s) via {plan.strategy} [{graph.metadata.get('planner', 'provided')}]"
        metrics = evaluate_plan(graph, request.task, max_nodes=cfg.max_nodes)

        # 2. schedule --------------------------------------------------------
        node_solver: NodeSolver = ProtocolSolver(self.solver, cfg) if self.solver is not None else LLMSubTaskSolver(cfg, retriever=self.retriever, verifier=self.verifier)
        if self.verifier is not None:
            # verification nodes belong to the verifier, not the solver (a RAG solver has nothing to retrieve for them)
            node_solver = VerifierNodeSolver(node_solver, self.verifier, cfg)
        scheduler = Scheduler(
            executor=self.executor,
            retry=RetryPolicy(base_delay_s=cfg.retry_base_delay_s, factor=cfg.retry_factor, max_delay_s=cfg.retry_max_delay_s),
            parallelism=plan.parallelism,
            max_output_chars=cfg.max_subtask_output_chars,
            component=self.spec.name,
        )
        expansions: list[dict[str, Any]] = []

        def should_stop() -> Optional[str]:
            # deadline, call cap or cost cap (Constraints.max_latency_s / max_llm_calls / max_cost_usd)
            return stop_reason(ctx)

        def remaining_calls() -> Optional[int]:
            # the scheduler reserves one call per dispatched node; once the cost cap is reached nothing may start
            if stop_reason(ctx) is not None:
                return 0
            return ctx.budget.remaining_calls()

        def make_solve(current: TaskGraph, depth: int) -> SolveClosure:
            async def _solve(task: SubTask, prior: dict[str, SubTaskResult]) -> SubTaskResult:
                res = await node_solver.solve(task, prior, graph=current, request=request, ctx=ctx, plan=plan, depth=depth)
                if isinstance(strategy, RecursiveDecomposer) and depth < plan.max_depth - 1 and not is_verification_result(res) and self._needs_expansion(res) and can_call(ctx, 2):
                    reason = res.error or f"confidence {res.confidence:.2f} below recursive_expand_below={cfg.recursive_expand_below}"
                    sub = await strategy.expand(task, reason=reason, depth=depth, request=request, plan=plan, ctx=ctx)
                    sub_report = await scheduler.run(sub, make_solve(sub, depth + 1), should_stop=should_stop, remaining_calls=remaining_calls)
                    synth = self.synthesizer.heuristic(task.description, sub, sub_report.results)
                    expansions.append({"task_id": task.id, "depth": depth + 1, "children": sub.ids(), "all_ok": sub_report.all_ok()})
                    if sub_report.completed_ids:
                        usage = Usage()
                        for r in sub_report.results.values():
                            usage = usage + r.usage
                        evidence_ids = sorted({e for r in sub_report.results.values() for e in r.evidence_ids})
                        child_evidence: dict[str, Evidence] = {}
                        for r in sub_report.results.values():
                            for e in r.evidence:
                                child_evidence.setdefault(e.id, e)
                        return SubTaskResult(
                            task_id=task.id,
                            status=Status.SUCCESS if sub_report.all_ok() else Status.PARTIAL,
                            output_text=synth.answer,
                            confidence=synth.confidence,
                            evidence_ids=evidence_ids,
                            evidence=list(child_evidence.values()),  # so the final verifier sees what the children used
                            gaps=synth.unresolved[:5],
                            usage=usage,
                            depth=depth,
                            metadata={"expanded": True, "children": sub.ids(), "child_results": {k: v.model_dump(mode="json") for k, v in sub_report.results.items()}},
                        )
                return res

            return _solve

        with ctx.step("schedule", parallelism=plan.parallelism) as st:
            report = await scheduler.run(graph, make_solve(graph, 0), should_stop=should_stop, remaining_calls=remaining_calls)
            st.summary = self._report_summary(report)

        # 3. replan ----------------------------------------------------------
        memory = ReflectionMemory(cap=cfg.reflection_memory)
        reflections: list[Reflection] = []
        replans = 0
        while plan.replan_on_failure and replans < cfg.max_replans:
            failed = replannable_failures(report)
            if not failed or should_stop():
                break
            with ctx.step(f"replan.{replans + 1}", failed=failed) as st:
                revision = await self.replanner.revise(graph=graph, report=report, request=request, ctx=ctx, plan=plan, memory=memory, round_no=replans + 1)
                if revision is None:
                    st.summary = "nothing replannable"
                    break
                memory.add(revision.reflection)
                reflections.append(revision.reflection)
                if revision.give_up or not revision.replacement_tasks:
                    st.summary = "replanner gave up"
                    ctx.warn(f"replanner gave up on {failed}: {revision.reflection.root_cause[:160]}")
                    break
                try:
                    graph = self.replanner.materialise(graph, revision, failed)
                except PlanValidationError as exc:
                    st.summary = f"revision rejected: {exc}"
                    ctx.warn(f"revision rejected ({exc}); keeping current results")
                    break
                replans += 1
                completed = {tid: r for tid, r in report.results.items() if r.ok and tid in graph.ids()}
                report = await scheduler.run(graph, make_solve(graph, 0), completed=completed, should_stop=should_stop, remaining_calls=remaining_calls)
                st.summary = f"replan {replans}: {self._report_summary(report)}"
        if replans:
            metrics = evaluate_plan(graph, request.task, max_nodes=cfg.max_nodes)

        # 4. synthesise ------------------------------------------------------
        with ctx.step("synthesize") as st:
            # always_append_original_task=False skips the SYNTHESIZE call when every node completed (heuristic
            # sections suffice); the LLM still composes partial results because gaps need an explanation
            use_llm = len(graph.tasks) > 1 and use_llm_path(ctx, cfg) and can_call(ctx) and bool(report.completed_ids) and (cfg.always_append_original_task or not report.all_ok())
            synthesis = await self.synthesizer.synthesize(task=request.task, graph=graph, results=report.results, request=request, ctx=ctx, use_llm=use_llm)
            st.summary = f"{'llm' if use_llm else 'heuristic'} composition; confidence={synthesis.confidence:.2f}"

        # 5. verify ----------------------------------------------------------
        # request context + every sub-task's evidence (including verification nodes' and expanded children's)
        all_evidence = self._collect_evidence(request, report)
        node_verdicts = {tid: v for tid, r in report.results.items() if (v := node_verdict(r)) is not None}
        verdict: Optional[Verdict] = None
        if self.verifier is not None and plan.verification != "none" and synthesis.answer.strip() and report.completed_ids:
            with ctx.step("verify", level=plan.verification) as st:
                try:
                    # claims only (no section headings), checked against the evidence the sub-tasks collected
                    verdict = await self.verifier.verify(claims_text(synthesis.answer), task=request.task, evidence=all_evidence)
                    charge_verdict_usage(verdict, ctx, what="final verifier verdict")  # overshoot is warned about, the verdict kept
                    st.summary = f"{verdict.method}: {'passed' if verdict.passed else 'failed'} score={verdict.score:.2f}"
                except Exception as exc:  # noqa: BLE001 - a broken verifier must not fail the run
                    ctx.warn(f"final verifier raised ({type(exc).__name__}); answer left unverified")
                    st.summary = "verifier error"

        # 6. confidence and status ----------------------------------------------
        confidence, basis, status = self._assess(plan, report, synthesis, verdict, ctx, request, node_verdicts)
        output = DecompositionOutput(
            plan=plan,
            graph=graph,
            results=report.results,
            synthesis=synthesis,
            metrics=metrics,
            waves=report.waves,
            replans=replans,
            reflections=reflections,
            expansions=expansions,
            verdict=verdict,
            node_verdicts=node_verdicts,
            subtask_success_rate=subtask_success_rate(report.results),
            failure_classes=failure_classes(report.results),
        )
        cited = {e for r in report.results.values() for e in r.evidence_ids}
        result = self.new_result(
            request,
            status=status,
            output=output,
            output_text=synthesis.answer.strip() or graph.to_ascii(),
            evidence=[e for e in all_evidence if e.id in cited] if cited else [],
            confidence=confidence,
            confidence_basis=basis,
            errors=[f"{tid}: {r.error}" for tid, r in report.results.items() if r.error],
            metadata={
                "strategy": plan.strategy,
                "planner": graph.metadata.get("planner", "provided"),
                "waves": report.waves,
                "replans": replans,
                "expansions": len(expansions),
                "retries": report.retries,
                "stopped_reason": report.stopped_reason,
                "accept_threshold": self._threshold(ctx, request),
                "node_verdicts": {tid: {"method": v.method, "passed": v.passed, "score": v.score} for tid, v in node_verdicts.items()},
                "plan_metrics": metrics.model_dump(mode="json"),
            },
        )
        result.steps.extend(report.steps)  # one StepRecord per sub-task, collected by the scheduler
        return result

    # ---------------------------------------------------------------- helpers
    async def _build_graph(self, request: TaskRequest, plan: DecompositionPlan, ctx: RunContext, strategy: Strategy, features: TaskFeatures) -> TaskGraph:
        provided = request.metadata.get("graph")
        if provided is not None:
            graph = TaskGraph.model_validate(provided) if not isinstance(provided, TaskGraph) else provided.model_copy(deep=True)
            if graph.metadata.get("planner") not in (None, "provided"):
                graph.metadata["original_planner"] = graph.metadata["planner"]
            graph.metadata["planner"] = "provided"
            issues = graph.validate_structure(max_nodes=self.config.max_nodes, detect_duplicates=self.config.loop_detection)
            if issues:
                raise PlanValidationError("provided graph is invalid: " + "; ".join(issues))
            return graph.infer_dependencies()
        graph = await strategy.build(request.task, request=request, plan=plan, ctx=ctx, features=features)
        issues = graph.validate_structure(max_nodes=self.config.max_nodes, detect_duplicates=self.config.loop_detection)
        if issues:
            ctx.warn(f"{plan.strategy} graph invalid after build ({'; '.join(issues)[:160]}); using heuristic graph")
            graph = strategy.heuristic(request.task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)
            graph.ensure_valid(max_nodes=self.config.max_nodes)
        return graph

    def _needs_expansion(self, res: SubTaskResult) -> bool:
        if not res.ok:
            return True
        return res.confidence < self.config.recursive_expand_below

    @staticmethod
    def _report_summary(report: ScheduleReport) -> str:
        n = len(report.results)
        return f"{len(report.completed_ids)}/{n} ok, {len(report.failed_ids)} failed, {len(report.skipped_ids)} skipped, retries={report.retries}, waves={report.waves_executed}/{len(report.waves)}" + (f", stopped={report.stopped_reason}" if report.stopped_reason else "")

    @staticmethod
    def _collect_evidence(request: TaskRequest, report: ScheduleReport) -> list[Evidence]:
        base = list(request.context) + evidence_from_inputs(request.inputs)
        seen = {e.id for e in base}
        for r in report.results.values():
            for e in r.evidence:
                if e.id not in seen:
                    base.append(e)
                    seen.add(e.id)
        for tid, r in report.results.items():
            if is_verification_result(r):
                continue  # a verdict is not a finding the final answer can be checked against
            if r.ok and r.output_text and f"result:{tid}" not in seen:
                base.append(Evidence(id=f"result:{tid}", content=r.output_text, source=f"subtask:{tid}", score=r.confidence, retriever="provided", metadata={"status": r.status.value}))
                seen.add(f"result:{tid}")
        return base

    @staticmethod
    def _threshold(ctx: RunContext, request: TaskRequest) -> float:
        return max(ctx.profile.min_confidence_to_accept, request.constraints.min_confidence)

    def _assess(
        self,
        plan: DecompositionPlan,
        report: ScheduleReport,
        synthesis: SynthesisOutput,
        verdict: Optional[Verdict],
        ctx: RunContext,
        request: TaskRequest,
        node_verdicts: Optional[dict[str, Verdict]] = None,
    ) -> tuple[float, str, Status]:
        """Derive (confidence, basis, status) from sub-task outcomes, verification state and thresholds.

        Verification nodes answered by the verifier are not counted as content sub-tasks; each verdict
        bounds the confidence like the final verdict does, and a rejected or unavailable check sends the
        run to review.
        """
        cfg = self.config
        content = {tid: r for tid, r in report.results.items() if not is_verification_result(r)}
        checks = {tid: r for tid, r in report.results.items() if is_verification_result(r)}
        node_verdicts = node_verdicts or {}
        n = len(content)
        n_ok = sum(1 for r in content.values() if r.ok)
        conf = synthesis.confidence
        parts = [f"{n_ok}/{n} sub-tasks completed"]
        if n > 1:
            parts.append(f"weakest-link aggregation with {cfg.composition_discount:.0%} composition discount")
        if ctx.offline:
            conf = min(conf, cfg.offline_confidence_cap)
            parts.append(f"offline structural estimate capped at {cfg.offline_confidence_cap:.2f}")
        rejected_checks: list[str] = []
        for tid, r in checks.items():
            nv = node_verdicts.get(tid)
            if nv is None:
                rejected_checks.append(tid)
                parts.append(f"verification node {tid} could not run ({r.error or 'no verdict'})")
                continue
            conf = min(conf, nv.score)
            parts.append(f"verification node {tid} {nv.method} {'passed' if nv.passed else 'failed'} (score {nv.score:.2f})")
            if not nv.passed:
                rejected_checks.append(tid)
        if verdict is not None:
            conf = min(conf, verdict.score)
            parts.append(f"external verifier {verdict.method} {'passed' if verdict.passed else 'failed'} (score {verdict.score:.2f})")
        elif plan.verification in ("external", "strict"):
            conf = min(conf, cfg.unverified_confidence_cap)
            parts.append(f"plan requires {plan.verification} verification but no verifier is injected; capped at {cfg.unverified_confidence_cap:.2f}")
        if report.stopped_reason:
            parts.append(f"stopped early: {report.stopped_reason}")
        threshold = self._threshold(ctx, request)
        conf = max(0.0, min(1.0, conf))
        if n_ok == 0:
            status = Status.FAILED
            parts.append("no sub-task completed")
        elif verdict is not None and not verdict.passed:
            status = Status.NEEDS_REVIEW
            parts.append("verifier rejected the composed answer")
        elif rejected_checks:
            status = Status.NEEDS_REVIEW
            parts.append(f"verification node(s) {', '.join(rejected_checks)} did not pass the sub-task results")
        elif conf < threshold:
            status = Status.NEEDS_REVIEW
            parts.append(f"below accept threshold {threshold:.2f}")
        elif n_ok < n:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS
            parts.append(f"meets accept threshold {threshold:.2f}")
        return conf, "; ".join(parts), status

    # -------------------------------------------------------------- direct API
    def _request(self, task: str | TaskRequest, *, domain: str = "general", inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = (), constraints: Optional[Constraints] = None, strategy: Optional[str] = None) -> TaskRequest:
        if isinstance(task, TaskRequest):
            if strategy:
                task.metadata["strategy"] = strategy
            return task
        return TaskRequest(task=task, domain=domain, inputs=dict(inputs or {}), context=list(context), constraints=constraints or Constraints(), metadata={"strategy": strategy} if strategy else {})

    async def plan_for(self, task: str | TaskRequest, *, domain: str = "general", inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = (), constraints: Optional[Constraints] = None) -> DecompositionPlan:
        """Return only the :class:`DecompositionPlan` for a task (no execution)."""
        request = self._request(task, domain=domain, inputs=inputs, context=context, constraints=constraints)
        ctx = RunContext(request, self.tracer, self.llm, self.spec.name)
        with self.tracer.span(f"{self.spec.name}.plan_for", self.spec.name):
            return await self.plan(request, ctx)

    def plan_for_sync(self, task: str | TaskRequest, **kw: Any) -> DecompositionPlan:
        return run_sync(self.plan_for(task, **kw))

    async def decompose(self, task: str | TaskRequest, *, domain: str = "general", inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = (), constraints: Optional[Constraints] = None, strategy: Optional[str] = None) -> TaskGraph:
        """Plan and decompose without executing: the graph other models or a human reviewer can inspect, edit and hand to :meth:`run_graph`."""
        request = self._request(task, domain=domain, inputs=inputs, context=context, constraints=constraints, strategy=strategy)
        ctx = RunContext(request, self.tracer, self.llm, self.spec.name)
        with self.tracer.span(f"{self.spec.name}.decompose", self.spec.name, request_id=request.request_id, domain=request.domain):
            plan = await self.plan(request, ctx)
            features: TaskFeatures = ctx.scratch["features"]
            graph = await self._build_graph(request, plan, ctx, get_strategy(plan.strategy, self.config), features)
        graph.metadata.update({"plan": plan.model_dump(mode="json"), "usage": ctx.usage.model_dump(mode="json"), "warnings": list(ctx.warnings), "domain": request.domain})
        return graph

    def decompose_sync(self, task: str | TaskRequest, **kw: Any) -> TaskGraph:
        return run_sync(self.decompose(task, **kw))

    async def run_graph(self, graph: TaskGraph, request: str | TaskRequest, **request_kwargs: Any) -> TaskResult:
        """Execute a pre-approved (possibly human-edited) graph instead of planning one (human checkpoint hook)."""
        req = request if isinstance(request, TaskRequest) else TaskRequest(task=request, **request_kwargs)
        req.metadata["graph"] = graph.model_dump(mode="json")
        plan_meta = graph.metadata.get("plan")
        if plan_meta and "strategy" not in req.metadata:
            req.metadata["strategy"] = plan_meta.get("strategy", graph.strategy) if isinstance(plan_meta, dict) else graph.strategy
        elif graph.strategy and "strategy" not in req.metadata:
            req.metadata["strategy"] = graph.strategy
        return await self.run(req)

    def run_graph_sync(self, graph: TaskGraph, request: str | TaskRequest, **request_kwargs: Any) -> TaskResult:
        return run_sync(self.run_graph(graph, request, **request_kwargs))

    def evaluate_plan(self, graph: TaskGraph, task: Optional[str] = None) -> PlanQualityReport:
        """Step-wise plan metrics (coverage, dependency correctness, parallel speedup, validity) without a model call."""
        return evaluate_plan(graph, task, max_nodes=self.config.max_nodes)


__all__ = ["DecompositionOutput", "TaskDecompositionModel"]
