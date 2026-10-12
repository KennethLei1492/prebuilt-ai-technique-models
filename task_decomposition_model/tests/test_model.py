"""End-to-end behaviour of TaskDecompositionModel: LLM-planned path, status rules, domain adaptation, budgets, injection, direct API."""
from __future__ import annotations

import pytest

from pipeline_core import ENGINEERING_RIGOR, Constraints, Evidence, MockLLM, ScriptedLLM, Status, TaskRequest, Tracer

from task_decomposition import DecompositionOutput, SubTask, TaskDecompositionModel, TaskGraph

from td_fixtures import BRIDGE_TASK, DEBUG_TASK, EXPLORE_TASK, PUMP_TASK, TRIVIAL_TASK, FakeExecutor, FakeRetriever, FakeSolver, FakeVerifier

PLAN_JSON = {
    "strategy": "dag",
    "max_depth": 3,
    "max_breadth": 4,
    "parallelism": 4,
    "verification": "criteria",
    "complexity_estimate": "medium",
    "estimated_steps": 4,
    "replan_on_failure": True,
    "rationale": "Three independent extractions feed one summary.",
}
GRAPH_JSON = {
    "tasks": [
        {"id": "t1", "description": "Extract the duty point from pump log A", "depends_on": [], "inputs": [], "expected_output": "flow and head with units", "verification_criteria": ["Units stated"], "tool_hint": "retrieve", "complexity": "low"},
        {"id": "t2", "description": "Extract the duty point from pump log B", "depends_on": [], "inputs": [], "expected_output": "flow and head with units", "verification_criteria": ["Units stated"], "tool_hint": "retrieve", "complexity": "low"},
        {"id": "t3", "description": "Summarise fleet status from both duty points", "depends_on": ["t1", "t2"], "inputs": [{"name": "a", "value": "$t1.output"}, {"name": "b", "value": "$t2.output"}], "expected_output": "fleet summary", "verification_criteria": ["Mentions both pumps"], "tool_hint": "aggregate", "complexity": "medium"},
    ],
    "notes": "Datasheet assumed at rated temperature.",
}


def _answer(text: str, conf: float = 0.9) -> dict:
    return {"answer": text, "confidence": conf, "status": "completed", "criteria_met": [True], "evidence_ids": [], "gaps": []}


async def test_llm_planned_path_with_scripted_llm(fast_model):
    llm = ScriptedLLM([
        PLAN_JSON,
        GRAPH_JSON,
        _answer("Pump A: 118 m3/h at 31 m."),
        _answer("Pump B: 122 m3/h at 33 m."),
        _answer("Both pumps within 3% of the 120 m3/h duty point."),
        {"answer": "Fleet status: both pumps on the curve; A 118 m3/h, B 122 m3/h.", "confidence": 0.9, "traceability": [{"task_id": "t1", "contribution": "pump A"}, {"task_id": "t2", "contribution": "pump B"}, {"task_id": "t3", "contribution": "summary"}], "unresolved": []},
    ])
    model = fast_model(llm, force_llm_path=True)
    result = await model.run(PUMP_TASK)
    assert result.status == Status.SUCCESS, (result.confidence_basis, result.errors)
    out: DecompositionOutput = result.output
    assert out.plan.strategy == "dag" and out.plan.rationale.startswith("[llm]")
    assert out.graph.metadata["planner"] == "llm" and out.graph.ids() == ["t1", "t2", "t3"]
    assert out.graph.get("t3").depends_on == ["t1", "t2"]
    assert result.output_text.startswith("Fleet status")
    assert result.usage.llm_calls == 6
    assert [c.tags[0] for c in llm.calls] == ["task_decomposition.plan", "task_decomposition.decompose", "task_decomposition.solve", "task_decomposition.solve", "task_decomposition.solve", "task_decomposition.synthesize"]
    # t3 received both upstream outputs through placeholders
    assert "Pump A: 118" in llm.calls[4].last_user_text() and "Pump B: 122" in llm.calls[4].last_user_text()
    assert [c.task_id for c in out.synthesis.traceability] == ["t1", "t2", "t3"]
    # stable blocks carry no volatile data; volatile task text is only in the user turn
    for c in llm.calls:
        assert all("Pump A" not in s for s in c.system_stable)
    step_names = [s.name for s in result.steps]
    assert "plan" in step_names and "decompose" in step_names and "schedule" in step_names and "synthesize" in step_names


async def test_llm_plan_failure_falls_back_to_heuristics(fast_model):
    llm = MockLLM(responder=lambda req: "this is not json" if "task_decomposition.plan" in req.tags else None)

    def responder(req):  # noqa: ANN001
        if "task_decomposition.plan" in req.tags:
            return "garbage"
        from pipeline_core import example_from_schema

        return example_from_schema(req.output_schema) if req.output_schema else "text"

    llm = MockLLM(responder=responder)
    model = fast_model(llm, force_llm_path=True)
    result = await model.run(TRIVIAL_TASK)
    assert result.output.plan.rationale.startswith("[heuristic]")
    assert any("LLM planning failed" in w for w in result.warnings)


async def test_trivial_task_is_one_call(fast_model):
    model = fast_model()
    result = await model.run(TRIVIAL_TASK)
    assert result.status == Status.SUCCESS and result.usage.llm_calls == 1
    assert result.output.plan.strategy == "direct" and len(result.output.graph.tasks) == 1
    assert result.metadata["plan"]["strategy"] == "direct"


async def test_domain_adaptation_safety_critical_vs_general(fast_model):
    model = fast_model()
    general = await model.run(TaskRequest(task=BRIDGE_TASK, domain="general"))
    civil = await model.run(TaskRequest(task=BRIDGE_TASK, domain="civil_structural_engineering"))
    assert general.status == Status.SUCCESS
    assert civil.status == Status.NEEDS_REVIEW
    assert general.output.plan.verification in ("criteria", "none") and civil.output.plan.verification == "strict"
    assert "verify" not in general.output.graph.ids() and "verify" in civil.output.graph.ids()
    assert "no verifier is injected" in civil.confidence_basis and "below accept threshold 0.85" in civil.confidence_basis
    llm: MockLLM = model.llm  # type: ignore[assignment]
    civil_calls = [c for c in llm.calls if "Risk level: safety_critical" in "\n".join(c.system_stable)]
    general_calls = [c for c in llm.calls if "Risk level: low" in "\n".join(c.system_stable)]
    assert civil_calls and general_calls
    assert all(ENGINEERING_RIGOR in c.system_stable for c in civil_calls)
    assert all(ENGINEERING_RIGOR not in c.system_stable for c in general_calls)
    assert all("Common units: kN" in "\n".join(c.system_stable) for c in civil_calls)


async def test_needs_review_below_requested_min_confidence(fast_model):
    model = fast_model()
    ok = await model.run(TaskRequest(task=PUMP_TASK, constraints=Constraints(min_confidence=0.5)))
    strict = await model.run(TaskRequest(task=PUMP_TASK, constraints=Constraints(min_confidence=0.95)))
    assert ok.status == Status.SUCCESS and strict.status == Status.NEEDS_REVIEW
    assert strict.metadata["accept_threshold"] == 0.95
    assert strict.confidence <= model.config.offline_confidence_cap


async def test_budget_enforcement_degrades_to_partial(fast_model):
    model = fast_model()
    result = await model.run(TaskRequest(task=BRIDGE_TASK, constraints=Constraints(max_llm_calls=2)))
    assert result.usage.llm_calls <= 2
    assert result.status in (Status.PARTIAL, Status.NEEDS_REVIEW)
    out: DecompositionOutput = result.output
    assert len(out.results) == len(out.graph.tasks)
    assert any((r.error or "").startswith("budget") or (r.error or "").startswith("not started: budget") for r in out.results.values())
    assert out.replans == 0  # budget failures are not replannable
    assert "stopped early: budget" in result.confidence_basis or result.metadata["stopped_reason"] == "budget"
    assert out.failure_classes.get("budget_or_deadline", 0) + out.failure_classes.get("skipped_dependency", 0) >= 1


async def test_cost_budget_stops_paid_calls_after_exhaustion(fast_model):
    """Regression: with only max_cost_usd set, no further LLM call is issued once the cap is exceeded."""
    llm = MockLLM(simulate_cost_as="claude-opus-5-5")
    model = fast_model(llm)
    result = await model.run(TaskRequest(task=BRIDGE_TASK, constraints=Constraints(max_cost_usd=1e-6)))
    out: DecompositionOutput = result.output
    assert len(out.graph.tasks) >= 3
    assert len(llm.calls) == 1, [c.tags for c in llm.calls]  # the first paid call trips the cap; nothing else is dispatched
    assert result.usage.llm_calls == 1 and result.usage.cost_usd > 1e-6
    assert result.status == Status.FAILED and result.metadata["stopped_reason"] == "budget"
    assert all((r.error or "").startswith("budget") for r in out.results.values())
    assert out.failure_classes == {"budget_or_deadline": len(out.graph.tasks)} and out.replans == 0
    assert "stopped early: budget" in result.confidence_basis
    # a cost cap that is never reached changes nothing
    free = await fast_model(MockLLM(simulate_cost_as="claude-opus-5-5")).run(TaskRequest(task=BRIDGE_TASK, constraints=Constraints(max_cost_usd=10.0)))
    assert free.ok and free.metadata["stopped_reason"] is None


async def test_nested_solver_overshoot_keeps_its_result_and_stops_dispatch(fast_model):
    """Regression: a nested SubTaskSolver that spends more than the reserved call keeps its successful output;
    the overshoot is charged and warned about, then the scheduler stops further dispatch."""

    class GreedySolver:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, request: TaskRequest):  # noqa: ANN202
            from pipeline_core import TaskResult, Usage

            self.calls += 1
            return TaskResult(request_id=request.request_id, component="greedy", output_text="GOOD ANSWER", confidence=0.9, usage=Usage(llm_calls=2))

    solver = GreedySolver()
    model = fast_model(solver=solver)
    result = await model.run(TaskRequest(task=PUMP_TASK, constraints=Constraints(max_llm_calls=1)))
    out: DecompositionOutput = result.output
    assert solver.calls == 1 and result.usage.llm_calls == 2
    first = next(r for r in out.results.values() if r.attempts)
    assert first.ok and first.output_text == "GOOD ANSWER" and first.error is None
    assert any(w.startswith("budget exceeded after nested sub-task") for w in result.warnings)
    assert result.metadata["stopped_reason"] == "budget" and out.replans == 0
    assert all((r.error or "").startswith("not started: budget") for r in out.results.values() if not r.ok)
    assert result.status in (Status.PARTIAL, Status.NEEDS_REVIEW) and "GOOD ANSWER" in result.output_text


async def test_always_append_original_task_false_skips_the_synthesis_call(fast_model):
    """Regression: the flag was dead configuration; False must skip SYNTHESIZE when every node completed."""
    from pipeline_core import example_from_schema

    SYNTH_JSON = {"answer": "Fleet status: both pumps on the curve.", "confidence": 0.9, "traceability": [{"task_id": "t3", "contribution": "summary"}], "unresolved": []}

    def responder(req):  # noqa: ANN001
        if "task_decomposition.decompose" in req.tags:
            return GRAPH_JSON
        if "task_decomposition.synthesize" in req.tags:
            return SYNTH_JSON  # example_from_schema would answer with confidence 0.0 and force NEEDS_REVIEW
        return example_from_schema(req.output_schema) if req.output_schema else "text"

    calls: dict[bool, list[str]] = {}
    answers: dict[bool, str] = {}
    for flag in (True, False):
        llm = MockLLM(responder=responder)
        model = fast_model(llm, force_llm_path=True, always_append_original_task=flag)
        result = await model.run(TaskRequest(task=PUMP_TASK, metadata={"strategy": "dag"}))
        assert result.ok and result.output.graph.ids() == ["t1", "t2", "t3"] and all(r.ok for r in result.output.results.values()), result.confidence_basis
        calls[flag] = [c.tags[0] for c in llm.calls]
        answers[flag] = result.output_text
    assert "task_decomposition.synthesize" in calls[True]
    assert "task_decomposition.synthesize" not in calls[False]
    assert len(calls[True]) == len(calls[False]) + 1
    assert answers[True].startswith("Fleet status") and answers[False].startswith(f"# {PUMP_TASK}")  # heuristic sections when skipped
    # with a failed node the LLM still composes the partial answer, flag or not
    def partial(req):  # noqa: ANN001
        if "task_decomposition.decompose" in req.tags:
            return GRAPH_JSON
        if "task_decomposition.solve" in req.tags and 'id="t1"' in req.last_user_text().split("</sub_task>")[0]:
            return {"answer": "", "confidence": 0.1, "status": "failed", "criteria_met": [], "evidence_ids": [], "gaps": ["no log"]}
        return example_from_schema(req.output_schema) if req.output_schema else "text"

    llm = MockLLM(responder=partial)
    model = fast_model(llm, force_llm_path=True, always_append_original_task=False, max_replans=0)
    result = await model.run(TaskRequest(task=PUMP_TASK, metadata={"strategy": "dag"}))
    assert not result.output.results["t1"].ok and any(c.tags[0] == "task_decomposition.synthesize" for c in llm.calls)


async def test_deadline_stops_dispatch_and_is_reported_as_deadline(fast_model):
    """Regression: a real max_latency_s stop is classed 'deadline' (not 'budget') and bucketed budget_or_deadline."""
    llm = MockLLM(latency_s=0.05)
    model = fast_model(llm)
    result = await model.run(TaskRequest(task=PUMP_TASK, constraints=Constraints(max_latency_s=0.02)))
    out: DecompositionOutput = result.output
    assert result.metadata["stopped_reason"] == "deadline", result.confidence_basis
    assert "stopped early: deadline" in result.confidence_basis
    assert out.failure_classes.get("budget_or_deadline", 0) >= 1 and "skipped_dependency" not in out.failure_classes
    assert out.replans == 0
    assert len(llm.calls) < len(out.graph.tasks) + 1  # at least the synthesis call was not made


async def test_zero_call_budget_still_returns_a_plan_and_graph(fast_model):
    model = fast_model()
    result = await model.run(TaskRequest(task=PUMP_TASK, constraints=Constraints(max_llm_calls=0)))
    assert result.status == Status.FAILED and result.usage.llm_calls == 0
    assert result.output is not None and len(result.output.graph.tasks) >= 3
    assert result.output_text.startswith("wave 1:")  # graph rendered when nothing could be solved


async def test_recursive_expansion_on_external_failure(fast_model):
    from pipeline_core import example_from_schema

    def responder(req):  # noqa: ANN001
        text = req.last_user_text()
        if '<sub_task id="t1">' in text and DEBUG_TASK in text.split("</sub_task>")[0]:
            return {"answer": "", "confidence": 0.1, "status": "failed", "criteria_met": [], "evidence_ids": [], "gaps": ["cannot reproduce without logs"]}
        return example_from_schema(req.output_schema) if req.output_schema else "x"

    model = fast_model(MockLLM(responder=responder))
    result = await model.run(TaskRequest(task=DEBUG_TASK, domain="data_engineering"))
    out: DecompositionOutput = result.output
    assert out.plan.strategy == "recursive"
    assert len(out.expansions) == 1 and out.expansions[0]["task_id"] == "t1" and out.expansions[0]["depth"] == 1
    assert out.results["t1"].ok and out.results["t1"].metadata["expanded"] is True
    assert len(out.results["t1"].metadata["children"]) >= 2
    # ADaPT: the failed whole-task attempt is not retried verbatim; it is decomposed (1 failed call + one per child)
    assert result.ok and result.usage.llm_calls == 1 + len(out.results["t1"].metadata["children"])
    assert out.results["t1"].attempts == 1
    assert out.results["t1"].metadata["children"][0] == "t1"  # ids are numbered in execution order


async def test_verifier_rejection_marks_needs_review_and_passing_verdict_is_recorded(fast_model):
    bad = FakeVerifier(passed=False)
    model = fast_model(verifier=bad)
    result = await model.run(TaskRequest(task=PUMP_TASK, domain="software_engineering"))
    assert result.status == Status.NEEDS_REVIEW and "verifier rejected" in result.confidence_basis
    assert result.output.verdict is not None and not result.output.verdict.passed and len(bad.calls) == 1
    good = FakeVerifier(passed=True, score=0.9)
    model2 = fast_model(verifier=good)
    result2 = await model2.run(TaskRequest(task=PUMP_TASK, domain="software_engineering"))
    assert result2.status == Status.SUCCESS and result2.output.verdict.passed
    assert result2.usage.llm_calls == len(result2.output.graph.tasks) + 1  # verifier usage charged from verdict.details


async def test_strict_verification_failure_triggers_replanning(fast_model):
    verifier = FakeVerifier(fail_when="flag deviations")
    model = fast_model(verifier=verifier)
    result = await model.run(TaskRequest(task=PUMP_TASK, domain="civil_structural_engineering"))
    out: DecompositionOutput = result.output
    assert out.plan.verification == "strict"
    assert out.replans >= 1 and out.reflections and "verifier" in out.reflections[0].root_cause
    assert any("narrowed from" in t.description for t in out.graph.tasks)
    assert result.status in (Status.NEEDS_REVIEW, Status.PARTIAL)
    assert any((r.error or "").startswith("verifier:") for r in out.results.values())
    assert any(s.name == "replan.1" for s in result.steps)


async def test_retriever_adds_evidence_to_prompts_and_result(fast_model):
    retriever = FakeRetriever()
    model = fast_model(retriever=retriever)
    result = await model.run(TaskRequest(task=PUMP_TASK, domain="general"))
    assert retriever.queries
    llm: MockLLM = model.llm  # type: ignore[assignment]
    assert any('id="doc-x"' in c.last_user_text() for c in llm.calls)
    assert any(e.id == "doc-x" for e in result.evidence)
    assert all("doc-x" in r.evidence_ids for r in result.output.results.values())


async def test_protocol_solver_receives_dependency_outputs_as_evidence(fast_model):
    solver = FakeSolver()
    model = fast_model(solver=solver)
    result = await model.run(TaskRequest(task=PUMP_TASK, domain="general"))
    out: DecompositionOutput = result.output
    assert result.ok and all(r.metadata["component"] == "fake_rag" for r in out.results.values())
    assert result.usage.llm_calls == len(out.graph.tasks) and result.usage.cost_usd == pytest.approx(0.01 * len(out.graph.tasks))
    assert model.llm.calls == []  # the protocol solver never touches the model's own LLM
    # nested usage is charged against the request budget, so a budget of 1 nested call stops the run early
    capped = await fast_model(solver=FakeSolver()).run(TaskRequest(task=PUMP_TASK, constraints=Constraints(max_llm_calls=1)))
    assert capped.usage.llm_calls <= 1 and capped.status in (Status.PARTIAL, Status.NEEDS_REVIEW, Status.FAILED)
    last = out.graph.ids()[-1]
    dep_request = next(r for r in solver.requests if r.metadata["subtask_id"] == last)
    assert dep_request.parent_id and any(e.id.startswith("result:") for e in dep_request.context)
    assert dep_request.metadata["root_task"] == PUMP_TASK
    assert all(r.output_text.startswith("solved:") for r in out.results.values())


async def test_executor_injection_and_parallelism_constraint(fast_model):
    executor = FakeExecutor()
    model = fast_model(executor=executor)
    result = await model.run(TaskRequest(task=BRIDGE_TASK, constraints=Constraints(max_parallelism=2)))
    assert result.ok and result.output.plan.parallelism == 2
    assert executor.concurrencies and all(c == 2 for c in executor.concurrencies)


async def test_strategy_override_and_tree_of_thoughts_offline(fast_model):
    model = fast_model()
    chain = await model.run(TaskRequest(task=BRIDGE_TASK, metadata={"strategy": "least_to_most"}))
    g = chain.output.graph
    assert chain.output.plan.strategy == "least_to_most" and g.depth() == len(g.tasks)
    tot = await model.run(TaskRequest(task=EXPLORE_TASK, domain="mechanical_engineering"))
    assert tot.output.plan.strategy == "tree_of_thoughts" and "candidate_scores" in tot.output.graph.metadata
    assert tot.status == Status.NEEDS_REVIEW  # high-risk domain without a verifier cannot auto-accept


async def test_direct_api_decompose_plan_for_and_run_graph(fast_model):
    model = fast_model()
    plan = await model.plan_for(PUMP_TASK, domain="mechanical_engineering")
    assert plan.strategy == "dag" and plan.verification == "external"
    graph = await model.decompose(PUMP_TASK, domain="mechanical_engineering")
    assert isinstance(graph, TaskGraph) and graph.metadata["plan"]["strategy"] == "dag" and graph.metadata["usage"]["llm_calls"] == 0
    edited = graph.with_tasks([t for t in graph.tasks if t.id != "verify"][:2])
    edited.strategy = "dag"
    result = await model.run_graph(edited, TaskRequest(task=PUMP_TASK, domain="general"))
    assert result.ok and result.output.graph.ids() == edited.ids() and result.metadata["planner"] == "provided"
    assert model.evaluate_plan(graph).valid
    sync_graph = model.decompose_sync(TRIVIAL_TASK)
    assert len(sync_graph.tasks) == 1 and model.plan_for_sync(TRIVIAL_TASK).strategy == "direct"
    bad = TaskGraph(root_task=PUMP_TASK, tasks=[SubTask(id="a", description="x", depends_on=["b"]), SubTask(id="b", description="y", depends_on=["a"])])
    failed = await model.run_graph(bad, PUMP_TASK)
    assert failed.status == Status.FAILED and any("PlanValidationError" in e for e in failed.errors)


async def test_result_evidence_and_tracing(fast_model):
    tracer = Tracer()
    model = TaskDecompositionModel(MockLLM(), tracer=tracer, retry_base_delay_s=0.0)
    ev = [Evidence(id="spec-1", content="Design flow 120 m3/h.", source="spec")]
    result = await model.run(TaskRequest(task=PUMP_TASK, context=ev, inputs={"documents": ["Log A: 118 m3/h"]}))
    assert {e.id for e in result.evidence} >= {"spec-1", "doc-1"}
    assert result.trace_id == tracer.trace_id and "task_decomposition.run" in tracer.tree()
    assert model.describe()["config"]["max_replans"] == 2 and model.describe()["solver"] == "llm"


def test_run_sync_and_config_overrides():
    model = TaskDecompositionModel(MockLLM(), config={"max_subtasks": 4}, max_replans=1)
    assert model.config.max_subtasks == 4 and model.config.max_replans == 1
    result = model.run_sync(TRIVIAL_TASK)
    assert result.ok and result.output.plan.strategy == "direct"


async def test_verify_nodes_use_max_effort_only_in_safety_critical_domains(fast_model):
    model = fast_model()
    civil = await model.run(TaskRequest(task=BRIDGE_TASK, domain="civil_structural_engineering"))
    llm: MockLLM = model.llm  # type: ignore[assignment]
    # solve-call tags are [template, version, "solve", <task id>, <tool hint>]
    final_verify = [c for c in llm.calls if c.tags[2:4] == ["solve", "verify"]]
    others = [c for c in llm.calls if c.tags[2] == "solve" and c.tags[3] != "verify"]
    assert "verify" in civil.output.graph.ids() and len(final_verify) == 1 and others
    assert final_verify[0].effort == model.config.safety_critical_verification_effort == "max"
    assert all(c.effort == model.config.solver_effort for c in others)  # intermediate checks keep the solver effort
    llm.reset()
    await model.run(TaskRequest(task=BRIDGE_TASK, domain="mechanical_engineering"))  # high risk, not safety-critical
    assert all(c.effort != "max" for c in llm.calls if c.tags[2] == "solve")
