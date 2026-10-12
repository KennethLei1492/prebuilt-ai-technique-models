"""Replanner pure logic (memory, revision application) and synthesis maths."""
from __future__ import annotations

import pytest

from pipeline_core import MockLLM, PlanValidationError, RunContext, Status, TaskRequest, Tracer

from task_decomposition import (
    Reflection,
    ReflectionMemory,
    Replanner,
    ScheduleReport,
    SubTask,
    SubTaskResult,
    Synthesizer,
    TaskGraph,
    aggregate_confidence,
    apply_revision,
    coverage_ratio,
    replannable_failures,
)


def _graph() -> TaskGraph:
    return TaskGraph(
        root_task="root",
        tasks=[
            SubTask(id="t1", description="look up the datasheet"),
            SubTask(id="t2", description="measure the duty point from the test log and compare it with the datasheet"),
            SubTask(id="t3", description="combine $t1.output and $t2.output", depends_on=["t1", "t2"], inputs={"a": "$t1.output", "b": "$t2.output"}),
        ],
    )


def test_reflection_memory_is_bounded_fifo():
    mem = ReflectionMemory(cap=2)
    for i in range(4):
        mem.add(Reflection(root_cause=f"c{i}", change=f"x{i}"))
    assert [r.root_cause for r in mem.entries] == ["c2", "c3"]
    assert "c3" in mem.render() and "c0" not in mem.render()
    assert ReflectionMemory().render().startswith("(no prior")


def test_apply_revision_repoints_dependents_and_placeholders():
    g = _graph()
    repl = [SubTask(id="t2a", description="extract the duty point"), SubTask(id="t2b", description="compare with datasheet", depends_on=["t2a"], inputs={"x": "$t2a.output"})]
    revised = apply_revision(g, ["t2"], repl)
    assert set(revised.ids()) == {"t1", "t3", "t2a", "t2b"}
    t3 = revised.get("t3")
    assert t3.depends_on == ["t1", "t2b"] and t3.inputs["b"] == "$t2b.output" and "$t2b.output" in t3.description
    assert revised.validate_structure() == [] and revised.metadata["revisions"] == 1
    with pytest.raises(PlanValidationError):
        apply_revision(g, ["t2"], [SubTask(id="t1", description="collides")])
    with pytest.raises(PlanValidationError):
        apply_revision(g, ["t2"], [])


def test_replannable_failures_excludes_budget_and_not_started():
    report = ScheduleReport(
        results={
            "a": SubTaskResult(task_id="a", status=Status.FAILED, error="budget: exhausted"),
            "b": SubTaskResult(task_id="b", status=Status.FAILED, error="RuntimeError: boom"),
            "c": SubTaskResult(task_id="c", status=Status.SKIPPED, error="not started: budget"),
            "d": SubTaskResult(task_id="d", status=Status.FAILED, error="verifier: unit mismatch"),
        }
    )
    assert replannable_failures(report) == ["b", "d"]


def test_apply_revision_remaps_placeholders_followed_by_punctuation():
    """Regression: ``$t2,`` / ``($t2)`` / ``$t2;`` must be re-pointed too, or the dependent keeps a dangling literal."""
    g = TaskGraph(
        root_task="root",
        tasks=[
            SubTask(id="t1", description="look up A"),
            SubTask(id="t2", description="look up B"),
            SubTask(id="t3", description="combine $t2, then $t1 and finally ${t2}", depends_on=["t1", "t2"], inputs={"x": "($t2)", "y": "$t2;", "z": "$t2.output", "keep": "$t20"}),
        ],
    )
    revised = apply_revision(g, ["t2"], [SubTask(id="t2a", description="look up B from the datasheet")])
    t3 = revised.get("t3")
    assert t3.description == "combine $t2a, then $t1 and finally ${t2a}"
    assert t3.inputs == {"x": "($t2a)", "y": "$t2a;", "z": "$t2a.output", "keep": "$t20"}  # $t20 is a different id
    assert t3.depends_on == ["t1", "t2a"] and revised.validate_structure() == []


def test_heuristic_revision_splits_once_then_gives_up():
    g = _graph()
    results = {"t2": SubTaskResult(task_id="t2", status=Status.FAILED, error="solver reported failure")}
    rev = Replanner().heuristic_revision(g, ["t2"], results, round_no=1)
    assert not rev.give_up and len(rev.replacement_tasks) == 2
    assert rev.replacement_tasks[1].depends_on == [rev.replacement_tasks[0].id]
    assert rev.reflection.affected_task_ids == ["t2"]
    revised = Replanner().materialise(g, rev, ["t2"])
    assert revised.validate_structure() == []
    again = Replanner().heuristic_revision(revised, [rev.replacement_tasks[0].id], {rev.replacement_tasks[0].id: results["t2"]}, round_no=2)
    assert again.give_up


async def test_llm_revision_path_and_fallback():
    good = {"reflection": {"root_cause": "step too broad", "change": "split", "affected_task_ids": []}, "replacement_tasks": [{"id": "r1", "description": "narrow step"}], "give_up": False}
    llm = MockLLM(responder=lambda req: good)
    req = TaskRequest(task="root")
    from task_decomposition import HeuristicPlanner, TaskDecompositionConfig

    cfg = TaskDecompositionConfig(force_llm_path=True)
    ctx = RunContext(req, Tracer(), llm, "task_decomposition")
    plan = HeuristicPlanner(cfg).plan("root task to do", profile=ctx.profile)
    report = ScheduleReport(results={"t2": SubTaskResult(task_id="t2", status=Status.FAILED, error="RuntimeError: boom")})
    rev = await Replanner(cfg).revise(graph=_graph(), report=report, request=req, ctx=ctx, plan=plan, memory=ReflectionMemory(), round_no=1)
    assert rev is not None and rev.replacement_tasks[0].id == "r1" and rev.reflection.affected_task_ids == ["t2"]
    empty = ScheduleReport(results={"t2": SubTaskResult(task_id="t2", status=Status.FAILED, error="budget: gone")})
    assert await Replanner(cfg).revise(graph=_graph(), report=empty, request=req, ctx=ctx, plan=plan, memory=ReflectionMemory(), round_no=1) is None


def test_aggregate_confidence_weakest_link_discount_and_coverage():
    assert aggregate_confidence([]) == 0.0
    assert aggregate_confidence([0.8]) == pytest.approx(0.8)
    assert aggregate_confidence([0.9, 0.9], composition_discount=0.1) == pytest.approx(0.81)
    assert aggregate_confidence([0.9, 0.3], composition_discount=0.0) == pytest.approx(0.5)  # min + allowance 0.2
    assert aggregate_confidence([0.9, 0.9], composition_discount=0.0, coverage=0.5) == pytest.approx(0.45)
    assert aggregate_confidence([1.5, -1.0]) <= 1.0


def test_heuristic_synthesis_sections_and_unresolved():
    g = _graph()
    results = {
        "t1": SubTaskResult(task_id="t1", output_text="datasheet says 120 m3/h", confidence=0.9, evidence_ids=["doc-1"]),
        "t2": SubTaskResult(task_id="t2", status=Status.FAILED, error="boom"),
        "t3": SubTaskResult(task_id="t3", status=Status.SKIPPED, error="dependency failed: t2"),
    }
    out = Synthesizer().heuristic("root", g, results)
    assert out.answer.startswith("# root") and "## t1" in out.answer and "datasheet says" in out.answer
    assert [c.task_id for c in out.traceability] == ["t1"]
    assert any("t2 (failed)" in u for u in out.unresolved) and any("t3 (skipped)" in u for u in out.unresolved)
    assert coverage_ratio(results, g) == pytest.approx(1 / 3)
    assert out.confidence < 0.5
    single = TaskGraph(root_task="r", tasks=[SubTask(id="t1", description="d")])
    assert Synthesizer().heuristic("r", single, {"t1": SubTaskResult(task_id="t1", output_text="only", confidence=0.7)}).answer == "only"


async def test_llm_synthesis_bounds_confidence_and_falls_back_on_garbage():
    g = _graph()
    results = {tid: SubTaskResult(task_id=tid, output_text=f"out {tid}", confidence=0.8) for tid in g.ids()}
    req = TaskRequest(task="root")
    good = MockLLM(responder=lambda r: {"answer": "composed", "confidence": 0.99, "traceability": [{"task_id": "t1", "contribution": "x"}], "unresolved": []})
    ctx = RunContext(req, Tracer(), good, "task_decomposition")
    out = await Synthesizer().synthesize(task="root", graph=g, results=results, request=req, ctx=ctx, use_llm=True)
    assert out.answer == "composed" and out.confidence <= aggregate_confidence([0.8, 0.8, 0.8]) + 1e-9
    bad = MockLLM(responder=lambda r: "not json")
    ctx2 = RunContext(req, Tracer(), bad, "task_decomposition")
    out2 = await Synthesizer().synthesize(task="root", graph=g, results=results, request=req, ctx=ctx2, use_llm=True)
    assert out2.answer.startswith("# root") and any("heuristic composition" in w for w in ctx2.warnings)
