"""Verification nodes with an injected verifier: routed to the verifier, recorded as verdicts, never composed as content.

Regression for the integration defect where the terminal ``verify`` node of high-risk profiles was sent to an
injected RAG solver, which answered "Insufficient evidence" (PARTIAL, confidence 0); that text was composed into
the answer and the final verifier rejected it, so every run ended NEEDS_REVIEW at ~0.18 confidence.
"""
from __future__ import annotations

from typing import Sequence

import pytest

from pipeline_core import Evidence, MockLLM, RunContext, Status, TaskRequest, TaskResult, Tracer, Usage, Verdict

from task_decomposition import (
    SubTask,
    SubTaskResult,
    Synthesizer,
    TaskGraph,
    claims_text,
    coverage_ratio,
    is_verification_node,
    is_verification_result,
)

from td_fixtures import BRIDGE_TASK, FakeSolver, FakeVerifier

PUMP_STATION_TASK = "Review the pump station design: check the corrosion allowance, verify the motor rating against the measured shaft power, and list open risks."
INSUFFICIENT = "Insufficient evidence to answer. No evidence was retrieved."


class RAGLikeSolver:
    """Stands in for ``PipelineRAGModel.solver``: grounded answers for content nodes, nothing to retrieve for "verify the results"."""

    def __init__(self) -> None:
        self.requests: list[TaskRequest] = []

    async def __call__(self, request: TaskRequest) -> TaskResult:
        self.requests.append(request)
        sid = str(request.metadata.get("subtask_id"))
        if request.task.lower().startswith("independently verify"):
            return TaskResult(request_id=request.request_id, component="fake_rag", status=Status.PARTIAL, output_text=INSUFFICIENT, confidence=0.0, usage=Usage(llm_calls=1))
        ev = Evidence(id=f"doc-{sid}", content=f"source for {request.task}", source="index", score=0.8, retriever="fused")
        return TaskResult(request_id=request.request_id, component="fake_rag", output_text=f"finding for {sid} [doc-{sid}]", confidence=0.85, evidence=[ev], usage=Usage(llm_calls=1, cost_usd=0.01))


class RaisingVerifier:
    """Verifies the final answer but raises when asked to check combined sub-task results."""

    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("verifier backend down")
        return Verdict(method="fake", passed=True, score=0.9)


class NodeRejectingVerifier(FakeVerifier):
    """Rejects only the verification-node call (the first one); passes the final composed answer."""

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict:
        verdict = await super().verify(answer, task=task, evidence=evidence)
        if len(self.calls) == 1:
            return Verdict(method="fake", passed=False, score=0.3, issues=["motor rating not compared with shaft power"], details={"usage": Usage(llm_calls=1).model_dump()})
        return verdict


def _requests_for(solver: RAGLikeSolver | FakeSolver, subtask_id: str) -> list[TaskRequest]:
    return [r for r in solver.requests if r.metadata.get("subtask_id") == subtask_id]


# --------------------------------------------------------------------- with a verifier


async def test_verify_node_goes_to_verifier_not_solver(fast_model):
    solver, verifier = RAGLikeSolver(), FakeVerifier(score=0.9)
    model = fast_model(solver=solver, verifier=verifier)
    result = await model.run(TaskRequest(task=PUMP_STATION_TASK, domain="mechanical_engineering"))
    out = result.output
    assert "verify" in out.graph.ids()
    content_ids = [t for t in out.graph.ids() if t != "verify"]

    # the solver never sees the verification node; the dependency-free "Verify the motor rating ..." node is content
    assert not _requests_for(solver, "verify")
    motor = [t for t in out.graph.tasks if t.description.lower().startswith("verify the motor rating")]
    assert motor and motor[0].tool_hint == "verify" and not motor[0].depends_on and not is_verification_node(motor[0])
    assert _requests_for(solver, motor[0].id)

    # node verdict: verifier called on the combined upstream outputs, root task, upstream evidence (no result:* entries)
    node_answer, node_task = verifier.calls[0]
    assert node_task == PUMP_STATION_TASK
    assert all(f"finding for {tid}" in node_answer for tid in content_ids)
    assert verifier.evidence[0] == [f"doc-{tid}" for tid in content_ids]
    v = out.results["verify"]
    assert is_verification_result(v) and v.status == Status.SUCCESS and v.confidence == pytest.approx(0.9)
    assert v.metadata["checked_results"] == content_ids and v.metadata["verdict"]["passed"] is True
    assert v.metadata["component"] == "fake"  # spec.name for pipeline components, else the verdict method
    assert out.node_verdicts["verify"].passed and result.metadata["node_verdicts"]["verify"]["score"] == pytest.approx(0.9)

    # the verdict is not content: not composed, not in traceability, not offered to the final verifier as a finding
    assert INSUFFICIENT not in result.output_text and "fake: passed" not in result.output_text and "## verify" not in result.output_text
    assert all(c.task_id != "verify" for c in out.synthesis.traceability)

    # final verification: claims only, with the evidence the sub-tasks collected
    final_answer, _ = verifier.calls[-1]
    assert len(verifier.calls) == 2 and not final_answer.lstrip().startswith("#")
    assert {f"doc-{tid}" for tid in content_ids} <= set(verifier.evidence[-1]) and "result:verify" not in verifier.evidence[-1]

    # status/confidence reflect the sub-tasks: offline cap 0.70 (< 0.75 mechanical threshold), not ~0.18
    assert result.confidence == pytest.approx(0.70)
    assert f"{len(content_ids)}/{len(content_ids)} sub-tasks completed" in result.confidence_basis
    assert "verification node verify fake passed (score 0.90)" in result.confidence_basis
    assert "rejected" not in result.confidence_basis and result.status == Status.NEEDS_REVIEW and "below accept threshold 0.75" in result.confidence_basis

    # usage: one call per content node from the solver + node verdict + final verdict (from verdict.details["usage"])
    assert result.usage.llm_calls == len(content_ids) + 2
    assert v.usage.llm_calls == 1


async def test_failed_node_verdict_flags_review_without_retry_or_replan(fast_model):
    verifier = NodeRejectingVerifier()
    model = fast_model(solver=FakeSolver(), verifier=verifier)
    result = await model.run(TaskRequest(task=PUMP_STATION_TASK, domain="mechanical_engineering"))
    out = result.output
    v = out.results["verify"]
    assert v.status == Status.NEEDS_REVIEW and v.attempts == 1 and v.confidence == pytest.approx(0.3)
    assert (v.error or "").startswith("verifier:") and out.failure_classes.get("verification_failed") == 1
    assert out.replans == 0 and not out.node_verdicts["verify"].passed
    assert result.status == Status.NEEDS_REVIEW and result.confidence <= 0.3
    assert "verification node(s) verify did not pass" in result.confidence_basis
    assert any(u.startswith("verify (needs_review): verifier:") for u in out.synthesis.unresolved)
    assert "motor rating not compared" not in result.output_text  # recorded as an open item, not as content


async def test_raising_verifier_on_verify_node_degrades_to_review(fast_model):
    model = fast_model(solver=FakeSolver(), verifier=RaisingVerifier())
    result = await model.run(TaskRequest(task=PUMP_STATION_TASK, domain="mechanical_engineering"))
    v = result.output.results["verify"]
    assert is_verification_result(v) and v.status == Status.NEEDS_REVIEW and v.metadata["verdict"] is None
    assert "verify" not in result.output.node_verdicts
    assert any("verification node verify" in w for w in result.warnings)
    assert result.status == Status.NEEDS_REVIEW and "verification node verify could not run" in result.confidence_basis


async def test_llm_solver_path_also_hands_verify_nodes_to_the_verifier(fast_model):
    verifier = FakeVerifier(score=0.95)
    model = fast_model(verifier=verifier)
    result = await model.run(TaskRequest(task=BRIDGE_TASK, domain="mechanical_engineering"))
    llm: MockLLM = model.llm  # type: ignore[assignment]
    assert "verify" in result.output.graph.ids()
    assert not [c for c in llm.calls if c.tags[2:4] == ["solve", "verify"]]  # no LLM solve for the verification node
    assert result.output.node_verdicts["verify"].passed and verifier.calls[0][1] == BRIDGE_TASK


# ------------------------------------------------------------------ without a verifier


async def test_without_verifier_verify_node_still_goes_to_solver(fast_model):
    solver = RAGLikeSolver()
    model = fast_model(solver=solver)
    result = await model.run(TaskRequest(task=PUMP_STATION_TASK, domain="mechanical_engineering"))
    out = result.output
    assert _requests_for(solver, "verify")  # unchanged behaviour: the solver answers every node
    assert not is_verification_result(out.results["verify"]) and out.node_verdicts == {}
    assert out.results["verify"].output_text == INSUFFICIENT
    assert "no verifier is injected" in result.confidence_basis


# ------------------------------------------------------------------------- pure helpers


def test_is_verification_node_requires_hint_and_dependencies():
    assert is_verification_node(SubTask(id="v", description="Independently verify the results", depends_on=["t1"], tool_hint="verify"))
    assert is_verification_node(SubTask(id="v", description="Check t1", depends_on=["t1"], tool_hint="Verification"))
    assert not is_verification_node(SubTask(id="t2", description="Verify the motor rating against shaft power", tool_hint="verify"))
    assert not is_verification_node(SubTask(id="t3", description="Summarise", depends_on=["t1"], tool_hint="aggregate"))


def test_claims_text_drops_heading_scaffolding():
    answer = "# Review the design\n\n## t1: Check the allowance\nAllowance is 3 mm [doc-1].\n\n## t2: List risks\n- seal supplier [doc-2]"
    assert claims_text(answer) == "Allowance is 3 mm [doc-1].\n\n- seal supplier [doc-2]"
    assert claims_text("# only a heading") == "# only a heading"


def _graph_with_verify() -> tuple[TaskGraph, dict[str, SubTaskResult]]:
    graph = TaskGraph(
        root_task="Check the allowance and list risks",
        tasks=[
            SubTask(id="t1", description="Check the allowance"),
            SubTask(id="t2", description="List risks"),
            SubTask(id="verify", description="Independently verify the combined sub-task results", depends_on=["t1", "t2"], tool_hint="verify"),
        ],
    )
    results = {
        "t1": SubTaskResult(task_id="t1", output_text="Allowance is 3 mm.", confidence=0.8),
        "t2": SubTaskResult(task_id="t2", output_text="Seal supplier change is open.", confidence=0.8),
        "verify": SubTaskResult(task_id="verify", status=Status.NEEDS_REVIEW, output_text="fake: failed (score 0.30)", confidence=0.3, error="verifier: units missing", metadata={"verification_node": True}),
    }
    return graph, results


def test_heuristic_synthesis_excludes_verdict_text_and_coverage():
    graph, results = _graph_with_verify()
    synth = Synthesizer().heuristic(graph.root_task, graph, results)
    assert "fake: failed" not in synth.answer and "## verify" not in synth.answer
    assert [c.task_id for c in synth.traceability] == ["t1", "t2"]
    assert "verify (needs_review): verifier: units missing" in synth.unresolved
    assert coverage_ratio(results, graph) == 1.0  # the verdict is not a missing content node
    assert synth.confidence == pytest.approx(0.8 * 0.9)
    # without the verification marker the same node is ordinary (failed) content, as before
    plain: dict[str, SubTaskResult] = {**results, "verify": results["verify"].model_copy(update={"metadata": {}})}
    assert coverage_ratio(plain, graph) == pytest.approx(2 / 3)


async def test_llm_synthesis_omits_verdicts_from_composition():
    graph, results = _graph_with_verify()
    llm = MockLLM()
    request = TaskRequest(task=graph.root_task)
    ctx = RunContext(request, Tracer(), llm, "task_decomposition")
    out = await Synthesizer().synthesize(task=graph.root_task, graph=graph, results=results, request=request, ctx=ctx, use_llm=True)
    prompt = str(llm.calls[-1].messages)
    assert 'id="t1"' in prompt and 'id="verify"' not in prompt and "fake: failed" not in prompt
    assert any(u.startswith("verify (needs_review)") for u in out.unresolved)
    assert all(c.task_id != "verify" for c in out.traceability)
