"""End-to-end behaviour of ParallelizationModel: LLM-planned path, status/confidence, budgets, domains, direct API."""
from __future__ import annotations

import itertools
from typing import Any

import pytest

from pipeline_core import ENGINEERING_RIGOR, Constraints, Executor, LLMError, LLMRequest, MockLLM, ScriptedLLM, Status, TaskRequest

from parallelization import ParallelizationModel
from parallelization.config import ParallelizationConfig
from parallelization.executor import ManualClock, ParallelExecutor, RetryPolicy

PLAN_JSON: dict[str, Any] = {
    "pattern": "sectioning",
    "partitions": [
        {"id": "thermal", "task": "Assess the thermal margin of the enclosure.", "depends_on": []},
        {"id": "structural", "task": "Assess the structural margin of the enclosure.", "depends_on": []},
        {"id": "summary", "task": "Summarise both margins.", "depends_on": ["thermal", "structural"]},
    ],
    "samples": 5,
    "aggregator": "llm",
    "verifiability": "free_form",
    "rationale": "Two independent assessments feeding a summary.",
}
SOLUTION = {"answer": "margin is adequate", "evidence": ["datasheet"], "rationale": "numbers", "confidence": "high"}
MERGE = {"answer": "Both margins are adequate.", "unresolved": [], "confidence": "high"}


def fast_executor(cfg: ParallelizationConfig | None = None) -> ParallelExecutor:
    return ParallelExecutor.from_config(cfg or ParallelizationConfig(), clock=ManualClock())


async def test_llm_planned_path_with_scripted_llm():
    llm = ScriptedLLM([PLAN_JSON, SOLUTION, SOLUTION, SOLUTION, MERGE])
    model = ParallelizationModel(llm, planner_mode="llm", executor=fast_executor())
    result = await model.run(TaskRequest(task="Assess the enclosure design.", domain="software_engineering"))
    assert result.status == Status.SUCCESS
    plan = result.metadata["plan"]
    assert plan["planner"] == "llm" and plan["pattern"] == "sectioning"
    assert [p["id"] for p in plan["partitions"]] == ["thermal", "structural", "summary"]
    assert result.output_text == "Both margins are adequate."
    assert result.usage.llm_calls == 5
    # the planner call carried the structured schema and the domain profile in the stable prefix
    first = llm.calls[0]
    assert first.output_schema is not None and "Domain: software_engineering" in "\n".join(first.system_stable)
    # dependency outputs were bound into the dependent sub-task's user turn
    summary_call = next(c for c in llm.calls if 'sub_task id="summary"' in c.last_user_text())
    assert "<dependencies>" in summary_call.last_user_text() and "margin is adequate" in summary_call.last_user_text()
    assert result.output.details["graph"]["summary"] == ["thermal", "structural"]


async def test_invalid_llm_plan_falls_back_to_heuristics_with_warning():
    llm = ScriptedLLM([{"pattern": "teleport", "rationale": "nonsense"}, {"answer": "step-by-step derivation", "confidence": "medium"}])
    model = ParallelizationModel(llm, planner_mode="llm", executor=fast_executor())
    result = await model.run("Derive the deflection formula for a simply supported beam step by step.")
    assert result.ok and result.metadata["plan"]["planner"] == "heuristic" and result.metadata["plan"]["pattern"] == "sequential"
    assert any("LLM planner output unusable" in w for w in result.warnings)
    assert result.usage.llm_calls == 2


async def test_offline_uses_heuristics_and_attaches_plan():
    model = ParallelizationModel(MockLLM(), executor=fast_executor())
    result = await model.run("What are the main failure modes of a bolted flange joint?")
    assert result.metadata["plan"]["planner"] == "heuristic" and result.metadata["plan"]["pattern"] == "skeleton_of_thought"
    assert result.status == Status.SUCCESS and result.output.aggregator == "concat"
    assert result.usage.llm_calls == 1 + len(result.output.details["points"])
    assert result.evidence and all(e.retriever == "provided" for e in result.evidence)


async def test_needs_review_when_votes_disagree():
    counter = itertools.count()

    def responder(req: LLMRequest) -> Any:
        return {"reasoning": "r", "final_answer": f"{next(counter)} kN"}

    model = ParallelizationModel(MockLLM(responder=responder), executor=fast_executor())
    result = await model.run("Calculate the bending stress for a 10 kN load on a 2 m beam.")
    assert result.output.pattern == "voting"
    assert result.status == Status.NEEDS_REVIEW
    assert result.confidence < 0.5 and "below acceptance threshold" in result.confidence_basis
    assert result.usage.llm_calls == 10  # not decisive -> sampled up to max_samples


async def test_needs_review_when_request_min_confidence_is_higher_than_domain():
    model = ParallelizationModel(MockLLM(), executor=fast_executor())
    ok = await model.run(TaskRequest(task="Derive the formula step by step."))
    strict = await model.run(TaskRequest(task="Derive the formula step by step.", constraints=Constraints(min_confidence=0.95)))
    assert ok.status == Status.SUCCESS and ok.confidence == pytest.approx(0.9)
    assert strict.status == Status.NEEDS_REVIEW and strict.metadata["threshold"] == 0.95


def failing_partition_llm(failing: str) -> MockLLM:
    def responder(req: LLMRequest) -> Any:
        text = req.last_user_text()
        if f'sub_task id="{failing}"' in text:
            raise LLMError("provider error on this branch")
        if "<solutions>" in text:
            return MERGE
        return SOLUTION

    return MockLLM(responder=responder)


async def test_partial_failure_is_partial_in_general_domain():
    cfg = ParallelizationConfig(retry_max_attempts=1)
    llm = failing_partition_llm("p2")
    model = ParallelizationModel(llm, config=cfg, executor=fast_executor(cfg))
    result = await model.run("Compare steel, aluminium and titanium for a bracket.")
    assert result.status == Status.PARTIAL
    assert result.output.failed_branches == ["p2"] and result.output.quorum_met
    # the merge call was told which branch is missing, and the basis reports the shortfall
    merge_call = next(c for c in llm.calls if "<solutions>" in c.last_user_text())
    assert "p2" in merge_call.last_user_text() and 'id="p2"' not in merge_call.last_user_text().split("<solutions>")[1].split("</solutions>")[0]
    assert "2/3 partitions solved" in result.confidence_basis
    assert result.confidence == pytest.approx(0.6 * (2 / 3) + 0.4 * 0.9, abs=1e-3)
    assert result.metadata["executor"]["failed"] == 1


async def test_partial_failure_blocks_answer_in_safety_critical_domain():
    cfg = ParallelizationConfig(retry_max_attempts=1)
    model = ParallelizationModel(failing_partition_llm("p2"), config=cfg, executor=fast_executor(cfg))
    result = await model.run(TaskRequest(task="Compare steel, aluminium and titanium for a bracket.", domain="civil_structural_engineering"))
    assert result.metadata["plan"]["failure_policy"] == "quorum" and result.metadata["plan"]["quorum"] == 1.0
    assert result.status == Status.NEEDS_REVIEW and result.confidence == 0.0
    assert not result.output.quorum_met and result.errors
    # two branches completed (the failing branch raised before the mock recorded usage) and no merge call was spent
    assert result.usage.llm_calls == 2
    assert not any("<solutions>" in c.last_user_text() for c in model.llm.calls)


async def test_domain_adaptation_changes_prompts_and_sampling():
    general_llm, civil_llm = MockLLM(), MockLLM()
    task = "Calculate the bending stress for a 10 kN load on a 2 m beam."
    g = await ParallelizationModel(general_llm, executor=fast_executor()).run(task)
    c = await ParallelizationModel(civil_llm, executor=fast_executor()).run(TaskRequest(task=task, domain="civil_structural_engineering"))
    assert g.metadata["plan"]["samples"] == 5 and c.metadata["plan"]["samples"] == 10
    assert g.usage.llm_calls == 5 and c.usage.llm_calls == 10  # unanimous mock votes stop after the first batch
    assert ENGINEERING_RIGOR in civil_llm.calls[0].system_stable and ENGINEERING_RIGOR not in general_llm.calls[0].system_stable
    assert "Domain: civil_structural_engineering" in "\n".join(civil_llm.calls[0].system_stable)
    assert c.metadata["plan"]["failure_policy"] == "quorum"


async def test_budget_enforcement_degrades_fan_out():
    llm = MockLLM()
    model = ParallelizationModel(llm, executor=fast_executor())
    result = await model.run(TaskRequest(task="Compare A, B, C, D, E and F for the housing.", constraints=Constraints(max_llm_calls=4)))
    assert result.ok and result.usage.llm_calls <= 4 and len(result.metadata["plan"]["partitions"]) == 3
    tight = await model.run(TaskRequest(task="What are the main risks of a redesign?", constraints=Constraints(max_llm_calls=1)))
    assert tight.ok and tight.usage.llm_calls == 1 and tight.metadata["plan"]["pattern"] == "sequential"


async def test_runtime_budget_guard_when_plan_underestimates():
    # the skeleton returns 10 points but only 5 calls remain after the skeleton call: expansions are capped, not failed
    def responder(req: LLMRequest) -> Any:
        if "skeleton of a long-form answer" in "\n".join(req.system_stable):
            return {"points": [f"point {i}" for i in range(10)]}
        return {"text": "expanded"}

    cfg = ParallelizationConfig(skeleton_min_points=3)
    model = ParallelizationModel(MockLLM(responder=responder), config=cfg, executor=fast_executor(cfg))
    result = await model.run(TaskRequest(task="What are the main risks of a redesign?", constraints=Constraints(max_llm_calls=6)))
    assert result.ok and result.usage.llm_calls == 6
    assert any("budget" in w for w in result.warnings)


async def test_speculative_pattern_uses_fast_model_and_verifier():
    llm = MockLLM()
    model = ParallelizationModel(llm, executor=fast_executor())
    result = await model.run(TaskRequest(task="Tell me about bolts.", constraints=Constraints(max_latency_s=10)))
    assert result.output.pattern == "speculative" and result.output.details["source"] == "fast"
    models = {c.model for c in llm.calls}
    assert "claude-haiku-5-5" in models  # fast path used the cheap model id without a date suffix
    assert result.usage.llm_calls in (2, 3)


async def test_map_reduce_pattern_end_to_end():
    text = "\n".join(f"2026-01-01T00:00:{i % 60:02d}Z worker ERROR timeout contacting inventory-service" for i in range(600))
    model = ParallelizationModel(MockLLM(), executor=fast_executor())
    result = await model.run(TaskRequest(task="Summarise the failure clusters in these logs.", inputs={"text": text}))
    assert result.output.pattern == "map_reduce" and result.status == Status.SUCCESS
    assert result.output.details["chunks"] >= 2 and result.output.details["reduce_levels"] >= 1
    assert result.usage.llm_calls == result.output.details["chunks"] + sum(1 for b in result.output.branches if b.stage == "reduce")


async def test_ensemble_pattern_end_to_end():
    model = ParallelizationModel(MockLLM(), executor=fast_executor())
    result = await model.run(TaskRequest(task="Recommend a root cause and justification for the recurring bearing failures in the pump train.", domain="civil_structural_engineering"))
    assert result.output.pattern == "ensemble" and result.usage.llm_calls == 7 and result.status == Status.SUCCESS
    stages = [b.stage for b in result.output.branches]
    assert stages.count("propose") == 6 and stages.count("aggregate") == 1


async def test_branch_solve_merge_end_to_end():
    model = ParallelizationModel(MockLLM(), executor=fast_executor())
    result = await model.run("Review this design note against the requirements.", inputs={"text": "The bracket shall carry 5 kN."})
    assert result.output.pattern == "branch_solve_merge" and result.status == Status.SUCCESS
    assert result.usage.llm_calls == 1 + len(result.output.details["branches"]) + 1


async def test_model_satisfies_executor_protocol_and_direct_api_accounts_usage():
    model = ParallelizationModel(MockLLM(), executor=fast_executor())
    assert isinstance(model, Executor) and isinstance(model.executor, Executor)

    async def double(x: int) -> int:
        return 2 * x

    assert await model.map(double, [1, 2, 3]) == [2, 4, 6]
    pr = await model.vote("Is 7 prime?", samples=3)
    assert pr.pattern == "voting" and pr.llm_calls == 3 and pr.usage.llm_calls == 3 and pr.confidence > 0.9
    usc = await model.vote("Describe the trade-offs", samples=3, free_form=True)
    assert usc.aggregator == "usc" and usc.llm_calls == 4
    sk = await model.skeleton_of_thought("List the risks of the redesign", domain="mechanical_engineering")
    assert sk.pattern == "skeleton_of_thought" and sk.answer
    sec = await model.sectioning("Assess the design", ["thermal", "structural"], aggregator="concat")
    assert sec.pattern == "sectioning" and sec.llm_calls == 2 and "[p1]" in sec.answer
    mr = await model.map_reduce(["a", "b", "c"], "Summarise: {item}", "Combine: {records}", fanin=2)
    assert mr.llm_calls == 6 and mr.details["reduce_levels"] == 2
    bsm = await model.branch_solve_merge("Evaluate the proposal")
    assert bsm.pattern == "branch_solve_merge" and bsm.llm_calls >= 5
    ens = await model.ensemble("Recommend a mitigation", proposers=2, layers=1)
    assert ens.llm_calls == 3


async def test_heterogeneous_proposers_are_opt_in():
    llm = MockLLM()
    cfg = ParallelizationConfig(homogeneous_ensemble=False, ensemble_proposer_models=["claude-sonnet-5-5", "claude-haiku-5-5"])
    model = ParallelizationModel(llm, config=cfg, executor=fast_executor(cfg))
    pr = await model.ensemble("Recommend a mitigation", proposers=2, layers=1)
    assert pr.llm_calls == 3
    assert [c.model for c in llm.calls[:2]] == ["claude-sonnet-5-5", "claude-haiku-5-5"] and llm.calls[2].model is None
    homogeneous = ParallelizationModel(MockLLM(), executor=fast_executor())
    await homogeneous.ensemble("Recommend a mitigation", proposers=2, layers=1)
    assert all(c.model is None for c in homogeneous.llm.calls)


def test_run_sync_and_describe_and_heuristic_plan():
    model = ParallelizationModel(MockLLM(), max_concurrency=4)
    assert model.config.max_concurrency == 4
    plan = model.heuristic_plan("Compare A, B and C.")
    assert plan.pattern == "sectioning" and plan.concurrency == 4
    d = model.describe()
    assert d["name"] == "parallelization" and d["config"]["max_concurrency"] == 4 and "executor" in d
    result = model.run_sync("Tell me about bolts.")
    assert result.ok and result.trace_id and model.tracer.tree()
