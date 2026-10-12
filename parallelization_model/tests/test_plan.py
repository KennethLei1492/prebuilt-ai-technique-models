"""Heuristic planner: routing table, domain policy, budget degradation, LLM-output validation."""
from __future__ import annotations

import pytest

from pipeline_core import Constraints, PlanValidationError, TaskRequest, get_profile

from parallelization.config import ParallelizationConfig
from parallelization.plan import HeuristicPlanner, ParallelPlan, PlannerOutput, chunk_count, detect_enumerations, extract_features

GENERAL = get_profile("general")
MECH = get_profile("mechanical_engineering")
CIVIL = get_profile("civil_structural_engineering")


def plan_for(task: str, *, domain: str = "general", **kw) -> ParallelPlan:
    req = TaskRequest(task=task, domain=domain, **kw)
    return HeuristicPlanner().plan(req, get_profile(domain), remaining_calls=req.constraints.max_llm_calls)


@pytest.mark.parametrize(
    "task, pattern",
    [
        ("Compare steel, aluminium and titanium for a bracket.", "sectioning"),
        ("What is the load capacity? Which bolt grade is required? How many bolts are needed?", "sectioning"),
        ("What are the main risks of switching to a microservice architecture?", "skeleton_of_thought"),
        ("Calculate the bending stress for a 10 kN load on a 2 m beam.", "voting"),
        ("Review this design against the requirements.", "branch_solve_merge"),
        ("Derive the Euler buckling load step by step.", "sequential"),
        ("Write a Python function that parses the log file.", "sequential"),
        ("Tell me about bolts.", "sequential"),
    ],
)
def test_routing_table(task: str, pattern: str):
    assert plan_for(task).pattern == pattern


def test_items_input_triggers_sectioning_with_one_partition_per_item():
    p = plan_for("Classify each ticket as bug, feature or question.", inputs={"items": ["Crash on save", "Add dark mode", "How do I export?"]})
    assert p.pattern == "sectioning" and len(p.partitions) == 3
    assert "Crash on save" in p.partitions[0].task
    assert p.expected_calls == 4 and p.expected_speedup > 1.5


def test_long_input_triggers_map_reduce_with_chunk_partitions():
    text = "line of log output\n" * 2000  # ~9.5k tokens
    p = plan_for("Summarise the failure clusters in these logs.", inputs={"text": text})
    assert p.pattern == "map_reduce"
    assert len(p.partitions) == chunk_count(len(text), 4000 * 4, 1600) >= 2
    assert p.aggregator == "llm"


def test_chunk_count():
    assert chunk_count(0, 100, 10) == 0
    assert chunk_count(10, 4, 2) == 5  # matches pipeline_core.chunk_text
    assert chunk_count(100, 100, 0) == 1


def test_high_risk_domain_changes_policy_and_sample_counts():
    general = plan_for("Calculate the bending stress for a 10 kN load.")
    civil = plan_for("Calculate the bending stress for a 10 kN load.", domain="civil_structural_engineering")
    assert general.failure_policy == "best_effort" and civil.failure_policy == "quorum" and civil.quorum == 1.0
    assert general.samples == 5 and civil.samples == 10  # no early stop below n0 in safety-critical domains
    mech = plan_for("Calculate the bending stress for a 10 kN load.", domain="mechanical_engineering")
    assert mech.failure_policy == "quorum" and mech.quorum == pytest.approx(0.8)


def test_high_value_free_form_in_high_risk_domain_uses_ensemble():
    p = plan_for("Recommend a root cause and justification for the recurring bearing failures in the pump train.", domain="civil_structural_engineering")
    assert p.pattern == "ensemble" and p.layers == 2 and p.samples == 3 and p.expected_calls == 7
    assert p.expected_speedup == pytest.approx(7 / 3, abs=1e-3)
    # the same task in a low-risk domain does not pay for an ensemble
    assert plan_for("Recommend a root cause and justification for the recurring bearing failures in the pump train.").pattern != "ensemble"


def test_amdahl_serial_share_note_is_added_when_above_target():
    ensemble = plan_for("Recommend a root cause and justification for the recurring bearing failures in the pump train.", domain="civil_structural_engineering")
    assert "serial share 43%" in ensemble.rationale and "Amdahl bound 2.3x" in ensemble.rationale
    wide = plan_for("Classify each ticket.", inputs={"items": [f"ticket {i}" for i in range(10)]})
    assert "serial share" not in wide.rationale  # 1 merge call of 11 is under the 30% target


def test_latency_bound_simple_task_uses_speculative():
    p = plan_for("Tell me about bolts.", constraints=Constraints(max_latency_s=10))
    assert p.pattern == "speculative" and p.expected_calls == 3


def test_budget_degrades_rather_than_fails():
    p = plan_for("Compare A, B, C, D, E and F for the housing.", constraints=Constraints(max_llm_calls=4))
    assert p.pattern == "sectioning" and len(p.partitions) == 3 and p.expected_calls <= 4
    assert "Budget" in p.rationale
    v = plan_for("Calculate the bending stress for a 10 kN load.", constraints=Constraints(max_llm_calls=3))
    assert v.pattern == "voting" and v.max_samples == 3 and v.expected_calls <= 3
    s = plan_for("What are the main risks of switching to a microservice architecture?", constraints=Constraints(max_llm_calls=2))
    assert s.pattern == "sequential"
    e = plan_for("Recommend a root cause and justification for the recurring bearing failures.", domain="civil_structural_engineering", constraints=Constraints(max_llm_calls=5))
    assert e.pattern in ("ensemble", "sequential") and e.expected_calls <= 5


def test_constraints_cap_concurrency_and_deterministic_disables_cache_gate():
    p = plan_for("Compare A, B and C.", constraints=Constraints(max_parallelism=2, deterministic=True))
    assert p.concurrency == 2 and p.cache_aware is False


def test_token_multiplier_cap_limits_voting():
    planner = HeuristicPlanner(ParallelizationConfig(token_multiplier_cap=4, voting_samples=5, voting_max_samples=10))
    req = TaskRequest(task="Calculate the bending stress for a 10 kN load.")
    p = planner.plan(req, GENERAL)
    assert p.max_samples == 4 and p.samples == 4


def test_detect_enumerations_and_features():
    assert detect_enumerations("Compare steel, aluminium and titanium.") == ["steel", "aluminium", "titanium"]
    assert len(detect_enumerations("For each of the following: pump, valve, filter.")) == 3
    f = extract_features(TaskRequest(task="Calculate the stress. Then compute the deflection?"), MECH)
    assert f.numeric_markers and f.risk_level == "high" and f.word_count == 7


def test_validate_llm_output_rejects_bad_plans():
    planner = HeuristicPlanner()
    good = planner.validate_llm_output({"pattern": "sectioning", "partitions": [{"id": "a", "task": "x"}, {"id": "b", "task": "y", "depends_on": ["a"]}], "rationale": "r"})
    assert isinstance(good, PlannerOutput)
    with pytest.raises(PlanValidationError):
        planner.validate_llm_output({"pattern": "teleport", "rationale": "r"})
    with pytest.raises(PlanValidationError):
        planner.validate_llm_output({"pattern": "sectioning", "partitions": [{"id": "a", "task": "x"}, {"id": "a", "task": "y"}], "rationale": "dup"})
    with pytest.raises(PlanValidationError):
        planner.validate_llm_output({"pattern": "sectioning", "partitions": [{"id": "a", "task": "x", "depends_on": ["b"]}, {"id": "b", "task": "y", "depends_on": ["a"]}], "rationale": "cycle"})


def test_finalize_sectioning_with_single_partition_falls_back_to_sequential():
    planner = HeuristicPlanner()
    req = TaskRequest(task="x")
    out = PlannerOutput(pattern="sectioning", partitions=[{"id": "a", "task": "only"}], rationale="r")
    plan = planner.finalize(out, req, GENERAL, planner="llm")
    assert plan.pattern == "sequential" and plan.planner == "llm" and plan.expected_calls == 1


def test_finalize_voting_free_form_uses_usc():
    planner = HeuristicPlanner()
    plan = planner.finalize(PlannerOutput(pattern="voting", samples=6, verifiability="free_form", rationale="r"), TaskRequest(task="x"), GENERAL, planner="llm")
    assert plan.aggregator == "usc" and plan.samples == 6
    plan2 = planner.finalize(PlannerOutput(pattern="voting", samples=6, verifiability="exact_match", rationale="r"), TaskRequest(task="x"), GENERAL, planner="llm")
    assert plan2.aggregator == "code"
    assert "pattern=voting" in plan2.summary()
