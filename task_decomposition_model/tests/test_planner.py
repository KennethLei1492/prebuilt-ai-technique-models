"""Heuristic planner decision rules, feature extraction and plan normalisation."""
from __future__ import annotations

import pytest

from pipeline_core import Constraints, Evidence, get_profile

from task_decomposition import DecompositionPlan, HeuristicPlanner, TaskDecompositionConfig, extract_features, normalize_plan

from td_fixtures import BRIDGE_TASK, CHAIN_TASK, DEBUG_TASK, EXPLORE_TASK, PUMP_TASK, TRIVIAL_TASK


@pytest.mark.parametrize(
    "task,expected",
    [
        (TRIVIAL_TASK, "direct"),
        (PUMP_TASK, "dag"),
        (BRIDGE_TASK, "dag"),
        (DEBUG_TASK, "recursive"),
        (EXPLORE_TASK, "tree_of_thoughts"),
        (CHAIN_TASK, "least_to_most"),
        ("Summarise the main risks in the attached design note and recommend mitigations.", "plan_and_solve"),
    ],
)
def test_strategy_selection(task: str, expected: str):
    planner = HeuristicPlanner()
    inputs = {"documents": ["x"]} if "attached" in task else None
    plan = planner.plan(task, profile=get_profile("general"), inputs=inputs)
    assert plan.strategy == expected, plan.rationale


def test_feature_extraction_counts_markers():
    f = extract_features(PUMP_TASK, profile=get_profile("mechanical_engineering"))
    assert f.independence_markers >= 1 and f.sequence_markers >= 1 and f.enumeration_items >= 3
    assert f.estimated_steps >= 3 and f.domain_hint_count == 2 and f.risk_level == "high"
    f2 = extract_features("Check this", inputs={"documents": ["d"]}, context=[Evidence(id="e", content="c")])
    assert f2.has_documents and f2.input_keys == ["documents"]
    f3 = extract_features("1. first item\n2. second item\n3. third item")
    assert f3.enumeration_items == 3


def test_deterministic_constraint_downgrades_sampling_strategies():
    planner = HeuristicPlanner()
    det = Constraints(deterministic=True)
    assert planner.plan(EXPLORE_TASK, profile=get_profile("general"), constraints=det).strategy == "dag"
    assert planner.plan(DEBUG_TASK, profile=get_profile("general"), constraints=det).strategy == "least_to_most"
    assert HeuristicPlanner(TaskDecompositionConfig(enable_search=False)).plan(EXPLORE_TASK, profile=get_profile("general")).strategy != "tree_of_thoughts"


def test_verification_follows_profile_risk():
    planner = HeuristicPlanner()
    assert planner.plan(PUMP_TASK, profile=get_profile("general")).verification == "criteria"
    assert planner.plan(TRIVIAL_TASK, profile=get_profile("general")).verification == "none"
    assert planner.plan(PUMP_TASK, profile=get_profile("software_engineering")).verification == "criteria"
    assert planner.plan(PUMP_TASK, profile=get_profile("mechanical_engineering")).verification == "external"
    assert planner.plan(PUMP_TASK, profile=get_profile("civil_structural_engineering")).verification == "strict"


def test_bounds_and_parallelism_follow_config_and_constraints():
    cfg = TaskDecompositionConfig(max_parallelism=3, max_subtasks=5)
    plan = HeuristicPlanner(cfg).plan(BRIDGE_TASK, profile=get_profile("general"), constraints=Constraints(max_parallelism=2))
    assert plan.strategy == "dag" and plan.parallelism == 2 and plan.max_breadth <= 5
    chain = HeuristicPlanner().plan(CHAIN_TASK, profile=get_profile("general"))
    assert chain.parallelism == 1 and chain.max_depth >= chain.max_breadth
    assert HeuristicPlanner().plan(PUMP_TASK, profile=get_profile("general"), strategy_override="least_to_most").strategy == "least_to_most"


def test_normalize_plan_clamps_and_raises_verification_floor():
    cfg = TaskDecompositionConfig(max_subtasks=4, max_parallelism=4, max_depth=3, deep_domain_depth=4)
    wild = DecompositionPlan(strategy="dag", max_depth=50, max_breadth=40, parallelism=64, verification="none", complexity_estimate="high", estimated_steps=12, replan_on_failure=True, rationale="llm said so")
    p = normalize_plan(wild, cfg, Constraints(max_parallelism=3), get_profile("civil_structural_engineering"))
    assert p.max_breadth == 4 and p.max_depth == 4 and p.parallelism == 3 and p.verification == "strict"
    tot = DecompositionPlan(strategy="tree_of_thoughts", max_depth=2, max_breadth=5, parallelism=5, verification="criteria", complexity_estimate="medium", estimated_steps=5, replan_on_failure=True, rationale="r")
    assert normalize_plan(tot, cfg, Constraints(deterministic=True), get_profile("general")).strategy == "dag"
    assert normalize_plan(tot, TaskDecompositionConfig(enable_search=False), Constraints(), get_profile("general")).strategy == "dag"
    direct = DecompositionPlan(strategy="direct", max_depth=3, max_breadth=3, parallelism=3, verification="strict", complexity_estimate="trivial", estimated_steps=1, replan_on_failure=True, rationale="r")
    d = normalize_plan(direct, cfg, Constraints(), get_profile("general"))
    assert (d.max_depth, d.max_breadth, d.parallelism, d.replan_on_failure, d.verification) == (1, 1, 1, False, "strict")  # never lowered
    assert normalize_plan(wild, TaskDecompositionConfig(max_replans=0), Constraints(), get_profile("general")).replan_on_failure is False


def test_plan_schema_clamps_bad_numbers():
    p = DecompositionPlan(strategy="dag", max_depth="abc", max_breadth=0, parallelism=999, verification="criteria", complexity_estimate="low", estimated_steps=-3, replan_on_failure=False, rationale="r")
    assert (p.max_depth, p.max_breadth, p.parallelism, p.estimated_steps) == (1, 1, 64, 1)
    assert p.is_parallel() and "strategy=dag" in p.describe()


def test_markers_are_counted_as_whole_words_once():
    """Regression: substring counting matched 'each' in 'Teach', 'per ' in 'paper ', 'once ' in 'announce ' and counted 'after that' twice."""
    none = extract_features("Teach the new hire to reach the beach, then review the onboarding paper and announce the result.")
    assert none.independence_markers == 0
    assert none.sequence_markers == 1  # only "then"
    once = extract_features("Measure the flow, after that compute the head.")
    assert once.sequence_markers == 1
    indep = extract_features("Report the duty point per pump and the efficiency for each unit.")
    assert indep.independence_markers == 2
    assert extract_features("Compute the head using the results of the flow test.").sequence_markers == 1


def test_sequence_marked_parts_are_not_an_enumeration():
    chain = extract_features(CHAIN_TASK)
    assert chain.enumeration_items == 0 and chain.sequence_markers >= 3 and chain.estimated_steps >= 4
    items = extract_features("Check the pump curve, verify the motor rating, check the relief valve, and list open risks.")
    assert items.enumeration_items == 4 and items.sequence_markers == 0
    mixed = extract_features("Measure the flow, measure the head, measure the power, then compute the efficiency.")
    assert mixed.enumeration_items == 3 and mixed.sequence_markers == 1
    assert HeuristicPlanner().plan(mixed and "Measure the flow, measure the head, measure the power, then compute the efficiency.", profile=get_profile("general")).strategy == "dag"
