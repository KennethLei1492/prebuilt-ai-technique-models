"""Heuristic planner, feature extraction, reconciliation and the LLM-planned path."""
from __future__ import annotations

import pytest

from pipeline_core import Constraints, DomainProfile, MockLLM, ScriptedLLM, Status, TaskRequest, get_profile, register_profile

from caching import CachePlan, CachingConfig, CachingModel, HeuristicPlanner, analyze_task

MECH = get_profile("mechanical_engineering")
GENERAL = get_profile("general")
SAFETY = get_profile("civil_structural_engineering")


# ------------------------------------------------------------------ features


def test_analyze_task_extracts_signals():
    f = analyze_task("What does clause 4.2 of ISO 2768 require for a 120 mm dimension today?")
    assert f.lookup_like and f.time_sensitive and not f.randomness
    assert f.identifier_count >= 1 and f.number_count >= 2
    assert "ISO2768" in f.guards and "120mm" in f.guards
    assert "lookup-like but time-sensitive" in f.conflicts()


def test_analyze_task_personal_and_random():
    assert analyze_task("Summarize my account history").personal
    assert analyze_task("Brainstorm five slogans").randomness
    assert not analyze_task("Explain how a PID controller works").personal


# ------------------------------------------------------------------ heuristic rules


@pytest.mark.parametrize(
    "task,expected_class,cacheable,semantic",
    [
        ("Brainstorm five different names for a torque sensor", "random", False, False),
        ("What is the current exchange rate for EUR to USD today?", "time_sensitive", True, False),
        ("Summarize my account history for me", "personal", True, False),
        ("What does ISO 2768-m specify for a 120 mm dimension?", "lookup", True, True),
        ("Compare the fatigue life of grade 8.8 and 10.9 bolts", "analysis", True, True),
    ],
)
def test_heuristic_task_classes(task, expected_class, cacheable, semantic):
    plan = HeuristicPlanner().plan_text(task, MECH)
    assert plan.task_class == expected_class
    assert plan.cacheable is cacheable
    assert ("semantic" in plan.tiers) is semantic
    if cacheable:
        assert plan.tiers[0] == "exact"


def test_time_sensitive_uses_short_ttl_and_personal_uses_personal_ttl():
    cfg = CachingConfig()
    p = HeuristicPlanner(cfg).plan_text("What is the latest firmware status right now?", GENERAL)
    assert p.ttl_s == cfg.time_sensitive_ttl_s and "time_sensitive" in p.risk_flags
    q = HeuristicPlanner(cfg).plan_text("Show my account balance", GENERAL)
    assert q.ttl_s == cfg.personal_ttl_s and "personal_data" in q.risk_flags


def test_identifier_density_enforces_strict_threshold():
    cfg = CachingConfig()
    loose = HeuristicPlanner(cfg).plan_text("Explain how a centrifugal pump works", GENERAL)
    strict = HeuristicPlanner(cfg).plan_text("Which torque applies to fastener ABC-1234 at 12 Nm?", GENERAL)
    assert loose.semantic_threshold == cfg.semantic_threshold and loose.error_budget == cfg.semantic_error_budget
    assert strict.semantic_threshold == cfg.strict_semantic_threshold and strict.error_budget == cfg.strict_error_budget
    assert strict.verify_hits and "identifier_or_numeric_sensitive" in strict.risk_flags


def test_domain_adaptation_safety_critical_disables_semantic_tier():
    task = "Explain how a centrifugal pump works"
    general = HeuristicPlanner().plan_text(task, GENERAL)
    safety = HeuristicPlanner().plan_text(task, SAFETY)
    assert "semantic" in general.tiers
    assert "semantic" not in safety.tiers
    assert safety.semantic_threshold >= 0.98 and safety.error_budget <= 0.01
    assert any(f.startswith("risk_level=safety_critical") for f in safety.risk_flags)
    # the switch is configuration, not code: allow it and the tier returns with strict settings
    allowed = HeuristicPlanner(CachingConfig(semantic_in_safety_critical=True)).plan_text(task, SAFETY)
    assert "semantic" in allowed.tiers and allowed.verify_hits


def test_prefix_ttl_promoted_for_standards_heavy_profiles():
    assert HeuristicPlanner().plan_text("What does clause 7 require?", MECH).prefix_ttl == "1h"
    assert HeuristicPlanner().plan_text("What does clause 7 require?", GENERAL).prefix_ttl == "5m"
    assert HeuristicPlanner().plan_text("hello", GENERAL, has_stable_prefix=False).prefix_ttl == "none"


def test_namespace_contains_every_determinant():
    plan = HeuristicPlanner().plan_text("Define FoS", MECH, model="claude-opus-5-5", prompt_version="2.1.0", corpus_version="snap-7", tenant="acme")
    assert plan.namespace.startswith("claude-opus-5-5:2.1.0:")
    assert "snap-7" in plan.namespace and "acme" in plan.namespace and plan.namespace.endswith(":mechanical_engineering")


def test_custom_registered_profile_changes_behaviour_without_code_changes():
    prof = register_profile(
        DomainProfile(name="test_aerospace_cache", description="aerospace structures", risk_level="safety_critical", min_confidence_to_accept=0.9, standards=["FAR 25"], document_types=["spec"]),
        overwrite=True,
    )
    plan = HeuristicPlanner().plan_text("What does FAR 25.571 require?", prof)
    assert "semantic" not in plan.tiers and plan.prefix_ttl == "1h"


# ------------------------------------------------------------------ reconciliation


def test_reconcile_never_relaxes_the_heuristic_floor():
    planner = HeuristicPlanner()
    heuristic = planner.plan_text("Which torque applies to fastener ABC-1234 at 12 Nm?", MECH)
    llm_plan = heuristic.model_copy(update={"semantic_threshold": 0.90, "error_budget": 0.1, "verify_hits": False, "tiers": ["semantic", "exact"], "identifier_guards": False})
    merged, overrides = planner.reconcile(llm_plan, heuristic, MECH)
    assert merged.semantic_threshold == heuristic.semantic_threshold
    assert merged.error_budget == heuristic.error_budget
    assert merged.verify_hits and merged.identifier_guards
    assert merged.tiers[0] == "exact"
    assert len(overrides) >= 4


def test_reconcile_keeps_more_conservative_llm_choices():
    planner = HeuristicPlanner()
    heuristic = planner.plan_text("Explain how a centrifugal pump works", GENERAL)
    llm_plan = heuristic.model_copy(update={"tiers": ["exact", "prefix"], "ttl_s": 3600.0, "semantic_threshold": 0.99})
    merged, overrides = planner.reconcile(llm_plan, heuristic, GENERAL)
    assert "semantic" not in merged.tiers and merged.ttl_s == 3600.0 and merged.semantic_threshold == 0.99
    assert overrides == []


def test_reconcile_blocks_llm_from_caching_random_prompts():
    planner = HeuristicPlanner()
    heuristic = planner.plan_text("Brainstorm five slogans", GENERAL)
    llm_plan = heuristic.model_copy(update={"cacheable": True, "tiers": ["exact"]})
    merged, overrides = planner.reconcile(llm_plan, heuristic, GENERAL)
    assert merged.cacheable is False and merged.tiers == [] and overrides


# ------------------------------------------------------------------ LLM-planned path


def _valid_plan_json(**over) -> dict:
    base = {
        "cacheable": True,
        "tiers": ["exact", "semantic", "prefix", "negative"],
        "ttl_s": 43200,
        "semantic_threshold": 0.98,
        "error_budget": 0.01,
        "verify_hits": True,
        "identifier_guards": True,
        "namespace": "ignored-by-reconcile",
        "prefix_ttl": "1h",
        "prewarm": False,
        "negative_ttl_s": 300,
        "task_class": "lookup",
        "risk_flags": ["identifier_or_numeric_sensitive"],
        "reason": "Stable standards lookup with identifiers; strict semantic settings.",
        "confidence": 0.9,
    }
    base.update(over)
    return base


async def test_llm_planned_path_with_scripted_llm():
    llm = ScriptedLLM([_valid_plan_json()])
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="What does IEC 61511 require for SIL 2 proof-test intervals?", domain="mechanical_engineering"))
    assert result.status == Status.SUCCESS
    assert result.usage.llm_calls == 1
    assert result.output.plan_source == "llm"
    plan = CachePlan.model_validate(result.metadata["plan"])
    assert plan.ttl_s == 43200 and plan.prefix_ttl == "1h" and plan.negative_ttl_s == 300
    assert plan.namespace.endswith(":mechanical_engineering")  # namespace always from heuristics
    # the planning prompt keeps stable text in system_stable and the task in the user turn
    req = llm.calls[0]
    assert req.output_schema is not None and req.effort == "low"
    assert any("cache policy planner" in b for b in req.system_stable)
    assert "IEC 61511" in req.last_user_text() and not any("IEC 61511" in b for b in req.system_stable)


async def test_llm_plan_is_reconciled_and_lowers_confidence():
    llm = ScriptedLLM([_valid_plan_json(semantic_threshold=0.85, error_budget=0.2, verify_hits=False)])
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="Which torque applies to fastener ABC-1234 at 12 Nm?", domain="mechanical_engineering"))
    assert result.output.plan_source == "llm" and result.output.plan_overrides
    assert result.output.plan.semantic_threshold >= 0.98 and result.output.plan.verify_hits
    assert any("plan override" in w for w in result.warnings)
    assert result.confidence < 0.9


async def test_invalid_llm_plan_falls_back_to_heuristics():
    llm = ScriptedLLM([{"cacheable": "maybe", "tiers": "all"}])
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="Explain how a centrifugal pump works"))
    assert result.status == Status.SUCCESS
    assert result.output.plan_source == "heuristic"
    assert any("heuristic plan" in w for w in result.warnings)


async def test_deterministic_constraint_forces_heuristic_planner():
    llm = ScriptedLLM([_valid_plan_json()])
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="Explain how a centrifugal pump works", constraints=Constraints(deterministic=True)))
    assert result.usage.llm_calls == 0 and result.output.plan_source == "heuristic"


async def test_offline_mock_llm_never_calls_the_model_in_auto_mode():
    llm = MockLLM()
    model = CachingModel(llm)
    await model.run("Explain how a centrifugal pump works")
    assert llm.calls == []
