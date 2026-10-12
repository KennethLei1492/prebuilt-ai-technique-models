"""Heuristic routing and planning rules (pure, no LLM)."""
from __future__ import annotations

import pytest

from pipeline_core import Evidence, TaskRequest, get_profile

from pipeline_rag import HeuristicPlanner, PipelineRAGConfig, RAGPlan, extract_task_features, route_for_features
from pipeline_rag.plan import answer_type_for_features
from pipeline_rag.router import QueryRouter, next_route

DOCS = [{"id": "d1", "text": "The casing corrosion allowance is 3 mm. " * 5}, {"id": "d2", "text": "Motor rated 75 kW at 1480 rpm. " * 5}]


def test_no_corpus_routes_to_no_retrieval() -> None:
    f = extract_task_features("What corrosion allowance applies to the casing?", {})
    assert not f.has_corpus
    assert route_for_features(f).route == "no_retrieval"


def test_transformation_task_routes_to_no_retrieval_even_with_corpus() -> None:
    f = extract_task_features("Rewrite the following paragraph in plain English", {"documents": DOCS})
    assert route_for_features(f).route == "no_retrieval"


def test_focused_question_is_single_step() -> None:
    f = extract_task_features("What corrosion allowance does the specification require for the pump casing?", {"documents": DOCS})
    d = route_for_features(f)
    assert d.route == "single_step"
    assert d.complexity == "simple"


def test_comparison_is_multi_step() -> None:
    f = extract_task_features("Compare the casing corrosion allowance with the impeller material requirement and explain the difference between them.", {"documents": DOCS})
    d = route_for_features(f)
    assert d.route == "multi_step"
    assert "compar" in d.reason


def test_global_question_is_agentic() -> None:
    f = extract_task_features("What recurring themes appear across all incident reports?", {"documents": DOCS})
    assert route_for_features(f).route == "agentic"


def test_upstream_context_counts_as_corpus() -> None:
    ev = [Evidence(id="e1", content="Casing corrosion allowance 3 mm.")]
    f = extract_task_features("What is the casing corrosion allowance?", {}, context=ev)
    assert f.has_corpus and f.has_context
    assert route_for_features(f).route == "single_step"


def test_identifier_detection_blocks_step_back_and_hyde() -> None:
    cfg = PipelineRAGConfig(enable_hyde=True)
    planner = HeuristicPlanner(cfg)
    f = extract_task_features("Why does relay K12-A fail at 85 degC per REQ-204?", {"documents": DOCS})
    assert f.has_identifiers and f.why_how
    plan = planner.plan_from_features(f, get_profile("general"))
    assert "step_back" not in plan.query_transforms
    assert "hyde" not in plan.query_transforms
    assert "identifiers present" in plan.rationale


def test_why_question_adds_step_back_and_hyde_when_enabled() -> None:
    cfg = PipelineRAGConfig(enable_hyde=True)
    planner = HeuristicPlanner(cfg)
    f = extract_task_features("Why does the casing need a corrosion allowance in seawater?", {"documents": DOCS})
    plan = planner.plan_from_features(f, get_profile("general"))
    assert plan.query_transforms[:2] == ["rewrite", "step_back"]
    assert "hyde" in plan.query_transforms
    assert plan.expected_answer_type == "explanatory"


def test_hyde_disabled_by_default() -> None:
    planner = HeuristicPlanner(PipelineRAGConfig())
    f = extract_task_features("Why does the casing need a corrosion allowance in seawater?", {"documents": DOCS})
    assert "hyde" not in planner.plan_from_features(f, get_profile("general")).query_transforms


def test_safety_critical_profile_forces_grading_verification_and_inline_citations() -> None:
    planner = HeuristicPlanner(PipelineRAGConfig(citation_style="footnote"))
    f_gen = extract_task_features("What is the design pressure?", {"documents": DOCS}, profile=get_profile("general"))
    f_sc = extract_task_features("What is the design pressure?", {"documents": DOCS}, profile=get_profile("civil_structural_engineering"))
    general = planner.plan_from_features(f_gen, get_profile("general"))
    safety = planner.plan_from_features(f_sc, get_profile("civil_structural_engineering"))
    assert general.citation_style == "footnote"
    assert safety.citation_style == "inline"
    assert safety.verify and safety.grade_context and safety.abstain_when_insufficient
    assert "safety_critical" in safety.rationale
    assert "safety_critical" not in general.rationale


def test_multi_step_plan_reserves_hops_and_multi_query() -> None:
    cfg = PipelineRAGConfig(max_hops=3)
    planner = HeuristicPlanner(cfg)
    f = extract_task_features("Trace the root cause of the vibration trip and relate it to the maintenance procedure.", {"documents": DOCS})
    plan = planner.plan_from_features(f, get_profile("general"))
    assert plan.route == "multi_step"
    assert plan.max_hops == 3
    assert "multi_query" in plan.query_transforms
    assert plan.k >= cfg.k_per_hop * 2


def test_agentic_plan_uses_hierarchy_under_auto() -> None:
    planner = HeuristicPlanner(PipelineRAGConfig(build_hierarchy="auto"))
    f = extract_task_features("Summarize the overall patterns across all documents.", {"documents": DOCS})
    plan = planner.plan_from_features(f, get_profile("general"))
    assert plan.route == "agentic" and plan.use_hierarchy
    assert plan.expected_answer_type == "summary"
    never = HeuristicPlanner(PipelineRAGConfig(build_hierarchy="never")).plan_from_features(f, get_profile("general"))
    assert not never.use_hierarchy


def test_plan_normalization_clamps_llm_values() -> None:
    cfg = PipelineRAGConfig(max_hops=4, max_query_variants=2)
    raw = RAGPlan(route="multi_step", query_transforms=["rewrite", "hyde", "step_back", "multi_query"], k=500, max_hops=99, context_token_budget=5)
    plan = raw.normalized(cfg)
    assert plan.k == 100
    assert plan.max_hops == 4
    assert plan.context_token_budget == 100
    assert "hyde" not in plan.query_transforms and len(plan.query_transforms) <= 2
    nr = RAGPlan(route="no_retrieval", query_transforms=["rewrite"], grade_context=True, max_hops=3).normalized(cfg)
    assert nr.query_transforms == [] and nr.max_hops == 1 and not nr.grade_context


def test_planner_from_request_and_describe() -> None:
    planner = HeuristicPlanner(PipelineRAGConfig())
    req = TaskRequest(task="List the mitigations for casing corrosion, strainer fouling, and motor overheating.", inputs={"documents": DOCS})
    plan = planner.plan(req, get_profile("general"))
    assert plan.expected_answer_type == "list"
    assert "route=" in plan.describe() and "rationale:" in plan.describe()


def test_answer_type_rules() -> None:
    assert answer_type_for_features(extract_task_features("How many pumps are installed?", {"documents": DOCS})) == "numerical"
    assert answer_type_for_features(extract_task_features("How do I align the coupling?", {"documents": DOCS})) == "procedural"
    assert answer_type_for_features(extract_task_features("Compare pump A versus pump B", {"documents": DOCS})) == "comparison"


def test_query_router_classify_and_next_route() -> None:
    router = QueryRouter(PipelineRAGConfig())
    assert router.classify("What is the design pressure?", {"documents": DOCS}).route == "single_step"
    assert next_route("no_retrieval") == "single_step"
    assert next_route("multi_step") == "agentic"
    assert next_route("agentic") is None


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PipelineRAGConfig(grade_lower_threshold=0.9, grade_upper_threshold=0.5)
    with pytest.raises(ValueError):
        PipelineRAGConfig(chunk_overlap_tokens=600, chunk_size_tokens=512)
    w = PipelineRAGConfig().confidence_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["support"] > w["self_reported"]
