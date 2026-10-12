"""Query features, the heuristic planner, budget degradation and plan clamping."""
from __future__ import annotations

import pytest

from pipeline_core import Constraints, DomainProfile, get_profile

from hybrid_retrieval.config import HybridRetrievalConfig
from hybrid_retrieval.plan import (
    FilterClause,
    HeuristicPlanner,
    PlannerOutput,
    RetrievalPlan,
    analyze_query,
    apply_budget,
    estimate_llm_calls,
    n_windows,
)

CFG = HybridRetrievalConfig()


def test_analyze_query_features() -> None:
    f = analyze_query('What is the torque for fastener "P/N 7342-118" on A320-NEO?')
    assert f.n_tokens == 10 and f.has_question_word and not f.is_exploratory
    assert "7342-118" in f.identifiers and "A320-NEO" in f.identifiers
    assert f.identifier_density == pytest.approx(3 / 10)  # P/N, 7342-118, A320-NEO
    assert f.quoted_phrases == ["P/N 7342-118"]
    assert analyze_query("").n_tokens == 0
    assert analyze_query("has anything like this happened before").is_exploratory
    assert "identifier_density" in f.render()


def test_heuristic_identifier_query_is_lexical_and_cheap() -> None:
    plan = HeuristicPlanner(CFG).plan("ERR-1203", profile=get_profile("software_engineering"))
    assert plan.query_type == "identifier" and plan.fusion == "convex"
    assert plan.use_sparse and not plan.use_dense  # bare identifier -> dense off
    assert plan.sparse_weight == pytest.approx(0.8) and plan.dense_weight == pytest.approx(0.2)
    assert plan.rerank == "none" and plan.query_transforms == [] and plan.expected_llm_calls == 0
    assert plan.source == "heuristic" and "identifier-dense" in plan.rationale


def test_heuristic_natural_language_general_query() -> None:
    plan = HeuristicPlanner(CFG).plan("Has anything similar to the pump cavitation event happened before?")
    assert plan.query_type == "exploratory" and plan.fusion == "rrf"
    assert plan.use_sparse and plan.use_dense
    assert plan.sparse_weight == pytest.approx(0.2)  # 0.3 default - 0.1 identifier-free question
    assert "hyde" in plan.query_transforms and "multi_query" in plan.query_transforms
    assert plan.rerank == "listwise" and plan.rerank_depth == 20
    assert plan.expected_llm_calls == estimate_llm_calls(plan, CFG) == 3
    assert plan.k_sparse == 100 and plan.k_dense == 100 and plan.k_final == 10


def test_heuristic_domain_adaptation_safety_critical_vs_general() -> None:
    q = "corrosion allowance casing"
    general = HeuristicPlanner(CFG).plan(q, profile=get_profile("general"))
    civil = HeuristicPlanner(CFG).plan(q, profile=get_profile("civil_structural_engineering"))
    assert general.rerank == "none" and civil.rerank == "listwise" and civil.rerank_depth == 30
    assert civil.sparse_weight > general.sparse_weight
    assert civil.fusion == "convex"  # calibrated domain weight
    custom = DomainProfile(name="x", description="x", retrieval_hints={"sparse_weight": 0.9}, risk_level="low")
    plan = HeuristicPlanner(CFG).plan("how do seals fail", profile=custom)
    assert plan.sparse_weight == pytest.approx(0.8) and plan.fusion == "convex"  # 0.9 - 0.1, clamped <= 0.9


def test_heuristic_keyword_query_uses_query2doc_and_long_questions_extract_keywords() -> None:
    kw = HeuristicPlanner(CFG).plan("bearing preload procedure", profile=get_profile("mechanical_engineering"))
    assert kw.query_type == "keyword" and "query2doc" in kw.query_transforms
    long = HeuristicPlanner(CFG).plan("Which of the pump station specifications define the acceptance criteria for the hydrostatic test of the casing and nozzles?")
    assert "keywords" in long.query_transforms and long.query_type == "natural_language"


def test_deterministic_constraint_disables_llm_work() -> None:
    plan = HeuristicPlanner(CFG).plan("Has anything similar to the pump cavitation event happened before?", constraints=Constraints(deterministic=True))
    assert plan.rerank == "none" and all(t == "keywords" for t in plan.query_transforms)
    assert plan.expected_llm_calls == 0 and any("deterministic" in d for d in plan.degradations)


def test_apply_budget_degrades_in_order() -> None:
    base = HeuristicPlanner(CFG).plan("Has anything similar to the pump cavitation event happened before?", profile=get_profile("civil_structural_engineering"))
    assert base.rerank_depth == 30 and estimate_llm_calls(base, CFG) == 4  # hyde + multi_query + 2 windows
    p2 = apply_budget(base.model_copy(deep=True), CFG, remaining_calls=2)  # transforms go first, reranking is kept
    assert p2.query_transforms == [] and p2.rerank == "listwise" and p2.rerank_depth == 30
    assert p2.expected_llm_calls == 2 and p2.degradations == ["budget: dropped hyde", "budget: dropped multi_query"]
    p1 = apply_budget(base.model_copy(deep=True), CFG, remaining_calls=1)
    assert p1.rerank == "listwise" and p1.rerank_depth == 20 and p1.expected_llm_calls == 1
    assert p1.degradations[-1] == "budget: rerank depth reduced to one window"
    p0 = apply_budget(base.model_copy(deep=True), CFG, remaining_calls=0)
    assert p0.query_transforms == [] and p0.rerank == "none" and p0.expected_llm_calls == 0
    pw = base.model_copy(deep=True)
    pw.rerank = "pointwise"
    pw = apply_budget(pw, CFG, remaining_calls=3)
    assert pw.rerank == "listwise" and "budget: pointwise -> listwise" in pw.degradations
    assert apply_budget(base.model_copy(deep=True), CFG, remaining_calls=None).degradations == []


def test_n_windows_and_estimate() -> None:
    assert n_windows(10, 20, 10) == 1 and n_windows(20, 20, 10) == 1
    assert n_windows(30, 20, 10) == 2 and n_windows(100, 20, 10) == 9
    plan = RetrievalPlan(query="q", rerank="pointwise", rerank_depth=7, query_transforms=["hyde", "keywords"])
    assert estimate_llm_calls(plan, CFG) == 8


def test_from_planner_output_clamps_and_drops_unknown_filters() -> None:
    out = PlannerOutput(
        query_type="mixed",
        query_variants=["  torque spec 7342-118 ", "", "7342-118 torque", "a", "b", "c", "d"],
        use_sparse=False,
        use_dense=False,
        sparse_weight=1.7,
        fusion="convex",
        k_sparse=5000,
        k_dense=0,
        k_final=25,
        rerank="listwise",
        rerank_depth=5,
        query_transforms=["hyde", "hyde", "keywords"],
        filters=[FilterClause(field="type", op="in", value="drawing, specification"), FilterClause(field="ghost", op="eq", value="x")],
        rationale="  because  ",
    )
    plan = RetrievalPlan.from_planner_output(out, query="torque spec 7342-118", config=CFG, mandatory_filters={"tenant": "acme"}, corpus_fields={"type": ["drawing"]})
    assert plan.use_sparse and plan.use_dense  # both off -> both on
    assert plan.sparse_weight == 1.0 and plan.dense_weight == 0.0
    assert plan.k_sparse == 1000 and plan.k_dense == 1 and plan.k_final == 25
    assert plan.rerank_depth == 25  # raised to k_final
    assert plan.query_transforms == ["hyde", "keywords"]
    assert len(plan.query_variants) == 4 and plan.query_variants[0] == "torque spec 7342-118"
    assert [f.field for f in plan.filters] == ["type"]
    assert plan.filter_dict() == {"type": {"$in": ["drawing", "specification"]}, "tenant": "acme"}
    assert plan.rationale == "because" and plan.source == "llm"
    assert plan.all_queries() == ["torque spec 7342-118", "7342-118 torque", "a", "b"]
    assert "filters=" in plan.describe() and "source=llm" in plan.describe()


def test_filter_clause_forms_and_mandatory_filters_win() -> None:
    assert FilterClause(field="rev", op="eq", value="B").to_filter() == {"rev": {"$eq": "B"}}
    assert FilterClause(field="sev", op="gte", value="2").to_filter() == {"sev": {"$gte": "2"}}
    plan = RetrievalPlan(query="q", filters=[FilterClause(field="type", op="eq", value="memo")], mandatory_filters={"type": "spec"})
    assert plan.filter_dict() == {"type": "spec"}


def test_deterministic_keeps_the_free_cross_encoder() -> None:
    plan = RetrievalPlan(query="q", rerank="cross_encoder", query_transforms=["hyde", "keywords"])
    out = apply_budget(plan, CFG, remaining_calls=None, deterministic=True)
    assert out.rerank == "cross_encoder" and out.query_transforms == ["keywords"] and out.expected_llm_calls == 0
    llm_plan = RetrievalPlan(query="q", rerank="pointwise", query_transforms=["multi_query"])
    assert apply_budget(llm_plan, CFG, remaining_calls=None, deterministic=True).rerank == "none"


def test_from_planner_output_never_disables_the_sparse_channel() -> None:
    out = PlannerOutput(
        query_type="natural_language",
        query_variants=["why does the pump cavitate"],
        use_sparse=False,
        use_dense=True,
        sparse_weight=0.3,
        fusion="rrf",
        k_sparse=100,
        k_dense=100,
        k_final=10,
        rerank="none",
        rerank_depth=20,
        query_transforms=[],
        filters=[],
        rationale="semantic question",
    )
    plan = RetrievalPlan.from_planner_output(out, query="why does the pump cavitate", config=CFG)
    assert plan.use_sparse and plan.use_dense and plan.source == "llm"
    assert "fusion=rrf" in plan.describe() and "sparse=" not in plan.describe()  # rank-based fusion shows no weights
