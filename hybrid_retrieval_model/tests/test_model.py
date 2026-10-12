"""HybridRetrievalModel end to end: LLM-planned path, status/confidence, domains, budgets, inputs.

The LLM-planned path is exercised with ``OnlineScriptedLLM`` (``hr_fixtures.py``), a
scripted ``BaseLLMClient`` that is *not* a ``MockLLM``: spec 3.1.2 routes every
``MockLLM`` (``ScriptedLLM`` included) to the heuristic planner because ``ctx.offline``.
"""
from __future__ import annotations

import json

import pytest
from hr_fixtures import OnlineScriptedLLM

from pipeline_core import ENGINEERING_RIGOR, Constraints, Evidence, MockLLM, ScriptedLLM, Status, TaskRequest
from pipeline_core.errors import ConfigurationError

from hybrid_retrieval import HybridRetrievalConfig, HybridRetrievalModel, RetrievalPlan

LLM_PLAN = {
    "query_type": "mixed",
    "query_variants": ["Which clause gives the impeller material?", "impeller material clause duplex"],
    "use_sparse": True,
    "use_dense": True,
    "sparse_weight": 0.65,
    "fusion": "convex",
    "k_sparse": 50,
    "k_dense": 50,
    "k_final": 4,
    "rerank": "none",
    "rerank_depth": 20,
    "query_transforms": [],
    "filters": [{"field": "type", "op": "eq", "value": "specification"}, {"field": "ghost", "op": "eq", "value": "x"}],
    "rationale": "Material clauses live in specifications; the lexical weight is raised because 'duplex' and grade designations are exact tokens.",
}

EXPLORATORY = "Has anything similar to the pump cavitation event happened before?"


async def test_llm_planned_path_with_scripted_plan(corpus) -> None:
    llm = OnlineScriptedLLM([LLM_PLAN])
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run("Which clause gives the impeller material?", domain="general")
    assert result.status == Status.SUCCESS, result.output_text
    plan = RetrievalPlan.model_validate(result.metadata["plan"])
    assert plan.source == "llm" and plan.fusion == "convex" and plan.sparse_weight == 0.65 and plan.dense_weight == pytest.approx(0.35)
    assert plan.rationale.startswith("Material clauses") and plan.k_final == 4 and plan.k_sparse == 50
    assert [f.field for f in plan.filters] == ["type"]  # unknown corpus field dropped
    assert result.evidence and all(e.metadata["type"] == "specification" for e in result.evidence) and len(result.evidence) <= 4
    assert result.evidence[0].metadata["doc_id"] == "spec-pump"
    assert result.usage.llm_calls == 1 and result.output.n_queries == 2
    req = llm.calls[0]
    assert req.output_schema is not None and req.effort == "high"
    assert "<task>" in req.last_user_text() and "fields: " in req.last_user_text() and "<query_features>" in req.last_user_text()
    assert any("Domain: general" in s for s in req.system_stable) and not any(ENGINEERING_RIGOR in s for s in req.system_stable)
    assert any("Decision rules" in s for s in req.system_stable)


async def test_llm_planned_path_with_rerank_and_budget(corpus) -> None:
    plan = {**LLM_PLAN, "rerank": "listwise", "rerank_depth": 6, "filters": [], "query_transforms": ["multi_query"]}
    llm = OnlineScriptedLLM([plan, {"queries": ["duplex impeller material clause"]}, {"ranking": [2, 1, 3, 4, 5, 6]}])
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run(TaskRequest(task="Which clause gives the impeller material?", domain="mechanical_engineering", constraints=Constraints(max_llm_calls=3)))
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW)
    assert result.usage.llm_calls == 3 and result.output.reranked and "multi_query" in result.output.transforms_applied
    assert result.evidence[0].retriever == "reranked" and result.output.rerank_method == "listwise"
    assert any(ENGINEERING_RIGOR in s for s in llm.calls[0].system_stable)
    assert any(ENGINEERING_RIGOR in s for s in llm.calls[-1].system_stable)  # high-risk profile: rerank prompt too (spec 3.3)
    assert [s.name for s in result.steps if s.name in ("plan", "retrieve", "assess")] == ["plan", "retrieve", "assess"]


async def test_llm_planner_failure_falls_back_to_heuristic(corpus) -> None:
    llm = OnlineScriptedLLM(["this is not a plan"], loop=True)
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run("ERR-1203 cause")
    assert result.metadata["plan"]["source"] == "heuristic"
    assert any("LLM planner failed" in w for w in result.warnings)
    assert result.evidence and result.evidence[0].metadata["doc_id"] == "runbook-retry"


async def test_offline_scripted_llm_uses_heuristic_planner(corpus) -> None:
    """Spec 3.1.2: any MockLLM (ScriptedLLM included) means ctx.offline, so the planner prompt is never issued."""
    llm = ScriptedLLM([LLM_PLAN], loop=True)
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run("Which clause gives the impeller material?", domain="general")
    assert result.metadata["plan"]["source"] == "heuristic" and not result.errors
    assert not any("Decision rules" in s for call in llm.calls for s in call.system_stable)


async def test_budget_zero_calls_degrades_instead_of_failing(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    model.index(corpus)
    result = await model.run(TaskRequest(task=EXPLORATORY, constraints=Constraints(max_llm_calls=0)))
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW) and not result.errors
    plan = result.metadata["plan"]
    assert plan["rerank"] == "none" and plan["query_transforms"] == [] and plan["degradations"]
    assert result.usage.llm_calls == 0 and result.evidence


async def test_budget_one_call_online_spends_it_on_planning(corpus) -> None:
    plan = {**LLM_PLAN, "rerank": "listwise", "filters": [], "query_transforms": ["hyde", "multi_query"]}
    llm = OnlineScriptedLLM([plan])
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run(TaskRequest(task="Which clause gives the impeller material?", constraints=Constraints(max_llm_calls=1)))
    assert result.usage.llm_calls == 1 and not result.errors
    p = result.metadata["plan"]
    assert p["source"] == "llm" and p["rerank"] == "none" and p["query_transforms"] == [] and p["degradations"]


async def test_cost_budget_exhaustion_mid_execute_keeps_fused_evidence(corpus) -> None:
    """Regression (critical): a BudgetExceededError from max_cost_usd inside transforms/rerank must degrade, not fail."""
    llm = MockLLM(simulate_cost_as="claude-opus-5-5")
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run(TaskRequest(task=EXPLORATORY, domain="mechanical_engineering", constraints=Constraints(max_cost_usd=0.0001)))
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW), result.output_text
    assert not result.errors and result.evidence and result.output is not None
    assert result.usage.llm_calls == 1  # the first call blew the cost budget; everything after it was skipped
    assert result.usage.cost_usd > 0.0001
    assert any("budget exhausted" in w or "cost budget" in w for w in result.warnings)
    assert not result.output.reranked and result.output.rerank_method == "none"
    assert all(s.status != Status.FAILED for s in result.steps)


async def test_latency_budget_exhaustion_mid_execute_keeps_fused_evidence(corpus) -> None:
    """Regression (critical): max_latency_s exhausted after the first call skips the remaining fan-outs with warnings."""
    llm = MockLLM(latency_s=0.3)
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run(TaskRequest(task=EXPLORATORY, domain="mechanical_engineering", constraints=Constraints(max_latency_s=0.2)))
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW) and not result.errors and result.evidence
    assert result.usage.llm_calls == 1
    assert any("latency budget" in w for w in result.warnings)
    assert not result.output.reranked


async def test_needs_review_below_min_confidence(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    model.index(corpus)
    result = await model.run(TaskRequest(task="ERR-1203", constraints=Constraints(min_confidence=0.99), domain="software_engineering"))
    assert result.status == Status.NEEDS_REVIEW and result.evidence
    assert any("below acceptance threshold 0.99" in w for w in result.warnings) and result.metadata["threshold"] == 0.99


async def test_safety_critical_profile_changes_plan_and_threshold(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    model.index(corpus)
    general = await model.run("corrosion allowance casing", domain="general")
    civil = await model.run("corrosion allowance casing", domain="civil_structural_engineering")
    assert general.metadata["threshold"] == 0.5 and civil.metadata["threshold"] == 0.85
    gp, cp = general.metadata["plan"], civil.metadata["plan"]
    assert gp["rerank"] == "none" and cp["rerank"] == "listwise" and cp["rerank_depth"] == 30
    assert cp["sparse_weight"] > gp["sparse_weight"]
    assert general.usage.llm_calls < civil.usage.llm_calls
    assert general.evidence[0].metadata["doc_id"] == civil.evidence[0].metadata["doc_id"] == "spec-pump"
    # a bare MockLLM returns empty permutations: the rerank is attempted but never claimed (no confidence bonus)
    assert not civil.output.reranked and civil.output.rerank_method == "none" and civil.evidence[0].retriever == "fused"
    assert civil.confidence == general.confidence or abs(civil.confidence - general.confidence) < 0.10


async def test_weak_evidence_warning_and_no_match(corpus) -> None:
    model = HybridRetrievalModel(MockLLM(), k_final=5)
    model.index(corpus)
    result = await model.run(TaskRequest(task="zzqx quantum flux capacitor", constraints=Constraints(deterministic=True)))
    assert result.status == Status.NEEDS_REVIEW
    assert result.confidence < 0.5 and any("weak evidence" in w or "no candidates" in w for w in result.warnings)


async def test_empty_index_is_a_failed_result_not_an_exception() -> None:
    model = HybridRetrievalModel(MockLLM())
    result = await model.run("anything")
    assert result.status == Status.FAILED and result.errors and "no documents indexed" in result.errors[0]
    assert "No documents are indexed" in result.output_text


async def test_documents_in_inputs_filters_k_and_explicit_plan(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    result = await model.run(TaskRequest(task="pump", inputs={"documents": corpus, "filters": {"type": "incident"}, "k": 2}))
    assert len(result.evidence) == 1 and result.evidence[0].metadata["type"] == "incident"
    assert result.metadata["plan"]["mandatory_filters"] == {"type": "incident"} and result.metadata["plan"]["k_final"] == 2
    assert any(s.name == "index" for s in result.steps)
    explicit = await model.run(TaskRequest(task="impeller", inputs={"plan": {"use_dense": False, "rerank": "none", "query": "ignored"}}))
    assert explicit.metadata["plan"]["source"] == "explicit" and explicit.metadata["plan"]["query"] == "impeller"
    assert explicit.evidence[0].retriever == "sparse"
    override = await model.run(TaskRequest(task="find the seal plan", inputs={"query": "API 682 plan 11"}))
    assert override.metadata["plan"]["query"] == "API 682 plan 11" and override.evidence[0].id == "spec-pump#2"


async def test_context_evidence_is_indexed_when_store_is_empty() -> None:
    model = HybridRetrievalModel(MockLLM())
    ctx_docs = [Evidence(id="up-1", content="The relief valve set pressure is 18 bar.", source="upstream"), Evidence(id="up-2", content="Schedule memo for the quarter.", source="upstream")]
    result = await model.run(TaskRequest(task="relief valve set pressure", context=ctx_docs, constraints=Constraints(deterministic=True)))
    assert result.evidence and result.evidence[0].metadata["doc_id"] == "up-1" and result.evidence[0].source == "upstream"


async def test_deterministic_constraint_skips_llm_planner(corpus) -> None:
    llm = OnlineScriptedLLM([LLM_PLAN])
    model = HybridRetrievalModel(llm)
    model.index(corpus)
    result = await model.run(TaskRequest(task="Which clause gives the impeller material?", constraints=Constraints(deterministic=True)))
    assert result.metadata["plan"]["source"] == "heuristic" and result.usage.llm_calls == 0 and llm.calls == []


async def test_output_text_warnings_are_deduplicated(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    model.index(corpus)
    result = await model.run("corrosion allowance casing", domain="civil_structural_engineering")
    text = result.output_text
    assert "Warnings:" in text
    rendered = text.split("Warnings: ", 1)[1].split(" | ")
    assert len(rendered) == len(set(rendered))


def test_inconsistent_config_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        HybridRetrievalModel(MockLLM(), config={"chunk_size_tokens": 64, "chunk_overlap_tokens": 64})
    with pytest.raises(ConfigurationError):
        HybridRetrievalConfig(rerank_window=5, rerank_stride=10)


def test_direct_api_persistence_config_and_describe(corpus, tmp_path) -> None:
    model = HybridRetrievalModel(MockLLM(), config={"k_final": 3}, rrf_k=30, domain="software_engineering")
    assert model.cfg.k_final == 3 and model.cfg.rrf_k == 30 and model.config["rrf_k"] == 30
    assert HybridRetrievalModel(MockLLM(), config=HybridRetrievalConfig(k_final=7), k_final=8).cfg.k_final == 8
    model.index(corpus)
    ev = model.retrieve_sync("getUserName", k=1)
    assert ev and ev[0].metadata["doc_id"] == "runbook-retry"
    path = model.save_index(tmp_path / "index.json")
    fresh = HybridRetrievalModel(MockLLM())
    assert fresh.load_index(path) == model.retriever.n_chunks
    assert [e.id for e in fresh.retrieve_sync("getUserName", k=1)] == [e.id for e in ev]
    d = model.describe()
    assert d["name"] == "hybrid_retrieval" and d["index"]["documents"] == 5 and d["llm"]["client"] == "mock"
    assert model.remove_documents(["memo-schedule"]) == 1 and model.store.documents.keys() >= {"spec-pump"}
    json.dumps(model.run_sync("ERR-1203").model_dump(mode="json"))  # result is JSON-serialisable
