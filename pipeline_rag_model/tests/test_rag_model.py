"""End-to-end behaviour of PipelineRAGModel: MockLLM path, ScriptedLLM-planned path, status, domain, budget, composition."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pytest

from pipeline_core import Constraints, Evidence, MockLLM, ScriptedLLM, Status, TaskRequest, Tracer, Usage, Verdict, register_profile, DomainProfile

from pipeline_rag import PipelineRAGConfig, PipelineRAGModel, RAGOutput, RAGPlan
from pipeline_rag.evaluation import sample_corpus
from pipeline_rag.model import aggregate_confidence

QUESTION = "What corrosion allowance does the specification require for the pump casing?"


def plan_json(**over: Any) -> dict[str, Any]:
    base = {
        "route": "single_step",
        "query_transforms": ["rewrite"],
        "k": 6,
        "grade_context": True,
        "compress": True,
        "max_hops": 1,
        "verify": True,
        "citation_style": "inline",
        "rationale": "focused lookup of one quantity in the specification",
        "context_token_budget": 1500,
        "abstain_when_insufficient": True,
        "escalate_on_abstention": True,
        "use_hierarchy": False,
        "expected_answer_type": "numerical",
        "planner": "llm",
    }
    base.update(over)
    return base


TRANSFORM_JSON = {"rewrite": "pump casing corrosion allowance specification", "sub_queries": [], "step_back": "", "hypothetical_answer": ""}
GRADE_JSON = {"grades": [{"doc_id": "PS-100", "label": "correct", "score": 0.95, "reason": "states the allowance"}, {"doc_id": "REQ-CW", "label": "correct", "score": 0.8, "reason": "requirement"}]}
GEN_JSON = {
    "answer": "The specification requires a 3 mm corrosion allowance for the pump casing in seawater service [PS-100]. The same minimum is set by REQ-CW-020 [REQ-CW].",
    "citations": [{"doc_id": "PS-100", "quote": "3 mm corrosion allowance"}, {"doc_id": "REQ-CW", "quote": "minimum 3 mm corrosion allowance"}],
    "unsupported_claims": [],
    "confidence": 0.9,
    "abstained": False,
    "abstention_reason": "",
}
SUPPORT_JSON = {
    "statements": [
        {"statement": "The specification requires a 3 mm corrosion allowance for the pump casing in seawater service.", "support": "fully", "evidence_ids": ["PS-100"]},
        {"statement": "The same minimum is set by REQ-CW-020.", "support": "fully", "evidence_ids": ["REQ-CW"]},
    ],
    "is_relevant": True,
    "is_useful": 5,
}


# ------------------------------------------------------------ MockLLM path


def test_mock_run_single_step_success() -> None:
    model = PipelineRAGModel(MockLLM())
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, domain="mechanical_engineering"))
    assert result.status == Status.SUCCESS
    out = result.output
    assert isinstance(out, RAGOutput) and out.route == "single_step" and out.routes_tried == ["single_step"]
    assert "3 mm" in out.answer and out.cited_ids and all(c in out.evidence_ids for c in out.cited_ids)
    assert out.grade_decision == "correct"
    assert out.reflection is not None and out.reflection.method == "heuristic"
    assert result.confidence >= 0.75  # mechanical_engineering threshold
    assert result.evidence[0].metadata.get("grade") in ("correct", "ambiguous")
    assert result.usage.llm_calls == 0
    step_names = [s.name for s in result.steps]
    for name in ("plan", "index", "transform", "retrieve", "grade", "compress", "generate", "verify"):
        assert name in step_names


def test_mock_run_multi_step_iterates() -> None:
    model = PipelineRAGModel(MockLLM(), max_hops=3)
    task = "What was the root cause of the high vibration trip on pump P-102 and which maintenance procedure was updated as a result?"
    result = model.run_sync(TaskRequest(task=task, inputs={"documents": sample_corpus()}))
    out = result.output
    assert out.route == "multi_step" and len(out.hops) >= 2
    assert out.hops[0].triggered_by == "initial" and out.hops[1].triggered_by == "schedule"
    assert "iterate" in [s.name for s in result.steps]
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW)


def test_mock_run_agentic_builds_hierarchy_and_summary_shape() -> None:
    model = PipelineRAGModel(MockLLM())
    result = model.run_sync(TaskRequest(task="What recurring themes appear across all the incident reports?", inputs={"documents": sample_corpus()}))
    out = result.output
    assert out.route == "agentic"
    plan = RAGPlan.model_validate(result.metadata["plan"])
    assert plan.use_hierarchy and plan.expected_answer_type == "summary"
    assert any(e.metadata.get("level", 0) >= 1 for e in result.evidence) or out.abstained


def test_no_corpus_abstains_offline_and_needs_review() -> None:
    model = PipelineRAGModel(MockLLM())
    result = model.run_sync(QUESTION)
    assert result.status == Status.NEEDS_REVIEW
    assert result.output.route == "no_retrieval" and result.output.abstained
    assert result.confidence <= 0.25 and "abstained" in result.confidence_basis
    assert "Insufficient evidence" in result.output_text


def test_unanswerable_question_escalates_then_abstains() -> None:
    model = PipelineRAGModel(MockLLM(), max_escalations=2)
    result = model.run_sync(TaskRequest(task="What warranty period does the pump vendor offer?", inputs={"documents": sample_corpus()}))
    out = result.output
    assert out.abstained and result.status == Status.NEEDS_REVIEW
    assert out.routes_tried == ["single_step", "multi_step", "agentic"]
    assert any("escalating" in w for w in result.warnings)
    no_esc = PipelineRAGModel(MockLLM(), max_escalations=0).run_sync(TaskRequest(task="What warranty period does the pump vendor offer?", inputs={"documents": sample_corpus()}))
    assert no_esc.output.routes_tried == ["single_step"]


def test_upstream_context_is_used_without_documents() -> None:
    model = PipelineRAGModel(MockLLM())
    ctx_ev = [Evidence(id="up-1", content="The specification requires a corrosion allowance of 3 mm for the pump casing.", source="upstream", retriever="provided")]
    result = model.run_sync(TaskRequest(task=QUESTION, context=ctx_ev))
    assert result.status == Status.SUCCESS and "[up-1]" in result.output.answer


def test_min_confidence_constraint_forces_review() -> None:
    model = PipelineRAGModel(MockLLM())
    ok = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    strict = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(min_confidence=0.99)))
    assert ok.status == Status.SUCCESS
    assert strict.status == Status.NEEDS_REVIEW and any("below threshold 0.99" in w for w in strict.warnings)


def test_footnote_citation_style_and_k_override() -> None:
    model = PipelineRAGModel(MockLLM(), citation_style="footnote")
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus(), "k": 2}))
    assert "[1]" in result.output_text and ("[1] PS-100" in result.output_text or "[1] REQ-CW" in result.output_text)
    assert RAGPlan.model_validate(result.metadata["plan"]).k == 2
    assert len(result.output.evidence_ids) <= 2


def test_full_corpus_gate_bypasses_ranking() -> None:
    model = PipelineRAGModel(MockLLM(), full_corpus_gate_tokens=100_000)
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    assert result.output.full_corpus_used
    assert "3 mm" in result.output.answer


# ------------------------------------------------------- ScriptedLLM path


def test_llm_planned_path_with_scripted_llm() -> None:
    llm = ScriptedLLM([plan_json(), TRANSFORM_JSON, GRADE_JSON, GEN_JSON, SUPPORT_JSON])
    model = PipelineRAGModel(llm, llm_mode="always")
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, domain="mechanical_engineering"))
    assert result.status == Status.SUCCESS, (result.warnings, result.errors)
    plan = RAGPlan.model_validate(result.metadata["plan"])
    assert plan.planner == "llm" and plan.k == 6 and plan.expected_answer_type == "numerical"
    assert plan.rationale.startswith("focused lookup")
    out = result.output
    assert out.answer == GEN_JSON["answer"]
    assert out.reflection is not None and out.reflection.method == "llm" and out.reflection.is_supported == "fully"
    assert out.confidence.support == 1.0 and out.confidence.citation_coverage == 1.0 and out.confidence.self_reported == 0.9
    assert result.usage.llm_calls == 5
    tags = [c.tags[0] for c in llm.calls]
    assert tags == ["pipeline_rag.plan", "pipeline_rag.transform", "pipeline_rag.grade", "pipeline_rag.generate", "pipeline_rag.support"]
    plan_req = llm.calls[0]
    assert plan_req.output_schema is not None and plan_req.effort == "high"
    assert QUESTION in plan_req.last_user_text() and "heuristic_suggestion" in plan_req.last_user_text()
    assert all("Domain: mechanical_engineering" in "\n".join(c.system_stable) for c in llm.calls)
    assert llm.calls[3].effort == "high" and llm.calls[2].effort == "low"
    grade_evidence = [e for e in result.evidence if e.id == "PS-100"]
    assert grade_evidence and grade_evidence[0].metadata["grade_score"] == 0.95


def test_llm_plan_is_normalised_and_domain_enforced() -> None:
    bad_plan = plan_json(route="single_step", k=9999, max_hops=50, verify=False, grade_context=False, citation_style="none", query_transforms=["hyde", "step_back", "rewrite"])
    llm = ScriptedLLM([bad_plan, GEN_JSON, SUPPORT_JSON])
    model = PipelineRAGModel(llm, llm_mode="always", config=PipelineRAGConfig(max_query_variants=2))
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, domain="civil_structural_engineering"))
    plan = RAGPlan.model_validate(result.metadata["plan"])
    assert plan.k == 100 and plan.max_hops == 1
    assert plan.verify and plan.grade_context and plan.citation_style == "inline"  # safety_critical overrides
    assert "hyde" not in plan.query_transforms and len(plan.query_transforms) <= 2
    assert llm.calls[-1].effort == "max"  # safety-critical verification effort


def test_llm_plan_failure_falls_back_to_heuristic() -> None:
    llm = ScriptedLLM(["not json at all {{{", GEN_JSON, SUPPORT_JSON])
    model = PipelineRAGModel(llm, llm_mode="always")
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    plan = RAGPlan.model_validate(result.metadata["plan"])
    assert plan.planner == "heuristic"
    assert any("planning failed" in w for w in result.warnings)
    assert result.status in (Status.PARTIAL, Status.SUCCESS, Status.NEEDS_REVIEW)


def test_llm_generation_abstention_escalates_once_then_reviews() -> None:
    abstain = {"answer": "", "citations": [], "unsupported_claims": [], "confidence": 0.2, "abstained": True, "abstention_reason": "no warranty terms in the documents"}
    llm = ScriptedLLM([plan_json(verify=False, grade_context=False, query_transforms=["rewrite"]), TRANSFORM_JSON, abstain, TRANSFORM_JSON, abstain])
    model = PipelineRAGModel(llm, llm_mode="always", max_escalations=1)
    result = model.run_sync(TaskRequest(task="What warranty period does the pump vendor offer?", inputs={"documents": sample_corpus()}))
    assert result.status == Status.NEEDS_REVIEW and result.output.abstained
    assert result.output.routes_tried == ["single_step", "multi_step"]
    assert result.confidence <= 0.25


def test_unsupported_claims_in_high_risk_domain_need_review() -> None:
    gen = dict(GEN_JSON, unsupported_claims=["The vendor warranty is 5 years."])
    llm = ScriptedLLM([plan_json(), TRANSFORM_JSON, GRADE_JSON, gen, SUPPORT_JSON])
    high = PipelineRAGModel(llm, llm_mode="always").run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, domain="electrical_engineering"))
    assert high.status == Status.NEEDS_REVIEW and any("unsupported" in w for w in high.warnings)
    llm2 = ScriptedLLM([plan_json(), TRANSFORM_JSON, GRADE_JSON, gen, SUPPORT_JSON])
    low = PipelineRAGModel(llm2, llm_mode="always").run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, domain="general"))
    assert low.status == Status.SUCCESS and low.output.unsupported_claims == ["The vendor warranty is 5 years."]


# ------------------------------------------------------ domain adaptation


def test_domain_adaptation_general_vs_safety_critical() -> None:
    docs = sample_corpus()
    general = PipelineRAGModel(MockLLM(), citation_style="none").run_sync(TaskRequest(task=QUESTION, inputs={"documents": docs}, domain="general"))
    safety = PipelineRAGModel(MockLLM(), citation_style="none").run_sync(TaskRequest(task=QUESTION, inputs={"documents": docs}, domain="chemical_process_engineering"))
    gp = RAGPlan.model_validate(general.metadata["plan"])
    sp = RAGPlan.model_validate(safety.metadata["plan"])
    assert gp.citation_style == "none" and sp.citation_style == "inline"
    assert "safety_critical" in sp.rationale and "safety_critical" not in gp.rationale
    assert "[" not in general.output_text and "[" in safety.output_text
    assert safety.output.confidence.threshold == 0.85 and general.output.confidence.threshold == 0.5


def test_custom_registered_profile_changes_behaviour_without_code() -> None:
    register_profile(
        DomainProfile(
            name="test_aero_structures",
            description="Aerospace structures test profile.",
            glossary={"FoS": "Factor of safety - ratio of allowable to applied load"},
            risk_level="safety_critical",
            min_confidence_to_accept=0.999,
        ),
        overwrite=True,
    )
    model = PipelineRAGModel(MockLLM())
    result = model.run_sync(TaskRequest(task="What FoS applies to the casing bolts?", inputs={"documents": sample_corpus()}, domain="test_aero_structures"))
    kinds = [v.kind for v in result.output.query_variants]
    assert "glossary" in kinds  # glossary expansion driven purely by the profile
    assert result.status == Status.NEEDS_REVIEW  # 0.999 threshold cannot be met


# ------------------------------------------------------------- budgets


def test_budget_one_call_goes_to_generation() -> None:
    llm = ScriptedLLM([GEN_JSON])
    model = PipelineRAGModel(llm, llm_mode="always")
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(max_llm_calls=1)))
    assert result.status != Status.FAILED
    assert result.usage.llm_calls == 1 and llm.calls[0].tags[0] == "pipeline_rag.generate"
    assert RAGPlan.model_validate(result.metadata["plan"]).planner == "heuristic"
    assert any("budget" in w for w in result.warnings)


def test_budget_two_calls_plan_then_generate() -> None:
    llm = ScriptedLLM([plan_json(), GEN_JSON])
    model = PipelineRAGModel(llm, llm_mode="always")
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(max_llm_calls=2)))
    assert result.usage.llm_calls == 2
    assert [c.tags[0] for c in llm.calls] == ["pipeline_rag.plan", "pipeline_rag.generate"]
    assert result.output.reflection is not None and result.output.reflection.method == "heuristic"


def test_cost_budget_exhaustion_degrades_without_wasting_calls() -> None:
    llm = MockLLM(simulate_cost_as="claude-opus-5-5")
    model = PipelineRAGModel(llm, llm_mode="always")
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(max_cost_usd=0.0000001)))
    assert result.status == Status.PARTIAL  # the plan call blew the cost budget; every later stage degraded to heuristics
    assert result.usage.llm_calls == 1 and len(llm.calls) == 1
    assert any("BudgetExceededError" in w for w in result.warnings)
    assert "3 mm" in result.output.answer  # extractive answer still delivered


def test_multi_hop_extractive_answer_covers_both_facets() -> None:
    model = PipelineRAGModel(MockLLM())
    task = "What was the root cause of the high vibration trip on pump P-102 and which procedure was updated?"
    result = model.run_sync(TaskRequest(task=task, inputs={"documents": sample_corpus()}))
    assert result.output.route == "multi_step"
    assert "misalignment" in result.output.answer.lower() and "MP-14" in result.output.answer


def test_summary_question_offline_uses_summary_nodes() -> None:
    model = PipelineRAGModel(MockLLM())
    result = model.run_sync(TaskRequest(task="What recurring themes appear across all the incident reports?", inputs={"documents": sample_corpus()}))
    out = result.output
    assert out.route == "agentic" and not out.abstained
    assert any(c.doc_id.endswith("#summary") for c in out.citations)
    assert result.status == Status.NEEDS_REVIEW  # low self-reported confidence for a heuristic summary


# ----------------------------------------------------------- composition


class FakeRetriever:
    version = "v1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        self.calls.append(query)
        return [Evidence(id="ext-1", content="External index: the specification requires a corrosion allowance of 3 mm for the pump casing.", source="ext", score=5.0, rank=1, retriever="dense")]


class FakeVerifier:
    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self.seen: list[str] = []

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict:
        self.seen.append(answer)
        return Verdict(method="fake", passed=self.passed, score=0.95 if self.passed else 0.2, issues=[] if self.passed else ["number mismatch"], details={"usage": Usage(llm_calls=2, cost_usd=0.01)})


class FakeExecutor:
    def __init__(self) -> None:
        self.mapped = 0

    async def map(self, fn: Any, items: Sequence[Any], *, concurrency: Optional[int] = None) -> list[Any]:
        self.mapped += 1
        return [await fn(it) for it in items]


class FakeCache:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.hits = 0

    async def get(self, key: str) -> Optional[Any]:
        v = self.store.get(key)
        if v is not None:
            self.hits += 1
        return v

    async def set(self, key: str, value: Any, *, ttl_s: Optional[float] = None, tags: Sequence[str] = ()) -> None:
        self.store[key] = value

    async def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int:
        return 0


def test_injected_retriever_verifier_executor_cache_are_used() -> None:
    retriever, verifier, executor, cache = FakeRetriever(), FakeVerifier(True), FakeExecutor(), FakeCache()
    tracer = Tracer()
    model = PipelineRAGModel(MockLLM(), retriever=retriever, verifier=verifier, executor=executor, cache=cache, tracer=tracer)
    result = model.run_sync(TaskRequest(task=QUESTION))
    assert result.status == Status.SUCCESS
    assert retriever.calls and executor.mapped >= 1
    assert "ext-1" in result.output.cited_ids
    assert result.output.verdict and result.output.verdict["method"] == "fake"
    assert result.output.reflection.method == "verifier"
    assert result.usage.llm_calls == 2 and result.usage.cost_usd == pytest.approx(0.01)  # verifier usage charged
    again = model.run_sync(TaskRequest(task=QUESTION))
    assert cache.hits >= 1 and again.status == Status.SUCCESS
    assert "pipeline_rag.run" in tracer.tree()


def test_failed_verifier_verdict_forces_review() -> None:
    model = PipelineRAGModel(MockLLM(), retriever=FakeRetriever(), verifier=FakeVerifier(False))
    result = model.run_sync(TaskRequest(task=QUESTION))
    assert result.status == Status.NEEDS_REVIEW and any("verifier fake failed" in w for w in result.warnings)


def test_result_can_feed_downstream_components() -> None:
    model = PipelineRAGModel(MockLLM())
    first = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    ev = first.as_evidence()
    assert ev.retriever == "provided" and "3 mm" in ev.content
    second = model.run_sync(TaskRequest(task="What corrosion allowance applies to the casing?", context=[ev]))
    assert second.output.route == "single_step" and "3 mm" in second.output.answer


def test_aggregate_confidence_renormalises_and_caps() -> None:
    w = PipelineRAGConfig().confidence_weights()
    c, basis = aggregate_confidence({"self_reported": 0.8, "citation_coverage": None, "support": None, "evidence_quality": None}, w)
    assert c == pytest.approx(0.8) and "self reported" in basis
    c2, _ = aggregate_confidence({"self_reported": 1.0, "citation_coverage": 1.0, "support": 1.0, "evidence_quality": 1.0}, w)
    assert c2 == pytest.approx(1.0)
    c3, b3 = aggregate_confidence({"self_reported": 0.9}, w, abstained=True)
    assert c3 == 0.25 and "abstained" in b3
    assert aggregate_confidence({}, w) == (0.0, "no confidence signals available")
