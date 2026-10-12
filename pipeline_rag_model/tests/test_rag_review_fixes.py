"""Regression tests for the adversarial review of pipeline_rag (one test per fixed defect)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from pipeline_core import (
    BaseLLMClient,
    ComponentSpec,
    Constraints,
    Evidence,
    LLMRequest,
    LLMResponse,
    MockLLM,
    PipelineComponent,
    RunContext,
    ScriptedLLM,
    Status,
    TaskRequest,
    TaskResult,
    Tracer,
    Verdict,
)

from pipeline_rag import PipelineRAGConfig, PipelineRAGModel, RAGPlan
from pipeline_rag.cli import load_documents
from pipeline_rag.context import extractive_compress
from pipeline_rag.evaluation import QAExample, RAGEvaluator, sample_corpus
from pipeline_rag.generator import split_statements
from pipeline_rag.iterate import MultiHopLoop
from pipeline_rag.model import aggregate_confidence

QUESTION = "What corrosion allowance does the specification require for the pump casing?"
ANSWER_TEXT = "The specification requires a corrosion allowance of 3 mm for the pump casing in seawater service."


def _tag(req: LLMRequest) -> str:
    return req.tags[0] if req.tags else ""


# ------------------------------------------------- escalation builds the hierarchy (major)


def test_escalation_into_agentic_builds_summary_nodes() -> None:
    model = PipelineRAGModel(MockLLM(), max_escalations=2)
    docs = sample_corpus()
    result = model.run_sync(TaskRequest(task="What warranty period does the pump vendor offer?", inputs={"documents": docs}))
    assert result.output.routes_tried == ["single_step", "multi_step", "agentic"]
    inline = next(iter(model._inline_indexes.values()))
    assert inline.summary_chunks(), "the escalated agentic pass must build RAPTOR summary nodes"
    assert "index_3" in [s.name for s in result.steps]
    retrieve_ids = [i for s in result.steps if s.name == "retrieve_3" for i in s.details["evidence_ids"]]
    assert any(i.endswith("#summary") for i in retrieve_ids)


# ------------------------------------------- empty retrieval still runs the CRAG fallback (major)


class EmptyRetriever:
    version = "empty-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        self.calls += 1
        return []


class AnswerRetriever:
    version = "fallback-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        self.calls += 1
        return [Evidence(id="web-1", content=ANSWER_TEXT, source="fallback", score=3.0, rank=1, retriever="web")]


def test_empty_retrieval_enters_crag_fallback() -> None:
    primary, fallback = EmptyRetriever(), AnswerRetriever()
    model = PipelineRAGModel(MockLLM(), retriever=primary, fallback_retriever=fallback)
    result = model.run_sync(TaskRequest(task=QUESTION))
    assert primary.calls >= 1 and fallback.calls >= 1
    out = result.output
    assert not out.abstained and "3 mm" in out.answer and "web-1" in out.cited_ids
    assert any("fallback retrieval" in w for w in result.warnings)


# ------------------------------------------------ nested verifier usage is charged (major)


class CostlyVerifier(PipelineComponent):
    """A component verifier that spends LLM calls inside its own run()."""

    spec = ComponentSpec(name="costly_verifier", description="test verifier")

    def __init__(self) -> None:
        super().__init__(MockLLM(simulate_cost_as="claude-opus-5-5"))
        self.seen: list[TaskRequest] = []

    async def plan(self, request: TaskRequest, ctx: RunContext) -> Any:
        return None

    async def execute(self, request: TaskRequest, plan: Any, ctx: RunContext) -> TaskResult:
        self.seen.append(request)
        for _ in range(2):
            await ctx.llm_call(LLMRequest.simple("check the answer"), step="check")
        verdict = Verdict(method="costly", passed=True, score=0.93)
        return self.new_result(request, output=verdict, confidence=0.93)


def test_component_verifier_runs_via_run_and_usage_rolls_up() -> None:
    verifier = CostlyVerifier()
    model = PipelineRAGModel(MockLLM(), verifier=verifier)
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(max_llm_calls=50)))
    assert result.output.verdict["method"] == "costly"
    child = verifier.seen[0]
    assert child.inputs == {"answer": result.output.answer} and child.context
    assert child.constraints.max_llm_calls == 50  # remaining budget handed down (0 calls spent offline before verify)
    assert result.usage.llm_calls == 2 and result.usage.cost_usd > 0  # charged with ctx.charge


def test_real_answer_verification_model_verdict_is_consumed() -> None:
    from answer_verification import AnswerVerificationModel

    model = PipelineRAGModel(MockLLM(), verifier=AnswerVerificationModel(MockLLM()))
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    assert result.output.verdict is not None and result.output.verdict["method"] == "answer_verification"
    assert result.output.reflection is not None and result.output.reflection.method == "verifier"


# ----------------------------------------------- retrieval_trigger_threshold is live (major)


def _hop_ctx(llm: Any) -> RunContext:
    return RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")


@pytest.mark.parametrize("threshold,expect_hop", [(0.2, True), (0.6, False)])
async def test_retrieval_trigger_threshold_changes_hop_behaviour(threshold: float, expect_hop: bool) -> None:
    step = {"reasoning_step": "The casing allowance is stated.", "is_relevant": True, "need_more_information": False, "need_probability": 0.4, "next_query": "casing material grade", "can_answer": True, "confidence": 0.9}
    llm = ScriptedLLM([step], loop=True)
    queries: list[str] = []

    async def retrieve(q: str, k: int) -> list[Evidence]:
        queries.append(q)
        return [Evidence(id=f"h-{len(queries)}", content=f"Material grade note {len(queries)}.")]

    loop = MultiHopLoop(PipelineRAGConfig(retrieval_trigger_threshold=threshold))
    _, records, _ = await loop.run("What allowance applies?", [], retrieve, max_hops=2, ctx=_hop_ctx(llm), use_llm=True)
    if expect_hop:
        assert queries == ["casing material grade"] and records[-1].triggered_by == "retrieval_trigger"
    else:
        assert queries == [] and records[-1].stop_reason == "model can answer"


# ----------------------------------------------- contextual_prefix='llm' is live (major)


def test_llm_contextual_prefixes_are_generated_once_per_index() -> None:
    def responder(req: LLMRequest) -> Any:
        if _tag(req) == "pipeline_rag.context_prefix":
            return {"context": "Situating context: pump casing design specification."}
        from pipeline_core.llm import example_from_schema

        return example_from_schema(req.output_schema) if req.output_schema else "ok"

    llm = MockLLM(responder)
    model = PipelineRAGModel(llm, llm_mode="always", contextual_prefix="llm")
    docs = sample_corpus()[:2]
    model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": docs}))
    prefix_calls = [c for c in llm.calls if _tag(c) == "pipeline_rag.context_prefix"]
    inline = next(iter(model._inline_indexes.values()))
    assert prefix_calls and len(prefix_calls) == len(inline.leaf_chunks())
    assert inline.llm_prefixed and all(c.prefix.startswith("Situating context") for c in inline.leaf_chunks())
    llm.reset()
    model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": docs}))
    assert not [c for c in llm.calls if _tag(c) == "pipeline_rag.context_prefix"]  # cached index: no second pass


def test_heuristic_prefix_mode_makes_no_prefix_calls() -> None:
    llm = MockLLM()
    PipelineRAGModel(llm, llm_mode="always").run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()[:2]}))
    assert not [c for c in llm.calls if _tag(c) == "pipeline_rag.context_prefix"]


# --------------------------------------------------------------- minor fixes


def test_numeric_markers_resolve_with_doc_n_ids() -> None:
    sts = split_statements("The allowance is 3 mm [1]. The casing is duplex [2].", ["doc-1", "doc-2"])
    assert [s.citation_ids for s in sts] == [["doc-1"], ["doc-2"]]


def test_plan_hops_never_exceed_config_cap() -> None:
    plan = RAGPlan(route="multi_step", max_hops=5, rationale="x").normalized(PipelineRAGConfig(max_hops=1))
    assert plan.max_hops == 1


def test_partial_reason_names_the_degraded_stage() -> None:
    class Broken:
        async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
            raise RuntimeError("index down")

    result = PipelineRAGModel(MockLLM(), retriever=Broken()).run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    assert result.status == Status.PARTIAL
    assert any("retrieval leg injected" in r for r in result.metadata["review_reasons"])
    assert not any("LLM stages" in r for r in result.metadata["review_reasons"])


def test_cap_only_reported_when_it_binds() -> None:
    w = PipelineRAGConfig().confidence_weights()
    c, basis = aggregate_confidence({"self_reported": 0.15}, w, abstained=True)
    assert c == pytest.approx(0.15) and "capped" not in basis
    c2, basis2 = aggregate_confidence({"self_reported": 0.9}, w, abstained=True)
    assert c2 == 0.25 and "capped at 0.25" in basis2


def test_colliding_auto_ids_across_sources_are_kept_apart() -> None:
    model = PipelineRAGModel(MockLLM())
    model.index(["The motor is rated 75 kW at 400 V."])
    result = model.run_sync(TaskRequest(task="What is the casing corrosion allowance and the motor rating?", inputs={"documents": ["The casing corrosion allowance is 3 mm."]}))
    contents = {e.id: e.content for e in result.evidence}
    assert any("3 mm" in c for c in contents.values()) and any("75 kW" in c for c in contents.values())
    assert any(i.startswith("inline:") for i in contents)
    assert any("different text" in w for w in result.warnings)


def test_persistent_index_does_not_overwrite_auto_ids() -> None:
    model = PipelineRAGModel(MockLLM())
    model.index(["The motor is rated 75 kW."])
    model.index(["The casing allowance is 3 mm."])
    assert set(model.builtin.documents) == {"doc-1", "doc-1-1"}


async def test_latency_budget_exhausted_makes_no_llm_calls() -> None:
    llm = MockLLM()
    model = PipelineRAGModel(llm, llm_mode="always")
    result = await model.run(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(max_latency_s=1e-9)))
    assert llm.calls == [] and result.usage.llm_calls == 0
    assert not any("BudgetExceededError" in w for w in result.warnings)
    assert result.status == Status.PARTIAL and "generate" in result.metadata["degraded_stages"]


def test_deterministic_constraint_limits_run_to_generation() -> None:
    llm = MockLLM()
    result = PipelineRAGModel(llm, llm_mode="always").run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}, constraints=Constraints(deterministic=True)))
    tags = [_tag(c) for c in llm.calls]
    assert tags and set(tags) == {"pipeline_rag.generate"}  # one generation per Self-Route pass, nothing else sampled
    assert len(tags) == len(result.output.routes_tried)
    assert RAGPlan.model_validate(result.metadata["plan"]).planner == "heuristic"


def test_evaluator_attributes_fp3_when_gold_lost_in_consolidation() -> None:
    model = PipelineRAGModel(MockLLM())
    ex = QAExample(question=QUESTION, reference_answer="3 mm", relevant_ids=["PS-100"])
    from pipeline_core import StepRecord

    from pipeline_rag import RAGOutput

    result = TaskResult(
        request_id="r",
        component="pipeline_rag",
        output=RAGOutput(question=QUESTION, answer="The allowance is 6 mm [X-1].", route="single_step"),
        evidence=[Evidence(id="X-1", content="The allowance is 6 mm for flanges.")],
        steps=[StepRecord(name="retrieve", component="pipeline_rag", details={"evidence_ids": ["PS-100#c0", "X-1"]})],
    )
    rec = asyncio.run(RAGEvaluator(model).score(ex, result))
    assert rec.failure_point == "FP3"


def test_cli_loads_id_to_text_mapping(tmp_path: Path) -> None:
    f = tmp_path / "docs.json"
    f.write_text(json.dumps({"PS-100": "Corrosion allowance 3 mm.", "MP-14": {"text": "Alignment procedure.", "source": "mp14.pdf"}}), encoding="utf-8")
    docs = load_documents([str(f)])
    assert [d["id"] for d in docs] == ["PS-100", "MP-14"] and docs[1]["source"] == "mp14.pdf"
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_documents([str(empty)])


class UnversionedRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        self.calls += 1
        return [Evidence(id="ext-1", content=ANSWER_TEXT, source="ext", score=2.0, rank=1)]


class DictCache:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.invalidated: list[Sequence[str]] = []

    async def get(self, key: str) -> Optional[Any]:
        return self.store.get(key)

    async def set(self, key: str, value: Any, *, ttl_s: Optional[float] = None, tags: Sequence[str] = ()) -> None:
        self.store[key] = value

    async def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int:
        self.invalidated.append(tags)
        n = len(self.store)
        self.store.clear()
        return n


async def test_unversioned_retriever_cache_can_be_invalidated() -> None:
    retr, cache = UnversionedRetriever(), DictCache()
    model = PipelineRAGModel(MockLLM(), retriever=retr, cache=cache)
    await model.retrieve(QUESTION)
    await model.retrieve(QUESTION)
    assert retr.calls == 1  # second call served from cache
    assert await model.invalidate_retrieval_cache() >= 1 and cache.invalidated == [["pipeline_rag"]]
    await model.retrieve(QUESTION)
    assert retr.calls == 2


def test_cache_errors_are_surfaced_once() -> None:
    class BadCache(DictCache):
        async def get(self, key: str) -> Optional[Any]:
            raise ConnectionError("redis down")

    result = PipelineRAGModel(MockLLM(), retriever=UnversionedRetriever(), cache=BadCache()).run_sync(TaskRequest(task=QUESTION))
    assert sum("retrieval cache error" in w for w in result.warnings) == 1
    assert result.status != Status.FAILED and "3 mm" in result.output.answer


def test_budget_truncation_fits_token_budget() -> None:
    from pipeline_core import estimate_tokens

    long = Evidence(id="L", content=" ".join(["word"] * 2000), metadata={})
    out = extractive_compress([long], "word", sentences_per_unit=50, token_budget=200, protect_code=False)
    assert sum(estimate_tokens(e.content) for e in out) <= 200


def test_executor_failure_falls_back_to_local_gather() -> None:
    class BrokenExecutor:
        async def map(self, fn: Any, items: Sequence[Any], *, concurrency: Optional[int] = None) -> list[Any]:
            raise RuntimeError("pool closed")

    result = PipelineRAGModel(MockLLM(), executor=BrokenExecutor()).run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    assert "3 mm" in result.output.answer
    assert result.status == Status.PARTIAL and "executor" in result.metadata["degraded_stages"]


async def test_route_llm_direct_api() -> None:
    llm = ScriptedLLM([{"route": "multi_step", "complexity": "complex", "reason": "chains two reports"}])
    model = PipelineRAGModel(llm, llm_mode="always")
    decision = await model.route_llm("Which pump failed and why?", {"documents": sample_corpus()})
    assert decision.route == "multi_step" and model.direct_usage.llm_calls == 1
    assert _tag(llm.calls[0]) == "pipeline_rag.route"
    offline = await PipelineRAGModel(MockLLM()).route_llm(QUESTION, {"documents": sample_corpus()})
    assert offline.route == "single_step"


# ---------------------------------------------- wrapped offline client (coordinator request)


class WrappingClient(BaseLLMClient):
    """Cache-style wrapper exposing the wrapped client as ``inner``; must never be called offline."""

    name = "wrapper"

    def __init__(self, inner: BaseLLMClient) -> None:
        super().__init__(inner.model, effort=inner.effort)
        self.inner = inner

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("offline path expected: the wrapped MockLLM must not be called")


def test_wrapped_mock_client_takes_offline_path() -> None:
    model = PipelineRAGModel(WrappingClient(MockLLM()))
    result = model.run_sync(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    assert result.usage.llm_calls == 0 and "3 mm" in result.output.answer
    assert RAGPlan.model_validate(result.metadata["plan"]).planner == "heuristic"
    gen = asyncio.run(model.generate(QUESTION, result.evidence))  # direct API: extractive, no client call
    assert "3 mm" in gen.answer
    assert RAGEvaluator(model).judge.use_llm is False
