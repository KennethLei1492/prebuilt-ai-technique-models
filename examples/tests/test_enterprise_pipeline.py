"""Offline tests proving the enterprise pipeline wiring (examples/enterprise_pipeline.py).

Everything runs on the deterministic ``MockLLM``: no credentials, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import enterprise_pipeline as ep  # noqa: E402
from pipeline_core import Executor, MockLLM, ResponseCache, Retriever, Status, TaskRequest, Tracer, Verifier  # noqa: E402

OK_STATUSES = (Status.SUCCESS, Status.PARTIAL, Status.NEEDS_REVIEW, Status.CACHED)
SIMPLE = ep.DEMO_TASKS[0]["task"]
COMPLEX = ep.DEMO_TASKS[1]["task"]
ALL_COMPONENTS = {"hybrid_retrieval", "task_decomposition", "answer_verification", "parallelization", "caching", "pipeline_rag"}


def make_pipeline(**kwargs) -> ep.EnterprisePipeline:  # type: ignore[no-untyped-def]
    pipeline = ep.build_pipeline(offline=True, tracer=Tracer(name="test"), **kwargs)
    pipeline.index(ep.CORPUS)
    return pipeline


@pytest.fixture(scope="module")
def pipeline() -> ep.EnterprisePipeline:
    return make_pipeline()


@pytest.fixture(scope="module")
def simple_result(pipeline: ep.EnterprisePipeline):  # type: ignore[no-untyped-def]
    return pipeline.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering"))


@pytest.fixture(scope="module")
def complex_result(pipeline: ep.EnterprisePipeline):  # type: ignore[no-untyped-def]
    return pipeline.run_sync(TaskRequest(task=COMPLEX, domain="mechanical_engineering"))


# ----------------------------------------------------------------- wiring


def test_components_satisfy_protocols(pipeline: ep.EnterprisePipeline) -> None:
    assert isinstance(pipeline.retrieval, Retriever) and isinstance(pipeline.retriever, Retriever)
    assert isinstance(pipeline.verification, Verifier) and isinstance(pipeline.verifier, Verifier)
    assert isinstance(pipeline.parallel.executor, Executor) and isinstance(pipeline.executor, Executor)
    assert isinstance(pipeline.caching, ResponseCache) and isinstance(pipeline.cache, ResponseCache)
    assert isinstance(pipeline.decomposition.solver, ep.RAGSubTaskSolver)  # SubTaskSolver callable
    assert pipeline.rag.retriever is pipeline.retriever and pipeline.rag.verifier is pipeline.verifier and pipeline.rag.executor is pipeline.executor
    assert pipeline.rag.cache is pipeline.caching.cache
    assert pipeline.decomposition.executor is pipeline.executor and pipeline.decomposition.verifier is pipeline.verifier


def test_every_component_shares_the_cached_client_and_tracer(pipeline: ep.EnterprisePipeline) -> None:
    client = pipeline.llm
    assert isinstance(client, ep.CachedLLMClient)
    assert isinstance(client, MockLLM), "offline detection (RunContext.offline) must survive the cache wrapper"
    assert client.backend is pipeline.caching.backend
    for name, component in pipeline.components.items():
        if name == "caching":
            assert component.llm is pipeline.inner_llm
        else:
            assert component.llm is client, name
        assert component.tracer is pipeline.tracer, name


def test_registry_names_match_components(pipeline: ep.EnterprisePipeline) -> None:
    assert set(pipeline.components) == ALL_COMPONENTS
    for name, component in pipeline.components.items():
        assert component.spec.name == name


# ------------------------------------------------------------- demo tasks


def test_simple_task_routes_to_rag(simple_result) -> None:  # type: ignore[no-untyped-def]
    assert simple_result.status in OK_STATUSES
    assert simple_result.output_text.strip()
    out = simple_result.output
    assert isinstance(out, ep.EnterpriseOutput)
    assert out.route.route == "rag" and out.component == "pipeline_rag"
    assert out.citations, "a RAG answer must carry evidence"
    assert "3 mm" in simple_result.output_text
    assert out.verdict is not None and out.verdict["method"] == "answer_verification"
    assert [s.name for s in out.stages][:5] == ["cache_lookup", "route", "solve", "verify_final", "store"]
    assert 0.0 <= simple_result.confidence <= 1.0 and simple_result.confidence_basis


def test_complex_task_routes_to_decomposition(complex_result) -> None:  # type: ignore[no-untyped-def]
    assert complex_result.status in OK_STATUSES
    assert complex_result.output_text.strip()
    out = complex_result.output
    assert out.route.route == "decompose" and out.component == "task_decomposition"
    assert len(out.sub_tasks) >= 4
    assert all(st["status"] in ("success", "partial") for st in out.sub_tasks), out.sub_tasks
    assert any(st["component"] == "pipeline_rag" for st in out.sub_tasks), "sub-tasks are solved by RAG"
    assert any(st["component"] == "answer_verification" for st in out.sub_tasks), "the verify node is owned by the verifier"
    assert out.verdict is not None
    # the gate explains itself when it withholds SUCCESS
    if complex_result.status == Status.NEEDS_REVIEW:
        assert out.review_reasons


def test_final_gate_respects_domain_threshold(simple_result, complex_result) -> None:  # type: ignore[no-untyped-def]
    for result in (simple_result, complex_result):
        out = result.output
        assert out.threshold == pytest.approx(0.75)  # mechanical_engineering profile
        if result.status == Status.SUCCESS:
            assert result.confidence >= out.threshold and out.verdict["passed"]
        verdict_score = out.verdict["score"]
        assert result.confidence <= max(out.component_confidence, verdict_score) + 1e-9


# ----------------------------------------------------------------- caching


def test_repeated_identical_run_is_served_from_the_shared_cache() -> None:
    p = make_pipeline()
    before = p.caching.backend.snapshot_stats().hits
    first = p.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering"))
    assert first.status in OK_STATUSES and first.usage.cache_hits == 0
    second = p.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering"))
    assert second.status == Status.CACHED
    assert second.usage.cache_hits >= 1
    assert second.output_text == first.output_text
    assert second.output.cache_hit is True
    assert p.caching.backend.snapshot_stats().hits > before
    assert p.result_cache_hits == 1
    assert p.tracer.total_usage.cache_hits >= 1


def test_llm_level_cache_is_shared_and_usage_rolls_up() -> None:
    # force the RAG model through its LLM path so the (mock) client is actually called
    p = make_pipeline(rag_config={"llm_mode": "always"})
    first = p.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering", metadata={"no_cache": True}))
    assert first.status in OK_STATUSES
    assert first.usage.llm_calls > 0
    stages = [s for s in first.steps if s.component == ep.COMPONENT]
    assert first.usage.llm_calls == sum(s.usage.llm_calls for s in stages)
    assert first.usage.llm_calls >= first.output.component_usage.llm_calls
    root = next(s for s in p.tracer.spans if s.name == f"{ep.COMPONENT}.run")
    assert root.usage.llm_calls == first.usage.llm_calls, "the root span rolls up every nested LLM call"
    # every request is either served from a cache tier or reaches the model; semantic-hit
    # equivalence checks are extra model calls that the cache charges to the caller
    m = p.llm.metrics
    assert m.requests == (first.usage.llm_calls - m.verification_calls) + m.exact_hits + m.semantic_hits
    second = p.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering", metadata={"no_cache": True}))
    assert second.status in OK_STATUSES
    assert p.llm.metrics.exact_hits > 0, "identical prompts hit the exact tier of the shared cache"
    assert second.usage.cache_hits > 0
    assert second.usage.llm_calls < first.usage.llm_calls


def test_reindexing_a_changed_document_invalidates_cached_answers() -> None:
    p = make_pipeline()
    p.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering"))
    version = p.corpus_version
    changed = dict(ep.CORPUS[1])
    changed["text"] = changed["text"].replace("3 mm", "4 mm")
    p.index([changed])
    assert p.corpus_version != version
    again = p.run_sync(TaskRequest(task=SIMPLE, domain="mechanical_engineering"))
    assert again.status != Status.CACHED
    assert "4 mm" in again.output_text


# ----------------------------------------------------------------- tracing


def test_span_tree_names_every_component(pipeline: ep.EnterprisePipeline, simple_result, complex_result) -> None:  # type: ignore[no-untyped-def]
    components = {s.component for s in pipeline.tracer.spans}
    assert ALL_COMPONENTS <= components, components
    tree = pipeline.tracer.tree()
    for name in ALL_COMPONENTS | {ep.COMPONENT}:
        assert f"[{name}]" in tree, name
    assert simple_result.trace_id == complex_result.trace_id == pipeline.tracer.trace_id
    assert pipeline.tracer.summary()["errors"] == 0


def test_report_renders(pipeline: ep.EnterprisePipeline, complex_result) -> None:  # type: ignore[no-untyped-def]
    text = pipeline.report(complex_result)
    for marker in ("STATUS", "ROUTE", "STAGES", "SUB-TASKS", "VERDICT", "CITATIONS", "USAGE", "CACHE", "TRACE"):
        assert marker in text
    assert "[task_decomposition]" in text and "[pipeline_rag]" in text


def test_main_runs_offline(capsys) -> None:  # type: ignore[no-untyped-def]
    assert ep.main(["--offline", "--no-trace"]) == 0
    out = capsys.readouterr().out
    assert "simple factual question" in out and "complex multi-part design review" in out
    assert "status=cached" in out
