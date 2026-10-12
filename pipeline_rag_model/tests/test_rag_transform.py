"""Query transformation and Reciprocal Rank Fusion (pure)."""
from __future__ import annotations

import pytest

from pipeline_core import MockLLM, RunContext, ScriptedLLM, TaskRequest, Tracer, get_profile

from pipeline_rag import PipelineRAGConfig, QueryTransformer, heuristic_variants, reciprocal_rank_fusion
from pipeline_rag.router import extract_task_features
from pipeline_rag.transform import glossary_expansion, keyword_query, split_sub_queries, step_back_query


def test_rrf_maths_matches_formula() -> None:
    a = [("x", 9.0), ("y", 5.0), ("z", 1.0)]
    b = [("y", 0.9), ("x", 0.8)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    scores = dict(fused)
    assert scores["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["z"] == pytest.approx(1 / 63)
    # x and y tie exactly -> deterministic tie-break on id
    assert [d for d, _ in fused] == ["x", "y", "z"]


def test_rrf_weights_and_duplicates() -> None:
    a = [("x", 1.0), ("x", 0.5), ("y", 0.1)]  # duplicate inside a ranking counted once
    b = [("y", 1.0)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60, weights=[1.0, 0.5]))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 0.5 / 61)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a, b], weights=[1.0])
    assert reciprocal_rank_fusion([]) == []


def test_rrf_ignores_input_scores() -> None:
    a = [("x", 1000.0), ("y", 0.001)]
    b = [("x", 0.0001), ("y", 0.00005)]
    assert [d for d, _ in reciprocal_rank_fusion([a, b])] == ["x", "y"]


def test_keyword_and_step_back_queries() -> None:
    q = "Why does relay K12-A fail at 85 degC under the derating rule?"
    kq = keyword_query(q)
    assert "K12-A" in kq and "85" in kq and "why" not in kq.lower().split()
    sb = step_back_query(q)
    assert "K12-A" not in sb and "general principle" in sb


def test_split_sub_queries_on_clauses() -> None:
    parts = split_sub_queries("What is the casing material and what is the design pressure?")
    assert len(parts) == 2
    assert any("casing" in p for p in parts) and any("pressure" in p for p in parts)


def test_heuristic_variants_keep_literal_first_and_respect_identifiers() -> None:
    q = "Why does relay K12-A fail at 85 degC?"
    f = extract_task_features(q, {"documents": ["x"]})
    vs = heuristic_variants(q, ["rewrite", "step_back", "multi_query"], features=f)
    assert vs[0].kind == "literal" and vs[0].text == q
    assert all(v.kind != "step_back" for v in vs)  # identifiers present
    vs2 = heuristic_variants("Why does the casing corrode in seawater?", ["rewrite", "step_back"])
    assert [v.kind for v in vs2] == ["literal", "rewrite", "step_back"]
    assert len(heuristic_variants(q, ["rewrite", "step_back", "multi_query"], max_variants=1)) <= 2


def test_glossary_expansion_uses_profile_terms() -> None:
    prof = get_profile("mechanical_engineering")
    out = glossary_expansion("What FoS applies to the casing bolts?", prof.glossary)
    assert out is not None and "Factor of safety" in out
    assert glossary_expansion("What is the design pressure?", prof.glossary) is None
    assert glossary_expansion("anything", {}) is None


async def test_transformer_offline_and_llm_paths() -> None:
    cfg = PipelineRAGConfig(max_query_variants=4)
    tr = QueryTransformer(cfg)
    offline = await tr.transform("Why does the casing corrode in seawater?", ["rewrite", "step_back"], use_llm=False)
    assert [v.kind for v in offline] == ["literal", "rewrite", "step_back"]

    llm = ScriptedLLM([{"rewrite": "casing corrosion seawater", "sub_queries": ["casing material", "seawater corrosion rate"], "step_back": "corrosion allowance principle", "hypothetical_answer": ""}])
    req = TaskRequest(task="q", inputs={})
    ctx = RunContext(req, Tracer(), llm, "pipeline_rag")
    out = await tr.transform("Why does the casing corrode in seawater?", ["rewrite", "multi_query", "step_back"], ctx=ctx, use_llm=True)
    kinds = [v.kind for v in out]
    assert kinds[0] == "literal" and "rewrite" in kinds and "multi_query" in kinds and "step_back" in kinds
    assert len(out) <= cfg.max_query_variants + 1
    assert llm.calls[0].output_schema is not None
    assert "pipeline_rag.transform" in llm.calls[0].tags


async def test_transformer_falls_back_when_llm_fails() -> None:
    tr = QueryTransformer(PipelineRAGConfig())
    llm = ScriptedLLM([])  # exhausted immediately -> LLMError
    ctx = RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")
    out = await tr.transform("What is the design pressure of the casing?", ["rewrite"], ctx=ctx, use_llm=True)
    assert out[0].kind == "literal" and len(out) >= 2
    assert any("transform failed" in w for w in ctx.warnings)


async def test_transformer_delegates_to_retriever_transforms() -> None:
    class R:
        async def transform_queries(self, question: str, wanted: list[str]) -> list[str]:
            return ["delegated query one", "delegated query two"]

    tr = QueryTransformer(PipelineRAGConfig())
    out = await tr.transform("What is the design pressure?", ["rewrite", "multi_query"], retriever=R(), use_llm=False)
    assert [v.text for v in out[1:]] == ["delegated query one", "delegated query two"]


async def test_transformer_preserves_call_budget() -> None:
    tr = QueryTransformer(PipelineRAGConfig())
    llm = MockLLM()
    from pipeline_core import Constraints

    ctx = RunContext(TaskRequest(task="q", constraints=Constraints(max_llm_calls=1)), Tracer(), llm, "pipeline_rag")
    out = await tr.transform("What is the design pressure?", ["rewrite"], ctx=ctx, use_llm=True)
    assert llm.calls == [] and len(out) >= 1
