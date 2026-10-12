"""HybridRetriever: indexing, filters, channel metadata, transforms, fusion weights, rerank reporting, confidence and the protocol API."""
from __future__ import annotations

from typing import Sequence

import pytest

from pipeline_core import ENGINEERING_RIGOR, ScriptedLLM, TaskRequest, Tracer, get_profile
from pipeline_core.component import RunContext
from pipeline_core.protocols import Retriever

from hybrid_retrieval.config import HybridRetrievalConfig
from hybrid_retrieval.dense import Embedder, HashingEmbedder
from hybrid_retrieval.plan import RetrievalPlan, n_windows
from hybrid_retrieval.retriever import HybridRetriever, query_coverage, retrieval_confidence

LONG_QUESTION = "What is the root cause of the cooling water pump cavitation incident and which corrective action was taken afterwards?"


def test_query_coverage_and_confidence_pure_functions() -> None:
    assert query_coverage("pump impeller material", ["The impeller material is duplex", "pump datasheet"]) == 1.0
    assert query_coverage("pump impeller material", ["schedule memo"]) == 0.0
    assert query_coverage("the of", ["x"]) == 1.0
    assert retrieval_confidence(coverage=1.0, overlap=0.0, n_results=0, k=10, reranked=False, both_channels=True) == (0.0, "no candidates matched the query (or the filters admitted none)")
    full, basis = retrieval_confidence(coverage=1.0, overlap=1.0, n_results=10, k=10, reranked=True, both_channels=True)
    assert full == 1.0 and "reranked" in basis and "overlap@10 = 1.00" in basis
    weak, _ = retrieval_confidence(coverage=0.0, overlap=0.0, n_results=1, k=10, reranked=False, both_channels=True)
    assert weak == pytest.approx(0.16)
    single, basis = retrieval_confidence(coverage=1.0, overlap=0.0, n_results=1, k=10, reranked=False, both_channels=False)
    assert single == pytest.approx(0.71) and "single channel" in basis
    plain, _ = retrieval_confidence(coverage=0.5, overlap=0.5, n_results=10, k=10, reranked=False, both_channels=True)
    boosted, _ = retrieval_confidence(coverage=0.5, overlap=0.5, n_results=10, k=10, reranked=True, both_channels=True)
    assert boosted == pytest.approx(plain + 0.10)


def test_index_add_remove_counts(corpus) -> None:
    r = HybridRetriever()
    n = r.index(corpus)
    assert n == r.n_chunks and r.n_documents == 5 and len(r.sparse) == n and len(r.dense) == n
    assert r.add_documents([{"id": "spec-pump", "text": "replacement casing text"}]) == 1
    assert r.n_documents == 5 and len(r.sparse) == r.n_chunks == len(r.dense)
    assert r.remove_documents(["spec-pump", "ghost"]) == 1
    assert r.n_documents == 4 and len(r.sparse) == r.n_chunks
    assert r.add_documents([]) == 0
    assert r.describe()["documents"] == 4 and r.describe()["sparse"]["backend"] == "python"


async def test_add_documents_without_ids_never_desyncs_channels() -> None:
    """Regression (major): id-less dicts after id-less strings used to evict doc-1 from both channels but not the store."""
    r = HybridRetriever()
    assert r.index(["The relief valve set pressure is 18 bar."]) == 1
    assert r.add_documents([{"text": "Schedule memo for the third quarter."}]) == 1
    assert set(r.store.documents) == {"doc-1", "doc-2"}
    assert r.n_chunks == 2 == len(r.sparse) == len(r.dense)
    res = await r.search(RetrievalPlan(query="relief valve set pressure", k_final=2))
    assert res.evidence and res.evidence[0].metadata["doc_id"] == "doc-1"
    assert r.add_documents(["a third plain document"]) == 1 and "doc-3" in r.store.documents and r.n_chunks == 3 == len(r.sparse)


async def test_search_fused_metadata_and_channels(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    plan = RetrievalPlan(query="impeller material duplex", k_final=5)
    res = await r.search(plan)
    assert res.evidence and res.evidence[0].metadata["doc_id"] == "spec-pump"
    top = res.evidence[0]
    assert top.retriever == "fused" and top.rank == 1 and set(top.metadata["channels"]) <= {"sparse", "dense"}
    assert top.metadata["fused_score"] == top.score and top.metadata["sparse_rank"] is not None
    assert [c.name for c in res.channels] == ["sparse", "dense"] and res.n_queries == 1
    assert 0.0 <= res.overlap_at_k <= 1.0 and res.confidence > 0.5 and res.filtered_chunks is None
    assert not res.reranked and res.usage.llm_calls == 0


async def test_filters_are_pushed_into_channels(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    res = await r.search(RetrievalPlan(query="pump", mandatory_filters={"type": "incident"}, k_final=5))
    assert res.evidence and all(e.metadata["type"] == "incident" for e in res.evidence)
    assert res.filtered_chunks == 1
    none = await r.search(RetrievalPlan(query="pump", mandatory_filters={"type": "drawing"}))
    assert none.evidence == [] and none.filtered_chunks == 0 and none.confidence == 0.0 and none.warnings
    numeric = await r.search(RetrievalPlan(query="steel", mandatory_filters={"severity": {"$lte": 2}}, k_final=10))
    assert numeric.evidence and all(int(e.metadata["severity"]) <= 2 for e in numeric.evidence)


async def test_single_channel_plans_and_empty_index() -> None:
    empty = HybridRetriever()
    res = await empty.search(RetrievalPlan(query="anything"))
    assert res.evidence == [] and "index is empty" in res.warnings[0]


async def test_sparse_only_and_dense_only(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    sparse = await r.search(RetrievalPlan(query="ERR-1203", use_dense=False, k_final=3))
    assert sparse.evidence[0].retriever == "sparse" and sparse.evidence[0].metadata["channels"] == ["sparse"]
    assert [c.name for c in sparse.channels] == ["sparse"] and sparse.overlap_at_k == 0.0
    dense = await r.search(RetrievalPlan(query="cavitation", use_sparse=False, k_final=3))
    assert dense.evidence and dense.evidence[0].retriever == "dense" and [c.name for c in dense.channels] == ["dense"]


async def test_query_variants_are_fused_with_rrf(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    res = await r.search(RetrievalPlan(query="impeller material", query_variants=["impeller material", "duplex stainless steel grade"], k_final=5))
    assert res.n_queries == 2
    assert res.evidence[0].metadata["doc_id"] in ("spec-pump", "datasheet-steel")
    assert max(e.metadata["variants_hit"] for e in res.evidence) >= 2


# ------------------------------------------------------------- fusion weights
def test_fusion_weights_follow_the_method() -> None:
    """Regression (major): rank-based rrf/interleave are unweighted (Cormack et al. 2009); weights only drive convex/dbsf."""
    assert HybridRetriever.fusion_weights(RetrievalPlan(query="q", fusion="rrf", sparse_weight=0.2, dense_weight=0.8)) is None
    assert HybridRetriever.fusion_weights(RetrievalPlan(query="q", fusion="interleave", sparse_weight=0.2, dense_weight=0.8)) is None
    assert HybridRetriever.fusion_weights(RetrievalPlan(query="q", fusion="convex", sparse_weight=0.2, dense_weight=0.8)) == [0.2, 0.8]
    assert HybridRetriever.fusion_weights(RetrievalPlan(query="q", fusion="dbsf", sparse_weight=0.6, dense_weight=0.4)) == [0.6, 0.4]


async def test_rrf_ignores_plan_weights_but_convex_uses_them(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    lex = await r.search(RetrievalPlan(query="corrosion allowance casing", fusion="rrf", sparse_weight=0.9, dense_weight=0.1, k_final=8))
    sem = await r.search(RetrievalPlan(query="corrosion allowance casing", fusion="rrf", sparse_weight=0.1, dense_weight=0.9, k_final=8))
    assert [(e.id, e.score) for e in lex.evidence] == [(e.id, e.score) for e in sem.evidence]
    clex = await r.search(RetrievalPlan(query="corrosion allowance casing", fusion="convex", sparse_weight=0.9, dense_weight=0.1, k_final=8))
    csem = await r.search(RetrievalPlan(query="corrosion allowance casing", fusion="convex", sparse_weight=0.1, dense_weight=0.9, k_final=8))
    assert [e.score for e in clex.evidence] != [e.score for e in csem.evidence]


async def test_dbsf_and_interleave_through_the_retriever(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    for method in ("dbsf", "interleave"):
        res = await r.search(RetrievalPlan(query="impeller material duplex", fusion=method, k_final=4))
        assert res.evidence and res.evidence[0].metadata["doc_id"] in ("spec-pump", "datasheet-steel"), method
        assert res.evidence[0].retriever == "fused" and len(res.evidence) <= 4


# ----------------------------------------------------------------- embedders
class _SpecEmbedder:
    """The spec's minimal protocol: ``embed`` + ``dimension`` only (no ``embed_queries``)."""

    def __init__(self) -> None:
        self._inner = HashingEmbedder(64)
        self.dimension = 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.embed(texts)


async def test_spec_minimal_embedder_works_end_to_end(corpus) -> None:
    """Regression (major): an embedder with only embed()+dimension used to fail isinstance and crash the dense leg."""
    emb = _SpecEmbedder()
    assert isinstance(emb, Embedder) and not hasattr(emb, "embed_queries")
    r = HybridRetriever(embedder=emb, llm=ScriptedLLM([{"passages": ["The impeller material is duplex stainless steel ASTM A890."]}]))
    r.index(corpus)
    dense = await r.search(RetrievalPlan(query="impeller material duplex", use_sparse=False, k_final=3))
    assert dense.evidence and dense.evidence[0].metadata["doc_id"] in ("spec-pump", "datasheet-steel")
    hyde = await r.search(RetrievalPlan(query="impeller material", query_transforms=["hyde"], k_final=3))
    assert "hyde" in hyde.transforms_applied and hyde.evidence


# -------------------------------------------------------- transforms / rerank
async def test_transforms_and_rerank_through_direct_llm(corpus) -> None:
    llm = ScriptedLLM(
        [
            {"queries": ["duplex steel impeller grade", "unrelated budget memo"]},  # multi_query (second drifts)
            {"passages": ["The impeller material is duplex stainless steel ASTM A890."]},  # hyde
            {"ranking": [2, 1]},  # listwise over 2 candidates
        ]
    )
    cfg = HybridRetrievalConfig(expansion_min_overlap=0.5)
    r = HybridRetriever(cfg, llm=llm)
    r.index(corpus)
    plan = RetrievalPlan(query="impeller material", query_transforms=["multi_query", "hyde", "keywords"], rerank="listwise", rerank_depth=2, k_final=2)
    res = await r.search(plan)
    assert set(res.transforms_applied) == {"multi_query", "hyde"}  # keywords equal to the query itself are not added
    assert res.n_queries == 2 and res.reranked and res.rerank_method == "listwise"
    assert res.evidence[0].retriever == "reranked" and len(res.evidence) == 2
    assert res.usage.llm_calls == 3 and r.usage.llm_calls == 3
    assert any("drifted" in w for w in res.warnings)


async def test_keywords_variant_feeds_only_the_sparse_leg(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    res = await r.search(RetrievalPlan(query=LONG_QUESTION, query_transforms=["keywords"], k_final=5))
    assert res.transforms_applied == ["keywords"] and res.n_queries == 2
    stats = {c.name: c for c in res.channels}
    assert stats["sparse"].queries == 2 and stats["dense"].queries == 1  # bag-of-words never reaches the dense leg
    assert res.evidence[0].metadata["doc_id"] == "incident-7" and res.usage.llm_calls == 0


async def test_empty_rerank_output_is_not_reported_as_reranked(corpus) -> None:
    """Regression (major): a rerank whose windows all returned nothing must keep fused labels, scores and confidence."""
    r = HybridRetriever(llm=ScriptedLLM([{"ranking": []}], loop=True))
    r.index(corpus)
    plain = await r.search(RetrievalPlan(query="impeller material duplex", k_final=3))
    attempted = await r.search(RetrievalPlan(query="impeller material duplex", rerank="listwise", rerank_depth=5, k_final=3))
    assert r.usage.llm_calls >= 1 and any("empty ranking" in w for w in attempted.warnings)
    assert not attempted.reranked and attempted.rerank_method == "none"
    assert [(e.id, e.score, e.retriever) for e in attempted.evidence] == [(e.id, e.score, e.retriever) for e in plain.evidence]
    assert attempted.confidence == plain.confidence and "fused order" in attempted.confidence_basis


async def test_rerank_without_llm_is_skipped_with_warning(corpus) -> None:
    r = HybridRetriever()
    r.index(corpus)
    res = await r.search(RetrievalPlan(query="impeller material", rerank="listwise", query_transforms=["hyde"]))
    assert not res.reranked and any("skipped" in w for w in res.warnings)
    assert res.evidence and res.usage.llm_calls == 0
    ce = await r.search(RetrievalPlan(query="impeller material", rerank="cross_encoder"))
    assert not ce.reranked and any("no cross-encoder" in w for w in ce.warnings)


async def test_engineering_rigor_in_rerank_and_transform_prompts_for_high_risk_profiles(corpus) -> None:
    llm = ScriptedLLM([{"queries": ["duplex impeller material clause"]}, {"ranking": [1, 2, 3]}], loop=True)
    r = HybridRetriever(llm=llm)
    r.index(corpus)
    plan = RetrievalPlan(query="impeller material duplex", query_transforms=["multi_query"], rerank="listwise", rerank_depth=3, k_final=3)
    await r.search(plan, profile=get_profile("civil_structural_engineering"))
    assert len(llm.calls) == 2 and all(any(ENGINEERING_RIGOR in s for s in c.system_stable) for c in llm.calls)
    assert all(any("civil_structural_engineering" in s for s in c.system_stable) for c in llm.calls)
    llm.calls.clear()
    await r.search(plan, profile=get_profile("general"))
    assert len(llm.calls) == 2 and not any(ENGINEERING_RIGOR in s for c in llm.calls for s in c.system_stable)


async def test_cost_budget_exhaustion_inside_search_keeps_fused_candidates(corpus) -> None:
    """Regression (critical): BudgetExceededError raised by ctx.llm_call during rerank must not escape search()."""
    llm = ScriptedLLM([{"ranking": [3, 1, 2]}], loop=True, simulate_cost_as="claude-opus-5-5")
    ctx = RunContext(TaskRequest(task="t", constraints={"max_cost_usd": 0.0001}), Tracer(), llm, "test")
    r = HybridRetriever(llm=llm)
    r.index(corpus)
    res = await r.search(RetrievalPlan(query="impeller material duplex", query_transforms=["multi_query"], rerank="listwise", rerank_depth=5, k_final=3), ctx=ctx)
    assert res.evidence and ctx.usage.llm_calls == 1 and ctx.budget.remaining_cost() == 0.0
    assert any("cost" in w and "skipped" in w for w in res.warnings)  # the transform that overspent degraded in place
    assert any("cost budget" in w and "stopped" in w for w in res.warnings)  # the rerank window was never bought
    assert not res.reranked and res.evidence[0].retriever == "fused"


# ------------------------------------------------------------- protocol API
async def test_retriever_protocol_direct_api(corpus) -> None:
    r = HybridRetriever(profile="software_engineering")
    r.index(corpus)
    assert isinstance(r, Retriever)
    ev = await r.retrieve("ERR-1203", k=2)
    assert 1 <= len(ev) <= 2 and ev[0].metadata["doc_id"] == "runbook-retry" and ev[0].rank == 1
    assert r.usage.llm_calls == 0
    plan = r.plan_for("Has anything similar to the pump cavitation event happened before?", k=3)
    assert plan.rerank == "none" and all(t == "keywords" for t in plan.query_transforms) and plan.k_final == 3
    assert plan.expected_llm_calls == 0  # estimate recomputed after the direct-API strip
    filtered = r.retrieve_sync("pump", k=3, filters={"type": "incident"})
    assert filtered and all(e.metadata["type"] == "incident" for e in filtered)


async def test_direct_rerank_option_uses_llm(corpus) -> None:
    llm = ScriptedLLM([{"ranking": [2, 1, 3]}], loop=True)
    cfg = HybridRetrievalConfig(direct_rerank=True)
    r = HybridRetriever(cfg, llm=llm, profile=get_profile("mechanical_engineering"))
    r.index(corpus)
    plan = r.plan_for("impeller material duplex", k=3)
    assert plan.rerank == "listwise" and plan.expected_llm_calls == n_windows(plan.rerank_depth, cfg.rerank_window, cfg.rerank_stride)
    ev = await r.retrieve("impeller material duplex", k=3)
    assert ev and ev[0].retriever == "reranked" and r.usage.llm_calls >= 1


async def test_save_load_round_trip(corpus, tmp_path) -> None:
    r = HybridRetriever()
    r.index(corpus)
    path = r.save(tmp_path / "idx.json")
    loaded = HybridRetriever.load(path)
    assert loaded.n_chunks == r.n_chunks and loaded.n_documents == r.n_documents
    a = [e.id for e in await r.retrieve("corrosion allowance casing", k=3)]
    b = [e.id for e in await loaded.retrieve("corrosion allowance casing", k=3)]
    assert a == b and a[0].startswith("spec-pump")
