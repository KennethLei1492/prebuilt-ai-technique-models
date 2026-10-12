"""Rerankers and query transforms: pure helpers plus LLM paths driven by ScriptedLLM."""
from __future__ import annotations

import re

import pytest

from pipeline_core import Evidence, ScriptedLLM, TaskRequest, Tracer
from pipeline_core.component import RunContext
from pipeline_core.errors import ConfigurationError

from hybrid_retrieval.query_transform import HyDE, KeywordExtractor, MultiQueryExpander, Query2Doc, content_overlap
from hybrid_retrieval.rerank import (
    CrossEncoderReranker,
    LLMListwiseReranker,
    LLMPointwiseReranker,
    NoopReranker,
    apply_window_permutation,
    make_reranker,
    repair_permutation,
    sliding_windows,
)

CANDS = [
    Evidence(id="c1", content="Schedule memo about commissioning", score=0.9, rank=1, retriever="fused"),
    Evidence(id="c2", content="The impeller shall be duplex stainless steel ASTM A890 Grade 4A", score=0.8, rank=2, retriever="fused"),
    Evidence(id="c3", content="Mechanical seal per API 682", score=0.7, rank=3, retriever="fused"),
]

# ISO timestamps and UUID-like fragments must never appear in cacheable stable blocks
VOLATILE_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}", re.IGNORECASE)


def _ctx(llm, **constraints) -> RunContext:
    return RunContext(TaskRequest(task="t", constraints=constraints), Tracer(), llm, "test")


def _unchanged(out: list[Evidence], cands: list[Evidence]) -> bool:
    return [(e.id, e.score, e.rank, e.retriever) for e in out] == [(e.id, e.score, e.rank, e.retriever) for e in cands]


# ------------------------------------------------------------ pure helpers
def test_sliding_windows_bottom_up() -> None:
    assert sliding_windows(10, 20, 10) == [(0, 10)]
    w = sliding_windows(100, 20, 10)
    assert len(w) == 9 and w[0] == (80, 100) and w[-1] == (0, 20)
    assert sliding_windows(0, 20, 10) == []
    with pytest.raises(ValueError):
        sliding_windows(5, 0, 1)


def test_repair_permutation() -> None:
    assert repair_permutation([3, 1, 2], 3) == [2, 0, 1]
    assert repair_permutation([3, 3, 9, "x", 1], 3) == [2, 0, 1]
    assert repair_permutation([], 3) == [0, 1, 2]
    assert repair_permutation([2], 3) == [1, 0, 2]


def test_apply_window_permutation() -> None:
    assert apply_window_permutation([0, 1, 2, 3, 4], 1, 4, [2, 0, 1]) == [0, 3, 1, 2, 4]
    assert apply_window_permutation([0, 1, 2], 0, 3, []) == [0, 1, 2]


# -------------------------------------------------------------- rerankers
async def test_listwise_reranker_reorders_with_scripted_llm() -> None:
    llm = ScriptedLLM([{"ranking": [2, 3, 1]}])
    rr = LLMListwiseReranker(llm, window=20, stride=10)
    out = await rr.rerank("impeller material", CANDS)
    assert [e.id for e in out] == ["c2", "c3", "c1"]
    assert out[0].retriever == "reranked" and out[0].rank == 1 and out[0].score > out[1].score > out[2].score
    assert out[0].metadata["fused_rank"] == 2 and out[0].metadata["rerank_method"] == "listwise"
    assert rr.usage.llm_calls == 1 and rr.applied is True
    req = llm.calls[0]
    assert req.output_schema is not None and "<passage" in req.last_user_text() and req.effort == "low"
    assert not any(VOLATILE_RE.search(s) for s in req.system_stable)  # no timestamps/uuids in stable blocks


async def test_listwise_reranker_tolerates_bad_output_and_budgets_through_ctx() -> None:
    llm = ScriptedLLM(["not json at all"])
    ctx = _ctx(llm)
    rr = LLMListwiseReranker(llm)
    out = await rr.rerank("q", CANDS, k=2, ctx=ctx)
    assert _unchanged(out, CANDS[:2]) and rr.applied is False  # fused order kept, truncated to k
    assert any("failed" in w for w in ctx.warnings)
    single = await LLMListwiseReranker(llm).rerank("q", CANDS[:1])
    assert _unchanged(single, CANDS[:1]) and len(llm.calls) == 1  # no call for a single candidate


async def test_failed_windows_keep_fused_order_without_claiming_a_rerank() -> None:
    """Regression (major): all windows failing or empty must not relabel evidence as reranked or rewrite scores."""
    for responses in (["garbage"], [{"ranking": []}]):
        llm = ScriptedLLM(responses, loop=True)
        rr = LLMListwiseReranker(llm)
        out = await rr.rerank("impeller material", CANDS)
        assert _unchanged(out, CANDS) and rr.applied is False and rr.windows_applied == 0
        assert all(e.retriever == "fused" for e in out) and [e.score for e in out] == [0.9, 0.8, 0.7]
        assert "rerank_score" not in out[0].metadata and rr.warnings


async def test_listwise_sliding_windows_make_one_call_per_window() -> None:
    cands = [Evidence(id=f"c{i}", content=f"passage {i}", score=1.0 - i / 100, rank=i + 1) for i in range(30)]
    llm = ScriptedLLM([{"ranking": []}], loop=True)
    ctx = _ctx(llm)
    rr = LLMListwiseReranker(llm, window=20, stride=10)
    out = await rr.rerank("q", cands, ctx=ctx)
    assert ctx.usage.llm_calls == 2 and rr.applied is False and [e.id for e in out] == [e.id for e in cands]


async def test_listwise_budget_exhaustion_mid_way_keeps_applied_windows() -> None:
    cands = [Evidence(id=f"c{i}", content=f"passage {i}", score=1.0 - i / 100, rank=i + 1) for i in range(30)]
    llm = ScriptedLLM([{"ranking": list(range(20, 0, -1))}], loop=True)  # reverse each window
    ctx = _ctx(llm, max_llm_calls=1)
    rr = LLMListwiseReranker(llm, window=20, stride=10)
    out = await rr.rerank("q", cands, ctx=ctx)
    assert ctx.usage.llm_calls == 1 and rr.applied is True and rr.windows_applied == 1
    assert [e.id for e in out[:10]] == [f"c{i}" for i in range(10)]  # the second (top) window was never reached
    assert [e.id for e in out[10:]] == [f"c{i}" for i in range(29, 9, -1)]  # bottom window reversed
    assert any("stopped" in w and "budget" in w for w in ctx.warnings)


async def test_pointwise_reranker_orders_by_grade() -> None:
    llm = ScriptedLLM([{"score": 2, "rationale": "off topic"}, {"score": 9, "rationale": "exact"}, {"score": 5, "rationale": "partial"}])
    ctx = _ctx(llm)
    rr = LLMPointwiseReranker(llm, concurrency=1)
    out = await rr.rerank("impeller material", CANDS, ctx=ctx)
    assert [e.id for e in out] == ["c2", "c3", "c1"] and rr.applied is True
    assert out[0].score == 0.9 and out[0].metadata["rerank_rationale"] == "exact" and out[0].metadata["graded"] is True
    assert ctx.usage.llm_calls == 3


async def test_pointwise_all_grades_failing_is_not_applied() -> None:
    llm = ScriptedLLM(["garbage"], loop=True)
    rr = LLMPointwiseReranker(llm, concurrency=1)
    out = await rr.rerank("q", CANDS)
    assert _unchanged(out, CANDS) and rr.applied is False and len(rr.warnings) == 3


async def test_pointwise_reranker_respects_call_budget() -> None:
    llm = ScriptedLLM([{"score": 9, "rationale": "exact"}])
    ctx = RunContext(TaskRequest(task="t", constraints={"max_llm_calls": 1}), Tracer(), llm, "test")
    out = await LLMPointwiseReranker(llm).rerank("q", CANDS, ctx=ctx)
    assert len(out) == 3 and out[0].id == "c1" and out[0].metadata["graded"] is True
    assert out[1].metadata["graded"] is False and any("limited" in w for w in ctx.warnings)


async def test_pointwise_skips_fan_out_when_budget_is_already_exhausted() -> None:
    llm = ScriptedLLM([{"score": 9, "rationale": "exact"}], loop=True)
    ctx = _ctx(llm, max_llm_calls=0)
    rr = LLMPointwiseReranker(llm)
    out = await rr.rerank("q", CANDS, ctx=ctx)
    assert _unchanged(out, CANDS) and rr.applied is False and ctx.usage.llm_calls == 0
    assert any("skipped" in w for w in ctx.warnings)


class _FakeExecutor:
    """Minimal ``pipeline_core.protocols.Executor``: sequential map that records its use."""

    def __init__(self) -> None:
        self.maps = 0
        self.concurrency = None

    async def map(self, fn, items, *, concurrency=None):
        self.maps += 1
        self.concurrency = concurrency
        return [await fn(i) for i in items]


async def test_pointwise_fan_out_goes_through_the_executor() -> None:
    llm = ScriptedLLM([{"score": 1, "rationale": "a"}, {"score": 8, "rationale": "b"}, {"score": 4, "rationale": "c"}])
    ex = _FakeExecutor()
    rr = LLMPointwiseReranker(llm, concurrency=3, executor=ex)
    out = await rr.rerank("q", CANDS)
    assert ex.maps == 1 and ex.concurrency == 3 and [e.id for e in out] == ["c2", "c3", "c1"]
    assert rr.applied is True and rr.usage.llm_calls == 3


class _FakeCrossEncoder:
    def predict(self, pairs):
        return [1.0 if "impeller" in passage else (0.5 if "seal" in passage else 0.1) for _q, passage in pairs]


async def test_cross_encoder_adapter_with_fake_model() -> None:
    rr = make_reranker("cross_encoder", cross_encoder=_FakeCrossEncoder())
    assert isinstance(rr, CrossEncoderReranker)
    out = await rr.rerank("impeller", CANDS, k=2)
    assert [e.id for e in out] == ["c2", "c3"] and out[0].retriever == "reranked" and out[0].score == 1.0
    assert out[0].metadata["rerank_method"] == "cross_encoder" and out[0].metadata["fused_rank"] == 2 and rr.applied is True
    assert await CrossEncoderReranker(_FakeCrossEncoder()).rerank("q", []) == []


async def test_noop_and_factory() -> None:
    out = await NoopReranker().rerank("q", CANDS, k=2)
    assert [e.id for e in out] == ["c1", "c2"]
    assert isinstance(make_reranker("none"), NoopReranker)
    assert isinstance(make_reranker("listwise", None), LLMListwiseReranker)
    assert isinstance(make_reranker("pointwise", None), LLMPointwiseReranker)
    with pytest.raises(ConfigurationError):
        await LLMListwiseReranker(None).rerank("q", CANDS)


# --------------------------------------------------------- query transforms
def test_content_overlap_and_compose_and_keywords() -> None:
    assert content_overlap("pump impeller material", "the impeller is made of duplex material") == pytest.approx(2 / 3)
    assert content_overlap("the of", "anything") == 1.0
    assert Query2Doc.compose("q", "doc", repeat=3) == "q q q doc"
    terms = KeywordExtractor().extract_heuristic("What is the root cause of ERR-1203 in the retry runbook?")
    assert terms[:3] == ["root", "cause", "err-1203"] and "the" not in terms and "what" not in terms


async def test_multi_query_drift_guard_and_dedupe() -> None:
    llm = ScriptedLLM([{"queries": ["impeller material for the pump", "Quarterly budget review", "pump impeller material", "PUMP impeller alloy"]}])
    ctx = _ctx(llm)
    out = await MultiQueryExpander(llm, n=3, min_overlap=0.5).expand("pump impeller material", ctx)
    assert out == ["impeller material for the pump", "PUMP impeller alloy"]  # drift dropped, original deduped
    assert ctx.usage.llm_calls == 1 and any("drifted" in w for w in ctx.warnings)


async def test_multi_query_failure_returns_empty_and_warns() -> None:
    llm = ScriptedLLM(["garbage"])
    ctx = _ctx(llm)
    assert await MultiQueryExpander(llm).expand("pump impeller material", ctx) == []
    assert any("failed" in w for w in ctx.warnings)


async def test_transforms_skip_when_budget_is_exhausted() -> None:
    llm = ScriptedLLM([{"queries": ["x"]}], loop=True)
    ctx = _ctx(llm, max_llm_calls=0)
    assert await MultiQueryExpander(llm).expand("pump impeller material", ctx) == []
    assert await HyDE(llm).generate("pump impeller material", ctx) == []
    assert await Query2Doc(llm).expand("pump impeller material", ctx) == "pump impeller material"
    assert ctx.usage.llm_calls == 0 and len(ctx.warnings) == 3 and all("budget" in w for w in ctx.warnings)


async def test_hyde_and_query2doc_with_scripted_llm() -> None:
    llm = ScriptedLLM([{"passages": ["The pump impeller material is duplex stainless steel.", "Unrelated passage about schedules."]}])
    hy = HyDE(llm, n=4, min_overlap=0.5)
    passages = await hy.generate("pump impeller material")
    assert passages == ["The pump impeller material is duplex stainless steel."] and hy.usage.llm_calls == 1

    q2d = Query2Doc(ScriptedLLM([{"pseudo_document": "ERR-1203 is raised when the heartbeat is lost."}]), repeat=2)
    expanded = await q2d.expand("ERR-1203 cause")
    assert expanded.startswith("ERR-1203 cause ERR-1203 cause ERR-1203 is raised")
    drift = Query2Doc(ScriptedLLM([{"pseudo_document": "Totally unrelated text."}]), min_overlap=0.5)
    assert await drift.expand("ERR-1203 cause") == "ERR-1203 cause"
    with pytest.raises(ConfigurationError):
        await HyDE(None).generate("q")


async def test_keyword_extractor_llm_path_falls_back_to_heuristic() -> None:
    llm = ScriptedLLM([{"keywords": ["ERR-1203", "retry", ""]}, "broken"])
    kx = KeywordExtractor(llm, use_llm=True)
    assert await kx.extract("ERR-1203 retry cause") == ["ERR-1203", "retry"]
    assert await kx.extract("ERR-1203 retry cause") == ["err-1203", "retry", "cause"]
