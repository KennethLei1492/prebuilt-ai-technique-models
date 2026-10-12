"""Multi-hop loop, Self-RAG critique maths and the support judge."""
from __future__ import annotations

import pytest

from pipeline_core import Evidence, RunContext, ScriptedLLM, TaskRequest, Tracer

from pipeline_rag import MultiHopLoop, PipelineRAGConfig, ReflectionJudge, critique_score, plan_hop_queries
from pipeline_rag.generator import split_statements
from pipeline_rag.iterate import flare_trigger, heuristic_support, merge_evidence, overall_support, should_retrieve, support_fraction

CORPUS = {
    "inc": Evidence(id="inc", content="Pump P-102 tripped on high vibration. Root cause: misalignment of 0.35 mm offset."),
    "proc": Evidence(id="proc", content="Procedure MP-14 requires a hot alignment check within 4 hours of first run."),
    "motor": Evidence(id="motor", content="The drive motor is rated 75 kW at 1480 rpm."),
}


def make_retrieve(log: list[str]):  # type: ignore[no-untyped-def]
    async def retrieve(query: str, k: int) -> list[Evidence]:
        log.append(query)
        q = query.lower()
        hits = []
        if "alignment" in q or "procedure" in q or "principle" in q:
            hits.append(CORPUS["proc"])
        if "vibration" in q or "root cause" in q or "trip" in q:
            hits.append(CORPUS["inc"])
        if "motor" in q:
            hits.append(CORPUS["motor"])
        return hits[:k]

    return retrieve


def test_critique_score_weights() -> None:
    w = PipelineRAGConfig().reflection_weights()
    assert w == {"is_relevant": 1.0, "is_supported": 1.0, "is_useful": 0.5}
    assert critique_score(is_relevant=True, is_supported="fully", is_useful=5, weights=w) == pytest.approx(1.0)
    assert critique_score(is_relevant=False, is_supported="no", is_useful=1, weights=w) == pytest.approx(0.0)
    mid = critique_score(is_relevant=True, is_supported="partially", is_useful=3, weights=w)
    assert mid == pytest.approx((1.0 + 0.5 + 0.5 * 0.5) / 2.5)
    heavy = critique_score(is_relevant=True, is_supported="no", is_useful=5, weights={"is_relevant": 1.0, "is_supported": 3.0, "is_useful": 0.5})
    assert heavy < mid  # raising the support weight punishes unsupported answers
    assert critique_score(is_relevant=True, is_supported="fully", is_useful=5, weights={"is_relevant": 0, "is_supported": 0, "is_useful": 0}) == 0.0


def test_support_maths_and_triggers() -> None:
    assert support_fraction(["fully", "partially", "no"]) == pytest.approx(0.5)
    assert support_fraction([]) == 0.0
    assert overall_support(["fully", "fully"]) == "fully"
    assert overall_support(["no", "no"]) == "no"
    assert overall_support(["fully", "no"]) == "partially"
    assert flare_trigger(0.5, 0.6) and not flare_trigger(0.7, 0.6)
    assert should_retrieve(0.3, 0.2) and not should_retrieve(0.1, 0.2) and should_retrieve(0.01, 0.0)


def test_plan_hop_queries_schedule() -> None:
    qs = plan_hop_queries("What was the root cause of the vibration trip and which procedure was updated?")
    assert len(qs) >= 2
    assert qs[-1].startswith("general principle")
    assert all(q.lower() != "what was the root cause of the vibration trip and which procedure was updated?" for q in qs)
    assert plan_hop_queries("Short question?", include_step_back=False) == []


def test_merge_evidence_dedupes() -> None:
    merged, added = merge_evidence([CORPUS["inc"]], [CORPUS["inc"], CORPUS["proc"]])
    assert [e.id for e in merged] == ["inc", "proc"] and added == 1


async def test_loop_offline_follows_schedule_and_stops() -> None:
    cfg = PipelineRAGConfig(k_per_hop=3)
    loop = MultiHopLoop(cfg)
    log: list[str] = []
    q = "What was the root cause of the vibration trip and which alignment procedure was updated?"
    evidence, hops, reasoning = await loop.run(q, [CORPUS["inc"]], make_retrieve(log), max_hops=4)
    assert hops[0].triggered_by == "initial" and hops[0].hop == 1
    assert len(hops) >= 2 and all(h.triggered_by == "schedule" for h in hops[1:])
    assert any(e.id == "proc" for e in evidence)
    assert hops[-1].stop_reason in ("schedule exhausted", "no further queries", "hop cap reached", "query repeated")
    assert reasoning == []
    assert len(log) == len(hops) - 1


async def test_loop_respects_hop_cap() -> None:
    loop = MultiHopLoop(PipelineRAGConfig())
    log: list[str] = []
    _, hops, _ = await loop.run("Compare the trip root cause with the alignment procedure and the motor rating.", [], make_retrieve(log), max_hops=2)
    assert len(hops) <= 2 and len(log) <= 1


async def test_loop_llm_path_uses_next_query_then_stops() -> None:
    cfg = PipelineRAGConfig()
    loop = MultiHopLoop(cfg)
    llm = ScriptedLLM(
        [
            {"reasoning_step": "The trip was caused by misalignment.", "is_relevant": True, "need_more_information": True, "next_query": "MP-14 alignment procedure update", "can_answer": False, "confidence": 0.8},
            {"reasoning_step": "MP-14 now requires a hot alignment check.", "is_relevant": True, "need_more_information": False, "next_query": "", "can_answer": True, "confidence": 0.9},
        ]
    )
    ctx = RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")
    log: list[str] = []
    evidence, hops, reasoning = await loop.run("Root cause and procedure?", [CORPUS["inc"]], make_retrieve(log), max_hops=4, ctx=ctx, use_llm=True)
    assert log == ["MP-14 alignment procedure update"]
    assert hops[1].triggered_by == "need_more_information" and hops[1].stop_reason == "model can answer"
    assert reasoning == ["The trip was caused by misalignment.", "MP-14 now requires a hot alignment check."]
    assert any(e.id == "proc" for e in evidence)
    assert len(llm.calls) == 2 and "pipeline_rag.hop" in llm.calls[0].tags


async def test_loop_flare_low_confidence_triggers_retrieval() -> None:
    cfg = PipelineRAGConfig(flare_confidence_threshold=0.6)
    loop = MultiHopLoop(cfg)
    llm = ScriptedLLM([{"reasoning_step": "The motor rating matters for the trip analysis.", "is_relevant": True, "need_more_information": False, "next_query": "", "can_answer": False, "confidence": 0.3}])
    ctx = RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")
    log: list[str] = []
    _, hops, _ = await loop.run("Why did it trip?", [CORPUS["inc"]], make_retrieve(log), max_hops=2, ctx=ctx, use_llm=True)
    assert hops[1].triggered_by == "low_confidence" and "motor" in log[0]


def test_heuristic_support_levels() -> None:
    sts = split_statements("The trip root cause was misalignment of 0.35 mm offset [inc]. The vendor offers a 5 year warranty [inc].", ["inc", "proc"])
    j = heuristic_support(sts, [CORPUS["inc"], CORPUS["proc"]])
    assert j.statements[0].support == "fully" and j.statements[0].evidence_ids == ["inc"]
    assert j.statements[1].support == "no"
    assert j.is_relevant and 1 <= j.is_useful <= 5


async def test_reflection_judge_heuristic_and_llm() -> None:
    cfg = PipelineRAGConfig()
    judge = ReflectionJudge(cfg)
    sts = split_statements("The trip root cause was misalignment of 0.35 mm offset [inc].", ["inc"])
    flags = judge.judge_heuristic(sts, [CORPUS["inc"]])
    assert flags.is_supported == "fully" and flags.method == "heuristic" and flags.critique_score > 0.8
    llm = ScriptedLLM([{"statements": [{"statement": "The trip root cause was misalignment of 0.35 mm offset.", "support": "partially", "evidence_ids": ["inc", "ghost"]}], "is_relevant": True, "is_useful": 4}])
    ctx = RunContext(TaskRequest(task="q", domain="civil_structural_engineering"), Tracer(), llm, "pipeline_rag")
    flags2 = await judge.judge_llm("q", "answer", sts, [CORPUS["inc"]], ctx, effort="max")
    assert flags2.method == "llm" and flags2.is_supported == "partially" and flags2.support_fraction == 0.5
    assert llm.calls[0].effort == "max"
    failing = ScriptedLLM([])
    ctx2 = RunContext(TaskRequest(task="q"), Tracer(), failing, "pipeline_rag")
    flags3 = await judge.judge_llm("q", "answer", sts, [CORPUS["inc"]], ctx2)
    assert flags3.method == "heuristic" and ctx2.warnings
