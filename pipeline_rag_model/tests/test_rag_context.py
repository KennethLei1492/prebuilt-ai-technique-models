"""CRAG grading, extractive compression, ordering and dedupe (pure)."""
from __future__ import annotations

from pipeline_core import Evidence, RunContext, ScriptedLLM, TaskRequest, Tracer, estimate_tokens

from pipeline_rag import ContextCompressor, ContextGrader, ContextOrderer, PipelineRAGConfig, crag_decision, deduplicate, extractive_compress, order_for_position
from pipeline_rag.context import label_for_score, lexical_relevance, looks_like_code_or_table, split_sentences


def ev(i: str, text: str, score: float = 0.0, **meta: object) -> Evidence:
    return Evidence(id=i, content=text, score=score, metadata=dict(meta))


def test_crag_decision_three_way() -> None:
    assert crag_decision([0.9, -1.0], 0.5, -0.9) == "correct"
    assert crag_decision([-0.95, -1.0], 0.5, -0.9) == "incorrect"
    assert crag_decision([0.1, -0.95], 0.5, -0.9) == "ambiguous"
    assert crag_decision([], 0.5, -0.9) == "incorrect"
    assert label_for_score(0.5, 0.5, -0.9) == "correct"
    assert label_for_score(-0.9, 0.5, -0.9) == "incorrect"
    assert label_for_score(0.0, 0.5, -0.9) == "ambiguous"


def test_lexical_relevance_weights_identifiers() -> None:
    q = "corrosion allowance for casing REQ-020"
    hit = lexical_relevance(q, "REQ-020 requires a corrosion allowance on the casing")
    miss = lexical_relevance(q, "the motor is rated 75 kW")
    assert hit > 0.9 and miss == -1.0
    partial = lexical_relevance(q, "corrosion allowance casing")  # identifier (double weight) missing
    assert -1.0 < partial < 0.5
    assert lexical_relevance("", "anything") == 0.0


async def test_grader_heuristic_filters_and_annotates() -> None:
    cfg = PipelineRAGConfig()
    grader = ContextGrader(cfg)
    evidence = [
        ev("a", "The specification requires a corrosion allowance of 3 mm for the pump casing."),
        ev("b", "The motor is rated 75 kW at 1480 rpm with service factor 1.15."),
        ev("c", "Casing corrosion is monitored by an annual thickness survey."),
    ]
    kept, decision, grades = await grader.grade("What corrosion allowance does the specification require for the pump casing?", evidence)
    assert decision == "correct"
    ids = [e.id for e in kept]
    assert "a" in ids and "b" not in ids  # unrelated strip dropped (knowledge-strip refinement)
    assert kept[0].metadata["grade"] == "correct" and isinstance(kept[0].metadata["grade_score"], float)
    assert len(grades) == 3
    kept2, decision2, _ = await grader.grade("What is the warranty period?", [evidence[1]])
    assert decision2 == "incorrect" and kept2 == []
    assert (await grader.grade("anything", []))[1] == "incorrect"


async def test_grader_llm_path_and_fallback() -> None:
    cfg = PipelineRAGConfig()
    grader = ContextGrader(cfg)
    evidence = [ev("a", "alpha text"), ev("b", "beta text")]
    llm = ScriptedLLM([{"grades": [{"doc_id": "a", "label": "correct", "score": 0.9, "reason": "answers"}, {"doc_id": "zzz", "label": "correct", "score": 1.0, "reason": "unknown id"}]}])
    ctx = RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")
    kept, decision, grades = await grader.grade("alpha?", evidence, ctx=ctx, use_llm=True)
    assert decision == "correct"
    by_id = {g.doc_id: g for g in grades}
    assert by_id["a"].score == 0.9 and "zzz" not in by_id
    assert by_id["b"].reason == "lexical fallback"  # skipped by the model -> graded lexically
    assert llm.calls[0].effort == cfg.grading_effort
    failing = ScriptedLLM([])
    ctx2 = RunContext(TaskRequest(task="q"), Tracer(), failing, "pipeline_rag")
    _, _, grades2 = await grader.grade("alpha?", evidence, ctx=ctx2, use_llm=True)
    assert all(g.reason == "lexical overlap" for g in grades2)
    assert any("grading failed" in w for w in ctx2.warnings)


def test_extractive_compress_keeps_relevant_sentences_and_spans() -> None:
    text = "The pump is blue. The casing corrosion allowance is 3 mm. The vendor is in Oslo. Design pressure is 16 bar."
    e = Evidence(id="a", content=text, span=(0, len(text)))
    out = extractive_compress([e], "casing corrosion allowance design pressure", token_budget=1000, sentences_per_unit=2)
    assert len(out) == 1
    assert "3 mm" in out[0].content and "16 bar" in out[0].content and "Oslo" not in out[0].content
    assert out[0].metadata["compressed"] is True and out[0].metadata["original_chars"] == len(text)
    assert out[0].span == (0, len(text))  # pointer back to the source span is kept
    # order of kept sentences follows the source order
    assert out[0].content.index("3 mm") < out[0].content.index("16 bar")


def test_extractive_compress_protects_code_and_budget() -> None:
    code = "def f(x):\n    return x * 2\n\nclass A:\n    pass\n"
    assert looks_like_code_or_table(code)
    table = "| part | qty |\n| A | 1 |\n| B | 2 |"
    assert looks_like_code_or_table(table)
    assert not looks_like_code_or_table("Plain prose about pumps. Another sentence here.")
    e_code = Evidence(id="c", content=code)
    out = extractive_compress([e_code], "return x", token_budget=1000, sentences_per_unit=1)
    assert out[0].content == code and "compressed" not in out[0].metadata
    prose = " ".join(f"Sentence number {i} about casing corrosion allowance." for i in range(40))
    many = [Evidence(id=f"p{i}", content=prose) for i in range(5)]
    out2 = extractive_compress(many, "casing corrosion", token_budget=60, sentences_per_unit=2)
    assert sum(estimate_tokens(o.content) for o in out2) <= 60
    assert len(out2) < len(many)


def test_order_for_position_puts_best_first_second_last() -> None:
    items = [ev("a", "x", grade_score=0.1), ev("b", "x", grade_score=0.9), ev("c", "x", grade_score=0.5), ev("d", "x", grade_score=0.7)]
    ordered = [e.id for e in order_for_position(items)]
    assert ordered[0] == "b" and ordered[-1] == "d"
    assert ordered == ["b", "c", "a", "d"]
    assert [e.id for e in order_for_position(items[:2])] == ["b", "a"]
    assert order_for_position([]) == []
    assert [e.id for e in ContextOrderer().order(items)] == ordered


def test_deduplicate_by_id_and_text() -> None:
    items = [ev("a", "Same text here."), ev("b", "same   text HERE."), ev("a", "other"), ev("c", "Different.")]
    out = deduplicate(items)
    assert [e.id for e in out] == ["a", "c"]


def test_split_sentences_keeps_list_lines() -> None:
    s = split_sentences("First sentence. Second one!\n- bullet a\n- bullet b")
    assert s == ["First sentence.", "Second one!", "- bullet a", "- bullet b"]


def test_compressor_uses_config_quota() -> None:
    cfg = PipelineRAGConfig(sentences_per_chunk_single_hop=1, sentences_per_chunk_multi_hop=3, context_token_budget=500)
    comp = ContextCompressor(cfg)
    text = "Alpha casing fact one. Beta casing fact two. Gamma casing fact three. Delta unrelated."
    single = comp.compress([ev("a", text)], "casing fact")
    multi = comp.compress([ev("a", text)], "casing fact", multi_hop=True)
    assert len(split_sentences(single[0].content)) == 1
    assert len(split_sentences(multi[0].content)) == 3
