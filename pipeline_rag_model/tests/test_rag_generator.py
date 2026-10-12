"""Citation parsing, extractive generation and rendering (pure) plus the LLM path."""
from __future__ import annotations

from pipeline_core import Evidence, RunContext, ScriptedLLM, TaskRequest, Tracer

from pipeline_rag import GenerationOutput, GroundedGenerator, PipelineRAGConfig, citation_coverage, render_answer, split_statements
from pipeline_rag.generator import cited_ids, extract_markers, extractive_answer, strip_markers


EVIDENCE = [
    Evidence(id="spec#1", content="The specification requires a corrosion allowance of 3 mm for the pump casing. Casing design pressure is 16 bar.", source="Spec"),
    Evidence(id="motor", content="The drive motor is rated 75 kW at 1480 rpm.", source="Motor"),
]


def test_marker_extraction_and_stripping() -> None:
    text = "Allowance is 3 mm [spec#1]. Motor is 75 kW [motor][spec#1, motor]."
    assert extract_markers(text) == ["spec#1", "motor", "spec#1, motor"]
    assert strip_markers(text) == "Allowance is 3 mm. Motor is 75 kW."


def test_split_statements_resolves_ids_indices_and_case() -> None:
    answer = "The allowance is 3 mm [spec#1]. The motor is rated 75 kW [2]. Unknown claim here [nope]. [MOTOR]"
    sts = split_statements(answer, [e.id for e in EVIDENCE])
    assert [s.text for s in sts] == ["The allowance is 3 mm.", "The motor is rated 75 kW.", "Unknown claim here."]
    assert sts[0].citation_ids == ["spec#1"]
    assert sts[1].citation_ids == ["motor"]
    assert sts[2].citation_ids == ["motor"] and sts[2].unresolved_ids == ["nope"]  # trailing [MOTOR] attached to previous sentence
    assert citation_coverage(sts) == 1.0
    assert citation_coverage([]) == 0.0
    sts2 = split_statements("No citations at all here.", ["spec#1"])
    assert citation_coverage(sts2) == 0.0


def test_cited_ids_union_and_order() -> None:
    out = GenerationOutput(answer="The motor claim [motor].", citations=[{"doc_id": "spec#1", "quote": "q"}, {"doc_id": "motor"}])
    sts = split_statements(out.answer, ["spec#1", "motor"])
    assert cited_ids(out, sts) == ["motor", "spec#1"]


def test_extractive_answer_cites_and_abstains() -> None:
    out = extractive_answer("What corrosion allowance does the specification require for the pump casing?", EVIDENCE)
    assert not out.abstained and "3 mm" in out.answer and "[spec#1]" in out.answer
    assert out.citations[0].doc_id == "spec#1" and "3 mm" in out.citations[0].quote
    assert 0.5 < out.confidence <= 0.95
    miss = extractive_answer("What is the vendor warranty period in years?", EVIDENCE)
    assert miss.abstained and miss.confidence < 0.3
    assert extractive_answer("anything", []).abstained
    lst = extractive_answer("List the casing allowance, the design pressure and the motor rating", EVIDENCE, answer_type="list")
    assert lst.answer.startswith("- ") and lst.answer.count("\n") >= 1


def test_render_answer_styles() -> None:
    out = GenerationOutput(answer="Allowance 3 mm [spec#1]. Motor 75 kW [motor].", citations=[])
    assert render_answer(out, EVIDENCE, "inline") == out.answer
    assert render_answer(out, EVIDENCE, "none") == "Allowance 3 mm. Motor 75 kW."
    foot = render_answer(out, EVIDENCE, "footnote")
    assert foot.startswith("Allowance 3 mm [1]. Motor 75 kW [2].")
    assert "[1] spec#1 (Spec)" in foot and "[2] motor (Motor)" in foot
    abst = GenerationOutput(abstained=True, abstention_reason="no evidence was retrieved")
    assert render_answer(abst, [], "inline") == "Insufficient evidence to answer. No evidence was retrieved."


def test_generation_output_clamps_confidence() -> None:
    assert GenerationOutput(confidence=7).confidence == 1.0
    assert GenerationOutput(confidence=-1).confidence == 0.0


async def test_generator_llm_path_puts_documents_in_user_turn() -> None:
    cfg = PipelineRAGConfig()
    gen = GroundedGenerator(cfg)
    llm = ScriptedLLM([{"answer": "The allowance is 3 mm [spec#1].", "citations": [{"doc_id": "spec#1", "quote": "3 mm"}], "unsupported_claims": [], "confidence": 0.9, "abstained": False, "abstention_reason": ""}])
    ctx = RunContext(TaskRequest(task="q", domain="mechanical_engineering"), Tracer(), llm, "pipeline_rag")
    out = await gen.generate_llm("What corrosion allowance applies?", EVIDENCE, ctx, answer_type="factual")
    assert out.answer.startswith("The allowance is 3 mm") and out.confidence == 0.9
    req = llm.calls[0]
    assert req.output_schema is not None and req.effort == cfg.generation_effort
    user = req.last_user_text()
    assert "<documents>" in user and 'id="spec#1"' in user and "What corrosion allowance applies?" in user
    stable = "\n".join(req.system_stable)
    assert "Domain: mechanical_engineering" in stable and "engineering rigor" in stable.lower()
    assert 'id="spec#1"' not in stable and "3 mm" not in stable and "What corrosion allowance applies?" not in stable  # volatile content never in the cached prefix


async def test_generator_llm_failure_falls_back_to_extractive() -> None:
    gen = GroundedGenerator(PipelineRAGConfig())
    llm = ScriptedLLM([])
    ctx = RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")
    out = await gen.generate_llm("What corrosion allowance does the specification require for the pump casing?", EVIDENCE, ctx)
    assert not out.abstained and "[spec#1]" in out.answer
    assert any("generation failed" in w for w in ctx.warnings)


async def test_generator_no_retrieval_prompt_without_evidence() -> None:
    gen = GroundedGenerator(PipelineRAGConfig())
    llm = ScriptedLLM([{"answer": "Paris.", "citations": [], "unsupported_claims": [], "confidence": 0.95, "abstained": False, "abstention_reason": ""}])
    ctx = RunContext(TaskRequest(task="q"), Tracer(), llm, "pipeline_rag")
    out = await gen.generate_llm("Capital of France?", [], ctx)
    assert out.answer == "Paris." and "pipeline_rag.no_retrieval" in llm.calls[0].tags
