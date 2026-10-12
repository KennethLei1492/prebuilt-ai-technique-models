"""Direct protocol API (answer, index, retrieve, route, plan_for, generate, solver) and the CLI."""
from __future__ import annotations

import json
from pathlib import Path

from pipeline_core import Evidence, MockLLM, Retriever, ScriptedLLM, Status, TaskRequest

from pipeline_rag import PipelineRAGModel
from pipeline_rag.cli import load_documents, main
from pipeline_rag.evaluation import sample_corpus

QUESTION = "What corrosion allowance does the specification require for the pump casing?"


def test_index_retrieve_satisfies_retriever_protocol() -> None:
    model = PipelineRAGModel(MockLLM())
    added = model.index(sample_corpus())
    assert added >= len(sample_corpus())
    assert isinstance(model, Retriever)
    import asyncio

    hits = asyncio.run(model.retrieve("corrosion allowance pump casing", k=3))
    assert hits and hits[0].retriever == "fused" and hits[0].rank == 1
    assert {h.id.split("#")[0] for h in hits} & {"PS-100", "REQ-CW", "FMEA-CW", "MP-14"}
    assert asyncio.run(PipelineRAGModel(MockLLM()).retrieve("anything")) == []
    model.clear_index()
    assert len(model.builtin) == 0


def test_answer_direct_api_uses_persistent_index() -> None:
    model = PipelineRAGModel(MockLLM())
    model.add_documents(sample_corpus())
    result = model.answer_sync(QUESTION, domain="mechanical_engineering")
    assert result.status == Status.SUCCESS and "3 mm" in result.output.answer
    # inline documents are fused with the persistent index
    result2 = model.answer_sync("What is the motor rated power?", documents=[{"id": "m", "text": "The motor is rated 75 kW."}], k=2)
    assert "75 kW" in result2.output.answer and set(result2.output.evidence_ids) <= {"m", "MD-7"}
    assert "inline" in result2.steps[1].details["sources"] and "builtin" in result2.steps[1].details["sources"]
    fresh = PipelineRAGModel(MockLLM()).answer_sync("What is the motor rated power?", documents=[{"id": "m", "text": "The motor is rated 75 kW."}], k=1)
    assert fresh.output.evidence_ids == ["m"] and "[m]" in fresh.output.answer


def test_route_and_plan_for_previews() -> None:
    model = PipelineRAGModel(MockLLM())
    assert model.route(QUESTION).route == "no_retrieval"
    assert model.route(QUESTION, {"documents": sample_corpus()}).route == "single_step"
    model.index(sample_corpus())
    assert model.route("Compare the casing allowance with the impeller material.").route == "multi_step"
    plan = model.plan_for(QUESTION, domain="civil_structural_engineering")
    assert plan.verify and plan.citation_style == "inline" and plan.planner == "heuristic"


async def test_generate_direct_api_heuristic_and_llm() -> None:
    evidence = [Evidence(id="s", content="The specification requires a corrosion allowance of 3 mm for the pump casing.")]
    model = PipelineRAGModel(MockLLM())
    out = await model.generate(QUESTION, evidence)
    assert "[s]" in out.answer and model.direct_usage.llm_calls == 0
    llm = ScriptedLLM([{"answer": "3 mm [s].", "citations": [{"doc_id": "s", "quote": "3 mm"}], "unsupported_claims": [], "confidence": 0.9, "abstained": False, "abstention_reason": ""}])
    model2 = PipelineRAGModel(llm, llm_mode="always")
    out2 = await model2.generate(QUESTION, evidence, domain="mechanical_engineering")
    assert out2.answer == "3 mm [s]." and model2.direct_usage.llm_calls == 1
    assert "Domain: mechanical_engineering" in "\n".join(llm.calls[0].system_stable)


async def test_solver_is_run() -> None:
    model = PipelineRAGModel(MockLLM())
    result = await model.solver(TaskRequest(task=QUESTION, inputs={"documents": sample_corpus()}))
    assert result.component == "pipeline_rag" and result.ok


def test_cli_offline_text_output(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main([QUESTION, "--offline", "--sample-corpus", "--domain", "mechanical_engineering", "--show-evidence"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[pipeline_rag] status=success" in out and "route: single_step" in out and "3 mm" in out and "evidence:" in out


def test_cli_offline_json_and_inputs(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.txt").write_text("The specification requires a corrosion allowance of 3 mm for the pump casing.", encoding="utf-8")
    (docs / "more.json").write_text(json.dumps([{"id": "motor", "text": "Motor rated 75 kW."}, "plain string doc"]), encoding="utf-8")
    loaded = load_documents([str(docs)])
    assert {d["id"] for d in loaded} == {"spec", "motor", "more-2"}
    code = main([QUESTION, "--offline", "--json", "--documents", str(docs), "--input", "k=3", "--input", 'note="hello"'])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["status"] == "success" and payload["metadata"]["plan"]["k"] == 3
    assert "[spec]" in payload["output"]["answer"]


def test_cli_no_documents_abstains_but_exits_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main([QUESTION, "--offline"])
    out = capsys.readouterr().out
    assert code == 0 and "status=needs_review" in out and "route: no_retrieval" in out


def test_cli_evaluate_flag_and_evaluation_main(capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    code = main([QUESTION, "--offline", "--sample-corpus", "--evaluate"])
    out = capsys.readouterr().out
    assert code == 0 and "evaluation: n=6 judge=heuristic" in out and "routes:" in out
    from pipeline_rag.evaluation import main as eval_main

    assert eval_main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("n=6 judge=heuristic") and "abstained" in out
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spec.txt").write_text("The casing corrosion allowance is 3 mm for seawater service. Design pressure is 16 bar at 80 degC.", encoding="utf-8")
    assert eval_main(["--offline", "--json", "--documents", str(docs), "--n", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] >= 1 and payload["judge"] == "heuristic"
