"""Metrics, the evaluation harness, the synthetic corpus and the CLI."""
from __future__ import annotations

import json
import math

import pytest

from pipeline_core import Evidence

from hybrid_retrieval import HybridRetrievalModel, HybridRetriever
from hybrid_retrieval.cli import load_documents, main, parse_input
from hybrid_retrieval.evaluation import (
    QueryJudgment,
    ablation,
    aggregate,
    collapse_to_documents,
    evaluate,
    evaluate_sync,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_ranking,
    success_at_k,
    synthetic_engineering_corpus,
)

QRELS = {"a": 3.0, "b": 0.0, "c": 1.0}


def test_metric_values() -> None:
    ranked = ["a", "b", "c", "d"]
    expected = (3.0 + 1.0 / math.log2(4)) / (3.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(ranked, QRELS, 10) == pytest.approx(expected)
    assert ndcg_at_k(["c", "a"], QRELS, 1) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, {"z": 0.0}, 10) == 0.0 and ndcg_at_k(ranked, QRELS, 0) == 0.0
    assert recall_at_k(ranked, QRELS, 1) == 0.5 and recall_at_k(ranked, QRELS, 3) == 1.0
    assert precision_at_k(ranked, QRELS, 4) == 0.5 and precision_at_k([], QRELS, 4) == 0.0
    assert mrr(["b", "d", "c"], QRELS) == pytest.approx(1 / 3) and mrr(["b"], QRELS) == 0.0 and mrr(["b", "a"], QRELS, k=1) == 0.0
    assert mrr(["a", "b"], QRELS, k=0) == 0.0 and mrr(["a", "b"], QRELS, k=None) == mrr(["a", "b"], QRELS)  # k=0 is a cutoff, not "no cutoff"
    assert success_at_k(["b", "c"], QRELS, 2) == 1.0 and success_at_k(["b"], QRELS, 1) == 0.0


def test_collapse_and_aggregate() -> None:
    ev = [Evidence(id="d1#2", content="x", metadata={"doc_id": "d1"}), Evidence(id="d1#0", content="y", metadata={"doc_id": "d1"}), Evidence(id="d2#0", content="z")]
    assert collapse_to_documents(ev) == ["d1", "d2"]
    j = QueryJudgment(query_id="q", query="q", qrels={"d1": 3.0}, kind="identifier")
    row = score_ranking(j, ["d1", "d2"], k=5, latency_s=0.01)
    assert row.ndcg == 1.0 and row.recall == 1.0 and row.mrr == 1.0 and row.precision == 0.2
    rep = aggregate([row, score_ranking(QueryJudgment(query_id="q2", query="q2", qrels={"d9": 1.0}, kind="natural_language"), ["d1"], k=5)], k=5)
    assert rep.n_queries == 2 and rep.ndcg_at_k == 0.5 and rep.recall_at_k == 0.5 and set(rep.by_kind) == {"identifier", "natural_language"}
    assert "nDCG@5=0.500" in rep.summary() and rep.worst(1)[0].query_id == "q2"


def test_synthetic_corpus_is_deterministic_and_judged() -> None:
    docs, judgments = synthetic_engineering_corpus(n_docs=16, seed=3)
    docs2, judgments2 = synthetic_engineering_corpus(n_docs=16, seed=3)
    assert docs == docs2 and [j.model_dump() for j in judgments] == [j.model_dump() for j in judgments2]
    assert len(docs) == 16 and len({d["id"] for d in docs}) == 16
    ids = {d["id"] for d in docs}
    assert all(set(j.qrels) <= ids for j in judgments) and {d["metadata"]["type"] for d in docs} == {"specification", "incident", "runbook", "datasheet"}
    assert {j.kind for j in judgments} >= {"identifier", "natural_language", "exploratory"}


async def test_evaluate_and_ablation_on_synthetic_corpus() -> None:
    docs, judgments = synthetic_engineering_corpus(n_docs=24)
    model = HybridRetrievalModel()
    model.index(docs)
    report = await evaluate(model, judgments, k=10)
    assert report.n_queries == len(judgments) and report.recall_at_k >= 0.8 and report.ndcg_at_k >= 0.7 and report.success_at_k >= 0.8
    assert report.usage.llm_calls == 0
    reports = await ablation(model, judgments[:10], k=10)
    assert set(reports) == {"sparse", "dense", "hybrid"} and reports["hybrid"].recall_at_k >= reports["dense"].recall_at_k - 1e-9
    mapping = evaluate_sync(model.retriever, {judgments[0].query: judgments[0].qrels}, k=5)
    assert mapping.n_queries == 1 and mapping.per_query[0].query_id == "q1"
    chunk_level = await evaluate(HybridRetriever(), judgments[:1], k=5, document_level=False)
    assert chunk_level.recall_at_k == 0.0  # empty retriever


def test_cli_helpers(tmp_path) -> None:
    assert parse_input("k=5") == ("k", 5) and parse_input('filters={"type":"spec"}') == ("filters", {"type": "spec"}) and parse_input("query=ERR-1203 cause") == ("query", "ERR-1203 cause")
    with pytest.raises(Exception):
        parse_input("novalue")
    (tmp_path / "a.md").write_text("# Alpha\nERR-9001 means the pump tripped.", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps([{"id": "j1", "text": "json doc"}, "bare string"]), encoding="utf-8")
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "l1", "text": "line doc"}) + "\n", encoding="utf-8")
    docs = load_documents([str(tmp_path)])
    assert {d["id"] for d in docs} == {"a", "j1", "b-2", "l1"} and docs[0]["metadata"]["type"] == "md"


def test_cli_offline_text_and_json(capsys, tmp_path) -> None:
    code = main(["Find the clauses that specify the pump impeller material and its corrosion allowance", "--offline", "--k", "3"])
    out = capsys.readouterr().out
    assert code == 0 and "Hybrid retrieval for" in out and "synthetic engineering corpus" in out and "Plan:" in out
    (tmp_path / "doc.md").write_text("ERR-7777 is raised when the valve positioner loses feedback.", encoding="utf-8")
    code = main(["ERR-7777", "--offline", "--json", "--domain", "software_engineering", "--documents", str(tmp_path), "--input", "k=1", "--save-index", str(tmp_path / "idx.json")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["status"] == "success" and payload["evidence"][0]["metadata"]["doc_id"] == "doc"
    assert payload["metadata"]["plan"]["query_type"] == "identifier" and "indexed 1 documents" in payload["note"]
    (tmp_path / "doc2.md").write_text("ERR-7778 is raised when the positioner feedback is noisy.", encoding="utf-8")
    code = main(["--documents", str(tmp_path / "doc.md"), "--documents", str(tmp_path / "doc2.md"), "ERR-7777", "--offline", "--json"])  # flag before the task, repeated
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and "indexed 2 documents" in payload["note"]
    code = main(["ERR-7777", "--offline", "--load-index", str(tmp_path / "idx.json"), "--deterministic", "--trace"])
    out = capsys.readouterr().out
    assert code == 0 and "loaded index" in out and "hybrid_retrieval.run" in out


def test_cli_evaluate_flag(capsys) -> None:
    code = main(["ERR-1203", "--offline", "--evaluate", "--max-llm-calls", "0"])
    out = capsys.readouterr().out
    assert code == 0 and "Evaluation (document level" in out and "[sparse]" in out and "[hybrid]" in out
