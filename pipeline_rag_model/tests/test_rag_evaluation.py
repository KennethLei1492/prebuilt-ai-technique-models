"""Evaluation metrics (pure) and the offline harness."""
from __future__ import annotations

import pytest

from pipeline_core import MockLLM

from pipeline_rag import PipelineRAGModel
from pipeline_rag.evaluation import (
    QAExample,
    RAGEvaluator,
    abstention_metrics,
    attribute_failure_point,
    citation_metrics,
    context_precision,
    context_recall,
    exact_match,
    faithfulness,
    sample_corpus,
    sample_qa,
    synthetic_qa,
    token_f1,
)
from pipeline_rag.generator import Statement


def test_answer_metrics() -> None:
    assert exact_match("The allowance is 3 mm [spec].", "3 mm") == 1.0
    assert exact_match("The allowance is 4 mm.", "3 mm") == 0.0
    assert token_f1("rated 75 kW at 1480 rpm", "75 kW at 1480 rpm") == pytest.approx(2 * (5 / 6) * 1.0 / (5 / 6 + 1.0))
    assert token_f1("", "x") == 0.0


def test_context_metrics_match_parent_documents() -> None:
    retrieved = ["PS-100#2", "MD-7", "HT-22#1"]
    assert context_precision(retrieved, ["PS-100", "REQ-CW"]) == pytest.approx(1 / 3)
    assert context_recall(retrieved, ["PS-100", "REQ-CW"]) == pytest.approx(0.5)
    assert context_recall([], ["x"]) == 0.0 and context_precision([], ["x"]) == 0.0


def test_faithfulness_and_citation_metrics() -> None:
    assert faithfulness([True, True, False]) == pytest.approx(2 / 3)
    assert faithfulness([]) == 0.0
    sts = [
        Statement(text="A", citation_ids=["d1", "d2"]),
        Statement(text="B", citation_ids=["d3"]),
        Statement(text="C", citation_ids=[]),
    ]
    supports = lambda text, doc: (text, doc) in {("A", "d1"), ("B", "d3")}  # noqa: E731
    recall, precision = citation_metrics(sts, supports)
    assert recall == pytest.approx(2 / 3)  # C has no citation, A and B each have a supporting one
    assert precision == pytest.approx(2 / 3)  # d2 does not support A
    assert citation_metrics([], supports) == (0.0, 0.0)


def test_abstention_metrics() -> None:
    m = abstention_metrics([(True, False), (False, True), (True, True), (False, False)])
    assert m["rate"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    assert abstention_metrics([]) == {"rate": 0.0, "precision": 0.0, "recall": 0.0}


def test_failure_point_attribution() -> None:
    assert attribute_failure_point(answerable=False, abstained=False, correct=None, gold_retrieved=None, gold_in_context=None, status="success") == "FP1"
    assert attribute_failure_point(answerable=False, abstained=True, correct=None, gold_retrieved=None, gold_in_context=None, status="success") is None
    assert attribute_failure_point(answerable=True, abstained=True, correct=None, gold_retrieved=False, gold_in_context=False, status="success") == "FP2"
    assert attribute_failure_point(answerable=True, abstained=False, correct=False, gold_retrieved=True, gold_in_context=False, status="success") == "FP3"
    assert attribute_failure_point(answerable=True, abstained=False, correct=False, gold_retrieved=True, gold_in_context=True, status="success") == "FP4"
    assert attribute_failure_point(answerable=True, abstained=False, correct=True, gold_retrieved=True, gold_in_context=True, status="failed") == "FP5"
    assert attribute_failure_point(answerable=True, abstained=False, correct=True, gold_retrieved=True, gold_in_context=True, status="success", complete=False) == "FP7"
    assert attribute_failure_point(answerable=True, abstained=False, correct=True, gold_retrieved=True, gold_in_context=True, status="success", complete=True) is None


def test_synthetic_qa_is_answerable_by_retrieval() -> None:
    qa = synthetic_qa(sample_corpus(), n=4)
    assert len(qa) == 4
    for ex in qa:
        assert ex.question.startswith("According to") and ex.reference_answer and ex.relevant_ids
    assert [e.question for e in synthetic_qa(sample_corpus(), n=4)] == [e.question for e in qa]  # deterministic per seed
    assert [e.question for e in synthetic_qa(sample_corpus(), n=4, seed=1)] == [e.question for e in synthetic_qa(sample_corpus(), n=4, seed=1)]


async def test_evaluator_offline_report_and_gates() -> None:
    model = PipelineRAGModel(MockLLM())
    evaluator = RAGEvaluator(model)
    report = await evaluator.evaluate(sample_qa(), documents=sample_corpus())
    assert report.n == len(sample_qa()) and report.judge == "heuristic"
    assert report.means["context_recall"] > 0.5
    assert report.means["faithfulness"] >= 0.9  # extractive answers are supported by construction
    assert report.means["citation_recall"] >= 0.8
    assert report.gates["faithfulness"] and report.gates["citation_recall"]
    assert report.abstention["recall"] == 1.0  # the unanswerable question was declined
    assert "single_step" in report.route_distribution
    assert report.records[-1].abstained and report.records[-1].failure_point is None
    assert "FP1" not in report.failure_points
    assert "n=6" in report.summary()
    assert report.usage.llm_calls == 0


def test_evaluator_sync_wrapper_subset() -> None:
    model = PipelineRAGModel(MockLLM())
    report = RAGEvaluator(model).evaluate_sync(sample_qa()[:2], documents=sample_corpus())
    assert report.n == 2 and report.means["exact_match"] >= 0.5
