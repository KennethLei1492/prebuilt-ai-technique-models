"""Checkpoint test: the package imports, registers, and runs end to end offline."""
from __future__ import annotations

import pipeline_core
from pipeline_core import MockLLM, Status
from pipeline_core.protocols import Retriever

from hybrid_retrieval import HybridRetrievalModel, HybridRetriever, RetrievalOutput, RetrievalPlan


def test_registered_component() -> None:
    assert pipeline_core.get("hybrid_retrieval") is HybridRetrievalModel
    assert HybridRetrievalModel.spec.name == "hybrid_retrieval"


def test_model_and_retriever_satisfy_retriever_protocol(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    assert isinstance(model, Retriever)
    assert isinstance(model.retriever, HybridRetriever)
    assert isinstance(model.retriever, Retriever)


async def test_run_offline_end_to_end(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    n_chunks = model.index(corpus)
    assert n_chunks >= len(corpus)
    result = await model.run("What material is specified for the pump impeller?")
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW)
    assert result.evidence, result.output_text
    assert result.evidence[0].metadata["doc_id"] == "spec-pump"
    assert isinstance(result.output, RetrievalOutput)
    assert result.output.n_results == len(result.evidence)
    plan = RetrievalPlan.model_validate(result.metadata["plan"])
    assert plan.source == "heuristic"
    assert "Hybrid retrieval for" in result.output_text
    assert 0.0 <= result.confidence <= 1.0 and result.confidence_basis
    assert any(s.name == "retrieve" for s in result.steps)


def test_run_sync_with_documents_in_inputs(corpus) -> None:
    model = HybridRetrievalModel(MockLLM())
    result = model.run_sync("ERR-1203 cause", inputs={"documents": corpus, "k": 3}, domain="software_engineering")
    assert result.ok or result.status == Status.NEEDS_REVIEW
    assert len(result.evidence) <= 3
    assert result.evidence[0].metadata["doc_id"] == "runbook-retry"
    assert result.metadata["plan"]["query_type"] == "identifier"
