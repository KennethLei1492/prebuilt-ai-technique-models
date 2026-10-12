"""Basic usage: answer questions from a handful of documents with citations.

Runs offline: ``make_llm()`` returns a ``MockLLM`` without credentials, and
the model's heuristic stages (routing, lexical grading, extractive answer,
lexical support check) produce a grounded, cited answer. With
``ANTHROPIC_API_KEY`` set the same code runs against Claude.

    python pipeline_rag_model/examples/basic_usage.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "pipeline_rag_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import TaskRequest, make_llm  # noqa: E402

from pipeline_rag import PipelineRAGModel, RAGPlan  # noqa: E402
from pipeline_rag.evaluation import sample_corpus  # noqa: E402


def main() -> None:
    os.environ.setdefault("PIPELINE_OFFLINE", os.environ.get("PIPELINE_OFFLINE", "0"))
    llm = make_llm()
    model = PipelineRAGModel(llm)
    documents = sample_corpus()

    questions = [
        "What corrosion allowance does the specification require for the pump casing?",
        "What was the root cause of the high vibration trip on pump P-102 and which procedure was updated?",
        "What recurring themes appear across all the incident reports?",
        "What warranty period does the pump vendor offer?",  # not in the corpus: the model should abstain
    ]
    for question in questions:
        result = model.run_sync(TaskRequest(task=question, inputs={"documents": documents}))
        plan = RAGPlan.model_validate(result.metadata["plan"])
        print("=" * 100)
        print("Q:", question)
        print(result.summary())
        print(f"plan: {plan.describe()}")
        if len(result.output.routes_tried) > 1:
            print("escalation:", " -> ".join(result.output.routes_tried))
        print(f"confidence: {result.confidence:.2f} ({result.confidence_basis})")
        print("A:", result.output_text)
        if result.warnings:
            print("warnings:", "; ".join(result.warnings))

    # The direct API: index once, ask many times; the model is also a Retriever.
    print("=" * 100)
    model.index(documents)
    hits = model.answer_sync("What is the vibration acceptance limit?", domain="mechanical_engineering")
    print("direct answer:", hits.output_text)
    print(model.tracer.tree().splitlines()[0])


if __name__ == "__main__":
    main()
