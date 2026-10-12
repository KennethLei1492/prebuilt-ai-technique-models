"""Domain adaptation: the same question under a general and a safety-critical profile.

Shows how a ``DomainProfile`` changes the plan (mandatory grading,
verification, inline citations, abstention), the prompts (domain glossary and
``ENGINEERING_RIGOR`` in the stable system prefix), the retrieval queries
(glossary expansion) and the acceptance threshold, without any code change.
Also registers a custom profile for a new field and runs the offline
evaluation harness.

    python pipeline_rag_model/examples/engineering_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "pipeline_rag_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import DomainProfile, TaskRequest, make_llm, register_profile  # noqa: E402

from pipeline_rag import PipelineRAGConfig, PipelineRAGModel, RAGPlan  # noqa: E402
from pipeline_rag.evaluation import RAGEvaluator, sample_corpus, sample_qa  # noqa: E402

QUESTION = "What corrosion allowance does the specification require for the pump casing, and what FoS margin does the test report demonstrate?"


def show(label: str, result) -> None:  # type: ignore[no-untyped-def]
    plan = RAGPlan.model_validate(result.metadata["plan"])
    print("-" * 100)
    print(f"[{label}] {result.summary()}")
    print(f"  plan: route={plan.route} transforms={plan.query_transforms} grade={plan.grade_context} verify={plan.verify} citations={plan.citation_style} abstain={plan.abstain_when_insufficient}")
    print(f"  rationale: {plan.rationale}")
    print(f"  variants: {[v.kind + ':' + v.text[:50] for v in result.output.query_variants]}")
    print(f"  confidence: {result.confidence:.2f} vs threshold {result.output.confidence.threshold:.2f} -> status {result.status.value}")
    print(f"  answer: {result.output_text[:400]}")


def main() -> None:
    llm = make_llm()
    documents = sample_corpus()

    # 1. Same question, two built-in profiles.
    general = PipelineRAGModel(llm, config=PipelineRAGConfig(citation_style="footnote"))
    safety = PipelineRAGModel(llm, config=PipelineRAGConfig(citation_style="footnote"))
    show("general", general.run_sync(TaskRequest(task=QUESTION, inputs={"documents": documents}, domain="general")))
    show("mechanical_engineering (high risk)", safety.run_sync(TaskRequest(task=QUESTION, inputs={"documents": documents}, domain="mechanical_engineering")))

    # 2. A new field registered by the user: no code changes in the model.
    register_profile(
        DomainProfile(
            name="offshore_rotating_equipment",
            description="Rotating equipment on offshore platforms: pumps, compressors, drivers and their inspection regimes.",
            glossary={"UT": "Ultrasonic thickness survey", "RPN": "Risk priority number from FMEA", "FoS": "Factor of safety - ratio of allowable to applied load"},
            units=["bar", "mm", "kW", "rpm", "mm/s"],
            document_types=["specification", "data sheet", "FMEA", "incident report", "maintenance procedure"],
            verification_rules=["Every pressure must state the reference (design, test, operating).", "Vibration limits must cite the ISO 10816 zone."],
            risk_level="safety_critical",
            min_confidence_to_accept=0.85,
            retrieval_hints={"sparse_weight": 0.65},
            standards=["ISO 10816-3", "API 610", "ISO 9906"],
        ),
        overwrite=True,
    )
    offshore = PipelineRAGModel(llm)
    show("offshore_rotating_equipment (custom, safety_critical)", offshore.run_sync(TaskRequest(task="What UT survey requirement protects the casing corrosion allowance?", inputs={"documents": documents}, domain="offshore_rotating_equipment")))

    # 3. Offline evaluation harness over the labelled sample set.
    print("-" * 100)
    report = RAGEvaluator(offshore).evaluate_sync(sample_qa(), documents=documents)
    print("evaluation:", report.summary())
    print("routes:", report.route_distribution, "failure points:", report.failure_points)


if __name__ == "__main__":
    main()
