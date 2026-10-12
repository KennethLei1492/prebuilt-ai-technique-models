"""Adapting the decomposition model to an engineering field and composing it with other components.

Demonstrates, offline:

1. registering a new ``DomainProfile`` (aerospace structures) with its own
   decomposition hints, units, verification rules and risk level, and how the
   plan, graph and prompts change without any code change;
2. injecting protocol objects: a ``Retriever`` (per-sub-question evidence), a
   ``Verifier`` (the external signal that can fail a node and trigger
   replanning) and a ``SubTaskSolver`` (another component's ``run``);
3. the offline evaluation harness.

    python task_decomposition_model/examples/engineering_example.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "task_decomposition_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import DomainProfile, Evidence, TaskRequest, TaskResult, Usage, Verdict, make_llm, register_profile  # noqa: E402

from task_decomposition import EvalCase, TaskDecompositionModel, evaluate  # noqa: E402

TASK = (
    "Assess the wing-to-fuselage attachment fitting: identify the limit and ultimate loads, "
    "compute the bearing and shear-out stresses for the fastener pattern, check the margins of safety against the allowables, "
    "and list the assumptions that must be confirmed by test."
)

DOCS = [
    Evidence(id="loads-rev-c", content="Limit load case 7: 42 kN tension, 9 kN shear at the fitting. Ultimate = 1.5 x limit.", source="loads report rev C", score=0.9, retriever="provided"),
    Evidence(id="mmpds-7075", content="7075-T73 plate: Ftu 460 MPa, Fsu 275 MPa, Fbru(e/D=2) 1000 MPa at room temperature.", source="material allowables", score=0.8, retriever="provided"),
]


class DocRetriever:
    """Stands in for HybridRetrievalModel: returns the fixed evidence for every sub-question."""

    async def retrieve(self, query: str, *, k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        return DOCS[:k]


class UnitsVerifier:
    """Stands in for AnswerVerificationModel: passes only answers that state a unit (external, rule-based signal)."""

    async def verify(self, answer: str, *, task: str, evidence: Sequence[Evidence] = ()) -> Verdict:
        has_unit = any(u in answer for u in ("kN", "MPa", "mm", "%")) or "mock" in answer  # MockLLM answers are stubs
        return Verdict(method="units_rule", passed=has_unit, score=0.9 if has_unit else 0.2, issues=[] if has_unit else ["no unit stated"], details={"usage": Usage().model_dump()})


async def rag_solver(request: TaskRequest) -> TaskResult:
    """Stands in for PipelineRAGModel.run: any ``SubTaskSolver`` callable works."""
    cited = [e.id for e in request.context][:2]
    return TaskResult(
        request_id=request.request_id,
        component="pipeline_rag",
        output_text=f"[rag] {request.task}: 42 kN limit / 63 kN ultimate, allowable 275 MPa (sources: {', '.join(cited) or 'none'})",
        confidence=0.82,
        evidence=list(request.context)[:2],
        usage=Usage(llm_calls=1),
    )


def register_aerospace_profile() -> DomainProfile:
    profile = DomainProfile(
        name="aerospace_structures",
        description="Airframe structural analysis: loads, stress, fatigue and damage tolerance of metallic and composite structure.",
        glossary={"MS": "Margin of safety = allowable / applied - 1", "DLL": "Design limit load", "DUL": "Design ultimate load (1.5 x DLL)"},
        units=["kN", "MPa", "mm", "N/mm"],
        document_types=["loads report", "stress report", "material allowables", "drawing", "test report"],
        verification_rules=[
            "Every margin of safety must be reported against the ultimate allowable with the load case identified.",
            "Material allowables must cite the source table and the temperature/basis (A- or B-basis).",
            "Units must be SI and dimensionally consistent.",
        ],
        risk_level="safety_critical",
        min_confidence_to_accept=0.85,
        decomposition_hints=["Separate load derivation from stress analysis", "Check every fastener failure mode (bearing, shear-out, net section) individually", "Attach a verification method to each margin"],
        standards=["FAR 25.305", "FAR 25.571", "MMPDS"],
    )
    return register_profile(profile, overwrite=True)


async def main() -> None:
    register_aerospace_profile()
    llm = make_llm()

    # 1. Domain adaptation: the same task, two profiles.
    model = TaskDecompositionModel(llm)
    general = await model.run(TaskRequest(task=TASK, domain="general", context=DOCS))
    aero = await model.run(TaskRequest(task=TASK, domain="aerospace_structures", context=DOCS))
    for name, res in (("general", general), ("aerospace_structures", aero)):
        print(f"== {name}: {res.summary()}")
        print(f"   plan: {res.output.plan.describe().splitlines()[0]}")
        print(f"   nodes: {res.output.graph.ids()}")
        print(f"   basis: {res.confidence_basis}")
    print()
    print(aero.output.graph.to_ascii())
    print()

    # 2. Composition: retriever + verifier + protocol solver.
    composed = TaskDecompositionModel(llm, retriever=DocRetriever(), verifier=UnitsVerifier(), solver=rag_solver)
    res = await composed.run(TaskRequest(task=TASK, domain="aerospace_structures"))
    print("== composed (retriever + verifier + RAG solver):", res.summary())
    print("   status:", res.status.value, "| verdict:", res.output.verdict.method if res.output.verdict else None, "| replans:", res.output.replans)
    print("   first sub-answer:", next(iter(res.output.results.values())).output_text[:110])
    print()

    # 3. Evaluation harness on a tiny in-domain suite.
    summary = await evaluate(
        TaskDecompositionModel(llm),
        [
            EvalCase(task=TASK, domain="aerospace_structures", expected_strategy="dag"),
            EvalCase(task="Convert 42 kN to lbf.", domain="aerospace_structures", expected_strategy="direct"),
        ],
    )
    print("== evaluation")
    print(summary.table())


if __name__ == "__main__":
    asyncio.run(main())
