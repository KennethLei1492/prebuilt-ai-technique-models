"""Basic usage of the answer verification model.

Runs offline without credentials (``make_llm()`` returns a MockLLM), so the
LLM-backed verifiers report themselves as inconclusive and the evidence-,
rule- and agreement-based verifiers do the work. With ``ANTHROPIC_API_KEY``
set the same code plans with one structured LLM call and runs the full set.

    python answer_verification_model/examples/basic_usage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "answer_verification_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import Evidence, TaskRequest, make_llm  # noqa: E402

from answer_verification import AnswerVerificationModel  # noqa: E402

SPEC = [
    Evidence(id="spec-4.2", content="Section 4.2: the pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces.", source="Pump specification rev C", score=1.0, retriever="provided"),
    Evidence(id="spec-t2", content="Table 2: discharge piping design pressure 16 bar at 80 degC.", source="Pump specification rev C", score=0.8, retriever="provided"),
]
TASK = "What corrosion allowance does the specification require for the pump casing?"


def main() -> None:
    model = AnswerVerificationModel(make_llm())

    print("== 1. run(): a grounded answer")
    result = model.run_sync(TaskRequest(task=TASK, inputs={"answer": "The specification requires a corrosion allowance of 3 mm on all wetted casing surfaces."}, context=SPEC))
    print(result.summary())
    print(result.output_text)
    print()

    print("== 2. run(): an answer the evidence contradicts")
    result = model.run_sync(TaskRequest(task=TASK, inputs={"answer": "The specification requires a corrosion allowance of 6 mm on all wetted casing surfaces."}, context=SPEC))
    print(result.summary())
    for reason in result.output.review_reasons:
        print("  review:", reason)
    print()

    print("== 3. Verifier protocol: verify(answer, task=..., evidence=...) -> Verdict")
    verdict = model.verify_sync("16 bar at 80 degC", task="What is the discharge piping design pressure?", evidence=SPEC)
    print(f"passed={verdict.passed} score={verdict.score:.2f} evidence={verdict.evidence_ids}")
    print()

    print("== 4. Domain adaptation: the same numeric answer in two domains")
    # Offline, pre-sampled candidates (e.g. from an upstream solver) are the LLM-free external signal for a bare number;
    # the general profile accepts on agreement, the safety-critical one still demands documents or tests.
    inputs = {"answer": "The required preload is 12.5 kN.", "candidates": ["12.5 kN", "12.5 kN", "12.5 kN"]}
    for domain in ("general", "civil_structural_engineering"):
        r = model.run_sync(TaskRequest(task="What bolt preload is required?", inputs=inputs, domain=domain))
        plan = r.metadata["plan"]
        print(f"{domain:30s} status={r.status.value:12s} confidence={r.confidence:.2f} accept>={plan['thresholds']['accept']:.2f} mandatory={plan['mandatory']}")
    print()

    print("== 5. Nothing conclusive ran: the answer is reported as unverified, never as SUCCESS")
    r = model.run_sync(TaskRequest(task="What bolt preload is required?", inputs={"answer": "The required preload is 12.5 kN."}, domain="general"))
    print(f"status={r.status.value} confidence={r.confidence:.2f} | {r.output.review_reasons[0]}")
    print()
    print(model.tracer.tree().splitlines()[0])


if __name__ == "__main__":
    main()
