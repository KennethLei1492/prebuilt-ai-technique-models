"""Engineering example: a custom safety-critical domain, rules, pre-sampled candidates and code tests.

Everything runs offline (MockLLM when no credentials are present). It shows:

1. registering a new ``DomainProfile`` changes verification posture with no code edits;
2. structured rules from ``inputs["rules"]`` (required sections, regex, numeric range);
3. upstream samples in ``inputs["candidates"]`` give self-consistency an LLM-free signal;
4. a code answer verified by executing the supplied tests in the sandbox.

    python answer_verification_model/examples/engineering_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "answer_verification_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import Constraints, DomainProfile, Evidence, TaskRequest, make_llm, register_profile  # noqa: E402

from answer_verification import AnswerVerificationModel  # noqa: E402

register_profile(
    DomainProfile(
        name="pressure_vessels",
        description="ASME BPVC Section VIII Division 1 pressure vessel design.",
        units=["MPa", "mm", "bar", "degC", "kN"],
        verification_rules=[
            "Every thickness must state the corrosion allowance included.",
            "Allowable stress must cite the material table and design temperature.",
        ],
        risk_level="safety_critical",
        min_confidence_to_accept=0.85,  # the safety-critical floor; offline fusion of grounding + rules tops out at 0.875
        standards=["ASME BPVC VIII-1", "ASME II-D"],
    ),
    overwrite=True,
)

EVIDENCE = [
    Evidence(id="ug-27", content="UG-27(c)(1): for cylindrical shells under internal pressure the required thickness is t = P R / (S E - 0.6 P). For the design case P = 1.5 MPa, R = 600 mm, S = 138 MPa, E = 1.0 the required thickness is 6.6 mm plus 3 mm corrosion allowance, 9.6 mm in total.", source="ASME VIII-1 (project calc note CN-12)", score=1.0, retriever="provided"),
    Evidence(id="ii-d", content="Section II-D Table 1A: SA-516 Gr 70 allowable stress S = 138 MPa at 150 degC.", source="ASME II-D", score=0.9, retriever="provided"),
]
TASK = "What minimum shell thickness is required for the SA-516 Gr 70 vessel at the design case?"
RULES = [
    {"type": "required_sections", "sections": ["Assumptions", "Result"], "description": "calculation note layout"},
    {"type": "regex", "pattern": r"corrosion allowance", "description": "states the corrosion allowance"},
    {"type": "range", "min": 5, "max": 15, "unit": "mm", "description": "plausible thickness band"},  # 'unit' selects the mm quantity stated as the result
]
GOOD = "Assumptions: P = 1.5 MPa, R = 600 mm, S = 138 MPa at 150 degC, E = 1.\nResult: the required thickness is 6.6 mm, to which the 3 mm corrosion allowance is added."
BAD = "Result: the required thickness is 4.1 mm including corrosion allowance."


def main() -> None:
    model = AnswerVerificationModel(make_llm())

    print("== Safety-critical custom domain: grounded answer with rules")
    r = model.run_sync(TaskRequest(task=TASK, inputs={"answer": GOOD, "rules": RULES}, context=EVIDENCE, domain="pressure_vessels"))
    print(r.summary())
    print("plan:", r.metadata["plan"]["verifiers"], "mandatory:", r.metadata["plan"]["mandatory"], "accept >=", r.metadata["plan"]["thresholds"]["accept"])
    print(r.output_text)
    print()

    print("== Same domain: answer that violates the evidence and the rules")
    r = model.run_sync(TaskRequest(task=TASK, inputs={"answer": BAD, "rules": RULES}, context=EVIDENCE, domain="pressure_vessels"))
    print(r.summary())
    for reason in r.output.review_reasons:
        print("  review:", reason)
    print()

    print("== General domain, same bad answer: lower bar, still flagged by the evidence")
    r = model.run_sync(TaskRequest(task=TASK, inputs={"answer": BAD}, context=EVIDENCE, domain="general"))
    print(r.summary(), "| escalated:", r.output.escalated)
    print()

    print("== Pre-sampled candidates give self-consistency an LLM-free signal")
    r = model.run_sync(TaskRequest(task="Compute the required thickness", inputs={"answer": "9.6 mm", "candidates": ["9.6 mm", "9.6 mm", "9.60 mm", "9.1 mm", "9.6 mm"]}, domain="pressure_vessels", context=EVIDENCE))
    sc = next(v for v in r.output.verdicts if v.method == "self_consistency")
    print(f"self-consistency agreement={sc.details['agreement_with_candidate']:.2f} clusters={sc.details['clusters']} passed={sc.passed}")
    print()

    print("== A candidate the samples outvote: the majority answer is re-verified by grounding and the rules before it replaces the candidate")
    r = model.run_sync(TaskRequest(task="Compute the required thickness", inputs={"answer": "9.1 mm", "candidates": ["9.6 mm", "9.6 mm", "9.60 mm", "9.6 mm"]}, domain="pressure_vessels", context=EVIDENCE))
    rep = r.output
    print(f"status={r.status.value} revised={rep.revised} source={rep.revision_source} verified={rep.revision_verified} rounds={rep.revision_rounds} final={rep.final_answer!r}")
    print("re-check:", ", ".join(f"{v.method}={'PASS' if v.passed else 'FAIL'}" for v in rep.revision_verdicts))
    print()

    print("== A domain without units and no rules: the rule verifier has nothing to check and says so")
    r = model.run_sync(TaskRequest(task="Compute the required thickness", inputs={"answer": "9.6 mm"}, domain="general"))
    rules = next(v for v in r.output.verdicts if v.method == "rules")
    print(f"status={r.status.value} rules inconclusive={rules.details.get('inconclusive')} | {r.output.review_reasons[0]}")
    print()

    print("== Code answer verified by executing the supplied tests")
    code = "```python\ndef required_thickness(p, r, s, e=1.0):\n    return p * r / (s * e - 0.6 * p)\n```"
    tests = "assert abs(required_thickness(1.5, 600, 138) - 6.56) < 0.05\nprint('1 passed')"
    r = model.run_sync(
        TaskRequest(task="Implement UG-27(c)(1) required thickness", inputs={"answer": code, "tests": tests}, domain="software_engineering", constraints=Constraints(allow_code_execution=True))
    )
    exe = next(v for v in r.output.verdicts if v.method == "code_execution")
    print(r.summary())
    print(f"code_execution passed={exe.passed} tests_passed={exe.details.get('tests_passed')} status={r.status.value}")


if __name__ == "__main__":
    main()
