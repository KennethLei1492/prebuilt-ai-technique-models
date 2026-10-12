"""Heuristic planner, feature extraction, budget fitting and policy."""
from __future__ import annotations

from pipeline_core import Constraints, DomainProfile, TaskRequest, get_profile

from answer_verification import (
    AnswerVerificationConfig,
    HeuristicPlanner,
    Thresholds,
    VerificationPlan,
    apply_policy,
    classify_answer_type,
    extract_features,
    fit_to_budget,
    policy_for_profile,
)
from answer_verification.plan import estimate_calls, sanitize_verifiers

CFG = AnswerVerificationConfig()


def _plan(task: str, answer: str, *, domain: str = "general", evidence: int = 0, inputs: dict | None = None, constraints: Constraints | None = None) -> VerificationPlan:
    req = TaskRequest(task=task, inputs=inputs or {}, domain=domain, constraints=constraints or Constraints())
    return HeuristicPlanner(CFG).plan(req, get_profile(domain), answer=answer, evidence_count=evidence)


def test_classify_answer_types():
    assert classify_answer_type(extract_features("Write a function", "```python\ndef add(a, b):\n    return a + b\n```")) == "code"
    assert classify_answer_type(extract_features("Calculate the stress", "The stress is 12.5 MPa.")) == "numerical"
    assert classify_answer_type(extract_features("How to install the pump", "Step 1. Lift the pump.\nStep 2. Align the shaft.\nStep 3. Torque the bolts.")) == "procedural"
    assert classify_answer_type(extract_features("Propose an architecture for the control system", " ".join(["We recommend a layered design with a supervisory controller and local loops."] * 6))) == "design"
    assert classify_answer_type(extract_features("What is the capital of France?", "Paris.")) == "factual"
    assert classify_answer_type(extract_features("Summarise the report", " ".join(["The report discusses several unrelated observations about operations."] * 12))) == "free_text"


def test_prose_containing_return_or_import_is_not_code():
    """Regression: `return` / `import X` in English prose made the planner call the answer 'code' and compile it."""
    assert classify_answer_type(extract_features("When is the pump back?", "The pump will return to service after 3 hours of cooling.")) == "numerical"
    assert classify_answer_type(extract_features("What is the duty?", "The import duty is 5 percent of the invoice value.")) == "numerical"
    assert classify_answer_type(extract_features("Which pump?", "Select the pump from Table 3 and return the form.")) != "code"
    assert classify_answer_type(extract_features("Which fire class?", "Class A fire; return the extinguisher to its bracket.")) != "code"
    assert classify_answer_type(extract_features("Write it", "import math\nreturn math.pi")) == "code", "two weak markers at line starts"
    assert classify_answer_type(extract_features("Write it", "const f = (x) => x + 1")) == "code"
    assert classify_answer_type(extract_features("Query", "SELECT name FROM pumps WHERE duty > 75")) == "code"


def test_numerical_plan_uses_self_consistency_and_rules():
    plan = _plan("Compute the required preload", "The preload is 12.5 kN.")
    assert plan.answer_type == "numerical"
    assert plan.verifiers[:2] == ["self_consistency", "rules"]
    assert plan.samples == CFG.samples_default
    assert plan.source == "heuristic"
    assert plan.estimated_calls > 0


def test_evidence_puts_grounding_first_and_enables_revision():
    plan = _plan("What does the spec require?", "A 3 mm corrosion allowance is required.", evidence=2)
    assert plan.verifiers[0] == "grounding"
    assert plan.allow_revision is True
    assert _plan("What does the spec require?", "A 3 mm corrosion allowance is required.").allow_revision is False


def test_code_with_tests_and_permission_uses_execution():
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    with_tests = _plan("Implement add", code, inputs={"tests": "assert add(1, 2) == 3"}, constraints=Constraints(allow_code_execution=True))
    assert "code_execution" in with_tests.verifiers and "rules" in with_tests.verifiers
    without_permission = _plan("Implement add", code, inputs={"tests": "assert add(1, 2) == 3"})
    assert "code_execution" not in without_permission.verifiers
    assert "self_consistency" in without_permission.verifiers


def test_safety_critical_domain_makes_grounding_and_rules_mandatory():
    general = _plan("What load combination applies?", "The governing combination is 1.35G + 1.5Q.", evidence=1)
    civil = _plan("What load combination applies?", "The governing combination is 1.35G + 1.5Q.", domain="civil_structural_engineering", evidence=1)
    assert general.mandatory == []
    assert civil.mandatory == ["grounding", "rules"]
    assert civil.thresholds.accept >= 0.85 and civil.thresholds.min_supported_fraction == 1.0
    assert civil.final_effort == "max" and general.final_effort == "high"
    assert civil.samples == CFG.samples_max


def test_deterministic_constraint_reduces_samples():
    plan = _plan("Compute the preload", "12.5 kN", constraints=Constraints(deterministic=True))
    assert plan.samples == CFG.samples_min


def test_fit_to_budget_degrades_samples_then_drops_verifiers():
    plan = VerificationPlan(answer_type="numerical", verifiers=["self_consistency", "rubric_judge", "rules"], samples=10, mandatory=["rules"])
    tight = fit_to_budget(plan, Constraints(max_llm_calls=4), CFG, planning_calls=1)
    assert tight.estimated_calls <= 3
    assert "rules" in tight.verifiers
    assert tight.samples < 10
    unlimited = fit_to_budget(plan, Constraints(), CFG)
    assert unlimited.verifiers == plan.verifiers and unlimited.samples == 10
    minimal = fit_to_budget(plan, Constraints(max_llm_calls=1), CFG, planning_calls=1)
    assert minimal.verifiers == ["rules"]


def test_estimate_calls_counts_usc_only_for_free_form():
    assert estimate_calls(["self_consistency"], 5, CFG, answer_type="numerical") == 5 + 1
    assert estimate_calls(["self_consistency"], 5, CFG, answer_type="free_text") == 5 + CFG.usc_selection_passes + 1 + 1
    assert estimate_calls(["rules", "code_execution"], 5, CFG) == 1


def test_sanitize_verifiers_normalises_and_filters():
    assert sanitize_verifiers(["Grounding", "self-consistency", "bogus", "code_execution", "grounding"], allow_code_execution=False) == ["grounding", "self_consistency"]
    assert sanitize_verifiers(["code execution"], allow_code_execution=True) == ["code_execution"]


def test_policy_for_profiles():
    low = policy_for_profile(get_profile("general"), CFG)
    high = policy_for_profile(get_profile("mechanical_engineering"), CFG)
    crit = policy_for_profile(get_profile("chemical_process_engineering"), CFG)
    assert low.mandatory == [] and low.require_no_contradictions is False
    assert high.mandatory == ["rules"] and high.require_no_contradictions is True
    assert crit.mandatory == ["grounding", "rules"] and crit.final_effort == "max" and crit.thresholds.accept >= 0.85


def test_apply_policy_enforces_floors_and_skips_grounding_without_evidence():
    plan = VerificationPlan(answer_type="factual", verifiers=["self_consistency"], thresholds=Thresholds(accept=0.5))
    policy = policy_for_profile(get_profile("civil_structural_engineering"), CFG)
    no_ev = apply_policy(plan, policy, has_evidence=False)
    assert "grounding" not in no_ev.verifiers and no_ev.mandatory == ["rules"]
    with_ev = apply_policy(plan, policy, has_evidence=True)
    assert with_ev.mandatory == ["grounding", "rules"] and with_ev.thresholds.accept >= 0.85 and with_ev.final_effort == "max"


def test_custom_domain_profile_changes_plan_without_code_edits():
    profile = DomainProfile(name="test_pressure_vessels", description="ASME VIII vessels", units=["MPa", "mm"], risk_level="safety_critical", min_confidence_to_accept=0.9)
    req = TaskRequest(task="What is the shell thickness?", domain="general")
    plan = HeuristicPlanner(CFG).plan(req, profile, answer="The shell thickness is 12 mm.", evidence_count=1)
    assert plan.mandatory == ["grounding", "rules"] and plan.thresholds.accept >= 0.9
