"""End-to-end runs: MockLLM, ScriptedLLM LLM-planned path, status gates, domains, budget, protocols, CLI."""
from __future__ import annotations

import json

import pytest

from pipeline_core import (
    BudgetExceededError,
    Constraints,
    DomainProfile,
    Evidence,
    LLMError,
    MockLLM,
    ScriptedLLM,
    Status,
    TaskRequest,
    Verdict,
    Verifier,
    register_profile,
)

from answer_verification import AnswerVerificationConfig, AnswerVerificationModel, VerificationReport
from answer_verification.cli import main as cli_main
from answer_verification.evaluation import evaluate, synthetic_examples

TASK = "What corrosion allowance does the specification require for the pump casing?"
GOOD = "The specification requires a corrosion allowance of 3 mm on all wetted casing surfaces."
BAD = "The specification requires a corrosion allowance of 6 mm on all wetted casing surfaces."
PLAN_JSON = {
    "answer_type": "numerical",
    "verifiers": ["rules"],
    "samples": 3,
    "use_evidence": True,
    "allow_revision": False,
    "judge_criteria": ["correctness"],
    "rationale": "deterministic rules suffice for this unit-bearing answer",
}
VERBALIZED = {"candidates": [{"answer": "12.5 kN", "probability": 0.8}, {"answer": "13 kN", "probability": 0.1}]}


# --------------------------------------------------------------- LLM-planned path


async def test_scripted_llm_planned_path():
    llm = ScriptedLLM([PLAN_JSON, VERBALIZED])
    model = AnswerVerificationModel(llm, config=AnswerVerificationConfig(llm_mode="force"))
    # mechanical_engineering declares units, so the planned `rules` verifier performs real unit checks
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs={"answer": "The preload is 12.5 kN"}, domain="mechanical_engineering"))
    plan = result.metadata["plan"]
    assert plan["source"] == "llm" and plan["verifiers"] == ["rules"] and plan["rationale"].startswith("deterministic rules")
    assert plan["thresholds"]["accept"] == 0.75, "the plan reports the enforced threshold (profile minimum)"
    assert result.usage.llm_calls == 2
    assert result.status == Status.SUCCESS
    methods = [v.method for v in result.output.verdicts]
    assert methods == ["rules", "verbalized_confidence"]
    assert result.output.verdicts[0].details["checks"], "rules performed real checks"
    assert result.confidence == pytest.approx((1.5 * 0.75 + 0.5 * 0.8) / 2.0, abs=1e-3)


async def test_advisory_confidence_alone_does_not_verify_an_answer():
    """Critical regression: with no verifier of record conclusive, the run must be NEEDS_REVIEW even if verbalised confidence is high."""
    llm = ScriptedLLM([PLAN_JSON, VERBALIZED])
    model = AnswerVerificationModel(llm, config=AnswerVerificationConfig(llm_mode="force"))
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs={"answer": "The preload is 12.5 kN"}, domain="general"))
    rules = next(v for v in result.output.verdicts if v.method == "rules")
    assert rules.details.get("inconclusive") is True, "general declares no units and no rules were supplied: nothing to check"
    assert result.status == Status.NEEDS_REVIEW and not result.output.accepted
    assert any("no verifier produced a conclusive verdict" in r for r in result.output.review_reasons)


async def test_scripted_llm_plan_with_self_consistency_samples():
    plan = {**PLAN_JSON, "verifiers": ["self_consistency", "rules"], "samples": 3}
    derive = {"final_answer": "12.5 kN", "key_steps": ["F = T / (K d)"]}
    llm = ScriptedLLM([plan, derive, derive, derive, VERBALIZED])
    model = AnswerVerificationModel(llm, config=AnswerVerificationConfig(llm_mode="force"))
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs={"answer": "12.5 kN"}))
    assert result.usage.llm_calls == 5 and result.status == Status.SUCCESS
    sc = next(v for v in result.output.verdicts if v.method == "self_consistency")
    assert sc.passed and sc.details["agreement_with_candidate"] == 1.0


async def test_invalid_llm_plan_falls_back_to_heuristics():
    llm = ScriptedLLM([{"garbage": True}], loop=True)
    model = AnswerVerificationModel(llm, config=AnswerVerificationConfig(llm_mode="force", elicit_verbalized_confidence=False))
    result = await model.run(TaskRequest(task="Compute the preload", inputs={"answer": "12.5 kN"}, constraints=Constraints(max_llm_calls=1)))
    assert result.metadata["plan"]["source"] == "heuristic"
    assert any("heuristic planner" in w for w in result.warnings)
    assert result.status != Status.FAILED


# ------------------------------------------------------------------ status gates


async def test_needs_review_below_requested_min_confidence():
    model = AnswerVerificationModel(MockLLM())
    # 2/3 agreement passes self-consistency (>= 0.6) but the fused confidence 0.67 is below the requested 0.9
    result = await model.run(TaskRequest(task="Compute the preload", inputs={"answer": "12.5 kN", "candidates": ["12.5 kN", "12.5 kN", "13 kN"]}, constraints=Constraints(min_confidence=0.9)))
    assert result.status == Status.NEEDS_REVIEW
    assert result.output.verdicts[0].method == "self_consistency" and result.output.verdicts[0].passed
    assert any("acceptance threshold 0.90" in r for r in result.output.review_reasons)
    assert result.metadata["plan"]["thresholds"]["accept"] == 0.9, "the plan shows the enforced threshold"


async def test_no_conclusive_verifier_is_needs_review_not_success():
    """Critical regression: the prior 0.5 equalled the general profile's thresholds and an unverified answer was SUCCESS."""
    model = AnswerVerificationModel(MockLLM())
    numeric = await model.run(TaskRequest(task="Compute", inputs={"answer": "2+2=5"}))
    assert numeric.status == Status.NEEDS_REVIEW and not numeric.output.accepted and numeric.confidence == 0.5
    assert all(v.details.get("inconclusive") for v in numeric.output.verdicts)
    assert any("no verifier produced a conclusive verdict" in r for r in numeric.output.review_reasons)
    prose = await model.run(TaskRequest(task="What is the capital of France?", inputs={"answer": "Paris."}))
    assert prose.status == Status.NEEDS_REVIEW and "not verified" in " ".join(prose.output.review_reasons)
    free = await model.run(TaskRequest(task="Summarise the design rationale in prose", inputs={"answer": " ".join(["The design balances cost and reliability through redundancy."] * 10)}))
    assert free.status == Status.NEEDS_REVIEW and "no conclusive verifier ran" in free.confidence_basis


async def test_contradicted_evidence_escalates(spec_evidence):
    model = AnswerVerificationModel(MockLLM())
    result = await model.run(TaskRequest(task=TASK, inputs={"answer": BAD}, context=spec_evidence))
    assert result.status == Status.NEEDS_REVIEW and result.output.escalated is True
    assert result.confidence < 0.5
    grounding = next(v for v in result.output.verdicts if v.method == "grounding")
    assert grounding.details["contradicted"] == 1


async def test_supported_evidence_is_accepted(spec_evidence):
    model = AnswerVerificationModel(MockLLM())
    result = await model.run(TaskRequest(task=TASK, inputs={"answer": GOOD}, context=spec_evidence))
    assert result.status == Status.SUCCESS and result.output.accepted
    assert result.evidence and result.evidence[0].id == "spec-1"
    assert result.output.metrics["grounding.supported_fraction"] == 1.0


# ------------------------------------------------------------ domain adaptation


async def test_domain_adaptation_safety_critical_vs_general():
    model = AnswerVerificationModel(MockLLM())
    inputs = {"answer": "12.5 kN", "candidates": ["12.5 kN"] * 3}  # pre-sampled agreement is the LLM-free external signal
    general = await model.run(TaskRequest(task="Compute the preload", inputs=inputs, domain="general"))
    critical = await model.run(TaskRequest(task="Compute the preload", inputs=inputs, domain="civil_structural_engineering"))
    assert general.status == Status.SUCCESS
    assert critical.status == Status.NEEDS_REVIEW
    assert any("safety-critical domain requires external evidence" in r for r in critical.output.review_reasons)
    assert critical.metadata["plan"]["thresholds"]["accept"] >= 0.85 > general.metadata["plan"]["thresholds"]["accept"]
    assert critical.metadata["plan"]["final_effort"] == "max"


async def test_high_risk_domain_blocks_contradictions_as_mandatory(spec_evidence):
    model = AnswerVerificationModel(MockLLM())
    result = await model.run(TaskRequest(task=TASK, inputs={"answer": BAD}, context=spec_evidence, domain="civil_structural_engineering"))
    reasons = " ".join(result.output.review_reasons)
    assert "mandatory verifier 'grounding' failed" in reasons and "contradicted" in reasons
    assert result.metadata["plan"]["mandatory"] == ["grounding", "rules"]


async def test_user_registered_profile_changes_behaviour_without_code_edits():
    register_profile(DomainProfile(name="test_hydraulics", description="Hydraulic systems", units=["bar", "L/min"], risk_level="high", min_confidence_to_accept=0.8), overwrite=True)
    model = AnswerVerificationModel(MockLLM())
    result = await model.run(TaskRequest(task="What is the relief pressure setting?", inputs={"answer": "The relief valve is set to 210.5"}, domain="test_hydraulics"))
    assert result.metadata["plan"]["answer_type"] == "numerical"
    rules = next(v for v in result.output.verdicts if v.method == "rules")
    assert not rules.passed, "domain units make a bare decimal a hard failure"
    assert result.status == Status.NEEDS_REVIEW and result.metadata["plan"]["mandatory"] == ["rules"]


# ----------------------------------------------------------------------- budget


async def test_budget_fitting_keeps_calls_within_limit():
    plan = {**PLAN_JSON, "verifiers": ["self_consistency", "rubric_judge", "rules"], "samples": 10}
    llm = ScriptedLLM([plan, {"final_answer": "12.5 kN", "key_steps": []}, VERBALIZED, {"final_answer": "12.5 kN", "key_steps": []}], loop=True)
    model = AnswerVerificationModel(llm, config=AnswerVerificationConfig(llm_mode="force"))
    result = await model.run(TaskRequest(task="Compute the preload", inputs={"answer": "12.5 kN"}, constraints=Constraints(max_llm_calls=3)))
    assert result.status != Status.FAILED
    assert result.usage.llm_calls <= 3
    assert result.metadata["plan"]["estimated_calls"] <= 2


async def test_budget_exhaustion_inside_a_verifier_is_contained():
    class Exploding:
        name = "self_consistency"

        async def _verify(self, vi, ctx):
            raise BudgetExceededError("LLM call budget of 0 exhausted")

    class Broken:
        name = "rubric_judge"

        async def _verify(self, vi, ctx):
            raise LLMError("model returned garbage")

    model = AnswerVerificationModel(MockLLM())
    model.verifiers["self_consistency"] = Exploding()  # type: ignore[assignment]
    model.verifiers["rubric_judge"] = Broken()  # type: ignore[assignment]
    result = await model.run(TaskRequest(task="Summarise the design rationale in prose", inputs={"answer": " ".join(["The design balances cost and reliability through redundancy."] * 10)}))
    assert result.status in (Status.PARTIAL, Status.NEEDS_REVIEW)
    assert any("budget" in w for w in result.warnings) or any("rubric_judge" in e for e in result.errors)
    assert all(isinstance(v, Verdict) for v in result.output.verdicts)


# -------------------------------------------------------------------- revision


async def test_majority_revision_is_reverified_by_rules_before_being_applied():
    model = AnswerVerificationModel(MockLLM())
    # mechanical_engineering declares units, so the rules re-check of the proposal is a real check, not a vacuous pass
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs={"answer": "20 kN", "candidates": ["12.5 kN"] * 4}, domain="mechanical_engineering"))
    report = result.output
    assert report.revised and report.revision_source == "self_consistency" and report.revision_verified
    assert report.final_answer == "12.5 kN" and report.revision_rounds == 1
    assert [v.method for v in report.revision_verdicts] == ["rules"] and report.revision_verdicts[0].passed
    assert report.revision_verdicts[0].details["checks"], "the confirming verdict performed at least one check"
    assert result.status == Status.NEEDS_REVIEW and any("revised by self_consistency (confirmed" in r for r in report.review_reasons)
    assert any(s.name == "revision.confirm" for s in result.steps)
    assert report.to_verdict().corrected_answer == "12.5 kN"


async def test_majority_revision_without_applicable_rules_rests_on_proposer_authority():
    """In a domain without units the rules re-check is inconclusive and the revision is applied on self-consistency's authority."""
    model = AnswerVerificationModel(MockLLM())
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs={"answer": "20 kN", "candidates": ["12.5 kN"] * 4}, domain="general"))
    report = result.output
    assert report.revised and not report.revision_verified and report.final_answer == "12.5 kN"
    assert [v.method for v in report.revision_verdicts] == ["rules"] and report.revision_verdicts[0].details.get("inconclusive") is True
    assert any("own authority" in i for i in report.issues)


async def test_multi_round_revision_follow_up_from_grounding(spec_evidence):
    """A failing grounding re-check may itself supply the next evidence-backed edit, which starts round 2."""
    revisions = iter(["The design pressure is 20 bar at 80 degC.", "The design pressure is 16 bar at 80 degC."])
    seen_answers: list[str] = []

    def responder(req):
        name = req.tags[0].split(".", 1)[1] if req.tags else ""
        text = req.last_user_text()
        if name == "claim_extract":
            claim = text.split("<answer>")[1].split("</answer>")[0].strip()
            seen_answers.append(claim)
            return {"claims": [{"text": claim, "type": "numerical"}]}
        if name == "claim_classify":
            ok = "16 bar" in text.split("<claims>")[1]
            return {"verdicts": [{"label": "supported" if ok else "contradicted", "evidence_ids": ["spec-2"], "note": "table 2"}]}
        if name == "revise":
            return {"revised_answer": next(revisions), "changes": ["pressure"]}
        return None

    model = AnswerVerificationModel(MockLLM(responder), config=AnswerVerificationConfig(llm_mode="force", elicit_verbalized_confidence=False, max_revision_rounds=2))
    result = await model.run(
        TaskRequest(
            task="What is the discharge piping design pressure?",
            inputs={"answer": "The design pressure is 25 bar at 80 degC."},
            context=spec_evidence,
            constraints=Constraints(deterministic=True),  # heuristic plan: grounding first, no LLM planning call
        )
    )
    report = result.output
    assert report.revised and report.revision_verified and report.revision_rounds == 2
    assert report.final_answer == "The design pressure is 16 bar at 80 degC."
    assert seen_answers[1:] == ["The design pressure is 20 bar at 80 degC.", "The design pressure is 16 bar at 80 degC."]


async def test_revision_without_external_confirmer_rests_on_proposer_authority():
    model = AnswerVerificationModel(MockLLM())
    result = await model.run(TaskRequest(task="What is the capital of France?", inputs={"answer": "Lyon", "candidates": ["Paris"] * 4}))
    report = result.output
    assert result.metadata["plan"]["verifiers"] == ["self_consistency"]
    assert report.revised and not report.revision_verified and report.final_answer == "Paris" and report.revision_rounds == 0
    assert any("own authority" in i for i in report.issues)


async def test_zero_revision_rounds_keeps_the_original_answer():
    model = AnswerVerificationModel(MockLLM(), max_revision_rounds=0)
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs={"answer": "20 kN", "candidates": ["12.5 kN"] * 4}))
    assert not result.output.revised and result.output.final_answer == "20 kN"
    assert any("disabled by policy" in i for i in result.output.issues)
    assert result.output.to_verdict().corrected_answer is None


async def test_revision_that_fails_reverification_is_not_applied():
    model = AnswerVerificationModel(MockLLM())
    inputs = {"answer": "20 kN", "candidates": ["12.5"] * 4, "rules": [{"type": "regex", "pattern": r"kN", "description": "unit stated"}]}
    result = await model.run(TaskRequest(task="Compute the bolt preload", inputs=inputs))
    report = result.output
    assert not report.revised and report.final_answer == "20 kN" and report.revision_rounds == 1
    assert any("not applied" in i and "rules" in i for i in report.issues)


async def test_release_threshold_is_a_floor_and_the_plan_reports_the_enforced_value():
    """Regression: release_threshold=0.3 showed accept 0.3 in the plan while execute() enforced the profile's 0.75."""
    low = AnswerVerificationModel(MockLLM(), release_threshold=0.3)
    result = await low.run(TaskRequest(task="Compute the preload", inputs={"answer": "12.5 kN"}, domain="mechanical_engineering"))
    assert result.metadata["plan"]["thresholds"]["accept"] == 0.75 == result.output.accept_threshold
    high = AnswerVerificationModel(MockLLM(), release_threshold=0.95)
    result = await high.run(TaskRequest(task="Compute the preload", inputs={"answer": "12.5 kN"}, domain="mechanical_engineering"))
    assert result.metadata["plan"]["thresholds"]["accept"] == 0.95 == result.output.accept_threshold


def test_verbalized_confidence_lives_in_uncertainty_module():
    from answer_verification import verbalized_confidence
    from answer_verification.uncertainty import verbalized_confidence as from_uncertainty
    from answer_verification.verifiers import elicit_verbalized_confidence

    assert verbalized_confidence is from_uncertainty is elicit_verbalized_confidence


# -------------------------------------------------------------- protocol / API


async def test_verify_protocol_returns_fused_verdict(spec_evidence):
    model = AnswerVerificationModel(MockLLM())
    assert isinstance(model, Verifier)
    good = await model.verify(GOOD, task=TASK, evidence=spec_evidence)
    bad = await model.verify(BAD, task=TASK, evidence=spec_evidence)
    assert good.passed and good.score > bad.score and not bad.passed
    assert good.evidence_ids == ["spec-1"] and good.details["status"] == "success"
    assert bad.issues and bad.details["escalated"] is True


async def test_injected_retriever_and_executor_are_used(spec_evidence):
    class FakeRetriever:
        calls: list[str] = []

        async def retrieve(self, query, *, k=10, filters=None):
            self.calls.append(query)
            return spec_evidence[:1]

    class CountingExecutor:
        def __init__(self):
            self.batches = 0

        async def map(self, fn, items, *, concurrency=None):
            self.batches += 1
            out = []
            for it in items:
                out.append(await fn(it))
            return out

    retriever, executor = FakeRetriever(), CountingExecutor()
    llm = MockLLM(lambda req: {"final_answer": "3 mm", "key_steps": []} if "answer_verification.derive" in req.tags else None)
    model = AnswerVerificationModel(llm, retriever=retriever, executor=executor, config=AnswerVerificationConfig(llm_mode="force", elicit_verbalized_confidence=False, llm_equivalence_clustering=False))
    model.planner = model.planner  # heuristic planner is still used for the plan when the LLM plan is invalid
    result = await model.run(TaskRequest(task=TASK, inputs={"answer": "3 mm"}, constraints=Constraints(max_llm_calls=20)))
    assert retriever.calls == [TASK]
    assert any(s.name == "retrieve" for s in result.steps)
    assert executor.batches >= 1
    assert "grounding" in result.metadata["plan"]["verifiers"]


async def test_task_prefix_supplies_candidate_and_draft_warning():
    model = AnswerVerificationModel(MockLLM())
    result = await model.run("Verify this answer: the required NPSH margin is 1.5 m")
    assert result.output.answer.startswith("the required NPSH margin")
    assert result.metadata["answer_source"] == "task"
    plain = await model.run("Explain NPSH margin")
    assert plain.metadata["answer_source"] == "task" and any("no candidate answer" in w for w in plain.warnings)


async def test_report_shape_and_rendering(spec_evidence):
    model = AnswerVerificationModel(MockLLM())
    report = await model.verify_report(GOOD, task=TASK, evidence=spec_evidence, domain="mechanical_engineering")
    assert isinstance(report, VerificationReport)
    text = report.render()
    assert "Status: SUCCESS" in text and "grounding: PASS" in text and report.final_answer == GOOD
    dumped = report.model_dump(mode="json")
    assert dumped["plan"]["answer_type"] in ("numerical", "factual", "free_text")


async def test_online_draft_path_generates_and_verifies_a_candidate(spec_evidence):
    draft = "The specification requires a corrosion allowance of 3 mm on all wetted casing surfaces [spec-1]."
    llm = MockLLM(lambda req: draft if "answer_verification.draft" in req.tags else None)
    model = AnswerVerificationModel(llm, config=AnswerVerificationConfig(llm_mode="force", elicit_verbalized_confidence=False))
    result = await model.run(TaskRequest(task=TASK, context=spec_evidence, constraints=Constraints(deterministic=True)))
    assert result.metadata["answer_source"] == "draft" and result.output.answer == draft
    assert any(s.name == "draft" for s in result.steps) and any("draft was generated" in w for w in result.warnings)
    grounding = next(v for v in result.output.verdicts if v.method == "grounding")
    assert grounding.passed and grounding.details["citation_precision"] == 1.0


def test_coerce_evidence_accepts_strings_dicts_and_evidence():
    from answer_verification import coerce_evidence

    out = coerce_evidence(["plain text", {"id": "d2", "text": "dict text", "source": "s", "metadata": {"k": 1}}, Evidence(id="e3", content="ev"), {"content": ""}])
    assert [e.id for e in out] == ["doc-1", "d2", "e3"] and out[1].source == "s" and out[1].metadata == {"k": 1}
    assert coerce_evidence("one") [0].id == "doc-1" and coerce_evidence(None) == [] and coerce_evidence({"content": "x"})[0].content == "x"


def test_cli_runs_supplied_tests_with_code_execution(tmp_path, capsys):
    tests = tmp_path / "tests_add.py"
    tests.write_text("assert add(2, 3) == 5\nprint('1 passed')\n", encoding="utf-8")
    code = cli_main(["Implement add", "--answer", "```python\ndef add(a, b):\n    return a + b\n```", "--tests", str(tests), "--domain", "software_engineering", "--offline", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    exe = next(v for v in data["output"]["verdicts"] if v["method"] == "code_execution")
    assert exe["passed"] and exe["details"]["tests_passed"] == 1 and data["status"] == "success"


def test_evaluation_metrics_pure_functions():
    from answer_verification.evaluation import auroc, brier_score, expected_calibration_error

    assert brier_score([1.0, 0.0], [True, False]) == 0.0 and brier_score([0.5, 0.5], [True, False]) == 0.25
    assert expected_calibration_error([0.9, 0.9, 0.1, 0.1], [True, True, False, False]) == pytest.approx(0.1)
    assert expected_calibration_error([1.0], [True]) == 0.0, "p == 1.0 lands in the top bin"
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5 and auroc([0.5], [True]) is None


async def test_describe_includes_config():
    model = AnswerVerificationModel(MockLLM(), samples_default=7)
    d = model.describe()
    assert d["name"] == "answer_verification" and d["config"]["samples_default"] == 7


# ------------------------------------------------------------------- evaluation


async def test_evaluation_harness_offline():
    model = AnswerVerificationModel(MockLLM())
    report = await evaluate(model, synthetic_examples(6))
    assert report.n == 6 and report.accuracy == 1.0 and report.auroc == 1.0
    assert report.brier < 0.25 and 0.0 <= report.ece <= 1.0
    assert report.llm_calls == 0 and "decision accuracy" in report.render()


# -------------------------------------------------------------------------- CLI


def test_cli_offline_json(capsys):
    code = cli_main(["Verify this answer: the required NPSH margin is 1.5 m and the pump provides 2.1 m", "--offline", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["component"] == "answer_verification" and data["status"] in ("success", "needs_review")
    assert data["metadata"]["plan"]["answer_type"] == "numerical"


def test_cli_offline_text_with_documents_and_inputs(capsys):
    code = cli_main([TASK, "--answer", GOOD, "--offline", "--domain", "mechanical_engineering", "--document", "spec-1: Section 4.2: the pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces.", "--input", 'rules=[{"type":"regex","pattern":"3 mm"}]', "--trace"])
    out = capsys.readouterr().out
    assert code == 0 and "status=success" in out and "grounding: PASS" in out and "answer_verification.run" in out


def test_cli_evaluate(capsys):
    assert cli_main(["--evaluate", "--offline", "--n", "4"]) == 0
    assert "Evaluated 4 labelled answers" in capsys.readouterr().out
