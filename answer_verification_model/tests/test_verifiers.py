"""Each verifier through its LLM path (dispatching MockLLM, llm_mode='force') and its offline path."""
from __future__ import annotations

from pipeline_core import Constraints, Evidence, LLMRequest, RunContext, TaskRequest, Tracer  # noqa: F401 - Evidence used in tests below

from answer_verification import (
    AnswerVerificationConfig,
    ChainOfVerificationVerifier,
    CodeExecutionVerifier,
    DebateVerifier,
    GroundingVerifier,
    RubricJudgeVerifier,
    RuleVerifier,
    SelfConsistencyVerifier,
    VerificationPlan,
)
from answer_verification.uncertainty import is_inconclusive
from answer_verification.verifiers import VerifyInputs, elicit_verbalized_confidence, pairwise_decision

FORCE = AnswerVerificationConfig(llm_mode="force")


def index_of_candidate(req: LLMRequest, needle: str) -> int:
    """1-based index of the ``[i] text`` line in the request containing ``needle``."""
    import re

    for m in re.finditer(r"^\[(\d+)\] (.*)$", req.last_user_text(), re.M):
        if needle in m.group(2):
            return int(m.group(1))
    return 1


AUTO = AnswerVerificationConfig()


def _ctx(llm, task: str, *, domain: str = "general", evidence=(), inputs=None, constraints=None) -> RunContext:
    req = TaskRequest(task=task, inputs=inputs or {}, context=list(evidence), domain=domain, constraints=constraints or Constraints())
    return RunContext(req, Tracer(name="test"), llm, "test")


def _plan(answer_type: str, verifier: str, **kw) -> VerificationPlan:
    return VerificationPlan(answer_type=answer_type, verifiers=[verifier], **kw)


# ------------------------------------------------------------ self-consistency


async def test_self_consistency_majority_agrees_with_candidate(dispatch_llm):
    answers = iter(["12.5 kN", "12.5 kN", "12.5 kN", "13 kN", "12.5 kN", "12.5 kN"])
    llm = dispatch_llm({"derive": lambda req: {"final_answer": next(answers), "key_steps": ["x"]}})
    ctx = _ctx(llm, "Compute the preload")
    v = await SelfConsistencyVerifier(FORCE).verify("The preload is 12.5 kN", task="Compute the preload", ctx=ctx, plan=_plan("numerical", "self_consistency", samples=5))
    assert v.passed and v.score >= 0.8 and v.corrected_answer is None
    assert v.details["samples"] == 5 and v.details["clusters"] == [5, 1], "pool = 4 agreeing samples + candidate, 1 dissenter"
    assert ctx.usage.llm_calls == 5


async def test_self_consistency_stops_early_and_proposes_majority(dispatch_llm):
    llm = dispatch_llm({"derive": {"final_answer": "12.5 kN", "key_steps": []}})
    ctx = _ctx(llm, "Compute the preload")
    v = await SelfConsistencyVerifier(FORCE).verify("20 kN", task="Compute the preload", ctx=ctx, plan=_plan("numerical", "self_consistency", samples=10))
    assert not v.passed and v.score == 0.0
    assert v.corrected_answer == "12.5 kN"
    assert v.details["stopped_early"] is True and v.details["samples"] < 10


async def test_self_consistency_uses_presampled_candidates_offline():
    from pipeline_core import MockLLM

    ctx = _ctx(MockLLM(), "Compute", inputs={"candidates": ["12.5 kN", "12.5 kN", "12.4 kN"]})
    v = await SelfConsistencyVerifier(AUTO).verify("12.5 kN", task="Compute", ctx=ctx, plan=_plan("numerical", "self_consistency", samples=3))
    assert not is_inconclusive(v) and v.passed and v.details["sampled_llm"] == 0 and ctx.usage.llm_calls == 0


async def test_self_consistency_inconclusive_offline_without_samples():
    from pipeline_core import MockLLM

    ctx = _ctx(MockLLM(), "Compute")
    v = await SelfConsistencyVerifier(AUTO).verify("12.5 kN", task="Compute", ctx=ctx, plan=_plan("numerical", "self_consistency"))
    assert is_inconclusive(v)


async def test_universal_self_consistency_for_free_text(dispatch_llm):
    candidate = "Use a layered architecture with a supervisory controller."
    samples = iter(["A layered architecture with a supervisory controller is best.", "Layered architecture, supervisory controller on top.", "Use a flat peer-to-peer design."])
    llm = dispatch_llm(
        {
            "derive": lambda req: {"final_answer": next(samples), "key_steps": []},
            "equivalence": lambda req: {"groups": [[1, 2, 4], [3]]},
            "usc_select": lambda req: {"selected_index": index_of_candidate(req, "Use a layered"), "rationale": "majority"},
        }
    )
    ctx = _ctx(llm, "Propose an architecture")
    v = await SelfConsistencyVerifier(FORCE).verify(candidate, task="Propose an architecture", ctx=ctx, plan=_plan("free_text", "self_consistency", samples=3))
    assert v.passed and v.details["usc_selected_index"] == 3 and len(v.details["usc_passes"]) == FORCE.usc_selection_passes
    assert v.details["agreement_with_candidate"] > 0.6
    assert ctx.usage.llm_calls == 3 + 1 + FORCE.usc_selection_passes


# ------------------------------------------------------- chain of verification


def _cove_table(*, revised: str, labels=("agree", "contradict")):
    return {
        "cove_plan": {"questions": [{"question": "What corrosion allowance applies?", "claim_excerpt": "3 mm allowance"}, {"question": "What is the design pressure?", "claim_excerpt": "25 bar"}]},
        "cove_answer": lambda req: {"answer": "3 mm" if "allowance" in req.last_user_text() else "16 bar", "confident": True, "evidence_ids": ["spec-1"]},
        "cove_crosscheck": {"labels": list(labels)},
        "revise": {"revised_answer": revised, "changes": ["25 bar -> 16 bar"]},
    }


async def test_cove_finds_contradiction_and_revises_with_evidence(dispatch_llm, spec_evidence):
    answer = "The casing has a 3 mm allowance. The design pressure is 25 bar at 80 degC."
    llm = dispatch_llm(_cove_table(revised=answer.replace("25 bar", "16 bar")))
    ctx = _ctx(llm, "Summarise the spec", evidence=spec_evidence)
    v = await ChainOfVerificationVerifier(FORCE).verify(answer, task="Summarise the spec", evidence=spec_evidence, ctx=ctx, plan=_plan("free_text", "chain_of_verification", allow_revision=True))
    assert not v.passed and v.details["contradict"] == 1 and v.details["agree"] == 1
    assert v.corrected_answer and "16 bar" in v.corrected_answer
    assert v.details["preservation"] >= FORCE.revision_min_preservation
    assert ctx.usage.llm_calls == 1 + 2 + 1 + 1


async def test_cove_rejects_oversized_revision_and_never_revises_without_evidence(dispatch_llm, spec_evidence):
    answer = "The casing has a 3 mm allowance. The design pressure is 25 bar at 80 degC."
    llm = dispatch_llm(_cove_table(revised="Completely different text about something else entirely, rewritten from scratch."))
    ctx = _ctx(llm, "Summarise", evidence=spec_evidence)
    v = await ChainOfVerificationVerifier(FORCE).verify(answer, task="Summarise", evidence=spec_evidence, ctx=ctx, plan=_plan("free_text", "chain_of_verification"))
    assert v.corrected_answer is None and any("revision rejected" in i for i in v.issues)
    ctx2 = _ctx(dispatch_llm(_cove_table(revised=answer)), "Summarise")
    v2 = await ChainOfVerificationVerifier(FORCE).verify(answer, task="Summarise", ctx=ctx2, plan=_plan("free_text", "chain_of_verification"))
    assert v2.corrected_answer is None and any("intrinsic self-correction" in i for i in v2.issues)
    assert ctx2.usage.llm_calls == 4  # no revise call without evidence


async def test_cove_keeps_its_verdict_when_the_optional_revision_call_fails(dispatch_llm, spec_evidence):
    """Regression: an LLMError in the REVISE call propagated and erased the contradiction findings."""
    from pipeline_core import LLMError

    def failing_revise(req: LLMRequest):
        raise LLMError("revise call failed")

    table = {**_cove_table(revised="unused"), "revise": failing_revise}
    answer = "The casing has a 3 mm allowance. The design pressure is 25 bar at 80 degC."
    ctx = _ctx(dispatch_llm(table), "Summarise the spec", evidence=spec_evidence)
    v = await ChainOfVerificationVerifier(FORCE).verify(answer, task="Summarise the spec", evidence=spec_evidence, ctx=ctx, plan=_plan("free_text", "chain_of_verification", allow_revision=True))
    assert not is_inconclusive(v) and not v.passed and v.details["contradict"] == 1
    assert v.corrected_answer is None and any("contradicted" in i for i in v.issues) and any("revision call failed" in i for i in v.issues)
    assert any("revision failed" in w for w in ctx.warnings)


async def test_cove_is_inconclusive_offline():
    from pipeline_core import MockLLM

    v = await ChainOfVerificationVerifier(AUTO).verify("x", task="t", ctx=_ctx(MockLLM(), "t"), plan=_plan("free_text", "chain_of_verification"))
    assert is_inconclusive(v)


# ------------------------------------------------------------------ grounding


async def test_grounding_llm_mode_reports_three_numbers(dispatch_llm, spec_evidence):
    llm = dispatch_llm(
        {
            "claim_extract": {"claims": [{"text": "The casing corrosion allowance is 3 mm.", "type": "numerical"}, {"text": "The design pressure is 25 bar.", "type": "numerical"}]},
            "claim_classify": {"verdicts": [{"label": "supported", "evidence_ids": ["spec-1"], "note": "4.2"}, {"label": "contradicted", "evidence_ids": ["spec-2"], "note": "16 bar"}]},
            "revise": {"revised_answer": "The casing allowance is 3 mm and the design pressure is 16 bar.", "changes": ["pressure"]},
        }
    )
    ctx = _ctx(llm, "Summarise", evidence=spec_evidence)
    answer = "The casing allowance is 3 mm and the design pressure is 25 bar."
    v = await GroundingVerifier(FORCE).verify(answer, task="Summarise", evidence=spec_evidence, ctx=ctx, plan=_plan("free_text", "grounding", allow_revision=True), require_no_contradictions=True)
    assert v.details["mode"] == "llm"
    assert v.details["claim_count"] == 2 and v.details["supported"] == 1 and v.details["contradicted"] == 1
    assert not v.passed and v.score == 0.25 and v.evidence_ids == ["spec-1"]
    assert v.corrected_answer and "16 bar" in v.corrected_answer


async def test_grounding_offline_lexical_mode(spec_evidence):
    from pipeline_core import MockLLM

    ctx = _ctx(MockLLM(), "Summarise", evidence=spec_evidence)
    v = await GroundingVerifier(AUTO).verify("The pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces.", task="Summarise", evidence=spec_evidence, ctx=ctx, plan=_plan("factual", "grounding"))
    assert v.details["mode"] == "lexical" and v.passed and v.score == 1.0 and ctx.usage.llm_calls == 0
    none = await GroundingVerifier(AUTO).verify("anything", task="t", ctx=_ctx(MockLLM(), "t"), plan=_plan("factual", "grounding"))
    assert is_inconclusive(none)


async def test_grounding_treats_a_one_line_answer_as_a_single_claim():
    """Regression: '9.6 mm' fell below split_claims' minimum and grounding was inconclusive even with matching evidence."""
    from pipeline_core import MockLLM

    ev = [Evidence(id="cn-12", content="Calc note CN-12: the required shell thickness including corrosion allowance is 9.6 mm.")]
    ctx = _ctx(MockLLM(), "Compute the required thickness", evidence=ev)
    v = await GroundingVerifier(AUTO).verify("9.6 mm", task="Compute the required thickness", evidence=ev, ctx=ctx, plan=_plan("numerical", "grounding"))
    assert not is_inconclusive(v) and v.passed and v.details["claim_count"] == 1 and v.evidence_ids == ["cn-12"]
    assert v.details["claims"][0]["claim"] == "9.6 mm"
    wrong = await GroundingVerifier(AUTO).verify("12 mm", task="Compute the required thickness", evidence=ev, ctx=ctx, plan=_plan("numerical", "grounding"))
    assert not wrong.passed and wrong.details["contradicted"] == 1


# --------------------------------------------------------------- rubric judge


async def test_rubric_judge_averages_passes_and_flags_low_criteria(dispatch_llm):
    llm = dispatch_llm({"rubric": {"scores": [{"criterion": "correctness", "score": 5, "rationale": "ok"}, {"criterion": "completeness", "score": 2, "rationale": "misses the margin"}], "overall": 4}})
    ctx = _ctx(llm, "Explain the design")
    v = await RubricJudgeVerifier(FORCE).verify("A design explanation.", task="Explain the design", ctx=ctx, plan=_plan("design", "rubric_judge", judge_criteria=["correctness", "completeness"]))
    assert v.details["passes"] == FORCE.judge_samples and ctx.usage.llm_calls == FORCE.judge_samples
    assert v.score == round(((3.5 + 4) / 2 - 1) / 4, 4)
    assert any(i.startswith("completeness") for i in v.issues)
    assert v.passed == (v.score >= 0.7)


async def test_pairwise_compare_counts_only_position_consistent_wins(dispatch_llm):
    winners = iter(["A", "B"])  # second call presents (B, A); 'B' there means original A -> consistent
    llm = dispatch_llm({"pairwise": lambda req: {"winner": next(winners), "rationale": "better"}})
    judge = RubricJudgeVerifier(FORCE, llm=llm)
    out = await judge.compare("answer one", "answer two", task="t")
    assert out == {"winner": "A", "position_consistent": True, "orders": ["A", "A"]}
    assert pairwise_decision("A", "B") == {"winner": "tie", "position_consistent": False, "orders": ["A", "B"]}


# ---------------------------------------------------------------------- rules


async def test_rule_verifier_uses_domain_units_and_inputs():
    from pipeline_core import MockLLM

    ctx = _ctx(MockLLM(), "Compute", domain="mechanical_engineering", inputs={"rules": [{"type": "required_sections", "sections": ["Assumptions"]}]})
    v = await RuleVerifier(AUTO).verify("Assumptions: none. The stress is 12.5 MPa.", task="Compute", ctx=ctx, plan=_plan("numerical", "rules"))
    assert v.passed and v.score == 0.75
    bad = await RuleVerifier(AUTO).verify("Assumptions: none. The stress is 12.5", task="Compute", ctx=ctx, plan=_plan("numerical", "rules"))
    assert not bad.passed and bad.details["hard_failures"] == 1, "final answer number without a unit is a hard failure"
    soft = await RuleVerifier(AUTO).verify("Assumptions: E = 1.0. The stress is 12.5 MPa.", task="Compute", ctx=ctx, plan=_plan("numerical", "rules"))
    assert soft.passed and soft.details["soft_failures"] == 1 and soft.score < 0.75, "dimensionless factors are only soft issues"


async def test_rule_verifier_is_inconclusive_when_no_check_applies():
    """Regression: zero applicable checks (general domain, no rules/reference) was reported as a conclusive PASS at 0.75."""
    from pipeline_core import MockLLM

    ctx = _ctx(MockLLM(), "Compute", domain="general")
    v = await RuleVerifier(AUTO).verify("2+2=5", task="Compute", ctx=ctx, plan=_plan("numerical", "rules"))
    assert is_inconclusive(v) and v.details["checks"] == [] and "no deterministic check applies" in v.details["reason"]
    with_rule = await RuleVerifier(AUTO).verify("2+2=5", task="Compute", ctx=_ctx(MockLLM(), "Compute", inputs={"reference": "4"}), plan=_plan("numerical", "rules"))
    assert not is_inconclusive(with_rule) and not with_rule.passed


# ------------------------------------------------------------- code execution


async def test_code_execution_verifier_runs_tests_when_allowed():
    from pipeline_core import MockLLM

    code = "```python\ndef add(a, b):\n    return a + b\n```"
    tests = "assert add(2, 3) == 5\nprint('1 passed')"
    allowed = _ctx(MockLLM(), "Implement add", inputs={"tests": tests}, constraints=Constraints(allow_code_execution=True))
    v = await CodeExecutionVerifier(AUTO).verify(code, task="Implement add", ctx=allowed, plan=_plan("code", "code_execution"))
    assert v.passed and v.score > 0.9 and v.details["tests_passed"] == 1
    denied = _ctx(MockLLM(), "Implement add", inputs={"tests": tests})
    d = await CodeExecutionVerifier(AUTO).verify(code, task="Implement add", ctx=denied, plan=_plan("code", "code_execution"))
    assert is_inconclusive(d)
    failing = await CodeExecutionVerifier(AUTO).verify("```python\ndef add(a, b):\n    return a * b\n```", task="Implement add", ctx=allowed, plan=_plan("code", "code_execution"))
    assert not failing.passed and failing.score < 0.5


async def test_code_execution_runs_unfenced_python_in_profiles_whose_first_language_is_not_python():
    """Regression: data_engineering lists sql first, so plain Python was 'no sandbox runtime for language sql'."""
    from pipeline_core import MockLLM

    tests = "assert add(2, 3) == 5\nprint('1 passed')"
    ctx = _ctx(MockLLM(), "Implement add", domain="data_engineering", inputs={"tests": tests}, constraints=Constraints(allow_code_execution=True))
    v = await CodeExecutionVerifier(AUTO).verify("def add(a, b):\n    return a + b\n", task="Implement add", ctx=ctx, plan=_plan("code", "code_execution"))
    assert not is_inconclusive(v) and v.passed and v.details["tests_passed"] == 1
    sql = await CodeExecutionVerifier(AUTO).verify("```sql\nSELECT 1;\n```", task="Query", ctx=ctx, plan=_plan("code", "code_execution"))
    assert is_inconclusive(sql) and "sql" in sql.details["reason"], "a fence that names another language is respected"
    explicit = _ctx(MockLLM(), "Implement add", domain="data_engineering", inputs={"tests": tests, "language": "scala"}, constraints=Constraints(allow_code_execution=True))
    assert is_inconclusive(await CodeExecutionVerifier(AUTO).verify("def add(a, b): return a + b", task="Implement add", ctx=explicit, plan=_plan("code", "code_execution")))


# --------------------------------------------------------------------- debate


async def test_debate_converges_and_judges(dispatch_llm):
    turn_prompts: list[str] = []

    def turn(req: LLMRequest):
        turn_prompts.append(req.last_user_text())
        return {"acceptable": True, "position": "the sizing is conservative", "key_issue": "none"}

    llm = dispatch_llm({"debate_turn": turn, "debate_judge": {"verdict": "correct", "confidence": 0.9, "unresolved_issues": []}})
    ctx = _ctx(llm, "Review the design")
    v = await DebateVerifier(FORCE).verify("Use a 75 kW motor.", task="Review the design", ctx=ctx, plan=_plan("design", "debate"))
    assert v.passed and v.score == 0.9 and v.details["converged"] is True and v.details["rounds"] == 1
    assert ctx.usage.llm_calls == FORCE.debate_agents + 1
    assert len(turn_prompts) == FORCE.debate_agents and len(set(turn_prompts)) == FORCE.debate_agents, "every round-1 agent gets a different persona framing"


async def test_debate_second_round_shows_peers_in_agent_specific_order_and_falls_back_when_the_judge_fails(dispatch_llm):
    from pipeline_core import LLMError

    from answer_verification import prompts as P

    votes = iter([True, False, True, True, True, True])  # round 1 splits, round 2 converges
    peers_seen: list[str] = []

    def turn(req: LLMRequest):
        text = req.last_user_text()
        peers_seen.append(text.split("<peer_positions>")[1].split("</peer_positions>")[0].strip())
        return {"acceptable": next(votes), "position": f"position {len(peers_seen)}", "key_issue": "sizing margin"}

    def failing_judge(req: LLMRequest):
        raise LLMError("judge unavailable")

    llm = dispatch_llm({"debate_turn": turn, "debate_judge": failing_judge})
    ctx = _ctx(llm, "Review the design")
    v = await DebateVerifier(FORCE).verify("Use a 75 kW motor.", task="Review the design", ctx=ctx, plan=_plan("design", "debate"))
    assert v.details["rounds"] == 2 and v.details["converged"] is True and v.details["judged"] is False
    assert v.passed and v.details["verdict"] == "correct" and v.details["judge_confidence"] == 1.0, "judge-less fallback: final-round vote share"
    assert any("judge call failed" in w for w in ctx.warnings)
    round2 = peers_seen[FORCE.debate_agents :]
    assert all("(none yet)" in p for p in peers_seen[: FORCE.debate_agents])
    assert len(set(round2)) > 1, "peer summaries are shuffled per agent"
    assert all(sorted(p.splitlines()) == sorted(round2[0].splitlines()) for p in round2), "same content, different order"
    assert len(P.DEBATE_FRAMINGS) >= FORCE.debate_agents


# -------------------------------------------------------- verbalised confidence


async def test_verbalized_confidence_reads_candidate_probability(dispatch_llm):
    llm = dispatch_llm({"verbalized_confidence": {"candidates": [{"answer": "12.5 kN", "probability": 0.8}, {"answer": "13 kN", "probability": 0.1}]}})
    ctx = _ctx(llm, "Compute")
    vi = VerifyInputs(task="Compute", answer="The preload is 12.5 kN", answer_type="numerical")
    v = await elicit_verbalized_confidence(ctx, vi, FORCE)
    assert v.method == "verbalized_confidence" and v.score == 0.8 and v.details["advisory"] is True


def test_build_request_layout_is_cache_friendly(spec_evidence):
    from pipeline_core import MockLLM

    from answer_verification import prompts as P
    from answer_verification.verifiers import build_request, documents_block

    ctx = _ctx(MockLLM(), "t", domain="mechanical_engineering")
    req: LLMRequest = build_request(ctx, P.DERIVE, schema=None, effort="low", documents=documents_block(spec_evidence), task="the task", framing="f")
    assert req.system_stable[0] == P.DERIVE.system
    assert any("Domain: mechanical_engineering" in b for b in req.system_stable)
    assert any("engineering rigor" in b.lower() for b in req.system_stable)
    assert "spec-1" in req.last_user_text() and "the task" in req.last_user_text()
    assert not any("spec-1" in b for b in req.system_stable)
