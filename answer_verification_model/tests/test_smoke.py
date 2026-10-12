"""Smoke tests: the package imports, registers, and runs end to end offline."""
from __future__ import annotations

from pipeline_core import MockLLM, Status, TaskRequest, Verdict, Verifier, get

from answer_verification import AnswerVerificationModel, VerificationReport


def test_registered_under_component_name():
    assert get("answer_verification") is AnswerVerificationModel
    assert AnswerVerificationModel.spec.name == "answer_verification"


async def test_offline_end_to_end_numeric_answer():
    model = AnswerVerificationModel(MockLLM())
    request = TaskRequest(
        task="What bolt preload is required for the flange joint?",
        inputs={"answer": "The required preload is 12.5 kN per bolt."},
        domain="mechanical_engineering",
    )
    result = await model.run(request)
    assert result.status in (Status.SUCCESS, Status.NEEDS_REVIEW)
    assert isinstance(result.output, VerificationReport)
    assert result.metadata["plan"]["answer_type"] == "numerical"
    assert result.metadata["plan"]["source"] == "heuristic"
    assert result.output_text.startswith("Verification of candidate answer")
    assert result.confidence_basis
    assert 0.0 <= result.confidence <= 1.0
    assert any(s.name.startswith("verify.") for s in result.steps)


async def test_direct_api_is_a_verifier():
    model = AnswerVerificationModel(MockLLM())
    assert isinstance(model, Verifier)
    verdict = await model.verify("Paris is the capital of France.", task="What is the capital of France?")
    assert isinstance(verdict, Verdict)
    assert verdict.method == "answer_verification"
    assert 0.0 <= verdict.score <= 1.0


def test_sync_wrapper():
    model = AnswerVerificationModel(MockLLM())
    verdict = model.verify_sync("42 mm", task="What is the shaft diameter?")
    assert isinstance(verdict, Verdict)
