"""Regressions found by the enterprise pipeline integration example."""
from __future__ import annotations

from answer_verification import AnswerVerificationModel
from answer_verification.claims import _content_tokens, _identifiers, extract_citations
from pipeline_core import Evidence, MockLLM


def test_citation_markers_with_hash_and_slash_are_not_claim_identifiers():
    text = "The casing corrosion allowance is 3 mm [SPEC-PS7-MECH#0] per clause 6.3.1 [docs/api610.md#2]."
    assert "spec-ps7-mech" not in _identifiers(text)
    assert not any("#" in t or "docs" == t for t in _content_tokens(text))
    assert "SPEC-PS7-MECH#0" in extract_citations(text)


async def test_verify_attaches_usage_and_status_to_verdict():
    model = AnswerVerificationModel(MockLLM())
    ev = [Evidence(id="SPEC#0", content="Pressure casings shall have a minimum corrosion allowance of 3 mm.", source="spec")]
    verdict = await model.verify("The corrosion allowance is 3 mm [SPEC#0].", task="What corrosion allowance applies?", evidence=ev)
    assert "usage" in verdict.details and isinstance(verdict.details["usage"], dict)
    assert "llm_calls" in verdict.details["usage"]
    assert verdict.details["component_status"] in ("success", "needs_review", "partial", "failed")
    assert verdict.details["request_id"]
