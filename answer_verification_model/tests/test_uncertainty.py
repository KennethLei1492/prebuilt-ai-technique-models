"""Pure logic: canonicalisation, agreement/entropy, adaptive stopping, calibration, fusion."""
from __future__ import annotations

import math

import pytest

from pipeline_core import Verdict

from answer_verification import (
    CalibrationParams,
    VerbalizedConfidence,
    aggregate_verdicts,
    canonicalize_answer,
    fit_platt,
    majority_stability,
    platt_scale,
    semantic_agreement,
    should_stop_sampling,
)
from answer_verification.uncertainty import ConfidenceCandidate, probability_for_candidate, similarity_ratio


def test_extract_number_picks_the_answer_number():
    """Regression: the last number in the text was used ('80' in '16 bar at 80 degC', '16' from 'M16')."""
    from answer_verification.uncertainty import extract_number

    assert extract_number("16 bar at 80 degC") == 16.0
    assert extract_number("16 bar at 80 degC", unit="degC") == 80.0, "unit family selects the quantity"
    assert extract_number("Use a 75 kW motor at 1480 rpm.", unit="kW") == 75.0
    assert extract_number("Use a 75 kW motor at 1480 rpm.", unit="rpm") == 1480.0
    assert extract_number("The preload is 12.5 kN per M16 bolt") == 12.5
    assert extract_number("Assumptions: P = 1.5 MPa, R = 600 mm. Result: the required thickness is 6.6 mm.") == 6.6, "result marker wins"
    assert extract_number("t = P R / (S E - 0.6 P) = 1.5 * 600 / 137.1 = 6.56 mm") == 6.56, "right-hand side of the last '='"
    assert extract_number("```python\nx = 42\n```\nThe answer is 7.") == 7.0, "numbers inside code are not the answer"
    assert extract_number("no numbers here") is None
    assert canonicalize_answer("16 bar at 80 degC", "numerical") == "16"
    assert canonicalize_answer("The preload is 12.5 kN per M16 bolt", "numerical") == "12.5"


def test_aggregate_verdicts_accepts_a_generator():
    verdicts = (v for v in [_v("rules", True, 0.75), _v("debate", False, 0.5, inconclusive=True)])
    conf, basis = aggregate_verdicts(verdicts)
    assert conf == 0.75 and "inconclusive: debate" in basis


def test_canonicalize_numbers_code_and_text():
    assert canonicalize_answer("The preload is 12.50 kN", "numerical") == canonicalize_answer("12.5kN", "numerical") == "12.5"
    assert canonicalize_answer("1,480 rpm", "numerical") == "1480"
    assert canonicalize_answer("def f(x):\n    return x  # comment", "code") == canonicalize_answer("def f(x): return x", "code")
    assert canonicalize_answer("The capital of France is Paris.", "factual") == canonicalize_answer("Paris is the capital of France", "factual")


def test_semantic_agreement_clusters_and_entropy():
    res = semantic_agreement(["12.5 kN", "12.5 kN", "13 kN", "12.5 kN"], answer_type="numerical")
    assert res.counts == [3, 1] and res.agreement == 0.75 and res.majority_index == 0
    assert math.isclose(res.entropy, -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    unanimous = semantic_agreement(["Paris", "paris.", "Paris"], answer_type="factual")
    assert unanimous.agreement == 1.0 and unanimous.normalized_entropy == 0.0
    assert semantic_agreement([]).agreement == 0.0


def test_semantic_agreement_accepts_injected_equivalence():
    res = semantic_agreement(["a", "b", "c"], equivalent=lambda x, y: True)
    assert res.counts == [3]


def test_majority_stability_and_adaptive_stopping():
    assert math.isclose(majority_stability([3]), 1 - 0.5**4)
    assert majority_stability([5]) > 0.95 >= majority_stability([3])
    assert majority_stability([3, 3]) == pytest.approx(0.5)
    assert not should_stop_sampling([2], min_samples=3)
    assert should_stop_sampling([5], threshold=0.95, min_samples=3)
    assert not should_stop_sampling([2, 1], threshold=0.95, min_samples=3)
    assert majority_stability([]) == 0.0


def test_platt_identity_and_fit():
    assert platt_scale(0.7) == 0.7
    assert platt_scale(0.7, CalibrationParams(a=1.0, b=2.0)) > 0.7
    scores = [0.9, 0.85, 0.8, 0.95, 0.3, 0.2, 0.35, 0.1]
    labels = [1, 1, 0, 1, 0, 0, 0, 0]
    params = fit_platt(scores, labels, iterations=200)
    assert platt_scale(0.9, params) > platt_scale(0.3, params)
    with pytest.raises(ValueError):
        fit_platt([], [])


def _v(method: str, passed: bool, score: float, **details):
    return Verdict(method=method, passed=passed, score=score, details=details)


def test_aggregate_verdicts_weighted_and_capped():
    conf, basis = aggregate_verdicts([_v("grounding", True, 1.0), _v("rules", True, 0.75)])
    assert conf == pytest.approx((1.5 * 1.0 + 1.5 * 0.75) / 3.0, abs=1e-4)
    assert "grounding=1.00" in basis
    capped, basis = aggregate_verdicts([_v("grounding", True, 1.0), _v("rules", False, 0.2)])
    assert capped == 0.2 and "capped by failed rules" in basis
    contra, basis = aggregate_verdicts([_v("self_consistency", True, 0.9), _v("grounding", False, 0.7, contradicted=1)])
    assert contra == 0.5 and "contradicted" in basis


def test_aggregate_ignores_inconclusive_and_uses_prior_when_nothing_ran():
    conf, basis = aggregate_verdicts([_v("rubric_judge", False, 0.5, inconclusive=True)])
    assert conf == 0.5 and "no conclusive verifier" in basis
    conf2, basis2 = aggregate_verdicts([_v("rules", True, 0.75), _v("debate", False, 0.5, inconclusive=True)])
    assert conf2 == 0.75 and "inconclusive: debate" in basis2


def test_probability_for_candidate_matches_by_canonical_key():
    dist = VerbalizedConfidence(candidates=[ConfidenceCandidate(answer="12.5 kN", probability=0.7), ConfidenceCandidate(answer="13 kN", probability=0.2)])
    assert probability_for_candidate(dist, "The preload is 12.5 kN", "numerical") == 0.7
    assert probability_for_candidate(dist, "99 kN", "numerical") == 0.0
    assert similarity_ratio("abc", "abc") == 1.0
