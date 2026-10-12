"""answer_verification - decide whether an answer is correct, grounded and complete.

Public surface::

    from answer_verification import AnswerVerificationModel, AnswerVerificationConfig, VerificationPlan

    model = AnswerVerificationModel()                       # MockLLM without credentials
    result = model.run_sync("Verify this answer: the bolt preload is 12.5 kN", domain="mechanical_engineering")
    verdict = model.verify_sync("12.5 kN", task="What is the bolt preload?")   # Verifier protocol

Importing the package registers ``AnswerVerificationModel`` with
:func:`pipeline_core.register` under the name ``answer_verification``.
"""
from __future__ import annotations

from .claims import Claim, ClaimVerdict, citation_metrics, grounding_metrics, lexical_support, split_claims
from .config import ALL_VERIFIERS, AnswerVerificationConfig, CalibrationParams
from .evaluation import EvaluationReport, VerificationExample, brier_score, evaluate, evaluate_sync, expected_calibration_error, synthetic_examples
from .model import AnswerVerificationModel, VerificationReport, coerce_evidence
from .plan import (
    HeuristicPlanner,
    LLMPlan,
    TaskFeatures,
    Thresholds,
    VerificationPlan,
    classify_answer_type,
    effective_accept_threshold,
    extract_features,
    fit_to_budget,
    looks_like_code,
)
from .policy import VerificationPolicy, apply_policy, policy_for_profile
from .uncertainty import (
    AgreementResult,
    VerbalizedConfidence,
    aggregate_verdicts,
    canonicalize_answer,
    extract_number,
    fit_platt,
    majority_stability,
    platt_scale,
    semantic_agreement,
    should_stop_sampling,
    verbalized_confidence,
)
from .units import UnitReport, check_units, convert, parse_quantities, unit_family
from .verifiers import (
    BaseVerifier,
    ChainOfVerificationVerifier,
    CodeExecutionVerifier,
    DebateVerifier,
    GroundingVerifier,
    RubricJudgeVerifier,
    RuleVerifier,
    SelfConsistencyVerifier,
    VerifyInputs,
    make_verifiers,
    run_rule_checks,
)

__version__ = "1.0.0"

__all__ = [
    "ALL_VERIFIERS",
    "AgreementResult",
    "AnswerVerificationConfig",
    "AnswerVerificationModel",
    "BaseVerifier",
    "CalibrationParams",
    "ChainOfVerificationVerifier",
    "Claim",
    "ClaimVerdict",
    "CodeExecutionVerifier",
    "DebateVerifier",
    "EvaluationReport",
    "GroundingVerifier",
    "HeuristicPlanner",
    "LLMPlan",
    "RubricJudgeVerifier",
    "RuleVerifier",
    "SelfConsistencyVerifier",
    "TaskFeatures",
    "Thresholds",
    "UnitReport",
    "VerbalizedConfidence",
    "VerificationExample",
    "VerificationPlan",
    "VerificationPolicy",
    "VerificationReport",
    "VerifyInputs",
    "aggregate_verdicts",
    "apply_policy",
    "brier_score",
    "canonicalize_answer",
    "check_units",
    "citation_metrics",
    "classify_answer_type",
    "coerce_evidence",
    "convert",
    "effective_accept_threshold",
    "evaluate",
    "evaluate_sync",
    "expected_calibration_error",
    "extract_features",
    "extract_number",
    "fit_platt",
    "fit_to_budget",
    "grounding_metrics",
    "lexical_support",
    "looks_like_code",
    "majority_stability",
    "make_verifiers",
    "parse_quantities",
    "platt_scale",
    "policy_for_profile",
    "run_rule_checks",
    "semantic_agreement",
    "should_stop_sampling",
    "split_claims",
    "synthetic_examples",
    "unit_family",
    "verbalized_confidence",
]
