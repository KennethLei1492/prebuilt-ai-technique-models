"""Verification policy: domain-derived defaults layered over a plan.

A :class:`VerificationPolicy` says which verifiers are mandatory, how they are
weighted, and what thresholds gate acceptance. :func:`policy_for_profile`
derives one from a :class:`pipeline_core.DomainProfile` so that a newly
registered domain changes behaviour without code edits; :func:`apply_policy`
enforces it on any plan (heuristic or LLM-produced).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import DomainProfile

from .config import AnswerVerificationConfig
from .plan import Thresholds, VerificationPlan


class VerificationPolicy(BaseModel):
    """Domain posture for verification."""

    model_config = ConfigDict(extra="forbid")

    verifiers: list[str] = Field(default_factory=list, description="Verifiers the policy adds if the plan lacks them")
    mandatory: list[str] = Field(default_factory=list, description="Verifiers whose failure forces NEEDS_REVIEW")
    weights: dict[str, float] = Field(default_factory=dict)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    escalate_below: float = Field(default=0.5, ge=0.0, le=1.0)
    allow_revision: bool = True
    max_revision_rounds: int = Field(default=2, ge=0, le=4)
    require_no_contradictions: bool = Field(default=False, description="Any contradicted claim blocks acceptance")
    final_effort: str = "high"
    risk_level: str = "medium"
    description: str = ""


def policy_for_profile(profile: DomainProfile, config: Optional[AnswerVerificationConfig] = None) -> VerificationPolicy:
    """Derive the policy from the domain profile.

    * ``low``/``medium`` risk: thresholds from the profile; nothing mandatory.
    * ``high``: rules added; grounding mandatory when evidence exists (enforced at run time).
    * ``safety_critical``: grounding and rules mandatory, no contradictions, accept >= 0.85,
      supported fraction 1.0, final judgement at effort ``max``.
    """
    cfg = config or AnswerVerificationConfig()
    # release_threshold is an extra floor: it can raise the profile's acceptance bar, never lower it
    accept = max(cfg.release_threshold or 0.0, profile.min_confidence_to_accept)
    thresholds = Thresholds(
        accept=accept,
        escalate=cfg.escalate_below,
        min_agreement=cfg.low_agreement_threshold,
        min_supported_fraction=cfg.grounding_min_supported,
        min_rubric_score=cfg.judge_min_score,
    )
    risk = profile.risk_level
    if risk == "safety_critical":
        return VerificationPolicy(
            verifiers=["grounding", "rules"],
            mandatory=["grounding", "rules"],
            weights=dict(cfg.weights),
            thresholds=thresholds.model_copy(
                update={
                    "accept": max(accept, 0.85),
                    "min_supported_fraction": cfg.grounding_min_supported_safety_critical,
                    "min_agreement": max(cfg.low_agreement_threshold, 0.8),
                }
            ),
            escalate_below=max(cfg.escalate_below, 0.6),
            allow_revision=cfg.allow_revision,
            max_revision_rounds=min(cfg.max_revision_rounds, 1),
            require_no_contradictions=True,
            final_effort="max",
            risk_level=risk,
            description="safety-critical: external evidence mandatory, zero tolerance for contradictions",
        )
    if risk == "high":
        return VerificationPolicy(
            verifiers=["rules"],
            mandatory=["rules"],
            weights=dict(cfg.weights),
            thresholds=thresholds,
            escalate_below=cfg.escalate_below,
            allow_revision=cfg.allow_revision,
            max_revision_rounds=cfg.max_revision_rounds,
            require_no_contradictions=True,
            final_effort="high",
            risk_level=risk,
            description="high risk: rules enforced, contradictions block acceptance",
        )
    return VerificationPolicy(
        verifiers=[],
        mandatory=[],
        weights=dict(cfg.weights),
        thresholds=thresholds,
        escalate_below=cfg.escalate_below,
        allow_revision=cfg.allow_revision,
        max_revision_rounds=cfg.max_revision_rounds,
        require_no_contradictions=False,
        final_effort="high",
        risk_level=risk,
        description="general posture: thresholds from the domain profile",
    )


def apply_policy(plan: VerificationPlan, policy: VerificationPolicy, *, has_evidence: bool) -> VerificationPlan:
    """Return ``plan`` with the policy's mandatory verifiers and floors enforced.

    Grounding is only added/mandated when evidence exists; the policy cannot
    conjure evidence, so without it the requirement is recorded as a warning by
    the model instead.
    """
    verifiers = list(plan.verifiers)
    mandatory = list(plan.mandatory)
    for v in policy.verifiers:
        if v == "grounding" and not has_evidence:
            continue
        if v not in verifiers:
            verifiers.append(v)
    for v in policy.mandatory:
        if v == "grounding" and not has_evidence:
            continue
        if v not in mandatory:
            mandatory.append(v)
        if v not in verifiers:
            verifiers.append(v)
    t = plan.thresholds
    thresholds = t.model_copy(
        update={
            "accept": max(t.accept, policy.thresholds.accept),
            "escalate": max(t.escalate, policy.escalate_below),
            "min_agreement": max(t.min_agreement, policy.thresholds.min_agreement),
            "min_supported_fraction": max(t.min_supported_fraction, policy.thresholds.min_supported_fraction),
            "min_rubric_score": max(t.min_rubric_score, policy.thresholds.min_rubric_score),
        }
    )
    efforts = ["low", "medium", "high", "xhigh", "max"]
    final_effort = plan.final_effort
    if efforts.index(policy.final_effort) > efforts.index(final_effort):
        final_effort = policy.final_effort  # type: ignore[assignment]
    return plan.model_copy(
        update={
            "verifiers": verifiers,
            "mandatory": mandatory,
            "thresholds": thresholds,
            "allow_revision": plan.allow_revision and policy.allow_revision,
            "final_effort": final_effort,
        }
    )
