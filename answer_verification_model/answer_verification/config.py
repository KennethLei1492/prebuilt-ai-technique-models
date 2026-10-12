"""Configuration for the answer verification model.

Every tunable lives here with a default taken from the verification
literature (see the README configuration table for the citation behind each
value). Nothing in the package hard-codes a domain: domain behaviour comes
from :class:`pipeline_core.DomainProfile` through :mod:`.policy`.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

VerifierName = Literal[
    "self_consistency",
    "chain_of_verification",
    "grounding",
    "rubric_judge",
    "rules",
    "code_execution",
    "debate",
]

ALL_VERIFIERS: tuple[str, ...] = (
    "self_consistency",
    "chain_of_verification",
    "grounding",
    "rubric_judge",
    "rules",
    "code_execution",
    "debate",
)

LLMMode = Literal["auto", "force", "never"]


class CalibrationParams(BaseModel):
    """Stored Platt-scaling parameters ``sigmoid(a * logit(p) + b)``.

    The identity (``a=1, b=0``) is the default until a labelled calibration set
    is available; :func:`answer_verification.uncertainty.fit_platt` fits them.
    """

    model_config = ConfigDict(extra="forbid")

    a: float = 1.0
    b: float = 0.0


class AnswerVerificationConfig(BaseModel):
    """All tunables of :class:`~answer_verification.model.AnswerVerificationModel`."""

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ general
    default_domain: str = Field(default="general", description="Domain used by the direct ``verify()`` API")
    llm_mode: LLMMode = Field(
        default="auto",
        description="auto: use the LLM unless ctx.offline; force: always call the LLM (ScriptedLLM tests, replay); never: heuristics only",
    )
    max_concurrency: int = Field(default=4, ge=1, le=64, description="Fan-out ceiling when no Executor is injected")
    plan_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    grade_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low", description="Effort for per-item grading/classification calls"
    )
    derive_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="medium", description="Effort for independent re-derivations and verification answers"
    )
    judge_model: Optional[str] = Field(
        default=None,
        description="Optional model id for judge calls; the literature recommends a different family from the generator",
    )

    # -------------------------------------------------- self-consistency
    samples_default: int = Field(default=5, ge=1, le=64, description="Default k_max for self-consistency")
    samples_min: int = Field(default=3, ge=1, le=64, description="First batch size before adaptive stopping is consulted")
    samples_max: int = Field(default=10, ge=1, le=64, description="Ceiling for high-stakes queries")
    sample_batch: int = Field(default=3, ge=1, le=16, description="Batch size for adaptive sampling")
    stability_threshold: float = Field(default=0.95, ge=0.5, le=1.0, description="Posterior P(majority is stable) that stops sampling")
    low_agreement_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Agreement below this flags the answer as uncertain")
    usc_candidates: int = Field(default=5, ge=2, le=16, description="Candidates shown to the Universal Self-Consistency selector")
    usc_selection_passes: int = Field(default=2, ge=1, le=5, description="Shuffled selection passes; modal choice wins")
    equivalence_jaccard: float = Field(default=0.6, ge=0.0, le=1.0, description="Lexical Jaccard at which two free-text answers are clustered together")
    llm_equivalence_clustering: bool = Field(
        default=True, description="For free-form answers, spend one call on LLM-judged bidirectional-entailment clustering (semantic entropy)"
    )

    # ------------------------------------------- chain of verification
    cove_max_questions: int = Field(default=10, ge=1, le=30)
    allow_revision: bool = Field(default=True, description="Allow evidence-backed minimal revisions")
    max_revision_rounds: int = Field(default=2, ge=0, le=4)
    revision_min_preservation: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum similarity ratio for an accepted revision")

    # --------------------------------------------------------- grounding
    grounding_evidence_per_claim: int = Field(default=5, ge=1, le=20)
    grounding_max_claims: int = Field(default=15, ge=1, le=100)
    grounding_min_supported: float = Field(default=0.9, ge=0.0, le=1.0, description="Supported fraction required to pass")
    grounding_min_supported_safety_critical: float = Field(default=1.0, ge=0.0, le=1.0)
    lexical_support_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Content-token overlap for the offline lexical support heuristic")

    # ------------------------------------------------------ rubric judge
    rubric_criteria: list[str] = Field(
        default_factory=lambda: ["correctness", "completeness", "groundedness", "clarity"],
        description="Default G-Eval style criteria; domain verification_rules are appended",
    )
    rubric_scale: int = Field(default=5, ge=2, le=10)
    judge_samples: int = Field(default=3, ge=1, le=10, description="Independent judge passes averaged (varied framing, not temperature)")
    judge_min_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Normalised rubric score required to pass")
    pairwise_swap: bool = Field(default=True, description="Evaluate both orderings and count only consistent wins")

    # ------------------------------------------------------------ debate
    debate_agents: int = Field(default=3, ge=2, le=6)
    debate_rounds: int = Field(default=2, ge=1, le=4)
    debate_max_rounds: int = Field(default=4, ge=1, le=6)
    debate_adaptive_stop: bool = Field(default=True, description="Stop the debate once all reviewers agree")

    # --------------------------------------------------------- retrieval
    retrieve_k: int = Field(default=8, ge=1, le=50, description="Evidence fetched from an injected Retriever when the request carries none")

    # ---------------------------------------------------- code execution
    code_timeout_s: float = Field(default=30.0, gt=0, le=600)
    code_max_output_chars: int = Field(default=4000, ge=200)

    # ------------------------------------------------- confidence fusion
    elicit_verbalized_confidence: bool = Field(default=True, description="Add a one-call top-k verbalised confidence signal when online")
    verbalized_top_k: int = Field(default=4, ge=1, le=10)
    verbalized_weight: float = Field(default=0.5, ge=0.0, le=5.0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "code_execution": 2.0,
            "rules": 1.5,
            "grounding": 1.5,
            "self_consistency": 1.0,
            "chain_of_verification": 1.0,
            "rubric_judge": 0.75,
            "debate": 0.5,
        },
        description="Fusion weights; external-evidence verifiers outrank LLM judgement",
    )
    release_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Extra acceptance floor. The enforced threshold is max(release_threshold, DomainProfile.min_confidence_to_accept, "
            "constraints.min_confidence) (0.85 or more for safety-critical profiles); a value below the profile minimum has no effect. "
            "None applies no extra floor."
        ),
    )
    escalate_below: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence below which the answer is abstained/escalated")
    calibration: CalibrationParams = Field(default_factory=CalibrationParams)

    # --------------------------------------------------------- behaviour
    draft_when_missing: bool = Field(
        default=True, description="If request.inputs lacks 'answer', draft one with the LLM and verify it (CLI convenience)"
    )
    max_answer_chars: int = Field(default=20000, ge=200)
