"""Verification plan schema and the offline :class:`HeuristicPlanner`.

The plan is what ``plan()`` decides from the natural-language task, the
candidate answer and the domain profile: which verifiers to run, how many
samples to draw, which thresholds gate acceptance. Online it is produced by
one structured LLM call (see :mod:`.model`); offline, or when the request is
deterministic, by :class:`HeuristicPlanner`.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Constraints, DomainProfile, TaskRequest

from .config import ALL_VERIFIERS, AnswerVerificationConfig

AnswerType = Literal["factual", "numerical", "code", "procedural", "design", "free_text"]


class Thresholds(BaseModel):
    """Acceptance thresholds; all are probabilities or fractions in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    accept: float = Field(default=0.6, ge=0.0, le=1.0, description="Fused confidence needed to accept the answer")
    escalate: float = Field(default=0.5, ge=0.0, le=1.0, description="Below this the answer is abstained / escalated to a human")
    min_agreement: float = Field(default=0.6, ge=0.0, le=1.0, description="Self-consistency agreement with the candidate")
    min_supported_fraction: float = Field(default=0.9, ge=0.0, le=1.0, description="Grounding: supported claims / claims")
    min_rubric_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Rubric judge normalised score")


class VerificationPlan(BaseModel):
    """What the model decided to do for one request."""

    model_config = ConfigDict(extra="forbid")

    answer_type: AnswerType = Field(description="Dominant type of the candidate answer")
    verifiers: list[str] = Field(description="Ordered verifier names to run", min_length=1)
    samples: int = Field(default=5, ge=1, le=64, description="k_max for self-consistency / candidates for USC")
    thresholds: Thresholds = Field(default_factory=Thresholds)
    use_evidence: bool = Field(default=True, description="Use request.context / inputs evidence (and the retriever) for grounding")
    allow_revision: bool = Field(default=True, description="Permit evidence-backed minimal revisions")
    judge_criteria: list[str] = Field(default_factory=list, description="Rubric criteria for the judge")
    mandatory: list[str] = Field(default_factory=list, description="Verifiers whose failure forces NEEDS_REVIEW")
    final_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(default="high", description="Effort for the final judgement calls")
    rationale: str = Field(default="", description="Why this plan")
    estimated_calls: int = Field(default=0, ge=0, description="Rough LLM-call estimate for budgeting")
    source: Literal["heuristic", "llm"] = "heuristic"

    def with_verifiers(self, verifiers: list[str]) -> "VerificationPlan":
        return self.model_copy(update={"verifiers": verifiers})


class LLMPlan(BaseModel):
    """Schema the online planner fills in; policy and budget fitting complete it."""

    model_config = ConfigDict(extra="forbid")

    answer_type: AnswerType
    verifiers: list[str] = Field(description="Subset of: self_consistency, chain_of_verification, grounding, rubric_judge, rules, code_execution, debate")
    samples: int = Field(description="Independent samples to draw (1-40); clamped by the model")
    use_evidence: bool
    allow_revision: bool
    judge_criteria: list[str]
    rationale: str


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

#: Strong code markers: a fence, a Python/JS definition, an SQL statement (upper-case keywords), an arrow function or a
#: statement terminator at the end of a line. Any one of these marks the answer as code. Case-sensitive on purpose:
#: "Select the pump from Table 3" and "Class A fire" are prose.
_CODE_STRONG = re.compile(r"```|\bdef \w+\(|\bclass \w+[:(]|\bSELECT\b.+\bFROM\b|=>|;\s*$", re.M)
#: Weak code markers: a line that *starts* with a lower-case ``import``/``from``/``return`` statement. Prose sentences
#: start with a capital ("The pump will return to service", "Import duties are 5 percent"), so the anchor and the case
#: keep ordinary English out; two weak markers are needed to call an unfenced snippet code.
_CODE_WEAK = re.compile(r"^[ \t]*(?:import \w+|from \w+(?:\.\w+)* import \w+|return\b)", re.M)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", re.I)
_UNIT_AFTER_NUMBER = re.compile(r"\d\s?(?:[a-zA-Zµ°%][a-zA-Z/²³0-9°]*)")
_NUMERIC_WORDS = re.compile(
    r"\b(calculate|compute|how (?:much|many)|value|stress|load|force|tolerance|voltage|current|power|rate|ratio|percent|"
    r"estimate|convert|sum|total|average|mean|margin|factor of safety|FoS|budget|capacity|dimension|size|"
    r"pressure|temperature|thickness|diameter|length|width|height|flow|speed|torque|frequency|resistance|mass|weight|"
    r"setting|set ?point|limit|allowance|clearance|preload|efficiency)\b",
    re.I,
)
_PROCEDURE_WORDS = re.compile(r"\b(step|steps|procedure|how to|install|configure|checklist|instructions|process|runbook|workflow)\b", re.I)
_DESIGN_WORDS = re.compile(
    r"\b(design|architecture|architect|trade-?off|approach|propose|recommend|strategy|option|alternative|rationale|should we|evaluate)\b", re.I
)
_FACT_QUESTION = re.compile(r"^\s*(what|which|who|when|where|is|are|does|do|did|was|were|can|has|have)\b", re.I)
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+", re.M)
_CITATION = re.compile(r"\[[\w\-:]+\]")


class TaskFeatures(BaseModel):
    """Cheap, deterministic features of (task, answer) used by the heuristic planner."""

    model_config = ConfigDict(extra="forbid")

    task_words: int
    answer_words: int
    answer_sentences: int
    numbers_in_answer: int
    units_in_answer: int
    code_markers: int = Field(description="Strong + weak code markers (see classify_answer_type)")
    strong_code_markers: int = Field(default=0, description="Fences, def/class, SQL SELECT ... FROM, =>, line-final ';'")
    list_items: int
    citations: int
    numeric_words: int
    procedure_words: int
    design_words: int
    fact_question: bool
    has_evidence: bool
    has_tests: bool
    has_reference: bool


def code_marker_counts(text: str) -> tuple[int, int]:
    """``(strong, weak)`` code-marker counts of ``text`` (see :data:`_CODE_STRONG` / :data:`_CODE_WEAK`)."""
    return len(_CODE_STRONG.findall(text)), len(_CODE_WEAK.findall(text))


def looks_like_code(text: str) -> bool:
    """The rule the planner uses to call an answer ``code``: one strong marker or two weak ones."""
    strong, weak = code_marker_counts(text)
    return strong >= 1 or weak >= 2


def extract_features(task: str, answer: str, *, evidence_count: int = 0, has_tests: bool = False, has_reference: bool = False) -> TaskFeatures:
    """Compute :class:`TaskFeatures` from text alone (no LLM)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()) if s.strip()]
    combined = f"{task}\n{answer}"
    body = _LIST_MARKER.sub("", answer)  # "1." / "Step 2." markers are not quantities
    body = re.sub(r"\b[Ss]tep\s+\d+\b", "step", body)
    strong, weak = code_marker_counts(answer)
    return TaskFeatures(
        task_words=len(task.split()),
        answer_words=len(answer.split()),
        answer_sentences=len(sentences),
        numbers_in_answer=len(_NUMBER.findall(body)),
        units_in_answer=len(_UNIT_AFTER_NUMBER.findall(answer)),
        code_markers=strong + weak,
        strong_code_markers=strong,
        list_items=len(_LIST_MARKER.findall(answer)),
        citations=len(_CITATION.findall(answer)),
        numeric_words=len(_NUMERIC_WORDS.findall(combined)),
        procedure_words=len(_PROCEDURE_WORDS.findall(combined)),
        design_words=len(_DESIGN_WORDS.findall(task)),
        fact_question=bool(_FACT_QUESTION.match(task)),
        has_evidence=evidence_count > 0,
        has_tests=has_tests,
        has_reference=has_reference,
    )


def classify_answer_type(f: TaskFeatures) -> AnswerType:
    """Decision rules mapping features to an :data:`AnswerType` (documented in the README)."""
    weak = f.code_markers - f.strong_code_markers
    if f.strong_code_markers >= 2 or (f.strong_code_markers >= 1 and f.answer_words < 400) or weak >= 2:
        return "code"
    if f.numbers_in_answer >= 1 and (f.units_in_answer >= 1 or f.numeric_words >= 1) and f.answer_words <= 250:
        return "numerical"
    if f.procedure_words >= 1 and (f.list_items >= 2 or f.answer_sentences >= 3):
        return "procedural"
    if f.design_words >= 1 and f.answer_words >= 40:
        return "design"
    if f.answer_words <= 60 and (f.fact_question or f.answer_sentences <= 2):
        return "factual"
    return "free_text"


def estimate_calls(
    plan_verifiers: list[str], samples: int, cfg: AnswerVerificationConfig, *, claims_hint: int = 8, answer_type: str = "free_text"
) -> int:
    """Rough LLM-call cost of a plan, used for budget fitting and reporting."""
    total = 0
    free_form = answer_type in ("free_text", "design", "procedural")
    for v in plan_verifiers:
        if v == "self_consistency":
            total += samples
            if free_form:
                total += cfg.usc_selection_passes + (1 if cfg.llm_equivalence_clustering else 0)
        elif v == "chain_of_verification":
            total += 2 + min(cfg.cove_max_questions, claims_hint) + (1 if cfg.allow_revision else 0)
        elif v == "grounding":
            total += 2 + (1 if cfg.allow_revision else 0)
        elif v == "rubric_judge":
            total += cfg.judge_samples
        elif v == "debate":
            total += cfg.debate_agents * cfg.debate_rounds + 1
        # rules and code_execution cost no LLM calls
    if cfg.elicit_verbalized_confidence:
        total += 1
    return total


def fit_to_budget(
    plan: VerificationPlan, constraints: Constraints, cfg: AnswerVerificationConfig, *, planning_calls: int = 1
) -> VerificationPlan:
    """Degrade samples first, then drop the cheapest-value verifiers, until the plan fits ``max_llm_calls``.

    ``planning_calls`` is the number of calls already spent (or reserved) for
    planning. Mandatory verifiers are never dropped (their failure to run is
    reported instead). ``rules`` and ``code_execution`` cost no LLM calls.
    """
    limit = constraints.max_llm_calls
    est = lambda v, s: estimate_calls(v, s, cfg, answer_type=plan.answer_type)  # noqa: E731
    if limit is None:
        return plan.model_copy(update={"estimated_calls": est(plan.verifiers, plan.samples)})
    budget = max(0, limit - planning_calls)
    verifiers = list(plan.verifiers)
    samples = plan.samples
    drop_order = ["debate", "chain_of_verification", "rubric_judge", "self_consistency", "grounding"]
    while est(verifiers, samples) > budget:
        if samples > cfg.samples_min and "self_consistency" in verifiers:
            samples = max(cfg.samples_min, samples - 2)
            continue
        if samples > 1 and "self_consistency" in verifiers and est([v for v in verifiers if v != "self_consistency"], 1) <= budget - 1:
            samples = max(1, samples - 1)
            continue
        candidates = [v for v in drop_order if v in verifiers and v not in plan.mandatory]
        if not candidates:
            break
        verifiers.remove(candidates[0])
    if not verifiers:
        verifiers = ["rules"]
    return plan.model_copy(update={"verifiers": verifiers, "samples": samples, "estimated_calls": est(verifiers, samples)})


def effective_accept_threshold(cfg: AnswerVerificationConfig, profile: DomainProfile, constraints: Constraints) -> float:
    """The acceptance threshold ``execute()`` enforces: ``max(release_threshold, profile.min_confidence_to_accept, constraints.min_confidence)``.

    ``release_threshold`` is an extra floor, never a way to accept below the domain profile's minimum, so the plan
    reports the value that is actually enforced.
    """
    return max(cfg.release_threshold or 0.0, profile.min_confidence_to_accept, constraints.min_confidence)


def sanitize_verifiers(names: list[str], *, allow_code_execution: bool) -> list[str]:
    """Keep known verifier names, in order, without duplicates; drop code execution when not allowed."""
    out: list[str] = []
    for n in names:
        n = n.strip().lower().replace("-", "_").replace(" ", "_")
        if n in ALL_VERIFIERS and n not in out:
            if n == "code_execution" and not allow_code_execution:
                continue
            out.append(n)
    return out


class HeuristicPlanner:
    """Offline planner: features + domain profile -> :class:`VerificationPlan`.

    Decision rules (also listed in the README):

    * ``numerical`` -> self-consistency (vote on canonical numbers) + rules (units); grounding if evidence.
    * ``code`` -> rules + code execution when allowed and tests exist, else USC self-consistency; rubric judge.
    * ``factual`` -> self-consistency; grounding when evidence exists.
    * ``procedural`` -> rubric judge; grounding when evidence exists; chain of verification for long answers.
    * ``design`` -> rubric judge; debate only for high-risk domains with budget; grounding when evidence exists.
    * ``free_text`` -> grounding when evidence exists, chain of verification, rubric judge.
    * Risk ``high`` adds rules; ``safety_critical`` makes grounding + rules mandatory and raises thresholds.
    * ``samples``: 5 by default, 3 when deterministic or tightly budgeted, up to ``samples_max`` for high-stakes.
    """

    def __init__(self, config: AnswerVerificationConfig | None = None) -> None:
        self.config = config or AnswerVerificationConfig()

    def plan(
        self,
        request: TaskRequest,
        profile: DomainProfile,
        *,
        answer: str,
        evidence_count: int = 0,
    ) -> VerificationPlan:
        cfg = self.config
        inputs = request.inputs
        has_tests = bool(inputs.get("tests"))
        has_reference = bool(inputs.get("reference"))
        f = extract_features(request.task, answer, evidence_count=evidence_count, has_tests=has_tests, has_reference=has_reference)
        answer_type = classify_answer_type(f)
        risk = profile.risk_level
        high_stakes = risk in ("high", "safety_critical")
        allow_code = request.constraints.allow_code_execution

        verifiers: list[str] = []
        reasons: list[str] = []
        if answer_type == "numerical":
            verifiers += ["self_consistency", "rules"]
            reasons.append("numerical answer: majority vote over independent derivations and unit/range rules")
        elif answer_type == "code":
            verifiers.append("rules")
            if allow_code and has_tests:
                verifiers.append("code_execution")
                reasons.append("code with tests: execution is the oracle")
            else:
                verifiers.append("self_consistency")
                reasons.append("code without executable tests: consensus selection over candidates")
            verifiers.append("rubric_judge")
        elif answer_type == "factual":
            verifiers.append("self_consistency")
            reasons.append("short factual answer: agreement across re-derivations")
        elif answer_type == "procedural":
            verifiers.append("rubric_judge")
            if f.answer_sentences >= 4:
                verifiers.append("chain_of_verification")
            reasons.append("procedural answer: rubric for completeness/order, verification questions for factual steps")
        elif answer_type == "design":
            verifiers.append("rubric_judge")
            if high_stakes and (request.constraints.max_llm_calls is None or request.constraints.max_llm_calls >= 12):
                verifiers.append("debate")
            reasons.append("design answer: rubric judgement; bounded debate for high-risk domains")
        else:  # free_text
            verifiers += ["chain_of_verification", "rubric_judge"]
            reasons.append("free text: factored verification questions plus rubric judgement")

        if f.has_evidence:
            verifiers.insert(0, "grounding")
            reasons.append("evidence available: atomic-claim grounding is the primary external signal")
        if high_stakes and "rules" not in verifiers:
            verifiers.append("rules")
            reasons.append(f"risk level {risk}: domain rules enforced")

        mandatory: list[str] = []
        thresholds = Thresholds(
            accept=effective_accept_threshold(cfg, profile, request.constraints),
            escalate=cfg.escalate_below,
            min_agreement=cfg.low_agreement_threshold,
            min_supported_fraction=cfg.grounding_min_supported,
            min_rubric_score=cfg.judge_min_score,
        )
        if risk == "safety_critical":
            mandatory = [v for v in ("grounding", "rules") if v in verifiers] or ["rules"]
            if "rules" not in verifiers:
                verifiers.append("rules")
            thresholds = thresholds.model_copy(
                update={
                    "accept": max(thresholds.accept, 0.85),
                    "min_supported_fraction": cfg.grounding_min_supported_safety_critical,
                    "min_agreement": max(thresholds.min_agreement, 0.8),
                }
            )
            reasons.append("safety-critical: grounding and rules mandatory, no contradictions tolerated, accept >= 0.85")
        thresholds = thresholds.model_copy(update={"accept": max(thresholds.accept, request.constraints.min_confidence)})

        samples = cfg.samples_default
        if request.constraints.deterministic:
            samples = min(samples, cfg.samples_min)
        elif high_stakes and answer_type in ("numerical", "factual"):
            samples = cfg.samples_max
        if answer_type in ("free_text", "design", "procedural"):
            samples = min(samples, cfg.usc_candidates)

        criteria = list(cfg.rubric_criteria) + list(profile.verification_rules)
        final_effort = "max" if risk == "safety_critical" else "high"
        plan = VerificationPlan(
            answer_type=answer_type,
            verifiers=sanitize_verifiers(verifiers, allow_code_execution=allow_code),
            samples=samples,
            thresholds=thresholds,
            use_evidence=True,
            allow_revision=cfg.allow_revision and (f.has_evidence or has_tests),
            judge_criteria=criteria,
            mandatory=mandatory,
            final_effort=final_effort,
            rationale="; ".join(reasons),
            source="heuristic",
        )
        return fit_to_budget(plan, request.constraints, cfg, planning_calls=0)
