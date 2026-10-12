"""Decomposition plan schema and the heuristic planner.

``plan()`` answers one question: *how* should this task be decomposed? The
answer is a :class:`DecompositionPlan`. Online it comes from one structured
LLM call; offline (``ctx.offline``) or when ``constraints.deterministic`` is
set it comes from :class:`HeuristicPlanner`, which reads cheap task features
and the domain profile.

The decision rules encode the dossier's strategy selector
(architecture recommendation 1): estimated step depth, independence of
sub-goals, need for environment feedback, verifiability of intermediate states
and the risk posture of the domain.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline_core import Constraints, DomainProfile, Evidence, tokenize_words

from .config import TaskDecompositionConfig

Strategy = Literal["direct", "plan_and_solve", "least_to_most", "dag", "recursive", "tree_of_thoughts"]
Verification = Literal["none", "criteria", "external", "strict"]
ComplexityEstimate = Literal["trivial", "low", "medium", "high"]

STRATEGIES: tuple[str, ...] = ("direct", "plan_and_solve", "least_to_most", "dag", "recursive", "tree_of_thoughts")


class DecompositionPlan(BaseModel):
    """How the task will be decomposed and executed.

    Flat and enum-typed so it doubles as the structured-output schema for the
    LLM planner (no numeric bounds in the schema; values are clamped on
    validation instead).
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Strategy = Field(..., description="direct | plan_and_solve | least_to_most | dag | recursive | tree_of_thoughts")
    max_depth: int = Field(..., description="Recursion / wave depth bound for this run")
    max_breadth: int = Field(..., description="Maximum sub-tasks per level")
    parallelism: int = Field(..., description="Concurrent sub-tasks per wave")
    verification: Verification = Field(..., description="none | criteria | external | strict")
    complexity_estimate: ComplexityEstimate = Field(..., description="trivial | low | medium | high")
    estimated_steps: int = Field(..., description="Estimated number of reasoning or action steps")
    replan_on_failure: bool = Field(..., description="Allow bounded replanning when a sub-task fails")
    rationale: str = Field(..., description="One or two sentences on why this strategy fits")

    @field_validator("max_depth", "max_breadth", "parallelism", "estimated_steps", mode="before")
    @classmethod
    def _at_least_one(cls, v: Any) -> int:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return 1
        return max(1, min(iv, 64))

    def is_parallel(self) -> bool:
        return self.strategy in ("dag", "tree_of_thoughts") and self.parallelism > 1

    def describe(self) -> str:
        return (
            f"strategy={self.strategy} steps~{self.estimated_steps} depth<={self.max_depth} breadth<={self.max_breadth} "
            f"parallelism={self.parallelism} verification={self.verification} complexity={self.complexity_estimate}\n"
            f"rationale: {self.rationale}"
        )


# ------------------------------------------------------------------ features

_SEQUENCE_MARKERS = (
    "then", "after that", "afterwards", "next", "finally", "once", "before", "after", "based on", "using the result",
    "using the results", "using those", "with the result", "with the results", "subsequently", "followed by", "in order to",
    "first,", "second,", "third,", "lastly",
)
_INDEPENDENCE_MARKERS = (
    "each", "every", "independently", "separately", "for all", "in parallel", "respectively", "per", "each of", "for every",
)
_EXPLORATION_MARKERS = (
    "alternatives", "alternative", "options", "trade-off", "tradeoff", "trade off", "compare", "best ordering", "optimal order",
    "which design", "choose between", "candidate", "candidates", "explore", "design space", "rank the", "select the best", "sequence of operations",
)
_INVESTIGATION_MARKERS = (
    "debug", "diagnose", "root cause", "investigate", "troubleshoot", "why does", "why is", "fails intermittently", "find the bug",
    "fix the failing", "incident", "failure analysis", "unexpected", "not working",
)
_ACTION_VERBS = (
    "check", "verify", "validate", "compute", "calculate", "list", "identify", "summarise", "summarize", "compare", "review",
    "design", "size", "select", "estimate", "derive", "write", "draft", "analyse", "analyze", "extract", "map", "assess",
    "evaluate", "determine", "document", "test", "implement", "propose", "recommend", "quantify", "trace", "confirm", "audit",
    "inspect", "measure", "plan", "schedule", "generate", "build", "create", "define", "specify", "update", "migrate", "refactor",
    "explain", "describe", "outline", "prioritise", "prioritize", "classify", "detect", "resolve", "fix", "investigate", "diagnose",
)
_ENUM_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•]|\([a-z]\)|[a-z][.)])\s+")


def _marker_regex(markers: Sequence[str]) -> "re.Pattern[str]":
    """One alternation over whole-word markers, longest first.

    Whole-word matching stops ``each`` matching ``teach``/``reach``, ``per`` matching ``paper`` and ``once``
    matching ``announce``; longest-first alternation counts ``after that`` or ``each of`` once instead of
    twice. ``(?<!\\w)``/``(?!\\w)`` rather than ``\\b`` so markers ending in punctuation (``first,``) work.
    """
    alts = sorted({m.strip() for m in markers if m.strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in alts) + r")(?!\w)", re.IGNORECASE)


_SEQUENCE_RE = _marker_regex(_SEQUENCE_MARKERS)
_INDEPENDENCE_RE = _marker_regex(_INDEPENDENCE_MARKERS)
_EXPLORATION_RE = _marker_regex(_EXPLORATION_MARKERS)
_INVESTIGATION_RE = _marker_regex(_INVESTIGATION_MARKERS)


def count_markers(text: str, markers: Sequence[str]) -> int:
    """Number of non-overlapping whole-word occurrences of any marker in ``text`` (case-insensitive)."""
    return len(_marker_regex(markers).findall(text))
# A comma-separated part that opens with an ordering word is a step of a chain, not an independent item:
# "first compute X, then size Y, then check Z" is a Least-to-Most chain even though it has three parts.
_SEQ_PART_RE = re.compile(
    r"^(?:and\s+)?(?:then|next|finally|lastly|afterwards|after that|subsequently|first|second|third|fourth|fifth|"
    r"using (?:the|those|that)|based on|with (?:the|those|that) results?|once |before |after )",
    re.IGNORECASE,
)


class TaskFeatures(BaseModel):
    """Cheap, deterministic signals extracted from the request."""

    model_config = ConfigDict(extra="forbid")

    word_count: int = 0
    sentence_count: int = 0
    clause_count: int = 0
    conjunctions: int = 0
    sequence_markers: int = 0
    independence_markers: int = 0
    exploration_markers: int = 0
    investigation_markers: int = 0
    enumeration_items: int = 0
    action_verbs: int = 0
    question_count: int = 0
    has_documents: bool = False
    input_keys: list[str] = Field(default_factory=list)
    domain_hint_count: int = 0
    risk_level: str = "low"
    estimated_steps: int = 1


def extract_features(
    task: str,
    *,
    inputs: Optional[dict[str, Any]] = None,
    context: Sequence[Evidence] = (),
    profile: Optional[DomainProfile] = None,
) -> TaskFeatures:
    """Compute :class:`TaskFeatures` for ``task`` without any model call."""
    text = task.strip()
    low = " " + text.lower() + " "
    words = tokenize_words(text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    enumeration = len(_ENUM_RE.findall(text))
    if enumeration == 0:
        # inline enumerations: "a, b, and c" with 3+ comma-separated *independent* items in one sentence;
        # parts that open with an ordering word ("then size the beam") belong to a chain and do not count
        for s in sentences:
            parts = [p.strip() for p in re.split(r",\s*|;\s*", s) if p.strip()]
            independent = [p for p in parts if not _SEQ_PART_RE.match(p)]
            if len(independent) >= 3:
                enumeration = max(enumeration, len(independent))
    clause_splits = re.split(r";|,\s*(?:and|then)\s+|\s+(?:and then|then)\s+|\s+and\s+(?=[a-z]+\s)", text)
    clause_count = max(len(sentences), len([c for c in clause_splits if len(c.split()) >= 2]))
    conj = len(re.findall(r"\b(?:and|then|also|plus|as well as)\b", low))
    seq = len(_SEQUENCE_RE.findall(low))
    indep = len(_INDEPENDENCE_RE.findall(low))
    explore = len(_EXPLORATION_RE.findall(low))
    invest = len(_INVESTIGATION_RE.findall(low))
    verbs = sum(1 for w in words if w in _ACTION_VERBS)
    questions = text.count("?")
    inputs = inputs or {}
    has_docs = bool(context) or any(k in inputs for k in ("documents", "document", "files", "logs", "spec", "specification", "code"))
    hints = len(profile.decomposition_hints) if profile else 0
    risk = profile.risk_level if profile else "low"
    steps = max(1, enumeration, verbs, clause_count, seq + 1)
    if questions > 1:
        steps = max(steps, questions)
    steps = min(steps, 24)
    return TaskFeatures(
        word_count=len(words),
        sentence_count=len(sentences),
        clause_count=clause_count,
        conjunctions=conj,
        sequence_markers=seq,
        independence_markers=indep,
        exploration_markers=explore,
        investigation_markers=invest,
        enumeration_items=enumeration,
        action_verbs=verbs,
        question_count=questions,
        has_documents=has_docs,
        input_keys=sorted(inputs.keys()),
        domain_hint_count=hints,
        risk_level=risk,
        estimated_steps=steps,
    )


# ------------------------------------------------------------- normalisation

_VERIFICATION_RANK: dict[str, int] = {"none": 0, "criteria": 1, "external": 2, "strict": 3}
_RISK_FLOOR: dict[str, str] = {"low": "none", "medium": "criteria", "high": "external", "safety_critical": "strict"}


def features_text(f: TaskFeatures) -> str:
    """Compact rendering of :class:`TaskFeatures` for the planner prompt."""
    return (
        f"steps~{f.estimated_steps} words={f.word_count} enumeration={f.enumeration_items} independence={f.independence_markers} "
        f"sequence={f.sequence_markers} exploration={f.exploration_markers} investigation={f.investigation_markers} "
        f"documents={str(f.has_documents).lower()} inputs={','.join(f.input_keys) or '-'} domain_hints={f.domain_hint_count} risk={f.risk_level}"
    )


def normalize_plan(plan: DecompositionPlan, config: TaskDecompositionConfig, constraints: Constraints, profile: DomainProfile) -> DecompositionPlan:
    """Clamp an LLM- or heuristic-produced plan to the configured and requested bounds.

    * breadth, depth and parallelism are clamped to the config and ``constraints.max_parallelism``;
    * ``tree_of_thoughts`` is downgraded to ``dag`` when search is disabled or the run is deterministic,
      ``recursive`` to ``least_to_most`` under ``deterministic``;
    * single-call strategies get breadth/depth/parallelism 1; non-parallel strategies get parallelism 1;
    * verification is raised to the floor implied by the profile's risk level (safety_critical -> strict,
      high -> external, medium -> criteria); it is never lowered;
    * replanning is disabled when ``max_replans`` is 0.
    """
    p = plan.model_copy()
    if p.strategy == "tree_of_thoughts" and (not config.enable_search or constraints.deterministic):
        p.strategy = "dag"
    if p.strategy == "recursive" and constraints.deterministic:
        p.strategy = "least_to_most"
    p.max_breadth = max(1, min(p.max_breadth, config.max_subtasks))
    p.max_depth = max(1, min(p.max_depth, max(config.max_depth, config.deep_domain_depth)))
    p.parallelism = max(1, min(p.parallelism, config.max_parallelism, constraints.max_parallelism))
    if p.strategy in ("direct", "plan_and_solve"):
        p.max_breadth = p.max_depth = p.parallelism = 1
    elif p.strategy == "least_to_most":
        p.parallelism = 1
        p.max_depth = max(p.max_depth, p.max_breadth)
    floor = _RISK_FLOOR.get(profile.risk_level, "criteria")
    if _VERIFICATION_RANK[p.verification] < _VERIFICATION_RANK[floor]:
        p.verification = floor  # type: ignore[assignment]
    if config.max_replans == 0 or p.strategy == "direct":
        p.replan_on_failure = False
    return p


# ------------------------------------------------------------------- planner


class HeuristicPlanner:
    """Rule-based strategy selection used offline, under ``deterministic`` and as the LLM fallback.

    Rules (in priority order):

    1. **Investigation** markers (debug, root cause, incident) -> ``recursive``
       (ADaPT: try the whole task first, decompose on failure; costs one call
       when the first attempt succeeds, so it needs no damping gate).
    2. **Exploration** markers (alternatives, trade-offs, orderings) on a task of
       8+ words with search enabled -> ``tree_of_thoughts`` (ToT/RAP:
       verifiable/scorable choices).
    3. **Damping** - fewer than ``decompose_min_steps`` estimated steps and a short
       task -> ``direct`` (Least-to-Most is neutral at 2 steps; Anthropic warns
       against over-decomposition).
    4. **Independence** markers or an enumeration of 3+ items that outnumbers
       sequence markers -> ``dag`` (LLMCompiler/ReWOO: parallel frontier).
    5. Sequence markers or 4+ estimated steps -> ``least_to_most``.
    6. Otherwise -> ``plan_and_solve`` (PS+: one planned call).

    ``deterministic`` requests downgrade sampling-heavy strategies
    (``tree_of_thoughts`` -> ``dag``, ``recursive`` -> ``least_to_most``).
    Verification level follows the profile risk: ``safety_critical`` -> strict,
    ``high`` -> external, ``medium`` -> criteria, ``low`` -> criteria when
    decomposing, none otherwise.
    """

    def __init__(self, config: Optional[TaskDecompositionConfig] = None) -> None:
        self.config = config or TaskDecompositionConfig()

    def plan(
        self,
        task: str,
        *,
        profile: DomainProfile,
        constraints: Optional[Constraints] = None,
        inputs: Optional[dict[str, Any]] = None,
        context: Sequence[Evidence] = (),
        features: Optional[TaskFeatures] = None,
        strategy_override: Optional[str] = None,
    ) -> DecompositionPlan:
        cfg = self.config
        constraints = constraints or Constraints()
        f = features or extract_features(task, inputs=inputs, context=context, profile=profile)
        steps = f.estimated_steps
        reasons: list[str] = []

        trivial = steps < cfg.decompose_min_steps and f.word_count < 40 and not f.has_documents and f.enumeration_items < 3
        if strategy_override and strategy_override in STRATEGIES:
            strategy = strategy_override
            reasons.append(f"strategy forced to {strategy}")
        elif f.investigation_markers:
            # ADaPT: try the whole task first (one call when it succeeds), expand only on failure
            strategy = "recursive"
            reasons.append("investigative task: granularity unknown up front, decompose as needed on failure")
        elif f.exploration_markers and f.word_count >= 8:
            strategy = "tree_of_thoughts"
            reasons.append("task asks to compare alternatives or orderings; candidate decompositions are scorable")
        elif trivial:
            strategy = "direct"
            reasons.append(f"~{steps} step(s) and {f.word_count} words: below the decomposition gate")
        elif (f.independence_markers or f.enumeration_items >= 3) and f.enumeration_items > f.sequence_markers:
            strategy = "dag"
            reasons.append(f"{max(f.enumeration_items, f.independence_markers)} independent items detected; run as a parallel DAG")
        elif f.sequence_markers or steps >= 4:
            strategy = "least_to_most"
            reasons.append(f"sequential markers={f.sequence_markers}, ~{steps} steps: solve easy-to-hard with answer carry-forward")
        else:
            strategy = "plan_and_solve"
            reasons.append("moderate task without clear structure: one planned call")

        if strategy == "tree_of_thoughts" and not cfg.enable_search:
            strategy = "dag"
            reasons.append("search disabled in config: downgraded to dag")
        if constraints.deterministic:
            if strategy == "tree_of_thoughts":
                strategy = "dag"
                reasons.append("deterministic constraint: search downgraded to dag")
            elif strategy == "recursive":
                strategy = "least_to_most"
                reasons.append("deterministic constraint: recursion downgraded to least_to_most")

        breadth = max(cfg.min_subtasks, min(cfg.max_subtasks, steps))
        if strategy in ("direct", "plan_and_solve"):
            breadth = 1
        depth = cfg.max_depth
        if steps >= cfg.deep_domain_steps or f.domain_hint_count >= 2:
            depth = max(depth, cfg.deep_domain_depth)
        if strategy in ("direct", "plan_and_solve"):
            depth = 1
        elif strategy == "least_to_most":
            depth = max(depth, breadth)  # a chain is as deep as it is long

        parallelism = 1
        if strategy in ("dag", "tree_of_thoughts", "recursive"):
            parallelism = max(1, min(cfg.max_parallelism, constraints.max_parallelism, breadth))

        risk = profile.risk_level
        if risk == "safety_critical":
            verification: Verification = "strict"
            reasons.append("safety-critical domain: external verification per sub-task and on the final answer")
        elif risk == "high":
            verification = "external"
        elif risk == "medium":
            verification = "criteria"
        else:
            verification = "criteria" if strategy != "direct" else "none"

        if steps <= 2 and f.word_count < 25:
            complexity: ComplexityEstimate = "trivial"
        elif steps <= 3:
            complexity = "low"
        elif steps <= 6:
            complexity = "medium"
        else:
            complexity = "high"
        if f.domain_hint_count:
            reasons.append(f"{f.domain_hint_count} domain decomposition hint(s) will be applied")

        return DecompositionPlan(
            strategy=strategy,  # type: ignore[arg-type]
            max_depth=depth,
            max_breadth=breadth,
            parallelism=parallelism,
            verification=verification,
            complexity_estimate=complexity,
            estimated_steps=steps,
            replan_on_failure=strategy != "direct" and cfg.max_replans > 0,
            rationale="; ".join(reasons),
        )
