"""Plan schema and heuristic planner.

``plan()`` answers one question: *how should this prompt be parallelised?*
The answer is a :class:`ParallelPlan`. Online it comes from one structured
LLM call (schema :class:`PlannerOutput`) that is then constrained by budget,
domain and configuration; offline (``ctx.offline``) or when
``constraints.deterministic`` is set, :class:`HeuristicPlanner` derives it
from task features alone.

The routing table follows the dossier's Amdahl-driven mode selection:
enumerable independent sub-tasks -> sectioning (DAG); point-independent long
form -> skeleton-of-thought; single verifiable answer -> sampling-and-voting;
long input -> map-reduce; evaluation / multi-constraint -> branch-solve-merge;
high-value free form -> layered ensemble; sequentially dependent reasoning ->
no parallelism (SoT-R's finding that unrouted parallelism hurts math, code
and writing).
"""
from __future__ import annotations

import math
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Constraints, DomainProfile, Effort, Evidence, PlanValidationError, TaskRequest, estimate_tokens

from .config import Aggregator, ParallelizationConfig
from .dag import topological_waves, validate_dag
from .estimator import estimate_fan_out

Pattern = Literal[
    "sequential",
    "sectioning",
    "skeleton_of_thought",
    "voting",
    "map_reduce",
    "branch_solve_merge",
    "ensemble",
    "speculative",
]
FailureMode = Literal["fail_fast", "best_effort", "quorum"]
Verifiability = Literal["exact_match", "verifiable", "free_form"]

PATTERNS: tuple[str, ...] = (
    "sequential",
    "sectioning",
    "skeleton_of_thought",
    "voting",
    "map_reduce",
    "branch_solve_merge",
    "ensemble",
    "speculative",
)


# --------------------------------------------------------------------------
# Plan schema
# --------------------------------------------------------------------------


class Partition(BaseModel):
    """One independent unit of work in a sectioning / DAG plan."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Short unique identifier, e.g. p1")
    task: str = Field(description="Self-contained sub-task")
    depends_on: list[str] = Field(default_factory=list, description="Ids of partitions whose output this one needs")


class RateLimits(BaseModel):
    """The *configured* client-side limits (informational; None = not configured).

    Enforcement is done by the executor's :class:`RateLimiter`, which is built
    once (``ParallelExecutor.from_config``) or injected; ``execute()`` warns
    when the executor's limiter disagrees with these values.
    """

    model_config = ConfigDict(extra="forbid")

    requests_per_minute: Optional[int] = None
    input_tokens_per_minute: Optional[int] = None
    output_tokens_per_minute: Optional[int] = None


class ParallelPlan(BaseModel):
    """How a task will be executed. Attached to ``result.metadata['plan']``."""

    model_config = ConfigDict(extra="forbid")

    pattern: Pattern
    partitions: list[Partition] = Field(default_factory=list, description="Sub-tasks for sectioning; empty for other patterns")
    concurrency: int = Field(default=8, ge=1, description="Fan-out width per wave")
    failure_policy: FailureMode = "best_effort"
    quorum: float = Field(default=0.8, gt=0.0, description="Required success fraction (or count when > 1) under quorum policy")
    rate_limits: RateLimits = Field(default_factory=RateLimits)
    samples: int = Field(default=1, ge=1, description="Initial independent samples (voting) or proposers per layer (ensemble)")
    max_samples: int = Field(default=1, ge=1, description="Upper bound after early-stopping checks (voting)")
    layers: int = Field(default=1, ge=1, description="Proposer layers (ensemble)")
    aggregator: Aggregator = Field(default="code", description="code = deterministic merge/vote, llm = merge prompt, usc = LLM-selected consistency, concat = ordered concatenation")
    verifiability: Verifiability = "free_form"
    cache_aware: bool = Field(default=True, description="Warm-then-burst fan-out when the shared prefix is cacheable")
    branch_effort: Effort = "medium"
    aggregator_effort: Effort = "high"
    rationale: str = ""
    expected_speedup: float = Field(default=1.0, ge=0.0)
    expected_calls: int = Field(default=1, ge=0, description="LLM calls the plan will spend in execute()")
    serial_fraction: float = Field(default=1.0, ge=0.0, le=1.0, description="Share of expected calls on the serial critical path (Amdahl bound = 1 / serial_fraction)")
    planner: Literal["heuristic", "llm"] = "heuristic"
    warnings: list[str] = Field(default_factory=list, description="Planning degradations (truncated partitions, folded items); surfaced as run warnings")

    def graph(self) -> dict[str, list[str]]:
        """Dependency graph of the partitions."""
        return {p.id: list(p.depends_on) for p in self.partitions}

    def summary(self) -> str:
        parts = [f"pattern={self.pattern}", f"concurrency={self.concurrency}", f"failure_policy={self.failure_policy}"]
        if self.partitions:
            parts.append(f"partitions={len(self.partitions)}")
        if self.pattern in ("voting", "ensemble"):
            parts.append(f"samples={self.samples}/{self.max_samples}")
        if self.pattern == "ensemble":
            parts.append(f"layers={self.layers}")
        parts.append(f"aggregator={self.aggregator}")
        parts.append(f"expected_speedup={self.expected_speedup:.2f}x")
        parts.append(f"expected_calls={self.expected_calls}")
        parts.append(f"planner={self.planner}")
        return " ".join(parts)


class PlannerOutput(BaseModel):
    """What the LLM planner decides; everything else is derived by :class:`HeuristicPlanner.finalize`."""

    model_config = ConfigDict(extra="forbid")

    pattern: Pattern
    partitions: list[Partition] = Field(default_factory=list, description="Only for sectioning: independent sub-tasks with ids")
    samples: int = Field(default=5, ge=1, le=40, description="Samples for voting / proposers for ensemble")
    aggregator: Aggregator = Field(default="llm")
    verifiability: Verifiability = "free_form"
    rationale: str = Field(description="One or two sentences on why this pattern fits")


# --------------------------------------------------------------------------
# Task features
# --------------------------------------------------------------------------

_SEQUENTIAL_MARKERS = (
    r"\bderiv(e|ation)\b",
    r"\bprove\b",
    r"\bproof\b",
    r"\bstep[- ]by[- ]step\b",
    r"\bprocedure\b",
    r"\bsequence of\b",
    r"\bin order\b",
    r"\bimplement\b",
    r"\bwrite (a |the )?(function|class|module|script|program|code|query)\b",
    r"\brefactor\b",
    r"\bdebug\b",
    r"\bfix the\b",
    r"\bmigrat(e|ion)\b",
    r"\bstory\b",
    r"\bpoem\b",
    r"\bessay\b",
    r"\bnarrative\b",
    r"\bwalk me through\b",
    r"\bthen\b.*\bthen\b",
)
_NUMERIC_MARKERS = (
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bwhat is the (value|result|total|sum|load|stress|force|voltage|current|pressure|flow|mass|factor|ratio)\b",
    r"\bsolve for\b",
    r"\btrue or false\b",
    r"\byes or no\b",
    r"\bclassify\b",
    r"\bwhich (one|of the following)\b",
    r"\bdetermine (the|whether)\b",
    r"\bestimate the\b",
    r"\bconvert\b",
    r"\d+(\.\d+)?\s*(kn|mpa|gpa|mm|kg|kpa|bar|v|ma|hz|khz|mhz|w|kw|n|m/s|rpm|nm|degc|k|%)\b",
)
_EVALUATION_MARKERS = (
    r"\bevaluate\b",
    r"\bassess\b",
    r"\breview\b",
    r"\bjudge\b",
    r"\bgrade\b",
    r"\bscore\b",
    r"\bcritique\b",
    r"\baudit\b",
    r"\bcompliance\b",
    r"\bcomply\b",
    r"\bagainst (the )?(criteria|requirements|standard|spec)",
    r"\bcheck (this|the|whether)\b",
)
_LONGFORM_MARKERS = (
    r"\blist\b",
    r"\benumerate\b",
    r"\boptions\b",
    r"\brisks?\b",
    r"\bconsiderations\b",
    r"\bchecklist\b",
    r"\bpros and cons\b",
    r"\badvantages\b",
    r"\bfactors\b",
    r"\bbest practices\b",
    r"\brecommendations\b",
    r"\bfaq\b",
    r"\boverview\b",
    r"\bsummari[sz]e the (main|key)\b",
    r"\bwhat are (the|some)\b",
    r"\bways to\b",
    r"\baspects\b",
    r"\bfailure modes\b",
    r"\bguidelines\b",
)
_HIGH_VALUE_MARKERS = (
    r"\broot cause\b",
    r"\bdesign review\b",
    r"\brecommend(ation)?\b",
    r"\bsafety case\b",
    r"\bjustif(y|ication)\b",
    r"\btrade[- ]?off\b",
    r"\bspecification\b",
)

_ENUM_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\bor\b|/)\s*", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"(?:^|\n|\s)(?:\(?\d{1,2}[.)]|[-*])\s+(?=\S)")
_COMPARE_RE = re.compile(r"\b(?:compare|contrast|versus|vs\.?)\b\s+(.+?)(?:[.?!]|$)", re.IGNORECASE)
_FOR_EACH_RE = re.compile(r"\bfor each (?:of )?(?:the )?(?:following|these)?\s*:?\s*(.+?)(?:[.?!]|$)", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.?!])\s+")
_CLAUSE_RE = re.compile(r"\s+(?:for|in|to|when|against|with|under|at|on|given|if|because|so that)\s+", re.IGNORECASE)


def _any(patterns: tuple[str, ...], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]


class TaskFeatures(BaseModel):
    """Surface features of a request used by the heuristic planner (and shown to the LLM planner)."""

    model_config = ConfigDict(extra="forbid")

    word_count: int = 0
    question_count: int = 0
    questions: list[str] = Field(default_factory=list)
    enumerations: list[str] = Field(default_factory=list, description="Independent items detected in the task text")
    numbered_items: int = 0
    input_items: int = Field(default=0, description="len(inputs['items']) when a list is supplied")
    input_tokens: int = Field(default=0, description="Estimated tokens of inputs text/documents plus context")
    has_documents: bool = False
    sequential_markers: list[str] = Field(default_factory=list)
    numeric_markers: list[str] = Field(default_factory=list)
    evaluation_markers: list[str] = Field(default_factory=list)
    longform_markers: list[str] = Field(default_factory=list)
    high_value_markers: list[str] = Field(default_factory=list)
    risk_level: str = "low"

    def summary(self) -> str:
        return (
            f"words={self.word_count} questions={self.question_count} enumerations={len(self.enumerations)} numbered={self.numbered_items} "
            f"input_items={self.input_items} input_tokens={self.input_tokens} sequential={len(self.sequential_markers)} numeric={len(self.numeric_markers)} "
            f"evaluation={len(self.evaluation_markers)} longform={len(self.longform_markers)} high_value={len(self.high_value_markers)} risk={self.risk_level}"
        )


def input_text(request: TaskRequest) -> str:
    """Concatenate text-like inputs (text, document(s), documents list, items) and context evidence."""
    parts: list[str] = []
    for key in ("text", "document", "content", "log", "logs", "transcript"):
        v = request.inputs.get(key)
        if isinstance(v, str):
            parts.append(v)
    docs = request.inputs.get("documents")
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, str):
                parts.append(d)
            elif isinstance(d, dict):
                parts.append(str(d.get("text") or d.get("content") or ""))
            elif isinstance(d, Evidence):
                parts.append(d.content)
    for ev in request.context:
        parts.append(ev.content)
    return "\n\n".join(p for p in parts if p)


def input_items(request: TaskRequest) -> list[Any]:
    """The ``items`` list when present (natural fan-out key: files, parts, clauses, tickets)."""
    items = request.inputs.get("items")
    if isinstance(items, (list, tuple)):
        return list(items)
    return []


def detect_enumerations(task: str) -> list[str]:
    """Find independent items in the task text: 'compare A, B and C', 'for each of X, Y', numbered lists, multiple questions."""
    found: list[str] = []
    for rx in (_COMPARE_RE, _FOR_EACH_RE):
        m = rx.search(task)
        if m:
            items = [s.strip(" .:") for s in _ENUM_SPLIT_RE.split(m.group(1)) if s.strip(" .:")]
            if len(items) >= 2:
                # a trailing clause ("... and titanium for a lightweight bracket") belongs to the
                # sentence, not to the last item
                items[-1] = _CLAUSE_RE.split(items[-1], maxsplit=1)[0].strip(" .:")
            items = [i for i in items if 1 <= len(i.split()) <= 8]
            if len(items) >= 2:
                found = items
                break
    if not found and _NUMBERED_RE.search("\n" + task):
        chunks = [c.strip() for c in _NUMBERED_RE.split("\n" + task) if c.strip()]
        # the first chunk is usually the lead-in sentence; keep it only if it looks like an item
        if len(chunks) >= 3:
            found = chunks[1:] if not chunks[0].endswith((":", "?")) and len(chunks[0].split()) > 12 else chunks[1:] if chunks[0].endswith(":") else chunks
            found = [c for c in found if c]
    if not found:
        questions = [s.strip() for s in _SENTENCE_RE.split(task) if s.strip().endswith("?")]
        if len(questions) >= 2:
            found = questions
    return found[:16]


def extract_features(request: TaskRequest, profile: DomainProfile) -> TaskFeatures:
    """Compute :class:`TaskFeatures` for a request."""
    task = request.task
    sentences = [s.strip() for s in _SENTENCE_RE.split(task) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    text = input_text(request)
    items = input_items(request)
    return TaskFeatures(
        word_count=len(task.split()),
        question_count=len(questions),
        questions=questions[:16],
        enumerations=detect_enumerations(task),
        numbered_items=len(_NUMBERED_RE.findall("\n" + task)),
        input_items=len(items),
        input_tokens=estimate_tokens(text),
        has_documents=bool(text),
        sequential_markers=_any(_SEQUENTIAL_MARKERS, task),
        numeric_markers=_any(_NUMERIC_MARKERS, task),
        evaluation_markers=_any(_EVALUATION_MARKERS, task),
        longform_markers=_any(_LONGFORM_MARKERS, task),
        high_value_markers=_any(_HIGH_VALUE_MARKERS, task),
        risk_level=profile.risk_level,
    )


# --------------------------------------------------------------------------
# Heuristic planner
# --------------------------------------------------------------------------


def expected_calls_for(plan: ParallelPlan, config: ParallelizationConfig) -> int:
    """LLM calls ``execute()`` will spend for ``plan`` (upper bound for voting)."""
    p = plan.pattern
    if p == "sequential":
        return 1
    if p == "sectioning":
        return len(plan.partitions) + (1 if plan.aggregator in ("llm", "usc") else 0)
    if p == "skeleton_of_thought":
        pts = min(config.skeleton_max_points, max(config.skeleton_min_points, len(plan.partitions) or config.skeleton_max_points))
        return 1 + pts + (1 if config.coherence_pass else 0)
    if p == "voting":
        return plan.max_samples + (1 if plan.aggregator == "usc" else 0)
    if p == "branch_solve_merge":
        return 1 + config.bsm_max_branches + 1
    if p == "map_reduce":
        n = max(1, len(plan.partitions))
        reducers = 0
        level = n
        while level > 1:
            level = math.ceil(level / config.reduce_fan_in)
            reducers += level
        return n + max(1, reducers)
    if p == "ensemble":
        return plan.layers * plan.samples + 1
    if p == "speculative":
        return 3
    return 1


def parallel_calls_for(plan: ParallelPlan, config: ParallelizationConfig) -> tuple[int, int]:
    """(serial_calls, parallel_calls) used by the speedup estimate."""
    total = expected_calls_for(plan, config)
    p = plan.pattern
    if p == "sequential":
        return 1, 0
    if p == "sectioning":
        return total - len(plan.partitions), len(plan.partitions)
    if p == "skeleton_of_thought":
        return total - (total - 1 - (1 if config.coherence_pass else 0)), total - 1 - (1 if config.coherence_pass else 0)
    if p == "voting":
        return total - plan.max_samples, plan.max_samples
    if p == "branch_solve_merge":
        return 2, total - 2
    if p == "map_reduce":
        n = max(1, len(plan.partitions))
        return total - n, n
    if p == "ensemble":
        return 1 + plan.layers - 1, plan.samples * plan.layers - (plan.layers - 1)
    if p == "speculative":
        return 1, 2
    return 1, 0


def chunk_count(n_chars: int, chunk_chars: int, overlap_chars: int) -> int:
    """Number of windows :func:`pipeline_core.chunk_text` produces for ``n_chars`` characters."""
    if n_chars <= 0:
        return 0
    step = max(1, chunk_chars - overlap_chars)
    return math.ceil(n_chars / step)


def truncate_partitions(partitions: list[Partition], cap: int) -> tuple[list[Partition], list[str]]:
    """Keep at most ``cap`` partitions as a *dependency-closed* subset.

    Partitions are admitted in topological order (dependencies first), so a
    kept node never depends on a dropped one and the result always passes
    :func:`validate_dag`. Returns ``(kept in original order, dropped ids)``.
    """
    if len(partitions) <= cap:
        return list(partitions), []
    ids = {p.id for p in partitions}
    graph = {p.id: [d for d in p.depends_on if d in ids] for p in partitions}
    order = [n for wave in topological_waves(graph) for n in wave]
    kept: set[str] = set()
    for node in order:
        if len(kept) >= cap:
            break
        if all(d in kept for d in graph[node]):
            kept.add(node)
    keep = [p.model_copy(update={"depends_on": [d for d in p.depends_on if d in kept]}) for p in partitions if p.id in kept]
    return keep, [p.id for p in partitions if p.id not in kept]


def fold_items(task: str, items: list[str], cap: int, *, label: str = "Item") -> list[Partition]:
    """One partition per item; when there are more than ``cap`` items the surplus is
    batched into the last partition instead of being dropped."""
    cap = max(1, cap)
    if len(items) <= cap:
        return [Partition(id=f"p{i + 1}", task=f"{task}\n{label} {i + 1}: {it}") for i, it in enumerate(items)]
    parts = [Partition(id=f"p{i + 1}", task=f"{task}\n{label} {i + 1}: {it}") for i, it in enumerate(items[: cap - 1])]
    rest = "\n".join(f"{label} {i + 1}: {it}" for i, it in enumerate(items) if i >= cap - 1)
    parts.append(Partition(id=f"p{cap}", task=f"{task}\nHandle each of these {label.lower()}s separately:\n{rest}"))
    return parts


def serial_fraction_for(plan: ParallelPlan, config: ParallelizationConfig) -> float:
    """Share of the plan's calls that sit on the serial critical path (planner-free: 1.0 for sequential)."""
    total = expected_calls_for(plan, config)
    if total <= 0:
        return 1.0
    if plan.pattern == "ensemble":
        serial = plan.layers + 1
    elif plan.pattern == "map_reduce":
        serial = max(1, total - max(1, len(plan.partitions)))
    else:
        serial, _ = parallel_calls_for(plan, config)
    return min(1.0, max(0.0, serial / total))


def expected_speedup_for(plan: ParallelPlan, config: ParallelizationConfig) -> float:
    """Critical-path speedup estimate for ``plan`` (relative units: one call = 1.0).

    Layered patterns (ensemble) have one serial stage per layer plus the
    aggregator, so they are estimated directly rather than through the
    single-fan-out shape of :func:`estimate_fan_out`.
    """
    if plan.pattern == "ensemble":
        work = plan.layers * plan.samples + 1
        waves = sum(math.ceil(plan.samples / max(1, plan.concurrency)) for _ in range(plan.layers))
        crit = waves + 1
        return round(work / crit, 4) if crit else 1.0
    if plan.pattern == "map_reduce":
        n = max(1, len(plan.partitions))
        levels = 0
        level = n
        while level > 1:
            level = math.ceil(level / config.reduce_fan_in)
            levels += 1
        levels = max(1, levels)
        work = expected_calls_for(plan, config)
        crit = math.ceil(n / max(1, plan.concurrency)) + levels
        return round(work / crit, 4)
    serial, parallel = parallel_calls_for(plan, config)
    return estimate_fan_out(parallel, serial_calls=serial, concurrency=plan.concurrency).speedup


class HeuristicPlanner:
    """Rule-based planner used offline, under ``deterministic`` and as the LLM planner's fallback.

    Also owns :meth:`finalize`, which turns any raw decision (heuristic or
    LLM) into a budget-, domain- and configuration-consistent plan.
    """

    def __init__(self, config: ParallelizationConfig | None = None) -> None:
        self.config = config or ParallelizationConfig()

    def partition_cap(self) -> int:
        """Most partitions a plan may have: ``min(max_partitions, token_multiplier_cap)``."""
        return max(2, min(self.config.max_partitions, int(self.config.token_multiplier_cap)))

    # ------------------------------------------------------------ routing
    def choose(self, features: TaskFeatures, request: TaskRequest, profile: DomainProfile) -> PlannerOutput:
        """Apply the routing table to task features."""
        cfg = self.config
        c = request.constraints
        high_risk = profile.risk_level in ("high", "safety_critical")

        # (a) explicit list input -> sectioning keyed by item
        if features.input_items >= 2:
            items = [_item_text(it) for it in input_items(request)]
            cap = self.partition_cap()
            parts = fold_items(request.task, items, cap)
            why = f"inputs['items'] supplies {len(items)} independent items; one worker per item, merged by an LLM synthesis call (sectioning)."
            if len(items) > cap:
                why += f" Items {cap}-{len(items)} are batched into partition p{cap} (partition cap {cap} = min(max_partitions, token_multiplier_cap))."
            return PlannerOutput(pattern="sectioning", partitions=parts, aggregator="llm", verifiability="free_form", rationale=why)

        # (d) long input -> map-reduce
        if features.input_tokens >= cfg.map_reduce_min_tokens:
            return PlannerOutput(pattern="map_reduce", aggregator="llm", verifiability="free_form", rationale=f"input is ~{features.input_tokens} tokens; chunked map with structured records and confidence-aware reduce (map-reduce).")

        # (f) sequentially dependent -> no parallelism (overrides everything but explicit items / long input)
        if features.sequential_markers and not features.enumerations:
            if features.numeric_markers and not _code_like(features):
                return self._voting(features, profile, "calculation with dependent steps: sampling-and-voting with numeric canonicalisation instead of skeleton expansion (engineering adaptation note).")
            return PlannerOutput(pattern="sequential", aggregator="code", verifiability="free_form", rationale=f"task markers {features.sequential_markers[:2]} indicate step-dependent content; SoT-R routes derivations, code and narrative to sequential generation.")

        # (c) single verifiable answer -> voting
        if features.numeric_markers and features.question_count <= 1 and not features.enumerations:
            return self._voting(features, profile, "single checkable answer: independent samples with majority vote (self-consistency / More Agents).")

        # (a) enumerations or several questions -> sectioning
        if len(features.enumerations) >= 2:
            enums = features.enumerations
            cap = self.partition_cap()
            parts = [Partition(id=f"p{i + 1}", task=_partition_task(request.task, e, features)) for i, e in enumerate(enums[: cap if len(enums) <= cap else cap - 1])]
            if len(enums) > cap:
                parts.append(Partition(id=f"p{cap}", task=f"{request.task}\nFocus only on each of: " + "; ".join(enums[cap - 1 :])))
            return PlannerOutput(pattern="sectioning", partitions=parts, aggregator="llm", verifiability="free_form", rationale=f"{len(enums)} independent sub-questions or items detected in the task text; solve concurrently in {len(parts)} partitions and merge (sectioning).")

        # (e) evaluation / multi-constraint -> branch-solve-merge
        if features.evaluation_markers:
            return PlannerOutput(pattern="branch_solve_merge", aggregator="llm", verifiability="free_form", rationale="evaluation or compliance task: let the model propose criteria, solve each independently, merge (Branch-Solve-Merge reduces judge bias).")

        # (b) point-independent long form -> skeleton-of-thought
        if features.longform_markers:
            return PlannerOutput(pattern="skeleton_of_thought", aggregator="concat", verifiability="free_form", rationale="long-form answer with independent points: skeleton then parallel expansion (Skeleton-of-Thought).")

        # (e') high-value free form in a high-risk domain -> ensemble when affordable
        if high_risk and (features.high_value_markers or features.word_count >= 25):
            budget_ok = c.max_llm_calls is None or c.max_llm_calls >= cfg.ensemble_layers * cfg.ensemble_proposers + 2
            if budget_ok:
                return PlannerOutput(pattern="ensemble", samples=cfg.ensemble_proposers, aggregator="llm", verifiability="free_form", rationale=f"high-value free-form task in a {profile.risk_level} domain: homogeneous proposers plus critical aggregation (MoA-Lite / Self-MoA).")

        # latency-bound simple task -> speculative race
        if c.max_latency_s is not None and c.max_latency_s <= 30 and not high_risk:
            return PlannerOutput(pattern="speculative", aggregator="code", verifiability="free_form", rationale="tight latency budget on a simple task: race a fast model against the default and verify the fast answer (speculative execution).")

        return PlannerOutput(pattern="sequential", aggregator="code", verifiability="free_form", rationale="no independent structure detected; the simplest approach that works is one careful call (Anthropic: start simple).")

    def _voting(self, features: TaskFeatures, profile: DomainProfile, why: str) -> PlannerOutput:
        return PlannerOutput(
            pattern="voting",
            samples=self.config.voting_initial(profile.risk_level),
            aggregator="code",
            verifiability="exact_match",
            rationale=why,
        )

    # ----------------------------------------------------------- finalize
    def finalize(
        self,
        decision: PlannerOutput,
        request: TaskRequest,
        profile: DomainProfile,
        *,
        remaining_calls: Optional[int] = None,
        planner: Literal["heuristic", "llm"] = "heuristic",
    ) -> ParallelPlan:
        """Turn a raw decision into a consistent plan: domain policy, budgets, limits, estimates."""
        cfg = self.config
        c: Constraints = request.constraints
        high_risk = profile.risk_level in ("high", "safety_critical")

        warnings: list[str] = []
        partitions = list(decision.partitions)
        if decision.pattern == "sectioning" and len(partitions) > self.partition_cap():
            partitions, dropped = truncate_partitions(partitions, self.partition_cap())
            warnings.append(f"plan: partitions {dropped} dropped (partition cap {self.partition_cap()} = min(max_partitions, token_multiplier_cap))")
        if decision.pattern == "sectioning":
            if len(partitions) < 2:
                decision = PlannerOutput(pattern="sequential", aggregator="code", verifiability=decision.verifiability, rationale=decision.rationale + " (fewer than two partitions; running sequentially)")
                partitions = []
            else:
                validate_dag({p.id: p.depends_on for p in partitions}, max_nodes=cfg.max_partitions)
        elif decision.pattern == "map_reduce":
            n_chunks = chunk_count(len(input_text(request)), cfg.chunk_tokens * 4, int(cfg.chunk_tokens * 4 * cfg.chunk_overlap_ratio))
            if n_chunks < 2:
                decision = PlannerOutput(pattern="sequential", aggregator="code", verifiability=decision.verifiability, rationale=decision.rationale + " (input fits in one chunk; running sequentially)")
                partitions = []
            else:
                partitions = [Partition(id=f"c{i + 1}", task=f"chunk {i + 1} of {n_chunks}") for i in range(n_chunks)]
        else:
            partitions = []

        concurrency = min(cfg.max_concurrency, c.max_parallelism)
        failure: FailureMode = "quorum" if high_risk else "best_effort"
        if profile.risk_level == "safety_critical":
            quorum = 1.0  # every branch must succeed before a safety-critical answer is assembled
        else:
            quorum = cfg.quorum_fraction

        samples = decision.samples
        max_samples = samples
        layers = 1
        aggregator: Aggregator = decision.aggregator
        if decision.pattern == "voting":
            samples = max(1, min(decision.samples, 40))
            # domain policy applies to every planner: high-risk profiles never start below their floor
            samples = max(samples, cfg.voting_initial(profile.risk_level))
            max_samples = max(samples, cfg.voting_max_samples)
            if decision.verifiability == "free_form":
                aggregator = "usc"
            else:
                aggregator = "code"
        elif decision.pattern == "ensemble":
            samples = max(1, min(decision.samples, 8)) if decision.samples else cfg.ensemble_proposers
            layers = cfg.ensemble_layers
            aggregator = "llm"
        elif decision.pattern in ("branch_solve_merge", "map_reduce"):
            aggregator = "llm"
        elif decision.pattern == "skeleton_of_thought":
            aggregator = "concat"
        elif decision.pattern == "sectioning" and aggregator not in ("llm", "concat"):
            aggregator = "llm"
        elif decision.pattern in ("sequential", "speculative"):
            aggregator = "code"

        plan = ParallelPlan(
            pattern=decision.pattern,
            partitions=partitions,
            concurrency=concurrency,
            failure_policy=failure,
            quorum=quorum,
            rate_limits=RateLimits(requests_per_minute=cfg.requests_per_minute, input_tokens_per_minute=cfg.input_tokens_per_minute, output_tokens_per_minute=cfg.output_tokens_per_minute),
            samples=samples,
            max_samples=max_samples,
            layers=layers,
            aggregator=aggregator,
            verifiability=decision.verifiability,
            cache_aware=cfg.cache_aware_fan_out and not c.deterministic,
            branch_effort=cfg.branch_effort,
            aggregator_effort=cfg.aggregator_effort,
            rationale=decision.rationale,
            planner=planner,
            warnings=warnings,
        )
        plan = self._apply_budget(plan, remaining_calls)
        plan = self._apply_token_cap(plan)
        plan.expected_calls = expected_calls_for(plan, cfg)
        plan.expected_speedup = expected_speedup_for(plan, cfg)
        serial_share = serial_fraction_for(plan, cfg)
        plan.serial_fraction = serial_share
        if plan.pattern != "sequential" and serial_share > cfg.serial_fraction_target:
            plan.rationale += f" Note: serial share {serial_share:.0%} exceeds the {cfg.serial_fraction_target:.0%} target; Amdahl bound {1 / serial_share:.1f}x."
        return plan

    def _apply_budget(self, plan: ParallelPlan, remaining: Optional[int]) -> ParallelPlan:
        """Degrade sample counts / partitions rather than fail when ``max_llm_calls`` is tight."""
        if remaining is None:
            return plan
        cfg = self.config
        note = " Budget: reduced to fit max_llm_calls."
        if remaining < 3:
            if plan.pattern != "sequential":
                plan.pattern = "sequential"
                plan.partitions = []
                plan.aggregator = "code"
                plan.rationale += note
            return plan
        if plan.pattern == "voting":
            usc = 1 if plan.aggregator == "usc" else 0
            cap = max(1, remaining - usc)
            if plan.max_samples > cap:
                plan.max_samples = cap
                plan.samples = min(plan.samples, cap)
                plan.rationale += note
        elif plan.pattern == "sectioning":
            agg = 1 if plan.aggregator in ("llm", "usc") else 0
            cap = max(2, remaining - agg)
            if len(plan.partitions) > cap:
                plan.partitions, dropped = truncate_partitions(plan.partitions, cap)
                plan.rationale += note + f" Partitions not run: {dropped}."
                plan.warnings.append(f"budget: partitions {dropped} dropped to fit max_llm_calls")
        elif plan.pattern == "skeleton_of_thought":
            if 1 + cfg.skeleton_min_points + (1 if cfg.coherence_pass else 0) > remaining:
                plan.pattern = "sequential"
                plan.aggregator = "code"
                plan.rationale += note
        elif plan.pattern == "branch_solve_merge":
            if 2 + cfg.bsm_min_branches > remaining:
                plan.pattern = "sequential"
                plan.aggregator = "code"
                plan.rationale += note
        elif plan.pattern == "ensemble":
            # need layers*samples + 1 aggregator
            while plan.layers * plan.samples + 1 > remaining and (plan.layers > 1 or plan.samples > 1):
                if plan.layers > 1:
                    plan.layers -= 1
                else:
                    plan.samples -= 1
                plan.rationale += note if not plan.rationale.endswith(note) else ""
            if plan.layers * plan.samples + 1 > remaining:
                plan.pattern = "sequential"
                plan.aggregator = "code"
        elif plan.pattern == "speculative" and remaining < 3:
            plan.pattern = "sequential"
            plan.aggregator = "code"
            plan.rationale += note
        elif plan.pattern == "map_reduce" and expected_calls_for(plan, cfg) > remaining:
            plan.pattern = "sequential"
            plan.partitions = []
            plan.aggregator = "code"
            plan.rationale += note + " Long input sent in a single call."
        return plan

    def _apply_token_cap(self, plan: ParallelPlan) -> ParallelPlan:
        """Keep the fan-out under the token-multiplier cap (Anthropic: ~15x of a single call)."""
        cap = int(self.config.token_multiplier_cap)
        if plan.pattern == "voting" and plan.max_samples > cap:
            plan.max_samples = cap
            plan.samples = min(plan.samples, cap)
        if plan.pattern == "sectioning" and len(plan.partitions) > cap:
            plan.partitions, dropped = truncate_partitions(plan.partitions, cap)
            plan.rationale += f" Token cap: partitions {dropped} not run."
            plan.warnings.append(f"token cap: partitions {dropped} dropped (token_multiplier_cap={self.config.token_multiplier_cap:g})")
        if plan.pattern == "ensemble" and plan.layers * plan.samples > cap:
            plan.samples = max(1, cap // plan.layers)
        return plan

    # ---------------------------------------------------------------- plan
    def plan(self, request: TaskRequest, profile: DomainProfile, *, remaining_calls: Optional[int] = None) -> ParallelPlan:
        """Full heuristic path: features -> routing -> finalize."""
        features = extract_features(request, profile)
        decision = self.choose(features, request, profile)
        return self.finalize(decision, request, profile, remaining_calls=remaining_calls, planner="heuristic")

    def validate_llm_output(self, raw: Any) -> PlannerOutput:
        """Validate the LLM planner's JSON; raise :class:`PlanValidationError` when unusable."""
        try:
            out = PlannerOutput.model_validate(raw)
        except Exception as exc:  # pydantic.ValidationError
            raise PlanValidationError(f"planner output failed validation: {exc}") from exc
        ids = [p.id for p in out.partitions]
        if len(ids) != len(set(ids)):
            raise PlanValidationError("planner produced duplicate partition ids")
        if out.partitions:
            validate_dag({p.id: p.depends_on for p in out.partitions}, max_nodes=self.config.max_partitions)
        return out


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return ", ".join(f"{k}={v}" for k, v in item.items())
    return str(item)


def _code_like(features: TaskFeatures) -> bool:
    return any("function" in m or "implement" in m or "debug" in m or "refactor" in m or "migrat" in m for m in features.sequential_markers)


def _partition_task(task: str, item: str, features: TaskFeatures) -> str:
    if item in features.questions:
        return item
    return f"{task}\nFocus only on: {item}"
