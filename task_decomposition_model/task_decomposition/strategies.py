"""Decomposition strategies: each turns a task into a :class:`TaskGraph`.

Every strategy has two paths that produce the same type:

* **LLM** - one structured ``DECOMPOSE_TEMPLATE`` call (plus at most
  ``plan_repair_attempts`` repair calls when the graph fails validation), used
  when the client is live and ``constraints.deterministic`` is not set.
* **Heuristic** - a deterministic splitter over task features and the domain
  profile, used offline, under ``deterministic`` and as the fallback when the
  LLM graph cannot be repaired (HuggingGPT planning F1 ~50%: validate, never
  trust).

Strategy catalogue (dossier architecture recommendation 1):

``direct``            one node, no decomposition (Anthropic damping rule)
``plan_and_solve``    one PS+ call that plans then executes (Wang et al.)
``least_to_most``     ordered chain with answer carry-forward (Zhou et al.)
``dag``               LLMCompiler/ReWOO dependency graph with ``$id`` placeholders
``recursive``         ADaPT: try the whole task, expand on failure to ``d_max``
``tree_of_thoughts``  propose several candidate graphs, score, keep the best
"""
from __future__ import annotations

import re
from abc import ABC
from typing import Any, ClassVar, Optional, Sequence

from pipeline_core import (
    BudgetExceededError,
    DomainProfile,
    Evidence,
    LLMError,
    RunContext,
    StructuredOutputError,
    TaskRequest,
    tokenize_words,
    validate_structured,
)

from .config import TaskDecompositionConfig
from .graph import SubTask, TaskGraph
from .plan import _ENUM_RE, DecompositionPlan, TaskFeatures, extract_features
from .prompting import build_request, render_documents, render_inputs
from .prompts import CANDIDATES_TEMPLATE, DECOMPOSE_TEMPLATE, EVALUATE_TEMPLATE, REFINE_TEMPLATE
from .schemas import CandidateGraphs, CandidateScores, GraphDraft, schema_for

_SEQ_START = re.compile(
    r"^(?:and\s+)?(?:then|next|finally|lastly|afterwards|after that|subsequently|using (?:the|those|that) results?|based on (?:the|those|that)|with (?:the|those|that) results?)\b[,\s]*",
    re.IGNORECASE,
)
_AGGREGATE_WORDS = ("summar", "overall", "combine", "aggregate", "consolidate", "report", "rank", "recommend", "conclude", "list open", "final")
_VERIFY_VERBS = ("verify", "check", "validate", "confirm", "audit", "test", "review")
_COMPUTE_VERBS = ("compute", "calculate", "size", "estimate", "derive", "quantify", "measure", "model")
_LOOKUP_VERBS = ("find", "look up", "lookup", "retrieve", "search", "locate", "identify", "extract", "list", "collect", "gather")
_STOPWORDS = frozenset(
    "the a an and or of to in for on with by from at as is are be this that these those it its into over under then than "
    "each every all any per using use used via about between against which what when where who how do does did not".split()
)


# ------------------------------------------------------------------ helpers


def use_llm_path(ctx: RunContext, config: TaskDecompositionConfig) -> bool:
    """LLM planning is used when the client is live (or forced) and the run is not deterministic."""
    if ctx.request.constraints.deterministic:
        return False
    return (not ctx.offline) or config.force_llm_path


def stop_reason(ctx: RunContext) -> Optional[str]:
    """Why no further LLM call may be issued: ``"deadline"``, ``"budget"`` (call *or* cost cap reached) or ``None``.

    Mirrors ``BudgetTracker`` so every gate (planner, strategies, scheduler, synthesis, verify) honours
    ``Constraints.max_llm_calls``, ``max_cost_usd`` and ``max_latency_s`` alike (architecture rule 3.2:
    check the budget before fan-outs and degrade rather than fail).
    """
    if ctx.deadline_exceeded():
        return "deadline"
    rem = ctx.budget.remaining_calls()
    if rem is not None and rem <= 0:
        return "budget"
    cost = ctx.budget.remaining_cost()
    if cost is not None and cost <= 0.0:
        return "budget"
    return None


def can_call(ctx: RunContext, n: int = 1) -> bool:
    """True when at least ``n`` LLM calls remain, the cost cap is not reached and the deadline has not passed."""
    if stop_reason(ctx) is not None:
        return False
    rem = ctx.budget.remaining_calls()
    return rem is None or rem >= n


def chainify(graph: TaskGraph) -> TaskGraph:
    """Make the graph a Least-to-Most chain: every node depends on all nodes before it (answer carry-forward)."""
    order = [tid for wave in graph.topological_waves() for tid in wave] if not graph.validate_structure() else graph.ids()
    seen: list[str] = []
    tasks: list[SubTask] = []
    for tid in order:
        t = graph.get(tid).model_copy(update={"depends_on": list(seen)})
        tasks.append(t)
        seen.append(tid)
    return graph.with_tasks(tasks)


def content_words(text: str) -> set[str]:
    """Lower-cased content words (stop-words and short tokens removed)."""
    return {w for w in tokenize_words(text) if len(w) > 2 and w not in _STOPWORDS}


def heuristic_segments(task: str, *, max_segments: int = 8) -> tuple[str, list[str]]:
    """Split a task into ``(preamble, segments)`` without a model.

    Enumerated lines win; otherwise sentences are split on semicolons, ``, and``,
    ``then`` and ``and <verb>`` joints. Segments shorter than two words are
    merged into their neighbour; more than ``max_segments`` are merged pairwise
    from the tail.
    """
    text = task.strip()
    preamble = ""
    items = [m for m in _ENUM_RE.split(text) if m and m.strip()]
    if len(_ENUM_RE.findall(text)) >= 2:
        first = text[: _ENUM_RE.search(text).start()].strip() if _ENUM_RE.search(text) else ""
        segs = [s.strip().rstrip(".;,") for s in items]
        if first and segs and segs[0].startswith(first):
            segs[0] = segs[0][len(first):].strip().rstrip(".;,")
            preamble = first.rstrip(":").strip()
        segs = [s for s in segs if s]
    else:
        if ":" in text and text.index(":") < len(text) * 0.5:
            head, _, rest = text.partition(":")
            if len(head.split()) <= 12 and rest.strip():
                preamble, text = head.strip(), rest.strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        segs = []
        for s in sentences:
            parts = re.split(r";\s*|,\s*(?=(?:and\s+)?(?:then|finally|next|lastly)\s)|\s+(?=and\s+(?:then|finally|next|lastly)\s)|,\s*and\s+(?!then|finally|next|lastly)|\s+(?=then\s)|,\s+(?=[a-z]+\s)|\s+and\s+(?=(?:" + "|".join(_VERIFY_VERBS + _COMPUTE_VERBS + _LOOKUP_VERBS + ("summarise", "summarize", "compare", "design", "write", "draft", "propose", "recommend", "assess", "evaluate", "determine", "document", "explain", "describe", "analyse", "analyze", "plan", "select", "map")) + r")\b)", s, flags=re.IGNORECASE)
            segs.extend(p.strip().rstrip(".;,") for p in parts if p and p.strip())
    merged: list[str] = []
    for s in segs:
        if merged and len(s.split()) < 2:
            merged[-1] = f"{merged[-1]}, {s}"
        else:
            merged.append(s)
    # scope clauses ("for each of the logs", "given the datasheet") are not work items:
    # fold them into the independent segments that follow
    folded: list[str] = []
    scope = ""
    for s in merged:
        if _SCOPE_START.match(s) and not _starts_with_action(s):
            scope = s.rstrip(",")
            continue
        if scope and not _SEQ_START.match(s):
            s = f"{scope}: {s[0].lower() + s[1:]}"
        folded.append(s)
    if not folded and scope:
        folded.append(scope)
    merged = folded
    while len(merged) > max_segments:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]}; {tail}"
    merged = [s[0].upper() + s[1:] if s else s for s in merged]
    return preamble, merged


_SCOPE_START = re.compile(r"^(?:for each|for every|for all|given|across|within|in each|per)\b", re.IGNORECASE)


def _starts_with_action(segment: str) -> bool:
    words = segment.lower().split()
    if not words:
        return False
    skip = {"for", "each", "every", "all", "of", "the", "given", "across", "within", "in", "per", "a", "an"}
    head = [w for w in words[:6] if w not in skip]
    first = head[0] if head else words[0]
    return any(first.startswith(v) for v in _VERIFY_VERBS + _COMPUTE_VERBS + _LOOKUP_VERBS + ("summar", "compar", "design", "write", "draft", "propos", "recommend", "assess", "evaluat", "determin", "document", "explain", "describ", "analys", "analyz", "plan", "select", "map", "review", "debug", "diagnos", "investigat", "fix", "implement", "build", "creat", "updat", "migrat", "refactor", "classif", "detect", "resolv", "trace", "report", "rank", "choose", "convert", "translat", "generat", "size"))


def _tool_hint(segment: str, *, has_docs: bool, depends: bool) -> str:
    low = segment.lower()
    first = low.split()[0] if low.split() else ""
    if any(first.startswith(v) for v in _VERIFY_VERBS):
        return "verify"
    if any(first.startswith(v) for v in _COMPUTE_VERBS):
        return "compute"
    if depends and any(w in low for w in _AGGREGATE_WORDS):
        return "aggregate"
    if has_docs and any(low.startswith(v) for v in _LOOKUP_VERBS):
        return "retrieve"
    return "llm"


def _complexity(segment: str) -> str:
    n = len(segment.split())
    return "low" if n <= 6 else ("high" if n >= 18 else "medium")


def _criteria(segment: str, profile: DomainProfile, *, has_docs: bool) -> list[str]:
    crit = [f"Addresses: {segment}"]
    if profile.units:
        crit.append("Every quantity carries a unit from the domain unit list")
    if has_docs:
        crit.append("Cites the supporting document ids")
    if profile.risk_level in ("high", "safety_critical") and profile.verification_rules:
        crit.append(profile.verification_rules[0])
    return crit


def heuristic_graph(
    task: str,
    *,
    plan: DecompositionPlan,
    profile: DomainProfile,
    config: TaskDecompositionConfig,
    features: Optional[TaskFeatures] = None,
    inputs: Optional[dict[str, Any]] = None,
    context: Sequence[Evidence] = (),
    strategy: str = "dag",
    chain: bool = False,
    max_segments: Optional[int] = None,
) -> TaskGraph:
    """Deterministic decomposition from task structure and the domain profile.

    Independent segments become parallel nodes; segments opening with a
    sequence marker (``then``, ``using the result`` ...) or summarising words
    depend on everything before them. Missing breadth is padded with a PS+
    variable-extraction node and the profile's ``decomposition_hints``; high
    and safety-critical profiles end with an independent ``verify`` node.
    """
    f = features or extract_features(task, inputs=inputs, context=context, profile=profile)
    has_docs = f.has_documents
    limit = max_segments or plan.max_breadth
    preamble, segs = heuristic_segments(task, max_segments=max(1, limit))
    want = min(config.min_subtasks, limit)
    tasks: list[SubTask] = []
    ids: list[str] = []
    if len(segs) < want and len(segs) < limit:
        # PS+ variable extraction comes first so its id reflects execution order
        desc = "Extract the relevant variables, inputs, constraints and assumptions from the task and documents"
        tasks.append(
            SubTask(
                id="t1",
                description=desc,
                expected_output=f"Result for: {desc}",
                verification_criteria=_criteria(desc, profile, has_docs=has_docs),
                tool_hint="llm",
                complexity="low",
                max_retries=config.max_retries_per_task,
            )
        )
        ids.append("t1")  # sequence-marked and summarising segments depend on everything before them, padding included
    offset = len(tasks)
    for i, seg in enumerate(segs, 1 + offset):
        tid = f"t{i}"
        # ``i`` numbers nodes (after the optional PS+ padding node); ``i - offset`` is the segment index,
        # so the "last segment aggregates" rule must compare the segment index, not the node id
        is_last_segment = (i - offset) == len(segs) and len(segs) > 1
        depends = chain or bool(_SEQ_START.match(seg)) or (is_last_segment and any(w in seg.lower() for w in _AGGREGATE_WORDS))
        desc = _SEQ_START.sub("", seg).strip() or seg
        desc = desc[0].upper() + desc[1:]
        deps = list(ids) if depends else []
        tasks.append(
            SubTask(
                id=tid,
                description=desc,
                depends_on=deps,
                inputs={"preamble": preamble} if preamble else {},
                expected_output=f"Result for: {desc}",
                verification_criteria=_criteria(desc, profile, has_docs=has_docs),
                tool_hint=_tool_hint(desc, has_docs=has_docs, depends=bool(deps)),
                complexity=_complexity(desc),  # type: ignore[arg-type]
                max_retries=config.max_retries_per_task,
            )
        )
        ids.append(tid)
    # padding with the profile's decomposition hints (how experts split work in this field)
    for hint in profile.decomposition_hints:
        if len(tasks) >= want or len(tasks) >= limit:
            break
        desc = f"{hint.rstrip('.')} for this task"
        tasks.append(
            SubTask(
                id=f"t{len(tasks) + 1}",
                description=desc,
                depends_on=[],
                expected_output=f"Result for: {desc}",
                verification_criteria=_criteria(desc, profile, has_docs=has_docs),
                tool_hint="llm",
                complexity="medium",
                max_retries=config.max_retries_per_task,
            )
        )
    if profile.risk_level in ("high", "safety_critical") and len(tasks) > 1 and len(tasks) < config.max_nodes:
        others = [t.id for t in tasks]
        tasks.append(
            SubTask(
                id="verify",
                description="Independently verify the combined sub-task results against the domain verification rules and flag every safety-relevant implication",
                depends_on=others,
                inputs={tid: f"${tid}.output" for tid in others},
                expected_output="Pass/fail per verification rule with the evidence used",
                verification_criteria=list(profile.verification_rules) or ["Each result was checked against an independent reference"],
                tool_hint="verify",
                complexity="medium",
                max_retries=config.max_retries_per_task,
            )
        )
    graph = TaskGraph(root_task=task, tasks=tasks, strategy=strategy, metadata={"planner": "heuristic", "preamble": preamble} if preamble else {"planner": "heuristic"})
    graph = graph.infer_dependencies()
    if chain:
        graph = chainify(graph)
    return graph


def structural_score(graph: TaskGraph, task: str, *, max_nodes: int = 25) -> float:
    """Model-free quality score in [0, 1]: keyword coverage, validity and parallel efficiency."""
    if not graph.tasks:
        return 0.0
    words = content_words(task)
    covered = content_words(" ".join(t.description for t in graph.tasks))
    coverage = len(words & covered) / len(words) if words else 1.0
    valid = 1.0 if not graph.validate_structure(max_nodes=max_nodes) else 0.0
    speed = min(1.0, graph.parallel_speedup_estimate() / max(1, len(graph.tasks)))
    size_penalty = 0.0 if 2 <= len(graph.tasks) <= 8 else 0.15
    return max(0.0, min(1.0, 0.55 * coverage + 0.3 * valid + 0.15 * speed - size_penalty))


def aggregate_votes(votes: Sequence[Sequence[float]], n_candidates: int, *, prune_below: float = 0.05) -> list[float]:
    """Average per-candidate scores over evaluator samples (ToT value voting); candidates any voter rated as impossible are pruned to 0."""
    totals = [0.0] * n_candidates
    counts = [0] * n_candidates
    pruned = [False] * n_candidates
    for vote in votes:
        for i, s in enumerate(vote[:n_candidates]):
            totals[i] += s
            counts[i] += 1
            if s < prune_below:
                pruned[i] = True
    return [0.0 if pruned[i] else (totals[i] / counts[i] if counts[i] else 0.0) for i in range(n_candidates)]


def _bounds_text(plan: DecompositionPlan, config: TaskDecompositionConfig, *, chain: bool) -> str:
    lo = min(config.min_subtasks, plan.max_breadth)
    if chain:
        return f"min={lo} max={plan.max_breadth} sub-questions, ordered from easiest to hardest; each may depend on the previous ones; strategy=least_to_most"
    return f"min={lo} max={plan.max_breadth} sub-tasks per level; max_depth={plan.max_depth}; max_nodes={config.max_nodes}; strategy={plan.strategy}"


# ----------------------------------------------------------------- base class


class Strategy(ABC):
    """Common machinery: LLM decomposition with validation and repair, heuristic fallback."""

    name: ClassVar[str] = "dag"
    chain: ClassVar[bool] = False

    def __init__(self, config: Optional[TaskDecompositionConfig] = None) -> None:
        self.config = config or TaskDecompositionConfig()

    # ------------------------------------------------------------ heuristic
    def heuristic(
        self,
        task: str,
        *,
        plan: DecompositionPlan,
        profile: DomainProfile,
        features: Optional[TaskFeatures] = None,
        inputs: Optional[dict[str, Any]] = None,
        context: Sequence[Evidence] = (),
    ) -> TaskGraph:
        return heuristic_graph(task, plan=plan, profile=profile, config=self.config, features=features, inputs=inputs, context=context, strategy=self.name, chain=self.chain)

    # ------------------------------------------------------------------ llm
    async def llm_decompose(
        self,
        ctx: RunContext,
        request: TaskRequest,
        *,
        task: str,
        plan: DecompositionPlan,
        feedback: str = "",
        step: str = "decompose.llm",
    ) -> Optional[TaskGraph]:
        """One structured decomposition call with bounded repair; ``None`` when no valid graph could be obtained."""
        cfg = self.config
        hints = "\n".join(f"- {h}" for h in ctx.profile.decomposition_hints) or "(none)"
        for attempt in range(cfg.plan_repair_attempts + 1):
            if not can_call(ctx):
                return None
            req = build_request(
                DECOMPOSE_TEMPLATE,
                profile=ctx.profile,
                values={
                    "task": task,
                    "bounds": _bounds_text(plan, cfg, chain=self.chain),
                    "hints": hints,
                    "inputs": render_inputs(request.inputs),
                    "documents": render_documents(request.context, max_items=cfg.max_context_docs),
                    "feedback": feedback,
                },
                schema=schema_for(GraphDraft),
                effort=cfg.planner_effort,
                tags=[self.name, f"attempt{attempt}"],
            )
            try:
                resp = await ctx.llm_call(req, step=step)
                draft = validate_structured(resp.parsed, GraphDraft)
            except BudgetExceededError:
                return None
            except (StructuredOutputError, LLMError) as exc:
                feedback = f"<previous_attempt_error>\n{type(exc).__name__}: {str(exc)[:300]}\n</previous_attempt_error>\nReturn a valid graph."
                ctx.warn(f"{self.name}: decomposition output invalid ({type(exc).__name__}); repairing" if attempt < cfg.plan_repair_attempts else f"{self.name}: decomposition failed after repair")
                continue
            graph = draft.to_graph(task, strategy=self.name, max_retries=cfg.max_retries_per_task)
            graph.metadata["planner"] = "llm"
            if self.chain:
                graph = chainify(graph)
            issues = graph.validate_structure(
                max_depth=None if self.chain else plan.max_depth,
                max_breadth=1 if self.chain else plan.max_breadth,
                max_nodes=cfg.max_nodes,
                detect_duplicates=cfg.loop_detection,
            )
            if len(graph.tasks) > (plan.max_breadth if self.chain else cfg.max_nodes):
                issues.append(f"{len(graph.tasks)} sub-tasks exceed the requested maximum")
            if not issues:
                return graph
            feedback = "<previous_attempt_issues>\n" + "\n".join(f"- {i}" for i in issues) + "\n</previous_attempt_issues>\nFix these issues and return a corrected graph."
            ctx.warn(f"{self.name}: invalid graph ({'; '.join(issues)[:200]}); {'repairing' if attempt < cfg.plan_repair_attempts else 'falling back to heuristics'}")
        return None

    async def build(
        self,
        task: str,
        *,
        request: TaskRequest,
        plan: DecompositionPlan,
        ctx: RunContext,
        features: Optional[TaskFeatures] = None,
    ) -> TaskGraph:
        """Produce the graph: LLM path when available, heuristic otherwise."""
        if use_llm_path(ctx, self.config):
            graph = await self.llm_decompose(ctx, request, task=task, plan=plan)
            if graph is not None:
                return graph
            ctx.warn(f"{self.name}: using heuristic decomposition")
        return self.heuristic(task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)


# ----------------------------------------------------------------- concrete


class DirectStrategy(Strategy):
    """No decomposition: one node that is the task itself (Anthropic damping rule)."""

    name = "direct"

    def heuristic(self, task: str, *, plan: DecompositionPlan, profile: DomainProfile, features: Optional[TaskFeatures] = None, inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = ()) -> TaskGraph:
        f = features or extract_features(task, inputs=inputs, context=context, profile=profile)
        node = SubTask(
            id="t1",
            description=task.strip(),
            expected_output="A complete answer to the task",
            verification_criteria=_criteria("the task as stated", profile, has_docs=f.has_documents),
            tool_hint="llm",
            complexity="low" if f.estimated_steps <= 2 else "medium",
            max_retries=self.config.max_retries_per_task,
        )
        return TaskGraph(root_task=task, tasks=[node], strategy=self.name, metadata={"planner": "heuristic"})

    async def build(self, task: str, *, request: TaskRequest, plan: DecompositionPlan, ctx: RunContext, features: Optional[TaskFeatures] = None) -> TaskGraph:
        return self.heuristic(task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)


class PlanAndSolve(DirectStrategy):
    """Zero-shot Plan-and-Solve (PS+): one call that writes the plan then executes it."""

    name = "plan_and_solve"

    def heuristic(self, task: str, *, plan: DecompositionPlan, profile: DomainProfile, features: Optional[TaskFeatures] = None, inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = ()) -> TaskGraph:
        graph = super().heuristic(task, plan=plan, profile=profile, features=features, inputs=inputs, context=context)
        graph.tasks[0].tool_hint = "plan_and_solve"
        graph.tasks[0].complexity = "medium"
        graph.strategy = self.name
        return graph


class LeastToMost(Strategy):
    """Ordered easy-to-hard chain; every sub-question sees all earlier answers (Zhou et al.).

    The author's order wins whenever the task states one (sequence markers or an
    enumerated list); only an unordered task is re-sorted easiest-first.
    """

    name = "least_to_most"
    chain = True

    def heuristic(self, task: str, *, plan: DecompositionPlan, profile: DomainProfile, features: Optional[TaskFeatures] = None, inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = ()) -> TaskGraph:
        f = features or extract_features(task, inputs=inputs, context=context, profile=profile)
        graph = heuristic_graph(task, plan=plan, profile=profile, config=self.config, features=f, inputs=inputs, context=context, strategy=self.name, chain=False)
        tasks = [t for t in graph.tasks if t.id != "verify"]
        verify = [t for t in graph.tasks if t.id == "verify"]
        if not f.sequence_markers and f.enumeration_items == 0:
            # no explicit order (no sequence marker, no numbered/inline list): easiest (shortest) first, the L2M ordering proxy
            tasks.sort(key=lambda t: (t.complexity != "low", len(t.description)))
        ordered = graph.with_tasks(tasks + verify)
        return chainify(ordered)


class DAGPlanner(Strategy):
    """LLMCompiler-style dependency graph with ``$id.output`` placeholders and a parallel frontier."""

    name = "dag"


class RecursiveDecomposer(Strategy):
    """ADaPT: attempt the whole task first; expand a node into 3-5 children only on failure, down to ``max_depth``."""

    name = "recursive"

    def heuristic(self, task: str, *, plan: DecompositionPlan, profile: DomainProfile, features: Optional[TaskFeatures] = None, inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = ()) -> TaskGraph:
        graph = DirectStrategy(self.config).heuristic(task, plan=plan, profile=profile, features=features, inputs=inputs, context=context)
        graph.strategy = self.name
        graph.metadata["recursive"] = True
        return graph

    async def build(self, task: str, *, request: TaskRequest, plan: DecompositionPlan, ctx: RunContext, features: Optional[TaskFeatures] = None) -> TaskGraph:
        return self.heuristic(task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)

    async def expand(
        self,
        node: SubTask,
        *,
        reason: str,
        depth: int,
        request: TaskRequest,
        plan: DecompositionPlan,
        ctx: RunContext,
    ) -> TaskGraph:
        """Decompose one failed node into a small sub-graph (LLM when live, heuristic otherwise)."""
        sub_plan = plan.model_copy(update={"max_breadth": max(1, min(self.config.max_subtasks, plan.max_breadth)), "max_depth": 2})
        if use_llm_path(ctx, self.config):
            feedback = f"<executor_feedback>\nThe whole sub-task was attempted at depth {depth} and did not succeed: {reason[:300]}\nSplit it into 3-5 smaller, shallower sub-tasks.\n</executor_feedback>"
            graph = await self.llm_decompose(ctx, request, task=node.description, plan=sub_plan, feedback=feedback, step=f"expand.{node.id}.llm")
            if graph is not None:
                graph.metadata["depth"] = depth + 1
                return graph
        graph = heuristic_graph(node.description, plan=sub_plan, profile=ctx.profile, config=self.config, inputs=request.inputs, context=request.context, strategy=self.name)
        if len(graph.tasks) <= 1:
            # nothing to split structurally: halve the description at the longest clause boundary
            words = node.description.split()
            half = max(1, len(words) // 2)
            a, b = " ".join(words[:half]), " ".join(words[half:])
            graph = TaskGraph(
                root_task=node.description,
                strategy=self.name,
                tasks=[
                    SubTask(id="t1", description=f"First part: {a}", expected_output="Partial result", max_retries=self.config.max_retries_per_task),
                    SubTask(id="t2", description=f"Second part: {b}", depends_on=["t1"], inputs={"first_part": "$t1.output"}, expected_output="Partial result", max_retries=self.config.max_retries_per_task),
                ],
            )
        graph.metadata["depth"] = depth + 1
        return graph


class TreeOfThoughtsPlanner(Strategy):
    """Bounded BFS over candidate decompositions with LLM value scoring (Yao et al. ToT).

    Level 1 proposes ``tot_breadth`` candidate graphs and scores each with
    ``tot_votes`` independent evaluator calls (different judging framings, no
    temperature). Each further level up to ``tot_depth`` keeps the two best
    candidates, asks for improved variants of each (``REFINE_TEMPLATE``) and
    votes again over parents plus children, so a refinement can never displace
    a better parent. The best-scoring graph wins; when no candidate reaches
    ``agreement_threshold`` the graph is flagged ``low_agreement``. Offline the
    candidates come from the heuristic splitter at several granularities and
    are scored structurally.
    """

    name = "tree_of_thoughts"
    keep_per_level: ClassVar[int] = 2

    _FRAMINGS: ClassVar[tuple[str, ...]] = (
        "Judge each candidate as a sceptical reviewer looking for missing requirements.",
        "Judge each candidate as the engineer who must execute it: are the steps solvable in isolation?",
        "Judge each candidate on how well its dependency structure matches the real data flow of the task.",
        "Judge each candidate on cost: does it avoid redundant or trivial steps while keeping coverage?",
        "Judge each candidate on verifiability: can every step be checked by an external reviewer or tool?",
    )

    def candidates_heuristic(self, task: str, *, plan: DecompositionPlan, profile: DomainProfile, features: Optional[TaskFeatures], inputs: Optional[dict[str, Any]], context: Sequence[Evidence]) -> list[TaskGraph]:
        """Model-free candidates differing in granularity and ordering."""
        cands: list[TaskGraph] = []
        for width in sorted({plan.max_breadth, max(2, plan.max_breadth // 2), min(self.config.max_subtasks, plan.max_breadth + 2)}):
            g = heuristic_graph(task, plan=plan, profile=profile, config=self.config, features=features, inputs=inputs, context=context, strategy=self.name, max_segments=width)
            cands.append(g)
        cands.append(chainify(heuristic_graph(task, plan=plan, profile=profile, config=self.config, features=features, inputs=inputs, context=context, strategy=self.name)))
        return cands[: self.config.tot_breadth]

    def select(self, candidates: Sequence[TaskGraph], scores: Sequence[float]) -> TaskGraph:
        best = max(range(len(candidates)), key=lambda i: (scores[i], -len(candidates[i].tasks)))
        chosen = candidates[best]
        chosen.metadata["candidate_scores"] = [round(s, 3) for s in scores]
        chosen.metadata["selected_index"] = best
        return chosen

    def heuristic(self, task: str, *, plan: DecompositionPlan, profile: DomainProfile, features: Optional[TaskFeatures] = None, inputs: Optional[dict[str, Any]] = None, context: Sequence[Evidence] = ()) -> TaskGraph:
        cands = self.candidates_heuristic(task, plan=plan, profile=profile, features=features, inputs=inputs, context=context)
        scores = [structural_score(c, task, max_nodes=self.config.max_nodes) for c in cands]
        return self.select(cands, scores)

    async def build(self, task: str, *, request: TaskRequest, plan: DecompositionPlan, ctx: RunContext, features: Optional[TaskFeatures] = None) -> TaskGraph:
        cfg = self.config
        if not use_llm_path(ctx, cfg):
            return self.heuristic(task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)
        n = cfg.tot_breadth
        votes = cfg.tot_votes
        rem = ctx.budget.remaining_calls()
        if rem is not None:
            # degrade sample counts before failing (keep room for solving)
            votes = max(1, min(votes, (rem - 1) // 2))
            if rem < 2:
                ctx.warn("tree_of_thoughts: budget too small for search; using heuristic candidates")
                return self.heuristic(task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)
        hints = "\n".join(f"- {h}" for h in ctx.profile.decomposition_hints) or "(none)"
        req = build_request(
            CANDIDATES_TEMPLATE,
            profile=ctx.profile,
            values={"task": task, "bounds": _bounds_text(plan, cfg, chain=False), "hints": hints, "documents": render_documents(request.context, max_items=cfg.max_context_docs), "n": n},
            schema=schema_for(CandidateGraphs),
            effort=cfg.planner_effort,
            tags=[self.name, "propose"],
        )
        candidates = await self._propose(ctx, req, task=task, plan=plan, step="tot.propose.llm", depth=1)
        if not candidates:
            ctx.warn("tree_of_thoughts: no valid candidate proposed; using heuristic candidates")
            return self.heuristic(task, plan=plan, profile=ctx.profile, features=features, inputs=request.inputs, context=request.context)
        if len(candidates) == 1:
            chosen = self.select(candidates, [1.0])
            chosen.metadata["search_depth"] = 1
            return chosen
        final = await self._vote(ctx, task, candidates, votes)
        depth_reached = 1
        # deeper levels: keep the best candidates, expand each into improved variants, vote again over parents + children
        for level in range(2, cfg.tot_depth + 1):
            if not can_call(ctx, 3):  # one expansion, one vote, one call left for solving
                break
            order = sorted(range(len(candidates)), key=lambda i: (-final[i], len(candidates[i].tasks)))
            parents = [candidates[i] for i in order[: self.keep_per_level]]
            children: list[TaskGraph] = []
            for parent in parents:
                if not can_call(ctx, 3):
                    break
                rreq = build_request(
                    REFINE_TEMPLATE,
                    profile=ctx.profile,
                    values={
                        "task": task,
                        "bounds": _bounds_text(plan, cfg, chain=False),
                        "hints": hints,
                        "documents": render_documents(request.context, max_items=cfg.max_context_docs),
                        "parent": parent.to_ascii(),
                        "n": max(2, n // 2),
                    },
                    schema=schema_for(CandidateGraphs),
                    effort=cfg.planner_effort,
                    tags=[self.name, "refine", f"level{level}"],
                )
                children.extend(await self._propose(ctx, rreq, task=task, plan=plan, step=f"tot.refine.{level}.llm", depth=level))
            if not children:
                break
            candidates = parents + children
            final = await self._vote(ctx, task, candidates, votes)
            depth_reached = level
        chosen = self.select(candidates, final)
        chosen.metadata["search_depth"] = depth_reached
        if max(final) < cfg.agreement_threshold:
            chosen.metadata["low_agreement"] = True
            ctx.warn(f"tree_of_thoughts: best candidate scored {max(final):.2f}, below agreement_threshold={cfg.agreement_threshold}; selection is weakly supported")
        return chosen

    async def _propose(self, ctx: RunContext, req: Any, *, task: str, plan: DecompositionPlan, step: str, depth: int) -> list[TaskGraph]:
        """One structured proposal call; returns only the candidates that pass validation within the plan bounds."""
        cfg = self.config
        out: list[TaskGraph] = []
        try:
            resp = await ctx.llm_call(req, step=step)
            drafts = validate_structured(resp.parsed, CandidateGraphs)
        except (BudgetExceededError, StructuredOutputError, LLMError) as exc:
            ctx.warn(f"tree_of_thoughts: candidate proposal failed ({type(exc).__name__})")
            return out
        for d in drafts.candidates:
            g = d.to_graph(task, strategy=self.name, max_retries=cfg.max_retries_per_task)
            g.metadata["planner"] = "llm"
            g.metadata["search_level"] = depth
            if g.tasks and not g.validate_structure(max_depth=plan.max_depth, max_breadth=plan.max_breadth, max_nodes=cfg.max_nodes, detect_duplicates=cfg.loop_detection):
                out.append(g)
        return out

    async def _vote(self, ctx: RunContext, task: str, candidates: Sequence[TaskGraph], votes: int) -> list[float]:
        """ToT value voting: ``votes`` independent low-effort evaluations under different framings, averaged (impossible -> 0)."""
        cfg = self.config
        rendered = "\n".join(f'<candidate index="{i}">\n{c.to_ascii()}\n</candidate>' for i, c in enumerate(candidates))
        collected: list[list[float]] = []
        for v in range(votes):
            if not can_call(ctx, 2):  # keep at least one call for solving
                break
            ereq = build_request(
                EVALUATE_TEMPLATE,
                profile=ctx.profile,
                values={"task": task, "candidates": rendered, "framing": self._FRAMINGS[v % len(self._FRAMINGS)]},
                schema=schema_for(CandidateScores),
                effort="low",
                tags=[self.name, "evaluate", f"vote{v}"],
            )
            try:
                eresp = await ctx.llm_call(ereq, step=f"tot.evaluate.{v}.llm")
                scores = validate_structured(eresp.parsed, CandidateScores)
            except (BudgetExceededError, StructuredOutputError, LLMError) as exc:
                ctx.warn(f"tree_of_thoughts: evaluation vote {v} failed ({type(exc).__name__})")
                continue
            vote = [0.5] * len(candidates)
            for s in scores.scores:
                if 0 <= s.index < len(candidates):
                    vote[s.index] = 0.0 if s.verdict == "impossible" else s.score
            collected.append(vote)
        if collected:
            return aggregate_votes(collected, len(candidates))
        return [structural_score(c, task, max_nodes=cfg.max_nodes) for c in candidates]


STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    "direct": DirectStrategy,
    "plan_and_solve": PlanAndSolve,
    "least_to_most": LeastToMost,
    "dag": DAGPlanner,
    "recursive": RecursiveDecomposer,
    "tree_of_thoughts": TreeOfThoughtsPlanner,
}


def get_strategy(name: str, config: Optional[TaskDecompositionConfig] = None) -> Strategy:
    """Instantiate the strategy registered under ``name`` (unknown names fall back to ``dag``)."""
    cls = STRATEGY_CLASSES.get(name, DAGPlanner)
    return cls(config)


__all__ = [
    "STRATEGY_CLASSES",
    "DAGPlanner",
    "DirectStrategy",
    "LeastToMost",
    "PlanAndSolve",
    "RecursiveDecomposer",
    "Strategy",
    "TreeOfThoughtsPlanner",
    "aggregate_votes",
    "can_call",
    "chainify",
    "content_words",
    "get_strategy",
    "heuristic_graph",
    "heuristic_segments",
    "stop_reason",
    "structural_score",
    "use_llm_path",
]
