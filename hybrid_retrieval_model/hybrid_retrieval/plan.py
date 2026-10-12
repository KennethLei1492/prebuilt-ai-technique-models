"""Retrieval plan schema, query feature analysis and the heuristic planner.

The plan is the auditable contract between *reasoning about the query* and
*executing retrieval* (dossier architecture recommendation 2): it names the
channels, weights, fusion method, reranker tier, k values, transforms and
filters, and carries a rationale. Two planners produce it:

* :class:`HeuristicPlanner` - deterministic rules over query features and the
  domain profile; used when ``ctx.offline`` or ``constraints.deterministic``.
* the LLM planner in :mod:`hybrid_retrieval.model` - one structured call that
  emits :class:`PlannerOutput`; converted and clamped by
  :meth:`RetrievalPlan.from_planner_output`.

Both paths end in :func:`apply_budget`, which degrades transforms and
reranking to fit ``Constraints.max_llm_calls`` instead of failing; cost and
latency budgets are checked at run time with :func:`budget_exhausted`.
"""
from __future__ import annotations

import math
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Constraints, DomainProfile, get_profile
from pipeline_core.utils import clamp

from .config import FusionMethod, HybridRetrievalConfig, RerankMode
from .sparse import DEFAULT_STOPWORDS, Tokenizer, looks_like_identifier

QueryType = Literal["identifier", "keyword", "natural_language", "exploratory", "mixed"]
QueryTransform = Literal["multi_query", "hyde", "query2doc", "keywords"]
FilterOp = Literal["eq", "ne", "in", "contains", "gte", "lte"]
PlanSource = Literal["heuristic", "llm", "explicit"]

_QUESTION_WORDS = frozenset("what which who whom whose when where why how does do is are can should could would will list explain describe compare find show give".split())
_EXPLORATORY_CUES = ("similar", "like this", "anything", "before", "history", "overview", "related", "examples", "precedent", "happened", "ever", "lessons", "background", "comparable")
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_QUOTE_RE = re.compile(r"\"([^\"]+)\"|'([^']{3,})'")


class QueryFeatures(BaseModel):
    """Cheap, deterministic features the planners reason over."""

    model_config = ConfigDict(extra="forbid")

    n_tokens: int = 0
    n_content_tokens: int = 0
    identifier_density: float = Field(default=0.0, description="Share of tokens that look like identifiers or codes")
    number_density: float = 0.0
    has_question_word: bool = False
    is_exploratory: bool = False
    quoted_phrases: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)

    def render(self) -> str:
        return (
            f"tokens={self.n_tokens} content_tokens={self.n_content_tokens} identifier_density={self.identifier_density:.2f} "
            f"numbers={int(round(self.number_density * max(1, self.n_tokens)))} question_word={str(self.has_question_word).lower()} "
            f"exploratory={str(self.is_exploratory).lower()} identifiers={self.identifiers} quoted={self.quoted_phrases}"
        )


def analyze_query(query: str) -> QueryFeatures:
    """Compute :class:`QueryFeatures` without any model call."""
    raw_tokens = re.findall(r"\S+", query)
    words = [t.strip(".,;:!?()[]{}\"'") for t in raw_tokens]
    words = [w for w in words if w]
    n = len(words)
    if n == 0:
        return QueryFeatures()
    identifiers = [w for w in words if looks_like_identifier(w) and not _NUMBER_RE.match(w)]
    numbers = [w for w in words if _NUMBER_RE.match(w)]
    lowered = [w.lower() for w in words]
    content = [w for w in lowered if w not in DEFAULT_STOPWORDS and len(w) > 1]
    q_lower = query.lower()
    return QueryFeatures(
        n_tokens=n,
        n_content_tokens=len(content),
        identifier_density=len(identifiers) / n,
        number_density=len(numbers) / n,
        has_question_word=lowered[0] in _QUESTION_WORDS or q_lower.rstrip().endswith("?"),
        is_exploratory=any(cue in q_lower for cue in _EXPLORATORY_CUES),
        quoted_phrases=[a or b for a, b in _QUOTE_RE.findall(query)],
        identifiers=identifiers,
    )


class FilterClause(BaseModel):
    """A metadata constraint expressible in a strict JSON schema (``value`` is a string; ``in`` is comma-separated)."""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: FilterOp
    value: str

    def to_filter(self) -> dict[str, Any]:
        if self.op == "eq":
            return {self.field: {"$eq": self.value}}
        if self.op == "in":
            return {self.field: {"$in": [v.strip() for v in self.value.split(",") if v.strip()]}}
        return {self.field: {f"${self.op}": self.value}}


class PlannerOutput(BaseModel):
    """What the LLM planner emits. All fields required, no numeric bounds (clamped later)."""

    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    query_variants: list[str] = Field(description="Original query first, then up to 3 reformulations preserving identifiers")
    use_sparse: bool
    use_dense: bool
    sparse_weight: float = Field(description="Lexical weight between 0 and 1; dense weight is its complement")
    fusion: FusionMethod
    k_sparse: int
    k_dense: int
    k_final: int
    rerank: RerankMode
    rerank_depth: int
    query_transforms: list[QueryTransform]
    filters: list[FilterClause]
    rationale: str


class RetrievalPlan(BaseModel):
    """The executable plan. Attached to ``result.metadata["plan"]``."""

    model_config = ConfigDict(extra="forbid")

    query: str
    query_type: QueryType = "mixed"
    query_variants: list[str] = Field(default_factory=list, description="Queries to run (original first); extra variants are fused with RRF")
    use_sparse: bool = True
    use_dense: bool = True
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    fusion: FusionMethod = "rrf"
    k_sparse: int = Field(default=100, ge=1)
    k_dense: int = Field(default=100, ge=1)
    k_final: int = Field(default=10, ge=1)
    rerank: RerankMode = "none"
    rerank_depth: int = Field(default=20, ge=1)
    query_transforms: list[QueryTransform] = Field(default_factory=list)
    filters: list[FilterClause] = Field(default_factory=list, description="Planner-inferred filters")
    mandatory_filters: dict[str, Any] = Field(default_factory=dict, description="Caller-supplied filters (ACL/tenant); the planner cannot remove them")
    rationale: str = ""
    source: PlanSource = "heuristic"
    features: QueryFeatures = Field(default_factory=QueryFeatures)
    expected_llm_calls: int = 0
    degradations: list[str] = Field(default_factory=list, description="Budget-driven changes applied by apply_budget")

    # ------------------------------------------------------------- helpers
    def filter_dict(self) -> dict[str, Any]:
        """Merged filter specification for the document store (mandatory filters win)."""
        out: dict[str, Any] = {}
        for c in self.filters:
            out.update(c.to_filter())
        out.update(self.mandatory_filters)
        return out

    def all_queries(self) -> list[str]:
        qs = [self.query] + [v for v in self.query_variants if v and v != self.query]
        seen: set[str] = set()
        out: list[str] = []
        for q in qs:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        return out

    def describe(self) -> str:
        channels = "+".join(c for c, on in (("sparse", self.use_sparse), ("dense", self.use_dense)) if on) or "none"
        parts = [
            f"query_type={self.query_type}",
            f"channels={channels}",
            (f"fusion={self.fusion}(sparse={self.sparse_weight:.2f}, dense={self.dense_weight:.2f})" if self.fusion in ("convex", "dbsf") else f"fusion={self.fusion}"),
            f"k={self.k_sparse}/{self.k_dense}->{self.k_final}",
            f"rerank={self.rerank}@{self.rerank_depth}" if self.rerank != "none" else "rerank=none",
            f"transforms={self.query_transforms or '[]'}",
            f"variants={len(self.all_queries())}",
        ]
        if self.filters or self.mandatory_filters:
            parts.append(f"filters={self.filter_dict()}")
        parts.append(f"llm_calls~{self.expected_llm_calls}")
        parts.append(f"source={self.source}")
        return " ".join(parts)

    @classmethod
    def from_planner_output(
        cls,
        out: PlannerOutput,
        *,
        query: str,
        config: HybridRetrievalConfig,
        features: QueryFeatures | None = None,
        mandatory_filters: dict[str, Any] | None = None,
        corpus_fields: dict[str, list[str]] | None = None,
    ) -> "RetrievalPlan":
        """Clamp an LLM plan into valid ranges; drop filters on unknown fields; keep the sparse channel on.

        ``k_sparse``/``k_dense`` are clamped to [1, 1000], ``k_final`` to [1, 100] and
        ``rerank_depth`` to [1, 200] (never below ``k_final``).
        """
        sw = clamp(float(out.sparse_weight), 0.0, 1.0)
        filters = list(out.filters)
        if corpus_fields is not None:
            filters = [f for f in filters if f.field in corpus_fields]
        variants = [v.strip() for v in out.query_variants if v and v.strip()][:4]
        # The sparse channel is never disabled by a planner (BM25 stays first-class; Thakur et al. 2021).
        # Only an explicit plan in inputs["plan"] may run dense-only.
        use_sparse, use_dense = True, bool(out.use_dense) or not bool(out.use_sparse)
        plan = cls(
            query=query,
            query_type=out.query_type,
            query_variants=variants,
            use_sparse=use_sparse,
            use_dense=use_dense,
            sparse_weight=sw,
            dense_weight=config.dense_weight_for(sw),
            fusion=out.fusion,
            k_sparse=int(clamp(out.k_sparse, 1, 1000)),
            k_dense=int(clamp(out.k_dense, 1, 1000)),
            k_final=int(clamp(out.k_final, 1, 100)),
            rerank=out.rerank,
            rerank_depth=int(clamp(out.rerank_depth, 1, 200)),
            query_transforms=list(dict.fromkeys(out.query_transforms)),
            filters=filters,
            mandatory_filters=dict(mandatory_filters or {}),
            rationale=out.rationale.strip(),
            source="llm",
            features=features or analyze_query(query),
        )
        if plan.rerank_depth < plan.k_final:
            plan.rerank_depth = plan.k_final
        return plan


# ---------------------------------------------------------------------------
# Cost model and budget
# ---------------------------------------------------------------------------


def n_windows(depth: int, window: int, stride: int) -> int:
    """Number of sliding windows a listwise reranker needs over ``depth`` passages."""
    if depth <= window:
        return 1
    return 1 + math.ceil((depth - window) / stride)


def estimate_llm_calls(plan: RetrievalPlan, config: HybridRetrievalConfig) -> int:
    calls = sum(1 for t in plan.query_transforms if t in ("multi_query", "hyde", "query2doc"))
    if plan.rerank == "listwise":
        calls += n_windows(plan.rerank_depth, config.rerank_window, config.rerank_stride)
    elif plan.rerank == "pointwise":
        calls += plan.rerank_depth
    return calls


def apply_budget(plan: RetrievalPlan, config: HybridRetrievalConfig, *, remaining_calls: Optional[int], deterministic: bool = False) -> RetrievalPlan:
    """Degrade the plan to fit the call budget: drop transforms first, then shrink and finally disable reranking.

    ``deterministic`` removes every step that samples an LLM (transforms other than
    the free ``keywords`` extraction, listwise/pointwise reranking); a configured
    ``cross_encoder`` reranker is kept because it is free and reproducible.
    Only ``max_llm_calls`` can be modelled up front; cost and latency budgets are
    enforced at run time by :func:`budget_exhausted` checks before every call.
    """
    if deterministic:
        llm_rerank = plan.rerank in ("listwise", "pointwise")
        if llm_rerank or any(t != "keywords" for t in plan.query_transforms):
            plan.degradations.append("deterministic: LLM transforms and reranking disabled")
        plan.query_transforms = [t for t in plan.query_transforms if t == "keywords"]
        if llm_rerank:  # a cross-encoder costs no LLM call and is reproducible, so it stays
            plan.rerank = "none"
    if remaining_calls is not None:
        for t in ("query2doc", "hyde", "multi_query"):
            if estimate_llm_calls(plan, config) <= remaining_calls:
                break
            if t in plan.query_transforms:
                plan.query_transforms.remove(t)
                plan.degradations.append(f"budget: dropped {t}")
        if plan.rerank == "pointwise" and estimate_llm_calls(plan, config) > remaining_calls:
            plan.rerank = "listwise"
            plan.degradations.append("budget: pointwise -> listwise")
        if plan.rerank == "listwise" and estimate_llm_calls(plan, config) > remaining_calls:
            plan.rerank_depth = max(plan.k_final, min(plan.rerank_depth, config.rerank_window))
            plan.degradations.append("budget: rerank depth reduced to one window")
        if plan.rerank != "none" and estimate_llm_calls(plan, config) > remaining_calls:
            plan.rerank = "none"
            plan.degradations.append("budget: reranking disabled")
    plan.expected_llm_calls = estimate_llm_calls(plan, config)
    return plan


def budget_exhausted(ctx: Any) -> Optional[str]:
    """Why ``ctx`` cannot afford another LLM call (spec 3.2), or ``None`` when it can.

    Checks, in order, the latency budget (``ctx.remaining_time()``), the call budget
    (``ctx.budget.remaining_calls()``) and the cost budget (``ctx.budget.remaining_cost()``).
    Callers consult it before every transform, rerank window or grading fan-out and
    degrade (skip the step, keep the fused order) instead of raising.
    ``ctx=None`` (direct API without a RunContext) is never exhausted.
    """
    if ctx is None:
        return None
    constraints = ctx.request.constraints
    if ctx.deadline_exceeded():
        return f"latency budget {constraints.max_latency_s}s exhausted"
    remaining_calls = ctx.budget.remaining_calls()
    if remaining_calls is not None and remaining_calls <= 0:
        return f"LLM call budget of {constraints.max_llm_calls} exhausted"
    remaining_cost = ctx.budget.remaining_cost()
    if remaining_cost is not None and remaining_cost <= 0.0:
        return f"cost budget ${constraints.max_cost_usd:.4f} exhausted (spent ${ctx.budget.spent.cost_usd:.4f})"
    return None


# ---------------------------------------------------------------------------
# Heuristic planner
# ---------------------------------------------------------------------------


class HeuristicPlanner:
    """Deterministic planner from query features and the domain profile.

    Rules (mirrored in the README):

    1. ``sparse_weight`` starts at the profile's ``retrieval_hints["sparse_weight"]``
       (else ``config.default_sparse_weight`` = 0.3, Bruch et al. 2023), +0.25 when
       at least half the tokens are identifiers, +0.10 when some are, +0.10 for
       quoted phrases, -0.10 for identifier-free questions; clamped to [0.1, 0.9].
    2. ``query_type``: identifier (>= 50% identifier tokens), keyword (<= 3 content
       tokens, no question word), exploratory (similarity cues), natural_language
       (question word or >= 6 tokens), else mixed.
    3. ``fusion``: convex when the profile supplies a calibrated weight or the query
       is identifier/keyword (exact terms must dominate); otherwise RRF.
    4. Both channels stay on except for a bare identifier of <= 2 tokens (dense off).
    5. Transforms: ``hyde`` for exploratory or short natural-language queries
       (< ``expansion_max_query_tokens`` content tokens); ``multi_query`` for
       natural-language/exploratory queries of >= 6 tokens; ``query2doc`` for short
       keyword queries; ``keywords`` (free) for long questions (> 12 tokens).
    6. ``rerank``: the configured tier for natural-language, exploratory and mixed
       queries, always for high/safety-critical profiles (depth 30 when safety-critical),
       never for bare identifier/keyword lookups in low/medium-risk domains.
    7. :func:`apply_budget` then fits the plan to ``max_llm_calls``/``deterministic``.
    """

    def __init__(self, config: HybridRetrievalConfig | None = None) -> None:
        self.config = config or HybridRetrievalConfig()

    def plan(
        self,
        query: str,
        *,
        profile: DomainProfile | None = None,
        constraints: Constraints | None = None,
        mandatory_filters: dict[str, Any] | None = None,
        remaining_calls: Optional[int] = None,
        k_final: int | None = None,
    ) -> RetrievalPlan:
        cfg = self.config
        profile = profile or get_profile("general")
        constraints = constraints or Constraints()
        f = analyze_query(query)
        reasons: list[str] = []

        hint = profile.retrieval_hints.get("sparse_weight")
        base = float(hint) if isinstance(hint, (int, float)) else cfg.default_sparse_weight
        sw = base
        reasons.append(f"domain '{profile.name}' sparse_weight base {base:.2f}")
        if f.identifier_density >= 0.5:
            sw += 0.25
            reasons.append("identifier-dense query -> +0.25 lexical")
        elif f.identifier_density > 0:
            sw += 0.10
            reasons.append("contains identifiers -> +0.10 lexical")
        if f.quoted_phrases:
            sw += 0.10
            reasons.append("quoted phrase -> +0.10 lexical")
        if f.has_question_word and f.identifier_density == 0:
            sw -= 0.10
            reasons.append("identifier-free question -> -0.10 lexical")
        sw = clamp(round(sw, 2), 0.1, 0.9)

        if f.identifier_density >= 0.5:
            qtype: QueryType = "identifier"
        elif f.is_exploratory:
            qtype = "exploratory"
        elif f.n_content_tokens <= 3 and not f.has_question_word:
            qtype = "keyword"
        elif f.has_question_word or f.n_tokens >= 6:
            qtype = "natural_language"
        else:
            qtype = "mixed"
        reasons.append(f"query_type={qtype}")

        fusion: FusionMethod = "convex" if (hint is not None or qtype in ("identifier", "keyword")) else cfg.fusion
        reasons.append(f"fusion={fusion} ({'calibrated domain weight' if hint is not None else 'exact terms dominate' if fusion == 'convex' else 'zero-config default'})")

        use_dense = not (qtype == "identifier" and f.n_tokens <= 2)
        if not use_dense:
            reasons.append("bare identifier lookup -> dense channel off")

        transforms: list[QueryTransform] = []
        if use_dense and (qtype == "exploratory" or (qtype == "natural_language" and f.n_content_tokens < cfg.expansion_max_query_tokens)):
            transforms.append("hyde")
            reasons.append("short/exploratory query -> HyDE")
        if qtype in ("natural_language", "exploratory") and f.n_tokens >= 6:
            transforms.append("multi_query")
            reasons.append("multi-facet question -> multi-query")
        if qtype == "keyword" and f.n_content_tokens < cfg.expansion_max_query_tokens and f.identifier_density == 0:
            transforms.append("query2doc")
            reasons.append("short keyword query -> Query2doc")
        if f.n_tokens > 12:
            transforms.append("keywords")
            reasons.append("long question -> keyword extraction for lexical leg")

        high_risk = profile.risk_level in ("high", "safety_critical")
        rerank: RerankMode = "none"
        depth = cfg.rerank_depth
        if high_risk:
            rerank = cfg.rerank if cfg.rerank != "none" else "listwise"
            depth = max(depth, 30) if profile.risk_level == "safety_critical" else depth
            reasons.append(f"{profile.risk_level} domain -> mandatory {rerank} rerank@{depth}")
        elif qtype in ("natural_language", "exploratory", "mixed") and cfg.rerank != "none":
            rerank = cfg.rerank
            reasons.append(f"precision matters -> {rerank} rerank")
        else:
            reasons.append("exact lexical lookup -> no rerank")

        plan = RetrievalPlan(
            query=query,
            query_type=qtype,
            query_variants=[query],
            use_sparse=True,
            use_dense=use_dense,
            sparse_weight=sw,
            dense_weight=cfg.dense_weight_for(sw),
            fusion=fusion,
            k_sparse=cfg.k_sparse,
            k_dense=cfg.k_dense,
            k_final=k_final or cfg.k_final,
            rerank=rerank,
            rerank_depth=max(depth, k_final or cfg.k_final),
            query_transforms=transforms,
            mandatory_filters=dict(mandatory_filters or {}),
            rationale="; ".join(reasons),
            source="heuristic",
            features=f,
        )
        return apply_budget(plan, cfg, remaining_calls=remaining_calls, deterministic=constraints.deterministic)
