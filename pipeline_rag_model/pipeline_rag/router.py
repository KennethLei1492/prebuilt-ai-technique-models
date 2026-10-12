"""Query routing: decide whether and how to retrieve.

Adaptive-RAG (Jeong et al. 2024) showed that a cheap complexity classifier
choosing between no / single-step / multi-step retrieval matches multi-step
accuracy at ~40% of its latency. This module provides:

* :func:`extract_task_features` - deterministic surface features of a task;
* :func:`route_for_features` - the heuristic routing rules (offline path);
* :class:`QueryRouter` - the same decision through one low-effort structured
  LLM call when a model is available, with the heuristics as fallback.

Routes: ``no_retrieval | single_step | multi_step | agentic``. ``agentic`` is
used for corpus-wide ("themes across all incidents") questions, which chunk
retrieval cannot answer (GraphRAG/RAPTOR); it builds document summaries and
runs the bounded reflection loop.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import DomainProfile, Evidence, LLMRequest, RunContext, sanitize_schema, tokenize_words

from .config import PipelineRAGConfig

Route = Literal["no_retrieval", "single_step", "multi_step", "agentic"]
ROUTE_ORDER: tuple[str, ...] = ("no_retrieval", "single_step", "multi_step", "agentic")

_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]{1,6}-\d{1,}[A-Z0-9-]*|[A-Z]{2,}\d+[A-Z0-9]*|[A-Z]\d{2,}[A-Z0-9]*|\d+[A-Z]{1,3}\d*|0x[0-9A-Fa-f]+|[a-z]+_[a-z_0-9]+|[a-z]+[A-Z][A-Za-z0-9]+|[A-Za-z_][A-Za-z0-9_]*\.[a-z_][A-Za-z0-9_]*\(\)?)\b"
)
_MULTI_HOP_WORDS = (
    "compare", "comparison", "difference between", "differences between", "versus", " vs ", " vs. ", "both ",
    "and then", "trace", "relationship", "relate", "led to", "lead to", "caused", "root cause", "impact of",
    "depend", "which of", "across", "each of", "respectively", "before and after", "between the", "chain",
)
_GLOBAL_WORDS = (
    "overall", "themes", "theme", "patterns", "pattern", "recurring", "trend", "summarize all", "summarise all",
    "across all", "all documents", "entire corpus", "whole corpus", "most frequent", "landscape", "in general across",
)
_NO_RETRIEVAL_WORDS = (
    "rewrite", "rephrase", "translate", "paraphrase", "convert the following", "format this", "proofread",
    "fix the grammar", "capitalize", "reverse the string", "sort the following",
)
_WHY_HOW_RE = re.compile(r"\b(why|how does|how do|how is|how are|how can|how should|what causes|what governs|principle)\b", re.I)
_NUMERIC_RE = re.compile(r"\b(how many|how much|value|rating|maximum|minimum|limit|tolerance|percent|ratio|torque|voltage|current|temperature|load|stress|margin|factor of safety)\b", re.I)
_QUESTION_WORDS = ("what", "which", "who", "when", "where", "why", "how", "does", "is", "are", "can", "should", "list", "explain", "describe", "summarize", "summarise")


class TaskFeatures(BaseModel):
    """Surface features of the task used by the heuristic planner and shown to the LLM planner."""

    model_config = ConfigDict(extra="forbid")

    word_count: int = 0
    question_count: int = 0
    question_words: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    has_identifiers: bool = False
    multi_hop_signals: list[str] = Field(default_factory=list)
    global_signals: list[str] = Field(default_factory=list)
    transformation_signals: list[str] = Field(default_factory=list)
    why_how: bool = False
    numeric: bool = False
    enumerations: int = Field(default=0, description="Number of comma/and separated items in the task")
    document_count: int = 0
    corpus_tokens: int = 0
    has_retriever: bool = False
    has_context: bool = False
    risk_level: str = "low"

    @property
    def has_corpus(self) -> bool:
        return self.document_count > 0 or self.has_retriever or self.has_context


class RouteDecision(BaseModel):
    """Outcome of routing. ``complexity`` mirrors Adaptive-RAG's labels."""

    model_config = ConfigDict(extra="forbid")

    route: Route
    complexity: Literal["trivial", "simple", "complex", "global"] = "simple"
    reason: str = ""


def _iter_documents(inputs: dict[str, Any]) -> list[Any]:
    docs = inputs.get("documents")
    if isinstance(docs, dict):
        return list(docs.values())
    if isinstance(docs, (list, tuple)):
        return list(docs)
    return []


def _estimate_corpus_tokens(inputs: dict[str, Any]) -> int:
    total = 0
    for d in _iter_documents(inputs):
        if isinstance(d, str):
            total += len(d)
        elif isinstance(d, dict):
            total += len(str(d.get("text") or d.get("content") or ""))
        elif isinstance(d, Evidence):
            total += len(d.content)
        else:
            total += len(str(getattr(d, "text", getattr(d, "content", ""))))
    return max(0, total // 4)


def extract_task_features(
    task: str,
    inputs: Optional[dict[str, Any]] = None,
    *,
    context: Sequence[Evidence] = (),
    has_retriever: bool = False,
    profile: Optional[DomainProfile] = None,
) -> TaskFeatures:
    """Compute the surface features the planner reasons over. Pure and deterministic."""
    inputs = inputs or {}
    low = f" {task.lower()} "
    words = tokenize_words(task)
    identifiers = sorted({m for m in _IDENTIFIER_RE.findall(task) if len(m) >= 3})
    multi = [w for w in _MULTI_HOP_WORDS if w in low]
    glob = [w for w in _GLOBAL_WORDS if w in low]
    transform = [w for w in _NO_RETRIEVAL_WORDS if w in low]
    qwords = [w for w in words[:3] if w in _QUESTION_WORDS]
    enumerations = len(re.findall(r",\s|\band\b", task.lower()))
    return TaskFeatures(
        word_count=len(words),
        question_count=task.count("?"),
        question_words=qwords,
        identifiers=identifiers,
        has_identifiers=bool(identifiers),
        multi_hop_signals=multi,
        global_signals=glob,
        transformation_signals=transform,
        why_how=bool(_WHY_HOW_RE.search(task)),
        numeric=bool(_NUMERIC_RE.search(task)),
        enumerations=enumerations,
        document_count=len(_iter_documents(inputs)),
        corpus_tokens=_estimate_corpus_tokens(inputs),
        has_retriever=has_retriever,
        has_context=len(context) > 0,
        risk_level=(profile.risk_level if profile else "low"),
    )


def route_for_features(features: TaskFeatures) -> RouteDecision:
    """Adaptive-RAG style complexity routing from surface features (pure)."""
    if not features.has_corpus:
        return RouteDecision(
            route="no_retrieval",
            complexity="trivial",
            reason="no documents, retriever or upstream evidence are available; answer from parametric knowledge with low confidence",
        )
    if features.transformation_signals and features.question_count == 0 and not features.multi_hop_signals:
        return RouteDecision(
            route="no_retrieval",
            complexity="trivial",
            reason=f"text transformation ({features.transformation_signals[0]!r}) needs no evidence",
        )
    if features.global_signals:
        return RouteDecision(
            route="agentic",
            complexity="global",
            reason=f"corpus-wide signal {features.global_signals[0]!r}: global questions need document summaries, not chunk search (GraphRAG/RAPTOR)",
        )
    score = len(features.multi_hop_signals)
    if features.question_count >= 2:
        score += 1
    if features.word_count > 30:
        score += 1
    if len(features.identifiers) >= 3:
        score += 1
    if features.enumerations >= 3 and features.why_how:
        score += 1
    if features.multi_hop_signals or score >= 2:
        reason = features.multi_hop_signals[0].strip() if features.multi_hop_signals else "several sub-questions"
        return RouteDecision(
            route="multi_step",
            complexity="complex",
            reason=f"multi-hop signal ({reason!r}): evidence from more than one document must be chained (IRCoT)",
        )
    return RouteDecision(route="single_step", complexity="simple", reason="focused question answerable from one retrieval round")


class QueryRouter:
    """Adaptive-RAG style router with heuristics offline and one low-effort LLM call online."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    def classify(self, task: str, inputs: Optional[dict[str, Any]] = None, *, features: Optional[TaskFeatures] = None) -> RouteDecision:
        """Heuristic classification (no LLM)."""
        feats = features or extract_task_features(task, inputs)
        return route_for_features(feats)

    async def classify_llm(self, task: str, features: TaskFeatures, ctx: RunContext) -> RouteDecision:
        """One structured, low-effort LLM call; falls back to heuristics on any failure."""
        from .prompts import ROUTE_PROMPT, make_request

        heuristic = route_for_features(features)
        try:
            req = make_request(
                ROUTE_PROMPT,
                ctx.profile,
                schema=sanitize_schema(RouteDecision.model_json_schema()),
                effort=self.config.grading_effort,
                model=self.config.sub_call_model,
                task=task,
                features=features.model_dump_json(),
                heuristic=heuristic.route,
            )
            resp = await ctx.llm_call(req, step="route")
            return RouteDecision.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - routing must never fail the run
            ctx.warn(f"LLM routing failed ({type(exc).__name__}); using heuristic route {heuristic.route}")
            return heuristic


def next_route(route: str) -> Optional[str]:
    """The next more expensive route for Self-Route style escalation, or None at the top."""
    i = ROUTE_ORDER.index(route)
    return ROUTE_ORDER[i + 1] if i + 1 < len(ROUTE_ORDER) else None


__all__ = [
    "ROUTE_ORDER",
    "QueryRouter",
    "Route",
    "RouteDecision",
    "TaskFeatures",
    "extract_task_features",
    "next_route",
    "route_for_features",
]

_ = LLMRequest  # re-exported for type checkers; requests are built in prompts.make_request
