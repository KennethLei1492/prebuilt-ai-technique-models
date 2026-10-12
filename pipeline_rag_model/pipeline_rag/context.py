"""Context processing between retrieval and generation.

* :class:`ContextGrader` - CRAG (Yan et al. 2024): score each document on
  [-1, 1]; at least one above the upper threshold -> ``correct`` (keep the
  relevant strips), all below the lower threshold -> ``incorrect`` (discard,
  fall back or abstain), otherwise ``ambiguous``. Offline the score is a
  lexical overlap; online it is one listwise low-effort LLM call.
* :class:`ContextCompressor` - RECOMP-style extractive compression (Xu et al.
  2024): keep the most question-relevant sentences of each unit within a
  token budget; never compress code, tables or identifier lists (LLMLingua
  pitfall); every compressed unit keeps its source span pointer (ALCE).
* :class:`ContextOrderer` - "Lost in the Middle" (Liu et al. 2024) and
  LongLLMLingua: best unit first, second best last, the rest in the middle.
* :func:`deduplicate` - exact and near-duplicate removal.

Everything except the LLM grading path is a pure function.
"""
from __future__ import annotations

import re
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, RunContext, estimate_tokens, normalize_text, render_evidence, sanitize_schema, tokenize_words

from .budget import note_degraded
from .config import PipelineRAGConfig
from .prompts import GRADE_PROMPT, make_request

GradeLabel = Literal["correct", "ambiguous", "incorrect"]
GradeDecision = Literal["correct", "ambiguous", "incorrect"]

_STOP = frozenset(
    "a an the of to in on for and or is are was were be been being do does did what which who when where why how can could "
    "should would will this that these those it its their there with from by as at into about".split()
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])|\n+")
_CODE_HINTS = re.compile(r"(^\s{4,}\S|[{};]\s*$|\bdef\b|\bclass\b|\breturn\b|=>|\|\s*\S+\s*\||^\s*[-*]\s+\S+:\s|\t)", re.M)


class DocGrade(BaseModel):
    """Relevance grade for one document (LLM output or heuristic)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    label: GradeLabel
    score: float = Field(default=0.0, description="-1 unrelated .. 1 directly answers")
    reason: str = ""


class GradeBatch(BaseModel):
    """Structured output of the listwise grading call."""

    model_config = ConfigDict(extra="forbid")

    grades: list[DocGrade]


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize_words(text) if t not in _STOP and len(t) > 1}


def lexical_relevance(query: str, text: str) -> float:
    """Overlap-based relevance on [-1, 1]: fraction of query content terms present in ``text``, mapped linearly.

    Identifier-like query terms (containing digits, hyphens or underscores) count double because
    exact-token matches dominate relevance for engineering lookups.
    """
    q = content_terms(query)
    if not q:
        return 0.0
    t = content_terms(text)
    weight = 0.0
    hit = 0.0
    for term in q:
        w = 2.0 if re.search(r"[\d_-]", term) else 1.0
        weight += w
        if term in t:
            hit += w
    frac = hit / weight if weight else 0.0
    return max(-1.0, min(1.0, 2.0 * frac - 1.0))


def label_for_score(score: float, upper: float, lower: float) -> GradeLabel:
    if score >= upper:
        return "correct"
    if score <= lower:
        return "incorrect"
    return "ambiguous"


def crag_decision(scores: Sequence[float], upper: float, lower: float) -> GradeDecision:
    """CRAG three-way decision over the graded scores (pure)."""
    if not scores:
        return "incorrect"
    if any(s >= upper for s in scores):
        return "correct"
    if all(s <= lower for s in scores):
        return "incorrect"
    return "ambiguous"


def split_sentences(text: str) -> list[str]:
    """Sentence/line split that keeps list items and code lines intact."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def looks_like_code_or_table(text: str) -> bool:
    """Heuristic: code, tables and identifier lists must not be compressed."""
    if "```" in text or "|" in text and text.count("|") >= 4:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    hinted = sum(1 for ln in lines if _CODE_HINTS.search(ln))
    return hinted / len(lines) >= 0.4


def deduplicate(evidence: Sequence[Evidence]) -> list[Evidence]:
    """Drop repeated ids and near-duplicate contents (normalised text), keeping the higher-scored copy's position."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    out: list[Evidence] = []
    for ev in evidence:
        key = normalize_text(ev.content)[:400]
        if ev.id in seen_ids or key in seen_text:
            continue
        seen_ids.add(ev.id)
        seen_text.add(key)
        out.append(ev)
    return out


def extractive_compress(
    evidence: Sequence[Evidence],
    query: str,
    *,
    token_budget: int,
    sentences_per_unit: int = 2,
    protect_code: bool = True,
) -> list[Evidence]:
    """Keep the ``sentences_per_unit`` most query-relevant sentences per unit, then fit the token budget.

    Units are processed in their given order (already relevance-ordered). A
    unit is only compressed when it has more sentences than the quota; code,
    tables and identifier lists are kept whole when ``protect_code`` is set.
    Compressed units record ``metadata["compressed"]``, ``original_chars``
    and keep ``span`` so citations resolve to the original text. Units that do
    not fit the remaining budget are dropped (lowest relevance last).
    """
    out: list[Evidence] = []
    used = 0
    for ev in evidence:
        text = ev.content
        compressed = False
        if not (protect_code and looks_like_code_or_table(text)):
            sents = split_sentences(text)
            if len(sents) > sentences_per_unit:
                scored = sorted(
                    ((lexical_relevance(query, s), -i, s) for i, s in enumerate(sents)),
                    key=lambda x: (-x[0], -x[1]),
                )
                keep_idx = sorted(-i for _, i, _ in scored[:sentences_per_unit])
                text = " ".join(sents[i] for i in keep_idx)
                compressed = True
        toks = estimate_tokens(text)
        if used + toks > token_budget:
            remaining = token_budget - used
            if remaining >= 40 and not (protect_code and looks_like_code_or_table(text)):
                # leave one token for the " ..." marker so the unit fits the budget exactly
                text = text[: max(0, remaining - 1) * 4].rsplit(" ", 1)[0] + " ..."
                toks = estimate_tokens(text)
                compressed = True
            else:
                continue
        used += toks
        meta = dict(ev.metadata)
        if compressed:
            meta.update(compressed=True, original_chars=len(ev.content))
        out.append(ev.model_copy(update={"content": text, "metadata": meta}))
    return out


def order_for_position(evidence: Sequence[Evidence], key: Optional[str] = None) -> list[Evidence]:
    """Best first, second best last, remainder by descending relevance in the middle (Lost in the Middle)."""
    items = list(evidence)
    if len(items) <= 2:
        return sorted(items, key=lambda e: -_rel(e, key))
    ranked = sorted(items, key=lambda e: -_rel(e, key))
    first, second, rest = ranked[0], ranked[1], ranked[2:]
    return [first, *rest, second]


def _rel(ev: Evidence, key: Optional[str]) -> float:
    if key and key in ev.metadata:
        try:
            return float(ev.metadata[key])
        except (TypeError, ValueError):
            pass
    g = ev.metadata.get("grade_score")
    if isinstance(g, (int, float)):
        return float(g)
    return float(ev.score)


class ContextGrader:
    """CRAG-style evidence gate."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    def grade_heuristic(self, question: str, evidence: Sequence[Evidence]) -> list[DocGrade]:
        cfg = self.config
        out = []
        for ev in evidence:
            s = lexical_relevance(question, f"{ev.metadata.get('prefix', '')} {ev.content}")
            out.append(DocGrade(doc_id=ev.id, label=label_for_score(s, cfg.grade_upper_threshold, cfg.grade_lower_threshold), score=s, reason="lexical overlap"))
        return out

    async def grade(
        self,
        question: str,
        evidence: Sequence[Evidence],
        *,
        ctx: Optional[RunContext] = None,
        use_llm: bool = False,
    ) -> tuple[list[Evidence], GradeDecision, list[DocGrade]]:
        """Return (evidence annotated with grades and filtered of 'incorrect' units, decision, raw grades)."""
        if not evidence:
            return [], "incorrect", []
        grades: list[DocGrade]
        if use_llm and ctx is not None:
            grades = await self._grade_llm(question, evidence, ctx)
        else:
            grades = self.grade_heuristic(question, evidence)
        by_id = {g.doc_id: g for g in grades}
        cfg = self.config
        decision = crag_decision([g.score for g in grades], cfg.grade_upper_threshold, cfg.grade_lower_threshold)
        kept: list[Evidence] = []
        for ev in evidence:
            g = by_id.get(ev.id)
            if g is None:
                g = DocGrade(doc_id=ev.id, label="ambiguous", score=0.0, reason="not graded")
            meta = dict(ev.metadata)
            meta.update(grade=g.label, grade_score=round(g.score, 4), grade_reason=g.reason)
            annotated = ev.model_copy(update={"metadata": meta})
            if g.label == "incorrect" and decision != "incorrect":
                continue  # knowledge-strip refinement: drop unrelated units when something relevant exists
            kept.append(annotated)
        if decision == "incorrect":
            return [], decision, grades
        return kept, decision, grades

    async def _grade_llm(self, question: str, evidence: Sequence[Evidence], ctx: RunContext) -> list[DocGrade]:
        cfg = self.config
        try:
            req = make_request(
                GRADE_PROMPT,
                ctx.profile,
                schema=sanitize_schema(GradeBatch.model_json_schema()),
                effort=cfg.grading_effort,
                model=cfg.sub_call_model,
                documents=render_evidence(evidence, max_chars_each=cfg.max_evidence_chars),
                question=question,
            )
            resp = await ctx.llm_call(req, step="grade")
            batch = GradeBatch.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - grading is advisory
            ctx.warn(f"LLM grading failed ({type(exc).__name__}); using lexical grades")
            note_degraded(ctx, "grade")
            return self.grade_heuristic(question, evidence)
        ids = {ev.id for ev in evidence}
        out: list[DocGrade] = []
        for g in batch.grades:
            if g.doc_id not in ids:
                continue
            score = max(-1.0, min(1.0, float(g.score)))
            out.append(DocGrade(doc_id=g.doc_id, label=g.label, score=score, reason=g.reason))
        graded = {g.doc_id for g in out}
        for ev in evidence:  # anything the model skipped is graded lexically
            if ev.id not in graded:
                s = lexical_relevance(question, ev.content)
                out.append(DocGrade(doc_id=ev.id, label=label_for_score(s, cfg.grade_upper_threshold, cfg.grade_lower_threshold), score=s, reason="lexical fallback"))
        return out


class ContextCompressor:
    """RECOMP-style extractive compression to a token budget."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    def compress(self, evidence: Sequence[Evidence], query: str, *, token_budget: Optional[int] = None, multi_hop: bool = False) -> list[Evidence]:
        cfg = self.config
        per_unit = cfg.sentences_per_chunk_multi_hop if multi_hop else cfg.sentences_per_chunk_single_hop
        return extractive_compress(
            evidence,
            query,
            token_budget=token_budget or cfg.context_token_budget,
            sentences_per_unit=per_unit,
            protect_code=cfg.protect_code_and_tables,
        )


class ContextOrderer:
    """Position-aware ordering (best first, second best last)."""

    def order(self, evidence: Sequence[Evidence]) -> list[Evidence]:
        return order_for_position(evidence)


__all__ = [
    "ContextCompressor",
    "ContextGrader",
    "ContextOrderer",
    "DocGrade",
    "GradeBatch",
    "GradeDecision",
    "GradeLabel",
    "content_terms",
    "crag_decision",
    "deduplicate",
    "extractive_compress",
    "label_for_score",
    "lexical_relevance",
    "looks_like_code_or_table",
    "order_for_position",
    "split_sentences",
]
