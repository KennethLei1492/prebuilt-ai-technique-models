"""Multi-hop iteration and Self-RAG style reflection.

* :class:`MultiHopLoop` runs the IRCoT / FLARE loop (Trivedi et al. 2023;
  Jiang et al. 2023): reason one step, detect an information need (an
  explicit ``need_more_information`` flag, a Self-RAG retrieve probability
  ``need_probability`` above ``retrieval_trigger_threshold``, or a self-rated
  confidence below the FLARE threshold), retrieve again with the new query,
  dedupe, repeat up to ``max_hops``. Offline the hop queries come from
  :func:`plan_hop_queries`, a deterministic facet/step-back schedule.
* :class:`ReflectionJudge` produces the Self-RAG reflection flags (Asai et
  al. 2024) ``is_relevant``, ``is_supported`` and ``is_useful`` as structured
  output, and :func:`critique_score` combines them with the configured weights.
  Offline the judgement is a lexical entailment proxy over the cited units.

The loop and the judge call the model only through ``ctx.llm_call``; the
aggregation maths (:func:`critique_score`, :func:`support_fraction`,
:func:`flare_trigger`, :func:`plan_hop_queries`, :func:`merge_evidence`) are
pure functions.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline_core import Evidence, RunContext, render_evidence, sanitize_schema

from .budget import calls_left, note_degraded
from .config import PipelineRAGConfig
from .context import deduplicate, lexical_relevance, split_sentences
from .generator import Statement
from .prompts import HOP_PROMPT, SUPPORT_PROMPT, make_request
from .transform import keyword_query, split_sub_queries, step_back_query

SupportLevel = Literal["fully", "partially", "no"]
RetrieveFn = Callable[[str, int], Awaitable[list[Evidence]]]
"""``await retrieve(query, k) -> evidence`` supplied by the model (fused, cached, traced)."""

SUPPORT_VALUE: dict[str, float] = {"fully": 1.0, "partially": 0.5, "no": 0.0}


# ----------------------------------------------------------------- schemas


class HopStep(BaseModel):
    """Structured output of one IRCoT/FLARE step."""

    model_config = ConfigDict(extra="forbid")

    reasoning_step: str = Field(default="", description="One sentence of reasoning toward the answer, grounded in the documents")
    is_relevant: bool = Field(default=True, description="At least one document bears on the question (Self-RAG IsRel)")
    need_more_information: bool = Field(default=False, description="A needed fact is missing from the documents")
    need_probability: float = Field(default=0.0, description="Probability (0-1) that another retrieval is needed (Self-RAG Retrieve token)")
    next_query: str = Field(default="", description="Self-contained search query for the missing fact")
    can_answer: bool = Field(default=False, description="The documents and reasoning so far suffice for a complete answer")
    confidence: float = Field(default=0.5, description="Probability (0-1) that reasoning_step is correct")

    @field_validator("confidence", "need_probability", mode="before")
    @classmethod
    def _clamp(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class HopRecord(BaseModel):
    """Trace of one executed hop."""

    model_config = ConfigDict(extra="forbid")

    hop: int
    query: str
    new_evidence: int
    reasoning_step: str = ""
    confidence: float = 0.0
    triggered_by: Literal["initial", "need_more_information", "retrieval_trigger", "low_confidence", "schedule"] = "initial"
    stop_reason: str = ""


class StatementSupport(BaseModel):
    """Self-RAG IsSup judgement for one answer statement."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    support: SupportLevel = "no"
    evidence_ids: list[str] = Field(default_factory=list)


class SupportJudgement(BaseModel):
    """Structured output of the reflection call (IsRel, IsSup per statement, IsUse)."""

    model_config = ConfigDict(extra="forbid")

    statements: list[StatementSupport] = Field(default_factory=list)
    is_relevant: bool = Field(default=True, description="Documents address the question at all (IsRel)")
    is_useful: int = Field(default=3, description="1 (useless) to 5 (fully answers the question) (IsUse)")

    @field_validator("is_useful")
    @classmethod
    def _clamp_useful(cls, v: int) -> int:
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return 3


class ReflectionFlags(BaseModel):
    """Aggregated Self-RAG flags for the final answer."""

    model_config = ConfigDict(extra="forbid")

    is_relevant: bool = True
    is_supported: SupportLevel = "no"
    is_useful: int = 3
    support_fraction: float = Field(default=0.0, description="Weighted fraction of statements supported (fully=1, partially=0.5)")
    unsupported_statements: list[str] = Field(default_factory=list)
    critique_score: float = Field(default=0.0, description="Weighted critique on [0, 1]")
    method: str = Field(default="heuristic", description="heuristic | llm | verifier")


# --------------------------------------------------------------- pure maths


def support_fraction(levels: Sequence[str]) -> float:
    """Weighted fraction of supported statements: fully=1, partially=0.5, no=0."""
    if not levels:
        return 0.0
    return sum(SUPPORT_VALUE.get(level, 0.0) for level in levels) / len(levels)


def overall_support(levels: Sequence[str]) -> SupportLevel:
    """Collapse per-statement levels into one IsSup flag (strict: any 'no' -> partially at best)."""
    if not levels:
        return "no"
    if all(level == "fully" for level in levels):
        return "fully"
    if all(level == "no" for level in levels):
        return "no"
    return "partially"


def critique_score(
    *,
    is_relevant: bool,
    is_supported: str,
    is_useful: int,
    weights: dict[str, float],
) -> float:
    """Self-RAG critique: weighted sum of normalised flags, scaled to [0, 1].

    ``weights`` keys are ``is_relevant``, ``is_supported``, ``is_useful`` (Asai
    et al. 2024 defaults 1.0 / 1.0 / 0.5). IsSup maps fully=1, partially=0.5,
    no=0; IsUse maps 1..5 to 0..1.
    """
    w_rel = float(weights.get("is_relevant", 1.0))
    w_sup = float(weights.get("is_supported", 1.0))
    w_use = float(weights.get("is_useful", 0.5))
    total = w_rel + w_sup + w_use
    if total <= 0:
        return 0.0
    rel = 1.0 if is_relevant else 0.0
    sup = SUPPORT_VALUE.get(is_supported, 0.0)
    use = (max(1, min(5, int(is_useful))) - 1) / 4.0
    return (w_rel * rel + w_sup * sup + w_use * use) / total


def flare_trigger(confidence: float, threshold: float) -> bool:
    """FLARE active retrieval: retrieve again when the step's confidence is below the threshold."""
    return confidence < threshold


def should_retrieve(need_probability: float, threshold: float) -> bool:
    """Self-RAG retrieval trigger: P(Retrieve=yes) above the threshold (0 means always retrieve)."""
    return need_probability > threshold


def plan_hop_queries(question: str, *, include_step_back: bool = True, max_queries: int = 4) -> list[str]:
    """Deterministic schedule of follow-up queries for offline multi-hop runs.

    Facet queries (one per sub-question or enumerated item) come first, then
    a step-back query for the governing principle. The literal question is
    excluded because the first hop already used it.
    """
    out: list[str] = []
    seen = {question.strip().lower(), keyword_query(question).lower()}
    for sq in split_sub_queries(question, max_parts=max_queries):
        if sq.lower() not in seen:
            out.append(sq)
            seen.add(sq.lower())
    if include_step_back:
        sb = step_back_query(question)
        if sb.lower() not in seen:
            out.append(sb)
    return out[:max_queries]


def merge_evidence(existing: Sequence[Evidence], new: Sequence[Evidence]) -> tuple[list[Evidence], int]:
    """Append new units not already present (by id or near-duplicate text); return (merged, added_count)."""
    merged = deduplicate([*existing, *new])
    return merged, len(merged) - len(deduplicate(existing))


MIN_QUESTION_RELEVANCE = -0.4
"""Unit-level IsRel floor: a unit is relevant when it carries >= 30% of the question's weighted content terms
(the same floor the extractive generator uses to decide it can answer at all)."""


def question_relevant(question: str, evidence: Sequence[Evidence], *, floor: float = MIN_QUESTION_RELEVANCE) -> bool:
    """Self-RAG IsRel proxy: does at least one evidence unit (prefix + content) mention the question's key terms?"""
    if not question.strip() or not evidence:
        return False
    return any(lexical_relevance(question, f"{ev.metadata.get('prefix', '')} {ev.content}") >= floor for ev in evidence)


def heuristic_support(statements: Sequence[Statement], evidence: Sequence[Evidence], *, question: str = "") -> SupportJudgement:
    """Lexical entailment proxy: a statement is 'fully' supported when a cited (or any) unit covers >= 75% of its terms.

    With ``question`` given, ``is_relevant`` (Self-RAG IsRel) is judged against
    the question through :func:`question_relevant`; without it the flag falls
    back to "some statement is supported", which only says the evidence was
    usable, not that it addressed the question.
    """
    by_id = {ev.id: ev for ev in evidence}
    out: list[StatementSupport] = []
    any_relevant = False
    for st in statements:
        candidates = [by_id[c] for c in st.citation_ids if c in by_id] or list(evidence)
        best = -1.0
        best_ids: list[str] = []
        for ev in candidates:
            for sent in split_sentences(ev.content) or [ev.content]:
                r = lexical_relevance(st.text, sent)
                if r > best:
                    best, best_ids = r, [ev.id]
                elif r == best and ev.id not in best_ids and r >= 0.5:
                    best_ids.append(ev.id)
        if best >= 0.5:
            level: SupportLevel = "fully"
        elif best >= -0.2:
            level = "partially"
        else:
            level = "no"
        if best >= -0.2:
            any_relevant = True
        out.append(StatementSupport(statement=st.text, support=level, evidence_ids=best_ids if level != "no" else []))
    frac = support_fraction([s.support for s in out])
    useful = 1 + round(4 * frac) if out else 1
    if question.strip():
        is_relevant = question_relevant(question, evidence)
        if not is_relevant:
            useful = min(useful, 2)  # supported sentences that do not address the question are of little use (IsUse)
    else:
        is_relevant = any_relevant or bool(evidence) and not statements
    return SupportJudgement(statements=out, is_relevant=is_relevant, is_useful=useful)


def flags_from_judgement(j: SupportJudgement, weights: dict[str, float], *, method: str) -> ReflectionFlags:
    """Aggregate a :class:`SupportJudgement` into :class:`ReflectionFlags` with the critique score."""
    levels = [s.support for s in j.statements]
    sup = overall_support(levels)
    return ReflectionFlags(
        is_relevant=j.is_relevant,
        is_supported=sup,
        is_useful=j.is_useful,
        support_fraction=support_fraction(levels),
        unsupported_statements=[s.statement for s in j.statements if s.support == "no"],
        critique_score=critique_score(is_relevant=j.is_relevant, is_supported=sup, is_useful=j.is_useful, weights=weights),
        method=method,
    )


# -------------------------------------------------------------------- loop


class MultiHopLoop:
    """IRCoT/FLARE retrieve-and-reason loop with a hard hop cap."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    async def run(
        self,
        question: str,
        initial: Sequence[Evidence],
        retrieve: RetrieveFn,
        *,
        max_hops: int,
        ctx: Optional[RunContext] = None,
        use_llm: bool = False,
    ) -> tuple[list[Evidence], list[HopRecord], list[str]]:
        """Return (accumulated evidence, hop records, reasoning steps).

        Hop 1 is the initial retrieval (already done by the caller). Each later
        hop issues one new query: offline from :func:`plan_hop_queries`, online
        from the HOP_PROMPT step's ``next_query`` (or the reasoning step itself
        when FLARE confidence is low).
        """
        cfg = self.config
        evidence = list(initial)
        records: list[HopRecord] = [HopRecord(hop=1, query=question, new_evidence=len(evidence), triggered_by="initial")]
        reasoning: list[str] = []
        schedule = plan_hop_queries(question)
        used_queries = {question.strip().lower()}
        for hop in range(2, max_hops + 1):
            if ctx is not None and ctx.deadline_exceeded():
                records[-1].stop_reason = "latency budget exhausted"
                break
            trigger: str
            query = ""
            step: Optional[HopStep] = None
            if use_llm and ctx is not None and calls_left(ctx) >= 2:
                step = await self._hop_llm(question, evidence, reasoning, ctx)
                if step is not None:
                    if step.reasoning_step:
                        reasoning.append(step.reasoning_step)
                    # Self-RAG retrieval trigger: P(Retrieve) above the configured threshold, with a query to run
                    triggered = should_retrieve(step.need_probability, cfg.retrieval_trigger_threshold) and bool(step.next_query.strip())
                    if step.can_answer and not step.need_more_information and not triggered:
                        records[-1].stop_reason = "model can answer"
                        break
                    if step.need_more_information and step.next_query.strip():
                        query, trigger = step.next_query.strip(), "need_more_information"
                    elif triggered and step.next_query.strip():
                        query, trigger = step.next_query.strip(), "retrieval_trigger"
                    elif flare_trigger(step.confidence, cfg.flare_confidence_threshold) and step.reasoning_step:
                        query, trigger = keyword_query(step.reasoning_step), "low_confidence"
            if not query:
                trigger = "schedule"
                while schedule and schedule[0].lower() in used_queries:
                    schedule.pop(0)
                if not schedule:
                    records[-1].stop_reason = "no further queries"
                    break
                query = schedule.pop(0)
            if query.lower() in used_queries:
                records[-1].stop_reason = "query repeated"
                break
            used_queries.add(query.lower())
            new = await retrieve(query, cfg.k_per_hop)
            evidence, added = merge_evidence(evidence, new)
            records.append(
                HopRecord(
                    hop=hop,
                    query=query,
                    new_evidence=added,
                    reasoning_step=step.reasoning_step if step else "",
                    confidence=step.confidence if step else 0.0,
                    triggered_by=trigger,  # type: ignore[arg-type]
                )
            )
            if not use_llm and not schedule:
                records[-1].stop_reason = "schedule exhausted"
                break
        if not records[-1].stop_reason:
            records[-1].stop_reason = "hop cap reached" if len(records) >= max_hops else "loop finished"
        return evidence, records, reasoning

    async def _hop_llm(self, question: str, evidence: Sequence[Evidence], reasoning: Sequence[str], ctx: RunContext) -> Optional[HopStep]:
        cfg = self.config
        try:
            req = make_request(
                HOP_PROMPT,
                ctx.profile,
                schema=sanitize_schema(HopStep.model_json_schema()),
                effort=cfg.grading_effort,
                model=cfg.sub_call_model,
                documents=render_evidence(evidence, max_chars_each=cfg.max_evidence_chars, max_items=20),
                question=question,
                reasoning="\n".join(reasoning) or "(none yet)",
            )
            resp = await ctx.llm_call(req, step="hop")
            return HopStep.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - fall back to the schedule
            ctx.warn(f"LLM hop step failed ({type(exc).__name__}); using scheduled query")
            note_degraded(ctx, "iterate")
            return None


# ------------------------------------------------------------------- judge


class ReflectionJudge:
    """Self-RAG IsRel / IsSup / IsUse judgement of a drafted answer."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    def judge_heuristic(self, statements: Sequence[Statement], evidence: Sequence[Evidence], *, question: str = "") -> ReflectionFlags:
        """Lexical support proxy (no LLM); ``question`` enables the question-based IsRel flag."""
        return flags_from_judgement(heuristic_support(statements, evidence, question=question), self.config.reflection_weights(), method="heuristic")

    async def judge_llm(
        self,
        question: str,
        answer: str,
        statements: Sequence[Statement],
        evidence: Sequence[Evidence],
        ctx: RunContext,
        *,
        effort: Optional[str] = None,
    ) -> ReflectionFlags:
        """One structured support call; falls back to the heuristic judgement on failure."""
        cfg = self.config
        if not statements or not evidence:
            return self.judge_heuristic(statements, evidence, question=question)
        try:
            req = make_request(
                SUPPORT_PROMPT,
                ctx.profile,
                schema=sanitize_schema(SupportJudgement.model_json_schema()),
                effort=effort or cfg.grading_effort,
                documents=render_evidence(evidence, max_chars_each=cfg.max_evidence_chars),
                question=question,
                answer=answer,
            )
            resp = await ctx.llm_call(req, step="reflect")
            judgement = SupportJudgement.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - verification is advisory, the status reflects it
            ctx.warn(f"LLM support judgement failed ({type(exc).__name__}); using lexical support")
            note_degraded(ctx, "reflect")
            return self.judge_heuristic(statements, evidence, question=question)
        if not judgement.statements:  # the model judged nothing: fall back per statement
            return self.judge_heuristic(statements, evidence, question=question)
        ids = {ev.id for ev in evidence}
        for s in judgement.statements:
            s.evidence_ids = [i for i in s.evidence_ids if i in ids]
        return flags_from_judgement(judgement, cfg.reflection_weights(), method="llm")


__all__ = [
    "HopRecord",
    "HopStep",
    "MIN_QUESTION_RELEVANCE",
    "MultiHopLoop",
    "ReflectionFlags",
    "ReflectionJudge",
    "RetrieveFn",
    "SUPPORT_VALUE",
    "StatementSupport",
    "SupportJudgement",
    "SupportLevel",
    "critique_score",
    "flags_from_judgement",
    "flare_trigger",
    "heuristic_support",
    "merge_evidence",
    "overall_support",
    "plan_hop_queries",
    "question_relevant",
    "should_retrieve",
    "support_fraction",
]
