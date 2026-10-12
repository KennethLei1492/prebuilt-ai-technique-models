"""Grounded generation with per-statement citations and abstention.

The generation contract follows ALCE (Gao et al. 2023): every factual
statement ends with one or more ``[doc-id]`` markers that resolve to an
evidence unit in the prompt, the model lists the ids it cited with a short
supporting quote, anything it cannot support goes to ``unsupported_claims``,
and when the evidence is insufficient it abstains instead of fabricating
(Barnett et al. 2024 FP1; Li et al. 2024 Self-Route "unanswerable").

Two paths produce the same :class:`GenerationOutput`:

* :meth:`GroundedGenerator.generate_llm` - one structured ``ctx.llm_call``
  with the ``GENERATE_PROMPT`` (or ``NO_RETRIEVAL_PROMPT``) template.
* :func:`extractive_answer` - the offline/heuristic path: the most
  question-relevant sentences of the ordered evidence, each cited verbatim.
  It is what ``MockLLM`` runs produce, so demos and tests behave sensibly
  without a model.

Everything that is not an LLM call (marker parsing, citation resolution,
coverage, rendering) is a pure function and unit-tested without a model.
"""
from __future__ import annotations

import re
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline_core import Evidence, RunContext, render_evidence, sanitize_schema

from .budget import note_degraded
from .config import PipelineRAGConfig
from .context import content_terms, lexical_relevance, split_sentences
from .prompts import GENERATE_PROMPT, NO_RETRIEVAL_PROMPT, make_request

CitationStyle = Literal["inline", "footnote", "none"]

_MARKER_RE = re.compile(r"\[([^\[\]\n]{1,80}?)\]")
_MARKER_STRIP_RE = re.compile(r"\s*\[[^\[\]\n]{1,80}?\]")


class Citation(BaseModel):
    """One cited evidence unit with the quote that supports the answer."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Evidence id exactly as given in the document id attribute")
    quote: str = Field(default="", description="Short verbatim quote from the document that supports the answer")


class GenerationOutput(BaseModel):
    """Structured output of the generation call (ALCE-style citations plus abstention)."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(default="", description="The answer; every factual statement ends with [doc-id] markers")
    citations: list[Citation] = Field(default_factory=list, description="Evidence units cited in the answer")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims the documents do not support")
    confidence: float = Field(default=0.0, description="Self-rated probability (0-1) that the answer is correct and complete")
    abstained: bool = Field(default=False, description="True when the evidence was insufficient to answer")
    abstention_reason: str = Field(default="", description="What was missing when abstaining")

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class Statement(BaseModel):
    """One atomic statement of the answer with its resolved citations."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citation_ids: list[str] = Field(default_factory=list, description="Marker ids that resolve to evidence")
    unresolved_ids: list[str] = Field(default_factory=list, description="Marker ids that match no evidence unit")


# ----------------------------------------------------------------- pure logic


def extract_markers(text: str) -> list[str]:
    """Return the citation marker ids in ``text`` in order of appearance (duplicates kept)."""
    return [m.group(1).strip() for m in _MARKER_RE.finditer(text)]


def strip_markers(text: str) -> str:
    """Remove ``[doc-id]`` markers and tidy whitespace before punctuation."""
    out = _MARKER_STRIP_RE.sub("", text)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def split_statements(answer: str, evidence_ids: Sequence[str]) -> list[Statement]:
    """Split an answer into sentence-level statements and resolve each one's markers.

    A marker resolves when it equals an evidence id, when it is a
    case-insensitive match, or when it is a number (``[1]``, ``[doc 1]``):
    numbers resolve to the auto-generated id ``doc-1`` when that id exists
    and otherwise to the first-based position within the evidence list.
    Markers that trail a sentence boundary are attributed to the sentence
    before them.
    """
    ids = list(evidence_ids)
    lower = {i.lower(): i for i in ids}
    sentences = [s for s in split_sentences(answer) if s]
    # markers that open a sentence (or stand alone) belong to the sentence before them
    merged: list[str] = []
    for s in sentences:
        lead = re.match(r"^(?:\s*\[[^\[\]\n]{1,80}?\]\s*)+[.]?", s)
        if lead and merged:
            merged[-1] = merged[-1] + " " + lead.group(0).strip()
            s = s[lead.end() :].strip()
            if not s:
                continue
        merged.append(s)
    out: list[Statement] = []
    for s in merged:
        text = strip_markers(s)
        if not text or len(content_terms(text)) == 0:
            continue
        resolved: list[str] = []
        unresolved: list[str] = []
        for m in extract_markers(s):
            for part in re.split(r"\s*[,;]\s*", m):
                part = part.strip()
                if not part:
                    continue
                hit = _resolve(part, ids, lower)
                if hit is None:
                    unresolved.append(part)
                elif hit not in resolved:
                    resolved.append(hit)
        out.append(Statement(text=text, citation_ids=resolved, unresolved_ids=unresolved))
    return out


def _resolve(marker: str, ids: Sequence[str], lower: dict[str, str]) -> Optional[str]:
    if marker in ids:
        return marker
    if marker.lower() in lower:
        return lower[marker.lower()]
    m = re.fullmatch(r"(?:doc[-_ ]?)?(\d+)", marker, re.I)
    if m:
        idx = int(m.group(1))
        candidate = f"doc-{idx}"  # the auto-generated id of a plain-string document wins over the position
        if candidate in ids:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
        if 1 <= idx <= len(ids):
            return ids[idx - 1]
    return None


def citation_coverage(statements: Sequence[Statement]) -> float:
    """Fraction of statements carrying at least one resolving citation (ALCE 'statements with citation')."""
    if not statements:
        return 0.0
    return sum(1 for s in statements if s.citation_ids) / len(statements)


def cited_ids(output: GenerationOutput, statements: Sequence[Statement]) -> list[str]:
    """Union of the ids cited in the answer text and in ``output.citations`` (order preserved)."""
    seen: list[str] = []
    for s in statements:
        for cid in s.citation_ids:
            if cid not in seen:
                seen.append(cid)
    for c in output.citations:
        if c.doc_id and c.doc_id not in seen:
            seen.append(c.doc_id)
    return seen


def render_answer(output: GenerationOutput, evidence: Sequence[Evidence], style: CitationStyle = "inline") -> str:
    """Render the answer for humans in the requested citation style.

    ``inline`` keeps the ``[doc-id]`` markers; ``footnote`` replaces them by
    numbered references and appends a source list; ``none`` strips them.
    """
    text = output.answer or ""
    if output.abstained and not text:
        reason = output.abstention_reason.strip()
        text = "Insufficient evidence to answer." + (f" {reason[0].upper() + reason[1:]}" + ("" if reason.endswith(".") else ".") if reason else "")
    if style == "none":
        return strip_markers(text)
    if style == "inline":
        return text
    by_id = {ev.id: ev for ev in evidence}
    order: list[str] = []

    def repl(m: re.Match[str]) -> str:
        nums = []
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            part = part.strip()
            if part not in order:
                order.append(part)
            nums.append(str(order.index(part) + 1))
        return "[" + ",".join(nums) + "]"

    body = _MARKER_RE.sub(repl, text)
    lines = [body, ""]
    for i, cid in enumerate(order, 1):
        ev = by_id.get(cid)
        src = f" ({ev.source})" if ev is not None and ev.source and ev.source != cid else ""
        lines.append(f"[{i}] {cid}{src}")
    return "\n".join(lines).rstrip()


def extractive_answer(
    question: str,
    evidence: Sequence[Evidence],
    *,
    answer_type: str = "factual",
    max_sentences: int = 3,
    min_relevance: float = -0.4,
    facets: Sequence[str] = (),
) -> GenerationOutput:
    """Offline grounded answer: the most question-relevant sentences, each cited verbatim.

    Sentences are scored with :func:`lexical_relevance`; the best
    ``max_sentences`` (more for list/summary/comparison shapes) are kept in
    evidence order and cited with their unit id. ``facets`` (sub-questions of
    a multi-hop question) each contribute their best sentence plus the
    sentence that follows it in the same unit (the answer continuation), and
    these facet picks bypass the one-sentence-per-unit rule. When no sentence
    reaches ``min_relevance`` (fewer than ~30% of the question's content
    terms) the output abstains, which is the RECOMP "return nothing"
    behaviour; a ``summary`` request falls back to the RAPTOR summary nodes
    (``metadata["level"] >= 1``) when they are present, with a low
    self-reported confidence (0.4) because nothing in them matched the
    question's key terms.
    """
    if not evidence:
        return GenerationOutput(abstained=True, abstention_reason="no evidence was retrieved", confidence=0.0)
    multi_item = answer_type in ("list", "summary", "comparison", "procedural")
    quota = max(max_sentences, 5) if multi_item else max_sentences
    scored: list[tuple[float, int, int, str, Evidence]] = []
    sentences: list[tuple[int, int, str, Evidence]] = []
    for ei, ev in enumerate(evidence):
        for si, sent in enumerate(split_sentences(ev.content)):
            if len(content_terms(sent)) < 2:
                continue
            sentences.append((ei, si, sent, ev))
            scored.append((lexical_relevance(question, sent), ei, si, sent, ev))
    if not scored:
        return GenerationOutput(abstained=True, abstention_reason="evidence contained no usable sentences", confidence=0.0)
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    best = scored[0][0]
    if best < min_relevance:
        summaries = [(ei, si, s, ev) for ei, si, s, ev in sentences if int(ev.metadata.get("level", 0) or 0) >= 1]
        if answer_type == "summary" and summaries:
            picked = []
            seen_units: set[int] = set()
            for ei, si, s, ev in summaries:  # first sentence of each summary node
                if ei not in seen_units:
                    seen_units.add(ei)
                    picked.append((0.0, ei, si, s, ev))
            return _assemble(picked[:quota], answer_type, confidence=0.4)
        return GenerationOutput(
            abstained=True,
            abstention_reason="retrieved evidence does not mention the question's key terms",
            confidence=max(0.0, (best + 1.0) / 4.0),
        )
    # facets of a list each match only part of the question; a single fact must not drag in weak neighbours
    floor = min(min_relevance, -0.75) if multi_item else max(min_relevance, best - 0.4)
    chosen = [s for s in scored[:quota] if s[0] >= floor]
    # multi-hop: every facet (sub-question) contributes its best sentence plus the sentence that follows it in
    # the same unit when that successor still shares a term with the facet. The successor rule is the
    # answer-continuation pattern of engineering records ("...tripped on high vibration." -> "Root cause: ...",
    # "Corrective action: ..."): the asked-for detail sits right after the sentence that names the subject, which
    # is also why the compressor keeps adjacent sentences (RECOMP adaptation, see config).
    facet_picks: set[tuple[int, int]] = set()
    for facet in facets:
        facet_best = max(((lexical_relevance(facet, s), ei, si, s, ev) for ei, si, s, ev in sentences), key=lambda x: (x[0], -x[1], -x[2]), default=None)
        if facet_best is None or facet_best[0] < min_relevance:
            continue
        picks = [facet_best]
        successor = min((t for t in sentences if t[0] == facet_best[1] and t[1] > facet_best[2]), key=lambda t: t[1], default=None)
        if successor is not None:
            rel = lexical_relevance(facet, successor[2])
            if rel > -1.0:  # shares at least one content term with the facet
                picks.append((rel, *successor))
        for pick in picks:
            pos = (pick[1], pick[2])
            facet_picks.add(pos)
            if all(pos != (c[1], c[2]) for c in chosen):
                chosen.append(pick)
    # one sentence per evidence unit unless the shape asks for many; facet picks are exempt because the
    # facets of a multi-part question are often answered by one record (incident -> root cause -> action)
    if not multi_item:
        seen_units: set[int] = set()
        filtered = []
        for s in chosen:
            if s[1] in seen_units and (s[1], s[2]) not in facet_picks:
                continue
            seen_units.add(s[1])
            filtered.append(s)
        chosen = filtered
    conf = (best + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    conf = max(0.0, min(0.95, 0.35 + 0.6 * conf))
    return _assemble(chosen, answer_type, confidence=conf)


def _assemble(chosen: list[tuple[float, int, int, str, Evidence]], answer_type: str, *, confidence: float) -> GenerationOutput:
    """Render chosen (score, unit index, sentence index, sentence, evidence) tuples as a cited answer."""
    chosen = sorted(chosen, key=lambda x: (x[1], x[2]))
    parts: list[str] = []
    cits: list[Citation] = []
    seen_ids: set[str] = set()
    for _rel, _ei, _si, sent, ev in chosen:
        sent_clean = sent.rstrip()
        has_end = bool(sent_clean) and sent_clean[-1] in ".!?:;"
        end = sent_clean[-1] if has_end else "."
        body = sent_clean[:-1].rstrip() if has_end else sent_clean
        parts.append(f"{body} [{ev.id}]{end}")  # ALCE convention: marker before the final punctuation
        if ev.id not in seen_ids:
            seen_ids.add(ev.id)
            cits.append(Citation(doc_id=ev.id, quote=sent[:200]))
    if answer_type == "list":
        answer = "\n".join(f"- {p}" for p in parts)
    else:
        answer = " ".join(parts)
    return GenerationOutput(answer=answer, citations=cits, confidence=confidence, abstained=False)


def citation_instruction(style: CitationStyle, require: bool) -> str:
    """Volatile instruction text telling the generator how strictly to cite."""
    if style == "none":
        return "Citations are optional for this answer; still list the ids you relied on in citations."
    base = "End every factual statement with [doc-id] markers using the document id attribute exactly."
    if require:
        base += " A statement without a resolving citation will be rejected."
    return base


# ------------------------------------------------------------------- class


class GroundedGenerator:
    """Produce a :class:`GenerationOutput` from ordered evidence (LLM online, extractive offline)."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    def generate_heuristic(self, question: str, evidence: Sequence[Evidence], *, answer_type: str = "factual", multi_hop: bool = False) -> GenerationOutput:
        """Extractive, citation-complete answer without an LLM.

        For multi-hop plans each facet of the question (its sub-questions)
        contributes its best-supported sentence, so a two-part question gets
        a two-part answer.
        """
        from .transform import split_sub_queries

        facets = split_sub_queries(question) if multi_hop else []
        return extractive_answer(question, evidence, answer_type=answer_type, facets=facets)

    async def generate_llm(
        self,
        question: str,
        evidence: Sequence[Evidence],
        ctx: RunContext,
        *,
        answer_type: str = "factual",
        citation_style: CitationStyle = "inline",
        require_citations: bool = True,
    ) -> GenerationOutput:
        """One structured generation call; falls back to the extractive answer on failure."""
        cfg = self.config
        schema = sanitize_schema(GenerationOutput.model_json_schema())
        try:
            if evidence:
                req = make_request(
                    GENERATE_PROMPT,
                    ctx.profile,
                    schema=schema,
                    effort=cfg.generation_effort,
                    documents=render_evidence(evidence, max_chars_each=cfg.max_evidence_chars),
                    question=question,
                    answer_type=answer_type,
                    citation_instruction=citation_instruction(citation_style, require_citations),
                    max_tokens=8000,
                )
            else:
                req = make_request(
                    NO_RETRIEVAL_PROMPT,
                    ctx.profile,
                    schema=schema,
                    effort=cfg.generation_effort,
                    question=question,
                    answer_type=answer_type,
                    max_tokens=8000,
                )
            resp = await ctx.llm_call(req, step="generate")
            out = GenerationOutput.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - generation must degrade, not raise
            ctx.warn(f"LLM generation failed ({type(exc).__name__}: {str(exc)[:120]}); using extractive answer")
            note_degraded(ctx, "generate")
            return self.generate_heuristic(question, evidence, answer_type=answer_type)
        if not out.answer.strip() and not out.abstained:
            out.abstained = True
            out.abstention_reason = out.abstention_reason or "the model returned an empty answer"
        return out


__all__ = [
    "Citation",
    "CitationStyle",
    "GenerationOutput",
    "GroundedGenerator",
    "Statement",
    "citation_coverage",
    "citation_instruction",
    "cited_ids",
    "extract_markers",
    "extractive_answer",
    "render_answer",
    "split_statements",
    "strip_markers",
]
