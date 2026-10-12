"""Query transformation and rank fusion.

Rewrite-Retrieve-Read (Ma et al. 2023) decouples the user's phrasing from the
retrieval query; step-back prompting (Zheng et al. 2024) retrieves for the
governing principle; multi-query covers facets; HyDE (Gao et al. 2023) embeds
a hypothetical answer. All variants are produced in one structured call when
a model is available, or by :func:`heuristic_variants` offline, and their
result lists are merged with Reciprocal Rank Fusion (Cormack et al. 2009,
k=60).

When the injected ``Retriever`` exposes its own ``transform``/``expand``
methods they are preferred (the hybrid retrieval model owns HyDE and
multi-query expansion in the full pipeline).
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import RunContext, sanitize_schema, tokenize_words

from .budget import calls_left, note_degraded
from .config import PipelineRAGConfig
from .prompts import TRANSFORM_PROMPT, make_request
from .router import TaskFeatures

TransformKind = Literal["literal", "rewrite", "multi_query", "step_back", "hyde", "hop", "glossary"]

_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being do does did what which who whom whose when where why how "
    "can could should would will shall may might must this that these those it its their there here with from by as at into about "
    "please tell me explain describe give show list find out between does the".split()
)
_ABSTRACTION_DROP = re.compile(r"\b(?:[A-Z]{1,6}-?\d+[A-Z0-9-]*|\d+(?:\.\d+)?\s*[A-Za-z%]*)\b")


class QueryVariants(BaseModel):
    """Structured output of the transform call (all fields present; empty when unused)."""

    model_config = ConfigDict(extra="forbid")

    rewrite: str = Field(default="", description="Search-friendly restatement keeping identifiers and units exact")
    sub_queries: list[str] = Field(default_factory=list, description="2-3 facet queries for multi-faceted questions")
    step_back: str = Field(default="", description="Query for the governing principle, standard or definition")
    hypothetical_answer: str = Field(default="", description="HyDE passage written as if it were the answering document")


class QueryVariant(BaseModel):
    """One retrieval query and the transform that produced it."""

    model_config = ConfigDict(extra="forbid")

    kind: TransformKind
    text: str
    weight: float = 1.0


def keyword_query(question: str) -> str:
    """Drop question words and stopwords, keep identifiers, numbers and content words in order."""
    toks = tokenize_words(question, lowercase=False)
    kept = [t for t in toks if t.lower() not in _STOPWORDS]
    return " ".join(kept) if kept else question.strip()


def step_back_query(question: str) -> str:
    """Abstract away identifiers and numbers to retrieve the governing principle (offline approximation)."""
    generic = _ABSTRACTION_DROP.sub(" ", question)
    toks = [t for t in tokenize_words(generic, lowercase=False) if t.lower() not in _STOPWORDS and not t.isdigit()]
    core = " ".join(toks[:8])
    return f"general principle requirements {core}".strip()


def glossary_expansion(question: str, glossary: dict[str, str]) -> Optional[str]:
    """Expand domain abbreviations found in the question with their glossary definitions (pure).

    ``DomainProfile.glossary`` maps terms such as ``FoS`` to ``Factor of safety -
    ratio of ...``; the head of the definition (before the first dash) is
    appended so lexical retrieval matches documents that spell the term out.
    Returns ``None`` when no glossary term occurs in the question.
    """
    if not glossary:
        return None
    expansions: list[str] = []
    for term, definition in sorted(glossary.items()):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", question):
            head = re.split(r"\s+[-–:]\s+|\s*\(", definition, maxsplit=1)[0].strip()
            if head and head.lower() not in question.lower():
                expansions.append(head)
    if not expansions:
        return None
    return " ".join([keyword_query(question), *expansions])


def split_sub_queries(question: str, max_parts: int = 3) -> list[str]:
    """Split a multi-part question on '?', ';', ' and ' (between clauses) and commas into facet queries."""
    parts = [p.strip() for p in re.split(r"\?|;|\band\b(?=\s+(?:what|which|how|why|when|where|who|does|is|are|list))", question) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r",\s+(?:and\s+)?|\s+and\s+", question) if len(p.split()) >= 3]
    out: list[str] = []
    for p in parts:
        kq = keyword_query(p)
        if kq and kq not in out and len(kq.split()) >= 1:
            out.append(kq)
    return out[:max_parts]


def heuristic_variants(
    question: str,
    transforms: Sequence[str],
    *,
    max_variants: int = 4,
    features: Optional[TaskFeatures] = None,
) -> list[QueryVariant]:
    """Offline query variants. The literal question is always kept so recall never drops below plain retrieval."""
    out: list[QueryVariant] = [QueryVariant(kind="literal", text=question.strip())]
    seen = {question.strip().lower()}

    def push(kind: TransformKind, text: str, weight: float = 1.0) -> None:
        t = " ".join(text.split())
        if t and t.lower() not in seen and len(out) < max_variants + 1:
            out.append(QueryVariant(kind=kind, text=t, weight=weight))
            seen.add(t.lower())

    if "rewrite" in transforms:
        push("rewrite", keyword_query(question))
    if "multi_query" in transforms:
        for sq in split_sub_queries(question):
            push("multi_query", sq, 0.8)
    if "step_back" in transforms and not (features and features.has_identifiers):
        push("step_back", step_back_query(question), 0.6)
    # HyDE needs a model; offline it is skipped.
    return out


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> list[tuple[str, float]]:
    """RRF: score(d) = sum_i w_i / (k + rank_i(d)); ranks are 1-based positions in each list.

    Pure function over ``[(doc_id, score), ...]`` lists; input scores are ignored
    (RRF needs no normalisation), which is why it is robust to heterogeneous
    retrievers. Ties break on doc id for determinism.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    w = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(w) != len(rankings):
        raise ValueError("weights must match rankings")
    fused: dict[str, float] = {}
    for wi, ranking in zip(w, rankings):
        seen: set[str] = set()
        rank = 0
        for doc_id, _score in ranking:
            if doc_id in seen:
                continue  # a repeated id does not consume a rank position
            seen.add(doc_id)
            rank += 1
            fused[doc_id] = fused.get(doc_id, 0.0) + wi / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class QueryTransformer:
    """Produce retrieval queries for a question: LLM when available, heuristics otherwise."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    async def transform(
        self,
        question: str,
        transforms: Sequence[str],
        *,
        ctx: Optional[RunContext] = None,
        use_llm: bool = False,
        features: Optional[TaskFeatures] = None,
        retriever: Any = None,
    ) -> list[QueryVariant]:
        """Return query variants (literal first). Falls back to heuristics on any LLM failure."""
        wanted = [t for t in transforms if t in ("rewrite", "multi_query", "step_back", "hyde")]
        if "hyde" in wanted and (not self.config.enable_hyde or (features and features.has_identifiers)):
            wanted.remove("hyde")
        if not wanted:
            return [QueryVariant(kind="literal", text=question.strip())]
        if retriever is not None and hasattr(retriever, "transform_queries"):
            try:
                texts = await retriever.transform_queries(question, wanted)  # delegated to the retriever's own transforms
                out = [QueryVariant(kind="literal", text=question.strip())]
                out.extend(QueryVariant(kind="rewrite", text=t) for t in texts if t and t.strip() != question.strip())
                return out[: self.config.max_query_variants + 1]
            except Exception:  # noqa: BLE001 - fall through to our own transforms
                pass
        if not use_llm or ctx is None:
            return heuristic_variants(question, wanted, max_variants=self.config.max_query_variants, features=features)
        if calls_left(ctx) < 2:  # keep at least one call (and some cost / time) for generation
            ctx.warn("skipping LLM query transforms to preserve the LLM budget")
            note_degraded(ctx, "transform")
            return heuristic_variants(question, wanted, max_variants=self.config.max_query_variants, features=features)
        try:
            req = make_request(
                TRANSFORM_PROMPT,
                ctx.profile,
                schema=sanitize_schema(QueryVariants.model_json_schema()),
                effort=self.config.grading_effort,
                model=self.config.sub_call_model,
                question=question,
                transforms=", ".join(wanted),
                max_variants=self.config.max_query_variants,
            )
            resp = await ctx.llm_call(req, step="transform")
            variants = QueryVariants.model_validate(resp.parsed)
        except Exception as exc:  # noqa: BLE001 - transforms are optional
            ctx.warn(f"LLM query transform failed ({type(exc).__name__}); using heuristic variants")
            note_degraded(ctx, "transform")
            return heuristic_variants(question, wanted, max_variants=self.config.max_query_variants, features=features)
        return self._from_llm(question, variants, wanted)

    def _from_llm(self, question: str, v: QueryVariants, wanted: Sequence[str]) -> list[QueryVariant]:
        out = [QueryVariant(kind="literal", text=question.strip())]
        seen = {question.strip().lower()}

        def push(kind: TransformKind, text: str, weight: float = 1.0) -> None:
            t = " ".join((text or "").split())
            if t and t.lower() not in seen and len(out) < self.config.max_query_variants + 1:
                out.append(QueryVariant(kind=kind, text=t, weight=weight))
                seen.add(t.lower())

        if "rewrite" in wanted:
            push("rewrite", v.rewrite)
        if "multi_query" in wanted:
            for sq in v.sub_queries[:3]:
                push("multi_query", sq, 0.8)
        if "step_back" in wanted:
            push("step_back", v.step_back, 0.6)
        if "hyde" in wanted:
            push("hyde", v.hypothetical_answer, 0.6)
        if len(out) == 1:  # the model returned nothing usable
            out.extend(heuristic_variants(question, wanted, max_variants=self.config.max_query_variants)[1:])
        return out


__all__ = [
    "QueryTransformer",
    "QueryVariant",
    "QueryVariants",
    "TransformKind",
    "glossary_expansion",
    "heuristic_variants",
    "keyword_query",
    "reciprocal_rank_fusion",
    "split_sub_queries",
    "step_back_query",
]
