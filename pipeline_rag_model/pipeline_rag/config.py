"""Configuration for the Pipeline RAG model.

Every tunable lives here with a literature-justified default. The README's
configuration table is generated from the field descriptions below, so keep
``description`` strings accurate: each one names the source that justifies
the default (see ``docs/research/pipeline-rag.json``).

Design rule: the LLM-facing output schemas in :mod:`pipeline_rag.plan` and
:mod:`pipeline_rag.generator` carry no numeric bounds (constrained decoders
reject them); bounds are enforced here and by clamping validators.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

LLMMode = Literal["auto", "always", "never"]
ContextualPrefix = Literal["none", "heuristic", "llm"]
HierarchyMode = Literal["never", "auto", "always"]
CitationStyle = Literal["inline", "footnote", "none"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class PipelineRAGConfig(BaseModel):
    """All tunables of :class:`pipeline_rag.model.PipelineRAGModel`."""

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ routing
    llm_mode: LLMMode = Field(
        default="auto",
        description=(
            "auto: use LLM stages when the client is a real model and heuristics when ctx.offline; "
            "always: call the LLM even with Mock/Scripted clients (tests of the LLM path); never: pure heuristics."
        ),
    )
    max_escalations: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Upward plan escalations on abstention (single_step -> multi_step -> agentic). Self-Route (Li et al. 2024) escalates cheap RAG to a richer plan on 'unanswerable'; dossier default depth 2.",
    )
    full_corpus_gate_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "If > 0 and the chunked corpus is at most this many tokens, retrieve every chunk instead of ranking (Anthropic Contextual Retrieval recommends bypassing retrieval below ~200K tokens). "
            "Disabled by default because evidence lives in the volatile user turn, so there is no prompt-cache win to pay for the larger context."
        ),
    )

    # ---------------------------------------------------------- retrieval
    default_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Evidence units kept after fusion for a single-step plan. Liu et al. (2024) show reader accuracy saturates at 10-20 documents; Anthropic's cascade keeps top-20 after reranking. 10 is the lower edge because evidence is then compressed to a token budget.",
    )
    k_per_hop: int = Field(
        default=5,
        ge=1,
        le=50,
        description="New evidence units retrieved per additional hop. Dossier default for IRCoT/FLARE loops: 4-8 new chunks per hop with dedupe.",
    )
    candidates_per_variant: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Candidates retrieved per query variant before Reciprocal Rank Fusion (dossier: retrieve top-50 per variant, fuse, then rerank).",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion constant. Cormack et al. (2009) fixed k=60 in a pilot and never re-tuned it; it is the default in every major hybrid search stack.",
    )
    max_query_variants: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Total query variants per hop (literal rewrite + multi-query + step-back). Dossier: 3-5 variants.",
    )
    enable_hyde: bool = Field(
        default=False,
        description="Generate a hypothetical answer passage as an extra query (Gao et al. 2023). Opt-in: HyDE injects hallucinated detail when the model lacks domain knowledge (private identifiers), so it is gated off and never used for identifier-heavy or high-risk queries.",
    )
    chunk_size_tokens: int = Field(
        default=512,
        ge=32,
        le=4000,
        description="Built-in chunker window. Dossier: 200-800 tokens (GraphRAG 600; Anthropic 'no more than a few hundred'); 512 matches the repository's hybrid retrieval default so chunk ids line up across models.",
    )
    chunk_overlap_tokens: int = Field(
        default=64,
        ge=0,
        description="Built-in chunker overlap (~12%). Dossier: 10-15% overlap (GraphRAG 100/600).",
    )
    contextual_prefix: ContextualPrefix = Field(
        default="heuristic",
        description="Prepend a short situating context to each chunk before indexing (Anthropic Contextual Retrieval: -35% to -49% top-20 failures). 'heuristic' derives it from source title and position without LLM calls; 'llm' generates a 50-100 token prefix per chunk with one sub-call each (CONTEXT_PREFIX_PROMPT, once per index, online only, budget-guarded; the heuristic prefix is kept offline, under constraints.deterministic or when the budget is short).",
    )
    llm_prefix_max_chunks: int = Field(
        default=200,
        ge=0,
        description="Cap on chunks per index that receive an LLM prefix when contextual_prefix='llm' (one sub-call each); later chunks keep the heuristic prefix.",
    )
    build_hierarchy: HierarchyMode = Field(
        default="auto",
        description="Build RAPTOR-style document summaries as extra index nodes (Sarthi et al. 2024). 'auto' builds them only for agentic/global plans, since summarisation costs one LLM call per document.",
    )

    # ------------------------------------------------------------ grading
    grade_upper_threshold: float = Field(
        default=0.5,
        ge=-1.0,
        le=1.0,
        description="CRAG upper threshold on the [-1, 1] relevance score: at least one document above it -> CORRECT. Yan et al. (2024) thresholds are dataset-specific; the dossier recommends starting at 0.5 and calibrating per domain.",
    )
    grade_lower_threshold: float = Field(
        default=-0.9,
        ge=-1.0,
        le=1.0,
        description="CRAG lower threshold: every document below it -> INCORRECT (discard, fall back or abstain). Dossier starting point -0.9.",
    )
    max_fallback_rounds: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Fallback retrieval rounds after an INCORRECT grade (CRAG). Dossier: one configured fallback with at most one round.",
    )

    # -------------------------------------------------------- compression
    context_token_budget: int = Field(
        default=2000,
        ge=100,
        description="Token budget for evidence passed to the generator. RAPTOR's collapsed-tree retrieval uses a 2,000-token budget (~top-20 nodes); the dossier recommends 2,000-4,000 for chat-sized models.",
    )
    sentences_per_chunk_single_hop: int = Field(
        default=2,
        ge=1,
        description="Extractive compression keeps this many question-relevant sentences per chunk on single-hop plans. RECOMP (Xu et al. 2024) keeps 1; we keep 2 because our lexical extractor is noisier than RECOMP's trained dual encoder and engineering answers depend on the adjacent sentence carrying the unit or condition.",
    )
    sentences_per_chunk_multi_hop: int = Field(
        default=3,
        ge=1,
        description="Sentences kept per chunk on multi-hop plans (RECOMP: top-2 for multi-hop; +1 for the same reason as above).",
    )
    protect_code_and_tables: bool = Field(
        default=True,
        description="Never compress chunks that look like code, tables or identifier lists (LLMLingua: token dropping is destructive for code and numbers).",
    )
    max_evidence_chars: int = Field(
        default=2000,
        ge=200,
        description="Maximum characters rendered per evidence unit in the generation prompt.",
    )

    # ---------------------------------------------------------- iteration
    max_hops: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum retrieve-and-reason hops for multi-step and agentic plans. IRCoT/Adaptive-RAG multi-step averaged 2.1-4.7 rounds; dossier default S=4.",
    )
    retrieval_trigger_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Self-RAG retrieval trigger: a multi-hop step retrieves again (with its next_query) when its need_probability (P(Retrieve)) exceeds this, even if it claims it can answer (Asai et al. 2024 used 0.2; 0 when citations are mandatory).",
    )
    flare_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="FLARE-style active retrieval: a reasoning step with self-rated confidence below this triggers another retrieval (Jiang et al. 2023; dossier range 0.4-0.8).",
    )
    reflection_weight_relevant: float = Field(default=1.0, ge=0.0, description="Self-RAG critique weight for IsRel (Asai et al. 2024 default 1.0).")
    reflection_weight_supported: float = Field(default=1.0, ge=0.0, description="Self-RAG critique weight for IsSup (default 1.0; raise for precision-critical domains).")
    reflection_weight_useful: float = Field(default=0.5, ge=0.0, description="Self-RAG critique weight for IsUse (default 0.5).")

    # ------------------------------------------------------- generation
    citation_style: CitationStyle = Field(
        default="inline",
        description="How citations are rendered: inline [doc-id] markers (ALCE convention), footnotes, or none.",
    )
    require_citations: bool = Field(
        default=True,
        description="Require at least one citation per factual statement (ALCE, Gao et al. 2023); statements without one count against confidence.",
    )
    min_citation_coverage: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of statements that must carry a resolving citation before the answer is accepted without review. ALCE/dossier release gate: citation recall >= 0.8.",
    )
    min_faithfulness: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Evaluation release gate on faithfulness (RAGAS, Es et al. 2024: 0.95 human agreement; dossier gate 0.9).",
    )
    abstain_when_insufficient: bool = Field(
        default=True,
        description="Allow the generator to answer 'insufficient evidence' instead of fabricating (Barnett FP1; RECOMP selective augmentation; Self-Route abstention).",
    )

    # ------------------------------------------------------- confidence
    confidence_weight_self_reported: float = Field(default=0.2, ge=0.0, description="Weight of the generator's self-rated confidence. Lowest weight: generations are critiqued against the evidence rather than trusted as produced (Self-RAG, Asai et al. 2024; ALCE: never trust self-reported citations).")
    confidence_weight_citation_coverage: float = Field(default=0.25, ge=0.0, description="Weight of citation coverage (ALCE citation recall).")
    confidence_weight_support: float = Field(default=0.35, ge=0.0, description="Weight of statement support / faithfulness. Highest weight: RAGAS faithfulness agrees with humans at 0.95.")
    confidence_weight_evidence: float = Field(default=0.2, ge=0.0, description="Weight of mean evidence relevance (CRAG grade) of the cited documents.")

    # ----------------------------------------------------------- effort
    planner_effort: Effort = Field(default="high", description="Effort for the one planning call (architecture rule: high for planning and synthesis).")
    grading_effort: Effort = Field(default="low", description="Effort for grading/routing many items (architecture rule: low for classification).")
    generation_effort: Effort = Field(default="high", description="Effort for grounded synthesis.")
    verification_effort_safety_critical: Effort = Field(default="max", description="Effort for the final support check in safety-critical domains (architecture rule).")
    sub_call_model: Optional[str] = Field(
        default=None,
        description="Optional cheaper model id for grading, transforms and summaries (e.g. claude-haiku-5-5). None keeps every call on the client's default model so prompt caches stay model-scoped.",
    )

    @model_validator(mode="after")
    def _check(self) -> "PipelineRAGConfig":
        if self.grade_lower_threshold >= self.grade_upper_threshold:
            raise ValueError("grade_lower_threshold must be below grade_upper_threshold")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        total = (
            self.confidence_weight_self_reported
            + self.confidence_weight_citation_coverage
            + self.confidence_weight_support
            + self.confidence_weight_evidence
        )
        if total <= 0:
            raise ValueError("confidence weights must sum to a positive number")
        return self

    def confidence_weights(self) -> dict[str, float]:
        """Normalised confidence weights keyed by signal name."""
        raw = {
            "self_reported": self.confidence_weight_self_reported,
            "citation_coverage": self.confidence_weight_citation_coverage,
            "support": self.confidence_weight_support,
            "evidence_quality": self.confidence_weight_evidence,
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def reflection_weights(self) -> dict[str, float]:
        """Self-RAG critique weights keyed by flag name."""
        return {
            "is_relevant": self.reflection_weight_relevant,
            "is_supported": self.reflection_weight_supported,
            "is_useful": self.reflection_weight_useful,
        }
