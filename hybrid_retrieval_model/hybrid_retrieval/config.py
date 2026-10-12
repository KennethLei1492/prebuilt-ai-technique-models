"""Configuration for the hybrid retrieval model.

Every tunable lives here with a literature-justified default. The README's
configuration table is generated from these fields, so keep the field
descriptions accurate: they are the single source of truth.

Where the architecture specification and the research dossier disagree, the
dossier wins and the reason is recorded in the field description:

* chunking: 400 tokens / 0 overlap (Smith 2024; Qu et al. 2024) instead of 512/64;
* fusion window: ``k_sparse = k_dense = 100`` (Bruch et al. 2023; Elastic 2024)
  instead of 50;
* convex combination: semantic weight alpha = 0.7, i.e. ``default_sparse_weight = 0.3``
  (Bruch et al. 2023) instead of 0.5. Domain profiles override this through
  ``retrieval_hints["sparse_weight"]``.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.errors import ConfigurationError

ChunkStrategy = Literal["fixed", "sentence", "structure", "token_budget"]
FusionMethod = Literal["rrf", "convex", "dbsf", "interleave"]
Normalization = Literal["minmax", "zscore", "theoretical"]
RerankMode = Literal["none", "listwise", "pointwise", "cross_encoder"]
BM25Variant = Literal["okapi", "plus"]
BM25Backend = Literal["python", "rank_bm25"]


class HybridRetrievalConfig(BaseModel):
    """All tunables of :class:`~hybrid_retrieval.model.HybridRetrievalModel`."""

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ chunking
    chunk_strategy: ChunkStrategy = Field(
        default="structure",
        description="Recursive boundary-aware splitting (headings > paragraphs > sentences). Fixed-size splitting "
        "was the robust baseline and semantic chunkers gave inconsistent gains (Qu et al. 2024; Smith 2024).",
    )
    chunk_size_tokens: int = Field(
        default=400, ge=16,
        description="Target chunk size. 200-400 tokens gave the best token-level recall/IoU; 800/400 was worst "
        "(Smith 2024; Qu et al. 2024). Deviates from the spec's 512 on the dossier's evidence.",
    )
    chunk_overlap_tokens: int = Field(
        default=0, ge=0,
        description="Overlap between consecutive chunks. Zero overlap was best for 200-400 token chunks; 10-25% only "
        "helps small-context embedders (Smith 2024). Deviates from the spec's 64.",
    )
    prepend_headers: bool = Field(
        default=True,
        description="Prefix each indexed chunk with the document title and section path (contextual header) for both "
        "legs while keeping raw text for display (Anthropic 2024; Karpukhin et al. 2020 title prefix).",
    )

    # -------------------------------------------------------------- sparse
    bm25_k1: float = Field(default=1.2, gt=0, description="BM25 term-frequency saturation; k1 in [1.2, 2.0] (Robertson & Zaragoza 2009).")
    bm25_b: float = Field(default=0.75, ge=0, le=1, description="BM25 length normalisation; b = 0.75 (Robertson & Zaragoza 2009).")
    bm25_variant: BM25Variant = Field(default="okapi", description="'okapi' (classic) or 'plus' (BM25+, lower-bounds long-document term scores).")
    bm25_delta: float = Field(default=1.0, ge=0, description="BM25+ delta; only used when bm25_variant='plus'.")
    bm25_backend: BM25Backend = Field(
        default="python",
        description="'python' is the deterministic built-in Okapi/BM25+ with the identifier-aware tokenizer; "
        "'rank_bm25' adapts the optional rank_bm25 package when installed.",
    )
    preserve_identifiers: bool = Field(
        default=True,
        description="Emit whole identifiers (ERR-1203, A320-NEO, getUserName) and their sub-tokens so part numbers, "
        "error codes and symbols match exactly (Robertson & Zaragoza 2009 design implication; CoIR, Li et al. 2025).",
    )
    remove_stopwords: bool = Field(
        default=True,
        description="Drop English function words from the BM25 channel so short passages do not win on 'the'/'before' matches; "
        "identifiers and their sub-tokens are never dropped (Robertson & Zaragoza 2009: idf alone under-penalises stopwords in small collections).",
    )

    # --------------------------------------------------------------- dense
    embedding_dimension: int = Field(default=512, ge=32, description="Dimension of the offline HashingEmbedder baseline.")
    dense_score_range: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="Theoretical range of the dense leg's cosine scores, used by theoretical min-max normalisation (Bruch et al. 2023).",
    )

    # -------------------------------------------------------------- fusion
    fusion: FusionMethod = Field(
        default="rrf",
        description="Zero-configuration default: RRF needs no normalisation or labelled queries (Cormack et al. 2009). "
        "The planner switches to 'convex' when a domain profile supplies a calibrated sparse_weight (Bruch et al. 2023).",
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF constant; k = 60 (Cormack et al. 2009; Elastic 2024). Tuned constants do not transfer (Bruch et al. 2023).")
    default_sparse_weight: float = Field(
        default=0.3, ge=0, le=1,
        description="Lexical weight of the convex combination when no domain hint exists; semantic alpha 0.7 was the best "
        "default with [0.6, 0.8] consistently good (Bruch et al. 2023). Deviates from the spec's 0.5.",
    )
    normalization: Normalization = Field(
        default="theoretical",
        description="Score normalisation for the convex combination; theoretical min-max beat observed min-max and z-score "
        "(Bruch et al. 2023). Falls back to observed min-max for channels without a known range (BM25).",
    )
    k_sparse: int = Field(default=100, ge=1, description="Sparse candidates for fusion; rank window max(size, 100) (Elastic 2024; Bruch et al. 2023). Spec said 50.")
    k_dense: int = Field(default=100, ge=1, description="Dense candidates for fusion; same justification as k_sparse.")
    k_final: int = Field(default=10, ge=1, description="Passages returned. 5-20 is the recommended band; reader accuracy saturates near 20 (Liu et al. 2023; Anthropic 2024).")

    # -------------------------------------------------------------- rerank
    rerank: RerankMode = Field(
        default="listwise",
        description="Default reranker tier when the planner enables reranking. Listwise LLM reranking is the strongest "
        "zero-shot reranker (Sun et al. 2023; Pradeep et al. 2023).",
    )
    rerank_depth: int = Field(default=20, ge=1, description="Candidates reranked; LLM tier over the top 20-30 costs one-fifth of full depth (Sun et al. 2023).")
    rerank_window: int = Field(default=20, ge=2, description="Listwise sliding-window size (RankGPT w = 20).")
    rerank_stride: int = Field(default=10, ge=1, description="Listwise sliding-window stride (RankGPT s = 10).")
    rerank_passage_chars: int = Field(default=1200, ge=100, description="Passage truncation in rerank prompts (~300 tokens; Sun et al. 2023).")
    pointwise_concurrency: int = Field(default=8, ge=1, description="Parallel pointwise grading calls (bounded by Constraints.max_parallelism).")

    # -------------------------------------------------- query transforms
    multi_query_n: int = Field(default=3, ge=1, description="Paraphrases generated by MultiQueryExpander and fused with RRF.")
    hyde_n: int = Field(default=4, ge=1, description="Hypothetical passages for HyDE; N = 4-8 with the query embedding as an extra vector (Gao et al. 2023).")
    query2doc_repeat: int = Field(default=5, ge=1, description="Times the original query is repeated before the pseudo-document (Wang et al. 2023).")
    query2doc_max_tokens: int = Field(default=128, ge=16, description="Pseudo-document length cap (Wang et al. 2023).")
    expansion_max_query_tokens: int = Field(default=6, ge=1, description="Router: HyDE/Query2doc fire only for short or exploratory queries (< 6 tokens) (Gao et al. 2023; Wang et al. 2023).")
    expansion_min_overlap: float = Field(
        default=0.1, ge=0, le=1,
        description="Drift guard: generated variants/passages sharing fewer content tokens than this fraction with the query are dropped (Wang et al. 2023 pitfall).",
    )

    # ---------------------------------------------------------- confidence
    weak_evidence_confidence: float = Field(default=0.3, ge=0, le=1, description="Below this the result warns 'weak evidence' so generators can abstain (Cohere 2025; Anthropic 2024).")
    direct_rerank: bool = Field(default=False, description="Whether the direct protocol API retrieve(query, k) also runs the configured reranker (costs LLM calls outside a RunContext).")
    max_documents_in_output_text: int = Field(default=10, ge=1, description="Rows rendered in output_text.")

    @model_validator(mode="after")
    def _check_consistency(self) -> "HybridRetrievalConfig":
        """Inconsistent settings raise ``ConfigurationError`` (spec 3.1.4: configuration errors are ``PipelineError`` subclasses)."""
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ConfigurationError(f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be smaller than chunk_size_tokens ({self.chunk_size_tokens})")
        if self.rerank_stride > self.rerank_window:
            raise ConfigurationError(f"rerank_stride ({self.rerank_stride}) must not exceed rerank_window ({self.rerank_window})")
        return self

    def dense_weight_for(self, sparse_weight: float) -> float:
        """Complement of the sparse weight (weights sum to 1)."""
        return round(1.0 - sparse_weight, 6)
