"""hybrid_retrieval - lexical + dense retrieval with fusion, reranking and query planning.

Import surface::

    from hybrid_retrieval import HybridRetrievalModel, HybridRetrievalConfig, RetrievalPlan

    model = HybridRetrievalModel()                 # MockLLM without credentials
    model.index([{"id": "spec-1", "text": "...", "metadata": {"type": "specification"}}])
    result = model.run_sync("Which clause specifies the impeller material?", domain="mechanical_engineering")
    evidence = await model.retrieve("impeller material", k=5)     # Retriever protocol

Importing the package registers ``HybridRetrievalModel`` with ``pipeline_core``
(``pipeline_core.get("hybrid_retrieval")``).
"""
from __future__ import annotations

from .config import HybridRetrievalConfig
from .dense import ANNBackend, CallableEmbedder, DenseIndex, Embedder, HashingEmbedder, SentenceTransformerEmbedder, VoyageEmbedder, cosine, mean_vector
from .fusion import channel_overlap, convex_combination, distribution_based_score_fusion, fuse, interleave, normalize_scores, reciprocal_rank_fusion
from .index import Chunk, Chunker, Document, DocumentStore, match_filter, split_sentences
from .model import HybridRetrievalModel, ResultRow, RetrievalOutput
from .plan import FilterClause, HeuristicPlanner, PlannerOutput, QueryFeatures, RetrievalPlan, analyze_query, apply_budget, estimate_llm_calls, n_windows
from .query_transform import HyDE, KeywordExtractor, MultiQueryExpander, Query2Doc, content_overlap
from .rerank import CrossEncoderReranker, LLMListwiseReranker, LLMPointwiseReranker, NoopReranker, Reranker, make_reranker, repair_permutation, sliding_windows
from .retriever import ChannelStats, HybridRetriever, RetrievalResult, query_coverage, retrieval_confidence
from .sparse import BM25Index, RankBM25Index, Tokenizer, looks_like_identifier, make_bm25

__version__ = "1.0.0"

__all__ = [
    "ANNBackend",
    "BM25Index",
    "CallableEmbedder",
    "ChannelStats",
    "Chunk",
    "Chunker",
    "CrossEncoderReranker",
    "DenseIndex",
    "Document",
    "DocumentStore",
    "Embedder",
    "FilterClause",
    "HashingEmbedder",
    "HeuristicPlanner",
    "HyDE",
    "HybridRetrievalConfig",
    "HybridRetrievalModel",
    "HybridRetriever",
    "KeywordExtractor",
    "LLMListwiseReranker",
    "LLMPointwiseReranker",
    "MultiQueryExpander",
    "NoopReranker",
    "PlannerOutput",
    "Query2Doc",
    "QueryFeatures",
    "RankBM25Index",
    "Reranker",
    "ResultRow",
    "RetrievalOutput",
    "RetrievalPlan",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "Tokenizer",
    "VoyageEmbedder",
    "analyze_query",
    "apply_budget",
    "channel_overlap",
    "content_overlap",
    "convex_combination",
    "cosine",
    "distribution_based_score_fusion",
    "estimate_llm_calls",
    "fuse",
    "interleave",
    "looks_like_identifier",
    "make_bm25",
    "make_reranker",
    "match_filter",
    "mean_vector",
    "n_windows",
    "normalize_scores",
    "query_coverage",
    "reciprocal_rank_fusion",
    "repair_permutation",
    "retrieval_confidence",
    "sliding_windows",
    "split_sentences",
]
