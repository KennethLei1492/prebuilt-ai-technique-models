"""pipeline_rag - end-to-end retrieval-augmented generation on ``pipeline_core``.

Route (Adaptive-RAG) -> transform queries (rewrite / multi-query / step-back /
HyDE, fused with RRF) -> retrieve (injected ``Retriever`` or the built-in
BM25 index) -> grade (CRAG) -> compress and order (RECOMP, Lost in the
Middle) -> generate with ``[doc-id]`` citations -> reflect (Self-RAG) ->
iterate for multi-hop questions (IRCoT / FLARE) -> escalate on abstention
(Self-Route).

Import surface::

    from pipeline_rag import PipelineRAGModel, PipelineRAGConfig, RAGPlan, RAGOutput

Importing the package registers :class:`PipelineRAGModel` with
``pipeline_core`` under the name ``"pipeline_rag"`` (``@register`` in
:mod:`pipeline_rag.model`).
"""
from __future__ import annotations

from .config import PipelineRAGConfig
from .context import ContextCompressor, ContextGrader, ContextOrderer, DocGrade, crag_decision, deduplicate, extractive_compress, order_for_position
from .generator import Citation, GenerationOutput, GroundedGenerator, Statement, citation_coverage, render_answer, split_statements
from .indexing import BM25Index, BuiltinRetriever, Chunk, IndexBuilder, SourceDocument, chunk_document, normalize_documents
from .iterate import HopRecord, HopStep, MultiHopLoop, ReflectionFlags, ReflectionJudge, SupportJudgement, critique_score, plan_hop_queries
from .model import ConfidenceSignals, PipelineRAGModel, RAGOutput, aggregate_confidence
from .plan import HeuristicPlanner, RAGPlan
from .router import QueryRouter, Route, RouteDecision, TaskFeatures, extract_task_features, route_for_features
from .transform import QueryTransformer, QueryVariant, heuristic_variants, reciprocal_rank_fusion

__version__ = "1.0.0"

__all__ = [
    "BM25Index",
    "BuiltinRetriever",
    "Chunk",
    "Citation",
    "ConfidenceSignals",
    "ContextCompressor",
    "ContextGrader",
    "ContextOrderer",
    "DocGrade",
    "GenerationOutput",
    "GroundedGenerator",
    "HeuristicPlanner",
    "HopRecord",
    "HopStep",
    "IndexBuilder",
    "MultiHopLoop",
    "PipelineRAGConfig",
    "PipelineRAGModel",
    "QueryRouter",
    "QueryTransformer",
    "QueryVariant",
    "RAGOutput",
    "RAGPlan",
    "ReflectionFlags",
    "ReflectionJudge",
    "Route",
    "RouteDecision",
    "SourceDocument",
    "Statement",
    "SupportJudgement",
    "TaskFeatures",
    "aggregate_confidence",
    "chunk_document",
    "citation_coverage",
    "crag_decision",
    "critique_score",
    "deduplicate",
    "extract_task_features",
    "extractive_compress",
    "heuristic_variants",
    "normalize_documents",
    "order_for_position",
    "plan_hop_queries",
    "reciprocal_rank_fusion",
    "render_answer",
    "route_for_features",
    "split_statements",
]
