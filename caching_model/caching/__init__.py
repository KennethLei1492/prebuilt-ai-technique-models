"""``caching`` - response caching, semantic caching, coalescing and prefix-cache optimisation.

Import surface::

    from caching import CachingModel, CachingConfig, CachePlan, CachedLLMClient

    model = CachingModel()                      # MockLLM without credentials
    client = model.wrap(make_llm())             # shared client for every other component
    report = model.explain("What does clause 4.2 of ISO 2768 require?", domain="mechanical_engineering")

Importing the package registers :class:`CachingModel` under the name ``"caching"``
with :func:`pipeline_core.register`.
"""
from __future__ import annotations

from .backends import BackendStats, CacheBackend, CacheEntry, FileBackend, InMemoryBackend, RedisBackend, SQLiteBackend, make_backend
from .config import BackendName, CachingConfig, EvictionPolicy, KeyNormalization, PrefixTTL
from .evaluation import (
    EvaluationReport,
    belady_opt_hits,
    compare_policies,
    cost_saved,
    false_hit_audit,
    latency_saved,
    prefix_savings,
    run_offline_evaluation,
    simulate_policy,
)
from .invalidation import DependencyFingerprint, InvalidationEvent, InvalidationManager, VersionRegistry
from .keys import (
    IDENTIFIER_PATTERNS,
    VOLATILE_PATTERNS,
    context_chain_key,
    exact_key,
    extract_guards,
    namespace_for,
    normalize_prompt_text,
    strip_volatile,
    tagged_keys,
)
from .middleware import CacheMetrics, CacheStore, CachedLLMClient
from .model import CacheReport, CachingModel, PlannerMode, SavingsEstimate
from .plan import CacheabilityFeatures, CachePlan, CacheTier, HeuristicPlanner, TaskClass, analyze_task
from .prefix import ArrivalStats, Invalidator, PrefixCacheAdvisor, PrefixReport, breakeven_uses, find_invalidators, find_unsorted_json, prefix_hit_ratio, recommend_ttl_from_gaps
from .prompts import CACHE_PLAN_TEMPLATE, EQUIVALENCE_TEMPLATE, TEMPLATES, EquivalenceVerdict
from .semantic import CallableEmbedder, Embedder, HashingEmbedder, SemanticCache, SemanticEntry, SemanticHit, SemanticMetrics, cosine
from .stampede import SingleFlight, xfetch_should_refresh

__version__ = "1.0.0"

__all__ = [
    "ArrivalStats",
    "BackendName",
    "BackendStats",
    "CACHE_PLAN_TEMPLATE",
    "CacheBackend",
    "CacheEntry",
    "CacheMetrics",
    "CachePlan",
    "CacheReport",
    "CacheStore",
    "CacheTier",
    "CacheabilityFeatures",
    "CachedLLMClient",
    "CachingConfig",
    "CachingModel",
    "CallableEmbedder",
    "DependencyFingerprint",
    "EQUIVALENCE_TEMPLATE",
    "Embedder",
    "EquivalenceVerdict",
    "EvaluationReport",
    "EvictionPolicy",
    "FileBackend",
    "HashingEmbedder",
    "HeuristicPlanner",
    "IDENTIFIER_PATTERNS",
    "InMemoryBackend",
    "InvalidationEvent",
    "InvalidationManager",
    "Invalidator",
    "KeyNormalization",
    "PlannerMode",
    "PrefixCacheAdvisor",
    "PrefixReport",
    "PrefixTTL",
    "RedisBackend",
    "SQLiteBackend",
    "SavingsEstimate",
    "SemanticCache",
    "SemanticEntry",
    "SemanticHit",
    "SemanticMetrics",
    "SingleFlight",
    "TEMPLATES",
    "TaskClass",
    "VOLATILE_PATTERNS",
    "VersionRegistry",
    "analyze_task",
    "belady_opt_hits",
    "breakeven_uses",
    "compare_policies",
    "context_chain_key",
    "cosine",
    "cost_saved",
    "exact_key",
    "extract_guards",
    "false_hit_audit",
    "find_invalidators",
    "find_unsorted_json",
    "latency_saved",
    "make_backend",
    "namespace_for",
    "normalize_prompt_text",
    "prefix_hit_ratio",
    "prefix_savings",
    "recommend_ttl_from_gaps",
    "run_offline_evaluation",
    "simulate_policy",
    "strip_volatile",
    "tagged_keys",
    "xfetch_should_refresh",
]
