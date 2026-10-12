"""Configuration for the caching model.

Every tunable has a literature-justified default; the README's configuration
table cites the source for each. Defaults follow the research dossier
``docs/research/caching.json``:

* eviction starts with LRU (Belady 1966; vLLM and SGLang defaults) and is
  size-aware in bytes, because GPTCache's entry-count eviction is a documented
  OOM pitfall;
* the semantic tier starts at a conservative cosine threshold of 0.95 with an
  explicit error budget (vCache) and exact-match guards on identifiers and
  numbers (Zhu et al. 2024);
* TTLs follow the dependency-based invalidation technique: 24 h positive TTL
  for grounded answers over versioned corpora, 5 min for time-sensitive
  answers, 5-15 min negative TTL (RFC 2308);
* provider prefix caching defaults to the 5-minute TTL and is promoted to
  1 hour when the measured reuse gap exceeds 5 minutes (reads refresh the
  TTL, so the 2x one-hour write pays off from the third use - Anthropic
  prompt-caching economics).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Effort

EvictionPolicy = Literal["lru", "lfu", "fifo"]
BackendName = Literal["memory", "sqlite", "file", "redis"]
PrefixTTL = Literal["5m", "1h"]


class KeyNormalization(BaseModel):
    """How prompt text is normalised before exact-match hashing.

    Lower-casing is off by default: identifiers, part numbers and code are
    case-sensitive (dossier engineering adaptation notes). Code blocks are
    never touched regardless of the other switches.

    Volatile stripping is **off** by default for the exact tier: a prompt that
    differs only in a timestamp, clock time, UUID or hex id (an incident window,
    a build id, a commit hash) asks a different question, and the exact tier is
    documented as zero-false-hit. Only explicitly *labelled* tracing ids
    (``request_id: ...``, ``trace_id=...``) are removed by default, because they
    never influence the answer. ``strip_volatile=True`` opts in to removing
    every :data:`caching.keys.VOLATILE_PATTERNS` match.
    """

    model_config = ConfigDict(extra="forbid")

    unicode_nfc: bool = Field(default=True, description="Apply Unicode NFC normalisation")
    collapse_whitespace: bool = Field(default=True, description="Trim and collapse runs of whitespace outside code blocks")
    lowercase: bool = Field(default=False, description="Lower-case prose (never code or identifiers)")
    strip_labelled_ids: bool = Field(default=True, description="Remove labelled tracing ids (request_id:/trace_id:/correlation_id=) before hashing")
    strip_volatile: bool = Field(default=False, description="Opt-in: also remove bare timestamps, clock times, UUIDs and 32-hex ids (changes the question; off for the exact tier)")
    stable_json: bool = Field(default=True, description="Re-serialise embedded JSON objects with sorted keys")


class CachingConfig(BaseModel):
    """All tunables of :class:`caching.CachingModel` and :class:`caching.CachedLLMClient`."""

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ backend
    backend: BackendName = Field(default="memory", description="Exact-match backend")
    eviction_policy: EvictionPolicy = Field(default="lru", description="LRU per Belady 1966 / vLLM / SGLang; LFU and FIFO available")
    max_entries: int = Field(default=10_000, ge=1)
    max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024, description="Size-aware capacity (GPTCache OOM pitfall)")
    sqlite_path: str = Field(default=":memory:", description="Path for the SQLite backend")
    file_dir: str = Field(default=".cache/caching", description="Directory for the file backend")
    redis_url: str = Field(default="redis://localhost:6379/0")
    key_prefix: str = Field(default="caching", description="Key prefix for shared stores (Redis)")
    tenant_salt: str = Field(default="", description="Per-tenant cache salt (vLLM cache_salt); isolates namespaces")
    normalization: KeyNormalization = Field(default_factory=KeyNormalization)

    # ---------------------------------------------------------------- TTLs
    default_ttl_s: Optional[float] = Field(default=86_400.0, gt=0, description="Positive TTL for grounded answers over versioned corpora (24 h)")
    time_sensitive_ttl_s: float = Field(default=300.0, gt=0, description="TTL for answers depending on live or time-sensitive data (5 min)")
    personal_ttl_s: float = Field(default=3_600.0, gt=0, description="TTL for personalised answers (exact tier only)")
    negative_ttl_s: float = Field(default=600.0, gt=0, description="Negative-cache TTL (RFC 2308: well below the positive TTL)")
    negative_caching: bool = Field(default=True, description="Cache deterministic failures (schema violations, refusals)")
    max_staleness_s: Optional[float] = Field(default=None, gt=0, description="Reject hits older than this even if the TTL has not elapsed")

    # ------------------------------------------------------------ semantic
    semantic_enabled: bool = True
    semantic_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Initial global cosine threshold for general embeddings")
    strict_semantic_threshold: float = Field(default=0.98, ge=0.0, le=1.0, description="Threshold for high-risk domains (engineering-strict profile)")
    semantic_top_k: int = Field(default=5, ge=1, le=100)
    semantic_error_budget: float = Field(default=0.02, ge=0.0, le=1.0, description="Maximum tolerated incorrect-hit rate (vCache error budget)")
    strict_error_budget: float = Field(default=0.01, ge=0.0, le=1.0)
    semantic_in_safety_critical: bool = Field(default=False, description="Allow the semantic tier for safety_critical profiles")
    verify_borderline_hits: bool = Field(default=True, description="LLM equivalence check for hits inside the borderline band")
    borderline_band: float = Field(default=0.02, ge=0.0, le=0.2, description="Similarity band above the threshold treated as borderline")
    identifier_guards: bool = Field(default=True, description="Require identical identifiers/numbers/units for semantic hits")
    context_turns: int = Field(default=3, ge=0, le=20, description="Prior turns hashed into the semantic key (MeanCache: 2-3)")
    semantic_max_entries: int = Field(default=5_000, ge=1)
    embedding_dimension: int = Field(default=256, ge=16, le=4096, description="Dimension of the offline HashingEmbedder built when no embedder is injected")

    # --------------------------------------------------------- concurrency
    coalesce_inflight: bool = Field(default=True, description="Single-flight coalescing of identical in-flight misses")
    xfetch_beta: float = Field(default=1.0, ge=0.0, description="XFetch eagerness; 1.0 is the paper's recommended default")
    bypass_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of bypassing the cache for diversity-requiring tasks (GPTCache knob, default 0)")

    # ------------------------------------------------------------- prefix
    prefix_min_tokens: Optional[int] = Field(default=None, ge=1, description="Override the model's minimum cacheable prefix")
    prefix_default_ttl: PrefixTTL = Field(default="5m")
    promote_to_1h_gap_s: float = Field(default=300.0, gt=0, description="Median reuse gap above which the 1-hour TTL is recommended")
    promote_to_1h_min_reuses_per_hour: int = Field(default=3, ge=1, description="Minimum observed uses of a prefix before the 1-hour TTL is recommended (the 2x write pays off from the third use)")
    apply_prefix_ttl: bool = Field(default=True, description="Wrapped client sends cache_ttl='1h' when the plan recommends the 1-hour provider TTL (never demotes an explicit request)")
    write_multiplier_5m: float = Field(default=1.25, gt=0)
    write_multiplier_1h: float = Field(default=2.0, gt=0)
    read_multiplier: float = Field(default=0.1, gt=0)

    # ------------------------------------------------------------ planner
    planner_effort: Effort = Field(default="low", description="Cacheability classification is routing work: cheap effort")
    planner_model: Optional[str] = Field(default=None, description="Optional cheaper model for the planning call")
    default_expected_reuses: int = Field(default=5, ge=1, description="Reuse assumption for savings estimates when none is given")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extra confidence floor on top of the domain profile")
