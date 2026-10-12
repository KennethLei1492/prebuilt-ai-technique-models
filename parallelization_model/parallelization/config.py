"""Configuration for the parallelization model.

Every tunable lives here with a default taken from the research dossier
(``docs/research/parallelization.json``). The README's configuration
table cites the source of each default; keep the two in sync.

Numbers that the literature leaves open (per-task timeouts, provider rate
limits) default to ``None``, meaning "not enforced", because a wrong guess is
worse than no limit: the dossier recommends deriving them from observed
latency percentiles and the provider's rate-limit headers.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Effort

Aggregator = Literal["code", "llm", "usc", "concat"]
PlannerMode = Literal["auto", "llm", "heuristic"]


class ParallelizationConfig(BaseModel):
    """All knobs of :class:`~parallelization.model.ParallelizationModel`.

    Grouped by the technique they control. See the README configuration table
    for the citation behind each default.
    """

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ concurrency
    max_concurrency: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Fan-out width per wave (LLMCompiler: 8-16 per level; Anthropic multi-agent: 3-5 workers).",
    )
    max_partitions: int = Field(default=16, ge=1, le=256, description="Upper bound on sub-tasks in one plan (LLMCompiler 8-16).")
    requests_per_minute: Optional[int] = Field(default=None, ge=1, description="Client-side RPM token bucket mirroring the provider limit.")
    input_tokens_per_minute: Optional[int] = Field(default=None, ge=1, description="Client-side ITPM token bucket.")
    output_tokens_per_minute: Optional[int] = Field(default=None, ge=1, description="Client-side OTPM token bucket.")
    rate_limit_burst_s: float = Field(default=5.0, gt=0, description="Bucket capacity in seconds of refill; keeps fan-out ramps gradual.")

    # ----------------------------------------------------------------- retry
    retry_max_attempts: int = Field(default=3, ge=1, le=10, description="Attempts per call (Google SRE: 3).")
    retry_base_delay_s: float = Field(default=0.5, ge=0.0, description="Backoff base (Brooker: 0.5-1 s).")
    retry_max_delay_s: float = Field(default=20.0, ge=0.0, description="Backoff cap (Brooker: 20-60 s).")
    retry_budget_ratio: float = Field(default=0.10, ge=0.0, le=1.0, description="Retries allowed as a fraction of requests (SRE: 10%).")
    retry_budget_min: int = Field(default=3, ge=0, description="Free retries so small runs can still retry at all.")
    retry_budget_window_s: float = Field(default=120.0, gt=0, description="Sliding window for the retry budget (SRE adaptive throttling: 2 min).")

    # ------------------------------------------------------- circuit breaker
    breaker_failure_threshold: int = Field(default=5, ge=1, description="Consecutive failures that open the breaker.")
    breaker_failure_rate: float = Field(default=0.5, gt=0.0, le=1.0, description="Failure rate over the window that opens the breaker.")
    breaker_window: int = Field(default=20, ge=1, description="Sliding window of recent calls for the failure rate.")
    breaker_recovery_s: float = Field(default=30.0, ge=0.0, description="Open duration before a half-open probe.")

    # ------------------------------------------------------ timeouts/hedging
    per_task_timeout_s: Optional[float] = Field(default=None, gt=0, description="Per-branch timeout; literature: 2-3x observed p95, so unset until measured.")
    hedge_after_s: Optional[float] = Field(default=None, gt=0, description="Issue a duplicate for idempotent branches after this delay (Dean & Barroso: p95).")
    hedge_fraction: float = Field(default=0.05, ge=0.0, le=1.0, description="Cap on hedged requests as a fraction of submissions.")

    # ------------------------------------------------------- failure policy
    quorum_fraction: float = Field(default=0.8, gt=0.0, le=1.0, description="Branches that must succeed before aggregation under quorum policy.")

    # --------------------------------------------------------------- voting
    voting_samples: int = Field(default=5, ge=1, description="Initial sample batch n0 (self-consistency / More Agents).")
    voting_max_samples: int = Field(default=10, ge=1, description="n_max after early-stopping check.")
    voting_samples_high_risk: int = Field(default=10, ge=1, description="Initial batch for high / safety-critical profiles (no early stop below this).")
    early_stop_margin: int = Field(default=2, ge=1, description="Leader margin in votes that stops sampling.")
    early_stop_leader_share: float = Field(default=0.6, gt=0.0, le=1.0, description="Leader share that stops sampling.")
    usc_max_candidates: int = Field(default=8, ge=2, description="Candidates shown to the Universal Self-Consistency selector (5-8).")

    # ------------------------------------------------------------- skeleton
    skeleton_min_points: int = Field(default=3, ge=1)
    skeleton_max_points: int = Field(default=10, ge=1)
    expansion_max_words: int = Field(default=150, ge=20, description="Per-point expansion cap (SoT: 100-200 words).")
    coherence_pass: bool = Field(default=False, description="Optional light coherence pass after concatenation (SoT).")

    # ----------------------------------------------------- branch-solve-merge
    bsm_min_branches: int = Field(default=3, ge=1)
    bsm_max_branches: int = Field(default=6, ge=1)

    # ------------------------------------------------------------ map-reduce
    chunk_tokens: int = Field(default=4000, ge=100, description="Map chunk size (LLM x MapReduce: 4k-16k).")
    chunk_overlap_ratio: float = Field(default=0.10, ge=0.0, lt=0.5, description="Prose overlap (10-15%).")
    map_reduce_min_tokens: int = Field(default=3000, ge=100, description="Input size that triggers map-reduce.")
    reduce_fan_in: int = Field(default=8, ge=2, description="Records per reducer (5-10).")

    # -------------------------------------------------------------- ensemble
    ensemble_layers: int = Field(default=2, ge=1, description="MoA-Lite: 2 layers.")
    ensemble_proposers: int = Field(default=3, ge=1, description="Proposers per layer (MoA-Lite: 3-6).")
    homogeneous_ensemble: bool = Field(default=True, description="Self-MoA: sample the best model instead of mixing models.")
    ensemble_proposer_models: list[str] = Field(default_factory=list, description="Opt-in heterogeneous proposers (used only when homogeneous_ensemble is False); proposer i uses models[i % len].")

    # --------------------------------------------------------------- budgets
    token_multiplier_cap: float = Field(default=15.0, ge=1.0, description="Max parallel calls relative to a single-call baseline (Anthropic: ~15x).")
    serial_fraction_target: float = Field(default=0.30, gt=0.0, lt=1.0, description="Keep planner+aggregator under this share of the critical path.")

    # ----------------------------------------------------------------- cache
    cache_aware_fan_out: bool = Field(default=True, description="Warm-then-burst fan-out when the shared prefix is cacheable.")

    # ----------------------------------------------------------------- models
    fast_model: str = Field(default="claude-haiku-5-5", description="Cheap model for the speculative fast path.")
    planner_effort: Effort = "high"
    branch_effort: Effort = "medium"
    map_effort: Effort = "low"
    aggregator_effort: Effort = "high"
    planner_mode: PlannerMode = Field(
        default="auto",
        description="auto = one structured planning call when online (heuristics offline / deterministic); llm = always attempt the LLM planner (used with ScriptedLLM/ReplayLLM); heuristic = never call the LLM to plan.",
    )
    ordered_tools: list[str] = Field(default_factory=list, description="Tools with ordering side effects; executed sequentially.")

    def voting_initial(self, risk_level: str) -> int:
        """Initial sample count for a domain risk level."""
        if risk_level in ("high", "safety_critical"):
            return max(self.voting_samples, self.voting_samples_high_risk)
        return self.voting_samples
