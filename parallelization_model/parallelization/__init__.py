"""parallelization - concurrent, resilient execution of LLM and tool work.

Quick start::

    from parallelization import ParallelizationModel

    model = ParallelizationModel()            # MockLLM without credentials
    result = model.run_sync("Compare steel, aluminium and titanium for a bracket", domain="mechanical_engineering")
    print(result.summary()); print(result.metadata["plan"]["pattern"]); print(result.output_text)

The model registers itself with ``pipeline_core`` under the name
``"parallelization"`` and exposes ``model.executor`` (an ``Executor``) for
other components.
"""
from __future__ import annotations

from .aggregation import (
    VoteResult,
    canonicalize_answer,
    combine_confidence,
    concat_in_order,
    confidence_from_vote,
    label_to_confidence,
    majority_vote,
    missing_branch_notice,
    success_fraction,
    vote_entropy,
    weighted_confidence,
)
from .config import ParallelizationConfig
from .dag import critical_path, to_mermaid, topological_waves, total_work, transitive_dependents, validate_dag
from .estimator import SpeedupEstimate, TradeoffRow, amdahl_bound, amdahl_speedup, estimate_fan_out, estimate_graph, recommended_concurrency, tradeoff_table
from .evaluation import EvaluationReport, RunMetrics, evaluate, metrics_from_result, percentile, synthetic_tasks
from .executor import (
    Bulkhead,
    CircuitBreaker,
    CircuitOpenError,
    ExecutorStats,
    FailurePolicy,
    ManualClock,
    MonotonicClock,
    ParallelExecutionError,
    ParallelExecutor,
    RateLimiter,
    RetryBudget,
    RetryPolicy,
    TaskOutcome,
    TokenBucket,
    TransientError,
)
from .model import ParallelizationModel, ParallelizationOutput, build_context
from .patterns import (
    BranchRecord,
    PatternResult,
    PatternRunner,
    PrefixWarmGate,
    ToolCall,
    ToolResult,
    branch_solve_merge,
    cache_aware_fan_out,
    ensemble,
    map_reduce,
    mixture_of_agents,
    parallel_tool_calls,
    skeleton_of_thought,
    speculative,
    voting,
)
from .plan import HeuristicPlanner, ParallelPlan, Partition, PlannerOutput, RateLimits, TaskFeatures, extract_features

__version__ = "1.0.0"

__all__ = [
    "BranchRecord",
    "Bulkhead",
    "CircuitBreaker",
    "CircuitOpenError",
    "EvaluationReport",
    "ExecutorStats",
    "FailurePolicy",
    "HeuristicPlanner",
    "ManualClock",
    "MonotonicClock",
    "ParallelExecutionError",
    "ParallelExecutor",
    "ParallelPlan",
    "ParallelizationConfig",
    "ParallelizationModel",
    "ParallelizationOutput",
    "Partition",
    "PatternResult",
    "PatternRunner",
    "PlannerOutput",
    "PrefixWarmGate",
    "RateLimiter",
    "RateLimits",
    "RetryBudget",
    "RetryPolicy",
    "RunMetrics",
    "SpeedupEstimate",
    "TaskFeatures",
    "TaskOutcome",
    "TokenBucket",
    "ToolCall",
    "ToolResult",
    "TradeoffRow",
    "TransientError",
    "VoteResult",
    "amdahl_bound",
    "amdahl_speedup",
    "branch_solve_merge",
    "build_context",
    "cache_aware_fan_out",
    "canonicalize_answer",
    "combine_confidence",
    "concat_in_order",
    "confidence_from_vote",
    "critical_path",
    "ensemble",
    "estimate_fan_out",
    "estimate_graph",
    "evaluate",
    "extract_features",
    "label_to_confidence",
    "majority_vote",
    "map_reduce",
    "metrics_from_result",
    "missing_branch_notice",
    "mixture_of_agents",
    "parallel_tool_calls",
    "percentile",
    "recommended_concurrency",
    "skeleton_of_thought",
    "speculative",
    "success_fraction",
    "synthetic_tasks",
    "to_mermaid",
    "topological_waves",
    "total_work",
    "tradeoff_table",
    "transitive_dependents",
    "validate_dag",
    "vote_entropy",
    "voting",
    "weighted_confidence",
]
