"""task_decomposition - turn a complex task into a validated, executable plan.

Quick start::

    from task_decomposition import TaskDecompositionModel

    model = TaskDecompositionModel()                       # MockLLM without credentials
    result = model.run_sync("For each pump log extract the duty point, then summarise", domain="mechanical_engineering")
    print(result.summary()); print(result.output.render())

Importing the package registers ``TaskDecompositionModel`` with
``pipeline_core.registry`` under the name ``task_decomposition``.
"""
from __future__ import annotations

from .config import Effort, TaskDecompositionConfig
from .evaluation import (
    CaseOutcome,
    EvalCase,
    EvaluationSummary,
    PlanQualityReport,
    compositionality_gap,
    coverage,
    dependency_correctness,
    evaluate,
    evaluate_plan,
    evaluate_sync,
    failure_classes,
    judge_coverage,
    llm_coverage,
    parallel_speedup,
    sample_cases,
)
from .graph import SubTask, TaskGraph, critical_path_length, find_cycle, find_placeholders, substitute_placeholders, topological_waves
from .model import DecompositionOutput, TaskDecompositionModel
from .plan import DecompositionPlan, HeuristicPlanner, TaskFeatures, extract_features, normalize_plan
from .replanner import ReflectionMemory, Replanner, apply_revision, replannable_failures
from .scheduler import BoundedExecutor, RetryPolicy, Scheduler, ScheduleReport, SubTaskResult
from .schemas import CoverageJudgement, GraphDraft, Reflection, Revision, SubTaskAnswer, SubTaskDraft, SynthesisOutput
from .solvers import LLMSubTaskSolver, ProtocolSolver, VerifierNodeSolver, evidence_from_inputs, is_verification_node, is_verification_result
from .strategies import (
    STRATEGY_CLASSES,
    DAGPlanner,
    DirectStrategy,
    LeastToMost,
    PlanAndSolve,
    RecursiveDecomposer,
    Strategy,
    TreeOfThoughtsPlanner,
    aggregate_votes,
    chainify,
    get_strategy,
    heuristic_graph,
    heuristic_segments,
    structural_score,
)
from .synthesis import Synthesizer, aggregate_confidence, claims_text, coverage_ratio

__version__ = "1.0.0"

__all__ = [
    "BoundedExecutor",
    "CaseOutcome",
    "CoverageJudgement",
    "DAGPlanner",
    "DecompositionOutput",
    "DecompositionPlan",
    "DirectStrategy",
    "Effort",
    "EvalCase",
    "EvaluationSummary",
    "GraphDraft",
    "HeuristicPlanner",
    "LLMSubTaskSolver",
    "LeastToMost",
    "PlanAndSolve",
    "PlanQualityReport",
    "ProtocolSolver",
    "VerifierNodeSolver",
    "RecursiveDecomposer",
    "Reflection",
    "ReflectionMemory",
    "Replanner",
    "RetryPolicy",
    "Revision",
    "STRATEGY_CLASSES",
    "ScheduleReport",
    "Scheduler",
    "Strategy",
    "SubTask",
    "SubTaskAnswer",
    "SubTaskDraft",
    "SubTaskResult",
    "SynthesisOutput",
    "Synthesizer",
    "TaskDecompositionConfig",
    "TaskDecompositionModel",
    "TaskFeatures",
    "TaskGraph",
    "TreeOfThoughtsPlanner",
    "aggregate_confidence",
    "claims_text",
    "aggregate_votes",
    "apply_revision",
    "chainify",
    "compositionality_gap",
    "coverage",
    "coverage_ratio",
    "critical_path_length",
    "dependency_correctness",
    "evaluate",
    "evaluate_plan",
    "evaluate_sync",
    "evidence_from_inputs",
    "extract_features",
    "failure_classes",
    "find_cycle",
    "find_placeholders",
    "get_strategy",
    "heuristic_graph",
    "heuristic_segments",
    "is_verification_node",
    "is_verification_result",
    "judge_coverage",
    "llm_coverage",
    "normalize_plan",
    "parallel_speedup",
    "replannable_failures",
    "sample_cases",
    "structural_score",
    "substitute_placeholders",
    "topological_waves",
]
