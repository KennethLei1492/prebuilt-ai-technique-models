"""Configuration for the task decomposition model.

Every tunable lives here with a default taken from the research dossier
(``docs/research/task-decomposition.draft.json`` and the two supporting
dossiers). The README's configuration table cites the source for each value;
keep the two in sync when changing a default.

Design stance (from the literature):

* Decompose lazily. Least-to-Most is roughly neutral on 2-step problems and
  pays off at 5+ steps, so decomposition is gated on an estimated step depth.
* Hard budgets everywhere: node cap, depth cap, replan cap, retry cap,
  reflection memory cap. Task Limit Exceeded is the dominant agent failure
  (AgentBench), and multi-layer retries amplify load (Google SRE).
* The planner is a proposer, not an oracle: plans are machine-validated
  (acyclic, dependencies exist, bounds) and repaired at most once before a
  heuristic fallback (Valmeekam et al., HuggingGPT planning F1 ~50%).
* Replanning is triggered only by external signals (solver failure, verifier
  verdict), never by the model re-reading its own output (Huang et al.).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class TaskDecompositionConfig(BaseModel):
    """Tunables for :class:`~task_decomposition.model.TaskDecompositionModel`."""

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------ gating
    decompose_min_steps: int = Field(
        default=3,
        ge=1,
        description="Estimated reasoning steps below which the task is answered directly (Least-to-Most: neutral at 2 steps, gains at 5+).",
    )
    min_subtasks: int = Field(default=3, ge=1, description="Lower bound on sub-tasks in a decomposition (ADaPT 3-5, dossier 3-8).")
    max_subtasks: int = Field(default=8, ge=1, description="Upper bound on sub-tasks per plan level (dossier 3-8).")
    max_nodes: int = Field(default=25, ge=1, description="Per-plan node cap after expansion/replanning (architecture recommendation 8).")

    # ------------------------------------------------------------- depth
    max_depth: int = Field(default=3, ge=1, le=6, description="Recursion depth bound d_max (ADaPT: 3; 4 for deeply compositional domains).")
    deep_domain_depth: int = Field(default=4, ge=1, le=6, description="Depth bound used when the task is estimated as highly compositional.")
    deep_domain_steps: int = Field(default=6, ge=1, description="Estimated step count at or above which deep_domain_depth applies.")

    # ------------------------------------------------------- parallelism
    max_parallelism: int = Field(default=8, ge=1, le=64, description="Concurrency ceiling for a wave (dossier: tool rate limits, typically 4-16).")

    # ------------------------------------------------ retries / replans
    max_retries_per_task: int = Field(default=1, ge=0, le=5, description="Retries per node before it is marked failed and replanning is considered (dossier: 1 retry).")
    retry_base_delay_s: float = Field(default=0.5, ge=0.0, description="Backoff base delay (SRE guidance: 0.5-1 s, factor 2, full jitter).")
    retry_factor: float = Field(default=2.0, ge=1.0, description="Exponential backoff factor.")
    retry_max_delay_s: float = Field(default=8.0, ge=0.0, description="Backoff ceiling per attempt.")
    max_replans: int = Field(default=2, ge=0, le=5, description="Bounded joiner/replanner iterations (LLMCompiler/dossier: 2-3).")
    reflection_memory: int = Field(default=3, ge=1, le=10, description="Reflexion episodic memory cap Omega (1-3 entries).")
    plan_repair_attempts: int = Field(default=1, ge=0, le=3, description="LLM repair passes for an invalid plan before the heuristic fallback (architecture recommendation 3).")

    # ------------------------------------------------------- recursion
    recursive_expand_below: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Solver confidence below which the recursive strategy expands a node; a weak prior unless a verifier is injected (Huang et al.).",
    )

    # ----------------------------------------------------------- search
    tot_breadth: int = Field(default=5, ge=1, le=10, description="Tree-of-Thoughts BFS breadth b (ToT: b=5).")
    tot_depth: int = Field(default=2, ge=1, le=4, description="Tree-of-Thoughts search depth (ToT generation tasks: depth 2).")
    tot_votes: int = Field(default=3, ge=1, le=10, description="Evaluator samples per state, averaged (ToT: v=3).")
    enable_search: bool = Field(default=True, description="Allow the tree_of_thoughts strategy; it costs several times a single call.")

    # --------------------------------------------------------- confidence
    composition_discount: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Confidence discount applied when composing several sub-answers (Self-Ask compositionality gap ~40% motivates a composition penalty).",
    )
    offline_confidence_cap: float = Field(default=0.7, ge=0.0, le=1.0, description="Ceiling for structural confidence when running on MockLLM.")
    unverified_confidence_cap: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ceiling when the plan requires an external verifier but none is injected (no external signal -> cannot auto-accept).",
    )
    agreement_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Self-consistency agreement considered 'confident' (dossier ~0.6).")

    # ----------------------------------------------------------- prompts
    planner_effort: Effort = Field(default="high", description="Effort for the planning and decomposition calls.")
    solver_effort: Effort = Field(default="medium", description="Effort for leaf sub-task solving.")
    synthesis_effort: Effort = Field(default="high", description="Effort for the final synthesis call.")
    safety_critical_verification_effort: Effort = Field(default="max", description="Effort reserved for final verification in safety-critical domains.")
    solver_model: Optional[str] = Field(default=None, description="Optional cheaper model for leaf sub-tasks (ReWOO / LangChain guidance). None keeps one model so caches stay scoped.")
    max_subtask_output_chars: int = Field(default=4000, ge=200, description="Truncate per-node outputs passed to dependents (ReWOO compact per-node context).")
    max_context_docs: int = Field(default=12, ge=1, description="Evidence items rendered into a sub-task prompt (reader accuracy saturates near 10-20 passages).")
    retrieve_k: int = Field(default=5, ge=1, description="Evidence retrieved per sub-question when a Retriever is injected (Self-Ask + search).")
    always_append_original_task: bool = Field(
        default=True,
        description="Synthesis re-asks the original task with every sub-answer in context (Least-to-Most) even when every node completed; False skips that SYNTHESIZE call on fully successful runs and composes heuristic sections instead (the LLM still composes partial runs).",
    )

    # ------------------------------------------------------------- mode
    force_llm_path: bool = Field(
        default=False,
        description="Use the LLM-driven planner even when the client is a MockLLM subclass (ScriptedLLM tests, replay). Heuristics remain the fallback on error.",
    )
    loop_detection: bool = Field(default=True, description="Flag duplicate sub-tasks (same description and inputs) as a plan defect (AgentBench looping).")
