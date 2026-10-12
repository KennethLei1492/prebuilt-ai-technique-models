"""LLM-facing structured-output schemas and their conversion to internal types.

These models are what the model is asked to emit (through ``output_schema``).
They are deliberately flat, enum-typed and free of numeric bounds, string
lengths or free-form dictionaries, because constrained decoders reject or
ignore those (see the prompting dossier's structured-output limits). Values
are clamped or normalised by validators after parsing instead.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline_core import sanitize_schema

from .graph import SubTask, TaskGraph

ToolHint = Literal["llm", "retrieve", "compute", "verify", "aggregate"]


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


class InputBinding(BaseModel):
    """One named input of a sub-task; ``value`` may contain ``$id`` placeholders."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str


class SubTaskDraft(BaseModel):
    """A sub-task as emitted by the decomposer LLM."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Short stable id such as t1, t2")
    description: str = Field(..., description="Self-contained instruction for this sub-task")
    depends_on: list[str] = Field(default_factory=list, description="Ids whose outputs are required first; empty when independent")
    inputs: list[InputBinding] = Field(default_factory=list, description="Named inputs; reference earlier outputs with $id.output")
    expected_output: str = Field(default="", description="What a correct output looks like")
    verification_criteria: list[str] = Field(default_factory=list, description="Externally checkable success criteria")
    tool_hint: ToolHint = Field(default="llm", description="llm | retrieve | compute | verify | aggregate")
    complexity: Literal["low", "medium", "high"] = "medium"

    def to_subtask(self, *, max_retries: int = 1) -> SubTask:
        return SubTask(
            id=self.id.strip() or "t",
            description=self.description.strip(),
            depends_on=[d.strip() for d in self.depends_on if d.strip()],
            inputs={b.name: b.value for b in self.inputs if b.name},
            expected_output=self.expected_output,
            verification_criteria=[c for c in self.verification_criteria if c.strip()],
            tool_hint=self.tool_hint,
            complexity=self.complexity,
            max_retries=max_retries,
        )


class GraphDraft(BaseModel):
    """The decomposer's answer: an ordered list of sub-tasks with explicit dependencies."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[SubTaskDraft] = Field(default_factory=list)
    notes: str = Field(default="", description="Assumptions or ambiguities worth surfacing (one or two sentences)")

    def to_graph(self, root_task: str, *, strategy: str = "", max_retries: int = 1) -> TaskGraph:
        tasks = [t.to_subtask(max_retries=max_retries) for t in self.tasks if t.description.strip()]
        graph = TaskGraph(root_task=root_task, tasks=tasks, strategy=strategy, metadata={"notes": self.notes} if self.notes else {})
        return graph.infer_dependencies()


class CandidateGraphs(BaseModel):
    """Several alternative decompositions for Tree-of-Thoughts style selection."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[GraphDraft] = Field(default_factory=list)


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., description="0-based index of the candidate")
    score: float = Field(..., description="0.0 (impossible) to 1.0 (sure) that executing this decomposition solves the task")
    verdict: Literal["sure", "likely", "impossible"] = "likely"
    issue: str = Field(default="", description="Main weakness, if any")

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return _clamp01(v)


class CandidateScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[CandidateScore] = Field(default_factory=list)


class SubTaskAnswer(BaseModel):
    """Structured answer to one sub-task."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="The result for this sub-task; complete and self-contained")
    confidence: float = Field(..., description="0.0-1.0 probability the answer is correct and complete")
    status: Literal["completed", "partial", "failed"] = Field(default="completed", description="failed when the inputs do not allow a correct answer")
    criteria_met: list[bool] = Field(default_factory=list, description="One entry per verification criterion, in order")
    evidence_ids: list[str] = Field(default_factory=list, description="Ids of documents that support the answer")
    gaps: list[str] = Field(default_factory=list, description="Missing information or unverifiable assumptions")

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return _clamp01(v)


class Reflection(BaseModel):
    """Reflexion-style structured note written after an externally observed failure."""

    model_config = ConfigDict(extra="forbid")

    root_cause: str
    change: str = Field(..., description="What to do differently")
    affected_task_ids: list[str] = Field(default_factory=list)


class Revision(BaseModel):
    """Replanner output: a reflection plus replacement sub-tasks for the failed region."""

    model_config = ConfigDict(extra="forbid")

    reflection: Reflection
    replacement_tasks: list[SubTaskDraft] = Field(default_factory=list, description="Sub-tasks that replace the failed ones; may depend on completed ids")
    give_up: bool = Field(default=False, description="True when the failure cannot be fixed by replanning")


class Contribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    contribution: str = Field(..., description="What this sub-task contributed to the final answer")


class SynthesisOutput(BaseModel):
    """Final composed answer with traceability to sub-tasks."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(..., description="0.0-1.0 probability the composed answer is correct and complete")
    traceability: list[Contribution] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list, description="Open questions, gaps or conflicts between sub-answers")

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return _clamp01(v)


class RequirementCoverage(BaseModel):
    """One atomic requirement of the task and the sub-tasks that address it (LLM coverage judge)."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(..., description="One atomic requirement stated or clearly implied by the task")
    covered_by: list[str] = Field(default_factory=list, description="Ids of sub-tasks that address this requirement; empty when none does")


class CoverageJudgement(BaseModel):
    """Coverage judge output: the task's requirements and which sub-tasks cover each."""

    model_config = ConfigDict(extra="forbid")

    requirements: list[RequirementCoverage] = Field(default_factory=list)

    def coverage(self, known_ids: set[str]) -> float:
        """Fraction of requirements covered by at least one sub-task that exists in the graph."""
        if not self.requirements:
            return 0.0
        covered = sum(1 for r in self.requirements if any(i in known_ids for i in r.covered_by))
        return covered / len(self.requirements)


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Sanitised JSON schema for structured output."""
    return sanitize_schema(model.model_json_schema())


def plan_and_solve_schema() -> dict[str, Any]:
    return schema_for(SubTaskAnswer)

