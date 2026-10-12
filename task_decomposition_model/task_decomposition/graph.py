"""Task graph: typed sub-tasks with dependencies plus pure DAG algorithms.

The graph is the one data structure every strategy produces and the scheduler
consumes. Everything here is plain Python over pydantic models: no LLM, so it
is unit-testable and cheap to validate before any execution (HuggingGPT-style
``id``/``dep``/``args`` plans, validated rather than trusted).

Placeholders
------------
Sub-task inputs may reference the output of another sub-task with
LLMCompiler-style placeholders: ``$t1``, ``${t1}``, ``$t1.output``. The
scheduler substitutes them once the dependency has finished;
:meth:`TaskGraph.infer_dependencies` derives implicit edges from them.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import PlanValidationError

Complexity = Literal["low", "medium", "high"]

PLACEHOLDER_RE = re.compile(r"\$\{?([A-Za-z0-9_][A-Za-z0-9_\-]*)\}?(?:\.(?:output|answer))?")


class SubTask(BaseModel):
    """One node of a :class:`TaskGraph`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Stable identifier, e.g. 't1'")
    description: str = Field(..., min_length=1, description="What to do, phrased so it can be solved in isolation")
    depends_on: list[str] = Field(default_factory=list, description="Ids of sub-tasks whose outputs are needed first")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Named inputs; values may contain $id placeholders")
    expected_output: str = Field(default="", description="Shape or content of the expected result")
    verification_criteria: list[str] = Field(default_factory=list, description="Checks an external verifier or reviewer can apply")
    tool_hint: str = Field(default="llm", description="llm | retrieve | compute | verify | aggregate | plan_and_solve | <tool name>")
    complexity: Complexity = "medium"
    max_retries: int = Field(default=1, ge=0, le=5)

    def placeholder_refs(self) -> list[str]:
        """Ids referenced through ``$id`` placeholders in inputs or description."""
        refs = find_placeholders(self.inputs) + find_placeholders(self.description)
        out: list[str] = []
        for r in refs:
            if r != self.id and r not in out:
                out.append(r)
        return out


# ----------------------------------------------------------------- placeholders


def find_placeholders(value: Any) -> list[str]:
    """Return every ``$id`` reference found in a string, list or dict (recursively, in order)."""
    found: list[str] = []
    if isinstance(value, str):
        found.extend(m.group(1) for m in PLACEHOLDER_RE.finditer(value))
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(find_placeholders(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            found.extend(find_placeholders(v))
    return found


def substitute_placeholders(value: Any, outputs: dict[str, str]) -> Any:
    """Replace ``$id`` / ``$id.output`` references with the text in ``outputs``.

    Unknown ids are left untouched so literal dollar amounts survive.
    """
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            ref = m.group(1)
            return outputs[ref] if ref in outputs else m.group(0)

        return PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: substitute_placeholders(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_placeholders(v, outputs) for v in value]
    return value


# ------------------------------------------------------------ DAG algorithms


def find_cycle(deps: dict[str, list[str]]) -> Optional[list[str]]:
    """Return one cycle (as a list of ids) if the dependency map has one, else ``None``."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {n: WHITE for n in deps}
    stack: list[str] = []

    def visit(n: str) -> Optional[list[str]]:
        colour[n] = GREY
        stack.append(n)
        for d in deps.get(n, []):
            if d not in colour:
                continue
            if colour[d] == GREY:
                i = stack.index(d)
                return stack[i:] + [d]
            if colour[d] == WHITE:
                cyc = visit(d)
                if cyc:
                    return cyc
        stack.pop()
        colour[n] = BLACK
        return None

    for node in deps:
        if colour[node] == WHITE:
            cyc = visit(node)
            if cyc:
                return cyc
    return None


def topological_waves(deps: dict[str, list[str]]) -> list[list[str]]:
    """Group nodes into waves: wave *i* contains nodes whose dependencies all lie in waves < *i*.

    Nodes in one wave are mutually independent and may run concurrently. Raises
    :class:`PlanValidationError` on a cycle or a missing dependency.
    """
    missing = sorted({d for ds in deps.values() for d in ds if d not in deps})
    if missing:
        raise PlanValidationError(f"dependencies reference unknown sub-tasks: {missing}")
    cyc = find_cycle(deps)
    if cyc:
        raise PlanValidationError(f"dependency cycle: {' -> '.join(cyc)}")
    remaining = {n: set(ds) for n, ds in deps.items()}
    done: set[str] = set()
    waves: list[list[str]] = []
    while remaining:
        ready = [n for n, ds in remaining.items() if ds <= done]
        if not ready:  # pragma: no cover - guarded by find_cycle
            raise PlanValidationError("no ready nodes; graph is not a DAG")
        waves.append(ready)
        done.update(ready)
        for n in ready:
            remaining.pop(n)
    return waves


def critical_path_length(deps: dict[str, list[str]], weights: Optional[dict[str, float]] = None) -> float:
    """Longest path through the DAG, in node weights (default 1 per node)."""
    if not deps:
        return 0.0
    w = weights or {n: 1.0 for n in deps}
    memo: dict[str, float] = {}

    def longest(n: str) -> float:
        if n in memo:
            return memo[n]
        best = max((longest(d) for d in deps.get(n, []) if d in deps), default=0.0)
        memo[n] = best + w.get(n, 1.0)
        return memo[n]

    if find_cycle(deps):
        raise PlanValidationError("cannot compute critical path of a cyclic graph")
    return max(longest(n) for n in deps)


# --------------------------------------------------------------------- graph


class TaskGraph(BaseModel):
    """A validated DAG of :class:`SubTask` nodes for one root task."""

    model_config = ConfigDict(extra="forbid")

    root_task: str = Field(..., min_length=1)
    tasks: list[SubTask] = Field(default_factory=list)
    strategy: str = Field(default="", description="Strategy that produced the graph")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---------------------------------------------------------------- access
    def ids(self) -> list[str]:
        return [t.id for t in self.tasks]

    def get(self, task_id: str) -> SubTask:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    def deps(self) -> dict[str, list[str]]:
        return {t.id: list(t.depends_on) for t in self.tasks}

    def dependents(self, task_id: str) -> list[str]:
        return [t.id for t in self.tasks if task_id in t.depends_on]

    def descendants(self, task_id: str) -> set[str]:
        out: set[str] = set()
        frontier = [task_id]
        while frontier:
            cur = frontier.pop()
            for d in self.dependents(cur):
                if d not in out:
                    out.add(d)
                    frontier.append(d)
        return out

    # ------------------------------------------------------------ mutation
    def infer_dependencies(self) -> "TaskGraph":
        """Add edges implied by ``$id`` placeholders (LLMCompiler-style) and drop self-edges."""
        known = set(self.ids())
        for t in self.tasks:
            deps = [d for d in t.depends_on if d != t.id]
            for ref in t.placeholder_refs():
                if ref in known and ref not in deps:
                    deps.append(ref)
            t.depends_on = deps
        return self

    def with_tasks(self, tasks: Iterable[SubTask]) -> "TaskGraph":
        return TaskGraph(root_task=self.root_task, tasks=list(tasks), strategy=self.strategy, metadata=dict(self.metadata))

    # ---------------------------------------------------------- validation
    def validate_structure(
        self,
        *,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        max_nodes: Optional[int] = None,
        detect_duplicates: bool = True,
    ) -> list[str]:
        """Return a list of defects (empty when the graph is valid)."""
        issues: list[str] = []
        if not self.tasks:
            issues.append("graph has no sub-tasks")
            return issues
        ids = self.ids()
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            issues.append(f"duplicate sub-task ids: {dupes}")
            return issues
        known = set(ids)
        for t in self.tasks:
            for d in t.depends_on:
                if d == t.id:
                    issues.append(f"{t.id} depends on itself")
                elif d not in known:
                    issues.append(f"{t.id} depends on unknown sub-task {d!r}")
            for ref in t.placeholder_refs():
                if ref in known and ref not in t.depends_on:
                    issues.append(f"{t.id} references ${ref} but does not depend on it")
        if issues:
            return issues
        cyc = find_cycle(self.deps())
        if cyc:
            issues.append(f"dependency cycle: {' -> '.join(cyc)}")
            return issues
        if max_nodes is not None and len(self.tasks) > max_nodes:
            issues.append(f"{len(self.tasks)} sub-tasks exceed the node cap {max_nodes}")
        waves = topological_waves(self.deps())
        if max_depth is not None and len(waves) > max_depth:
            issues.append(f"depth {len(waves)} exceeds max_depth {max_depth}")
        if max_breadth is not None:
            widest = max(len(w) for w in waves)
            if widest > max_breadth:
                issues.append(f"wave breadth {widest} exceeds max_breadth {max_breadth}")
        if detect_duplicates:
            seen: dict[tuple[str, str], str] = {}
            for t in self.tasks:
                key = (" ".join(t.description.lower().split()), repr(sorted(t.inputs.items(), key=lambda kv: kv[0])))
                if key in seen:
                    issues.append(f"{t.id} duplicates {seen[key]} (same description and inputs; possible loop)")
                else:
                    seen[key] = t.id
        return issues

    def ensure_valid(self, **bounds: Any) -> "TaskGraph":
        issues = self.validate_structure(**bounds)
        if issues:
            raise PlanValidationError("; ".join(issues))
        return self

    # ------------------------------------------------------------ metrics
    def topological_waves(self) -> list[list[str]]:
        return topological_waves(self.deps())

    def critical_path_length(self) -> int:
        return int(critical_path_length(self.deps())) if self.tasks else 0

    def depth(self) -> int:
        return len(self.topological_waves()) if self.tasks else 0

    def breadth(self) -> int:
        return max((len(w) for w in self.topological_waves()), default=0)

    def parallel_speedup_estimate(self) -> float:
        """Total nodes / critical path: the latency ratio of sequential vs wave execution."""
        cp = self.critical_path_length()
        return len(self.tasks) / cp if cp else 1.0

    # ---------------------------------------------------------- rendering
    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for t in self.tasks:
            label = t.description.replace('"', "'")
            if len(label) > 60:
                label = label[:57] + "..."
            lines.append(f'    {t.id}["{t.id}: {label}"]')
        for t in self.tasks:
            for d in t.depends_on:
                lines.append(f"    {d} --> {t.id}")
        return "\n".join(lines)

    def to_ascii(self) -> str:
        if not self.tasks:
            return "(empty graph)"
        lines: list[str] = []
        for i, wave in enumerate(self.topological_waves(), 1):
            lines.append(f"wave {i}:")
            for tid in wave:
                t = self.get(tid)
                dep = f" <- {', '.join(t.depends_on)}" if t.depends_on else ""
                lines.append(f"  [{t.id}] ({t.tool_hint}, {t.complexity}) {t.description}{dep}")
        return "\n".join(lines)
