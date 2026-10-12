"""Pure DAG helpers: validation, topological waves, critical path.

A graph is ``Mapping[node_id, Sequence[dependency_id]]``. These functions
have no LLM or asyncio dependency so they can be unit-tested exhaustively
and reused by the planner (LLMCompiler-style plans must be acyclic with
resolvable references), the executor (wave scheduling) and the estimator
(Amdahl / critical-path speedup).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from pipeline_core import PlanValidationError

Graph = Mapping[str, Sequence[str]]


def normalize_graph(graph: Graph) -> dict[str, list[str]]:
    """Return a copy with deterministic ordering and deduplicated dependencies."""
    out: dict[str, list[str]] = {}
    for node, deps in graph.items():
        seen: list[str] = []
        for d in deps:
            if d not in seen:
                seen.append(d)
        out[str(node)] = [str(d) for d in seen]
    return out


def validate_dag(graph: Graph, *, max_nodes: int | None = None) -> None:
    """Raise :class:`PlanValidationError` on unknown dependencies, self-loops or cycles."""
    g = normalize_graph(graph)
    if max_nodes is not None and len(g) > max_nodes:
        raise PlanValidationError(f"graph has {len(g)} nodes; maximum is {max_nodes}")
    for node, deps in g.items():
        for d in deps:
            if d == node:
                raise PlanValidationError(f"node '{node}' depends on itself")
            if d not in g:
                raise PlanValidationError(f"node '{node}' depends on unknown node '{d}'")
    # Kahn's algorithm; leftover nodes form a cycle.
    indeg = {n: len(deps) for n, deps in g.items()}
    ready = [n for n, k in indeg.items() if k == 0]
    visited = 0
    while ready:
        n = ready.pop()
        visited += 1
        for m, deps in g.items():
            if n in deps:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
    if visited != len(g):
        cyclic = sorted(n for n, k in indeg.items() if k > 0)
        raise PlanValidationError(f"graph contains a cycle among {cyclic}")


def topological_waves(graph: Graph) -> list[list[str]]:
    """Group nodes into waves; every node's dependencies are in earlier waves.

    Nodes inside a wave are independent and may run concurrently. Insertion
    order is preserved inside a wave so scheduling is deterministic.
    """
    g = normalize_graph(graph)
    validate_dag(g)
    level: dict[str, int] = {}

    def depth(n: str) -> int:
        if n in level:
            return level[n]
        level[n] = 0 if not g[n] else 1 + max(depth(d) for d in g[n])
        return level[n]

    for n in g:
        depth(n)
    waves: list[list[str]] = [[] for _ in range(max(level.values(), default=-1) + 1)]
    for n in g:  # insertion order
        waves[level[n]].append(n)
    return waves


def critical_path(graph: Graph, durations: Mapping[str, float] | None = None) -> tuple[float, list[str]]:
    """Longest weighted path through the DAG.

    ``durations`` maps node -> latency (default 1.0 per node). Returns the
    path length and the node sequence from source to sink.
    """
    g = normalize_graph(graph)
    validate_dag(g)
    dur = {n: float(durations.get(n, 1.0)) if durations else 1.0 for n in g}
    best: dict[str, float] = {}
    prev: dict[str, str | None] = {}

    def finish(n: str) -> float:
        if n in best:
            return best[n]
        if not g[n]:
            best[n] = dur[n]
            prev[n] = None
        else:
            p = max(g[n], key=finish)
            best[n] = finish(p) + dur[n]
            prev[n] = p
        return best[n]

    for n in g:
        finish(n)
    if not best:
        return 0.0, []
    sink = max(best, key=lambda n: best[n])
    path: list[str] = []
    cur: str | None = sink
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return best[sink], path


def total_work(graph: Graph, durations: Mapping[str, float] | None = None) -> float:
    """Sum of node durations (the sequential-execution latency)."""
    return sum(float(durations.get(n, 1.0)) if durations else 1.0 for n in graph)


def dependents(graph: Graph, node: str) -> list[str]:
    """Nodes that depend directly on ``node``."""
    return [n for n, deps in graph.items() if node in deps]


def transitive_dependents(graph: Graph, nodes: Iterable[str]) -> set[str]:
    """Every node downstream of any node in ``nodes`` (used to skip after a failure)."""
    g = normalize_graph(graph)
    out: set[str] = set()
    frontier = list(nodes)
    while frontier:
        n = frontier.pop()
        for m in dependents(g, n):
            if m not in out:
                out.add(m)
                frontier.append(m)
    return out


def to_mermaid(graph: Graph) -> str:
    """Render the DAG as a Mermaid flowchart (handy in READMEs and traces)."""
    lines = ["flowchart TD"]
    g = normalize_graph(graph)
    for n in g:
        lines.append(f"    {n}")
    for n, deps in g.items():
        for d in deps:
            lines.append(f"    {d} --> {n}")
    return "\n".join(lines)
