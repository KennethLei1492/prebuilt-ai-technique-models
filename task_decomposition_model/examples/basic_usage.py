"""Basic usage of the task decomposition model.

Runs offline: ``make_llm()`` returns a deterministic MockLLM when no Anthropic
credential is available, so the plan, graph and status logic are real while
the sub-task answers are stubs. With ``ANTHROPIC_API_KEY`` set the same code
runs against Claude.

    python task_decomposition_model/examples/basic_usage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "task_decomposition_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import Constraints, TaskRequest, make_llm  # noqa: E402

from task_decomposition import TaskDecompositionModel  # noqa: E402

TASK = "For each of the three pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status."


def main() -> None:
    model = TaskDecompositionModel(make_llm())

    # 1. One call does everything: plan -> decompose -> schedule -> synthesise.
    result = model.run_sync(TASK)
    print(result.summary())
    print(result.confidence_basis)
    print()
    print(result.output.render())
    print()

    # 2. The same task in a safety-critical domain: strict verification, a verify node,
    #    and NEEDS_REVIEW because no external verifier is injected.
    critical = model.run_sync(TaskRequest(task=TASK, domain="civil_structural_engineering"))
    print(f"general -> {result.status.value} (verification={result.output.plan.verification})")
    print(f"civil_structural_engineering -> {critical.status.value} (verification={critical.output.plan.verification}, verify node={'verify' in critical.output.graph.ids()})")
    print()

    # 3. Plan-only / human checkpoint: inspect or edit the graph before executing it.
    graph = model.decompose_sync("Debug why the nightly ETL job fails intermittently and propose a fix with a regression test.", domain="data_engineering")
    print("plan:", graph.metadata["plan"]["strategy"], "|", graph.metadata["plan"]["rationale"])
    print(graph.to_mermaid())
    print()

    # 4. Budgets are enforced on every call; the run degrades instead of failing.
    budgeted = model.run_sync(TaskRequest(task=TASK, constraints=Constraints(max_llm_calls=2)))
    print(f"with max_llm_calls=2 -> status={budgeted.status.value} calls={budgeted.usage.llm_calls} stopped={budgeted.metadata.get('stopped_reason')}")
    print()
    print(model.tracer.tree().splitlines()[0])


if __name__ == "__main__":
    main()
