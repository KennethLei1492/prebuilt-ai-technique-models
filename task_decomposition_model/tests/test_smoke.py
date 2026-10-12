"""Checkpoint test: the package imports, registers, and runs end to end offline."""
from __future__ import annotations

from pipeline_core import MockLLM, Status, get

from task_decomposition import DecompositionOutput, TaskDecompositionModel


def test_registered_under_component_name():
    assert get("task_decomposition") is TaskDecompositionModel
    assert TaskDecompositionModel.spec.name == "task_decomposition"


async def test_end_to_end_offline_dag():
    model = TaskDecompositionModel(MockLLM())
    task = "For each of the three pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status."
    result = await model.run(task)
    assert result.ok, (result.status, result.errors, result.warnings)
    assert result.status == Status.SUCCESS
    out = result.output
    assert isinstance(out, DecompositionOutput)
    assert out.plan.strategy == "dag"
    assert len(out.graph.tasks) >= 3
    assert all(r.ok for r in out.results.values())
    assert result.output_text
    assert result.metadata["plan"]["strategy"] == "dag"
    assert result.usage.llm_calls == len(out.graph.tasks)  # plan + decompose + synthesis are heuristic offline
    assert 0.0 < result.confidence <= model.config.offline_confidence_cap
    assert any(s.name == "decompose" for s in result.steps) and any(s.name.startswith("subtask.") for s in result.steps)
