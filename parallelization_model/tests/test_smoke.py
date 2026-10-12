"""Checkpoint test: the package imports, registers, plans offline and runs end to end with MockLLM."""
from __future__ import annotations

import pipeline_core
from pipeline_core import MockLLM, Status, TaskRequest

from parallelization import ParallelizationModel, ParallelizationOutput


def test_registered_under_component_name():
    assert pipeline_core.get("parallelization") is ParallelizationModel
    assert ParallelizationModel.spec.name == "parallelization"


async def test_offline_end_to_end_sectioning():
    model = ParallelizationModel(MockLLM())
    result = await model.run(TaskRequest(task="Compare steel, aluminium and titanium for a lightweight bracket.", domain="mechanical_engineering"))
    assert result.status in (Status.SUCCESS, Status.PARTIAL)
    assert result.component == "parallelization"
    assert isinstance(result.output, ParallelizationOutput)
    assert result.output.pattern == "sectioning"
    assert result.metadata["plan"]["pattern"] == "sectioning"
    assert len(result.metadata["plan"]["partitions"]) == 3
    assert result.output_text
    assert result.usage.llm_calls == 4  # 3 partitions + 1 merge
    assert 0.0 < result.confidence <= 1.0
    assert result.confidence_basis


def test_run_sync_sequential_fallback_for_dependent_task():
    model = ParallelizationModel(MockLLM())
    result = model.run_sync("Derive the deflection formula for a simply supported beam step by step.")
    assert result.ok
    assert result.output.pattern == "sequential"
    assert result.usage.llm_calls == 1
