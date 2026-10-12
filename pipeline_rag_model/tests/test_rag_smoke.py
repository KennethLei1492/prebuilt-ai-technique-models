"""Checkpoint test: the package imports, registers, and answers end-to-end offline."""
from __future__ import annotations

import pipeline_core
from pipeline_core import MockLLM, Status, TaskRequest

from pipeline_rag import PipelineRAGModel, RAGOutput, RAGPlan

DOCS = [
    {
        "id": "spec-pump",
        "source": "Pump Specification PS-100 Rev C",
        "text": (
            "Section 4.2 Casing. The pump casing shall be ASTM A216 WCB carbon steel. "
            "The specification requires a corrosion allowance of 3 mm for the pump casing in seawater service. "
            "Casing design pressure is 16 bar at 80 degC."
        ),
    },
    {
        "id": "spec-motor",
        "source": "Motor Data Sheet MD-7",
        "text": (
            "The drive motor is rated 75 kW at 1480 rpm. Service factor 1.15. "
            "Insulation class F with class B temperature rise. The motor shall be suitable for VFD operation."
        ),
    },
    {
        "id": "test-report",
        "source": "Hydrostatic Test Report HT-22",
        "text": (
            "Hydrostatic test of casing serial 0451 held 24 bar for 30 minutes with no visible leakage. "
            "Test water temperature 18 degC. Witnessed by the client inspector."
        ),
    },
]


def test_registered() -> None:
    assert pipeline_core.get("pipeline_rag") is PipelineRAGModel
    assert PipelineRAGModel.spec.name == "pipeline_rag"


def test_offline_end_to_end_single_step() -> None:
    model = PipelineRAGModel(MockLLM())
    request = TaskRequest(
        task="What corrosion allowance does the specification require for the pump casing?",
        inputs={"documents": DOCS},
    )
    result = model.run_sync(request)
    assert result.status in (Status.SUCCESS, Status.PARTIAL), result.warnings
    assert isinstance(result.output, RAGOutput)
    plan = RAGPlan.model_validate(result.metadata["plan"])
    assert plan.route == "single_step"
    assert plan.planner == "heuristic"
    assert "3 mm" in result.output_text
    assert "[spec-pump]" in result.output.answer
    assert result.confidence > 0.5
    assert result.usage.llm_calls == 0  # offline: every stage is heuristic
    assert any(e.id == "spec-pump" for e in result.evidence)
    assert result.output.cited_ids == ["spec-pump"]
