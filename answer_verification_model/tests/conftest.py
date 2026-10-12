"""Shared fixtures: a dispatching MockLLM responder keyed on the prompt template name."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from pipeline_core import Evidence, LLMRequest, MockLLM, example_from_schema

Table = dict[str, Any]


def template_name(req: LLMRequest) -> str:
    """The template that produced ``req`` (first tag), without the package prefix."""
    tag = req.tags[0] if req.tags else ""
    return tag.split(".", 1)[1] if "." in tag else tag


def make_responder(table: Table) -> Callable[[LLMRequest], Any]:
    """Return a MockLLM responder that answers per template; unknown templates get schema examples."""

    def responder(req: LLMRequest) -> Any:
        name = template_name(req)
        if name in table:
            value = table[name]
            return value(req) if callable(value) else value
        if req.output_schema is not None:
            return example_from_schema(req.output_schema)
        return f"[mock] {req.last_user_text()[:80]}"

    return responder


@pytest.fixture
def dispatch_llm():
    """Factory: ``dispatch_llm({"derive": {...}, ...})`` -> MockLLM."""

    def factory(table: Table) -> MockLLM:
        return MockLLM(make_responder(table))

    return factory


@pytest.fixture
def spec_evidence() -> list[Evidence]:
    return [
        Evidence(id="spec-1", content="Section 4.2: the pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces.", source="pump spec", score=1.0, retriever="provided"),
        Evidence(id="spec-2", content="Table 2: discharge piping design pressure 16 bar at 80 degC.", source="pump spec", score=0.8, retriever="provided"),
    ]
