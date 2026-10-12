"""Checkpoint tests: the package imports, registers, and runs end to end offline."""
from __future__ import annotations

import pipeline_core
from pipeline_core import MockLLM, Status, TaskRequest

from caching import CachePlan, CacheReport, CachingConfig, CachingModel


def test_package_registers_component():
    assert pipeline_core.get("caching") is CachingModel
    assert CachingModel.spec.name == "caching"


def test_default_config_matches_literature_defaults():
    cfg = CachingConfig()
    assert cfg.eviction_policy == "lru"
    assert cfg.semantic_threshold == 0.95
    assert cfg.semantic_error_budget == 0.02
    assert cfg.negative_ttl_s < cfg.default_ttl_s  # RFC 2308: negative TTL well below positive
    assert cfg.prefix_default_ttl == "5m"


async def test_run_offline_end_to_end_with_mock_llm():
    model = CachingModel(MockLLM())
    result = await model.run(TaskRequest(task="What does clause 4.2 of ISO 2768 require for a 120 mm dimension?", domain="mechanical_engineering"))
    assert result.status == Status.SUCCESS
    assert isinstance(result.output, CacheReport)
    assert result.usage.llm_calls == 0  # heuristic planner offline: no LLM call
    plan = CachePlan.model_validate(result.metadata["plan"])
    assert plan.cacheable and plan.tiers[0] == "exact"
    assert "semantic" in plan.tiers and plan.semantic_threshold >= 0.98  # identifiers present -> strict
    assert result.output_text.startswith("Cache plan for")
    assert 0.0 < result.confidence <= 1.0 and result.confidence_basis
    assert {s.name for s in result.steps} >= {"plan", "analyze", "keys", "prefix_audit", "estimate", "store"}


def test_run_sync_and_explain():
    model = CachingModel(MockLLM())
    report = model.explain("Brainstorm five names for a torque sensor product line")
    assert report.plan.cacheable is False and report.plan.task_class == "random"
    assert any("Do not response-cache" in r for r in report.recommendations)
