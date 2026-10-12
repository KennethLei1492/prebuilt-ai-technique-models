"""CachedLLMClient middleware, direct API, protocols, status/confidence and budget behaviour."""
from __future__ import annotations

import asyncio

import pytest

from pipeline_core import (
    Constraints,
    Evidence,
    LLMRequest,
    LLMResponse,
    MockLLM,
    ResponseCache,
    ScriptedLLM,
    Status,
    StructuredOutputError,
    TaskRequest,
    Usage,
    get_profile,
)

from caching import CacheReport, CachedLLMClient, CachingConfig, CachingModel, InMemoryBackend, SemanticCache


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _req(text: str, **meta) -> LLMRequest:
    return LLMRequest.simple(text, system=["You answer engineering questions about standards."], metadata=meta)


# ------------------------------------------------------------------ middleware


async def test_exact_hit_returns_cached_response_with_cache_hit_usage():
    inner = MockLLM(simulate_cost_as="claude-opus-5-5")
    client = CachedLLMClient(inner, clock=FakeClock())
    r1 = await client.complete(_req("What does ISO 2768-m specify for a 120 mm dimension?"))
    r2 = await client.complete(_req("What does ISO 2768-m specify for a 120 mm dimension?"))
    assert not r1.cached and r2.cached
    assert r2.usage.cache_hits == 1 and r2.usage.llm_calls == 0 and r2.usage.cost_usd == 0.0
    assert r2.text == r1.text and len(inner.calls) == 1
    assert getattr(r2, "cache")["cache_tier"] == "exact"
    assert client.metrics.exact_hits == 1 and client.metrics.misses == 1 and client.metrics.cost_saved_usd > 0


async def test_whitespace_variants_hit_exact_tier():
    client = CachedLLMClient(MockLLM(), clock=FakeClock())
    await client.complete(_req("Define   factor of safety"))
    r = await client.complete(_req("Define factor of safety"))
    assert r.cached and client.metrics.exact_hits == 1


async def test_semantic_tier_serves_paraphrase_but_not_identifier_change():
    cfg = CachingConfig(semantic_threshold=0.6, strict_semantic_threshold=0.6, borderline_band=0.0)
    inner = MockLLM()
    client = CachedLLMClient(inner, config=cfg, clock=FakeClock())
    await client.complete(_req("What does clause 4.2 of ISO 2768 require for a 12 mm dimension?"))
    r = await client.complete(_req("What does clause 4.2 of ISO 2768 require for a 12 mm dimension please?"))
    assert r.cached and getattr(r, "cache")["cache_tier"] == "semantic"
    r2 = await client.complete(_req("What does clause 4.2 of ISO 2768 require for a 120 mm dimension?"))
    assert not r2.cached and len(inner.calls) == 2


async def test_borderline_semantic_hit_is_verified_by_llm_and_rejected():
    # the scripted inner client answers the first question, then returns a "not equivalent" verdict
    inner = ScriptedLLM(["answer one", {"equivalent": False, "confidence": 0.9, "reason": "differs"}, "answer two"])
    from caching import HashingEmbedder, cosine

    a, b = "Explain pump cavitation in simple words", "Explain pump cavitation in simple terms"
    sim = cosine(*HashingEmbedder().embed([a, b]))
    thr = round(sim - 0.05, 3)  # the paraphrase lands inside the borderline band above the threshold
    cfg = CachingConfig(semantic_threshold=thr, strict_semantic_threshold=thr, borderline_band=0.2)
    client = CachedLLMClient(inner, config=cfg, clock=FakeClock())
    await client.complete(_req(a))
    r = await client.complete(_req(b))
    assert not r.cached and r.text == "answer two"
    assert client.metrics.semantic_rejected_by_verifier == 1
    verify_req = inner.calls[1]
    assert verify_req.output_schema is not None and verify_req.effort == "low" and verify_req.metadata.get("no_cache")


async def test_time_sensitive_prompt_uses_short_ttl_and_skips_semantic():
    clock = FakeClock()
    cfg = CachingConfig()
    client = CachedLLMClient(MockLLM(), config=cfg, clock=clock)
    await client.complete(_req("What is the current status of the build right now?"))
    assert client.last_plan is not None and client.last_plan.ttl_s == cfg.time_sensitive_ttl_s
    assert "semantic" not in client.last_plan.tiers
    clock.t += cfg.time_sensitive_ttl_s + 1
    r = await client.complete(_req("What is the current status of the build right now?"))
    assert not r.cached


async def test_non_cacheable_prompts_bypass_unless_deterministic():
    inner = MockLLM()
    client = CachedLLMClient(inner, clock=FakeClock())
    await client.complete(_req("Brainstorm five slogans"))
    await client.complete(_req("Brainstorm five slogans"))
    assert client.metrics.bypassed == 2 and len(inner.calls) == 2
    await client.complete(_req("Brainstorm five slogans", deterministic=True))
    r = await client.complete(_req("Brainstorm five slogans", deterministic=True))
    assert r.cached  # Constraints.deterministic -> prefer the cache even for diversity tasks


async def test_no_cache_metadata_bypasses():
    client = CachedLLMClient(MockLLM(), clock=FakeClock())
    await client.complete(_req("Define FoS", no_cache=True))
    r = await client.complete(_req("Define FoS", no_cache=True))
    assert not r.cached and client.metrics.bypassed == 2


async def test_single_flight_coalesces_identical_concurrent_requests():
    inner = MockLLM(latency_s=0.02)
    client = CachedLLMClient(inner, clock=FakeClock())
    reqs = [_req("Define factor of safety") for _ in range(4)]
    out = await asyncio.gather(*(client.complete(r) for r in reqs))
    assert len(inner.calls) == 1
    assert sum(1 for r in out if r.cached) == 3 and client.metrics.coalesced == 3


async def test_negative_caching_of_structured_output_failures():
    def responder(req: LLMRequest):
        return "this is not json {"

    inner = MockLLM(responder)
    client = CachedLLMClient(inner, clock=FakeClock())
    req = LLMRequest.simple("Give me JSON", system=["rules"], schema={"type": "object", "properties": {"a": {"type": "string"}}})
    with pytest.raises(StructuredOutputError):
        await client.complete(req)
    with pytest.raises(StructuredOutputError) as exc:
        await client.complete(req)
    assert "negative cache" in str(exc.value) and len(inner.calls) == 1
    assert client.metrics.negative_stores == 1 and client.metrics.negative_hits == 1


async def test_invalidate_by_document_tag_through_client():
    client = CachedLLMClient(MockLLM(), clock=FakeClock())
    await client.complete(_req("What does doc d1 say about derating?", document_ids=["d1"]))
    assert client.invalidate(tags=["doc:d1"]) >= 1
    r = await client.complete(_req("What does doc d1 say about derating?", document_ids=["d1"]))
    assert not r.cached


async def test_max_staleness_rejects_old_hits():
    clock = FakeClock()
    client = CachedLLMClient(MockLLM(), config=CachingConfig(max_staleness_s=100), clock=clock)
    await client.complete(_req("Define FoS"))
    clock.t += 200
    r = await client.complete(_req("Define FoS"))
    assert not r.cached and client.metrics.stale_rejections == 1


async def test_xfetch_early_refresh_recomputes_before_expiry():
    clock = FakeClock()
    inner = MockLLM()
    # rand=1e-12 -> ln(rand) very negative -> refresh triggers when an entry has recompute cost and is near expiry
    client = CachedLLMClient(inner, config=CachingConfig(default_ttl_s=100), clock=clock, rand=lambda: 1e-12)
    await client.complete(_req("Define FoS"))
    entry = client.backend.get_entry(client.key_for(_req("Define FoS")), touch=False)
    assert entry is not None
    entry.recompute_cost_s = 5.0
    client.backend._store(entry)
    clock.t += 90
    r = await client.complete(_req("Define FoS"))
    assert not r.cached and client.metrics.early_refreshes == 1 and len(inner.calls) == 2


# ------------------------------------------------------------------ direct API / protocols


async def test_model_wrap_shares_backend_and_cache_protocol():
    model = CachingModel(MockLLM(), clock=FakeClock())
    client = model.wrap(MockLLM())
    assert isinstance(client, CachedLLMClient) and client.backend is model.backend
    await client.complete(_req("Define FoS"))
    assert len(model.backend) == 1
    assert isinstance(model.cache, ResponseCache) and isinstance(model, ResponseCache)
    await model.set("k", {"v": 1}, tags=["doc:x"])
    assert await model.get("k") == {"v": 1}
    assert await model.invalidate(tags=["doc:x"]) == 1 and await model.get("k") is None
    rep = model.savings_report()
    assert rep["clients"] == 1 and rep["misses"] == 1


def test_model_direct_helpers():
    model = CachingModel(MockLLM())
    plan = model.plan_for("What does ISO 2768 require?", domain="mechanical_engineering")
    assert plan.cacheable and plan.namespace.endswith(":mechanical_engineering")
    req = _req("What does ISO 2768 require?")
    assert model.key_for(req) == model.key_for(req) and len(model.key_for(req)) == 64
    rep = model.analyze_prefix(req, model="claude-opus-5-5")
    assert rep.model == "claude-opus-5-5" and not rep.meets_minimum
    fp = model.fingerprint([Evidence(id="d1", content="x")])
    assert fp.document_ids == ["d1"]
    model.backend.set("k", 1, tags=["doc:d1"])
    assert model.invalidate_documents(["d1"]).removed_exact == 1
    assert model.bump_version("ns") == 2
    d = model.describe()
    assert d["name"] == "caching" and d["config"]["eviction_policy"] == "lru"


# ------------------------------------------------------------------ run(): status, confidence, domain, budget


async def test_run_grounded_prompt_includes_dependency_fingerprint_and_tags():
    model = CachingModel(MockLLM())
    ctx = [Evidence(id="spec-1", content="Clause 4.2: general tolerances ..."), Evidence(id="spec-2", content="Clause 5: ...")]
    result = await model.run(TaskRequest(task="What does clause 4.2 require for a 120 mm dimension?", context=ctx, domain="mechanical_engineering", inputs={"corpus_version": "snap-3", "expected_reuses": 10}))
    rep: CacheReport = result.output
    assert rep.dependency_fingerprint is not None and rep.dependency_fingerprint.document_ids == ["spec-1", "spec-2"]
    assert "doc:spec-1" in rep.tags and "snap-3" in rep.namespace
    assert rep.savings.expected_reuses == 10 and rep.savings.exact_tier_saved_usd > 0
    assert result.evidence == ctx


async def test_run_stores_provided_response_and_reports_existing_entry():
    model = CachingModel(MockLLM(), clock=FakeClock())
    task = "What does clause 4.2 require for a 120 mm dimension?"
    r1 = await model.run(TaskRequest(task=task, inputs={"response": {"text": "2 mm"}}))
    assert r1.output.stored and not r1.output.cached_entry_exists
    r2 = await model.run(TaskRequest(task=task))
    assert r2.output.cached_entry_exists and r2.output.exact_key == r1.output.exact_key
    stored = await model.get(r1.output.exact_key)
    assert stored["text"] == "2 mm" and LLMResponse.model_validate(stored).text == "2 mm"  # stored as an LLMResponse


async def test_domain_adaptation_changes_run_output():
    model = CachingModel(MockLLM())
    task = "Explain how a centrifugal pump works"
    general = await model.run(TaskRequest(task=task, domain="general"))
    safety = await model.run(TaskRequest(task=task, domain="chemical_process_engineering"))
    assert "semantic" in general.output.plan.tiers and "semantic" not in safety.output.plan.tiers
    assert safety.output.plan.semantic_threshold > general.output.plan.semantic_threshold
    assert safety.metadata["confidence_threshold"] == 0.85 and general.metadata["confidence_threshold"] == 0.5
    # the prefix is audited on the domain prompt: the safety-critical request carries ENGINEERING_RIGOR, the general one does not
    from pipeline_core import ENGINEERING_RIGOR

    safety_req = model._representative_request(TaskRequest(task=task, domain="chemical_process_engineering"), safety.output.plan, get_profile("chemical_process_engineering"), "claude-opus-5-5")
    general_req = model._representative_request(TaskRequest(task=task), general.output.plan, get_profile("general"), "claude-opus-5-5")
    assert ENGINEERING_RIGOR in safety_req.system_stable and ENGINEERING_RIGOR not in general_req.system_stable
    assert safety.output.prefix.prefix_tokens > general.output.prefix.prefix_tokens  # rigor block + richer profile text


async def test_needs_review_when_confidence_below_profile_threshold():
    model = CachingModel(MockLLM())
    # conflicting signals (lookup + time-sensitive + personal) lower planner confidence below the 0.85 safety-critical floor
    result = await model.run(TaskRequest(task="What is the current load rating for my project today according to clause 7?", domain="civil_structural_engineering"))
    assert result.status == Status.NEEDS_REVIEW
    assert result.confidence < 0.85 and any("below required" in w for w in result.warnings)
    ok = await model.run(TaskRequest(task="What is the current load rating for my project today according to clause 7?", domain="general"))
    assert ok.status == Status.SUCCESS  # same prompt is acceptable for a low-risk profile


async def test_needs_review_when_request_min_confidence_is_higher():
    model = CachingModel(MockLLM())
    result = await model.run(TaskRequest(task="Explain how a centrifugal pump works", constraints=Constraints(min_confidence=0.99)))
    assert result.status == Status.NEEDS_REVIEW


async def test_volatile_prefix_lowers_confidence_and_warns():
    model = CachingModel(MockLLM())
    clean = await model.run(TaskRequest(task="Explain how a centrifugal pump works"))
    dirty = await model.run(TaskRequest(task="Explain how a centrifugal pump works", inputs={"system_prompt": "Report generated at 2026-01-02T03:04:05Z for request_id: abc"}))
    assert dirty.confidence < clean.confidence
    assert dirty.output.prefix.invalidators and any("volatile content" in w for w in dirty.warnings)


async def test_budget_zero_llm_calls_forces_heuristic_plan():
    llm = ScriptedLLM([{"cacheable": True}])
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="Explain how a centrifugal pump works", constraints=Constraints(max_llm_calls=0)))
    assert result.status == Status.SUCCESS and result.usage.llm_calls == 0 and result.output.plan_source == "heuristic"


async def test_budget_cost_exceeded_is_contained():
    llm = ScriptedLLM([{"cacheable": True}], simulate_cost_as="claude-opus-5-5")
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="Explain how a centrifugal pump works", constraints=Constraints(max_cost_usd=0.0000001)))
    assert result.status in (Status.PARTIAL, Status.FAILED)
    assert result.errors and "BudgetExceededError" in result.errors[0]


async def test_wrapped_client_inside_component_counts_cache_hits_in_usage():
    """A component using the wrapped client makes no extra LLM call on a repeat; the hit is counted by the client.

    ``TaskResult.usage.cache_hits`` stays 0 because of the core ``_max_usage`` quirk
    (README limitations), so the hit is asserted on the client's own counters.
    """
    model = CachingModel(MockLLM())
    inner = MockLLM(simulate_cost_as="claude-opus-5-5")
    shared = model.wrap(inner, profile=get_profile("mechanical_engineering"))
    from pipeline_core import ComponentSpec, PipelineComponent, RunContext, TaskResult

    class Consumer(PipelineComponent):
        spec = ComponentSpec(name="consumer_test", description="t")

        async def plan(self, request, ctx):
            return None

        async def execute(self, request, plan, ctx) -> TaskResult:
            r = await ctx.llm_call(_req(request.task))
            return self.new_result(request, output_text=r.text, confidence=1.0, confidence_basis="t")

    c = Consumer(shared)
    a = await c.run("Define factor of safety")
    b = await c.run("Define factor of safety")
    assert a.usage.llm_calls == 1 and a.usage.cost_usd > 0
    assert b.usage.llm_calls == 0 and b.usage.cost_usd == 0.0 and b.output_text == a.output_text
    assert len(inner.calls) == 1
    assert model.metrics().exact_hits == 1 and shared.total_usage.cache_hits == 1
    assert shared.total_usage.llm_calls == 1 and inner.total_usage.llm_calls == 1


def test_cli_offline_text_and_json(capsys, monkeypatch):
    from caching.cli import main

    monkeypatch.setenv("PIPELINE_OFFLINE", "0")  # the CLI sets it to "1"; monkeypatch restores it afterwards

    assert main(["What does ISO 2768 require for a 120 mm dimension?", "--domain", "mechanical_engineering", "--offline", "--input", "expected_reuses=8"]) == 0
    out = capsys.readouterr().out
    assert "[caching] status=success" in out and "Cache plan for" in out and '"cacheable": true' in out
    assert main(["Brainstorm five slogans", "--offline", "--json"]) == 0
    import json

    data = json.loads(capsys.readouterr().out)
    assert data["metadata"]["plan"]["cacheable"] is False and data["component"] == "caching"
