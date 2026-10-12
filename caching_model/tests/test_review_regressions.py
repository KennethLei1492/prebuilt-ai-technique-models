"""Regression tests for the adversarial review findings (one or more per critical/major fix, plus minors)."""
from __future__ import annotations

import asyncio
import gc

import pytest

from pipeline_core import (
    LLMError,
    LLMRefusalError,
    LLMRequest,
    MockLLM,
    ScriptedLLM,
    Status,
    TaskRequest,
    Usage,
)

from caching import (
    CachedLLMClient,
    CachingConfig,
    CachingModel,
    FileBackend,
    HashingEmbedder,
    HeuristicPlanner,
    SingleFlight,
    SQLiteBackend,
    analyze_task,
    cosine,
)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


SYSTEM = ["You answer engineering questions about standards."]


def _req(text: str, **meta) -> LLMRequest:
    return LLMRequest.simple(text, system=SYSTEM, metadata=meta)


def _borderline_config(a: str, b: str) -> CachingConfig:
    """Threshold placed just under the similarity of ``a`` and ``b`` so ``b`` is a borderline hit for ``a``."""
    sim = cosine(*HashingEmbedder().embed([a, b]))
    thr = round(sim - 0.05, 3)
    return CachingConfig(semantic_threshold=thr, strict_semantic_threshold=thr, borderline_band=0.2)


# --------------------------------------------------------------------------- critical: exact key


async def test_exact_tier_never_serves_a_different_time_window_or_object_id():
    inner = MockLLM()
    client = CachedLLMClient(inner, clock=FakeClock())
    a = await client.complete(_req("Summarise incidents between 2026-03-01T10:00Z and 2026-03-01T12:00Z"))
    b = await client.complete(_req("Summarise incidents between 2026-03-02T10:00Z and 2026-03-02T12:00Z"))
    c = await client.complete(_req("What changed in build 0123456789abcdef0123456789abcdef?"))
    d = await client.complete(_req("What changed in build fedcba9876543210fedcba9876543210?"))
    assert not any(r.cached for r in (a, b, c, d)) and len(inner.calls) == 4


# --------------------------------------------------------------------------- major: ttl_s = 0


async def test_metadata_cache_ttl_zero_means_do_not_retain_and_never_raises():
    inner = MockLLM()
    client = CachedLLMClient(inner, clock=FakeClock())
    r1 = await client.complete(_req("Define factor of safety", cache_ttl_s=0))
    r2 = await client.complete(_req("Define factor of safety", cache_ttl_s=0))
    assert r1.text and not r2.cached and len(inner.calls) == 2
    assert client.metrics.stores == 0 and len(client.backend) == 0


async def test_cache_store_and_protocol_set_with_ttl_zero_drop_instead_of_raising():
    model = CachingModel(MockLLM(), clock=FakeClock())
    await model.set("k", {"text": "a"}, ttl_s=60)
    await model.cache.set("k", {"text": "b"}, ttl_s=0)  # "do not retain" also drops the old value
    assert await model.get("k") is None
    assert model.backend.set("k2", 1, ttl_s=-5) is None and await model.get("k2") is None


async def test_llm_plan_with_ttl_zero_does_not_fail_the_run():
    llm = ScriptedLLM([{"cacheable": True, "tiers": ["exact"], "ttl_s": 0, "reason": "do not keep", "confidence": 0.9}])
    model = CachingModel(llm, planner_mode="always")
    result = await model.run(TaskRequest(task="Define factor of safety", inputs={"response": "FoS = capacity / demand"}))
    assert result.status == Status.SUCCESS and result.output.plan_source == "llm"
    assert result.output.plan.ttl_s == 0 and not result.output.stored
    assert any("do not retain" in w for w in result.warnings) and len(model.backend) == 0


# --------------------------------------------------------------------------- major: plan() fallback


@pytest.mark.parametrize("exc", [LLMRefusalError("refused"), LLMError("connection error")])
async def test_refused_or_failed_planning_call_falls_back_to_heuristic_plan(exc):
    def responder(req: LLMRequest):
        raise exc

    model = CachingModel(MockLLM(responder), planner_mode="always")
    result = await model.run(TaskRequest(task="Explain how a centrifugal pump works"))
    assert result.status == Status.SUCCESS and result.output is not None
    assert result.output.plan_source == "heuristic"
    assert any("using heuristic plan" in w for w in result.warnings)


# --------------------------------------------------------------------------- major: version bump


async def test_bump_version_invalidates_exact_tier_entries():
    model = CachingModel(MockLLM(), clock=FakeClock())
    inner = MockLLM()
    client = model.wrap(inner)
    req = _req("Define factor of safety")
    await client.complete(req)
    assert (await client.complete(req)).cached
    ev_before = len(model.invalidation.log)
    model.bump_version(client.last_plan.namespace)
    assert model.invalidation.log[ev_before].removed_exact == 1
    r = await client.complete(req)
    assert not r.cached and len(inner.calls) == 2


async def test_wrapped_client_rejects_entries_written_under_an_older_version():
    model = CachingModel(MockLLM(), clock=FakeClock())
    client = model.wrap(MockLLM())
    req = _req("Define factor of safety")
    await client.complete(req)
    model.invalidation.versions.bump(client.last_plan.namespace)  # registry bump without a purge
    r = await client.complete(req)
    assert not r.cached and client.metrics.version_rejections == 1
    assert (await client.complete(req)).cached  # the recomputed entry carries the new version


# --------------------------------------------------------------------------- major: verification usage


async def test_verified_borderline_hit_reports_the_verification_call_in_usage():
    a, b = "Explain pump cavitation in simple words", "Explain pump cavitation in simple terms"
    inner = ScriptedLLM(["answer one", {"equivalent": True, "confidence": 0.9, "reason": "same question"}], simulate_cost_as="claude-opus-5-5")
    client = CachedLLMClient(inner, config=_borderline_config(a, b), clock=FakeClock())
    await client.complete(_req(a))
    r = await client.complete(_req(b))
    assert r.cached and r.text == "answer one" and getattr(r, "cache")["verified"] is True
    assert r.usage.cache_hits == 1 and r.usage.llm_calls == 1 and r.usage.cost_usd > 0  # visible to ctx budgets / tracer
    m = client.metrics
    assert m.semantic_verified == 1 and m.verification_calls == 1 and m.verification_cost_usd == pytest.approx(r.usage.cost_usd)
    assert client.total_usage.llm_calls == 2 == len(inner.calls)


async def test_rejected_borderline_hit_counts_verification_plus_recompute():
    a, b = "Explain pump cavitation in simple words", "Explain pump cavitation in simple terms"
    inner = ScriptedLLM(["answer one", {"equivalent": False, "confidence": 0.9, "reason": "differs"}, "answer two"])
    client = CachedLLMClient(inner, config=_borderline_config(a, b), clock=FakeClock())
    await client.complete(_req(a))
    r = await client.complete(_req(b))
    assert not r.cached and r.text == "answer two" and r.usage.llm_calls == 2
    assert client.semantic is not None
    # the rejected candidate was not "served": its eviction rank is unchanged
    assert all(e.hits == 0 for bucket in client.semantic._ns.values() for e in bucket.values())


async def test_semantic_hit_count_rises_only_when_served():
    cfg = CachingConfig(semantic_threshold=0.6, strict_semantic_threshold=0.6, borderline_band=0.0)
    client = CachedLLMClient(MockLLM(), config=cfg, clock=FakeClock())
    await client.complete(_req("What does clause 4.2 of ISO 2768 require for a 12 mm dimension?"))
    r = await client.complete(_req("What does clause 4.2 of ISO 2768 require for a 12 mm dimension please?"))
    assert r.cached
    assert client.semantic is not None
    hits = [e.hits for bucket in client.semantic._ns.values() for e in bucket.values()]
    assert hits == [1]


# --------------------------------------------------------------------------- major: malformed stored values


async def test_response_stored_by_run_is_served_by_a_wrapped_client():
    model = CachingModel(MockLLM(), clock=FakeClock())
    inner = MockLLM()
    client = model.wrap(inner)
    req = _req("What tolerance applies to a 120 mm dimension under ISO 2768-m?")
    assert model.key_for(req) == client.key_for(req)  # same key rule as the client (unpriced model kept)
    result = await model.run(TaskRequest(task="store a verified answer", inputs={"llm_request": req, "response": "+/- 0.5 mm"}))
    assert result.output.stored and result.output.exact_key == client.key_for(req)
    r = await client.complete(req)
    assert r.cached and r.text == "+/- 0.5 mm" and not inner.calls


async def test_malformed_entry_under_a_client_key_is_a_miss_not_an_exception():
    model = CachingModel(MockLLM(), clock=FakeClock())
    inner = MockLLM()
    client = model.wrap(inner)
    req = _req("Define factor of safety")
    await model.cache.set(client.key_for(req), {"answer": 42})  # not an LLMResponse
    r = await client.complete(req)
    assert not r.cached and len(inner.calls) == 1 and client.metrics.corrupt_entries == 1
    assert (await client.complete(req)).cached  # the bad entry was replaced by a valid one


# --------------------------------------------------------------------------- major: per-request domain


async def test_shared_client_applies_the_request_domain_profile():
    model = CachingModel(MockLLM(), clock=FakeClock())
    shared = model.wrap(MockLLM())  # default ('general') profile, as in ARCHITECTURE section 5
    await shared.complete(_req("Explain how a centrifugal pump works"))
    assert "semantic" in shared.last_plan.tiers
    await shared.complete(_req("Explain how a centrifugal pump works", domain="civil_structural_engineering"))
    plan = shared.last_plan
    assert "semantic" not in plan.tiers and plan.namespace.endswith(":civil_structural_engineering")
    # unknown domains keep the wrap-time profile instead of silently relaxing to 'general' rules
    strict = model.wrap(MockLLM(), profile="civil_structural_engineering")
    await strict.complete(_req("Explain how a centrifugal pump works", domain="no_such_domain"))
    assert "semantic" not in strict.last_plan.tiers


# --------------------------------------------------------------------------- minors


@pytest.mark.parametrize("backend_kind", ["file", "sqlite"])
def test_retagging_an_entry_forgets_its_old_tags(tmp_path, backend_kind):
    backend = FileBackend(tmp_path / "fb") if backend_kind == "file" else SQLiteBackend(tmp_path / "c.sqlite")
    backend.set("k", {"v": 1}, tags=["doc:x"])
    backend.set("k", {"v": 2}, tags=["doc:y"])
    assert backend.invalidate(tags=["doc:x"]) == 0 and backend.get("k") == {"v": 2}
    assert backend.invalidate(tags=["doc:y"]) == 1


def test_embedding_dimension_is_honoured():
    assert CachingModel(MockLLM(), embedding_dimension=64).semantic.embedder.dimension == 64
    assert CachedLLMClient(MockLLM(), config=CachingConfig(embedding_dimension=128)).semantic.embedder.dimension == 128


def test_tenant_salt_never_appears_in_namespaces_or_reports():
    model = CachingModel(MockLLM(), tenant_salt="s3cr3t-tenant-salt")
    report = model.explain("Define factor of safety")
    assert "s3cr3t" not in report.namespace and all("s3cr3t" not in t for t in report.tags)
    assert ":salt-" in report.namespace
    other = CachingModel(MockLLM(), tenant_salt="another-salt").explain("Define factor of safety")
    assert other.namespace != report.namespace and other.exact_key != report.exact_key


async def test_run_does_not_skew_backend_hit_miss_stats():
    model = CachingModel(MockLLM(), clock=FakeClock())
    await model.run(TaskRequest(task="Define factor of safety"))
    await model.run(TaskRequest(task="Define factor of safety", inputs={"response": "x"}))
    await model.run(TaskRequest(task="Define factor of safety"))
    stats = model.backend.snapshot_stats()
    assert stats.hits == 0 and stats.misses == 0 and stats.writes == 1


async def test_single_flight_leader_failure_without_followers_logs_nothing():
    loop = asyncio.get_running_loop()
    seen: list[dict] = []
    old = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: seen.append(context))
    try:
        async def boom():
            raise ValueError("x")

        with pytest.raises(ValueError):
            await SingleFlight().run("k", boom)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(old)
    assert not any("never retrieved" in str(c.get("message", "")) for c in seen)


@pytest.mark.parametrize(
    "task",
    [
        "Explain the roll, pitch and yaw axes of an aircraft",
        "Compare the alternatives for M8 bolt grades",
        "What is a MAC address?",
        "What does HTTP status 404 mean?",
        "Tell me what ISO 2768 is",
        "Which PPE (personal protective equipment) is required for welding?",
        "Is a unique constraint enforced on the part number column?",
    ],
)
def test_ordinary_engineering_wording_is_not_misclassified(task):
    f = analyze_task(task)
    assert not f.randomness and not f.personal and not f.time_sensitive, f.matched_terms
    from pipeline_core import get_profile

    plan = HeuristicPlanner().plan_text(task, get_profile("general"))
    assert plan.cacheable and plan.task_class not in ("random", "personal", "time_sensitive")


def test_risk_phrases_still_trigger():
    assert analyze_task("Roll a die for me").randomness
    assert analyze_task("Write a short story about a bridge").randomness
    assert analyze_task("What is the status of the nightly build?").time_sensitive
    assert analyze_task("Update my email address").personal


async def test_wrapped_client_requests_1h_provider_ttl_when_the_plan_recommends_it():
    inner = MockLLM()
    client = CachedLLMClient(inner, profile="mechanical_engineering", clock=FakeClock())
    await client.complete(_req("What does clause 7 require?"))
    assert client.last_plan.prefix_ttl == "1h" and inner.calls[-1].cache_ttl == "1h"
    assert client.metrics.prefix_ttl_promotions == 1
    off = MockLLM()
    await CachedLLMClient(off, profile="mechanical_engineering", config=CachingConfig(apply_prefix_ttl=False)).complete(_req("What does clause 7 require?"))
    assert off.calls[-1].cache_ttl == "5m"


async def test_fan_out_prewarms_the_provider_prefix_once():
    class WarmableMock(MockLLM):
        def __init__(self) -> None:
            super().__init__(latency_s=0.01)
            self.warmed = 0

        async def prewarm(self, request: LLMRequest) -> Usage:
            self.warmed += 1
            await asyncio.sleep(0.01)
            return Usage(llm_calls=1, cache_creation_input_tokens=100)

    inner = WarmableMock()
    client = CachedLLMClient(inner, clock=FakeClock())
    reqs = [_req(f"Check clause {i} of the stress report", fan_out=True) for i in range(4)]
    out = await asyncio.gather(*(client.complete(r) for r in reqs))
    assert inner.warmed == 1 and client.metrics.prewarms == 1
    assert sum(r.usage.cache_creation_input_tokens for r in out) == 100  # charged once, to one caller
    await client.complete(_req("Check clause 9 of the stress report", fan_out=True))
    assert inner.warmed == 1  # still warm inside the provider TTL
