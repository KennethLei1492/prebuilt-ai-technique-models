"""Semantic tier, prefix advisor, invalidation and evaluation metrics (no LLM)."""
from __future__ import annotations

from pipeline_core import Evidence, LLMRequest, PromptTemplate, Usage

from caching import (
    DependencyFingerprint,
    HashingEmbedder,
    InMemoryBackend,
    InvalidationManager,
    PrefixCacheAdvisor,
    SemanticCache,
    VersionRegistry,
    belady_opt_hits,
    breakeven_uses,
    compare_policies,
    cosine,
    false_hit_audit,
    find_invalidators,
    find_unsorted_json,
    prefix_hit_ratio,
    prefix_savings,
    recommend_ttl_from_gaps,
    run_offline_evaluation,
)


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------ semantic


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dimension=64)
    v1, v2 = e.embed(["torque of fastener ABC", "torque of fastener ABC"])
    assert v1 == v2 and abs(cosine(v1, v2) - 1.0) < 1e-9
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6
    assert cosine(e.embed_one("pump impeller cavitation"), e.embed_one("tax return deadline")) < 0.3


def test_semantic_cache_hits_paraphrase_and_respects_guards():
    sc = SemanticCache(threshold=0.6, clock=FakeClock())
    sc.add("What does clause 4.2 of ISO 2768 require for a 12 mm dimension?", {"answer": "x"}, namespace="ns")
    hit = sc.lookup("What does clause 4.2 of ISO 2768 require for a 12 mm dimension please?", namespace="ns")
    assert hit is not None and hit.similarity >= 0.6
    # same wording, different number -> guards reject regardless of similarity
    assert sc.lookup("What does clause 4.2 of ISO 2768 require for a 120 mm dimension?", namespace="ns") is None
    assert sc.metrics.guard_rejections >= 1
    # other namespace never matches
    assert sc.lookup("What does clause 4.2 of ISO 2768 require for a 12 mm dimension?", namespace="other") is None


def test_semantic_cache_context_chain_separates_follow_ups():
    sc = SemanticCache(threshold=0.6, clock=FakeClock())
    sc.add("and for the next clause?", "A", namespace="ns", context_key="ctx-a")
    assert sc.lookup("and for the next clause?", namespace="ns", context_key="ctx-b") is None
    assert sc.lookup("and for the next clause?", namespace="ns", context_key="ctx-a") is not None
    assert sc.metrics.context_rejections == 1


def test_semantic_cache_adaptive_threshold_feedback():
    sc = SemanticCache(threshold=0.9, error_budget=0.02, clock=FakeClock())
    e = sc.add("explain cavitation", "A", namespace="ns")
    assert e.threshold == 0.9
    new = sc.record_feedback(e.key, similarity=0.93, correct=False)
    assert new is not None and new > 0.93  # raised above the offending similarity
    for _ in range(5):
        sc.record_feedback(e.key, similarity=0.99, correct=True)
    assert e.error_rate() < 0.2
    assert sc.record_feedback("missing", similarity=0.5, correct=True) is None


def test_semantic_cache_ttl_expiry():
    clock = FakeClock()
    sc = SemanticCache(threshold=0.99, clock=clock)
    sc.add("a question about pumps", 1, namespace="ns", ttl_s=10)
    assert sc.lookup("a question about pumps", namespace="ns") is not None
    clock.t = 20
    assert sc.lookup("a question about pumps", namespace="ns") is None and len(sc) == 0


def test_semantic_cache_evicts_least_used_oldest_first():
    clock = FakeClock()
    sc = SemanticCache(threshold=0.99, max_entries=2, clock=clock)
    sc.add("a question about pumps", 1, namespace="ns")
    clock.t = 1
    sc.add("a question about valves", 2, namespace="ns")
    assert sc.lookup("a question about valves", namespace="ns") is not None  # valves now has a hit
    clock.t = 2
    sc.add("a question about seals", 3, namespace="ns")
    assert len(sc) == 2 and sc.metrics.evictions == 1
    assert sc.lookup("a question about pumps", namespace="ns") is None  # zero hits, oldest -> evicted
    assert sc.lookup("a question about valves", namespace="ns") is not None


def test_semantic_invalidate_by_tag_and_exact_key():
    sc = SemanticCache(clock=FakeClock())
    sc.add("q1", 1, namespace="ns", tags=["doc:a"], exact_key="ek1")
    sc.add("q2", 2, namespace="ns", tags=["doc:b"])
    assert sc.invalidate(tags=["doc:a"]) == 1
    assert sc.invalidate(keys=["ek1"]) == 0
    assert sc.invalidate(tags=["doc:b"]) == 1 and len(sc) == 0


# ------------------------------------------------------------------ prefix advisor


def test_breakeven_matches_documented_multipliers():
    assert breakeven_uses(1.25, 0.1) == 2
    assert breakeven_uses(2.0, 0.1) == 3


def test_ttl_recommendation_from_gaps():
    assert recommend_ttl_from_gaps([]) == "5m"
    assert recommend_ttl_from_gaps([60, 120, 90]) == "5m"
    assert recommend_ttl_from_gaps([600, 700, 650]) == "1h"
    # a 40-minute cadence is bridged by the 1h TTL (reads refresh it): 1h, not keep-alive
    assert recommend_ttl_from_gaps([2400, 2500]) == "1h"
    assert recommend_ttl_from_gaps([1800]) == "5m"  # only two uses observed: not enough to pay for the 2x write
    assert recommend_ttl_from_gaps([7200, 8000]) == "keep-alive"  # two pings at 0.1x beat an uncached call
    assert recommend_ttl_from_gaps([40000, 40000]) == "none"  # ~11 h gaps: keep-alive reads cost more than a miss


def test_find_invalidators_flags_timestamps_dates_uuids_but_not_constant_json():
    blocks = [
        "Rules. Generated at 2026-01-02T03:04:05Z.",
        'Config: {"zeta": 1, "alpha": 2}',
        "Today is 2026-01-02",
        "stable text only",
    ]
    inv = find_invalidators(blocks)
    kinds = {i.kind for i in inv}
    assert {"iso_timestamp", "date"} <= kinds
    assert "unsorted_json" not in kinds and all(i.block_index not in (1, 3) for i in inv)
    assert find_unsorted_json(blocks) == [(1, '{"zeta": 1, "alpha": 2}')]
    # the package's own few-shot JSON examples are byte-constant and must not be reported as invalidators
    from caching import CACHE_PLAN_TEMPLATE

    rendered = CACHE_PLAN_TEMPLATE.render(task="x", features="{}", model="m", has_stable_prefix="true", fan_out="false")
    assert find_invalidators(rendered.system_blocks) == []
    long_req = LLMRequest.simple("q", system=["Standard clause text. " * 400 + ' {"zeta": 1, "alpha": 2}'])
    rep = PrefixCacheAdvisor().analyze(long_req, model="claude-opus-5-5")
    assert rep.cache_friendly and not rep.invalidators and any("sort_keys" in r for r in rep.recommendations)
    assert find_invalidators(["clean"], tools=[{"name": "t", "description": "id 123e4567-e89b-12d3-a456-426614174000"}])[0].block_index == -1


def test_prefix_report_minimum_and_recommendations():
    adv = PrefixCacheAdvisor(clock=FakeClock())
    short = LLMRequest.simple("question", system=["short rules"])
    rep = adv.analyze(short, model="claude-opus-5-5")
    assert not rep.meets_minimum and rep.recommended_ttl == "none" and rep.score < 0.8
    assert any("below the 512-token minimum" in r for r in rep.recommendations)
    long_block = "Standard clause text. " * 400
    rep2 = adv.analyze(LLMRequest.simple("question", system=[long_block]), model="claude-opus-5-5")
    assert rep2.meets_minimum and rep2.cache_friendly and rep2.breakeven_uses == 2 and rep2.write_cost_usd > 0
    rep3 = adv.analyze(LLMRequest.simple("question", system=[long_block + " Generated at 2026-01-02T03:04:05Z"]), model="claude-opus-5-5")
    assert not rep3.cache_friendly and rep3.invalidators


def test_prefix_advisor_observes_arrivals_and_recommends_1h():
    clock = FakeClock()
    adv = PrefixCacheAdvisor(clock=clock)
    req = LLMRequest.simple("q", system=["Standard clause text. " * 400])
    for t in (0, 600, 1200, 1800):
        clock.t = t
        adv.observe(req, model="claude-opus-5-5")
    stats = adv.arrival_stats(req, model="claude-opus-5-5")
    assert stats.arrivals == 4 and stats.median_gap_s == 600 and stats.recommended_ttl == "1h"
    assert adv.analyze(req, model="claude-opus-5-5").recommended_ttl == "1h"


def test_prefix_template_analysis_and_hit_metrics():
    tpl = PromptTemplate(name="t", system="Rules " * 600, instructions="<task>{task}</task>")
    rep = PrefixCacheAdvisor().analyze_template(tpl, model="claude-opus-5-5")
    assert rep.meets_minimum and rep.stable_blocks == 1
    u = Usage(input_tokens=100, cache_read_input_tokens=900)
    assert abs(prefix_hit_ratio(u) - 0.9) < 1e-9 and PrefixCacheAdvisor.verify_hit(u)
    assert not PrefixCacheAdvisor.verify_hit(Usage(input_tokens=10))


async def test_warm_then_fan_out_serialises_first_call():
    order: list[str] = []

    async def call(req: LLMRequest):
        order.append(req.last_user_text())
        from pipeline_core import LLMResponse

        return LLMResponse(text=req.last_user_text())

    reqs = [LLMRequest.simple(f"q{i}") for i in range(4)]
    out = await PrefixCacheAdvisor().warm_then_fan_out(call, reqs, concurrency=2)
    assert [r.text for r in out] == ["q0", "q1", "q2", "q3"] and order[0] == "q0"


# ------------------------------------------------------------------ invalidation


def test_dependency_fingerprint_changes_with_content():
    ev = [Evidence(id="d1", content="alpha"), Evidence(id="d2", content="beta")]
    fp1 = DependencyFingerprint.from_evidence(ev)
    fp2 = DependencyFingerprint.from_evidence([Evidence(id="d1", content="alpha"), Evidence(id="d2", content="beta v2")])
    assert fp1.fingerprint != fp2.fingerprint and fp2.changed_since(fp1) == ["d2"]
    assert fp1.tags() == ["doc:d1", "doc:d2"]
    assert DependencyFingerprint.from_evidence(ev, corpus_version="v2").fingerprint != fp1.fingerprint


def test_invalidation_manager_coordinates_tiers_and_logs_causes():
    clock = FakeClock()
    backend = InMemoryBackend(clock=clock)
    sem = SemanticCache(clock=clock)
    mgr = InvalidationManager(backend, sem, clock=clock)
    backend.set("k1", 1, tags=["doc:d1", "model:m"], namespace="ns")
    sem.add("what is d1", 1, namespace="ns", tags=["doc:d1"])
    ev = mgr.invalidate_documents(["d1"])
    assert ev.removed_exact == 1 and ev.removed_semantic == 1 and ev.cause == "document_updated"
    backend.set("k2", 2, tags=["model:m"])
    assert mgr.invalidate_model("m").removed_exact == 1
    assert mgr.causes() == {"document_updated": 2, "model_changed": 1}
    before = DependencyFingerprint.from_documents({"d1": "a"})
    after = DependencyFingerprint.from_documents({"d1": "b"})
    assert mgr.on_dependency_change(before, after) is not None
    assert mgr.on_dependency_change(after, after) is None


def test_version_registry_bump():
    vr = VersionRegistry()
    assert vr.current("ns") == 1 and vr.is_current("ns", 1)
    assert vr.bump("ns") == 2 and not vr.is_current("ns", 1)
    sem = SemanticCache(clock=FakeClock())
    sem.add("q", 1, namespace="ns")
    mgr = InvalidationManager(InMemoryBackend(), sem, versions=vr)
    assert mgr.bump_version("ns") == 3 and len(sem) == 0


# ------------------------------------------------------------------ evaluation


def test_belady_opt_is_an_upper_bound_on_every_policy():
    trace = ["a", "b", "c", "a", "b", "d", "a", "b", "c", "d", "a", "e", "a", "b"]
    rates = compare_policies(trace, capacity=2)
    assert rates["opt"] >= max(rates["lru"], rates["lfu"], rates["fifo"])
    assert belady_opt_hits(["a", "b", "a", "b"], 1) == 0 and belady_opt_hits(["a", "a", "a"], 1) == 2
    assert belady_opt_hits(["a"], 0) == 0


def test_false_hit_audit_guards_remove_identifier_false_hits():
    pairs = [
        ("What does clause 4.2 require for 5 mm?", "What does clause 4.2 require for 5 mm please?", True),
        ("What does clause 4.2 require for 5 mm?", "What does clause 4.2 require for 50 mm?", False),
        ("Torque for ABC-123?", "Torque for ABC-124?", False),
    ]
    guarded = false_hit_audit(pairs, threshold=0.6, guards=True)
    unguarded = false_hit_audit(pairs, threshold=0.6, guards=False)
    assert guarded["false_hit_rate"] == 0.0 and guarded["precision"] == 1.0
    assert unguarded["false_hit_rate"] > 0.0


def test_prefix_savings_and_offline_evaluation_report():
    ps = prefix_savings("claude-opus-5-5", 1000, uses=10, ttl="5m")
    assert ps["saved_usd"] > 0 and 70 < ps["saved_pct"] < 90
    assert prefix_savings("claude-opus-5-5", 1000, uses=1, ttl="1h")["saved_usd"] < 0  # single use loses money
    rep = run_offline_evaluation(trace_length=200, capacity=16)
    assert rep.belady_opt_hit_rate >= max(rep.hit_rate_by_policy.values())
    assert rep.semantic_audit["false_hit_rate"] == 0.0
    assert rep.cost_saved_usd_per_1000_hits > 0
