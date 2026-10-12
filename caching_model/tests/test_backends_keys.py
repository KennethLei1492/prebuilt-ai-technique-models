"""Pure-logic tests: backends (eviction, TTL, tags), keys, normalisation, guards, stampede helpers."""
from __future__ import annotations

import asyncio
import math

import pytest

from pipeline_core import CacheError, LLMRequest, Message

from caching import (
    InMemoryBackend,
    KeyNormalization,
    SQLiteBackend,
    SingleFlight,
    context_chain_key,
    exact_key,
    extract_guards,
    make_backend,
    namespace_for,
    normalize_prompt_text,
    strip_volatile,
    tagged_keys,
    xfetch_should_refresh,
)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ------------------------------------------------------------------ backends


def test_lru_eviction_by_entry_count():
    b = InMemoryBackend(max_entries=2, policy="lru", clock=FakeClock())
    b.set("a", 1)
    b.set("b", 2)
    assert b.get("a") == 1  # touch a -> b is now least recently used
    b.set("c", 3)
    assert b.get("b") is None and b.get("a") == 1 and b.get("c") == 3
    assert b.stats.evictions == 1


def test_lfu_and_fifo_victim_selection():
    clock = FakeClock()
    lfu = InMemoryBackend(max_entries=2, policy="lfu", clock=clock)
    lfu.set("a", 1)
    lfu.set("b", 2)
    lfu.get("a")
    lfu.get("a")
    lfu.set("c", 3)
    assert lfu.get("b") is None and lfu.get("a") == 1
    fifo = InMemoryBackend(max_entries=2, policy="fifo", clock=clock)
    fifo.set("a", 1)
    clock.advance(1)
    fifo.set("b", 2)
    fifo.get("a")  # recency does not matter for FIFO
    clock.advance(1)
    fifo.set("c", 3)
    assert fifo.get("a") is None and fifo.get("b") == 2


def test_size_aware_eviction_in_bytes():
    b = InMemoryBackend(max_entries=100, max_bytes=70, clock=FakeClock())
    b.set("a", "x" * 30)
    b.set("b", "y" * 30)
    assert len(b) == 2
    b.set("c", "z" * 30)  # pushes bytes over the limit -> oldest evicted
    assert len(b) <= 2 and b.bytes_used <= 70 and b.get("a") is None


def test_ttl_expiry_with_fake_clock():
    clock = FakeClock()
    b = InMemoryBackend(clock=clock)
    b.set("k", "v", ttl_s=10)
    clock.advance(9.9)
    assert b.get("k") == "v"
    clock.advance(0.2)
    assert b.get("k") is None and b.stats.expirations == 1


def test_tag_invalidation_and_purge():
    clock = FakeClock()
    b = InMemoryBackend(clock=clock)
    b.set("k1", 1, tags=["doc:a", "model:m"])
    b.set("k2", 2, tags=["doc:b"])
    b.set("k3", 3, tags=["doc:a"], ttl_s=1)
    assert b.invalidate(tags=["doc:a"]) == 2
    assert b.get("k2") == 2 and len(b) == 1
    clock.advance(5)
    assert b.purge_expired() == 0


def test_sqlite_backend_roundtrip_and_tags(tmp_path):
    clock = FakeClock()
    db = SQLiteBackend(tmp_path / "cache.sqlite", max_entries=2, clock=clock)
    db.set("a", {"text": "hello"}, tags=["doc:1"], ttl_s=100, namespace="ns")
    clock.advance(1)
    db.set("b", 2, ttl_s=100)
    clock.advance(1)
    e = db.get_entry("a")  # touches "a": "b" is now the least recently used
    assert e is not None and e.value == {"text": "hello"} and e.hits == 1 and e.namespace == "ns"
    clock.advance(1)
    db.set("c", 3, ttl_s=100)  # LRU by last access evicts "b"
    assert sorted(db.keys()) == ["a", "c"] and db.stats.evictions == 1
    assert db.invalidate(tags=["doc:1"]) == 1 and db.get("a") is None
    clock.advance(1000)
    assert db.get("c") is None  # TTL expiry
    db.close()


def test_make_backend_variants(tmp_path):
    assert make_backend("memory").name == "memory"
    assert make_backend("sqlite").name == "sqlite"
    assert make_backend("file", file_dir=str(tmp_path / "files")).name == "file"
    with pytest.raises(CacheError):
        make_backend("unknown")


def test_file_backend_persists_across_instances(tmp_path):
    from caching import FileBackend

    d = tmp_path / "fb"
    a = FileBackend(d)
    a.set("k", {"v": 1}, tags=["doc:x"])
    b = FileBackend(d)
    assert b.get("k") == {"v": 1}
    assert b.invalidate(tags=["doc:x"]) == 1
    assert b.get("k") is None


# ------------------------------------------------------------------ keys


def test_normalisation_preserves_code_blocks_and_keeps_bare_volatile_values():
    text = "Request id: abc-123  at 2026-03-01T10:00:00Z\n\n```python\nX  =   1\n```"
    out = normalize_prompt_text(text)
    assert "```python\nX  =   1\n```" in out  # byte-identical code block
    assert "abc-123" not in out  # labelled tracing id stripped by default
    assert "2026-03-01T10:00:00Z" in out  # a bare timestamp is part of the question: kept by default
    opt_in = normalize_prompt_text(text, KeyNormalization(strip_volatile=True))
    assert "2026-03-01T10:00:00Z" not in opt_in and "<volatile>" in opt_in


def test_exact_key_differs_for_prompts_differing_only_in_volatile_looking_values():
    """Regression (critical): the zero-false-hit exact tier must not merge different time windows or object ids."""
    pairs = [
        ("List incidents between 2026-03-01T10:00Z and 2026-03-01T12:00Z", "List incidents between 2026-03-02T10:00Z and 2026-03-02T12:00Z"),
        ("Schedule the pump test at 10:30 AM", "Schedule the pump test at 11:30 PM"),
        ("What changed in commit 0123456789abcdef0123456789abcdef?", "What changed in commit fedcba9876543210fedcba9876543210?"),
        ("Status of job 123e4567-e89b-12d3-a456-426614174000", "Status of job 223e4567-e89b-12d3-a456-426614174000"),
    ]
    for a, b in pairs:
        assert exact_key(LLMRequest.simple(a, system=["rules"]), "m") != exact_key(LLMRequest.simple(b, system=["rules"]), "m"), (a, b)
    # labelled tracing ids never change the answer and still normalise away
    assert exact_key(LLMRequest.simple("Define FoS. request_id: r-1"), "m") == exact_key(LLMRequest.simple("Define FoS. request_id: r-2"), "m")


def test_lowercase_is_opt_in_and_never_touches_code():
    opts = KeyNormalization(lowercase=True)
    assert normalize_prompt_text("Part ABC-1", opts) == "part abc-1"
    assert normalize_prompt_text("```\nABC\n```", opts) == "```\nABC\n```"


def test_exact_key_is_stable_under_whitespace_but_sensitive_to_determinants():
    r1 = LLMRequest.simple("What is   the FoS?", system=["rules"])
    r2 = LLMRequest.simple("What is the FoS?", system=["rules"])
    assert exact_key(r1, "m") == exact_key(r2, "m")
    assert exact_key(r1, "m") != exact_key(r1, "other-model")
    assert exact_key(r1, "m", "1.0") != exact_key(r1, "m", "1.1")
    assert exact_key(r1, "m", tenant_salt="acme") != exact_key(r1, "m", tenant_salt="globex")
    assert exact_key(r1, "m", extra_keys={"corpus_version": "a"}) != exact_key(r1, "m", extra_keys={"corpus_version": "b"})
    r3 = r1.model_copy(update={"max_tokens": 42})
    assert exact_key(r1, "m") != exact_key(r3, "m")


def test_namespace_and_tags():
    ns = namespace_for("claude-opus-5-5", prompt_version="1.0.0", corpus_version="s1", tenant="t")
    assert ns == "claude-opus-5-5:1.0.0:-:-:s1:-:t"
    tags = tagged_keys(model="m", prompt_template="caching.plan", prompt_version="1.0.0", document_ids=["d1", "d2"], namespace="ns", extra=["x"])
    assert tags == ["model:m", "template:caching.plan@1.0.0", "doc:d1", "doc:d2", "ns:ns", "x"]


def test_extract_guards_blurs_nothing_important():
    assert extract_guards("fastener ABC-1234 at 12 Nm") >= {"ABC-1234", "12nm", "12"}
    assert extract_guards("clause 4.2 of ISO 2768 for 5 mm") != extract_guards("clause 4.2 of ISO 2768 for 50 mm")
    assert extract_guards("REQ-120 shall") != extract_guards("REQ-121 shall")
    assert extract_guards("version 1.2.3") >= {"1.2.3"}


def test_context_chain_key_depends_on_prior_turns_only():
    assert context_chain_key(["a", "b"], 3) == context_chain_key(["a", "b"], 3)
    assert context_chain_key(["a", "b"], 3) != context_chain_key(["a", "c"], 3)
    assert context_chain_key(["x", "a", "b"], 2) == context_chain_key(["y", "a", "b"], 2)
    assert context_chain_key(["a"], 0) == context_chain_key(["b"], 0)


def test_strip_volatile_patterns():
    s = strip_volatile("uuid 123e4567-e89b-12d3-a456-426614174000 time 10:30 PM trace_id: xyz")
    assert "123e4567" not in s and "10:30 PM" not in s and "xyz" not in s


# ------------------------------------------------------------------ stampede


def test_xfetch_refreshes_earlier_for_expensive_entries():
    # rand -> e^-1 makes ln(rand) = -1 so the effective expiry is pulled forward by exactly delta * beta
    rand = lambda: math.exp(-1)  # noqa: E731
    assert xfetch_should_refresh(now=95.0, expires_at=100.0, recompute_cost_s=6.0, beta=1.0, rand=rand) is True
    assert xfetch_should_refresh(now=95.0, expires_at=100.0, recompute_cost_s=4.0, beta=1.0, rand=rand) is False
    assert xfetch_should_refresh(now=95.0, expires_at=100.0, recompute_cost_s=6.0, beta=0.0, rand=rand) is False
    assert xfetch_should_refresh(now=100.0, expires_at=100.0, recompute_cost_s=0.0, rand=rand) is True
    assert xfetch_should_refresh(now=100.0, expires_at=None, recompute_cost_s=5.0, rand=rand) is False


async def test_single_flight_coalesces_concurrent_calls():
    sf = SingleFlight()
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "result"

    results = await asyncio.gather(*(sf.run("k", work) for _ in range(5)))
    assert calls == 1
    assert all(r == "result" for r, _ in results)
    assert sum(1 for _, coalesced in results if coalesced) == 4
    assert sf.leaders == 1 and sf.coalesced == 4 and not sf.inflight("k")


async def test_single_flight_propagates_errors_to_followers():
    sf = SingleFlight()

    async def boom():
        await asyncio.sleep(0.01)
        raise ValueError("x")

    results = await asyncio.gather(*(sf.run("k", boom) for _ in range(3)), return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in results)
