"""Tests for pipeline_core. All offline."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from pipeline_core import (
    AnthropicLLM,
    BudgetExceededError,
    ComponentSpec,
    Constraints,
    Evidence,
    LLMRequest,
    Message,
    MockLLM,
    PipelineComponent,
    PromptTemplate,
    ScriptedLLM,
    Status,
    StructuredOutputError,
    TaskRequest,
    TaskResult,
    Tracer,
    Usage,
    canonical_json,
    chunk_text,
    example_from_schema,
    get_profile,
    list_profiles,
    render_evidence,
    sanitize_schema,
    stable_hash,
)
from pipeline_core.component import RunContext
from pipeline_core.protocols import Retriever, Verdict


class Plan(BaseModel):
    strategy: str
    samples: int


class Dummy(PipelineComponent):
    spec = ComponentSpec(name="dummy", description="test", capabilities=["echo"])

    async def plan(self, request: TaskRequest, ctx: RunContext) -> Plan:
        resp = await ctx.llm_call(LLMRequest.simple("plan", schema=sanitize_schema(Plan.model_json_schema())), step="plan.llm")
        return Plan.model_validate(resp.parsed)

    async def execute(self, request: TaskRequest, plan: Plan, ctx: RunContext) -> TaskResult:
        with ctx.step("work"):
            r = await ctx.llm_call(LLMRequest.simple(request.task))
        return self.new_result(request, output={"text": r.text}, output_text=r.text, confidence=0.8, confidence_basis="mock")


# ----------------------------------------------------------------- contracts


def test_usage_is_additive():
    u = Usage(input_tokens=10, output_tokens=5, llm_calls=1, cost_usd=0.5) + Usage(input_tokens=1, cache_read_input_tokens=100, llm_calls=2)
    assert u.input_tokens == 11 and u.llm_calls == 3 and u.cost_usd == 0.5
    assert u.total_prompt_tokens == 111
    assert 0.9 < u.cache_hit_ratio < 0.91


def test_request_derive_inherits_and_links():
    parent = TaskRequest(task="parent", domain="software_engineering", constraints=Constraints(max_llm_calls=3), metadata={"k": 1})
    child = parent.derive("child", inputs={"x": 1})
    assert child.parent_id == parent.request_id
    assert child.domain == "software_engineering"
    assert child.constraints.max_llm_calls == 3
    assert child.inputs == {"x": 1}


def test_result_as_evidence_roundtrip():
    r = TaskResult(request_id="r1", component="c", output_text="answer", confidence=0.7)
    ev = r.as_evidence()
    assert ev.content == "answer" and ev.score == 0.7 and ev.retriever == "provided"


# --------------------------------------------------------------------- utils


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == canonical_json({"a": [1, {"c": 3, "d": 2}], "b": 1})
    assert stable_hash({"a": 1}) == stable_hash({"a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_chunk_text_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij", "ij"]
    assert chunk_text("", 4, 1) == []
    with pytest.raises(ValueError):
        chunk_text("x", 4, 4)


# ----------------------------------------------------------------------- llm


def test_mock_llm_structured_output_conforms_to_schema():
    schema = sanitize_schema(Plan.model_json_schema())
    ex = example_from_schema(schema)
    Plan.model_validate(ex)


def test_sanitize_schema_sets_additional_properties_false_and_strips_titles():
    s = sanitize_schema({"title": "T", "type": "object", "properties": {"a": {"type": "string", "title": "A", "default": "x"}}})
    assert s["additionalProperties"] is False
    assert "title" not in s and "title" not in s["properties"]["a"] and "default" not in s["properties"]["a"]


def test_example_from_schema_handles_refs_enums_arrays():
    schema = {
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}, "required": ["n"]}},
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "minItems": 2},
            "flag": {"anyOf": [{"type": "null"}, {"type": "boolean"}]},
        },
        "required": ["kind", "items", "flag"],
    }
    ex = example_from_schema(schema)
    assert ex == {"kind": "a", "items": [{"n": 3}, {"n": 3}], "flag": True}


@pytest.mark.asyncio
async def test_scripted_llm_returns_in_order_and_raises_when_exhausted():
    llm = ScriptedLLM(["one", {"x": 1}])
    r1 = await llm.generate("a")
    r2 = await llm.generate("b", schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    assert r1.text == "one" and r2.parsed == {"x": 1}
    with pytest.raises(Exception):
        await llm.generate("c")


@pytest.mark.asyncio
async def test_structured_parse_failure_is_typed():
    llm = MockLLM(responder=lambda req: "not json at all")
    with pytest.raises(StructuredOutputError):
        await llm.generate("x", schema={"type": "object", "properties": {}})


def test_request_fingerprints_distinguish_prefix_from_full():
    a = LLMRequest.simple("q1", system=["sys"])
    b = LLMRequest.simple("q2", system=["sys"])
    assert a.prefix_fingerprint("m") == b.prefix_fingerprint("m")
    assert a.fingerprint("m") != b.fingerprint("m")


def test_anthropic_kwargs_follow_current_api_rules():
    client = AnthropicLLM(api_key="test-key")
    req = LLMRequest.simple(
        "question",
        system=["stable rules", "domain profile"],
        schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False},
        effort="low",
    )
    req.system_volatile = "volatile"
    kw = client.build_kwargs(req)
    assert kw["model"] == "claude-opus-5-5"
    assert "temperature" not in kw and "top_p" not in kw
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    # cache breakpoint sits on the LAST stable block, volatile block has none
    blocks = kw["system"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert blocks[2]["text"] == "volatile" and "cache_control" not in blocks[2]
    assert kw["messages"] == [{"role": "user", "content": "question"}]


def test_anthropic_kwargs_1h_ttl_and_no_cache():
    client = AnthropicLLM(api_key="k", thinking=None)
    kw = client.build_kwargs(LLMRequest.simple("q", system=["s"], cache_ttl="1h"))
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "thinking" not in kw
    kw2 = client.build_kwargs(LLMRequest.simple("q", system=["s"], cache=False))
    assert "cache_control" not in kw2["system"][0]


# ----------------------------------------------------------------- component


@pytest.mark.asyncio
async def test_component_runs_plan_then_execute_with_tracing():
    comp = Dummy(MockLLM())
    res = await comp.run("do the thing", domain="mechanical_engineering")
    assert res.status is Status.SUCCESS
    assert res.usage.llm_calls == 2
    assert [s.name for s in res.steps] == ["plan", "work"]
    assert res.metadata["plan"] == {"strategy": "mock", "samples": 0}
    assert res.trace_id == comp.tracer.trace_id
    names = {s.name for s in comp.tracer.spans}
    assert {"dummy.run", "plan", "plan.llm", "work", "llm"} <= names
    assert comp.tracer.total_usage.llm_calls == 2


@pytest.mark.asyncio
async def test_component_contains_budget_errors():
    comp = Dummy(MockLLM())
    res = await comp.run("x", constraints=Constraints(max_llm_calls=1))
    assert res.status in (Status.PARTIAL, Status.FAILED)
    assert any("budget" in e.lower() for e in res.errors)


@pytest.mark.asyncio
async def test_component_raise_on_error_propagates():
    class Boom(Dummy):
        async def execute(self, request, plan, ctx):
            raise RuntimeError("boom")

    comp = Boom(MockLLM(), raise_on_error=True)
    with pytest.raises(RuntimeError):
        await comp.run("x")
    comp2 = Boom(MockLLM())
    res = await comp2.run("x")
    assert res.status is Status.FAILED and "boom" in res.errors[0]


def test_run_sync_wrapper():
    res = Dummy(MockLLM()).run_sync("sync task")
    assert res.ok


@pytest.mark.asyncio
async def test_run_context_latency_budget():
    comp = Dummy(MockLLM(latency_s=0.05))
    res = await comp.run("x", constraints=Constraints(max_latency_s=0.01))
    assert res.status in (Status.PARTIAL, Status.FAILED)


# ------------------------------------------------------------------ prompts


def test_prompt_template_render_and_fingerprint_stability():
    t = PromptTemplate(name="t", version="1.0.0", system="S", instructions="Do {x} and {y}", examples=[("in", "out")])
    r = t.render(x="A", y="B")
    assert r.user == "Do A and B"
    assert r.system_blocks[0] == "S" and "<examples>" in r.system_blocks[1]
    assert t.fingerprint() == PromptTemplate(name="t", version="1.0.0", system="S", instructions="Do {x} and {y}", examples=[("in", "out")]).fingerprint()
    with pytest.raises(KeyError):
        t.render(x="A")


def test_render_evidence_is_citable_xml():
    out = render_evidence([Evidence(id="doc-1", content="alpha", source="a.txt", score=0.5)])
    assert 'id="doc-1"' in out and "alpha" in out and out.startswith("<documents>")


# ------------------------------------------------------------------- domain


def test_domain_profiles_present_and_fallback():
    assert {"general", "software_engineering", "mechanical_engineering", "civil_structural_engineering"} <= set(list_profiles())
    p = get_profile("mechanical_engineering")
    assert "MPa" in p.units and p.risk_level == "high"
    assert get_profile("does-not-exist").name == "general"
    assert "Domain: mechanical_engineering" in p.system_text()


# ---------------------------------------------------------------- protocols


def test_protocols_are_structural():
    class R:
        async def retrieve(self, query, *, k=10, filters=None):
            return []

    assert isinstance(R(), Retriever)
    v = Verdict(method="m", passed=True, score=0.9)
    assert v.score == 0.9


def test_tracer_export_and_tree(tmp_path):
    tr = Tracer()
    with tr.span("outer", "c"):
        with tr.span("inner", "c"):
            tr.record_usage(Usage(input_tokens=5, llm_calls=1, cost_usd=0.01))
    rows = tr.export(tmp_path / "trace.jsonl")
    assert len(rows) == 2 and (tmp_path / "trace.jsonl").exists()
    assert tr.summary()["usage"]["llm_calls"] == 1
    assert "outer" in tr.tree() and "inner" in tr.tree()
    outer = next(s for s in tr.spans if s.name == "outer")
    assert outer.usage.llm_calls == 1  # child usage rolls up


def test_offline_detection_sees_through_wrapper_clients():
    from pipeline_core import is_offline_client

    class Wrapper:
        def __init__(self, inner):
            self.inner = inner

    assert is_offline_client(MockLLM())
    assert is_offline_client(Wrapper(Wrapper(MockLLM())))
    assert not is_offline_client(Wrapper(object()))
    assert not is_offline_client(AnthropicLLM(api_key="k"))


@pytest.mark.asyncio
async def test_run_context_is_offline_for_wrapped_mock():
    class Wrapped(MockLLM.__mro__[1]):  # BaseLLMClient subclass that wraps a mock, like a cache
        name = "wrapped"

        def __init__(self, inner):
            super().__init__(inner.model)
            self.inner = inner

        async def _complete(self, request):
            return await self.inner._complete(request)

    seen = {}

    class Probe(Dummy):
        async def plan(self, request, ctx):
            seen["offline"] = ctx.offline
            return Plan(strategy="h", samples=1)

    await Probe(Wrapped(MockLLM())).run("x")
    assert seen["offline"] is True


@pytest.mark.asyncio
async def test_llm_call_propagates_domain_and_deterministic_to_request_metadata():
    seen = []
    llm = MockLLM(responder=lambda req: (seen.append(dict(req.metadata)), "ok")[1])
    await Dummy(llm).run("x", domain="mechanical_engineering", constraints=Constraints(deterministic=True))
    assert seen and all(m.get("domain") == "mechanical_engineering" and m.get("deterministic") is True for m in seen)


def test_max_usage_keeps_cache_hits():
    from pipeline_core.component import _max_usage

    merged = _max_usage(Usage(llm_calls=2, cost_usd=0.1), Usage(llm_calls=1, cache_hits=3))
    assert merged.llm_calls == 2 and merged.cache_hits == 3
