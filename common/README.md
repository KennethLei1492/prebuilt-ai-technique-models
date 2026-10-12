# `pipeline_core` - shared foundation for all six models

Every model folder in this repository depends on this package and on nothing
else inside the repository. It defines the contracts that make the models
composable, the Claude client they all share, and the cross-cutting concerns
(tracing, budgets, prompt-cache-aware prompting, domain adaptation) that a
company deployment needs and that no single model should own.

## What lives here

| Module | Purpose |
| --- | --- |
| `contracts.py` | `TaskRequest`, `TaskResult`, `Evidence`, `Constraints`, `Usage`, `StepRecord`, `BudgetTracker` - the types every component consumes and produces. |
| `component.py` | `PipelineComponent` base class (plan -> execute life cycle), `RunContext` (budget, tracing, steps, LLM calls), `ComponentSpec`. |
| `llm.py` | `LLMRequest`/`LLMResponse`; `AnthropicLLM` (production), `MockLLM`, `ScriptedLLM`, `RecordingLLM`, `ReplayLLM` (offline/test). Pricing table and cacheable-minimum table. |
| `prompts.py` | `PromptTemplate` with stable-prefix/volatile-suffix split, XML helpers, evidence rendering, shared rule snippets. |
| `domain.py` | `DomainProfile` registry: general, software, mechanical, electrical, civil/structural, systems, data, chemical/process engineering. |
| `protocols.py` | `Retriever`, `Verifier`, `Executor`, `ResponseCache`, `Verdict`, `SubTaskSolver` - structural interfaces for model-to-model wiring. |
| `telemetry.py` | `Tracer`/`Span` with OpenTelemetry GenAI attribute names, JSONL export, ASCII span tree. |
| `registry.py` | `@register` decorator and `discover()` for orchestrators. |
| `config.py` | Environment settings and `make_llm()` factory (falls back to `MockLLM` without credentials). |
| `errors.py` | Typed exception hierarchy. |
| `utils.py` | Canonical JSON, stable hashing, token estimation, chunking, `run_sync`, bounded gather. |

## The component life cycle

```python
from pipeline_core import PipelineComponent, ComponentSpec, RunContext, TaskRequest, TaskResult, LLMRequest, register

@register
class MyModel(PipelineComponent):
    spec = ComponentSpec(name="my_model", description="...", capabilities=["..."])

    async def plan(self, request: TaskRequest, ctx: RunContext):
        # 1. Reason about the prompt: one structured LLM call, heuristics when ctx.offline
        ...

    async def execute(self, request: TaskRequest, plan, ctx: RunContext) -> TaskResult:
        with ctx.step("retrieve"):
            resp = await ctx.llm_call(LLMRequest.simple("...", system=[ctx.profile.system_text()]))
        return self.new_result(request, output=..., output_text=..., confidence=..., confidence_basis=...)

result = MyModel().run_sync("Summarise the failure modes in these test logs", domain="mechanical_engineering")
print(result.summary()); print(MyModel().tracer.tree())
```

`run()` never raises by default: failures become `TaskResult(status=FAILED)`
with the error recorded, so orchestrators can decide what to do. Budgets
(`Constraints.max_llm_calls`, `max_cost_usd`, `max_latency_s`) are enforced on
every `ctx.llm_call`.

## Claude usage rules baked into `AnthropicLLM`

These follow the current Claude API behaviour (2026) and Anthropic's prompt-caching guidance:

- Default model `claude-opus-5-5`; exact IDs only (no date suffixes).
- Thinking is adaptive and always on; depth is controlled with `output_config.effort` (sent explicitly because Opus 5.5 defaults to `medium`).
- No `temperature`/`top_p`/`top_k` (rejected by current models). Determinism comes from caching and self-consistency, not sampling knobs.
- Structured output via `output_config.format = {"type": "json_schema", ...}`; schemas are sanitised (`additionalProperties: false`, no `title`/`default`).
- Prompt caching: `cache_control` goes on the **last stable system block**; volatile text is appended after it. Caches are prefix matches, so templates keep stable content first. Minimum cacheable prefix is 512 tokens on current models (see `MIN_CACHEABLE_TOKENS`).
- Server-side refusal fallbacks (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`) are on by default; disable with `use_fallbacks=False` on Bedrock/Vertex.
- `stop_reason == "refusal"` raises `LLMRefusalError`; truncated structured output raises `LLMTruncatedError`.
- Requests with `max_tokens > 16000` stream automatically and use `get_final_message()`.
- Cost is estimated from the pricing table and accumulated into `Usage.cost_usd`.

## Offline mode

`make_llm()` returns a `MockLLM` when `PIPELINE_OFFLINE=1` or when no
credential is discoverable. `MockLLM` fabricates schema-conformant JSON for
structured requests, so every model's `plan()` still returns a valid plan and
every test runs in CI without secrets. Components additionally ship heuristic
planners that are used when `ctx.offline` is true so demos produce meaningful
behaviour, not just valid shapes.

## Domain adaptation

`TaskRequest.domain` names a `DomainProfile`. Profiles carry glossary, units,
document types, verification rules, standards, risk level and retrieval
weights. Components render `ctx.profile.system_text()` into their stable
system prompt, read `retrieval_hints["sparse_weight"]`, and apply
`verification_rules` and `min_confidence_to_accept`. To support a new field:

```python
from pipeline_core import DomainProfile, register_profile
register_profile(DomainProfile(name="aerospace_structures", description="...", units=["kN", "MPa"], risk_level="safety_critical", standards=["FAR 25"]))
```

## Running the tests

```bash
python -m pytest common/tests -q
```
