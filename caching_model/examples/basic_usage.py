"""Basic usage of the caching model.

Runs offline: ``make_llm()`` returns a deterministic ``MockLLM`` when no
Anthropic credential is configured, so this script is safe to run anywhere.

    cd <repo root>
    PYTHONPATH="common;caching_model" python caching_model/examples/basic_usage.py     # Windows
    PYTHONPATH="common:caching_model" python caching_model/examples/basic_usage.py     # POSIX
"""
from __future__ import annotations

import asyncio

from pipeline_core import LLMRequest, TaskRequest, make_llm

from caching import CachingModel


async def main() -> None:
    llm = make_llm()  # MockLLM without credentials, AnthropicLLM with them
    model = CachingModel(llm)

    # 1. Plan / explain caching for a prompt (the component API) --------------
    for task, domain in [
        ("What does clause 4.2 of ISO 2768 require for a 120 mm dimension?", "mechanical_engineering"),
        ("What is the current status of the nightly build right now?", "software_engineering"),
        ("Brainstorm five names for a new torque sensor product line", "general"),
    ]:
        result = await model.run(TaskRequest(task=task, domain=domain, inputs={"expected_reuses": 10}))
        print(result.summary())
        print(result.output_text)
        print("-" * 78)

    # 2. Use the model as middleware: wrap any client ---------------------------
    cached = model.wrap(llm)
    req = LLMRequest.simple("Define factor of safety for a bolted joint.", system=["You are a careful mechanical engineer."])
    first = await cached.complete(req)
    second = await cached.complete(req)  # exact hit: no tokens, no cost, no LLM call
    print(f"first call cached={first.cached} llm_calls={first.usage.llm_calls}")
    print(f"second call cached={second.cached} cache_hits={second.usage.cache_hits} tier={getattr(second, 'cache', {}).get('cache_tier')}")
    print("savings report:", {k: model.savings_report()[k] for k in ("requests", "exact_hits", "misses", "hit_rate")})

    # 3. The ResponseCache protocol other models accept -------------------------
    await model.cache.set("my-key", {"answer": 42}, ttl_s=60, tags=["doc:spec-1"])
    print("protocol get:", await model.cache.get("my-key"))
    print("invalidated by document tag:", await model.cache.invalidate(tags=["doc:spec-1"]))
    print(model.tracer.tree())


if __name__ == "__main__":
    asyncio.run(main())
