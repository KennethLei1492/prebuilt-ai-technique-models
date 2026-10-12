"""Domain adaptation: the same prompt under a general, a high-risk and a safety-critical profile,
plus a custom registered profile, grounded answers with dependency invalidation, and a
prefix-cache audit of a standards-heavy prompt. Runs offline.

    PYTHONPATH="common;caching_model" python caching_model/examples/engineering_example.py
"""
from __future__ import annotations

import asyncio

from pipeline_core import DomainProfile, Evidence, LLMRequest, TaskRequest, make_llm, register_profile

from caching import CachingModel


async def main() -> None:
    model = CachingModel(make_llm())
    task = "What does clause 4.2 of ISO 2768 require for a 120 mm dimension?"

    # 1. One prompt, three risk postures --------------------------------------
    print("=== Domain adaptation ===")
    for domain in ("general", "mechanical_engineering", "civil_structural_engineering"):
        r = await model.run(TaskRequest(task=task, domain=domain))
        p = r.output.plan
        print(
            f"{domain:<30} status={r.status.value:<12} tiers={p.tiers} threshold={p.semantic_threshold} "
            f"budget={p.error_budget} prefix_ttl={p.prefix_ttl} confidence={r.confidence:.2f} (needs >= {r.metadata['confidence_threshold']})"
        )

    # 2. A new field registered by configuration, no code changes -------------
    register_profile(
        DomainProfile(
            name="aerospace_structures",
            description="Airframe structural substantiation.",
            units=["kN", "MPa"],
            standards=["FAR 25", "MMPDS"],
            document_types=["stress report", "drawing"],
            risk_level="safety_critical",
            min_confidence_to_accept=0.9,
        ),
        overwrite=True,
    )
    r = await model.run(TaskRequest(task="Which FAR 25.571 damage-tolerance criteria apply to a 2.5 mm skin?", domain="aerospace_structures"))
    print(f"{'aerospace_structures':<30} status={r.status.value:<12} tiers={r.output.plan.tiers} reason={r.output.plan.reason[:90]}...")

    # 3. Grounded answers: dependency fingerprint and tag invalidation --------
    print("\n=== Dependency-based invalidation ===")
    docs = [Evidence(id="iso2768-s4", content="Clause 4.2: general tolerances for linear dimensions, class m ..."), Evidence(id="iso2768-s5", content="Clause 5: angular dimensions ...")]
    r1 = await model.run(TaskRequest(task=task, context=docs, domain="mechanical_engineering", inputs={"corpus_version": "snap-2026-q3", "response": {"text": "+/- 0.5 mm (class m)"}}))
    print("stored under key", r1.output.exact_key[:16], "tags", r1.output.tags)
    r2 = await model.run(TaskRequest(task=task, context=docs, domain="mechanical_engineering", inputs={"corpus_version": "snap-2026-q3"}))
    print("entry exists before re-index:", r2.output.cached_entry_exists)
    ev = model.invalidate_documents(["iso2768-s4"])
    print(f"document updated -> purged exact={ev.removed_exact} semantic={ev.removed_semantic}")
    changed = [Evidence(id="iso2768-s4", content="Clause 4.2 (rev 2): general tolerances ..."), docs[1]]
    r3 = await model.run(TaskRequest(task=task, context=changed, domain="mechanical_engineering", inputs={"corpus_version": "snap-2026-q3"}))
    print("key changes with content:", r3.output.exact_key != r1.output.exact_key, "| changed docs:", r3.output.dependency_fingerprint.changed_since(r1.output.dependency_fingerprint))

    # 4. Prefix-cache audit of a standards-heavy prompt ------------------------
    print("\n=== Provider prefix-cache audit ===")
    standard_text = "ISO 2768-1 Table 1. General tolerances for linear dimensions, tolerance class m: " + "over 30 up to 120: +/- 0.3 mm; over 120 up to 400: +/- 0.5 mm. " * 60
    good = LLMRequest.simple(task, system=["You are a tolerance analyst.", standard_text])
    bad = LLMRequest.simple(task, system=["You are a tolerance analyst. Report generated at 2026-03-01T09:00:00Z", standard_text])
    for name, req in (("stable prefix", good), ("volatile prefix", bad)):
        rep = model.analyze_prefix(req, model="claude-opus-5-5")
        print(f"{name:<16} tokens={rep.prefix_tokens} meets_min={rep.meets_minimum} invalidators={[i.kind for i in rep.invalidators]} ttl={rep.recommended_ttl} breakeven={rep.breakeven_uses} score={rep.score}")
        for rec in rep.recommendations[:2]:
            print("   -", rec)

    # 5. Middleware with the engineering profile: guards block near-miss hits --
    print("\n=== Semantic guards under the mechanical profile ===")
    client = model.wrap(make_llm(), profile="mechanical_engineering")
    sys_block = ["You answer questions about ISO 2768 general tolerances."]
    await client.complete(LLMRequest.simple("What tolerance applies to a 120 mm dimension in class m?", system=sys_block))
    r_same = await client.complete(LLMRequest.simple("What tolerance applies to a 120 mm dimension in class m ?", system=sys_block))
    r_diff = await client.complete(LLMRequest.simple("What tolerance applies to a 12 mm dimension in class m?", system=sys_block))
    print(f"same question re-asked: cached={r_same.cached}; 12 mm instead of 120 mm: cached={r_diff.cached}")
    print("metrics:", {k: v for k, v in client.metrics.model_dump().items() if v})


if __name__ == "__main__":
    asyncio.run(main())
