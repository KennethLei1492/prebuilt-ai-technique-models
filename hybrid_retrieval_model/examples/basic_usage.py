"""Basic usage of the hybrid retrieval model.

Runs offline: ``make_llm()`` returns a deterministic ``MockLLM`` when no
Anthropic credential is configured, and the heuristic planner takes over.

    python hybrid_retrieval_model/examples/basic_usage.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "hybrid_retrieval_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import make_llm  # noqa: E402

from hybrid_retrieval import HybridRetrievalModel  # noqa: E402

DOCUMENTS = [
    {
        "id": "spec-pump",
        "text": (
            "# Pump specification PS-4471 rev B\n\n"
            "## 4.1 Casing\nThe casing shall be ASTM A216 WCB carbon steel with a corrosion allowance of 3 mm.\n\n"
            "## 4.2 Impeller\nThe impeller material shall be duplex stainless steel ASTM A890 Grade 4A (CD3MN).\n\n"
            "## 4.3 Shaft seal\nMechanical seal per API 682 Category 2, plan 11 flush."
        ),
        "metadata": {"title": "Pump specification PS-4471", "type": "specification", "revision": "B"},
    },
    {
        "id": "incident-4821",
        "text": (
            "# Incident INC-4821: cooling water pump cavitation\n\n"
            "Pump P-203 showed cavitation noise and a 15% head loss after the suction strainer clogged. "
            "Root cause: NPSH available fell below NPSH required. Corrective action: differential pressure alarm at 0.25 bar."
        ),
        "metadata": {"title": "Incident INC-4821", "type": "incident", "service": "cooling"},
    },
    {
        "id": "runbook-err-1203",
        "text": (
            "# Runbook ERR-1203\n\nERR-1203 is raised when the heartbeat from the field bus is lost for longer than the configured timeout. "
            "Restore communication, check the bus terminator and clear the fault."
        ),
        "metadata": {"title": "Runbook ERR-1203", "type": "runbook"},
    },
]


async def main() -> None:
    llm = make_llm()  # MockLLM without credentials
    model = HybridRetrievalModel(llm)
    n_chunks = model.index(DOCUMENTS)
    print(f"indexed {len(DOCUMENTS)} documents into {n_chunks} chunks\n")

    # 1. Component API: plan -> execute, with a readable report
    result = await model.run("Which clause specifies the pump impeller material?")
    print(result.summary())
    print(result.output_text)
    print()

    # 2. The plan is auditable
    print("plan:", json.dumps({k: result.metadata["plan"][k] for k in ("query_type", "fusion", "sparse_weight", "dense_weight", "rerank", "query_transforms", "source")}))
    print()

    # 3. Direct Retriever-protocol API (no LLM calls): what Pipeline RAG and verification consume
    evidence = await model.retrieve("ERR-1203", k=2)
    for ev in evidence:
        print(f"retrieve(): {ev.id} score={ev.score:.3f} via {ev.retriever} channels={ev.metadata['channels']} -> {ev.short(80)}")
    print()

    # 4. Domain adaptation: the same query planned for two domains
    for domain in ("general", "mechanical_engineering"):
        res = await model.run("corrosion allowance casing", domain=domain)
        plan = res.metadata["plan"]
        print(f"{domain:<24} fusion={plan['fusion']:<6} sparse_weight={plan['sparse_weight']:.2f} rerank={plan['rerank']}@{plan['rerank_depth']} status={res.status.value} confidence={res.confidence:.2f}")

    print("\nspan tree:\n" + model.tracer.tree())


if __name__ == "__main__":
    asyncio.run(main())
