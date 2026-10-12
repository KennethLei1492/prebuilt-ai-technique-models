"""Engineering-domain example: custom DomainProfile, mandatory filters, budgets, persistence and evaluation.

Runs offline (``make_llm()`` falls back to ``MockLLM`` without credentials).

    python hybrid_retrieval_model/examples/engineering_example.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "common", ROOT / "hybrid_retrieval_model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_core import Constraints, DomainProfile, TaskRequest, make_llm, register_profile  # noqa: E402

from hybrid_retrieval import HybridRetrievalModel  # noqa: E402
from hybrid_retrieval.evaluation import ablation, evaluate, synthetic_engineering_corpus  # noqa: E402

# A new field is a profile, not a code change: identifiers dominate, risk is safety critical.
register_profile(
    DomainProfile(
        name="rotating_equipment",
        description="Pumps, compressors and turbines: specifications, datasheets, incident reports and maintenance runbooks.",
        glossary={"NPSH": "Net positive suction head", "MAWP": "Maximum allowable working pressure"},
        units=["bar", "degC", "MPa", "mm", "rpm", "m3/h"],
        document_types=["specification", "datasheet", "incident", "runbook"],
        verification_rules=["Material designations must cite the ASTM/EN standard.", "Pressures must state the reference temperature."],
        risk_level="safety_critical",
        min_confidence_to_accept=0.8,
        retrieval_hints={"sparse_weight": 0.65, "reason": "tag numbers, material grades and clause ids are exact tokens"},
        standards=["API 610", "API 682", "ASME B73.1", "ISO 21940"],
    ),
    overwrite=True,
)


async def main() -> None:
    llm = make_llm()
    model = HybridRetrievalModel(llm, domain="rotating_equipment", k_final=5)
    documents, judgments = synthetic_engineering_corpus(n_docs=24)
    model.index(documents)
    print(f"indexed {len(documents)} synthetic documents -> {model.retriever.n_chunks} chunks\n")

    # 1. Domain-adapted plan: higher lexical weight, convex fusion, mandatory reranking at depth 30
    request = TaskRequest(
        task="What corrosion allowance is specified for the pump casing?",
        domain="rotating_equipment",
        inputs={"filters": {"type": "specification"}},  # mandatory ACL/type filter the planner cannot remove
    )
    result = await model.run(request)
    print(result.summary())
    print(result.output_text)
    print()

    # 2. Budget enforcement: with max_llm_calls=0 the plan degrades (no HyDE, no reranking) instead of failing
    tight = TaskRequest(task="previous cavitation incidents on cooling water pumps", domain="rotating_equipment", constraints=Constraints(max_llm_calls=0))
    res = await model.run(tight)
    print("budget run:", res.summary())
    print("  degradations:", res.metadata["plan"]["degradations"])
    print("  status:", res.status.value, "| warnings:", res.warnings[:2])
    print()

    # 3. Same query in the general profile: different plan, lower acceptance threshold
    general = await model.run("previous cavitation incidents on cooling water pumps", domain="general")
    print("general run:", general.summary(), "| threshold", general.metadata["threshold"])
    print("  plan:", general.metadata["plan"]["fusion"], general.metadata["plan"]["sparse_weight"], general.metadata["plan"]["rerank"])
    print()

    # 4. Persistence: save, reload, same answers
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.json"
        model.save_index(path)
        fresh = HybridRetrievalModel(llm, domain="rotating_equipment")
        fresh.load_index(path)
        a = [e.id for e in await model.retrieve("ERR-1202", k=3)]
        b = [e.id for e in await fresh.retrieve("ERR-1202", k=3)]
        print("persistence round-trip identical:", a == b, a)
    print()

    # 5. Evaluation with the lexical baseline reported alongside the hybrid result
    report = await evaluate(model, judgments, k=10)
    print("evaluation:\n  " + report.summary().replace("\n", "\n  "))
    reports = await ablation(model, judgments, k=10)
    for name in ("sparse", "dense", "hybrid"):
        print("  " + reports[name].summary().splitlines()[0])


if __name__ == "__main__":
    asyncio.run(main())
