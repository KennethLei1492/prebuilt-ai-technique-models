"""Domain adaptation: the same tasks under a general and a safety-critical profile.

Demonstrates how a ``DomainProfile`` changes behaviour without code changes:

* risk level ``high``/``safety_critical`` switches the failure policy to
  ``quorum`` (every branch must succeed before a safety-critical answer is
  assembled), raises the initial vote batch from 5 to 10 samples and adds the
  engineering-rigor rules to every stable prompt prefix;
* ``min_confidence_to_accept`` is higher, so weak agreement becomes
  ``NEEDS_REVIEW`` instead of ``SUCCESS``;
* a brand-new profile registered at runtime (``pressure_vessels``) is picked up
  immediately.

Also shows a long log triaged with map-reduce, a compliance review with
branch-solve-merge and parallel tool calls with an ordered side-effecting tool.

Run from the repository root::

    PYTHONPATH="common;parallelization_model" python parallelization_model/examples/engineering_example.py
"""
from __future__ import annotations

import asyncio

from pipeline_core import DomainProfile, TaskRequest, make_llm, register_profile

from parallelization import ParallelizationModel, ToolCall
from parallelization.evaluation import metrics_from_result

register_profile(
    DomainProfile(
        name="pressure_vessels",
        description="Pressure vessel design and inspection under ASME BPVC Section VIII.",
        units=["MPa", "bar", "mm", "degC", "kN"],
        verification_rules=["Every allowable stress must cite the code table and the metal design condition.", "Corrosion allowance must be stated."],
        risk_level="safety_critical",
        min_confidence_to_accept=0.85,
        standards=["ASME BPVC VIII-1", "API 510", "PED 2014/68/EU"],
    ),
    overwrite=True,
)


def show(label: str, result) -> None:
    plan = result.metadata["plan"]
    m = metrics_from_result(result)
    print(f"[{label}] {result.summary()}")
    print(f"   pattern={plan['pattern']} samples={plan['samples']}/{plan['max_samples']} failure_policy={plan['failure_policy']} quorum={plan['quorum']} threshold={result.metadata['threshold']}")
    print(f"   expected_speedup={m.expected_speedup}x calls={m.llm_calls} failed_branches={m.failed_branches} status={m.status}")
    print(f"   basis: {result.confidence_basis}")


async def main() -> None:
    llm = make_llm()
    model = ParallelizationModel(llm)
    calc = "Calculate the required shell thickness for a 2 m diameter vessel at 1.5 MPa design pressure with allowable stress 138 MPa."

    for domain in ("general", "mechanical_engineering", "pressure_vessels"):
        res = await model.run(TaskRequest(task=calc, domain=domain))
        show(domain, res)
        stable = "\n".join(llm.calls[-1].system_stable) if getattr(llm, "calls", None) else ""
        print(f"   rigor rules in prompt: {'Apply engineering rigor' in stable}; profile in prompt: {f'Domain: {domain}' in stable}\n")

    # Long inspection log -> map-reduce with structured records and a confidence-aware reduce
    log = "\n".join(f"2026-03-{(i % 28) + 1:02d} nozzle N{i % 9} UT reading {6.1 + (i % 7) * 0.05:.2f} mm (nominal 8.0 mm, min 5.8 mm)" for i in range(500))
    res = await model.run(TaskRequest(task="Summarise the wall-loss trend per nozzle and flag anything near the minimum.", inputs={"text": log}, domain="pressure_vessels"))
    show("map-reduce", res)
    print(f"   chunks={res.output.details.get('chunks')} reduce_levels={res.output.details.get('reduce_levels')}\n")

    # Compliance review -> branch-solve-merge over model-proposed criteria
    res = await model.run(
        TaskRequest(
            task="Review this design note for compliance with the code requirements.",
            inputs={"text": "Shell: SA-516 Gr.70, t = 14 mm, design 1.5 MPa at 200 degC, corrosion allowance 3 mm, joint efficiency 0.85."},
            domain="pressure_vessels",
        )
    )
    show("branch-solve-merge", res)
    print(f"   criteria={res.output.details.get('branches')}\n")

    # Parallel tool calls: lookups run concurrently, the PLM write runs sequentially under its own bulkhead
    async def lookup_material(grade: str) -> dict:
        return {"grade": grade, "allowable_mpa": 138}

    async def plm_write(record: str) -> str:
        return f"written:{record}"

    calls = [
        ToolCall(call_id="1", name="lookup_material", args={"grade": "SA-516-70"}),
        ToolCall(call_id="2", name="lookup_material", args={"grade": "SA-240-316L"}),
        ToolCall(call_id="3", name="plm_write", args={"record": "thickness-calc-v3"}),
    ]
    results = await model.parallel_tool_calls(calls, {"lookup_material": lookup_material, "plm_write": plm_write}, ordered_tools=["plm_write"])
    for r in results:
        print(f"   tool {r.name}[{r.call_id}] ok={r.ok} -> {r.result}")
    print("\nExecutor snapshot:", {k: v for k, v in model.executor.snapshot().items() if k in ("submitted", "succeeded", "failed", "retries", "breaker")})


if __name__ == "__main__":
    asyncio.run(main())
