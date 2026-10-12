"""Basic usage of the parallelization model.

Runs offline: ``make_llm()`` returns a deterministic ``MockLLM`` when no
Anthropic credential is configured, so this script always produces output.
With ``ANTHROPIC_API_KEY`` set the same code plans with one structured Claude
call and executes the chosen pattern for real.

Run from the repository root::

    PYTHONPATH="common;parallelization_model" python parallelization_model/examples/basic_usage.py
"""
from __future__ import annotations

import asyncio

from pipeline_core import Constraints, TaskRequest, make_llm

from parallelization import ParallelizationModel


def show(result) -> None:
    plan = result.metadata["plan"]
    print(result.summary())
    print(f"  plan: {plan['pattern']} (planner={plan['planner']}, concurrency={plan['concurrency']}, failure_policy={plan['failure_policy']}, expected_calls={plan['expected_calls']}, expected_speedup={plan['expected_speedup']}x)")
    print(f"  why:  {plan['rationale']}")
    print(f"  confidence basis: {result.confidence_basis}")
    print(f"  output: {result.output_text[:200]!r}")
    print()


async def main() -> None:
    model = ParallelizationModel(make_llm())

    # 1. Enumerable items in the task text -> sectioning (one worker per item, LLM merge)
    show(await model.run("Compare steel, aluminium and titanium for a lightweight bracket."))

    # 2. Point-independent long form -> skeleton-of-thought (skeleton, parallel expansion, concat)
    show(await model.run("What are the main failure modes of a bolted flange joint?"))

    # 3. Single checkable answer -> sampling-and-voting with early stopping
    show(await model.run("Calculate the bending stress for a 2 m beam with a 10 kN central load and section modulus 200 cm3."))

    # 4. Step-dependent reasoning -> no parallelism (SoT-R finding)
    show(await model.run("Derive the deflection formula for a simply supported beam step by step."))

    # 5. Structured list input -> one partition per item, budget-capped
    show(
        await model.run(
            TaskRequest(
                task="Classify each ticket as bug, feature or question and give a one-line reason.",
                inputs={"items": ["Crash on save", "Add dark mode", "How do I export?", "Login loops forever"]},
                constraints=Constraints(max_llm_calls=4),
            )
        )
    )

    # 6. Direct API: the executor and the patterns are usable without run()
    async def fetch(part: str) -> str:
        return f"{part}: ok"

    print("executor.map ->", await model.map(fetch, ["P-100", "P-200", "P-300"]))
    vote = await model.vote("Is 97 a prime number?", samples=3)
    print(f"vote -> {vote.answer!r} confidence={vote.confidence} calls={vote.llm_calls} basis={vote.confidence_basis}")
    print("\nSpan tree:\n" + model.tracer.tree()[:1200])


if __name__ == "__main__":
    asyncio.run(main())
