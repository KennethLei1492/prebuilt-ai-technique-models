"""Command-line entry point.

Examples (run from the repository root; use ``;`` instead of ``:`` in
``PYTHONPATH`` on Windows)::

    PYTHONPATH="common:task_decomposition_model" python -m task_decomposition --offline \\
        "For each of the three pump logs extract the duty point, then summarise the fleet status" --domain mechanical_engineering
    python -m task_decomposition "..." --json --input documents="spec text" --strategy dag --max-calls 6
    python -m task_decomposition "..." --plan-only
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

from pipeline_core import Constraints, MockLLM, TaskRequest, list_profiles, make_llm, run_sync

from .model import DecompositionOutput, TaskDecompositionModel
from .plan import STRATEGIES


def parse_inputs(pairs: Sequence[str]) -> dict[str, Any]:
    """``key=value`` pairs -> dict; values that parse as JSON become typed, ``documents`` always becomes a list."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--input expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if key in ("documents", "document"):
            if isinstance(value, str):
                value = [value]
            out.setdefault("documents", []).extend(value if isinstance(value, list) else [value])
        else:
            out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m task_decomposition", description="Decompose a task into a validated plan, execute it and synthesise the answer.")
    p.add_argument("task", help="the task in natural language")
    p.add_argument("--domain", default="general", help=f"DomainProfile name (one of: {', '.join(list_profiles())})")
    p.add_argument("--offline", action="store_true", help="force the deterministic MockLLM (no network, no credentials)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of the text summary")
    p.add_argument("--input", action="append", default=[], metavar="KEY=VALUE", help="structured input; repeatable; JSON values are parsed; documents=... adds a document")
    p.add_argument("--strategy", choices=list(STRATEGIES), help="force a decomposition strategy")
    p.add_argument("--max-calls", type=int, default=None, help="LLM call budget for the run")
    p.add_argument("--max-parallelism", type=int, default=8, help="concurrency ceiling for a wave")
    p.add_argument("--deterministic", action="store_true", help="prefer heuristic planning paths (reproducible runs)")
    p.add_argument("--plan-only", action="store_true", help="plan and decompose only; print the graph without executing it")
    p.add_argument("--trace", action="store_true", help="print the tracer span tree after the run")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    llm = MockLLM() if args.offline else make_llm()
    model = TaskDecompositionModel(llm)
    constraints = Constraints(max_llm_calls=args.max_calls, max_parallelism=args.max_parallelism, deterministic=args.deterministic)
    request = TaskRequest(task=args.task, domain=args.domain, inputs=parse_inputs(args.input), constraints=constraints, metadata={"strategy": args.strategy} if args.strategy else {})

    if args.plan_only:
        graph = run_sync(model.decompose(request))
        if args.json:
            print(json.dumps({"plan": graph.metadata.get("plan"), "graph": graph.model_dump(mode="json"), "metrics": model.evaluate_plan(graph).model_dump(mode="json")}, indent=2, ensure_ascii=False))
        else:
            plan = graph.metadata.get("plan") or {}
            print(f"strategy={plan.get('strategy')} steps~{plan.get('estimated_steps')} verification={plan.get('verification')} parallelism={plan.get('parallelism')}")
            print(f"rationale: {plan.get('rationale')}")
            print()
            print(graph.to_ascii())
            print()
            print(graph.to_mermaid())
        return 0

    result = run_sync(model.run(request))
    output = result.output if isinstance(result.output, DecompositionOutput) else None
    if args.json:
        payload: dict[str, Any] = {
            "summary": result.summary(),
            "status": result.status.value,
            "confidence": result.confidence,
            "confidence_basis": result.confidence_basis,
            "output_text": result.output_text,
            "plan": result.metadata.get("plan"),
            "output": output.model_dump(mode="json") if output else result.output,
            "usage": result.usage.model_dump(mode="json"),
            "warnings": result.warnings,
            "errors": result.errors,
            "trace_id": result.trace_id,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(result.summary())
        print(f"confidence basis: {result.confidence_basis}")
        print()
        if output is not None:
            print(output.render())
            print()
        print("OUTPUT")
        print(result.output_text or "(no output)")
        if result.warnings:
            print()
            print("WARNINGS")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print()
            print("ERRORS")
            for e in result.errors:
                print(f"  - {e}")
    if args.trace:
        print()
        print(model.tracer.tree())
    return 0 if result.ok or result.status.value == "needs_review" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
