"""Command-line entry point: ``python -m caching "task" [--domain NAME] [--offline] [--json] [--input key=value ...]``.

Prints ``result.summary()``, the plan and the human-readable report, or the
full :class:`pipeline_core.TaskResult` as JSON with ``--json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

from pipeline_core import TaskRequest, list_profiles, make_llm


def parse_inputs(pairs: Sequence[str]) -> dict[str, Any]:
    """Turn ``key=value`` pairs into a dict; values are parsed as JSON when possible."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"--input expects key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            out[k.strip()] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m caching", description="Plan and explain response caching for a prompt.")
    p.add_argument("task", help="The prompt / task to classify")
    p.add_argument("--domain", default="general", help=f"Domain profile (one of: {', '.join(list_profiles())})")
    p.add_argument("--offline", action="store_true", help="Force the deterministic MockLLM (heuristic planner)")
    p.add_argument("--json", action="store_true", help="Print the TaskResult as JSON")
    p.add_argument("--input", action="append", default=[], metavar="KEY=VALUE", help="Structured input (repeatable); JSON values accepted")
    p.add_argument("--model", default=None, help="Model id to plan and price for (default: client model or claude-opus-5-5)")
    p.add_argument("--expected-reuses", type=int, default=None, help="Reuse assumption for the savings estimate")
    p.add_argument("--trace", action="store_true", help="Print the span tree after the result")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline:
        os.environ["PIPELINE_OFFLINE"] = "1"
    from .model import CachingModel  # imported here so --offline is honoured by make_llm()

    inputs = parse_inputs(args.input)
    if args.model:
        inputs["model"] = args.model
    if args.expected_reuses is not None:
        inputs["expected_reuses"] = args.expected_reuses
    model = CachingModel(make_llm())
    result = model.run_sync(TaskRequest(task=args.task, inputs=inputs, domain=args.domain))
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str))
    else:
        print(result.summary())
        print()
        print("Plan:")
        print(json.dumps(result.metadata.get("plan", {}), indent=2, ensure_ascii=False))
        print()
        print(result.output_text)
        if result.warnings:
            print()
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print()
            print("Errors:")
            for e in result.errors:
                print(f"  - {e}")
    if args.trace:
        print()
        print(model.tracer.tree())
    return 0 if result.ok or result.status.value == "needs_review" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
