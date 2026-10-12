"""Command-line interface.

Examples::

    python -m parallelization --offline "Compare steel, aluminium and titanium for a bracket" --domain mechanical_engineering
    python -m parallelization --offline --json "Classify each ticket" --input 'items=["Crash on save","Add dark mode"]'
    python -m parallelization --offline "Summarise failures" --input text=@build.log --max-calls 12
    python -m parallelization --offline --plan-only "What are the risks of ..."
    python -m parallelization --offline --evaluate --domain software_engineering
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pipeline_core import Constraints, TaskRequest, list_profiles, make_llm


def parse_input(spec: str) -> tuple[str, Any]:
    """``key=value`` -> (key, value). JSON values are decoded; ``@path`` reads a file."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--input expects key=value, got {spec!r}")
    key, raw = spec.split("=", 1)
    key = key.strip()
    if raw.startswith("@"):
        return key, Path(raw[1:]).read_text(encoding="utf-8")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        if "," in raw and key in ("items", "documents"):
            return key, [s.strip() for s in raw.split(",") if s.strip()]
        return key, raw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m parallelization", description="Plan and run prompt-level parallel reasoning.")
    p.add_argument("task", nargs="?", help="Natural-language task")
    p.add_argument("--domain", default="general", help=f"DomainProfile name (one of: {', '.join(list_profiles())})")
    p.add_argument("--offline", action="store_true", help="Force the deterministic MockLLM")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON")
    p.add_argument("--input", action="append", default=[], metavar="KEY=VALUE", help="Structured input; JSON values decoded; @path reads a file; repeatable")
    p.add_argument("--max-calls", type=int, default=None, help="Constraints.max_llm_calls")
    p.add_argument("--max-parallelism", type=int, default=None, help="Constraints.max_parallelism")
    p.add_argument("--max-latency", type=float, default=None, help="Constraints.max_latency_s")
    p.add_argument("--min-confidence", type=float, default=None, help="Constraints.min_confidence")
    p.add_argument("--deterministic", action="store_true", help="Constraints.deterministic (heuristic plan, no cache-aware gating)")
    p.add_argument("--plan-only", action="store_true", help="Print the heuristic plan and exit")
    p.add_argument("--evaluate", action="store_true", help="Run the offline evaluation harness over the synthetic task set")
    p.add_argument("--trace", action="store_true", help="Print the span tree after the run")
    return p


def _request(args: argparse.Namespace) -> TaskRequest:
    inputs: dict[str, Any] = {}
    for spec in args.input:
        k, v = parse_input(spec)
        inputs[k] = v
    c: dict[str, Any] = {}
    if args.max_calls is not None:
        c["max_llm_calls"] = args.max_calls
    if args.max_parallelism is not None:
        c["max_parallelism"] = args.max_parallelism
    if args.max_latency is not None:
        c["max_latency_s"] = args.max_latency
    if args.min_confidence is not None:
        c["min_confidence"] = args.min_confidence
    if args.deterministic:
        c["deterministic"] = True
    return TaskRequest(task=args.task, inputs=inputs, domain=args.domain, constraints=Constraints(**c))


def format_result(result: Any, plan: dict[str, Any]) -> str:
    """Human-readable rendering: summary, plan, metrics, output text."""
    lines = [result.summary(), ""]
    lines.append(f"Plan: pattern={plan.get('pattern')} concurrency={plan.get('concurrency')} failure_policy={plan.get('failure_policy')} quorum={plan.get('quorum')} aggregator={plan.get('aggregator')} planner={plan.get('planner')}")
    lines.append(f"      expected_calls={plan.get('expected_calls')} expected_speedup={plan.get('expected_speedup')}x samples={plan.get('samples')}/{plan.get('max_samples')} partitions={len(plan.get('partitions', []))}")
    lines.append(f"      rationale: {plan.get('rationale', '')}")
    out = result.output
    if out is not None:
        data = out.model_dump() if hasattr(out, "model_dump") else out
        lines.append(f"Run:  llm_calls={data.get('llm_calls')} measured_speedup={data.get('measured_speedup')}x branches={len(data.get('branches', []))} failed={data.get('failed_branches')} cache_warmups={data.get('cache_warmups')} cache_waits={data.get('cache_waits')}")
    lines.append(f"Confidence basis: {result.confidence_basis}")
    if result.warnings:
        lines.append("Warnings: " + "; ".join(result.warnings))
    if result.errors:
        lines.append("Errors: " + "; ".join(result.errors))
    lines.append("")
    lines.append("Output:")
    lines.append(result.output_text or "(empty)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    from .evaluation import evaluate
    from .model import ParallelizationModel

    # --offline selects the MockLLM for this invocation only; the process environment is not mutated
    model = ParallelizationModel(make_llm(offline=True) if args.offline else make_llm())
    if args.evaluate:
        report = evaluate(model, domain=args.domain)
        print(json.dumps(report.model_dump(exclude={"per_run"}), indent=2) if args.json else report.table())
        return 0
    if not args.task:
        build_parser().print_usage()
        print("error: a task is required unless --evaluate is given", file=sys.stderr)
        return 2
    request = _request(args)
    if args.plan_only:
        plan = model.heuristic_plan(request)
        print(json.dumps(plan.model_dump(mode="json"), indent=2) if args.json else plan.summary() + "\n" + plan.rationale)
        return 0
    result = model.run_sync(request)
    plan = result.metadata.get("plan", {})
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(format_result(result, plan))
    if args.trace:
        print("\nTrace:\n" + model.tracer.tree())
    return 0 if result.ok or result.status.value == "needs_review" else 1


__all__ = ["build_parser", "format_result", "main", "parse_input"]
