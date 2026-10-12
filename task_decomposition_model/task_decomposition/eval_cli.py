"""``python -m task_decomposition.eval_cli [--json] [--cases file.json] [--judge keyword|llm] [--online]``.

The evaluation harness entry point lives in its own module, which the package
``__init__`` does not import, so ``runpy`` executes it exactly once: running
``python -m task_decomposition.evaluation`` instead emits Python's
``RuntimeWarning`` about a module found in ``sys.modules`` before execution.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from pipeline_core import MockLLM, make_llm, run_sync

from .evaluation import EvalCase, evaluate, sample_cases
from .model import TaskDecompositionModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m task_decomposition.eval_cli", description="Offline evaluation of the task decomposition model.")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    parser.add_argument("--cases", help="JSON file with a list of EvalCase objects (defaults to the built-in sample suite)")
    parser.add_argument("--judge", choices=("keyword", "llm"), default="keyword", help="coverage metric: model-free keyword judge or one low-effort LLM judge call per case")
    parser.add_argument("--online", action="store_true", help="use make_llm() instead of forcing the MockLLM")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the sample suite (or ``--cases``) on the deterministic ``MockLLM``, or on ``make_llm()`` with ``--online``."""
    args = build_parser().parse_args(argv)
    if args.cases:
        with open(args.cases, encoding="utf-8") as fh:
            cases = [EvalCase.model_validate(c) for c in json.load(fh)]
    else:
        cases = sample_cases()
    model = TaskDecompositionModel(make_llm() if args.online else MockLLM())
    summary = run_sync(evaluate(model, cases, judge=args.judge))  # works inside a running loop too (notebooks, async tests)
    if args.json:
        print(json.dumps(summary.model_dump(mode="json"), indent=2))
    else:
        print(summary.table())
        for o in summary.outcomes:
            print(f"- [{o.status}] {o.strategy:<16} conf={o.confidence:.2f} calls={o.llm_calls} cov={o.plan.coverage:.2f} speedup={o.plan.parallel_speedup:.2f} | {o.task[:70]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
