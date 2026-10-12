"""Command-line interface.

Examples::

    python -m answer_verification "Verify this answer: the NPSH margin is 1.5 m" --offline
    python -m answer_verification "What is the bolt preload?" --answer "12.5 kN" --domain mechanical_engineering --json
    python -m answer_verification "Is the casing material correct?" --answer "Duplex stainless steel" \
        --document "doc-1: The casing shall be ASTM A995 Gr 4A duplex stainless steel" --offline
    python -m answer_verification --evaluate --offline       # run the synthetic evaluation harness
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from pipeline_core import Constraints, Evidence, MockLLM, Status, TaskRequest, list_profiles, make_llm

from .config import AnswerVerificationConfig
from .model import AnswerVerificationModel


def parse_inputs(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse ``key=value`` pairs; values are JSON when parseable, ``@path`` reads a file."""
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"--input expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        if v.startswith("@") and Path(v[1:]).exists():
            v = Path(v[1:]).read_text(encoding="utf-8")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def parse_documents(docs: Sequence[str]) -> list[Evidence]:
    """``id: text`` or plain text (ids are assigned) or ``@path`` files, one Evidence each."""
    out: list[Evidence] = []
    for i, d in enumerate(docs, 1):
        source = ""
        if d.startswith("@") and Path(d[1:]).exists():
            source = d[1:]
            d = Path(d[1:]).read_text(encoding="utf-8")
        doc_id, text = f"doc-{i}", d
        if ": " in d and len(d.split(": ", 1)[0]) <= 24 and " " not in d.split(": ", 1)[0]:
            doc_id, text = d.split(": ", 1)
        out.append(Evidence(id=doc_id, content=text, source=source, retriever="provided"))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m answer_verification", description="Verify a candidate answer with research-grounded verifiers.")
    ap.add_argument("task", nargs="?", default=None, help="Natural-language task; 'Verify this answer: X' supplies the candidate inline")
    ap.add_argument("--answer", help="Candidate answer (shortcut for --input answer=...)")
    ap.add_argument("--domain", default="general", help=f"DomainProfile name; one of {', '.join(list_profiles())}")
    ap.add_argument("--offline", action="store_true", help="Use the deterministic MockLLM (no network)")
    ap.add_argument("--json", action="store_true", help="Print the full TaskResult as JSON")
    ap.add_argument("--input", action="append", default=[], metavar="KEY=VALUE", help="Structured input (JSON values parsed; @file reads a file)")
    ap.add_argument("--document", action="append", default=[], metavar="TEXT", help="Evidence document ('id: text', plain text or @file); repeatable")
    ap.add_argument("--tests", metavar="PATH", help="Python test file to run against code answers (implies --allow-code-execution)")
    ap.add_argument("--allow-code-execution", action="store_true", help="Permit the sandboxed code execution verifier")
    ap.add_argument("--max-calls", type=int, default=None, help="LLM call budget")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="Extra acceptance threshold")
    ap.add_argument("--deterministic", action="store_true", help="Prefer heuristic/cached paths (minimum samples)")
    ap.add_argument("--llm-mode", choices=["auto", "force", "never"], default="auto")
    ap.add_argument("--trace", action="store_true", help="Print the span tree after the result")
    ap.add_argument("--evaluate", action="store_true", help="Run the synthetic evaluation harness instead of a single task")
    ap.add_argument("--n", type=int, default=12, help="Number of synthetic examples for --evaluate")
    return ap


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI; returns the process exit code."""
    _configure_stdout()
    args = build_parser().parse_args(argv)
    if args.offline:
        os.environ["PIPELINE_OFFLINE"] = "1"
    llm = MockLLM() if args.offline else make_llm()
    config = AnswerVerificationConfig(llm_mode=args.llm_mode)
    model = AnswerVerificationModel(llm, config=config)

    if args.evaluate:
        from .evaluation import evaluate_sync, synthetic_examples

        report = evaluate_sync(model, synthetic_examples(args.n), domain=args.domain)
        print(json.dumps(report.model_dump(mode="json"), indent=2) if args.json else report.render())
        return 0

    if not args.task:
        build_parser().error("a task is required unless --evaluate is given")
    inputs = parse_inputs(args.input)
    if args.answer:
        inputs["answer"] = args.answer
    allow_code = args.allow_code_execution
    if args.tests:
        inputs["tests"] = Path(args.tests).read_text(encoding="utf-8")
        allow_code = True
    constraints = Constraints(
        max_llm_calls=args.max_calls,
        allow_code_execution=allow_code,
        deterministic=args.deterministic,
        min_confidence=args.min_confidence,
    )
    request = TaskRequest(task=args.task, inputs=inputs, context=parse_documents(args.document), domain=args.domain, constraints=constraints)
    result = model.run_sync(request)

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    else:
        print(result.summary())
        print()
        print("Plan:")
        plan = result.metadata.get("plan", {})
        for key in ("answer_type", "verifiers", "mandatory", "samples", "thresholds", "estimated_calls", "source", "rationale"):
            if key in plan:
                print(f"  {key}: {json.dumps(plan[key]) if not isinstance(plan[key], str) else plan[key]}")
        print()
        print(result.output_text)
        if result.warnings:
            print()
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print("Errors:")
            for e in result.errors:
                print(f"  - {e}")
        if args.trace:
            print()
            print(model.tracer.tree())
    return 0 if result.status != Status.FAILED else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
