"""Command line: ``python -m pipeline_rag "task" [--domain NAME] [--offline] [--json] [--input key=value ...]``.

Documents can be supplied with ``--documents PATH`` (a text/markdown file, a
JSON file holding a list of documents, or a directory of text files; repeat
the flag for several), with ``--input documents=...`` or, for a quick demo,
``--sample-corpus`` (the synthetic engineering corpus from
:mod:`pipeline_rag.evaluation`). Without documents the router selects
``no_retrieval`` and the model answers (or abstains) from its own knowledge.
``--evaluate`` additionally runs the evaluation harness
(:class:`pipeline_rag.evaluation.RAGEvaluator`) over the same documents.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from pipeline_core import Constraints, MockLLM, TaskRequest, list_profiles, make_llm

from .config import PipelineRAGConfig
from .model import PipelineRAGModel


def _parse_inputs(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--input expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def load_documents(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Load documents from files or directories.

    A ``.json`` file may hold a list of documents (strings or dicts), a
    ``{"documents": [...]}`` object or an ``{id: text}`` / ``{id: {...}}``
    mapping (the same shapes ``run()`` accepts); any other file is one plain
    text document named after its stem. A path that yields no document is an
    error (``SystemExit``) rather than a silent ``no_retrieval`` run.
    """
    docs: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise SystemExit(f"document path not found: {p}")
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        before = len(docs)
        for f in files:
            if f.suffix.lower() == ".json":
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    items: list[Any] = data
                elif isinstance(data, dict) and isinstance(data.get("documents"), list):
                    items = data["documents"]
                elif isinstance(data, dict):  # {id: text} or {id: {text, source, ...}} mapping
                    items = [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in data.items()]
                else:
                    raise SystemExit(f"{f}: JSON must be a list of documents, an object with a 'documents' list or an {{id: text}} mapping")
                for i, item in enumerate(items, 1):
                    if isinstance(item, str):
                        docs.append({"id": f"{f.stem}-{i}", "text": item, "source": f.name})
                    elif isinstance(item, dict):
                        docs.append({"id": str(item.get("id") or f"{f.stem}-{i}"), "text": str(item.get("text") or item.get("content") or ""), "source": item.get("source") or f.name, "metadata": item.get("metadata") or {}})
            else:
                docs.append({"id": f.stem, "text": f.read_text(encoding="utf-8", errors="replace"), "source": f.name})
        if len(docs) == before or all(not d["text"].strip() for d in docs[before:]):
            raise SystemExit(f"no documents found under {p}")
    return docs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pipeline_rag", description="Retrieval-augmented generation with routing, grading, citations and verification.")
    ap.add_argument("task", help="The question or task in natural language")
    ap.add_argument("--domain", default="general", help=f"DomainProfile name (one of: {', '.join(list_profiles())})")
    ap.add_argument("--offline", action="store_true", help="Use the deterministic MockLLM and heuristic stages")
    ap.add_argument("--json", action="store_true", help="Print the full result as JSON")
    ap.add_argument("--input", action="append", default=[], metavar="KEY=VALUE", help="Structured input (JSON value or string); repeatable")
    ap.add_argument("--documents", action="append", default=[], metavar="PATH", help="File, JSON list or directory of documents; repeatable")
    ap.add_argument("--sample-corpus", action="store_true", help="Use the built-in synthetic engineering corpus (demo)")
    ap.add_argument("--k", type=int, default=None, help="Override the number of evidence units kept")
    ap.add_argument("--max-calls", type=int, default=None, help="LLM call budget (Constraints.max_llm_calls)")
    ap.add_argument("--llm-mode", choices=["auto", "always", "never"], default=None, help="Force LLM or heuristic stages")
    ap.add_argument("--show-evidence", action="store_true", help="Print the evidence shown to the generator")
    ap.add_argument("--trace", action="store_true", help="Print the span tree after the result")
    ap.add_argument("--evaluate", action="store_true", help="Also run the offline evaluation harness over the documents (sample QA set or synthetic questions)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.offline:
        os.environ["PIPELINE_OFFLINE"] = "1"
    inputs = _parse_inputs(args.input)
    if args.documents:
        inputs["documents"] = load_documents(args.documents)
    if args.sample_corpus:
        from .evaluation import sample_corpus

        inputs["documents"] = sample_corpus()
    if args.k:
        inputs["k"] = args.k

    llm = MockLLM() if args.offline else make_llm()
    overrides: dict[str, Any] = {}
    if args.llm_mode:
        overrides["llm_mode"] = args.llm_mode
    model = PipelineRAGModel(llm, config=PipelineRAGConfig(**overrides))
    constraints = Constraints(max_llm_calls=args.max_calls) if args.max_calls is not None else Constraints()
    request = TaskRequest(task=args.task, inputs=inputs, domain=args.domain, constraints=constraints)
    result = model.run_sync(request)

    if args.json:
        payload = result.model_dump(mode="json")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result.summary())
        plan = result.metadata.get("plan") or {}
        if plan:
            print("plan:")
            for key in ("route", "query_transforms", "k", "max_hops", "grade_context", "compress", "verify", "citation_style", "expected_answer_type", "planner"):
                if key in plan:
                    print(f"  {key}: {plan[key]}")
            if plan.get("rationale"):
                print(f"  rationale: {plan['rationale']}")
        out = result.output
        if out is not None:
            routes = getattr(out, "routes_tried", None)
            if routes and len(routes) > 1:
                print(f"escalation: {' -> '.join(routes)}")
            if getattr(out, "hops", None) and len(out.hops) > 1:
                print("hops:")
                for h in out.hops:
                    print(f"  {h.hop}. {h.query[:80]!r} (+{h.new_evidence}, {h.triggered_by}{', ' + h.stop_reason if h.stop_reason else ''})")
        print(f"confidence: {result.confidence:.2f} - {result.confidence_basis}")
        if result.warnings:
            print("warnings:")
            for w in result.warnings:
                print(f"  - {w}")
        if result.errors:
            print("errors:")
            for e in result.errors:
                print(f"  - {e}")
        print("answer:")
        print(result.output_text or "(no answer)")
        if args.show_evidence and result.evidence:
            print("evidence:")
            for ev in result.evidence:
                print(f"  [{ev.id}] ({ev.source}, grade={ev.metadata.get('grade', '-')}) {ev.short(140)}")
        if args.trace:
            print(model.tracer.tree())
    if args.evaluate:
        from .evaluation import RAGEvaluator, sample_corpus, sample_qa, synthetic_qa

        docs = inputs.get("documents")
        if docs and not args.sample_corpus:
            examples = synthetic_qa(docs, n=10)
            for ex in examples:
                ex.domain = args.domain
        else:
            docs, examples = sample_corpus(), sample_qa()
        report = RAGEvaluator(model).evaluate_sync(examples, documents=docs)
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
        else:
            print("evaluation:", report.summary())
            print("routes:", report.route_distribution, "failure points:", report.failure_points or "none")
    return 0 if result.status.value != "failed" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
