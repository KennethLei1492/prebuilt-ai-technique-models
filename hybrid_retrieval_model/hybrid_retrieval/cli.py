"""Command-line entry point: ``python -m hybrid_retrieval "task" [options]``.

Examples::

    python -m hybrid_retrieval --offline "Find the clause that specifies the pump impeller material"
    python -m hybrid_retrieval "ERR-1203 cause" --domain software_engineering --documents docs/ --documents extra.md --json
    python -m hybrid_retrieval "corrosion allowance" --input k=5 --input 'filters={"type":"specification"}'
    python -m hybrid_retrieval --offline --evaluate "any task"      # metrics on the synthetic corpus

Without ``--documents`` or ``--load-index`` the CLI indexes the built-in
synthetic engineering corpus so the command always produces a meaningful result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from pipeline_core import Constraints, TaskRequest, make_llm, run_sync

from .evaluation import ablation, evaluate, synthetic_engineering_corpus
from .model import HybridRetrievalModel


def parse_input(item: str) -> tuple[str, Any]:
    """``key=value`` -> ``(key, value)``; the value is parsed as JSON when possible."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"--input expects key=value, got {item!r}")
    key, raw = item.split("=", 1)
    try:
        return key.strip(), json.loads(raw)
    except json.JSONDecodeError:
        return key.strip(), raw


def load_documents(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Load documents from .json (list), .jsonl, .txt/.md files or directories of them."""
    docs: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.suffix.lower() in (".txt", ".md", ".json", ".jsonl"))
        else:
            files = [p]
        for f in files:
            if f.suffix.lower() == ".json":
                data = json.loads(f.read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else data.get("documents", [data])
                for i, d in enumerate(items):
                    docs.append(d if isinstance(d, dict) else {"id": f"{f.stem}-{i + 1}", "text": str(d)})
            elif f.suffix.lower() == ".jsonl":
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                    if line.strip():
                        d = json.loads(line)
                        docs.append(d if isinstance(d, dict) else {"id": f"{f.stem}-{i + 1}", "text": str(d)})
            else:
                docs.append({"id": f.stem, "text": f.read_text(encoding="utf-8", errors="replace"), "metadata": {"title": f.stem, "source": str(f), "type": f.suffix.lstrip(".") or "text"}})
    return docs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m hybrid_retrieval", description="Hybrid (BM25 + dense) retrieval with fusion, reranking and query planning.")
    ap.add_argument("task", help="Natural-language retrieval task (used as the query unless --input query=... is given)")
    ap.add_argument("--domain", default="general", help="DomainProfile name (general, mechanical_engineering, software_engineering, ...)")
    ap.add_argument("--offline", action="store_true", help="Force the deterministic MockLLM (heuristic planner, no network)")
    ap.add_argument("--json", action="store_true", help="Print the full TaskResult as JSON")
    ap.add_argument("--input", action="append", default=[], metavar="KEY=VALUE", help="Request input (JSON values allowed): k=5, query=..., filters={...}")
    ap.add_argument("--documents", action="append", default=[], metavar="PATH", help="File or directory to index (.json/.jsonl/.txt/.md); repeat the flag for several paths")
    ap.add_argument("--load-index", metavar="PATH", help="Load an index saved with --save-index")
    ap.add_argument("--save-index", metavar="PATH", help="Save the index after the run")
    ap.add_argument("--k", type=int, default=None, help="Number of passages to return (overrides config k_final)")
    ap.add_argument("--max-llm-calls", type=int, default=None, help="Budget: maximum LLM calls for the run")
    ap.add_argument("--deterministic", action="store_true", help="Heuristic plan, no LLM transforms or reranking")
    ap.add_argument("--evaluate", action="store_true", help="Also run the evaluation harness (synthetic corpus only) and print nDCG/recall with a channel ablation")
    ap.add_argument("--trace", action="store_true", help="Print the span tree after the run")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline:
        os.environ["PIPELINE_OFFLINE"] = "1"
    llm = make_llm()
    model = HybridRetrievalModel(llm, domain=args.domain)

    judgments = None
    note = ""
    if args.load_index:
        n = model.load_index(args.load_index)
        note = f"loaded index from {args.load_index} ({n} chunks)"
    if args.documents:
        docs = load_documents(args.documents)
        n = model.add_documents(docs)
        note = (note + "; " if note else "") + f"indexed {len(docs)} documents ({n} chunks)"
    if model.retriever.n_chunks == 0:
        docs, judgments = synthetic_engineering_corpus()
        model.index(docs)
        note = f"no documents given: indexed the built-in synthetic engineering corpus ({len(docs)} documents)"

    inputs: dict[str, Any] = dict(parse_input(item) for item in args.input)
    if args.k is not None:
        inputs["k"] = args.k
    constraints = Constraints(max_llm_calls=args.max_llm_calls, deterministic=args.deterministic)
    request = TaskRequest(task=args.task, inputs=inputs, domain=args.domain, constraints=constraints)
    result = model.run_sync(request)

    if args.save_index:
        model.save_index(args.save_index)

    if args.json:
        payload = result.model_dump(mode="json")
        payload["note"] = note
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        if note:
            print(f"note: {note}")
        print(result.summary())
        print(result.output_text)
        if result.errors:
            print("Errors: " + " | ".join(result.errors))
        if args.trace:
            print(model.tracer.tree())

    if args.evaluate:
        if judgments is None:
            docs, judgments = synthetic_engineering_corpus()
            print("evaluation uses the synthetic corpus (your documents have no judgments)")
            model.index(docs)
        report = run_sync(evaluate(model, judgments, k=10))
        reports = run_sync(ablation(model, judgments, k=10))
        print("\nEvaluation (document level, k=10):")
        print("  " + report.summary().replace("\n", "\n  "))
        for name in ("sparse", "dense", "hybrid"):
            print("  " + reports[name].summary().splitlines()[0])
    return 0 if result.status.value != "failed" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
