"""Render the research bibliography from the dossiers in docs/research/.

Outputs
-------
* ``docs/RESEARCH_BIBLIOGRAPHY.md`` - index: one row per model/topic with source
  counts, verification counts and links.
* ``docs/bibliography/<concept>.md`` - the full, annotated source list for one
  concept plus its techniques, recommendations, pitfalls and metrics.

Usage (from the repository root)::

    python tools/build_bibliography.py            # uses docs/research/<concept>.json
    python tools/build_bibliography.py --drafts   # falls back to *.draft.json when a final file is missing

Each dossier is the structured output of the research workflow: a concept
summary, papers (title, authors, year, venue, url, arxiv_id, citation_signal,
key_contribution, design_implication, verification status), implementable
techniques with literature-backed defaults, architecture recommendations,
pitfalls and evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "docs" / "research"
OUTPUT = ROOT / "docs" / "RESEARCH_BIBLIOGRAPHY.md"
PER_CONCEPT_DIR = ROOT / "docs" / "bibliography"

ORDER = [
    ("hybrid-retrieval", "Hybrid Retrieval Model", "hybrid_retrieval_model/"),
    ("task-decomposition", "Task Decomposition Model", "task_decomposition_model/"),
    ("answer-verification", "Answer Verification Model", "answer_verification_model/"),
    ("parallelization", "Parallelization Model", "parallelization_model/"),
    ("caching", "Caching Model", "caching_model/"),
    ("pipeline-rag", "Pipeline RAG Model", "pipeline_rag_model/"),
    ("prompting-techniques", "Cross-cutting: Prompting Techniques", "all models"),
    ("pipeline-architecture", "Cross-cutting: Pipeline Architecture and Efficiency", "main README / orchestration"),
]

LEGEND = (
    "Verification legend: _verified_ = metadata confirmed against the arXiv record or venue page; "
    "_verified, metadata corrected_ = found, but title/author/year were adjusted to the authoritative record; "
    "_not independently verified_ = reported by the researcher and critic but not machine-checked "
    "(typically books, blogs, vendor documentation and non-arXiv venues)."
)


def load_dossier(key: str, allow_drafts: bool) -> tuple[dict | None, str]:
    verified = RESEARCH_DIR / f"{key}.json"
    draft = RESEARCH_DIR / f"{key}.draft.json"
    if verified.exists():
        return json.loads(verified.read_text(encoding="utf-8")), "final"
    if allow_drafts and draft.exists():
        return json.loads(draft.read_text(encoding="utf-8")), "draft"
    return None, "missing"


def badge(status: str) -> str:
    return {
        "verified": "verified",
        "verified-corrected": "verified, metadata corrected",
        "unverified": "not independently verified",
    }.get(status, status)


def fmt_paper(p: dict, idx: int) -> str:
    arxiv = f" arXiv:{p['arxiv_id']}" if p.get("arxiv_id") else ""
    b = badge(str(p.get("verification", "")))
    lines = [
        f"{idx}. **{p.get('authors', '')} ({p.get('year', '')}).** *{p.get('title', '')}.* {p.get('venue', '')}.{arxiv}  ",
        f"   <{p.get('url', '')}>" + (f"  _({b})_" if b else ""),
        f"   - Why it is highly rated: {p.get('citation_signal', '')}",
        f"   - Key contribution: {p.get('key_contribution', '')}",
        f"   - Design implication: {p.get('design_implication', '')}",
    ]
    if p.get("verification_evidence"):
        lines.append(f"   - Verification evidence: {p['verification_evidence']}")
    return "\n".join(lines)


def fmt_technique(t: dict) -> str:
    srcs = "; ".join(t.get("source_titles", []))
    return (
        f"- **{t.get('name', '')}** - {t.get('description', '')}\n"
        f"  - When to use: {t.get('when_to_use', '')}\n"
        f"  - Trade-offs: {t.get('tradeoffs', '')}\n"
        f"  - Literature defaults: {t.get('default_parameters', '')}\n"
        f"  - Sources: {srcs}"
    )


def build(allow_drafts: bool) -> tuple[str, dict[str, str]]:
    """Return (index markdown, {concept key: per-concept markdown})."""
    index: list[str] = ["# Research Bibliography", ""]
    index.append(
        "This bibliography is generated from the research dossiers in `docs/research/` by "
        "`tools/build_bibliography.py`. Each dossier was produced by a staged workflow: a literature "
        "researcher with web access compiled the sources, a completeness critic added missing seminal work "
        "and recorded corrections, and citation verification checked entries against arXiv metadata (and, "
        "where run, against venue pages). The per-model files list every source with its verification "
        "status; model READMEs cite only works listed there."
    )
    index.append("")
    rows = [
        "| Model / topic | Sources | Verified | Dossier | Full list |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    statuses: list[str] = []
    per_concept: dict[str, str] = {}
    total = 0
    total_verified = 0
    for key, title, used_by in ORDER:
        dossier, status = load_dossier(key, allow_drafts)
        statuses.append(f"- `{key}`: {status}")
        if dossier is None:
            continue
        papers = dossier.get("papers", [])
        verified = sum(1 for p in papers if str(p.get("verification", "")).startswith("verified"))
        total += len(papers)
        total_verified += verified
        rows.append(f"| {title} | {len(papers)} | {verified} | {status} | [bibliography/{key}.md](bibliography/{key}.md) |")

        sec: list[str] = [f"# {title}", ""]
        sec.append(f"_Used by: {used_by}._  Dossier: `docs/research/{key}.json` ({status}). Back to the [index](../RESEARCH_BIBLIOGRAPHY.md).")
        sec += ["", "## Summary", "", dossier.get("summary", ""), "", "## Sources", "", LEGEND, ""]
        for i, p in enumerate(papers, 1):
            sec.append(fmt_paper(p, i))
            sec.append("")
        if dossier.get("dropped_papers"):
            sec += ["## Dropped during verification", ""]
            for d in dossier["dropped_papers"]:
                sec.append(f"- {d.get('title', '')}: {d.get('reason', '')}")
            sec.append("")
        if dossier.get("critique_corrections"):
            sec += ["## Corrections recorded by the completeness critic", ""]
            for c in dossier["critique_corrections"]:
                sec.append(f"- {c}")
            sec.append("")
        sec += ["## Implementable techniques", ""]
        for t in dossier.get("techniques", []):
            sec.append(fmt_technique(t))
            sec.append("")
        for heading, field in (
            ("Architecture recommendations", "architecture_recommendations"),
            ("Documented pitfalls", "pitfalls"),
            ("Evaluation metrics", "evaluation_metrics"),
            ("Engineering adaptation notes", "engineering_adaptation_notes"),
            ("Composition notes", "composition_notes"),
        ):
            sec += [f"## {heading}", ""]
            for r in dossier.get(field, []):
                sec.append(f"- {r}")
            sec.append("")
        per_concept[key] = "\n".join(sec)

    index.append(f"Total sources: {total}; machine-verified: {total_verified}.")
    index.append("")
    index += rows
    index += ["", "Dossier files:", ""]
    index += statuses
    index += [
        "",
        "## How to cite from a README",
        "",
        "Use `Author et al. (Year), *Title*, URL` and make sure the title appears in the model's file above. "
        "`python tools/check_citations.py` cross-checks every README against the dossiers.",
        "",
    ]
    return "\n".join(index), per_concept


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the research bibliography from docs/research dossiers.")
    ap.add_argument("--drafts", action="store_true", help="allow *.draft.json when a final dossier is missing")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    text, per_concept = build(args.drafts)
    args.output.write_text(text, encoding="utf-8")
    PER_CONCEPT_DIR.mkdir(parents=True, exist_ok=True)
    for key, body in per_concept.items():
        (PER_CONCEPT_DIR / f"{key}.md").write_text(body, encoding="utf-8")
    print(f"wrote {args.output} ({len(text):,} chars) and {len(per_concept)} files under {PER_CONCEPT_DIR}")


if __name__ == "__main__":
    main()
