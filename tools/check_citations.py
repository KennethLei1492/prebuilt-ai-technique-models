"""Cross-check README citations against the research dossiers.

For every ``*.md`` under the model folders (plus ``docs/COMPOSITION_PATTERNS.md``
and the root ``README.md``), find citation-like fragments - italicised titles
(``*Title*``) on lines that look like citations, arXiv URLs and arXiv
identifiers - and confirm each one appears in some dossier in
``docs/research/``. Titles are matched case-insensitively after normalising
punctuation; a title counts as present if it equals a dossier title, is a
prefix of one (or vice versa), or is at least 85% similar after stripping a
trailing alias such as "(RankGPT)".

Exit status is non-zero when an unknown citation is found, so this can run in CI::

    python tools/check_citations.py            # report
    python tools/check_citations.py --strict   # also fail on titles found only in draft dossiers
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "docs" / "research"
MODEL_FOLDERS = [
    "hybrid_retrieval_model",
    "task_decomposition_model",
    "answer_verification_model",
    "parallelization_model",
    "caching_model",
    "pipeline_rag_model",
]

# single-asterisk italics with non-space edges; never **bold**
ITALIC_TITLE = re.compile(r"(?<!\*)\*(?=\S)([^*\n]{12,200}?)(?<=\S)\*(?!\*)")
ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
ARXIV_ID = re.compile(r"\barXiv:(\d{4}\.\d{4,5})")
# a line must look like a citation (year in parentheses, "et al.", a URL or arXiv) before its italics are checked
CITATION_LINE = re.compile(r"\(\d{4}[a-z]?\)|et al\.|https?://|arXiv")
# strip trailing aliases such as "(RankGPT)" or "(vLLM)"
ALIAS_SUFFIX = re.compile(r"\s*\((?:[A-Za-z0-9+\-/ .]{2,40})\)\s*$")


def norm(s: str) -> str:
    s = s.lower().replace("×", "x")
    s = re.sub(r"[‘’“”'\"`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def load_known() -> tuple[set[str], set[str], set[str]]:
    final_titles: set[str] = set()
    draft_titles: set[str] = set()
    ids: set[str] = set()
    for f in RESEARCH_DIR.glob("*.json"):
        if f.name == "arxiv_lookup.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        target = draft_titles if f.name.endswith(".draft.json") else final_titles
        for p in d.get("papers", []):
            target.add(norm(p.get("title", "")))
            venue = p.get("venue") or ""
            if "also known as:" in venue:  # alternate titles recorded during verification
                target.add(norm(venue.split("also known as:", 1)[1]))
            aid = p.get("arxiv_id") or ""
            m = re.search(r"(\d{4}\.\d{4,5})", aid) or ARXIV_URL.search(p.get("url", ""))
            if m:
                ids.add(m.group(1))
    return final_titles, draft_titles, ids


def title_known(t: str, known: set[str]) -> bool:
    for candidate in (t, ALIAS_SUFFIX.sub("", t)):
        n = norm(candidate)
        if not n:
            continue
        if n in known:
            return True
        if len(n) < 12:
            continue
        for k in known:
            if k.startswith(n) or n.startswith(k):
                return True
            if difflib.SequenceMatcher(None, n, k).ratio() >= 0.85:
                return True
    return False


def scan(path: Path, final_titles: set[str], draft_titles: set[str], ids: set[str]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    problems: list[str] = []
    for m in ITALIC_TITLE.finditer(text):
        t = m.group(1).strip()
        if len(t.split()) < 2:
            continue
        line = text.count("\n", 0, m.start()) + 1
        if not CITATION_LINE.search(lines[line - 1]):
            continue  # italic emphasis in prose, not a citation
        if title_known(t, final_titles):
            continue
        if title_known(t, draft_titles):
            problems.append(f"{path.relative_to(ROOT)}:{line}: title only in DRAFT dossier: {t!r}")
        else:
            problems.append(f"{path.relative_to(ROOT)}:{line}: title not found in any dossier: {t!r}")
    for rx in (ARXIV_URL, ARXIV_ID):
        for m in rx.finditer(text):
            if m.group(1) not in ids:
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{path.relative_to(ROOT)}:{line}: arXiv id {m.group(1)} not in any dossier")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="fail on titles found only in draft dossiers")
    args = ap.parse_args()
    final_titles, draft_titles, ids = load_known()
    files = [ROOT / "README.md", ROOT / "docs" / "COMPOSITION_PATTERNS.md"]
    for folder in MODEL_FOLDERS:
        files += sorted((ROOT / folder).rglob("*.md"))
    problems: list[str] = []
    for f in files:
        if f.exists():
            problems += scan(f, final_titles, draft_titles, ids)
    hard = [p for p in problems if "not found" in p or "not in any dossier" in p]
    soft = [p for p in problems if p not in hard]
    for p in problems:
        print(p)
    print(f"\nchecked {len(files)} files: {len(hard)} unknown citation(s), {len(soft)} draft-only citation(s)")
    if hard or (args.strict and soft):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
