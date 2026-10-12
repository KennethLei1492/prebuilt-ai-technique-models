"""Machine-verify dossier citations against arXiv metadata.

Two modes:

``fetch``  - query the arXiv API for every arXiv identifier referenced by the
            dossiers (politely: small batches, long pauses, retries on 429) and
            store the authoritative title / first author / year in a lookup file.
``apply``  - compare every dossier paper that carries an arXiv id with the
            lookup and set its ``verification`` field:

            * ``verified``            - title similarity >= 0.85 and year within one
            * ``verified-corrected``  - the id resolves to the same work but title or
                                        year differed; dossier metadata is corrected
            * ``unverified``          - no lookup entry, or the id resolves to a
                                        different work (the stale id is kept in
                                        ``verification_evidence`` for a human to fix)

Usage::

    python tools/verify_arxiv.py fetch   [--lookup PATH]
    python tools/verify_arxiv.py apply   [--lookup PATH] [--dry-run]

Non-arXiv sources (books, blogs, vendor documentation, journal-only papers) are
left ``unverified`` by this tool; they need a human or agent check against the
venue page. The bibliography marks them explicitly.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "docs" / "research"
DEFAULT_LOOKUP = ROOT / "docs" / "research" / "arxiv_lookup.json"
NS = {"a": "http://www.w3.org/2005/Atom"}
ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def final_dossiers() -> list[Path]:
    return sorted(p for p in RESEARCH_DIR.glob("*.json") if not p.name.endswith(".draft.json") and p.name != "arxiv_lookup.json")


def paper_arxiv_id(p: dict) -> str | None:
    m = ID_RE.search(p.get("arxiv_id") or "") or re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", p.get("url") or "")
    return m.group(1) if m else None


def cmd_fetch(lookup_path: Path, batch_size: int, pause: float) -> None:
    ids: set[str] = set()
    for f in final_dossiers():
        for p in json.loads(f.read_text(encoding="utf-8")).get("papers", []):
            aid = paper_arxiv_id(p)
            if aid:
                ids.add(aid)
    found: dict[str, dict] = json.loads(lookup_path.read_text(encoding="utf-8")) if lookup_path.exists() else {}
    todo = [x for x in sorted(ids) if x not in found]
    print(f"{len(ids)} arXiv ids referenced; {len(found)} already resolved; fetching {len(todo)}")
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        url = f"https://export.arxiv.org/api/query?id_list={','.join(batch)}&max_results={len(batch)}"
        data = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "prebuilt-ai-models-bibliography-check/1.0"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                wait = 45 * (attempt + 1)
                print(f"  batch {i // batch_size}: HTTP {e.code}; waiting {wait}s")
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001
                print(f"  batch {i // batch_size}: {e}; waiting 30s")
                time.sleep(30)
        if data is None:
            print(f"  batch {i // batch_size}: giving up")
            continue
        root = ET.fromstring(data)
        for entry in root.findall("a:entry", NS):
            m = ID_RE.search(entry.findtext("a:id", default="", namespaces=NS))
            title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=NS)).strip()
            if not m or title.lower().startswith("error"):
                continue
            authors = [a.findtext("a:name", default="", namespaces=NS) for a in entry.findall("a:author", NS)]
            found[m.group(1)] = {
                "title": title,
                "first_author": authors[0] if authors else "",
                "n_authors": len(authors),
                "year": entry.findtext("a:published", default="", namespaces=NS)[:4],
            }
        lookup_path.write_text(json.dumps(found, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  batch {i // batch_size}: resolved {len(found)} total")
        time.sleep(pause)
    print(f"done: {len(found)} of {len(ids)} resolved -> {lookup_path}")


def cmd_apply(lookup_path: Path, dry_run: bool) -> None:
    lookup: dict[str, dict] = json.loads(lookup_path.read_text(encoding="utf-8"))
    grand = {"verified": 0, "verified-corrected": 0, "unverified-nonarxiv": 0, "unverified-missing": 0, "unverified-mismatch": 0}
    for f in final_dossiers():
        d = json.loads(f.read_text(encoding="utf-8"))
        counts = dict.fromkeys(grand, 0)
        for p in d.get("papers", []):
            aid = paper_arxiv_id(p)
            if not aid:
                p["verification"] = "unverified"
                p.setdefault("verification_evidence", "no arXiv identifier; needs venue-page check")
                counts["unverified-nonarxiv"] += 1
                continue
            rec = lookup.get(aid)
            if rec is None:
                p["verification"] = "unverified"
                p["verification_evidence"] = f"arXiv {aid} not resolved by the metadata lookup"
                counts["unverified-missing"] += 1
                continue
            sim = similarity(p.get("title", ""), rec["title"])
            try:
                year_ok = abs(int(p.get("year") or 0) - int(rec["year"] or 0)) <= 1
            except ValueError:
                year_ok = False
            if sim >= 0.85 and year_ok:
                p["verification"] = "verified"
                p["verification_evidence"] = f"arXiv {aid}: title match {sim:.2f}, year {rec['year']}"
                counts["verified"] += 1
            elif sim >= 0.6:
                changes = []
                if sim < 0.97:
                    changes.append(f"title '{p.get('title')}' -> '{rec['title']}'")
                    p["title"] = rec["title"]
                if not year_ok:
                    changes.append(f"year {p.get('year')} -> {rec['year']}")
                    p["year"] = int(rec["year"])
                p["verification"] = "verified-corrected"
                p["verification_evidence"] = f"arXiv {aid}: " + ("; ".join(changes) or f"title match {sim:.2f}")
                counts["verified-corrected"] += 1
            else:
                p["verification"] = "unverified"
                p["verification_evidence"] = (
                    f"arXiv {aid} resolves to a different work ('{rec['title']}', {rec['year']}); id is probably wrong"
                )
                counts["unverified-mismatch"] += 1
        for k, v in counts.items():
            grand[k] += v
        d["_status"] = (
            "research + completeness critique; arXiv metadata verification applied "
            f"({counts['verified'] + counts['verified-corrected']} of {len(d.get('papers', []))} machine-verified; "
            "non-arXiv sources need a venue-page check)"
        )
        print(f"{f.stem}: {counts}")
        if not dry_run:
            f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total:", grand)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    f.add_argument("--batch-size", type=int, default=25)
    f.add_argument("--pause", type=float, default=12.0)
    a = sub.add_parser("apply")
    a.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    a.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "fetch":
        cmd_fetch(args.lookup, args.batch_size, args.pause)
    else:
        cmd_apply(args.lookup, args.dry_run)


if __name__ == "__main__":
    main()
