# Research dossiers

One JSON dossier per concept, produced by a three-stage literature workflow on
2026-10-11:

1. **Research** - an agent with web search and page fetch compiled sources for
   the concept (top-venue papers, highly cited preprints, authoritative
   engineering documentation), each with its key contribution and the design
   implication for a production implementation, plus implementable techniques
   with literature-backed default parameters, architecture recommendations,
   pitfalls, evaluation metrics, engineering-adaptation notes and composition
   notes.
2. **Critique** - a second agent audited the dossier for missing seminal work
   and factual errors, adding verified sources and recording corrections.
3. **Verify** - `tools/verify_arxiv.py` resolves every arXiv identifier against
   the arXiv API and compares title and year: matches are marked
   `verification: "verified"`, same-work-different-metadata entries are
   corrected and marked `"verified-corrected"`, and identifiers that resolve to
   a different work are flagged in `verification_evidence` for a human to fix.
   Sources without an arXiv identifier (books, standards, vendor documentation,
   journal-only papers) stay `"unverified"` until checked against their venue
   page; the bibliography labels them explicitly. `arxiv_lookup.json` is the
   cached metadata the verifier used.

Files:

| File | Meaning |
| --- | --- |
| `<concept>.json` | Final dossier after critique and verification. Source of truth for citations. |
| `<concept>.draft.json` | Snapshot after stage 1 only. Kept for provenance; superseded by the final file. |

Concepts: `hybrid-retrieval`, `task-decomposition`, `answer-verification`,
`parallelization`, `caching`, `pipeline-rag`, and the cross-cutting
`prompting-techniques` and `pipeline-architecture`.

`python tools/build_bibliography.py` renders all final dossiers into
[`../RESEARCH_BIBLIOGRAPHY.md`](../RESEARCH_BIBLIOGRAPHY.md).

Dossier schema (abridged):

```json
{
  "concept": "...",
  "summary": "state-of-the-art summary",
  "papers": [{"title", "authors", "year", "venue", "url", "arxiv_id", "citation_signal", "key_contribution", "design_implication", "verification"}],
  "techniques": [{"name", "description", "source_titles", "when_to_use", "tradeoffs", "default_parameters"}],
  "architecture_recommendations": ["..."],
  "pitfalls": ["..."],
  "evaluation_metrics": ["..."],
  "engineering_adaptation_notes": ["..."],
  "composition_notes": ["..."],
  "critique_corrections": ["..."],
  "dropped_papers": [{"title", "reason"}]
}
```
