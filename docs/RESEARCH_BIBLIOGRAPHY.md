# Research Bibliography

This bibliography is generated from the research dossiers in `docs/research/` by `tools/build_bibliography.py`. Each dossier was produced by a staged workflow: a literature researcher with web access compiled the sources, a completeness critic added missing seminal work and recorded corrections, and citation verification checked entries against arXiv metadata (and, where run, against venue pages). The per-model files list every source with its verification status; model READMEs cite only works listed there.

Total sources: 456; machine-verified: 382.

| Model / topic | Sources | Verified | Dossier | Full list |
| --- | ---: | ---: | --- | --- |
| Hybrid Retrieval Model | 69 | 56 | final | [bibliography/hybrid-retrieval.md](bibliography/hybrid-retrieval.md) |
| Task Decomposition Model | 50 | 44 | final | [bibliography/task-decomposition.md](bibliography/task-decomposition.md) |
| Answer Verification Model | 67 | 64 | final | [bibliography/answer-verification.md](bibliography/answer-verification.md) |
| Parallelization Model | 55 | 39 | final | [bibliography/parallelization.md](bibliography/parallelization.md) |
| Caching Model | 37 | 22 | final | [bibliography/caching.md](bibliography/caching.md) |
| Pipeline RAG Model | 57 | 55 | final | [bibliography/pipeline-rag.md](bibliography/pipeline-rag.md) |
| Cross-cutting: Prompting Techniques | 68 | 61 | final | [bibliography/prompting-techniques.md](bibliography/prompting-techniques.md) |
| Cross-cutting: Pipeline Architecture and Efficiency | 53 | 41 | final | [bibliography/pipeline-architecture.md](bibliography/pipeline-architecture.md) |

Dossier files:

- `hybrid-retrieval`: final
- `task-decomposition`: final
- `answer-verification`: final
- `parallelization`: final
- `caching`: final
- `pipeline-rag`: final
- `prompting-techniques`: final
- `pipeline-architecture`: final

## How to cite from a README

Use `Author et al. (Year), *Title*, URL` and make sure the title appears in the model's file above. `python tools/check_citations.py` cross-checks every README against the dossiers.
