# Prebuilt AI Technique Models

Six research-grounded, production-oriented AI pipeline models that work alone
or together, built on one shared contract so they compose into a single
efficient pipeline:

| Model | Folder | What it does | Protocol it provides |
| --- | --- | --- | --- |
| **Hybrid Retrieval** | [`hybrid_retrieval_model/`](hybrid_retrieval_model/README.md) | BM25 + dense (+ learned-sparse hooks) retrieval, Reciprocal Rank Fusion or convex fusion, LLM/cross-encoder reranking, HyDE and multi-query expansion, metadata filtering, nDCG/recall evaluation | `Retriever` |
| **Task Decomposition** | [`task_decomposition_model/`](task_decomposition_model/README.md) | Turns a complex prompt into a validated DAG of sub-tasks (Least-to-Most, Plan-and-Solve, LLMCompiler-style DAG, ADaPT recursive, Tree-of-Thoughts), schedules waves, re-plans on failure, synthesises with traceability | orchestrator; consumes `SubTaskSolver`, `Executor` |
| **Answer Verification** | [`answer_verification_model/`](answer_verification_model/README.md) | Self-consistency, Chain-of-Verification, claim-level grounding (FActScore-style), rubric judging with position-bias control, domain rule checks (units, required sections), code execution, calibrated confidence | `Verifier` |
| **Parallelization** | [`parallelization_model/`](parallelization_model/README.md) | Bounded concurrency, token-bucket rate limiting, jittered retries, circuit breaker, failure policies, DAG wave execution, skeleton-of-thought, map-reduce, ensembles, cache-aware fan-out, speedup estimation | `Executor` |
| **Caching** | [`caching_model/`](caching_model/README.md) | Exact-match and semantic response caches with TTL/LRU/tags, request coalescing, Anthropic prompt-prefix cache advisor (placement, TTL choice, pre-warming, hit verification), dependency-based invalidation | `ResponseCache`; wraps any LLM client |
| **Pipeline RAG** | [`pipeline_rag_model/`](pipeline_rag_model/README.md) | End-to-end retrieval-augmented generation: complexity routing (Adaptive-RAG), query transformation, CRAG-style context grading, compression and lost-in-the-middle ordering, cited generation, Self-RAG reflection, multi-hop iteration, RAGAS-style evaluation | consumes `Retriever`, `Verifier`, `Executor`, `ResponseCache` |

Everything sits on [`common/pipeline_core`](common/README.md): the request and
result contracts, the Claude client (with a deterministic offline mock),
tracing, budgets, prompt templates and domain profiles. The binding
specification is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the way the
models fit together is in [`docs/COMPOSITION_PATTERNS.md`](docs/COMPOSITION_PATTERNS.md);
the literature behind every design decision is in
[`docs/RESEARCH_BIBLIOGRAPHY.md`](docs/RESEARCH_BIBLIOGRAPHY.md).

---

## Why this design

1. **Every model reasons before it acts.** Each one implements `plan(request)`:
   a single structured LLM call (or a heuristic planner offline) that reads the
   natural-language task and chooses a strategy, sample counts, retrievers,
   parallelism or cache tiers. The plan is attached to the result so decisions
   are auditable. This is the "reason about the prompt, then execute" behaviour
   the models were asked to have.
2. **One contract, six implementations.** All models consume a `TaskRequest`
   and return a `TaskResult` with typed `output`, `output_text`, `evidence`,
   `confidence`, `confidence_basis`, `steps`, `usage` and a `status`
   (`SUCCESS`, `PARTIAL`, `NEEDS_REVIEW`, `FAILED`, `CACHED`). Orchestrators
   branch on status and roll up usage without adapters.
3. **Composition through protocols, not imports.** A model never imports
   another. It accepts a `Retriever`, `Verifier`, `Executor` or
   `ResponseCache`. The orchestrator wires concrete instances; tests inject
   fakes. Any model can be swapped for a vendor product that satisfies the
   same protocol.
4. **Adaptable to any engineering discipline by configuration.**
   `TaskRequest.domain` names a `DomainProfile` carrying glossary, units,
   document types, verification rules, standards, risk level and retrieval
   weights. Eight profiles ship (general, software, mechanical, electrical,
   civil/structural, systems, data, chemical/process engineering); registering
   a new one changes retrieval weights, verification rules and acceptance
   thresholds across all six models with no code change.
5. **Research-grounded defaults.** Every tunable has a default justified by a
   verified source: RRF `k=60` (Cormack et al., 2009), convex fusion weights in
   the 0.6-0.8 range (Bruch et al., 2023), BM25 `k1=1.2, b=0.75` (Robertson &
   Zaragoza, 2009), self-consistency sample counts (Wang et al., 2022),
   lost-in-the-middle context ordering (Liu et al., 2024), bounded
   self-correction only with external signals (Huang et al., 2024), prompt
   cache placement and TTL economics (Anthropic documentation). The dossiers
   that back these are in `docs/research/`.
6. **Runs offline, measures everything.** Without credentials the shared
   `MockLLM` produces schema-conformant outputs, so demos, tests and CI need no
   secrets. With credentials, every call goes through one traced, budgeted
   client; `tracer.tree()` shows the whole pipeline and `result.usage.cost_usd`
   is the real roll-up.

---

## Quick start

```bash
pip install -e ".[dev,retrieval]"      # installs pipeline_core and all six packages
python -m pytest -q                    # every suite, fully offline
python tools/quality_gate.py           # tests + CLIs + examples + static API-rule checks
```

Run any model from the command line without credentials. After `pip install -e .`
every package is importable, so no path setup is needed:

```bash
python -m pipeline_rag --offline --domain mechanical_engineering "What corrosion allowance does the specification require for the pump casing?"
```

Without installing, put `common` and the model folder on `PYTHONPATH`. Python
separates entries with `:` on Linux and macOS but with `;` on Windows, even
inside Git Bash:

```bash
PYTHONPATH="common;pipeline_rag_model" python -m pipeline_rag --offline "What corrosion allowance does the specification require for the pump casing?"
```

Set `ANTHROPIC_API_KEY` (or sign in with `ant auth login`) to run against
Claude. The default model is `claude-opus-5-5` with adaptive thinking and
`effort="high"`; override with `PIPELINE_MODEL` / `PIPELINE_EFFORT`.

Run all six together on a built-in pump-station corpus. A simple factual
question is routed straight to RAG, and a four-part design review is routed
through decomposition:

```bash
python examples/enterprise_pipeline.py --offline
```

The report shows each stage's status, confidence, citations, cache and
executor statistics, and the full span tree across all seven components.

---

## The models at a glance

Each model exposes the same two surfaces: `run(request)` (full life cycle with
plan, tracing, budgets) and a direct API used by the other models.

```python
from pipeline_core import TaskRequest, Tracer, make_llm
from hybrid_retrieval import HybridRetrievalModel
from task_decomposition import TaskDecompositionModel
from answer_verification import AnswerVerificationModel
from parallelization import ParallelizationModel
from caching import CachingModel
from pipeline_rag import PipelineRAGModel

tracer = Tracer()
llm = CachingModel(tracer=tracer).wrap(make_llm())          # exact + semantic + prefix-cache-aware client

retriever = HybridRetrievalModel(llm, tracer=tracer)        # Retriever
retriever.index(documents)                                  # list[dict(id, text, metadata)] or list[str]

verifier = AnswerVerificationModel(llm, tracer=tracer)      # Verifier
executor = ParallelizationModel(llm, tracer=tracer).executor  # Executor

rag = PipelineRAGModel(llm, retriever=retriever, verifier=verifier, executor=executor, tracer=tracer)
planner = TaskDecompositionModel(llm, solver=rag.solver, executor=executor, verifier=verifier, tracer=tracer)

result = planner.run_sync(TaskRequest(
    task="Review the pump station design: check hydraulic sizing, verify the motor rating against the duty point, and list open risks.",
    inputs={"documents": documents},
    domain="mechanical_engineering",
))
print(result.status, result.confidence, result.confidence_basis)
print(result.output_text)
print(tracer.tree())
```

| Model | `run()` answers | Direct API |
| --- | --- | --- |
| Hybrid Retrieval | "find the evidence for X" | `retrieve(query, k=10, filters=None) -> list[Evidence]`, `index(documents)` |
| Task Decomposition | "do this multi-part task" | `decompose(task) -> TaskGraph`; `solver=` and `executor=` injection |
| Answer Verification | "is this answer right?" (`inputs["answer"]`) | `verify(answer, task=..., evidence=...) -> Verdict` |
| Parallelization | "do these N things concurrently" | `executor.map(fn, items)`, `executor.dag(graph, fn)`, `patterns.*` |
| Caching | "is this prompt cacheable, and how?" | `wrap(llm) -> CachedLLMClient`, `cache.get/set/invalidate` |
| Pipeline RAG | "answer this from the corpus with citations" | `answer(question, documents=None) -> TaskResult` |

---

## How the models work hand in hand

```
request ─▶ cache (exact → semantic → coalesce) ─▶ complexity router
            │                                        │ simple          │ complex
            │                                        ▼                 ▼
            │                              Pipeline RAG         Task Decomposition (DAG)
            │                                 │  ▲                     │ waves via Parallelization
            │                                 │  │ Retriever           ▼
            │                                 │  └── Hybrid Retrieval  Pipeline RAG per sub-task
            │                                 │                        │
            │                                 ▼                        ▼
            │                              Answer Verification ◀── synthesis
            │                                 │
            └──────── store ◀─────────────────┘
                                   result (status, confidence, evidence, usage, trace)
```

- The **cache** is outermost: the same `CachedLLMClient` is handed to every
  model, so repeated planning calls, reranking calls and verification calls are
  served from the exact or semantic tier, and the prefix advisor keeps every
  stable system prompt cache-friendly for Claude's own prompt cache.
- **Routing** uses the decomposition planner's complexity estimate: simple
  questions skip decomposition and go straight to RAG.
- **Decomposition** emits a DAG; **parallelization** executes each wave of
  independent sub-tasks concurrently under rate limits with retries and a
  circuit breaker; each sub-task is a **RAG** run, which pulls evidence from
  **hybrid retrieval** and asks **verification** to grade grounding before
  returning.
- **Verification** gates the synthesised answer; below the domain's acceptance
  threshold the result is `NEEDS_REVIEW` instead of `SUCCESS`.
- A single **Tracer** gives one span tree and one cost roll-up for the run.

Patterns, recipes and the literature behind each edge are in
[`docs/COMPOSITION_PATTERNS.md`](docs/COMPOSITION_PATTERNS.md). The complete,
runnable wiring is [`examples/enterprise_pipeline.py`](examples/enterprise_pipeline.py).
Its `EnterprisePipeline` class is the recommended starting point for a
deployment:

```python
import sys; sys.path.insert(0, "examples")                   # run from the repository root
from enterprise_pipeline import EnterprisePipeline

pipe = EnterprisePipeline(domain="mechanical_engineering")   # uses make_llm(): Claude with credentials, MockLLM without
pipe.index(documents)
result = pipe.run_sync("Check the NPSH margin against API 610 and list the open risks from the trip report.")
print(result.status, result.confidence)
print(result.output_text)
print(pipe.tracer.tree())
```

It adds thin adapters on top of the packages, each documented in the file:

| Adapter | Why it exists |
| --- | --- |
| `ReentrantExecutor` | Gives each nesting level of fan-out (decomposition waves, then retrieval and verification inside each sub-task) its own bulkhead while sharing rate limits, retries and the circuit breaker, and adds a trace span. |
| `TracedRetriever` | Adds a span per retrieval and exposes the corpus version used in result-cache keys. |
| `GatedVerifier` | Strips citation markers and headings before verification and attaches the run's usage to the verdict. |
| `RAGSubTaskSolver` | Answers leaf sub-tasks with RAG and sends the decomposition's final `verify` node to the verifier. |
| `ComplexityRouter` | Uses the decomposition planner's step estimate to send simple questions straight to RAG. |

The pipeline also caches whole results keyed by task, domain and corpus
version. A repeated request is served without recomputation, and re-indexing
the corpus invalidates it.

---

## Adapting to any task or engineering field

Three levers, from cheapest to most specific:

1. **Domain profile** (`TaskRequest.domain`). Register a `DomainProfile` with
   your glossary, units, document types, verification rules, standards, risk
   level and retrieval weights. Every model reads it: retrieval shifts weight
   to lexical matching for identifier-heavy fields, verification applies your
   rules and unit checks, decomposition follows your hints, and acceptance
   thresholds tighten for safety-critical work.
2. **Model config** (`<Name>Config`). Each package has one pydantic config with
   literature-backed defaults documented in its README (fusion method, sample
   counts, thresholds, concurrency, TTLs, hops).
3. **Plan override**. Pass a pre-built plan (for example a `RetrievalPlan` or
   `RAGPlan`) through `request.inputs["plan"]` or call the direct API to bypass
   planning when the strategy is known.

```python
from pipeline_core import DomainProfile, register_profile
register_profile(DomainProfile(
    name="aerospace_structures",
    description="Airframe structural analysis and certification.",
    units=["kN", "MPa", "mm", "g"],
    verification_rules=["Every margin must state the limit and ultimate load cases.", "Cite the applicable FAR/CS paragraph."],
    standards=["FAR Part 25", "CS-25", "MMPDS"],
    risk_level="safety_critical",
    min_confidence_to_accept=0.9,
    retrieval_hints={"sparse_weight": 0.65},
))
```

---

## Repository layout

```
common/pipeline_core/          shared contracts, Claude client, mock, tracing, prompts, domain profiles, protocols
hybrid_retrieval_model/        package hybrid_retrieval   + README, tests, examples
task_decomposition_model/      package task_decomposition + README, tests, examples
answer_verification_model/     package answer_verification+ README, tests, examples
parallelization_model/         package parallelization    + README, tests, examples
caching_model/                 package caching            + README, tests, examples
pipeline_rag_model/            package pipeline_rag       + README, tests, examples
examples/enterprise_pipeline.py  all six wired together; examples/tests/ proves the wiring
docs/ARCHITECTURE.md           binding implementation contract
docs/COMPOSITION_PATTERNS.md   how to combine the models, with sources
docs/RESEARCH_BIBLIOGRAPHY.md  verified literature per model (generated)
docs/research/*.json           raw research dossiers (research -> critique -> verification)
tools/quality_gate.py          CI gate: tests, CLIs, examples, API-rule static checks
tools/build_bibliography.py    regenerates the bibliography from the dossiers
```

---

## Claude API conventions used throughout

- Default `claude-opus-5-5`; exact model IDs only. Cheaper sub-calls may use
  `claude-sonnet-5-5`/`claude-haiku-5-5` via `LLMRequest.model`, but the main
  loop stays on one model because prompt caches are model-scoped.
- Adaptive thinking is always on; depth is set with `output_config.effort`
  (`low` for routing/grading, `high` for planning/synthesis, `max` for final
  verification in safety-critical domains).
- No `temperature`/`top_p`/`top_k`; variation for self-consistency comes from
  independent calls with varied framing.
- Structured outputs via `output_config.format` with sanitised JSON schemas,
  validated with pydantic.
- Prompt caching: stable system blocks first with `cache_control` on the last
  one; retrieved documents and the question after it; no timestamps or ids in
  the prefix. The caching model's advisor audits this and recommends TTLs.
- Server-side refusal fallbacks are on by default; refusals and truncation are
  typed errors.

---

## Evaluation and quality

- `python -m pytest -q` runs every suite offline.
- `python tools/quality_gate.py` additionally runs every CLI and example and
  checks the static API rules from `docs/ARCHITECTURE.md` section 6.
- Each package's `evaluation.py` implements the metrics the literature uses for
  that concept (nDCG@k/recall@k/MRR; plan coverage and speedup; verification
  accuracy and Brier score; speedup, utilisation and retries; hit rate and cost
  saved; faithfulness, answer relevance, context precision/recall).
- Evaluate on a cost-accuracy frontier against simple baselines, as Kapoor et
  al. (2024) recommend; the tracer supplies the cost axis.

---

## How the research was done

Each concept has a dossier in `docs/research/` produced by a three-stage
workflow: a literature researcher with web access compiled top-venue papers,
highly cited preprints and authoritative engineering documentation (with each
source's key contribution and the design implication for production); a
completeness critic added missing seminal work and corrected claims; citation
verifiers confirmed every entry against its arXiv or venue page and dropped
anything unverifiable. READMEs cite only sources that survived verification.
`tools/build_bibliography.py` renders the dossiers into
`docs/RESEARCH_BIBLIOGRAPHY.md`.

## Limitations

- The offline `MockLLM` demonstrates control flow and shapes, not answer
  quality; quality claims apply to runs against Claude.
- The built-in `HashingEmbedder` is a dependency-free baseline; production
  dense retrieval should use a real embedding model (Voyage, sentence
  transformers) through the `Embedder` protocol.
- Semantic caching trades exactness for savings; keep it off for
  time-sensitive, personalised or numerically precise queries (the cache
  planner flags these, but the policy is yours).
- Confidence values are calibrated heuristically unless you fit calibration
  parameters on your own labelled data with each package's `evaluation.py`.
