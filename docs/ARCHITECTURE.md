# Architecture and Implementation Contract

This document is the binding specification for every model folder in the
repository. It exists so that six independently developed models compose
without adapters, behave identically under tracing and budgeting, and can be
re-targeted to any engineering domain by configuration alone.

Read it together with [`common/README.md`](../common/README.md) (the shared
package) and [`RESEARCH_BIBLIOGRAPHY.md`](RESEARCH_BIBLIOGRAPHY.md) (the
literature each design decision cites).

---

## 1. Repository layout

```
Prebuilt AI Technique Models/
├── README.md                       Main guide: what each model does, how they combine
├── pyproject.toml                  One installable distribution containing all packages
├── requirements.txt
├── common/                         pipeline_core - shared contracts (see common/README.md)
├── docs/
│   ├── ARCHITECTURE.md             This file
│   ├── RESEARCH_BIBLIOGRAPHY.md    Index of sources per model (generated)
│   ├── bibliography/<model>.md     Full annotated source lists (generated)
│   ├── COMPOSITION_PATTERNS.md     Recipes for combining models
│   └── research/<model>.json       Research dossiers with verification status
├── tools/
│   ├── quality_gate.py             CI gate: tests, CLIs, examples, static API-rule checks
│   ├── check_citations.py          README citations must exist in the dossiers
│   ├── verify_arxiv.py             Machine-verify dossier citations against arXiv
│   └── build_bibliography.py       Regenerate the bibliography from the dossiers
├── hybrid_retrieval_model/         package: hybrid_retrieval
├── task_decomposition_model/       package: task_decomposition
├── answer_verification_model/      package: answer_verification
├── parallelization_model/          package: parallelization
├── caching_model/                  package: caching
├── pipeline_rag_model/             package: pipeline_rag
└── examples/
    ├── enterprise_pipeline.py      All six models wired into one orchestrated pipeline
    └── tests/
```

Every `<name>_model/` folder has the same shape:

```
<name>_model/
├── README.md             How it is used, how it was built, the research behind each choice
├── <package>/
│   ├── __init__.py       Exports the model class, config, plan types; calls pipeline_core.register
│   ├── model.py          <Name>Model(PipelineComponent): spec, plan(), execute()
│   ├── config.py         <Name>Config(BaseModel): every tunable with literature-justified defaults
│   ├── plan.py           Pydantic plan schema(s) + HeuristicPlanner used when ctx.offline
│   ├── prompts.py        Versioned PromptTemplates (stable system first, volatile last)
│   ├── <technique>.py    One module per technique family
│   ├── evaluation.py     Metrics and an offline evaluation harness for the model
│   └── cli.py            python -m <package> "task" [--domain X] [--offline] [--json]
├── examples/
│   ├── basic_usage.py
│   └── engineering_example.py
└── tests/
    └── test_*.py         Offline (MockLLM / ScriptedLLM), fast, deterministic
```

Installation: `pip install -e .` from the repository root installs all seven
packages; `pytest` at the root runs every suite.

---

## 2. Core contracts (summary)

| Type | Role |
| --- | --- |
| `TaskRequest(task, inputs, context, constraints, domain, request_id, parent_id, metadata)` | What a component receives. `task` is natural language. `inputs` carries structured data (documents, code, parameters). `context` is upstream `Evidence`. |
| `TaskResult(request_id, component, status, output, output_text, evidence, confidence, confidence_basis, steps, usage, warnings, errors, metadata, trace_id)` | What a component returns. `output` is model-specific and typed; `output_text` is always filled. |
| `Evidence(id, content, source, score, rank, retriever, metadata, span)` | The one currency for retrieved/cited information. |
| `Constraints(max_cost_usd, max_latency_s, max_llm_calls, max_parallelism, min_confidence, allow_external_tools, allow_code_execution, deterministic)` | Enforced by `RunContext`. |
| `Usage` | Additive accounting (tokens, cache tokens, calls, cache hits, cost, latency). |
| `RunContext` | Per-run: `llm_call()`, `step()`, `charge()`, `warn()`, `profile`, `offline`, `budget`, `remaining_time()`. |
| `Verdict(method, passed, score, issues, corrected_answer, evidence_ids, details)` | Output of a verification method. |
| Protocols `Retriever`, `Verifier`, `Executor`, `ResponseCache`, `SubTaskSolver` | How models accept each other as dependencies. |

---

## 3. Rules every model must follow

### 3.1 Life cycle and reasoning

1. Subclass `PipelineComponent`; define `spec: ComponentSpec` with a stable `name` equal to the package name (`hybrid_retrieval`, `task_decomposition`, `answer_verification`, `parallelization`, `caching`, `pipeline_rag`). Decorate with `@register`.
2. `plan(request, ctx)` **reads the natural-language task and decides strategy**. It returns a pydantic plan model. It uses at most one structured LLM call (`ctx.llm_call` with `output_schema`) and must fall back to a `HeuristicPlanner` when `ctx.offline` is true or `request.constraints.deterministic` is set. The plan is attached to `result.metadata["plan"]` automatically.
3. `execute(request, plan, ctx)` carries the plan out. Wrap each logical phase in `with ctx.step("name")` so usage and timing are attributed. Call the LLM only through `ctx.llm_call` (budget + tracing). Nested components are invoked via their `run()`; charge their usage with `ctx.charge(result.usage)`.
4. Never raise out of `execute` for expected conditions (empty retrieval, verification failure, partial parallel failure); return `status=PARTIAL|NEEDS_REVIEW|FAILED` with `errors`/`warnings`. Raise `PipelineError` subclasses only for programming/configuration errors.
5. Always set `confidence` (0-1) and a human-readable `confidence_basis`. Compare against `ctx.profile.min_confidence_to_accept` and `request.constraints.min_confidence`; below either, set `status=NEEDS_REVIEW`.
6. Expose a **direct API** in addition to `run()` for programmatic use (for example `retriever.retrieve(query, k)`, `verifier.verify(answer, task=...)`, `executor.map(fn, items)`, `cache.get/set`). The direct API is what other models consume through the protocols.

### 3.2 LLM usage

- Build requests with `LLMRequest`; put `ctx.profile.system_text()` and the template's stable system text in `system_stable`; put retrieved documents and the question in the user turn (or `system_volatile`). Never interpolate timestamps, request IDs or random values into stable blocks.
- Use `output_schema` (sanitised JSON schema or pydantic `model_json_schema()` through `sanitize_schema`) for every machine-read answer. Validate with `Model.model_validate`.
- Choose `effort` per call: `low` for classification/routing/grading of many items, `high` for planning and synthesis, `max` only for final verification in safety-critical domains.
- Do not pass `temperature`; self-consistency obtains diversity by varying the prompt framing (persona, ordering, instruction wording) and by issuing independent calls.
- Model IDs: `claude-opus-5-5` default; cheaper sub-calls may use `claude-sonnet-5-5` or `claude-haiku-5-5` through `LLMRequest.model`, but keep the main loop on one model (caches are model-scoped).
- Respect `ctx.remaining_time()` and `ctx.budget.remaining_calls()` before fan-outs; degrade sample counts rather than fail.

### 3.3 Prompting

- Every prompt is a versioned `PromptTemplate` in `prompts.py`. Bump `version` when wording changes; caches key on the fingerprint.
- Structure: role and goal; rules (numbered); inputs in XML tags (`<documents>`, `<task>`, `<answer>`); explicit output specification. Use `render_evidence` for documents so the model can cite `[doc-id]`.
- Include `ENGINEERING_RIGOR` from `pipeline_core.prompts` when `ctx.profile.risk_level in ("high", "safety_critical")`.
- Keep few-shot examples in `PromptTemplate.examples` (rendered in the stable prefix).

### 3.4 Domain adaptation

- Read `ctx.profile` for: `retrieval_hints["sparse_weight"]` (hybrid retrieval), `verification_rules` and `units` (verification), `decomposition_hints` (decomposition), `risk_level`/`min_confidence_to_accept` (all), `code_languages` (code execution verifiers), `document_types` (chunking/indexing choices).
- Never hard-code a domain. A new `DomainProfile` registered by a user must change behaviour without code edits.

### 3.5 Testing and documentation

- Tests run offline with `MockLLM` (schema-conformant stubs) and `ScriptedLLM` (deterministic multi-step paths). Cover: plan heuristics, each technique module's pure logic (fusion maths, DAG validation, cache eviction, retry/backoff timing with a fake clock, aggregation), the end-to-end `run()` path, status/confidence behaviour, and domain adaptation.
- Each README must contain: purpose; architecture diagram (ASCII); how it reasons about a prompt (the plan schema and the decision rules); quick start (Python and CLI); configuration table with defaults **and the paper that justifies each default**; techniques implemented with citations; how to adapt to an engineering domain; how it composes with the other five models; evaluation (metrics + how to run the harness); limitations; "How it was built" (research -> design decisions).
- Cite sources as `Author et al. (Year), *Title*` with the arXiv or venue URL; cite only sources present in `docs/research/<model>.json`.

---

## 4. Per-model specifications

### 4.1 Hybrid Retrieval (`hybrid_retrieval`)

Purpose: find the most relevant evidence for a query from a document corpus by combining lexical, dense and (optionally) learned-sparse retrieval, fusing scores, and reranking.

Required modules and classes:

| Module | Contents |
| --- | --- |
| `index.py` | `Document(id, text, metadata)`, `Chunk(id, doc_id, text, metadata, span)`, `DocumentStore` (add/remove/filter by metadata, persistence to JSON), `Chunker` with strategies: fixed window with overlap, sentence-aware, structure-aware (markdown headings / code blocks / numbered clauses), token-budgeted. |
| `sparse.py` | `BM25Index` implementing Okapi BM25 (`k1=1.2`, `b=0.75`) with configurable tokenizer (lowercase, optional stemming hook, identifier-preserving tokenization for code/part numbers); BM25+ variant. Pure Python, no required deps; use `rank_bm25` if installed. |
| `dense.py` | `Embedder` protocol (`embed(texts) -> list[list[float]]`, `dimension`); `HashingEmbedder` (deterministic character-n-gram feature hashing, cosine; documented as offline baseline), `CallableEmbedder`, `VoyageEmbedder` (lazy import of `voyageai`), `SentenceTransformerEmbedder` (lazy). `DenseIndex` with brute-force cosine (numpy if present, pure Python fallback) and optional HNSW hook. |
| `fusion.py` | `reciprocal_rank_fusion(rankings, k=60)`, `convex_combination(score_lists, weights, normalization="minmax"|"zscore"|"theoretical")`, `distribution_based_score_fusion`, `interleave`; all pure functions over `list[list[(doc_id, score)]]`. |
| `rerank.py` | `Reranker` protocol; `LLMListwiseReranker` (RankGPT-style sliding window permutation generation with structured output), `LLMPointwiseReranker` (0-10 relevance with rationale, parallel), `CrossEncoderReranker` adapter (lazy import), `NoopReranker`. |
| `query_transform.py` | `MultiQueryExpander` (N paraphrases, fused by RRF), `HyDE` (hypothetical document generation then dense retrieval), `Query2Doc`, `KeywordExtractor` for the sparse leg; all accept a `RunContext` for budgeted LLM calls. |
| `retriever.py` | `HybridRetriever` implementing the `Retriever` protocol: runs sparse and dense legs concurrently, applies metadata filters, fuses, optionally reranks, returns `list[Evidence]` with `retriever` field set to `sparse|dense|fused|reranked` and `metadata` carrying per-leg scores and ranks. |
| `plan.py` | `RetrievalPlan(query_variants, use_sparse, use_dense, sparse_weight, dense_weight, fusion, k_sparse, k_dense, k_final, rerank, rerank_depth, filters, query_type, rationale)`; `HeuristicPlanner` using query features (length, identifier/number density, question words, domain `sparse_weight`). |
| `evaluation.py` | `ndcg_at_k`, `recall_at_k`, `mrr`, `precision_at_k`; `evaluate(retriever, qrels)` harness; a small synthetic engineering corpus generator for self-tests. |

Defaults (justify in README from the dossier): RRF `k=60`; convex `alpha` 0.5 general / domain `sparse_weight`; `k_sparse=k_dense=50`, `k_final=10`, `rerank_depth=20`; chunk 512 tokens / 64 overlap.

Direct API: `HybridRetrievalModel.retrieve(query, k=10, filters=None)`; `.index(documents)`; `.add_documents`.

### 4.2 Task Decomposition (`task_decomposition`)

Purpose: turn a complex task into an executable plan (sequence or DAG of sub-tasks with dependencies, inputs, expected outputs and verification criteria), schedule it, execute via a pluggable solver, re-plan on failure, and synthesise.

| Module | Contents |
| --- | --- |
| `graph.py` | `SubTask(id, description, depends_on, inputs, expected_output, verification_criteria, tool_hint, complexity, max_retries)`, `TaskGraph` (validation: acyclic, dependencies exist, depth/breadth bounds; topological waves; critical path length; `to_mermaid()`). |
| `strategies.py` | Strategy classes producing a `TaskGraph`: `LeastToMost` (sequential easy-to-hard chain), `PlanAndSolve` (single plan then solve), `DAGPlanner` (LLMCompiler-style parallel DAG with placeholder variable passing `$1.output`), `RecursiveDecomposer` (ADaPT: try solve; on failure or low confidence decompose further, depth bound), `TreeOfThoughtsPlanner` (bounded BFS over candidate decompositions with LLM state evaluation). |
| `scheduler.py` | `Scheduler` executing waves with an `Executor` protocol (default: bounded `asyncio.gather`), passing dependency outputs into sub-task inputs, honouring budget/time, per-task retries, and collecting `StepRecord`s. |
| `replanner.py` | `Replanner` (Reflexion-style: reflect on failures, produce revised graph; bounded iterations). |
| `synthesis.py` | `Synthesizer` combining sub-task results into the final answer with traceability (which sub-task produced what). |
| `plan.py` | `DecompositionPlan(strategy, max_depth, max_breadth, parallelism, verification, rationale, complexity_estimate)`; `HeuristicPlanner` (task length, conjunctions, enumerations, domain `decomposition_hints`). |
| `evaluation.py` | Plan quality metrics: coverage (LLM judge or keyword), dependency correctness, parallel speedup estimate (critical path / total), cost. |

Direct API: `TaskDecompositionModel.decompose(task) -> TaskGraph`; `.run(request)` with `solver: SubTaskSolver` injected (default: LLM solves each sub-task with evidence from `request.context`).

### 4.3 Answer Verification (`answer_verification`)

Purpose: decide whether an answer is correct, grounded and complete; estimate calibrated confidence; correct it when external signals allow.

| Module | Contents |
| --- | --- |
| `verifiers.py` | `BaseVerifier` + implementations of the `Verifier` protocol: `SelfConsistencyVerifier` (N independent re-derivations with varied framing, agreement via normalised match or LLM-selected consensus - Universal Self-Consistency), `ChainOfVerificationVerifier` (plan verification questions -> answer each independently without seeing the draft (factored) -> revise), `GroundingVerifier` (split answer into atomic claims -> classify each as supported/contradicted/unsupported against evidence -> FActScore-style supported fraction, citation precision/recall), `RubricJudgeVerifier` (G-Eval style criteria scoring with structured output; pairwise mode with position swapping to counter position bias), `RuleVerifier` (schema/regex/units/numeric-range/required-sections from domain `verification_rules`; unit consistency via a small unit parser), `CodeExecutionVerifier` (runs tests in a subprocess sandbox with timeout when `constraints.allow_code_execution`), `DebateVerifier` (optional, bounded rounds). |
| `uncertainty.py` | `verbalized_confidence` elicitation, `semantic_agreement` (cluster sampled answers by LLM-judged equivalence; entropy over clusters as uncertainty), calibration helpers (`platt_scale` placeholder with stored parameters), `aggregate_verdicts(verdicts, weights) -> (confidence, basis)`. |
| `policy.py` | `VerificationPolicy(verifiers, weights, thresholds, escalate_below, allow_revision, max_revision_rounds)` and domain-derived defaults (safety-critical -> grounding + rules mandatory, higher thresholds). |
| `plan.py` | `VerificationPlan(answer_type: factual|numerical|code|procedural|design|free_text, verifiers, samples, thresholds, rationale)`; `HeuristicPlanner` from task/answer features. |
| `evaluation.py` | Verification quality metrics: accuracy of pass/fail vs labels, Brier score / ECE for confidence, cost per verified answer. |

Design constraint from the literature: intrinsic self-correction is unreliable, so revisions are only accepted when backed by an external signal (evidence, executed tests, rules, or strong multi-sample agreement). Document this in the README.

Direct API: `AnswerVerificationModel.verify(answer, task=..., evidence=...) -> Verdict`; `run(request)` where `request.inputs["answer"]` is the candidate.

### 4.4 Parallelization (`parallelization`)

Purpose: execute LLM and tool work concurrently and safely: bounded concurrency, rate limiting, retries, circuit breaking, partial-failure policies, and the parallel reasoning patterns from the literature.

| Module | Contents |
| --- | --- |
| `executor.py` | `ParallelExecutor` implementing `Executor`: `map`, `gather`, `submit`, `dag(graph, fn)` (wave scheduling), with `max_concurrency`, `TokenBucket` rate limiter (requests/min and tokens/min), per-task timeout, `RetryPolicy` (exponential backoff with full jitter, retry only on retryable error classes), `CircuitBreaker` (closed/open/half-open), `Bulkhead` per resource, `FailurePolicy` = `fail_fast | best_effort | quorum(n)`, idempotency keys, cancellation propagation, progress callbacks. Clock injectable for tests. |
| `patterns.py` | `skeleton_of_thought(ctx, prompt)` (skeleton -> parallel point expansion -> merge), `branch_solve_merge`, `map_reduce(items, map_prompt, reduce_prompt, fanin)`, `ensemble` / `mixture_of_agents` (N proposers -> aggregator), `speculative(fast, slow, verifier)` (race a cheap path against an expensive one), `cache_aware_fan_out` (send one request, await its first response/cache write, then fire the rest - Anthropic prompt-cache concurrency guidance), `parallel_tool_calls` helper. |
| `estimator.py` | Amdahl/critical-path speedup estimates, cost-latency trade-off table for candidate concurrency levels, recommended `max_concurrency` from rate limits. |
| `plan.py` | `ParallelPlan(pattern, partitions, concurrency, failure_policy, rate_limits, rationale, expected_speedup)`; `HeuristicPlanner` (detect enumerations/lists/independent sub-questions; constraints.max_parallelism). |
| `evaluation.py` | Measured speedup, utilisation, p50/p95 latency, retries, breaker trips. |

Direct API: `ParallelizationModel.executor` (an `Executor`), `run(request)` for prompt-level parallel reasoning.

### 4.5 Caching (`caching`)

Purpose: avoid paying twice: exact-match response caching, semantic caching with safety guards, request coalescing, provider prompt-cache optimisation, and invalidation.

| Module | Contents |
| --- | --- |
| `backends.py` | `CacheBackend` ABC: `InMemoryBackend` (LRU/LFU/TTL with max entries and max bytes), `SQLiteBackend` (thread-safe, TTL, tags), `FileBackend`, `RedisBackend` (lazy import, same interface). Entries carry `created_at`, `ttl_s`, `tags`, `hits`, `version`. |
| `keys.py` | `exact_key(request, model, prompt_version)` from `LLMRequest.fingerprint` plus normalisation options (whitespace, case, stable JSON); `tagged_keys` for invalidation by document id / prompt template / model. |
| `semantic.py` | `SemanticCache` with pluggable `Embedder` (reuse `hybrid_retrieval.dense.Embedder` protocol shape but defined locally to avoid a package dependency), cosine threshold (default 0.95 for paraphrase-safe hits, configurable), optional LLM equivalence check on borderline hits (`verify_hits`), per-namespace isolation, metrics for false-hit risk. Document when semantic caching is unsafe (time-sensitive, personalised, numeric-precision queries). |
| `middleware.py` | `CachedLLMClient(BaseLLMClient)` wrapping any client: exact -> semantic -> in-flight coalescing (single flight for identical keys) -> inner call -> store; respects `Constraints.deterministic`; sets `LLMResponse.cached` and `Usage.cache_hits`; negative caching for deterministic failures. |
| `prefix.py` | `PrefixCacheAdvisor`: analyses an `LLMRequest`/`PromptTemplate` for silent invalidators (timestamps, UUIDs, unsorted JSON), checks the prefix against `MIN_CACHEABLE_TOKENS`, recommends TTL from observed inter-arrival gaps (5m vs 1h vs keep-alive), schedules `prewarm` through `AnthropicLLM.prewarm`, and verifies hits from `usage.cache_read_input_tokens`. |
| `invalidation.py` | Tag-based and version-based invalidation; `DependencyFingerprint` over retrieved document ids so RAG answers are invalidated when sources change. |
| `plan.py` | `CachePlan(cacheable, tiers, ttl_s, semantic_threshold, namespace, reason)`; `HeuristicPlanner` classifies cacheability (time-sensitive words, personal data markers, randomness requests). |
| `evaluation.py` | Hit rate, cost saved (from pricing), latency saved, false-hit audit sample. |

Direct API: `CachingModel.wrap(llm) -> CachedLLMClient`; `.cache` (a `ResponseCache`); `run(request)` to plan/explain caching for a prompt.

### 4.6 Pipeline RAG (`pipeline_rag`)

Purpose: the end-to-end retrieval-augmented generation pipeline: route, transform queries, retrieve, grade and compress context, generate grounded answers with citations, verify, and iterate for multi-hop questions.

| Module | Contents |
| --- | --- |
| `router.py` | `QueryRouter`: Adaptive-RAG style classification into `no_retrieval | single_step | multi_step | agentic`, with heuristics offline. |
| `transform.py` | Query rewriting, step-back prompting, multi-query, HyDE (delegates to a `Retriever`'s own transforms when available). |
| `context.py` | `ContextGrader` (CRAG-style per-document relevance: correct/ambiguous/incorrect, with fallback to web/alternate retriever hook), `ContextCompressor` (extractive selection of relevant sentences, token budget), `ContextOrderer` (places the most relevant documents at the beginning and end of the context to counter lost-in-the-middle effects), deduplication. |
| `generator.py` | Grounded generation with `[doc-id]` citations and structured output `{answer, citations[], unsupported_claims[], confidence}`; abstention when evidence is insufficient. |
| `iterate.py` | Multi-hop loop (IRCoT/FLARE style): generate partial reasoning, detect information need, retrieve again, bounded hops; `Self-RAG` style reflection flags (`is_relevant`, `is_supported`, `is_useful`) as structured outputs. |
| `indexing.py` | `IndexBuilder` using an injected `Retriever`-with-index (default: a minimal built-in BM25 over `request.inputs["documents"]` so the package works standalone), optional hierarchical summaries (RAPTOR-style) built with the LLM. |
| `evaluation.py` | RAGAS-style metrics via LLM judge: faithfulness, answer relevance, context precision, context recall; citation precision/recall; a small synthetic QA generator over the corpus. |
| `plan.py` | `RAGPlan(route, query_transforms, k, grade_context, compress, max_hops, verify, citation_style, rationale)`; `HeuristicPlanner`. |

Dependencies are injected: `retriever: Retriever | None`, `verifier: Verifier | None`, `executor: Executor | None`, `cache: ResponseCache | None`. Without them the package uses its minimal built-ins so it works in isolation; with them it becomes the full pipeline.

---

## 5. Composition contract

- A component **never imports another model package**. It accepts protocol objects.
- `examples/enterprise_pipeline.py` wires: `CachingModel.wrap(llm)` -> shared client; `HybridRetrievalModel` as `Retriever`; `AnswerVerificationModel` as `Verifier`; `ParallelizationModel.executor` as `Executor`; `PipelineRAGModel(retriever, verifier, executor)`; `TaskDecompositionModel(solver=rag.run)` for complex tasks; final verification; a single `Tracer` shared by all.
- Usage rolls up: each nested `run()` returns `usage`; the caller `ctx.charge()`s it. The root result's `usage.cost_usd` is the total.
- One `Tracer` instance is passed to every component so `tracer.tree()` shows the whole pipeline.

---

## 6. Known limitations and roadmap

Recorded by the package engineers as `core_change_requests` during implementation; none blocks use of the models today.

| Area | Limitation | Planned change |
| --- | --- | --- |
| `RunContext` budgets | `llm_call()` checks the remaining call budget before each call, but N concurrent callers can all pass the check before any completes, so a fan-out can overshoot `max_llm_calls` by up to the fan-out width. The task decomposition scheduler reserves calls per wave locally to compensate. | Add `BudgetTracker.try_reserve(n)` / `ctx.reserve_calls(n)` so fan-outs claim budget atomically up front. |
| `MockLLM` structured stubs | `example_from_schema` emits arrays with `minItems` elements (usually zero), so offline runs exercise the heuristic fallbacks more than the LLM-planned branches. Packages cover the LLM path with `ScriptedLLM` instead. | Emit one example element for arrays without `minItems` at shallow depth. |
| pytest layout | Every model keeps its own `tests/test_smoke.py` etc.; `--import-mode=importlib` (set in `pyproject.toml`) avoids basename collisions, which means test modules cannot import sibling test modules by bare name. Shared fixtures go in a uniquely named module re-exported from `conftest.py` (see `task_decomposition_model/tests/`). | None needed; documented convention. |
| Offline detection (resolved) | `RunContext.offline` was `isinstance(llm, MockLLM)`, so a cache-wrapped mock looked online and components parsed empty stubs. | Done: `pipeline_core.is_offline_client()` follows `inner` chains; `RunContext` uses it. |
| Verifier cost roll-up (resolved) | `AnswerVerificationModel.verify()` returned a `Verdict` without its usage, so verification cost vanished from parent results. | Done: `Verdict.details` now carries `usage`, `component_status` and `request_id`. |
| Cost budget | `max_cost_usd` is enforced after each call (by `BudgetTracker.charge`), so a single call can overshoot it once; call and latency budgets are checked before calls. | Pre-call check against `remaining_cost()` with an estimated request cost. |
| Citation verification | 382 of 456 dossier sources are machine-verified against arXiv metadata; 74 non-arXiv sources (books, standards, vendor documentation, journal-only papers) are marked "not independently verified" pending a venue-page check. | Extend `tools/verify_arxiv.py` with DOI (Crossref) lookups. |

## 7. Quality bar

- `pytest` green at the root with no network.
- `python -m <package> --offline "..."` produces a meaningful result for each package.
- No `temperature`, no `budget_tokens`, no date-suffixed model IDs anywhere.
- No stable prompt block contains `datetime`, `uuid` or request-specific text.
- README claims match the code (every documented class and default exists).
