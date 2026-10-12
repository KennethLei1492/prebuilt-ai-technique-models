# Composition Patterns

How the six models combine into one pipeline, and why each combination is
the right one according to the literature. The patterns below are the ones
Anthropic documents in *Building Effective Agents* (Schluntz & Zhang, 2024)
and that Zaharia et al. (2024) describe as the building blocks of *compound
AI systems*; each is mapped onto the models in this repository and the
protocols in `pipeline_core.protocols`.

The models never import each other. Every edge in the diagrams below is one
of four protocol objects: `Retriever`, `Verifier`, `Executor`, `ResponseCache`
(plus the `SubTaskSolver` callable). Wiring happens in one place, the
orchestrator (see `examples/enterprise_pipeline.py`).

---

## 0. The reference pipeline

```
                 ┌──────────────────────────────────────────────────────────────┐
  TaskRequest ──▶│ CachingModel.wrap(llm)  → one CachedLLMClient shared by all │
                 └──────────────┬───────────────────────────────────────────────┘
                                │ (exact → semantic → coalesced → provider prompt cache)
                                ▼
                 ┌─────────────────────────┐   simple    ┌────────────────────────────┐
                 │ Complexity router       │────────────▶│ PipelineRAGModel           │
                 │ (TaskDecomposition.plan)│             │  route → transform → retrieve│
                 └────────────┬────────────┘             │  → grade/compress/order    │
                              │ complex                  │  → generate+cite → verify  │
                              ▼                          └───────┬────────────┬───────┘
                 ┌─────────────────────────┐                     │            │
                 │ TaskDecompositionModel  │   sub-tasks         │ Retriever  │ Verifier
                 │  DAG of sub-tasks       │──────┐              ▼            ▼
                 │  waves via Executor     │      │   ┌──────────────────┐ ┌──────────────────────┐
                 └────────────┬────────────┘      │   │ HybridRetrieval  │ │ AnswerVerification   │
                              │ synthesis         └──▶│ sparse+dense     │ │ self-consistency,     │
                              ▼                       │ fusion+rerank    │ │ CoVe, grounding, rules│
                 ┌─────────────────────────┐          └──────────────────┘ └──────────────────────┘
                 │ Final verification gate │                     ▲
                 │ (AnswerVerificationModel)│                    │ Executor (bounded fan-out,
                 └────────────┬────────────┘                     │  rate limits, retries)
                              ▼                       ┌──────────────────┐
                        TaskResult + Tracer tree      │ Parallelization  │
                                                      └──────────────────┘
```

Data flow, in order:

1. **Cache first.** `CachingModel.wrap(llm)` returns a `CachedLLMClient`. Every
   component is constructed with it, so identical or near-identical calls
   anywhere in the pipeline are served from the exact or semantic tier, and
   the prefix advisor keeps system prompts cache-friendly. FrugalGPT (Chen,
   Zaharia & Zou, 2023) calls this *LLM approximation* and reports it as one of
   the three levers that together cut cost by up to 98% at matched accuracy.
2. **Route by complexity.** The decomposition model's `plan()` is itself a
   cheap, structured classification of task complexity. Single-fact questions
   go straight to RAG; multi-part tasks are decomposed. Adaptive-RAG (Jeong et
   al., 2024) shows routing by complexity beats always-simple and
   always-complex strategies on both accuracy and cost.
3. **Decompose into a DAG.** Independent sub-tasks are discovered up front
   (LLMCompiler, Kim et al., 2023) so they can run concurrently; dependent
   ones receive upstream outputs through placeholders.
4. **Execute waves in parallel.** The `Executor` from the parallelization
   model bounds concurrency, enforces provider rate limits, retries with
   jittered backoff and trips a circuit breaker; partial failures follow an
   explicit policy instead of crashing the run.
5. **Each sub-task is a RAG run.** `PipelineRAGModel` is the `SubTaskSolver`.
   It uses the hybrid retriever for evidence (BM25 for identifiers, standards
   clauses and part numbers; dense for paraphrase; fused with RRF or a
   convex combination; reranked for precision) and generates a cited answer.
6. **Verify before trusting.** The verification model grades grounding against
   the retrieved evidence, re-derives numeric answers with self-consistency,
   runs domain rules (units, required sections), and executes code when
   allowed. Revisions are accepted only with an external signal, following
   Huang et al. (2024), *Large Language Models Cannot Self-Correct Reasoning Yet*.
7. **Synthesise and gate.** Sub-task results are merged with traceability, then
   the whole answer passes the verification gate once more. Below the domain's
   `min_confidence_to_accept`, the result is marked `NEEDS_REVIEW` for a human.
8. **One trace.** A single `Tracer` is shared, so `tracer.tree()` shows every
   span and `result.usage` is the rolled-up cost of the entire run.

---

## 1. Prompt chaining (sequential)

Pattern: fixed sequence of calls, each consuming the previous output, with a
programmatic check between steps.

Models: RAG's internal chain (rewrite -> retrieve -> grade -> generate ->
verify) is a prompt chain; decomposition with the `LeastToMost` or
`PlanAndSolve` strategy is a prompt chain over sub-tasks.

```python
rag = PipelineRAGModel(llm, retriever=retriever, verifier=verifier)
first = await rag.run(request.derive("Which standard governs the impeller material?"))
second = await rag.run(request.derive(f"Given {first.output_text}, what corrosion allowance applies?", context=first.evidence))
```

When to use: tasks with a known order and a checkable intermediate (schema
validation, verifier verdict). Add a verification step between stages rather
than hoping the final step catches errors.

## 2. Routing

Pattern: classify the input, then dispatch to a specialised path.

Models: `TaskDecompositionModel.plan()` (complexity), `PipelineRAGModel`'s
`QueryRouter` (no-retrieval / single / multi-step / agentic),
`CachingModel.plan()` (cacheable or not), `HybridRetrievalModel.plan()`
(sparse-heavy vs dense-heavy by query type). Routing to cheaper models for
easy inputs is the RouteLLM result (Ong et al., 2024): a learned router kept
quality while cutting cost by over 2x on MT-Bench.

Rule of thumb: route with `effort="low"` and a tiny structured schema; the
router call must cost less than the savings it enables.

## 3. Parallelization (sectioning and voting)

Pattern: run independent calls concurrently and aggregate.

- **Sectioning** - split work into independent parts: decomposition DAG waves,
  map-reduce over documents, skeleton-of-thought expansion of answer sections
  (Ning et al., 2023 report up to 2.39x latency reduction).
- **Voting** - run the same task several times and aggregate: self-consistency
  (Wang et al., 2022; +17.9 points on GSM8K for PaLM-540B), ensembles /
  mixture-of-agents, repeated sampling with a verifier (Brown et al., 2024).

Models: `ParallelizationModel.executor` is the engine; `AnswerVerificationModel`'s
`SelfConsistencyVerifier` and `HybridRetrievalModel`'s multi-query expansion
are its most frequent callers.

Cache-aware fan-out: send one request first, wait for its first token (the
cache entry becomes readable only after the first response starts streaming),
then fire the remaining N-1 so they read the prefix cache instead of all
paying the write premium. `patterns.cache_aware_fan_out` implements this.

## 4. Orchestrator-workers

Pattern: a central LLM decides dynamically what sub-tasks are needed, delegates
them, and synthesises.

Models: `TaskDecompositionModel` is the orchestrator; the `SubTaskSolver` (a
RAG run, a tool, another component) is the worker; `ParallelizationModel`
schedules workers; `Synthesizer` merges. Anthropic's multi-agent research
system (Hadfield, Zhang & Li, 2025) reports that explicit effort scaling per
sub-task and structured artifacts (not chat transcripts) between orchestrator
and workers are what make this pattern reliable; the decomposition model's
`SubTask.complexity` and typed `TaskResult` passing follow that advice.

Failure modes to design against (Cemri et al., 2025, *Why Do Multi-Agent LLM
Systems Fail?*): unclear termination, step repetition, lost context between
agents, and absent final verification. The pipeline counters them with DAG
depth/breadth bounds, idempotency keys, typed artifacts and the mandatory
final verification gate.

## 5. Evaluator-optimizer

Pattern: one call produces, another evaluates and gives feedback, loop until
accepted or budget exhausted.

Models: `AnswerVerificationModel` is the evaluator; the producing component
(RAG generator, decomposition synthesiser) is the optimizer; the
`Replanner` closes the loop for plans. Bounded iterations (default 2) and the
"external signal" rule prevent the degradation that unbounded self-correction
shows in Huang et al. (2024).

## 6. Caching layers

Three tiers, cheapest first:

| Tier | Where | Key | Hit condition | Risk |
| --- | --- | --- | --- | --- |
| Exact | `CachedLLMClient` | `LLMRequest.fingerprint()` (model, prompts, schema, effort) | byte-identical request | staleness (mitigate with TTL and tag invalidation) |
| Semantic | `SemanticCache` | embedding of the normalised user text within a namespace | cosine >= threshold (default 0.95) and optional LLM equivalence check | false hits on near-duplicates with different numbers or dates |
| Provider prefix | Anthropic prompt cache | the stable system prefix | identical prefix within TTL (5 min / 1 h) | silent invalidation by volatile text in the prefix |

Invalidation: RAG answers are tagged with the ids of the documents they cite
(`DependencyFingerprint`), so re-indexing a document invalidates every cached
answer that depended on it.

## 7. Guardrails and gating

Statuses are the pipeline's control signal:

- `SUCCESS` - verified above threshold.
- `PARTIAL` - some sub-tasks failed but output is usable; warnings list what is missing.
- `NEEDS_REVIEW` - confidence below `DomainProfile.min_confidence_to_accept` (0.5 general, 0.75 mechanical/electrical, 0.85 safety-critical civil and process engineering).
- `FAILED` - no usable output.
- `CACHED` - served without recomputation.

Downstream systems should branch on `status`, never on free text.

## 8. Recipes

### Document question answering over an engineering corpus

```python
tracer = Tracer()
llm = CachingModel(tracer=tracer).wrap(make_llm())
retriever = HybridRetrievalModel(llm, tracer=tracer); retriever.index(documents)
verifier = AnswerVerificationModel(llm, tracer=tracer)
rag = PipelineRAGModel(llm, retriever=retriever, verifier=verifier, tracer=tracer)
result = await rag.run(TaskRequest(task=question, domain="mechanical_engineering"))
```

### Multi-part design review

```python
executor = ParallelizationModel(llm, tracer=tracer).executor
decomp = TaskDecompositionModel(llm, solver=rag.run, executor=executor, tracer=tracer)
result = await decomp.run(TaskRequest(task=review_brief, inputs={"documents": documents}, domain="systems_engineering"))
final = await verifier.verify(result.output_text, task=review_brief, evidence=result.evidence)
```

### Batch processing many similar requests cheaply

Use the caching model with semantic tier on, the parallelization executor
with rate limits matching the provider tier, and `effort="low"` on routing
calls. Measure `usage.cache_hit_ratio` and `cost_usd` from the trace; the
healthy signature is rising `cache_read_input_tokens` with flat
`cache_creation_input_tokens`.

---

## 9. Evaluation discipline

Kapoor et al. (2024), *AI Agents That Matter*, show that many reported agent
gains disappear once cost is put on the same axis as accuracy. Every
configuration change in this pipeline should therefore be evaluated on a
cost-accuracy Pareto frontier against simple baselines (zero-shot strong model;
retry; escalation). Each model ships an `evaluation.py` with its own metrics
(nDCG/recall for retrieval; plan coverage and speedup for decomposition;
verification accuracy and Brier score; speedup and utilisation; hit rate and
cost saved; RAGAS-style faithfulness and context precision/recall). The
shared `Tracer` provides the cost axis.

## Sources

All sources above are listed with verification status in
[`RESEARCH_BIBLIOGRAPHY.md`](RESEARCH_BIBLIOGRAPHY.md) under *Cross-cutting:
Pipeline Architecture and Efficiency* and the per-model sections.
