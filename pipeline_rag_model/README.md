# Pipeline RAG (`pipeline_rag`)

The end-to-end retrieval-augmented generation component: it reads a
natural-language task, decides **whether and how** to retrieve (Adaptive-RAG
routing), rewrites the query into several retrieval variants fused with
Reciprocal Rank Fusion, retrieves from an injected `Retriever` and/or its
built-in BM25 index, grades the evidence (CRAG), compresses and orders it
(RECOMP, Lost in the Middle), generates an answer whose every statement
carries a `[doc-id]` citation, reflects on support (Self-RAG), iterates for
multi-hop questions (IRCoT / FLARE), and escalates to a richer plan when the
cheap one abstains (Self-Route). It implements `pipeline_core.PipelineComponent`
and is itself a `Retriever` and a `SubTaskSolver`, so the other five models
plug in without adapters.

Everything runs offline: without credentials `make_llm()` returns a `MockLLM`,
the `HeuristicPlanner` decides the plan, lexical graders replace the LLM
graders and an extractive, citation-complete generator replaces the LLM
generator. The same code runs against Claude when `ANTHROPIC_API_KEY` is set.

---

## 1. Purpose

Engineering question answering over specifications, incident reports,
procedures, datasheets and requirements has three properties that a plain
"retrieve-then-read" chain handles badly. Questions range from trivial
lookups to corpus-wide "what recurs across all incidents" (fixed-k retrieval
over-spends on the first and cannot answer the last; Jeong et al. 2024;
Edge et al. 2024). Readers are distracted by irrelevant passages and lose
facts in the middle of long contexts (Liu et al. 2024; Yoran et al. 2024).
And a fabricated number or clause is worse than no answer, so every statement
must be traceable to a source span and the system must be able to say
"insufficient evidence" (Gao et al. 2023 ALCE; Barnett et al. 2024 FP1).

This package therefore treats RAG as a planned, conditional, bounded-loop
graph (Gao et al. 2024 Modular RAG) rather than a fixed chain, and returns a
typed `RAGOutput` with citations, unsupported claims, reflection flags, the
route taken and four confidence signals, so downstream verification and human
review know exactly what they are looking at.

## 2. Architecture

```
TaskRequest(task, inputs{documents, question, k}, context: Evidence, domain, constraints)
                                   │
                                   ▼
┌───────────────────────── PipelineRAGModel.plan() ──────────────────────────────┐
│ extract_task_features() -> TaskFeatures (identifiers, multi-hop / global cues,   │
│   question words, corpus size, retriever / context availability, risk level)   │
│ offline | deterministic | budget < 2 calls ─▶ HeuristicPlanner.plan_from_features │
│ online ─▶ PLAN_PROMPT (1 structured call, effort=high, seeded with the heuristic │
│           suggestion) ─▶ _enforce_domain() ─▶ RAGPlan.normalized()              │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   ▼  RAGPlan(route, query_transforms, k, grade_context, compress,
                                      max_hops, verify, citation_style, rationale, ...)
┌───────────────────────── PipelineRAGModel.execute() ───────────────────────────┐
│ index     indexing.py   IndexBuilder -> BuiltinRetriever (BM25 + contextual     │
│                         prefix, LLM prefixes when contextual_prefix='llm');     │
│                         RAPTOR summary nodes when plan.use_hierarchy (again on  │
│                         an escalated pass whose plan needs them: index_N step)  │
│ ┌─ one pass per route (Self-Route escalation loop, <= max_escalations) ───────┐ │
│ │ transform  transform.py  literal + rewrite | multi_query | step_back | hyde  │ │
│ │                          (+ glossary expansion from the DomainProfile)      │ │
│ │ retrieve   model.py      every variant x every source (injected Retriever,  │ │
│ │                          builtin index, inline documents, upstream context) │ │
│ │                          via Executor or gather_limited -> RRF k=60 -> top-k │ │
│ │ grade      context.py    ContextGrader (CRAG correct | ambiguous | incorrect)│ │
│ │                          incorrect -> fallback retrieval round(s)           │ │
│ │ iterate    iterate.py    MultiHopLoop (multi_step / agentic): reason, detect │ │
│ │                          need, retrieve again, dedupe, <= max_hops           │ │
│ │ compress   context.py    ContextCompressor (extractive, token budget) ->     │ │
│ │                          ContextOrderer (best first, second best last)      │ │
│ │ generate   generator.py  GroundedGenerator: {answer, citations[],           │ │
│ │                          unsupported_claims[], confidence, abstained}       │ │
│ │ verify     iterate.py    injected Verifier | ReflectionJudge (Self-RAG       │ │
│ │                          IsRel / IsSup / IsUse)                              │ │
│ │ abstained and escalate_on_abstention? -> next_route() and repeat            │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ finish    model.py      aggregate_confidence(self_reported, citation_coverage,  │
│                         support, evidence_quality) vs threshold -> Status       │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   ▼
TaskResult(output=RAGOutput, output_text, evidence[grade, fused_score, legs, compressed],
           confidence, confidence_basis, steps[plan, index, transform, retrieve, grade,
           iterate, compress, generate, verify],
           metadata{plan, routes_tried, review_reasons, degraded_stages})
```

| Module | Contents |
| --- | --- |
| `router.py` | `Route`, `ROUTE_ORDER`, `TaskFeatures`, `RouteDecision`, `extract_task_features`, `route_for_features`, `QueryRouter` (`classify`; `classify_llm`, used by `PipelineRAGModel.route_llm`), `next_route` |
| `transform.py` | `QueryVariant`, `QueryVariants` (LLM schema), `QueryTransformer`, `heuristic_variants`, `keyword_query`, `step_back_query`, `split_sub_queries`, `glossary_expansion`, `reciprocal_rank_fusion` |
| `context.py` | `DocGrade`, `GradeBatch`, `ContextGrader`, `ContextCompressor`, `ContextOrderer`, `lexical_relevance`, `label_for_score`, `crag_decision`, `extractive_compress`, `order_for_position`, `deduplicate`, `looks_like_code_or_table`, `split_sentences` |
| `generator.py` | `Citation`, `GenerationOutput`, `Statement`, `GroundedGenerator` (`generate_llm`, `generate_heuristic`), `extractive_answer`, `split_statements`, `citation_coverage`, `cited_ids`, `render_answer`, `extract_markers`, `strip_markers`, `citation_instruction` |
| `iterate.py` | `HopStep`, `HopRecord`, `StatementSupport`, `SupportJudgement`, `ReflectionFlags`, `MultiHopLoop`, `ReflectionJudge`, `critique_score`, `support_fraction`, `overall_support`, `flare_trigger`, `should_retrieve`, `plan_hop_queries`, `merge_evidence`, `heuristic_support`, `question_relevant` |
| `indexing.py` | `SourceDocument`, `Chunk`, `normalize_documents`, `chunk_document`, `heuristic_prefix`, `BM25Index`, `BuiltinRetriever` (a `Retriever`), `IndexBuilder` (`build`, `add`, `apply_llm_prefixes`, `build_hierarchy`), `lead_summary` |
| `evaluation.py` | `QAExample`, `EvalRecord`, `EvaluationReport`, `Judge`, `RAGEvaluator`, `exact_match`, `token_f1`, `context_precision`, `context_recall`, `faithfulness`, `citation_metrics`, `abstention_metrics`, `attribute_failure_point`, `synthetic_qa`, `synthetic_qa_llm`, `sample_corpus`, `sample_qa`, `main` |
| `plan.py` | `RAGPlan`, `HeuristicPlanner`, `answer_type_for_features` |
| `prompts.py` | Versioned `PromptTemplate`s: `PLAN_PROMPT`, `ROUTE_PROMPT`, `TRANSFORM_PROMPT`, `GRADE_PROMPT`, `GENERATE_PROMPT`, `NO_RETRIEVAL_PROMPT`, `HOP_PROMPT`, `SUPPORT_PROMPT`, `CONTEXT_PREFIX_PROMPT`, `SUMMARY_PROMPT`, `JUDGE_FAITHFULNESS_PROMPT`, `JUDGE_RELEVANCE_PROMPT`, `SYNTHETIC_QA_PROMPT`; `make_request`, `stable_system_blocks` |
| `config.py` | `PipelineRAGConfig` (42 tunables, section 5) |
| `budget.py` | `calls_left` (the one budget view every stage uses: 0 once the latency or cost budget is spent, else the remaining calls), `note_degraded` / `degraded_stages` (stages that fell back, the basis of `PARTIAL`) |
| `model.py` | `PipelineRAGModel` (`spec`, `plan()`, `execute()`, direct API incl. `route_llm`, `invalidate_retrieval_cache`), `RAGOutput`, `ConfidenceSignals`, `SummaryOut`, `ContextPrefixOut`, `aggregate_confidence` |
| `cli.py` | `python -m pipeline_rag "task" [--domain] [--offline] [--json] [--input k=v] [--documents PATH] [--sample-corpus] [--k] [--max-calls] [--llm-mode] [--show-evidence] [--trace] [--evaluate]` |

## 3. How it reasons about a prompt

### 3.1 The plan (`RAGPlan`)

| Field | Meaning |
| --- | --- |
| `route` | `no_retrieval \| single_step \| multi_step \| agentic` (Adaptive-RAG complexity classes; `agentic` = corpus-wide questions over RAPTOR summary nodes with the reflection loop) |
| `query_transforms` | Subset of `rewrite \| multi_query \| step_back \| hyde`; the literal question is always retrieved as well and the lists are fused with RRF |
| `k` | Evidence units kept after fusion (clamped to 1-100) |
| `grade_context` | Run the CRAG gate before generation |
| `compress` | Extractive compression to `context_token_budget` |
| `max_hops` | Retrieve-and-reason hops (1 for `no_retrieval`/`single_step`, 2-`max_hops` otherwise) |
| `verify` | Run the injected `Verifier` or the built-in Self-RAG reflection after generation |
| `citation_style` | `inline \| footnote \| none` rendering of `[doc-id]` markers |
| `rationale` | One or two sentences naming the task features that drove the decision |
| `context_token_budget` | Token budget for the evidence passed to the generator |
| `abstain_when_insufficient` | Answer "insufficient evidence" instead of guessing |
| `escalate_on_abstention` | Self-Route: on abstention re-run with `next_route()` |
| `use_hierarchy` | Build and search RAPTOR-style document and corpus summaries (also built when a Self-Route escalation reaches `agentic`) |
| `expected_answer_type` | `factual \| numerical \| procedural \| explanatory \| list \| comparison \| summary \| code`; guides the generator's shape |
| `planner` | `heuristic \| llm`: who produced the plan |

The plan is attached to `result.metadata["plan"]` and printed by the CLI
(`plan.describe()`); `RAGPlan.normalized(config)` makes any plan (LLM or
heuristic) consistent with its route and the config: `no_retrieval` drops
transforms, grading, compression and hierarchy; `single_step` forces one hop;
multi-hop routes clamp hops to `[2, max_hops]`; `agentic` turns the hierarchy
on unless `build_hierarchy="never"`; `hyde` is removed unless `enable_hyde`;
the transform list is cut to `max_query_variants`; escalation is disabled when
`max_escalations == 0`.

### 3.2 Heuristic planner (offline, `constraints.deterministic`, or fewer than two calls left)

`extract_task_features` computes word and question counts, leading question
words, identifiers (part numbers `P-102`, requirement ids `REQ-CW-020`, error
codes, `snake_case`/`camelCase` symbols), multi-hop cues (`compare`, `root
cause`, `relationship`, `led to`, `across`, ...), global cues (`themes`,
`patterns`, `recurring`, `across all`, ...), transformation cues (`rewrite`,
`translate`, `format this`, ...), why/how and numeric cues, enumerations,
inline document count and size, whether a retriever or upstream `context` is
available, and the profile's `risk_level`. The rules
(`router.route_for_features`, `plan.HeuristicPlanner.plan_from_features`):

1. **Route.** No documents, retriever or context -> `no_retrieval` (answer
   from parametric knowledge, low confidence). Pure text transformation with
   no question -> `no_retrieval`. Any global cue -> `agentic`. Any multi-hop
   cue, or two of {two questions, more than 30 words, three or more
   identifiers, three or more enumerated items with a why/how} ->
   `multi_step`. Otherwise `single_step`.
2. **Transforms.** Always `rewrite` (keyword query keeping identifiers and
   units). `step_back` for why/how questions **without** identifiers.
   `multi_query` for multi-hop routes, two or more questions or two or more
   enumerated items. `hyde` only when `enable_hyde` is set, the domain is not
   high-risk, the route is `single_step` and there are no identifiers. With
   identifiers present the rationale records that HyDE and step-back were
   withheld (they hallucinate or over-generalise private tokens).
3. **k.** `default_k` for `single_step`; `max(default_k, 2 * k_per_hop)` for
   multi-hop routes; 100 (everything) when the inline corpus is under
   `full_corpus_gate_tokens`.
4. **Domain.** `risk_level in {high, safety_critical}` forces
   `grade_context`, `verify`, `abstain_when_insufficient` and inline citations
   regardless of route, and the rationale says so. Otherwise
   `citation_style` comes from the config (`none` for `no_retrieval`).
5. **Hops and hierarchy.** `max_hops` for `multi_step`/`agentic`, 1 otherwise;
   `use_hierarchy` for `agentic`.
6. **Answer shape** (`answer_type_for_features`): global cue -> `summary`;
   comparison cue -> `comparison`; `list` or three or more enumerations ->
   `list`; numeric cue -> `numerical`; "how" -> `procedural`; why/how ->
   `explanatory`; else `factual`.

Examples (offline, sample corpus): "What corrosion allowance does the
specification require for the pump casing?" -> `route=single_step
transforms=rewrite k=10 hops=1 answer=factual`; "What was the root cause of the
high vibration trip on pump P-102 and which procedure was updated?" ->
`route=multi_step transforms=rewrite,multi_query hops=4` with the rationale
"multi-hop signal ('root cause') ...; identifiers present: literal query kept,
no HyDE/step-back over private tokens"; "What recurring themes appear across
all the incident reports?" -> `route=agentic use_hierarchy=True
answer=summary`.

### 3.3 LLM planner (online)

One structured call (`PLAN_PROMPT` v1.0.0, `effort=planner_effort`,
`output_schema=sanitize_schema(RAGPlan.model_json_schema())`). The stable
system block holds the role and six numbered rules (cheapest route that can
answer; always `rewrite`, `step_back` for principle questions, `multi_query`
for multi-faceted ones, never `hyde`/`step_back` over identifiers; k 5-10 for
focused and 10-20 for broad questions; hops 1 or 2-4; high-risk domains must
grade, verify, abstain and cite inline; explain the decision), then
`ctx.profile.system_text()` and `ENGINEERING_RIGOR` for high-risk profiles.
The user turn holds `<task>`, `<task_features>` (the `TaskFeatures` JSON),
`<corpus>` (inline document count and size, persistent index size, whether an
external retriever is injected, upstream evidence count) and
`<heuristic_suggestion>` (the heuristic plan's `describe()`), so the model
refines a defensible default rather than planning from nothing. The answer is
validated with `RAGPlan.model_validate`, marked `planner="llm"`, passed through
`_enforce_domain` (high-risk rules re-applied, `hyde`/`step_back` removed when
identifiers are present, `no_retrieval` forced when there is no corpus) and
`normalized()`. Any failure (`LLMError`, `StructuredOutputError`, validation)
falls back to the heuristic plan with a warning; `inputs["k"]` overrides `k`
either way.

Budget-awareness runs through every stage, not only planning: the LLM plan
needs two calls left (one must remain for generation), LLM transforms three,
LLM grading two, generation one; summaries for the hierarchy stop when fewer
than three calls remain; a spent cost budget (`max_cost_usd`) or latency
budget (`max_latency_s`) counts as zero calls, so no stage attempts a call it
cannot afford (`budget.calls_left`, shared by the model, the transformer and
the multi-hop loop). Stages degrade to their heuristic twins and warn;
nothing fails.

`constraints.deterministic` forces the heuristic planner and also every
advisory stage (transforms, grading, hops, reflection, summaries, contextual
prefixes); generation is then the only stage that samples the model, once per
Self-Route pass, so a deterministic run is as reproducible as the client (or
its response cache) makes that one call.

### 3.4 Execution decisions

- **Sources.** `_sources()` assembles the injected `Retriever`, the persistent
  built-in index (`model.index(...)`), a per-request built-in index over
  `inputs["documents"]` (cached by content hash, eight most recent) and, as a
  separate ranking, `request.context`. Every query variant goes to every
  source; legs run through the injected `Executor` (falling back to
  `gather_limited`) under `constraints.max_parallelism`; a failing leg warns
  and contributes nothing; once the latency budget is spent only the first
  (literal) variant is fanned out. When two sources return the same id with
  different text (e.g. the auto-generated `doc-1` of plain-string documents in
  the persistent and the inline index) the later one is renamed
  `<source>:<id>` with a warning, so no evidence is lost or misattributed;
  `model.index()` itself gives a different document a fresh `doc-N-k` id
  instead of overwriting an indexed one. Rankings are fused with weighted RRF (literal and
  rewrite 1.0, multi-query 0.8, step-back/HyDE 0.6, glossary 0.8) and the top
  `k` become `Evidence(retriever="fused", metadata{fused_score, legs,
  leg_score, leg_rank})`.
- **Full-corpus gate.** When `full_corpus_gate_tokens > 0` and the whole
  built-in corpus fits, ranking is bypassed and every chunk is used.
- **Grading.** `ContextGrader` scores each unit on [-1, 1] (lexical overlap
  offline; one listwise low-effort call online), applies the CRAG decision
  (any score >= `grade_upper_threshold` -> `correct`; all <=
  `grade_lower_threshold` -> `incorrect`; else `ambiguous`), drops `incorrect`
  strips when something relevant exists, and on `incorrect` (including a
  retrieval that returned nothing at all) runs up to `max_fallback_rounds` of
  fallback retrieval (keyword and step-back queries against
  `fallback_retriever` or the regular sources) before abstaining.
- **Iteration.** For `multi_step`/`agentic` with `max_hops > 1`,
  `MultiHopLoop` issues one new query per hop: online from `HOP_PROMPT`
  (`next_query` when `need_more_information` is set or the Self-RAG retrieve
  probability `need_probability` exceeds `retrieval_trigger_threshold`, or the
  reasoning step itself when its confidence is below
  `flare_confidence_threshold`), offline from
  `plan_hop_queries` (facet queries, then a step-back query). New units are
  lexically graded against the question, merged with dedupe, and the loop
  stops on `can_answer` (unless the retrieve probability still exceeds the
  trigger), an exhausted schedule, a repeated query, the hop cap or the
  latency budget.
- **Compression and ordering.** `extractive_compress` keeps the
  `sentences_per_chunk_*` most question-relevant sentences per unit, never
  touches code, tables or identifier lists, records
  `metadata{compressed, original_chars}` while keeping `span`, drops units
  that do not fit the budget; `order_for_position` places the best unit first
  and the second best last.
- **Generation.** Online: `GENERATE_PROMPT` (or `NO_RETRIEVAL_PROMPT` without
  evidence) with `effort=generation_effort`, documents in the user turn,
  structured `GenerationOutput`. Offline: `extractive_answer`, the most
  question-relevant sentences cited verbatim, one per unit for single facts,
  up to five for list/summary/comparison/procedural shapes, plus, for
  multi-hop plans, each facet's best sentence and its successor in the same
  unit (the answer-continuation pattern of incident records: trip -> root
  cause -> corrective action); it abstains when no sentence carries at least
  30 % of the question's weighted terms, except that a `summary` request
  falls back to the RAPTOR summary nodes with self-reported confidence 0.4.
- **Verification.** An injected verifier that is a `PipelineComponent` (the
  answer verification model) is invoked through its `run()` with a child
  request (`inputs={"answer": ...}`, the evidence as `context`, the parent's
  constraints narrowed to the remaining call / cost / latency budget); its
  `result.usage` is charged with `ctx.charge` and the `Verdict` comes from
  `result.output.to_verdict()`. A plain `Verifier` protocol object is called
  through `verify(answer, task=question, evidence=...)` and may report its
  spend as `details["usage"]`. The verdict is stored in `output.verdict` and
  its score becomes the `support` signal. Otherwise `ReflectionJudge` produces `ReflectionFlags`
  (`SUPPORT_PROMPT` online with `effort=max` in safety-critical domains,
  `generation_effort` in high-risk ones, `grading_effort` otherwise; a lexical
  entailment proxy offline, with `IsRel` judged against the question).
- **Escalation.** If the generation abstained, `escalate_on_abstention` is set,
  a corpus exists and the deadline is not exceeded, the route is escalated
  (`single_step -> multi_step -> agentic`) up to `max_escalations` times; each
  pass gets its own step records (`transform_2`, `retrieve_2`, ...) and
  `output.routes_tried` lists the path. A pass whose plan uses the hierarchy
  first builds the summary nodes the earlier plan skipped (`index_3`).

### 3.5 Confidence and status

`aggregate_confidence` takes the weighted mean of the signals that exist
(weights renormalised over the present ones): `self_reported` (the
generator's own probability), `citation_coverage` (fraction of statements
with a resolving `[doc-id]`), `support` (Self-RAG IsSup fraction or the
verifier's score) and `evidence_quality` (mean CRAG grade of the cited units
mapped to [0, 1]). An abstention caps the result at 0.25 (the basis says
"capped" only when the cap actually lowered the value). The threshold is
`max(profile.min_confidence_to_accept, constraints.min_confidence)` and the
basis string records every term, e.g. `self reported 0.75 (w 0.20), citation
coverage 1.00 (w 0.25), support 1.00 (w 0.35), evidence quality 0.83 (w 0.20)
-> 0.92; accept threshold 0.75 (mechanical_engineering)`.

Status rules (`_finish`, reasons also stored in
`metadata["review_reasons"]`): `NEEDS_REVIEW` when the answer abstained, when
confidence is below the threshold, when the injected verifier failed, when the
reflection says `IsRel=false` (the evidence does not address the question; the
answer is best effort), and, in high/safety-critical domains, when any
statement is unsupported or citation coverage is below
`min_citation_coverage`. A run that would otherwise succeed but had a stage
fall back (an LLM stage failing or running out of budget before generation, a
failing retrieval leg, executor or verifier) is `PARTIAL`; the stages are
recorded explicitly (`metadata["degraded_stages"]`) and named in the reason,
e.g. `stages degraded to their fallback: retrieval leg injected`. Expected
conditions never raise.

## 4. Quick start

### Python

```python
from pipeline_core import make_llm, TaskRequest, Constraints
from pipeline_rag import PipelineRAGModel, RAGPlan
from pipeline_rag.evaluation import sample_corpus

model = PipelineRAGModel(make_llm())            # MockLLM without credentials -> heuristic stages
documents = [
    {"id": "PS-100", "source": "Pump Specification PS-100 Rev C",
     "text": "Section 4.2 Casing. The pump casing shall be ASTM A216 WCB carbon steel with a 3 mm corrosion allowance ..."},
    {"id": "INC-2041", "source": "Incident Report INC-2041",
     "text": "Pump P-102 tripped on high vibration. Root cause: misalignment of 0.35 mm offset ... procedure update MP-14 Rev D ..."},
]

result = model.run_sync(TaskRequest(
    task="What corrosion allowance does the specification require for the pump casing?",
    inputs={"documents": documents},
    domain="mechanical_engineering",
))
print(result.summary())                      # [pipeline_rag] status=success confidence=0.92 calls=0 ...
print(result.output_text)                    # "... 3 mm corrosion allowance ... [PS-100]."
plan = RAGPlan.model_validate(result.metadata["plan"]); print(plan.describe())
out = result.output                          # RAGOutput
print(out.route, out.routes_tried, out.cited_ids, out.unsupported_claims, out.abstained)
print(out.confidence.basis, out.reflection.is_supported if out.reflection else None)

# Direct API: index once, ask many times; the model is also a Retriever
model.index(sample_corpus())
answer = model.answer_sync("What is the vibration acceptance limit?", domain="mechanical_engineering", k=5)
import asyncio
hits = asyncio.run(model.retrieve("MP-14 alignment tolerance", k=3))
print(model.route("Compare the casing allowance with the impeller material.").route)   # multi_step
print(model.plan_for("Why does the casing corrode?", domain="chemical_process_engineering").verify)
decision = asyncio.run(model.route_llm("Which pump failed and why?"))   # one low-effort call online; heuristic offline

# Budgets degrade stages instead of failing
cheap = model.run_sync(TaskRequest(task="What is the motor rated power?", inputs={"documents": documents},
                                   constraints=Constraints(max_llm_calls=1)))
```

Inputs understood by `run()`: `inputs["documents"]` (list of strings, dicts
with `id`/`text` or `content`/`source`/`metadata`, `Evidence`, or a
`{id: text}` mapping), `inputs["question"]` (overrides the task text as the
question), `inputs["k"]`, and `request.context` (upstream `Evidence`, fused
as its own ranking). `result.as_evidence()` feeds the answer to the next
component; `model.solver` is the `SubTaskSolver` for task decomposition.

### CLI

```bash
cd "<repo root>"
PYTHONPATH="common:pipeline_rag_model" python -m pipeline_rag --offline --sample-corpus \
    "What corrosion allowance does the specification require for the pump casing?" --domain mechanical_engineering --show-evidence
python -m pipeline_rag "Which procedure was updated after the P-102 trip?" --documents ./docs --json
python -m pipeline_rag "root cause of the vibration trip" --input k=5 --max-calls 3 --trace
python -m pipeline_rag --offline --sample-corpus --evaluate "any task"      # answer + evaluation report
python -m pipeline_rag.evaluation --offline                                 # the harness alone (section 9)
```

Without `--documents`/`--sample-corpus` the router selects `no_retrieval` and
the model answers from its own knowledge with low confidence (offline it
abstains and exits 0 with `status=needs_review`). On Windows use `;` as the
`PYTHONPATH` separator. `--llm-mode always` forces the LLM path with a
scripted client; `--llm-mode never` forces pure heuristics.

## 5. Configuration (`PipelineRAGConfig`)

Pass a `PipelineRAGConfig(...)` as `config=` or any field as a keyword
argument (`PipelineRAGModel(llm, max_hops=3)`). The README table is kept in
step with the field descriptions in `config.py`.

| Field | Default | Justification (dossier source) |
| --- | --- | --- |
| `llm_mode` | `auto` | `auto` uses LLM stages with a real client and heuristics when `ctx.offline`; `always`/`never` force a path (tests of the LLM path use `always` with `ScriptedLLM`). Architecture rule 3.1.2: plan must fall back to heuristics offline |
| `max_escalations` | `2` | Self-Route escalates cheap RAG to a richer plan on "unanswerable", never downward; dossier default depth 2: Li et al. (2024), *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (Self-Route)*, https://arxiv.org/abs/2407.16833 |
| `full_corpus_gate_tokens` | `0` | Anthropic recommends bypassing retrieval below ~200K tokens: Anthropic (2024), *Introducing Contextual Retrieval*, https://www.anthropic.com/news/contextual-retrieval. Off by default because evidence lives in the volatile user turn, so there is no prompt-cache win to pay for the larger context |
| `default_k` | `10` | Reader accuracy saturates at 10-20 documents while retriever recall keeps rising: Liu et al. (2024), *Lost in the Middle: How Language Models Use Long Contexts*, https://arxiv.org/abs/2307.03172; Anthropic (2024) keeps top-20 after reranking. 10 is the lower edge because evidence is then compressed to a token budget |
| `k_per_hop` | `5` | 4-8 new units per hop with dedupe in retrieve-and-reason loops: Trivedi et al. (2023), *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)*, https://arxiv.org/abs/2212.10509; Jiang et al. (2023), *Active Retrieval Augmented Generation (FLARE)*, https://arxiv.org/abs/2305.06983 |
| `candidates_per_variant` | `50` | Retrieve top-50 per variant, fuse, then rerank (dossier query fan-out defaults); fusion needs deep per-list candidates: Cormack et al. (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf |
| `rrf_k` | `60` | Cormack et al. (2009) fixed k = 60 in a pilot and never re-tuned it; the hybrid retrieval model uses the same constant so fused scores are comparable |
| `max_query_variants` | `4` | 3-5 variants (1 rewrite, 2-3 sub-queries, 1 step-back): Ma et al. (2023), *Query Rewriting for Retrieval-Augmented Large Language Models*, https://arxiv.org/abs/2305.14283; Zheng et al. (2024), *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*, https://arxiv.org/abs/2310.06117 |
| `enable_hyde` | `False` | HyDE injects hallucinated detail when the model lacks domain knowledge (private identifiers), so it is opt-in and never used over identifiers or in high-risk domains: Gao et al. (2023), *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*, https://arxiv.org/abs/2212.10496 |
| `chunk_size_tokens` | `512` | Dossier range 200-800 tokens (GraphRAG uses 600; Anthropic "no more than a few hundred"): Edge et al. (2024), *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, https://arxiv.org/abs/2404.16130; Anthropic (2024). 512 matches the architecture contract so chunk ids line up with the hybrid retrieval model |
| `chunk_overlap_tokens` | `64` | 10-15 % overlap (GraphRAG 100/600): Edge et al. (2024) |
| `contextual_prefix` | `heuristic` | A situating prefix per chunk cut top-20 retrieval failures by 35-49 %: Anthropic (2024). `heuristic` derives it from title and position at zero cost; `llm` generates a 50-100 token prefix per chunk (`CONTEXT_PREFIX_PROMPT`, one sub-call per chunk, once per index, online only and budget-guarded; offline, under `deterministic` or on a short budget the heuristic prefix is kept) |
| `llm_prefix_max_chunks` | `200` | Cost cap for `contextual_prefix='llm'`: chunks per index that receive an LLM prefix; later chunks keep the heuristic one. Anthropic (2024) prices contextual prefixes as a one-off indexing cost, so it is bounded rather than per query |
| `build_hierarchy` | `auto` | Summary nodes answer document- and corpus-level questions that chunk retrieval cannot (+20 points on QuALITY): Sarthi et al. (2024), *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, https://arxiv.org/abs/2401.18059. `auto` builds them only for agentic plans because summarisation costs one call per document |
| `grade_upper_threshold` | `0.5` | CRAG thresholds are dataset-specific ((0.59, -0.99) to (0.95, -0.91)); start at 0.5 and calibrate per domain: Yan et al. (2024), *Corrective Retrieval Augmented Generation (CRAG)*, https://arxiv.org/abs/2401.15884 |
| `grade_lower_threshold` | `-0.9` | CRAG INCORRECT branch (discard, fall back or abstain); dossier starting point -0.9: Yan et al. (2024) |
| `max_fallback_rounds` | `1` | One configured fallback retriever with at most one round after an INCORRECT grade: Yan et al. (2024) |
| `context_token_budget` | `2000` | RAPTOR's collapsed-tree retrieval fills a 2,000-token budget (~top-20 nodes): Sarthi et al. (2024); compression keeps 5-10 % of retrieved tokens with <= ~10 % relative loss: Xu et al. (2024), *RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation*, https://arxiv.org/abs/2310.04408 |
| `sentences_per_chunk_single_hop` | `2` | RECOMP's extractive compressor keeps the top sentence; we keep 2 because the lexical extractor is noisier than RECOMP's trained dual encoder and engineering answers depend on the adjacent sentence carrying the unit or condition: Xu et al. (2024) |
| `sentences_per_chunk_multi_hop` | `3` | RECOMP keeps top-2 for multi-hop (+1 for the same reason): Xu et al. (2024) |
| `protect_code_and_tables` | `True` | Token dropping is destructive for code and numbers: Jiang et al. (2023), *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*, https://arxiv.org/abs/2310.05736; the dossier's engineering notes mandate extractive-only compression for code and logs |
| `max_evidence_chars` | `2000` | Characters rendered per evidence unit in generation prompts (about one 512-token chunk); keeps the volatile user turn bounded so the stable prefix stays cacheable |
| `max_hops` | `4` | IRCoT / Adaptive-RAG multi-step averaged 2.1-4.7 rounds; dossier default S = 4: Trivedi et al. (2023); Jeong et al. (2024), *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*, https://arxiv.org/abs/2403.14403 |
| `retrieval_trigger_threshold` | `0.2` | Self-RAG retrieval trigger: a hop retrieves again when the step's `need_probability` (P(Retrieve)) > 0.2, even if the model claims it can answer; 0 when citations are mandatory: Asai et al. (2024), *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, https://arxiv.org/abs/2310.11511 |
| `flare_confidence_threshold` | `0.6` | FLARE retrieves again when a sentence's confidence falls below a threshold in the 0.4-0.8 range: Jiang et al. (2023) |
| `reflection_weight_relevant` | `1.0` | Self-RAG critique weight for IsRel: Asai et al. (2024) |
| `reflection_weight_supported` | `1.0` | Self-RAG weight for IsSup; raise it in precision-critical domains: Asai et al. (2024) |
| `reflection_weight_useful` | `0.5` | Self-RAG weight for IsUse: Asai et al. (2024) |
| `citation_style` | `inline` | `[doc-id]` markers per statement are the ALCE convention: Gao et al. (2023), *Enabling Large Language Models to Generate Text with Citations (ALCE)*, https://arxiv.org/abs/2305.14627 |
| `require_citations` | `True` | At least one citation per factual statement; uncited statements count against confidence: Gao et al. (2023) |
| `min_citation_coverage` | `0.8` | Release gate on citation recall >= 0.8 (dossier) and review trigger in high-risk domains: Gao et al. (2023) |
| `min_faithfulness` | `0.9` | RAGAS faithfulness agrees with humans at 0.95; dossier gate 0.9: Es et al. (2024), *RAGAs: Automated Evaluation of Retrieval Augmented Generation*, https://arxiv.org/abs/2309.15217 |
| `abstain_when_insufficient` | `True` | Missing content answered anyway is failure point 1: Barnett et al. (2024), *Seven Failure Points When Engineering a Retrieval Augmented Generation System*, https://arxiv.org/abs/2401.05856; selective augmentation may return nothing: Xu et al. (2024); explicit "unanswerable": Li et al. (2024) |
| `confidence_weight_self_reported` | `0.2` | Lowest weight: a generation is judged by an external critique against the evidence rather than trusted as produced: Asai et al. (2024); never trust self-reported citations: Gao et al. (2023) |
| `confidence_weight_citation_coverage` | `0.25` | ALCE citation recall as a first-class quality signal: Gao et al. (2023) |
| `confidence_weight_support` | `0.35` | Highest weight: statement-level faithfulness is the best-validated RAG metric (0.95 human agreement): Es et al. (2024) |
| `confidence_weight_evidence` | `0.2` | Mean CRAG grade of the cited documents: Yan et al. (2024) |
| `planner_effort` | `high` | Architecture rule 3.2: `high` for planning and synthesis |
| `grading_effort` | `low` | Architecture rule 3.2: `low` for classification, routing and grading of many items |
| `generation_effort` | `high` | Architecture rule 3.2: `high` for grounded synthesis |
| `verification_effort_safety_critical` | `max` | Architecture rule 3.2: `max` only for final verification in safety-critical domains |
| `sub_call_model` | `None` | Optional cheaper model for grading, transforms and summaries (dossier cost control "model tiering"); `None` keeps one model so prompt caches stay model-scoped (architecture rule 3.2) |

`PipelineRAGConfig` validates that the lower grade threshold is below the
upper one, that overlap is smaller than the chunk size and that the confidence
weights sum to a positive number; `confidence_weights()` and
`reflection_weights()` return the normalised dictionaries the model uses.

## 6. Techniques implemented

| Technique | Where | Citation(s) |
| --- | --- | --- |
| Complexity routing into no / single-step / multi-step / agentic with upward escalation on abstention | `router.py`, `plan.py`, `model.py` | Jeong et al. (2024); Li et al. (2024); Gao et al. (2024), *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*, https://arxiv.org/abs/2407.21059 |
| Rewrite-Retrieve-Read keyword rewrite, multi-query facet split, step-back abstraction, HyDE (opt-in), glossary expansion from the domain profile | `transform.py` | Ma et al. (2023); Zheng et al. (2024); Gao et al. (2023) HyDE |
| Weighted Reciprocal Rank Fusion across variants and sources | `transform.py`, `model.py` | Cormack et al. (2009) |
| Sentence-aware chunking with a situating prefix indexed alongside the text; Okapi BM25 (k1 = 1.2, b = 0.75) built-in retriever | `indexing.py` | Anthropic (2024) |
| RAPTOR-style summary nodes (document level, corpus level) searched together with leaf chunks (collapsed tree); clusters seeded by document structure instead of UMAP + GMM | `indexing.py` | Sarthi et al. (2024); Edge et al. (2024) for the corpus-wide question class |
| CRAG three-way evidence gate with knowledge-strip refinement and fallback retrieval | `context.py`, `model.py` | Yan et al. (2024) |
| Extractive compression to a token budget with code/table protection and source-span pointers | `context.py` | Xu et al. (2024); Jiang et al. (2023) LLMLingua |
| Best-first / second-best-last context ordering | `context.py` | Liu et al. (2024); Jiang et al. (2024), *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*, https://arxiv.org/abs/2310.06839 |
| Citation-grounded generation with per-statement `[doc-id]` markers, cited quotes, unsupported-claim listing, abstention; statement splitting and citation resolution | `generator.py` | Gao et al. (2023) ALCE; Barnett et al. (2024) |
| Prompts that state evidence may be irrelevant and must be ignored, and that retrieved text is data, not instructions | `prompts.py` | Lin et al. (2024), *RA-DIT: Retrieval-Augmented Dual Instruction Tuning*, https://arxiv.org/abs/2310.01352; Yoran et al. (2024), *Making Retrieval-Augmented Language Models Robust to Irrelevant Context*, https://arxiv.org/abs/2310.01558; Greshake et al. (2023), *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*, https://arxiv.org/abs/2302.12173 |
| IRCoT / FLARE retrieve-and-reason loop: one reasoning step per hop, information-need detection, low-confidence trigger, bounded hops | `iterate.py` | Trivedi et al. (2023); Jiang et al. (2023) FLARE |
| Self-RAG reflection flags IsRel / IsSup / IsUse as structured output with weighted critique score | `iterate.py` | Asai et al. (2024) |
| Four-signal confidence with abstention cap and domain threshold | `model.py` | Es et al. (2024); Gao et al. (2023) ALCE; Asai et al. (2024) |
| RAGAS-style faithfulness and answer relevance judges, context precision/recall, ALCE citation recall/precision, abstention metrics, failure-point attribution, synthetic in-domain QA | `evaluation.py` | Es et al. (2024); Saad-Falcon et al. (2024), *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*, https://arxiv.org/abs/2311.09476; Gao et al. (2023) ALCE; Barnett et al. (2024) |
| Prompt layout: stable rules first, domain profile and `ENGINEERING_RIGOR` as further stable blocks, documents then question in the user turn, XML tags, strict JSON schemas without numeric bounds | `prompts.py` | Liu et al. (2024); architecture rules 3.2-3.3 |

LLM contextual prefixes are implemented (`contextual_prefix='llm'`; or call
`IndexBuilder.apply_llm_prefixes` with your own `Prefixer`). Provided as
hooks, not implementations: web-search fallback (inject any `Retriever` as
`fallback_retriever`), the retriever's own transforms (`retriever.transform_queries(question, wanted)` is preferred
when present), dense retrieval and reranking (inject the hybrid retrieval
model). GraphRAG entity graphs, trained routers and NLI-based citation
checking are not implemented; see section 10.

## 7. Adapting to an engineering domain

No code changes: register a `DomainProfile` and name it in the request.

```python
from pipeline_core import DomainProfile, register_profile, TaskRequest, make_llm
from pipeline_rag import PipelineRAGModel

register_profile(DomainProfile(
    name="offshore_rotating_equipment",
    description="Rotating equipment on offshore platforms: pumps, compressors, drivers and their inspection regimes.",
    glossary={"UT": "Ultrasonic thickness survey", "RPN": "Risk priority number from FMEA",
              "FoS": "Factor of safety - ratio of allowable to applied load"},
    units=["bar", "mm", "kW", "rpm", "mm/s"],
    document_types=["specification", "data sheet", "FMEA", "incident report", "maintenance procedure"],
    verification_rules=["Every pressure must state the reference (design, test, operating).",
                        "Vibration limits must cite the ISO 10816 zone."],
    risk_level="safety_critical",      # -> grading, verification, inline citations, abstention mandatory
    min_confidence_to_accept=0.85,     # -> NEEDS_REVIEW below 0.85
    retrieval_hints={"sparse_weight": 0.65},   # consumed by the injected hybrid retriever
    standards=["ISO 10816-3", "API 610", "ISO 9906"],
))

model = PipelineRAGModel(make_llm())
result = model.run_sync(TaskRequest(
    task="What UT survey requirement protects the casing corrosion allowance?",
    inputs={"documents": documents},
    domain="offshore_rotating_equipment",
))
```

What the profile changes: `risk_level` forces `grade_context`, `verify`,
`abstain_when_insufficient` and inline citations in both planners, adds
`ENGINEERING_RIGOR` to every stable system block, raises the reflection
effort (`max` for safety-critical) and turns unsupported statements or
citation coverage below `min_citation_coverage` into `NEEDS_REVIEW`;
`min_confidence_to_accept` sets the acceptance threshold; `glossary` drives
`glossary_expansion` (a "UT" question also retrieves "Ultrasonic thickness
survey") so lexical retrieval matches documents that spell the term out;
`system_text()` (glossary, units, standards, verification rules) is appended
to every stable system prompt (plan, transform, grade, generate, hop,
support, summary) so prompt caches stay warm per domain.
`examples/engineering_example.py` runs this end to end, including the
evaluation harness.

Dossier guidance for engineering corpora encoded in the defaults: identifiers
switch off HyDE and step-back and weight double in lexical grading; code,
tables and identifier lists are never compressed; numbers, units and clause
numbers must be copied exactly (generation rule 7); incident reports get
single-step plans for "what happened in INC-2041" and the agentic plan over
summary nodes for "what recurs"; requirement traces ("REQ -> design -> test
-> incident") are multi-step IRCoT plans.

## 8. Composing with the other five models

| Model | Direction | Interface |
| --- | --- | --- |
| Hybrid Retrieval | accepts | `PipelineRAGModel(retriever=HybridRetrievalModel(...))`: every query variant is sent through `await retrieve(query, k=candidates_per_variant, filters=None)`; results are fused with the built-in and inline indexes by RRF; `retriever.transform_queries(question, wanted)` is used instead of the local transforms when the retriever exposes it; `retriever.version` is part of the retrieval cache key |
| Answer Verification | accepts | `PipelineRAGModel(verifier=AnswerVerificationModel(...))`: invoked through `await verifier.run(request.derive(question, inputs={"answer": answer}, context=evidence, constraints=<remaining budget>))`; `result.usage` is charged with `ctx.charge` (architecture 3.1.3) and `result.output.to_verdict()` becomes `output.verdict`, its `score` the `support` signal, `passed=False` forces `NEEDS_REVIEW`. Any other `Verifier` is called through `verify(answer, task=question, evidence=context)` and may report spend as `details["usage"]` |
| Parallelization | accepts | `PipelineRAGModel(executor=ParallelizationModel(...).executor)`: retrieval fan-out (variants x sources) runs through `await executor.map(fn, jobs, concurrency=constraints.max_parallelism)`; on failure it falls back to `gather_limited` |
| Caching | accepts | `PipelineRAGModel(CachingModel.wrap(llm), cache=caching_model.cache)`: the wrapped client caches prompts (stable blocks first, never timestamps or ids); the `ResponseCache` caches retrieval results under `stable_hash("pipeline_rag.retrieve", source, index_version, query, k, filters)` with tags `pipeline_rag`, `index:<version>`, `source:<name>` for invalidation when an index changes. A retriever without a `version` is keyed by its identity and `model.retrieval_cache_epoch`; call `await model.invalidate_retrieval_cache()` after re-indexing it. The first cache error of a model is surfaced as a warning |
| Task Decomposition | provides | `TaskDecompositionModel(solver=rag.solver)`: each sub-task becomes one RAG run; `result.as_evidence()` returns the cited answer as `Evidence(retriever="provided")` for the next sub-task or a final verification pass |
| Any consumer | provides | `PipelineRAGModel` **is a** `Retriever` (`await model.retrieve(query, k=10, filters=None)` over the indexed corpus, no LLM calls) and exposes `answer()`, `generate()`, `route()`, `route_llm()`, `plan_for()`, `index()` and `invalidate_retrieval_cache()` |

The component never imports another model package; it accepts protocol
objects only. Usage of nested components rolls up through `ctx.charge()` and
one shared `Tracer` shows the whole pipeline (`tracer.tree()` contains
`pipeline_rag.run` with `plan`, `index`, `transform`, `retrieve`, `grade`,
`iterate`, `compress`, `generate`, `verify` spans and the `llm` spans inside
them; a component verifier's own `run` span nests under `verify`). The repository-level `examples/enterprise_pipeline.py` described in
`docs/ARCHITECTURE.md` section 5 is where all six are wired together.

## 9. Evaluation

Metrics (`evaluation.py`, all pure functions): `exact_match` and `token_f1`
against a reference answer (SQuAD-style normalisation, citation markers
stripped); `context_precision` and `context_recall` of retrieved unit ids
against labelled relevant ids (chunk ids match their parent document);
`faithfulness` (RAGAS: fraction of answer statements supported by the
context); `citation_metrics` (ALCE citation recall and precision over
`Statement`s); `abstention_metrics` (rate, precision and recall of abstaining
on unanswerable questions); `attribute_failure_point` maps each record to
Barnett et al.'s FP1 (missing content answered), FP2 (gold not retrieved),
FP3 (retrieved but lost in consolidation), FP4 (in context but not
extracted), FP5 (wrong format / run failed) or FP7 (incomplete).
`RAGEvaluator` separates FP2 from FP3 by comparing the ids every `retrieve*`
step produced (before grading and compression) with the evidence that
reached the generator.

`RAGEvaluator(model)` runs `QAExample`s through the model and judges
faithfulness and answer relevance with the LLM (`JUDGE_FAITHFULNESS_PROMPT`,
`JUDGE_RELEVANCE_PROMPT`, `effort=low`) when a real client is available or
with lexical proxies offline (`report.judge` says which). `EvaluationReport`
carries per-metric means, the abstention triple, route distribution,
escalation rate, failure-point counts, usage and release gates:
`faithfulness >= min_faithfulness`, `citation_recall >= min_citation_coverage`,
`answer_relevance >= 0.8`. `synthetic_qa(documents, n)` builds in-domain
questions whose reference sentence must be retrieved at rank 1 (the ARES
filter, heuristic); `synthetic_qa_llm` does the same with `SYNTHETIC_QA_PROMPT`;
`sample_corpus()` and `sample_qa()` are the labelled demo set (five
answerable questions, one unanswerable).

```bash
python -m pipeline_rag.evaluation --offline                          # sample corpus + labelled questions
python -m pipeline_rag.evaluation --documents ./docs --n 20 --json   # synthetic questions over your corpus
python -m pipeline_rag --offline --sample-corpus --evaluate "any task"
```

```python
from pipeline_rag.evaluation import RAGEvaluator, QAExample, sample_corpus, sample_qa, synthetic_qa
report = RAGEvaluator(model).evaluate_sync(sample_qa(), documents=sample_corpus())
print(report.summary())            # n=6 judge=heuristic exact_match=0.60 token_f1=0.37 faithfulness=1.00 ...
print(report.failure_points, report.route_distribution, report.gates)
report2 = await RAGEvaluator(model).evaluate(synthetic_qa(my_docs, n=20), documents=my_docs)
```

Offline, the extractive generator is faithful and fully cited by construction
(faithfulness 1.0, citation recall 1.0, abstention recall 1.0 on the sample
set), so those gates pass trivially; `exact_match`, `token_f1` and
`answer_relevance` are the informative offline numbers, and the lexical
answer-relevance proxy fails its 0.8 gate on the sample set (0.75). Treat the
offline run as a self-test, not a benchmark: automatic judges, context
relevance in particular, agree with humans only moderately (RAGAS 0.70),
so calibrate on 150-300 human-labelled questions per domain (ARES) and
re-measure judge agreement when the model changes.

## 10. Limitations and failure modes

- **Missing content answered anyway (FP1).** The gate, abstention and the
  escalation cap reduce it, but an LLM generator can still answer from
  parametric knowledge; in low-risk domains unsupported claims are listed, not
  rejected. Use a high-risk profile or an injected verifier when that matters
  (Barnett et al. 2024).
- **Thresholds do not transfer.** CRAG thresholds ranged from (0.59, -0.99)
  to (0.95, -0.91) across datasets (Yan et al. 2024); the defaults 0.5 / -0.9
  and the lexical grader's 30 % relevance floor must be calibrated per corpus.
- **Lexical heuristics are proxies.** Offline grading, support judgement,
  IsRel and the extractive generator use token overlap without stemming
  ("updated" does not match "update"); they are deterministic and honest
  about coverage but can miss paraphrases and can pick a topical sentence
  over the one carrying the asked-for detail. The multi-hop
  answer-continuation rule mitigates the latter for incident-style records
  only.
- **Confidence is heuristic**, not a calibrated probability: for extractive
  answers coverage and support are 1.0 by construction, so the self-reported
  and evidence-quality terms carry the information. The weights follow the
  literature's ordering (faithfulness > citation recall > self-report;
  Es et al. 2024; Gao et al. 2023) but were not fitted.
- **Compression trades citation recall for correctness** (ALCE: 73.6 ->
  68.9/65.3 with summaries/snippets). Compression here is extractive and
  keeps spans, but a dropped sentence can remove the only support for a
  later statement; disable `compress` when citations are mandatory and the
  context fits.
- **Position effects remain** inside a unit: ordering is per unit (best
  first, second best last; Liu et al. 2024), not per sentence.
- **HyDE and step-back hallucinate or over-generalise** private identifiers
  (Gao et al. 2023 HyDE; Zheng et al. 2024); both are withheld whenever
  identifiers are detected, which also withholds them for questions that
  merely mention a part number.
- **The built-in index is BM25 only.** Vocabulary mismatch between a
  plain-English question and a formal specification needs the dense channel of
  the hybrid retrieval model; the built-in index is a standalone fallback, not
  a production retriever.
- **The hierarchy is structural, not clustered.** RAPTOR's UMAP + GMM
  clustering is replaced by one summary per document plus one corpus node;
  offline summaries are lead sentences. Corpus-wide themes across many
  documents therefore read as a digest and are flagged `NEEDS_REVIEW` when
  the question's terms do not appear in the nodes (IsRel = false).
- **Automatic judges are unreliable**, especially for context relevance
  (RAGAS 0.70 agreement; Es et al. 2024; Saad-Falcon et al. 2024); treat
  evaluation scores comparatively and validate against human labels.
- **Untrusted evidence.** Prompts mark retrieved text as data and the loop is
  bounded (hops, escalations, call and cost budgets), but there is no
  ingestion-time scan for instruction-like content and no per-source trust
  weighting (Greshake et al. 2023; Zou et al. (2025), *PoisonedRAG: Knowledge
  Corruption Attacks to Retrieval-Augmented Generation of Large Language
  Models*, https://arxiv.org/abs/2402.07867); a poisoned document is graded
  like any other.
- **Context precision / recall are label-based.** Architecture 4.6 lists
  them among the LLM-judged RAGAS metrics; here they compare retrieved ids
  with labelled relevant ids (faithfulness and answer relevance do have LLM
  judges), so they need labelled or synthetic questions.
- **Domain document types are not used for indexing.** `DomainProfile.
  document_types` does not change chunk size, code/table protection or
  prefix style; only `risk_level`, `min_confidence_to_accept` and the
  glossary change behaviour (`retrieval_hints` are left to an injected
  retriever).
- **Nested spend is charged after the fact.** A component verifier receives
  the remaining budget, but `BudgetTracker.charge` raises only after a paid
  call, so the final spend can exceed `max_cost_usd` by the last call; the
  overspend is recorded as a warning, not discarded.
- **Not implemented** from the dossier: GraphRAG community summaries, trained
  complexity routers, NLI-based citation checking, abstractive compression,
  an index store with versioned blue/green rebuilds (the `version`
  fingerprint and cache tags are the hooks for it).

## 11. How it was built

1. **Research** (`docs/research/pipeline-rag.json`,
   `prompting-techniques.json`, `pipeline-architecture.json`). The dossier's
   summary settled the shape: RAG as a modular, conditional, looping graph
   (Gao et al. 2024 Modular RAG) with a complexity router at the entry
   (Jeong et al. 2024) and upward-only escalation on abstention (Li et al.
   2024). Two empirical facts drove the context stage: reader accuracy falls
   from 75.8 % to 53.8 % when the gold passage moves from first to tenth and
   saturates near k = 20 (Liu et al. 2024), and RAG fails at seven predictable
   seams (Barnett et al. 2024). The technique entries fixed the defaults: RRF
   k = 60 and top-50 per variant (Cormack et al. 2009), 3-5 query variants
   with HyDE gated behind a domain-knowledge flag (Ma et al. 2023; Zheng et
   al. 2024; Gao et al. 2023 HyDE), CRAG thresholds 0.5 / -0.9 with one
   fallback round (Yan et al. 2024), a 2,000-token evidence budget (Sarthi et
   al. 2024; Xu et al. 2024), S = 4 hops (Trivedi et al. 2023), Self-RAG
   weights 1.0 / 1.0 / 0.5 and trigger 0.2 (Asai et al. 2024), FLARE
   threshold in 0.4-0.8 (Jiang et al. 2023), citation recall >= 0.8 and
   faithfulness >= 0.9 as release gates (Gao et al. 2023 ALCE; Es et al.
   2024). The prompting dossier contributed the layout rules (stable blocks
   first, documents before the question, XML delimiters, retrieved text is
   data: Greshake et al. 2023); the RAG dossier's rule that generations are
   critiqued against evidence rather than trusted as produced (Asai et al.
   2024; Gao et al. 2023 ALCE) is why self-reported confidence has the
   lowest weight.
2. **Design decisions.** A declarative `RAGPlan` separates reasoning from
   execution so orchestrators can inspect, override (`inputs["k"]`) and
   audit it; the heuristic planner encodes the same rules the LLM planner is
   prompted with and seeds the LLM call, so offline and online behaviour
   agree and the LLM refines rather than invents. Every LLM-facing schema
   (`RAGPlan`, `QueryVariants`, `GradeBatch`, `GenerationOutput`, `HopStep`,
   `SupportJudgement`) carries no numeric bounds (constrained decoders reject
   them); bounds are enforced by validators and `normalized()`. Dependencies
   are protocol objects with built-in stand-ins (BM25 index, Self-RAG judge,
   `gather_limited`, no cache) so the package works alone and becomes the full
   pipeline when wired. Budgets are checked before every call with a reserve
   for generation, so a tight `max_llm_calls` degrades the plan rather than
   failing the run. Abstention is a first-class output with a confidence cap,
   escalation is upward only and bounded, and the status rules make
   high-risk domains conservative by default.
3. **Implementation.** Pure-function cores (RRF, CRAG decision, lexical
   relevance, extractive compression, position ordering, statement splitting
   and citation resolution, critique maths, hop scheduling, metrics and
   failure-point attribution) are unit-tested without a model; LLM paths are
   covered with `ScriptedLLM` (planned path, normalisation and domain
   enforcement, plan failure fallback, abstention escalation, budgets); the
   `MockLLM` path covers routing, iteration, hierarchy, escalation, domain
   adaptation and composition with fake `Retriever`/`Verifier`/`Executor`/
   `ResponseCache` objects. `tests/` holds 107 tests that run in a few
   seconds; the CLI, `python -m pipeline_rag.evaluation` and both examples run
   with no credentials.
