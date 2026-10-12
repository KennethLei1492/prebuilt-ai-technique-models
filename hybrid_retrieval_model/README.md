# Hybrid Retrieval (`hybrid_retrieval`)

Lexical (BM25) + dense retrieval over a chunked document store, fused with
Reciprocal Rank Fusion or a normalised convex combination, optionally
expanded (multi-query, HyDE, Query2doc, keyword extraction) and reranked
(RankGPT-style listwise LLM, pointwise LLM, cross-encoder), **planned per
query** from the task text and the engineering domain profile. It implements
the `pipeline_core.protocols.Retriever` protocol, so the Pipeline RAG,
Answer Verification and Task Decomposition models consume it unchanged.

Everything runs offline: without credentials `make_llm()` returns a `MockLLM`,
the `HeuristicPlanner` decides the plan, and the deterministic
`HashingEmbedder` stands in for a production embedding model.

---

## 1. Purpose

Engineering corpora mix exact tokens (part numbers `PS-4471`, error codes
`ERR-1203`, clause ids, material grades `ASTM A890 Grade 4A`) with paraphrased
natural language. Dense-only retrieval under-performs BM25 out of domain
(Thakur et al. 2021), while BM25 alone misses vocabulary mismatch (Karpukhin
et al. 2020). This package keeps both channels, fuses them, pushes metadata
filters into every channel, reranks a bounded candidate set, and returns
`Evidence` with provenance (per-leg scores and ranks, section path, character
span) plus a calibrated-by-diagnostics confidence so downstream generators can
abstain on weak evidence.

## 2. Architecture

```
                 TaskRequest(task, inputs{documents, query, k, filters, plan}, domain, constraints)
                                              │
                                              ▼
 ┌──────────────────────────────  HybridRetrievalModel.plan()  ──────────────────────────────┐
 │  index inputs["documents"] (DocumentStore + Chunker)                                      │
 │  analyze_query() -> QueryFeatures                                                          │
 │  offline / deterministic / budget=0 ──▶ HeuristicPlanner        online ──▶ PLANNER_PROMPT  │
 │                                              │                 (1 structured LLM call,      │
 │                                              │                  PlannerOutput -> clamp)     │
 │                                              └────────────┬──────────────┘                 │
 │                                                     apply_budget()  ──▶  RetrievalPlan     │
 └────────────────────────────────────────────────────────────┬──────────────────────────────┘
                                                              ▼
 ┌──────────────────────────────  HybridRetriever.search(plan)  ─────────────────────────────┐
 │ filters ──▶ allow-list pushed into every channel (never post-filtered)                     │
 │ query transforms (query_transform.py): multi_query | hyde | query2doc | keywords           │
 │        │                                                                                   │
 │        ▼  for each query variant, concurrently                                             │
 │   ┌─────────────┐      ┌─────────────┐                                                     │
 │   │ BM25Index   │      │ DenseIndex  │   sparse.py / dense.py                              │
 │   │ (Okapi/+)   │      │ (cosine)    │                                                     │
 │   └──────┬──────┘      └──────┬──────┘                                                     │
 │          └───────┬────────────┘                                                            │
 │                  ▼  fusion.py: rrf | convex(theoretical min-max) | dbsf | interleave        │
 │           per-variant fused list ──▶ RRF across variants                                   │
 │                  ▼  rerank.py: listwise (sliding window) | pointwise | cross_encoder | none │
 │           top rerank_depth ──▶ top k_final  ──▶ list[Evidence] (retriever=sparse|dense|    │
 │                                                  fused|reranked, per-leg metadata)         │
 │           query_coverage(), channel_overlap() ──▶ retrieval_confidence()                   │
 └────────────────────────────────────────────────────────────┬──────────────────────────────┘
                                                              ▼
            HybridRetrievalModel.execute(): threshold = max(profile.min_confidence_to_accept,
            constraints.min_confidence); below it -> NEEDS_REVIEW; TaskResult(output=RetrievalOutput,
            evidence, confidence, confidence_basis, metadata{plan, index, threshold})
```

| Module | Contents |
| --- | --- |
| `index.py` | `Document`, `Chunk`, `Chunker` (`fixed`, `sentence`, `structure`, `token_budget`), `match_filter`, `DocumentStore` (add/remove/filter, JSON persistence) |
| `sparse.py` | `Tokenizer` (identifier-preserving, sub-tokens, optional stemmer/stopwords), `BM25Index` (Okapi / BM25+), `RankBM25Index` adapter, `make_bm25` |
| `dense.py` | `Embedder` protocol, `HashingEmbedder`, `CallableEmbedder`, `VoyageEmbedder`, `SentenceTransformerEmbedder`, `ANNBackend` hook, `DenseIndex` |
| `fusion.py` | `reciprocal_rank_fusion`, `convex_combination`, `normalize_scores`, `distribution_based_score_fusion`, `interleave`, `channel_overlap`, `fuse` |
| `rerank.py` | `Reranker` protocol, `LLMListwiseReranker`, `LLMPointwiseReranker`, `CrossEncoderReranker`, `NoopReranker`, `sliding_windows`, `repair_permutation` |
| `query_transform.py` | `MultiQueryExpander`, `HyDE`, `Query2Doc`, `KeywordExtractor`, `content_overlap` drift guard |
| `retriever.py` | `HybridRetriever` (the `Retriever`), `RetrievalResult`, `ChannelStats`, `query_coverage`, `retrieval_confidence` |
| `plan.py` | `RetrievalPlan`, `PlannerOutput`, `FilterClause`, `QueryFeatures`, `analyze_query`, `HeuristicPlanner`, `apply_budget`, `estimate_llm_calls` |
| `prompts.py` | Versioned `PromptTemplate`s: planner, multi-query, HyDE, Query2doc, keywords, listwise and pointwise rerank |
| `model.py` | `HybridRetrievalModel` (`spec`, `plan()`, `execute()`, direct API), `RetrievalOutput`, `ResultRow` |
| `evaluation.py` | `ndcg_at_k`, `recall_at_k`, `precision_at_k`, `mrr`, `success_at_k`, `evaluate`, `ablation`, `synthetic_engineering_corpus` |
| `cli.py` | `python -m hybrid_retrieval "task" [--domain] [--offline] [--json] [--input k=v] [--documents ...] [--evaluate]` |

## 3. How it reasons about a prompt

### 3.1 The plan (`RetrievalPlan`)

| Field | Meaning |
| --- | --- |
| `query`, `query_type` | The query (task text or `inputs["query"]`); `identifier \| keyword \| natural_language \| exploratory \| mixed` |
| `query_variants` | Original query first, then reformulations; variants are run and fused with RRF |
| `use_sparse`, `use_dense` | Channels to run (both on unless a bare identifier lookup; planners never switch the sparse channel off) |
| `sparse_weight`, `dense_weight` | Channel weights (sum to 1) for the score-based fusions `convex` and `dbsf`; the rank-based `rrf` and `interleave` ignore them (unweighted RRF, Cormack et al. 2009) |
| `fusion` | `rrf \| convex \| dbsf \| interleave` |
| `k_sparse`, `k_dense`, `k_final` | Fusion window per channel and passages returned |
| `rerank`, `rerank_depth` | `none \| listwise \| pointwise \| cross_encoder` and how many fused candidates are reranked |
| `query_transforms` | Subset of `multi_query \| hyde \| query2doc \| keywords` |
| `filters`, `mandatory_filters` | Planner-inferred `FilterClause`s (only on fields that exist in the corpus) and caller-supplied ACL/tenant filters the planner cannot remove |
| `rationale`, `source`, `features` | Audit trail: why, who decided (`heuristic \| llm \| explicit`), the `QueryFeatures` used |
| `expected_llm_calls`, `degradations` | Cost estimate and the budget-driven changes `apply_budget` made |

The plan is attached to `result.metadata["plan"]` and rendered in
`output_text` (`Plan: ...`, `Rationale: ...`, `Degradations: ...`).

### 3.2 Heuristic planner (offline, `constraints.deterministic`, or no budget)

`analyze_query` computes token count, identifier density, number density,
question words, exploratory cues (`similar`, `before`, `history`, ...) and
quoted phrases. Rules:

1. `sparse_weight` starts at the profile's `retrieval_hints["sparse_weight"]`
   (else `default_sparse_weight` = 0.3), +0.25 if at least half the tokens are
   identifiers, +0.10 if some are, +0.10 for quoted phrases, -0.10 for an
   identifier-free question; clamped to [0.1, 0.9].
2. `query_type`: identifier (>= 50 % identifier tokens), exploratory (cues),
   keyword (<= 3 content tokens, no question word), natural_language (question
   word or >= 6 tokens), else mixed.
3. `fusion`: `convex` when the profile supplies a calibrated weight or the query
   is identifier/keyword (exact terms must dominate); otherwise `rrf`.
4. Both channels stay on except for a bare identifier of <= 2 tokens (dense off).
5. Transforms: `hyde` for exploratory or short natural-language queries
   (< `expansion_max_query_tokens` content tokens); `multi_query` for
   natural-language/exploratory queries of >= 6 tokens; `query2doc` for short
   keyword queries without identifiers; `keywords` (free, heuristic) for
   questions longer than 12 tokens.
6. `rerank`: the configured tier for natural-language, exploratory and mixed
   queries; always for `high`/`safety_critical` profiles (depth 30 for
   safety-critical); never for bare identifier/keyword lookups in low- or
   medium-risk domains (`general`, `software_engineering`, `data_engineering`).
7. `apply_budget` fits the plan to `Constraints.max_llm_calls`: transforms are
   dropped first (`query2doc`, `hyde`, `multi_query`), then pointwise becomes
   listwise, then the listwise depth shrinks to one window, then reranking is
   disabled. `deterministic` removes every step that samples an LLM (a
   configured `cross_encoder` is free and reproducible, so it stays). Nothing fails.
8. `max_cost_usd` and `max_latency_s` cannot be modelled up front, so they are
   enforced at run time: before every transform, rerank window or grading
   fan-out `budget_exhausted(ctx)` is consulted and an exhausted budget skips
   the step with a warning; a `BudgetExceededError` raised by a call that spent
   the last of the budget keeps the fused candidates already computed. Budgets
   never empty the evidence list (spec 3.1.4 / 3.2).

Example (`--offline`): `"ERR-1203"` in `software_engineering` ->
`query_type=identifier channels=sparse fusion=convex(sparse=0.80) rerank=none llm_calls~0`;
`"Has anything similar to the pump cavitation event happened before?"` in
`general` -> `exploratory channels=sparse+dense fusion=rrf(sparse=0.20) rerank=listwise@20 transforms=['hyde','multi_query']`.

### 3.3 LLM planner (online)

One structured call (`PLANNER_PROMPT`, version 1.0.0, `effort="high"`,
`output_schema = sanitize_schema(PlannerOutput.model_json_schema())`). The
stable system blocks hold the role, nine numbered decision rules, two few-shot
examples, `ctx.profile.system_text()` and `ENGINEERING_RIGOR` for high-risk
profiles; the user turn holds `<task>`, `<query>`, `<query_features>`,
`<corpus>` (document count and metadata fields with sample values),
`<budget>` and `<defaults>`. The answer is validated with
`PlannerOutput.model_validate`, clamped by `RetrievalPlan.from_planner_output`
(weights to [0,1], `k_sparse`/`k_dense` to [1,1000], `k_final` to [1,100],
`rerank_depth` to [1,200] and never below `k_final`, filters restricted to
existing fields, at most 4 variants, the sparse channel always kept on) and
then budgeted. Any `LLMError`/`StructuredOutputError` falls back to the
heuristic plan with a warning; a planning call that exhausts the cost or
latency budget falls back to a heuristic plan without LLM steps. An explicit
plan in `inputs["plan"]` bypasses both.

Offline note: `ctx.offline` is true for any `MockLLM`, including
`ScriptedLLM`, and the heuristic planner is used whenever it is (spec 3.1.2);
the LLM planner runs only with a real client. The transforms and rerankers the
heuristic plan selects still go through `ctx.llm_call`, so a `ScriptedLLM`
drives them deterministically in tests. The LLM-*planned* path is covered by
`tests/hr_fixtures.py::OnlineScriptedLLM`, a scripted `BaseLLMClient` that is
not a `MockLLM` subclass.

## 4. Quick start

### Python

```python
import asyncio
from pipeline_core import make_llm, TaskRequest, Constraints
from hybrid_retrieval import HybridRetrievalModel

model = HybridRetrievalModel(make_llm())                       # MockLLM without credentials
model.index([
    {"id": "spec-pump", "text": "# Pump specification PS-4471\n## 4.2 Impeller\nThe impeller shall be duplex stainless steel ASTM A890 Grade 4A.",
     "metadata": {"type": "specification", "revision": "B"}},
    {"id": "runbook", "text": "ERR-1203 is raised when the heartbeat from the field bus is lost.", "metadata": {"type": "runbook"}},
])

result = model.run_sync("Which clause specifies the impeller material?", domain="mechanical_engineering")
print(result.summary())          # [hybrid_retrieval] status=... confidence=0.xx calls=n ...
print(result.output_text)        # plan, rationale, channels, ranked passages, confidence basis
print(result.metadata["plan"])   # the RetrievalPlan as JSON

# Direct Retriever-protocol API (no LLM calls by default)
evidence = asyncio.run(model.retrieve("ERR-1203", k=3, filters={"type": "runbook"}))
for ev in evidence:
    print(ev.id, ev.score, ev.retriever, ev.metadata["channels"], ev.metadata["sparse_rank"], ev.metadata["dense_rank"])

# Budgets degrade the plan instead of failing
res = model.run_sync(TaskRequest(task="previous cavitation incidents", constraints=Constraints(max_llm_calls=0)))
print(res.metadata["plan"]["degradations"])

model.save_index("index.json"); model.load_index("index.json")
```

Inputs understood by `run()`: `inputs["documents"]` (list of dicts with
`id`/`text`/`metadata`, strings, `Document`s or `Evidence`; upserted by id),
`inputs["query"]` (overrides the task text as the query), `inputs["k"]`,
`inputs["filters"]` (mandatory), `inputs["plan"]` (explicit plan fields).
When the store is empty and the request carries upstream `context` evidence,
that evidence is indexed so the model can re-rank it.

### CLI

```bash
cd "<repo root>"
export PYTHONPATH="common:hybrid_retrieval_model"        # Windows Python splits on ';' : PYTHONPATH="common;hybrid_retrieval_model"
python -m hybrid_retrieval --offline "Find the clauses that specify the pump impeller material and its corrosion allowance"
python -m hybrid_retrieval "ERR-1203 cause" --domain software_engineering --documents ./docs --documents extra.md --json
python -m hybrid_retrieval "corrosion allowance" --input k=5 --input 'filters={"type":"specification"}' --max-llm-calls 2
python -m hybrid_retrieval --offline --evaluate "any task"          # nDCG/Recall/MRR + sparse/dense/hybrid ablation
```

Without `--documents`/`--load-index` the CLI indexes the built-in synthetic
engineering corpus (24 documents) so the command always returns a meaningful
result. `--documents` takes one path per flag (repeat it), so it can appear
before or after the task. Windows Python uses `;` as the `PYTHONPATH`
separator; with `:` the interpreter reports `No module named hybrid_retrieval`.

## 5. Configuration (`HybridRetrievalConfig`)

| Field | Default | Justification (dossier source) |
| --- | --- | --- |
| `chunk_strategy` | `structure` | Recursive, boundary-aware splitting beat semantic chunkers on natural documents: Qu et al. (2024), *Is Semantic Chunking Worth the Computational Cost?*, https://arxiv.org/abs/2410.13070; Smith & Troynikov (2024), *Evaluating Chunking Strategies for Retrieval*, https://www.trychroma.com/research/evaluating-chunking |
| `chunk_size_tokens` | `400` | 200-400 tokens gave the best token-level recall/IoU; 800/400 was worst: Smith & Troynikov (2024); Qu et al. (2024). **Deviates from the spec's 512.** |
| `chunk_overlap_tokens` | `0` | Zero overlap was best for 200-400-token chunks; 10-25 % only helps small-context embedders: Smith & Troynikov (2024). **Deviates from the spec's 64.** |
| `prepend_headers` | `True` | Contextual chunk headers cut top-20 retrieval failures by 35-49 %: Anthropic (2024), *Introducing Contextual Retrieval*, https://www.anthropic.com/news/contextual-retrieval; title prefixes: Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering*, https://arxiv.org/abs/2004.04906 |
| `bm25_k1` | `1.2` | k1 in [1.2, 2.0]: Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*, https://nowpublishers.com/article/Details/INR-019 |
| `bm25_b` | `0.75` | Robertson & Zaragoza (2009) |
| `bm25_variant` | `okapi` | Classic BM25; `plus` lower-bounds long-document term scores (Robertson & Zaragoza 2009 discussion of length normalisation) |
| `bm25_delta` | `1.0` | BM25+ delta, used only with `variant="plus"` |
| `bm25_backend` | `python` | Pure-Python inverted index (no required dependency); `rank_bm25` adapts the optional package. Robertson & Zaragoza (2009): BM25 must remain a first-class channel |
| `preserve_identifiers` | `True` | Whole identifiers plus sub-tokens so part numbers and symbols match exactly: Robertson & Zaragoza (2009); Li et al. (2025), *CoIR: A Comprehensive Benchmark for Code Information Retrieval Models*, https://arxiv.org/abs/2407.02883 |
| `remove_stopwords` | `True` | In small collections idf alone under-penalises function words and short passages win on them (Robertson & Zaragoza 2009 on idf); identifiers are never dropped |
| `embedding_dimension` | `512` | Dimension of the offline `HashingEmbedder` baseline (replaced by a production embedder; see Muennighoff et al. (2023), *MTEB*, https://arxiv.org/abs/2210.07316 on choosing one in-domain) |
| `dense_score_range` | `(-1.0, 1.0)` | Theoretical cosine range for theoretical min-max normalisation: Bruch et al. (2023), *An Analysis of Fusion Functions for Hybrid Retrieval*, https://arxiv.org/abs/2210.11934 |
| `fusion` | `rrf` | Zero-configuration default: unweighted RRF with k = 60 (the plan's weights apply only to `convex`/`dbsf`): Cormack et al. (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf; the planner switches to `convex` when a domain weight exists or the query is an identifier/keyword lookup (Bruch et al. 2023) |
| `rrf_k` | `60` | Cormack et al. (2009); Elastic (2024), *Reciprocal rank fusion (RRF) retriever*, https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html; tuned constants do not transfer (Bruch et al. 2023) |
| `default_sparse_weight` | `0.3` | Semantic alpha 0.7 was the best default with [0.6, 0.8] consistently good: Bruch et al. (2023); corroborated by Weaviate (2023), *Unlocking the Power of Hybrid Search*, https://weaviate.io/blog/hybrid-search-fusion-algorithms. **Deviates from the spec's 0.5.** |
| `normalization` | `theoretical` | Theoretical min-max beat observed min-max and z-score: Bruch et al. (2023); BM25 (no known range) falls back to observed min-max |
| `k_sparse`, `k_dense` | `100` | Rank window `max(size, 100)`: Elastic (2024); deep candidate pools make fusion gains real: Ma et al. (2021), *A Replication Study of Dense Passage Retriever*, https://arxiv.org/abs/2104.05740. **Deviates from the spec's 50.** |
| `k_final` | `10` | 5-20 passages; reader accuracy saturates near 20: Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*, https://arxiv.org/abs/2307.03172; Anthropic (2024) |
| `rerank` | `listwise` | Reranker tier used when the planner enables reranking; listwise LLM reranking is the strongest zero-shot reranker: Sun et al. (2023), *Is ChatGPT Good at Search? (RankGPT)*, https://arxiv.org/abs/2304.09542; Pradeep et al. (2023), *RankZephyr*, https://arxiv.org/abs/2312.02724 |
| `rerank_depth` | `20` | LLM tier over the top 20-30 costs one-fifth of full depth: Sun et al. (2023) |
| `rerank_window`, `rerank_stride` | `20`, `10` | RankGPT sliding window w = 20, s = 10: Sun et al. (2023); Pradeep et al. (2023) |
| `rerank_passage_chars` | `1200` | ~300-token passage truncation in rerank prompts: Sun et al. (2023); truncation guidance: Cohere (2025), *Reranking Best Practices*, https://docs.cohere.com/docs/reranking-best-practices |
| `pointwise_concurrency` | `8` | Parallel pointwise grades (bounded by `Constraints.max_parallelism`); monoT5-style pointwise pattern: Nogueira et al. (2020), *Document Ranking with a Pretrained Sequence-to-Sequence Model*, https://arxiv.org/abs/2003.06713 |
| `multi_query_n` | `3` | Paraphrases fused with RRF (Cormack et al. 2009) |
| `hyde_n` | `4` | N = 4-8 hypothetical passages averaged with the query vector: Gao et al. (2023), *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*, https://arxiv.org/abs/2212.10496 |
| `query2doc_repeat` | `5` | Query repeated 5x before the pseudo-document: Wang et al. (2023), *Query2doc: Query Expansion with Large Language Models*, https://arxiv.org/abs/2303.07678 |
| `query2doc_max_tokens` | `128` | Pseudo-document cap: Wang et al. (2023) |
| `expansion_max_query_tokens` | `6` | Router gate: expansion only for short/exploratory queries: Gao et al. (2023); Wang et al. (2023) |
| `expansion_min_overlap` | `0.1` | Drift guard against plausible-but-wrong expansions: Wang et al. (2023) pitfall |
| `weak_evidence_confidence` | `0.3` | Below this the result warns so generators can abstain: Cohere (2025); Anthropic (2024) |
| `direct_rerank` | `False` | Whether the protocol API `retrieve()` runs the LLM reranker (costs calls outside a `RunContext`) |
| `max_documents_in_output_text` | `10` | Rows rendered in `output_text` |

## 6. Techniques implemented

| Technique | Where | Citation(s) |
| --- | --- | --- |
| Okapi BM25 / BM25+ with identifier-aware tokenisation | `sparse.py` | Robertson & Zaragoza (2009); Thakur et al. (2021), *BEIR*, https://arxiv.org/abs/2104.08663 |
| Dense bi-encoder retrieval with asymmetric query/document embedding and pre-filtering | `dense.py` | Karpukhin et al. (2020); Wang et al. (2022), *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)*, https://arxiv.org/abs/2212.03533; Xiao et al. (2024), *C-Pack: Packed Resources For General Chinese Embeddings (BGE)*, https://arxiv.org/abs/2309.07597; Patel et al. (2024), *ACORN*, https://arxiv.org/abs/2403.04871 |
| Reciprocal Rank Fusion | `fusion.py` | Cormack et al. (2009); Elastic (2024) |
| Normalised convex combination (theoretical min-max, min-max, z-score) | `fusion.py` | Bruch et al. (2023); Weaviate (2023) |
| Distribution-based score fusion, interleaving | `fusion.py` | Baselines/diagnostics around the fusion interface recommended by Bruch et al. (2023) |
| RankGPT-style listwise sliding-window reranking with structured permutation output and deterministic repair | `rerank.py` | Sun et al. (2023); Pradeep et al. (2023) |
| Pointwise 0-10 LLM grading with rationale, parallel, budget-aware | `rerank.py` | Nogueira & Cho (2019), *Passage Re-ranking with BERT*, https://arxiv.org/abs/1901.04085; Nogueira et al. (2020) |
| Cross-encoder adapter | `rerank.py` | Nogueira & Cho (2019); Thakur et al. (2021): BM25 + cross-encoder was best on 16/18 BEIR datasets |
| Multi-query expansion fused by RRF | `query_transform.py` | Cormack et al. (2009) |
| HyDE (hypothetical documents averaged with the query embedding) | `query_transform.py` | Gao et al. (2023) |
| Query2doc (query x5 + pseudo-document) on the lexical channel | `query_transform.py` | Wang et al. (2023) |
| Router-gated, drift-guarded expansion | `plan.py`, `query_transform.py` | Gao et al. (2023); Wang et al. (2023); Gospodinov et al. (2023), *Doc2Query--: When Less is More*, https://arxiv.org/abs/2301.03266 (grounding filters for generated text) |
| Structure-aware chunking at 400/0 with contextual headers | `index.py` | Smith & Troynikov (2024); Qu et al. (2024); Anthropic (2024); Luan et al. (2021), *Sparse, Dense, and Attentional Representations for Text Retrieval*, https://arxiv.org/abs/2005.00181 (short passages for single-vector indexes) |
| Metadata filters pushed into every channel, mandatory ACL filters | `index.py`, `retriever.py` | Patel et al. (2024) |
| Declarative, auditable retrieval plan with budget degradation | `plan.py` | Dossier architecture recommendations 1-2; Sun et al. (2023) and Wang et al. (2023) cost figures |
| Confidence from coverage / channel agreement / fill / rerank | `retriever.py` | Dossier observability recommendation (channel overlap as drift signal); Cohere (2025) on detecting "nothing relevant" |
| nDCG@k, Recall@k, MRR, Precision@k, Success@k; ablation with BM25 baseline | `evaluation.py` | Järvelin & Kekäläinen (2002), *Cumulated Gain-Based Evaluation of IR Techniques*, https://www.fi.muni.cz/~xkohout7/Research/clanky_cizi/evaluation/p422-jarvelin.pdf; Ma et al. (2021); Lin et al. (2021), *Pyserini*, https://arxiv.org/abs/2102.10073 |
| Prompt layout: stable rules + examples first, passages then query last, XML tags, data-not-instructions rule, strict JSON schemas validated in Python | `prompts.py` | Anthropic (2025), *Claude prompting best practices - Long context prompting*, https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices (long documents before the query, XML-tagged inputs, quote-then-answer grounding) |
| Budget-aware degradation at run time (cost, latency and call budgets checked before every LLM call; fused order kept when exhausted; `reranked` reported only when a reranker changed the order) | `retriever.py`, `rerank.py`, `query_transform.py`, `plan.py` | Sun et al. (2023) and Wang et al. (2023) latency/cost figures motivate the gating; Cohere (2025) on bounding rerank depth |

Provided as hooks, not implementations: `ANNBackend` (HNSW/FAISS),
`VoyageEmbedder`, `SentenceTransformerEmbedder`, `CrossEncoderReranker`
(all lazy imports). Learned-sparse (SPLADE) and multi-vector (ColBERTv2/PLAID)
channels from the dossier are not implemented; the fusion layer accepts any
additional `(doc_id, score)` list, so they can be added as a third channel.

## 7. Adapting to an engineering domain

No code changes: register a `DomainProfile` and name it in the request.

```python
from pipeline_core import DomainProfile, register_profile, TaskRequest
from hybrid_retrieval import HybridRetrievalModel

register_profile(DomainProfile(
    name="rotating_equipment",
    description="Pumps, compressors and turbines: specifications, datasheets, incidents, runbooks.",
    units=["bar", "degC", "MPa", "mm", "rpm"],
    document_types=["specification", "datasheet", "incident", "runbook"],
    risk_level="safety_critical",            # -> mandatory listwise rerank at depth 30, ENGINEERING_RIGOR in the planner prompt
    min_confidence_to_accept=0.8,            # -> NEEDS_REVIEW below 0.8
    retrieval_hints={"sparse_weight": 0.65}, # -> convex fusion, lexical weight 0.65 (+/- query adjustments)
    standards=["API 610", "API 682"],
))

model = HybridRetrievalModel(domain="rotating_equipment")
model.index(documents)
result = model.run_sync(TaskRequest(
    task="What corrosion allowance is specified for the pump casing?",
    domain="rotating_equipment",
    inputs={"filters": {"type": "specification"}},   # mandatory filter: pushed into BM25 and dense search
))
```

What the profile changes: `retrieval_hints["sparse_weight"]` seeds the convex
weight and switches fusion to `convex`; `risk_level` forces reranking (depth
30 when safety-critical) and adds `ENGINEERING_RIGOR` to the planner's stable
system block; `min_confidence_to_accept` sets the review threshold;
`system_text()` (glossary, units, standards, verification rules) is appended
to every stable system prompt (planner, transforms, rerankers) so prompt
caches stay warm per domain. `examples/engineering_example.py` runs this end
to end, including budget degradation, persistence and evaluation.

Dossier guidance encoded in the defaults for engineering corpora: identifier-
aware lexical analysis for code and logs (camelCase/snake_case sub-tokens);
clause/heading-aware chunking with the section path kept in metadata and in
the indexed header; filters on `type`, `revision`, `severity`, `service`
pushed into the channels; higher lexical weight for specifications and part
catalogues; HyDE for exploratory "has this happened before" incident queries.

## 8. Composing with the other five models

| Model | Direction | Interface |
| --- | --- | --- |
| Pipeline RAG | consumes | `HybridRetrievalModel` **is a** `Retriever`: `await retrieve(query, k=10, filters=None) -> list[Evidence]`. Evidence carries `source`, `span`, `metadata[section_path, doc_id, sparse_rank, dense_rank, fused_score, channels]`, ready for `render_evidence()` with documents before the question |
| Answer Verification | consumes | Same `Retriever` for retrieve-verify-retrieve loops; `confidence < weak_evidence_confidence` and the `"weak evidence"` warning tell the verifier to flag insufficient evidence instead of answering |
| Task Decomposition | consumes | Sub-questions become `inputs["query"]`/`inputs["plan"]["query_variants"]`; the planner skips its own expansion when variants are supplied (`SubTaskSolver = model.run`) |
| Parallelization | accepts | `HybridRetrievalModel(executor=...)` routes pointwise-rerank fan-out through an `Executor`; channel legs already run concurrently with `asyncio` |
| Caching | accepts | `HybridRetrievalModel(CachingModel.wrap(llm))`; prompts keep stable blocks first (rules, examples, profile) and never contain timestamps or ids, so planner/rerank prefixes are cacheable; `DocumentStore.version` and `RetrievalPlan` JSON are the natural cache keys for fused candidates |

The component never imports another model package; it accepts protocol objects
(`Executor`, any `BaseLLMClient`) and exposes `Retriever`.

## 9. Evaluation

Metrics (`evaluation.py`): `ndcg_at_k` (graded, headline), `recall_at_k`,
`precision_at_k`, `mrr`, `success_at_k`; `evaluate(retriever, judgments, k=10)`
collapses chunk hits to parent documents and reports per-kind breakdowns
(identifier / natural_language / exploratory); `ablation()` reports
sparse-only, dense-only and hybrid so the BM25 baseline is always visible
(Ma et al. 2021). `synthetic_engineering_corpus(n_docs, seed)` produces a
deterministic judged corpus (specifications, incidents, runbooks, datasheets).

```bash
python -m hybrid_retrieval --offline --evaluate "any task"
```

```python
from hybrid_retrieval.evaluation import evaluate, ablation, synthetic_engineering_corpus, QueryJudgment
docs, judgments = synthetic_engineering_corpus(24)
model.index(docs)
report = await evaluate(model, judgments, k=10); print(report.summary()); print(report.worst(3))
reports = await ablation(model, judgments, k=10)   # {"sparse": ..., "dense": ..., "hybrid": ...}
```

Offline, with the hashing embedder, the synthetic corpus is easy by design
(nDCG@10 1.0 hybrid vs about 0.9 dense-only); it is a self-test, not a
benchmark. For a real deployment build BEIR-style judgments from engineer
queries (>= 100 for stable alpha fitting; Bruch et al. 2023) and gate CI on a
1.0-point nDCG@10 regression tolerance (Lin et al. 2021). Per-run
diagnostics (`RetrievalOutput`): per-channel candidate counts and top scores,
`overlap_at_k`, `coverage`, transforms applied, rerank method, and the
confidence basis.

## 10. Limitations and failure modes

- **Dense-only retrieval out of domain** under-performs BM25 (DPR -47.7 % on
  BEIR; Thakur et al. 2021). Neither planner disables the sparse channel
  (`from_planner_output` forces `use_sparse=True`); only an explicit
  `inputs["plan"]` can run dense-only.
- **Under-tuned BM25 or untuned fusion weights hide hybrid gains** (Ma et al.
  2021). k1/b and `sparse_weight` are configuration; `ablation()` reports the
  baseline. No automatic grid search is shipped.
- **Tuned RRF constants do not transfer** (Bruch et al. 2023); `rrf_k` stays 60
  unless re-tuned per collection.
- **Position bias downstream** (Liu et al. 2023): this model returns a ranked
  list capped at `k_final`; the RAG model's context orderer is responsible for
  first/last placement.
- **LLM expansion and listwise reranking are slow and costly** (Query2doc > 2 s
  per query; RankGPT 11-32 s at depth 100; Wang et al. 2023; Sun et al. 2023).
  They are router-gated, depth-limited (20-30) and budget-degraded; cache
  them with the Caching model.
- **Generated expansions can drift** toward plausible-but-wrong content; the
  overlap guard drops them and warns, but cannot catch semantically wrong text
  that shares vocabulary.
- **The offline `HashingEmbedder` is a lexical-overlap baseline**, not a
  semantic model; dense quality in demos does not reflect production embedders,
  and E5/BGE-style prefixes must be configured on the embedder (Wang et al.
  2022; Xiao et al. 2024).
- **Confidence is heuristic**, derived from coverage, channel agreement, fill
  and reranking; it is not a calibrated probability. Safety-critical profiles
  therefore route most results to `NEEDS_REVIEW` by design.
- **Post-filtering collapses recall**; filters are pre-applied as allow-lists,
  which is exact for the brute-force index but an `ANNBackend` that ignores
  `allowed_ids` is bypassed when filters are present (Patel et al. 2024).
- **Chunking defaults are per collection**; 400/0 is the literature default,
  not a universal optimum (Smith & Troynikov 2024).
- **A bare `MockLLM` cannot rerank**: it returns empty permutations, so offline
  runs keep the fused order, warn, and report `reranked=False` /
  `rerank_method="none"` with no confidence bonus (`reranker.applied`). The same
  holds online when every window or grade fails.
- **The `Embedder` protocol is the spec's `embed(texts)` + `dimension`**;
  `embed_queries` is an optional hook for asymmetric models. A spec-minimal
  embedder therefore works, but E5/BGE-style query prefixes are only applied
  when the embedder implements the hook (the built-in ones do).

## 11. How it was built

1. **Research** (`docs/research/hybrid-retrieval.draft.json`,
   `prompting-techniques.draft.json`): the fusion literature settled the
   defaults (RRF k = 60 zero-config, convex with theoretical min-max and
   semantic alpha 0.7 once a domain weight exists), BEIR fixed the "always
   keep BM25" rule, RankGPT/RankZephyr fixed the listwise window/stride and
   depth, HyDE/Query2doc fixed N, repetition and the router gate, and the
   chunking studies moved the spec's 512/64 to 400/0.
2. **Design decisions**: a declarative `RetrievalPlan` separates reasoning
   from execution so humans and orchestrators can inspect or override it; the
   heuristic planner mirrors the rules the LLM planner is prompted with, so
   offline and online behaviour agree; filters are allow-lists inside the
   channels; every LLM output is a strict JSON schema validated and clamped
   in Python; costs are estimated before execution and degraded to fit the
   budget; direct-API calls are free of LLM calls unless asked.
3. **Implementation**: pure-function cores (fusion, normalisation, window
   permutation repair, confidence, metrics, budget degradation) are unit-tested
   without an LLM; transforms and rerankers are driven with `ScriptedLLM`, the
   LLM-planned component path with the non-mock `OnlineScriptedLLM` stub, and
   `MockLLM` covers the offline end-to-end path; the CLI and two examples run
   with no credentials. `tests/` holds 129 tests that complete in a few seconds.
4. **Adversarial review**: a second pass found and fixed budget exhaustion
   escaping `execute()`, rerank bonuses granted for reranks that did nothing,
   store/channel divergence for id-less documents, a stricter-than-spec
   `Embedder` protocol, weighted RRF, and offline planning that bypassed the
   heuristic planner; each fix carries a regression test.
