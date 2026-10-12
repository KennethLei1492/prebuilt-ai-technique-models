# Caching model (`caching`)

Avoid paying twice. `CachingModel` is the caching component of the pipeline
library: exact-match response caching with versioned, chained keys; a
*verified* semantic cache with identifier guards and per-entry adaptive
thresholds; single-flight coalescing and XFetch early refresh against
stampedes; negative caching of deterministic failures; a provider
prefix-cache advisor; and dependency-based invalidation for grounded (RAG)
answers. It is a **policy engine**, not a dictionary: every request is first
classified into a `CachePlan` that decides which tiers may serve it and for
how long.

It plugs into the other five models in two ways: `CachingModel.wrap(llm)`
returns a `CachedLLMClient` that any component can use as its `llm`, and
`CachingModel.cache` is the `pipeline_core.ResponseCache` protocol object
components accept as a dependency.

All citations below are to the verified research dossier
`docs/research/caching.json`; the pipeline-level design sources (deterministic
workflows first, cost cascades, OTel usage counters) come from
`docs/research/pipeline-architecture.json`. Sources the dossier marks
`unverified` (vendor documentation, RFC 2308, Belady, ARC, XFetch) are cited
for the specific facts they document; every quantitative claim about semantic
caching rests on verified papers.

---

## Purpose

| Need | What the model does |
| --- | --- |
| Repeated prompts cost full price | Exact tier returns the stored `LLMResponse` with `cached=True`, `Usage(cache_hits=1)` and zero cost. |
| Paraphrased repeats of stable lookups | Semantic tier serves hits above a cosine threshold, but only when identifiers/numbers/units are identical and the conversation context matches; borderline hits are verified by one cheap LLM call. |
| Parallel workers ask the same thing at once | Single-flight coalescing: one call, every waiter gets the result. |
| Hot entries expire under load | XFetch probabilistic early refresh using the stored recompute latency. |
| A prompt fails deterministically (schema, refusal) | Negative cache with a short, bounded TTL. |
| Provider prompt caching silently never hits | `PrefixCacheAdvisor` lints stable blocks for timestamps, dates, UUIDs, request ids and nonces (JSON with unsorted keys is reported as information only), checks the 512-4096-token minimum, picks 5m / 1h / keep-alive from observed reuse gaps, computes break-even and savings, and serialises the first call before a fan-out. The wrapped client requests the 1-hour provider TTL when the plan recommends it and prewarms the prefix once when a request is marked `fan_out`. |
| A source document changed | `DependencyFingerprint` of the evidence is folded into the key and each entry is tagged `doc:<id>`; ingestion purges by tag; a re-index bumps the namespace version, which purges the namespace from both tiers and makes wrapped clients reject entries written under an older version. |
| Should this prompt be cached at all? | `run(request)` plans and explains caching for a prompt and returns a `CacheReport`. |

---

## Architecture

```
                                 TaskRequest(task, inputs, context, domain)
                                                  |
                                                  v
 +----------------------------------------- CachingModel ------------------------------------------+
 |  plan()                                                                                          |
 |   +-- HeuristicPlanner.plan()  (plan.py)  ---- always: the safety floor                          |
 |   +-- CACHE_PLAN_TEMPLATE -> ctx.llm_call(output_schema=CachePlan)  ---- online only, 1 call     |
 |   +-- HeuristicPlanner.reconcile(llm_plan, heuristic)  ---- LLM may tighten, never relax         |
 |                                                  | CachePlan                                     |
 |  execute()                                       v                                               |
 |   analyze -> keys (exact_key, namespace, tags, DependencyFingerprint) -> prefix_audit            |
 |   (PrefixCacheAdvisor) -> estimate (SavingsEstimate) -> store (optional) -> CacheReport          |
 +--------------------------------------------------------------------------------------------------+
            |  wrap(llm)                                    |  .cache / get / set / invalidate
            v                                               v
 +------------------- CachedLLMClient (middleware.py) --------------------+   +--- CacheStore ---+
 |  LLMRequest  (profile = metadata['domain'] or the wrap-time profile)   |   | ResponseCache    |
 |   -> HeuristicPlanner.plan_for_llm_request  (cacheable? tiers? ttl?)   |   | protocol facade  |
 |   -> provider prefix policy (cache_ttl='1h' when planned; prewarm once |   +------------------+
 |                              per prefix when metadata['fan_out'])      |
 |   -> exact tier     backends.py  (InMemory LRU/LFU/FIFO, SQLite, File, |
 |                                   Redis; TTL, tags, bytes + entries;   |
 |                                   version-stamped)                     |
 |        | miss / stale / old version / XFetch refresh (stampede.py)     |   +--- invalidation.py ---+
 |   -> semantic tier  semantic.py  (Embedder, guards, context chain,     |   | tags, versions,       |
 |                                   per-entry thresholds, error budget)  |   | DependencyFingerprint |
 |        | miss                                                          |   +-----------------------+
 |   -> SingleFlight (stampede.py)  -> inner.complete()  -> store both    |
 |        | StructuredOutput/Refusal/Truncated error -> negative entry    |   +--- prefix.py ---------+
 |   <- LLMResponse(cached=True, usage.cache_hits=1, .cache={tier,...})   |   | lint, min tokens,     |
 +------------------------------------------------------------------------+   | TTL, break-even,      |
                                                                              | prewarm, verify_hit   |
   keys.py: SHA-256 chain  H(namespace) -> tools -> system blocks -> volatile  +-----------------------+
            -> messages -> extra determinants (model, template version,
            output schema, effort, max_tokens, tenant salt, corpus, fingerprint)
   evaluation.py: hit rate vs Belady OPT, false-hit audit, cost/latency saved
```

| Module | Contents |
| --- | --- |
| `backends.py` | `CacheBackend` ABC; `InMemoryBackend` (LRU / LFU / FIFO, bounded by entries **and bytes**), `SQLiteBackend`, `FileBackend`, `RedisBackend` (lazy import). `CacheEntry` carries `created_at`, `ttl_s`, `tags`, `hits`, `version`, `size_bytes`, `recompute_cost_s`, `negative`. Injectable clock. |
| `keys.py` | `exact_key` (SHA-256 hash chain), `namespace_for`, `tagged_keys`, `normalize_prompt_text` (NFC, whitespace, labelled tracing-id stripping, stable JSON; bare timestamps / clock times / UUIDs / hex ids are kept by default; code blocks untouched; lower-casing and full volatile stripping opt-in), `extract_guards`, `context_chain_key`. |
| `semantic.py` | `Embedder` protocol (same shape as `hybrid_retrieval.dense.Embedder`, defined locally), `HashingEmbedder` (offline baseline), `CallableEmbedder`, `SemanticCache` (namespaces, guards, context chain, top-k, per-entry thresholds, feedback, value-aware eviction, metrics). |
| `stampede.py` | `xfetch_should_refresh`, `SingleFlight`. |
| `middleware.py` | `CachedLLMClient(BaseLLMClient)` (per-request domain profile, 1h prefix TTL promotion, fan-out prewarm, version checks, corrupt-entry recovery, verification usage accounting), `CacheStore` (the `ResponseCache`), `CacheMetrics`. |
| `prefix.py` | `PrefixCacheAdvisor`, `PrefixReport`, `find_invalidators`, `find_unsorted_json` (informational), `recommend_ttl_from_gaps`, `breakeven_uses`, `prefix_hit_ratio`, `verify_hit`, `warm_then_fan_out`. |
| `invalidation.py` | `DependencyFingerprint`, `VersionRegistry`, `InvalidationManager` (tag/document/template/model purge, version bump, cause log). |
| `plan.py` | `CachePlan`, `CacheabilityFeatures`, `analyze_task`, `HeuristicPlanner` (`plan`, `plan_text`, `plan_for_llm_request`, `reconcile`). |
| `prompts.py` | `CACHE_PLAN_TEMPLATE` (v1.0.0), `EQUIVALENCE_TEMPLATE` (v1.0.0), `EquivalenceVerdict`. |
| `model.py` | `CachingModel`, `CacheReport`, `SavingsEstimate`. |
| `config.py` | `CachingConfig`, `KeyNormalization`. |
| `evaluation.py` | Belady OPT simulation, policy comparison, false-hit audit, cost/latency savings, synthetic engineering workload, `run_offline_evaluation`. |
| `cli.py` | `python -m caching "task" [--domain] [--offline] [--json] [--input k=v] [--model] [--expected-reuses] [--trace]`. |

---

## How it reasons about a prompt

### The plan schema (`CachePlan`)

| Field | Meaning |
| --- | --- |
| `cacheable` | Whether any response-cache tier may serve the prompt. |
| `tiers` | Ordered subset of `exact`, `semantic`, `prefix`, `negative`. `exact` is always first when cacheable. |
| `ttl_s` | TTL in seconds (`null` = no expiry; `0` = do not retain: nothing is stored, never an error). |
| `semantic_threshold`, `error_budget` | Initial cosine threshold for new semantic entries and the maximum tolerated incorrect-hit rate that drives per-entry threshold learning. |
| `verify_hits` | Run the equivalence check on borderline semantic hits. |
| `identifier_guards` | Require identical identifiers / numbers / units for semantic hits. |
| `namespace` | `{model}:{template version}:{tools hash}:{schema hash}:{corpus version}:{embedding}:{tenant}:{domain}`. |
| `prefix_ttl`, `prewarm` | Provider prefix TTL to request (`5m`, `1h`, `none`) and whether to warm the prefix before a fan-out. |
| `negative_ttl_s` | TTL for cached deterministic failures. |
| `task_class` | `lookup`, `analysis`, `generative`, `time_sensitive`, `personal`, `random`. |
| `risk_flags`, `reason`, `confidence` | Why, in words; planner confidence (0-1). |

### Heuristic rules (offline, `Constraints.deterministic`, and always as the floor)

`analyze_task` extracts features: time-sensitive words, personal-data markers,
randomness/diversity requests, lookup vs analysis wording, identifier and
number density (`extract_guards`), and conflicting signals. Then, in order:

1. **Diversity requested** (brainstorm, random, "N different", poem, joke, "write a story") -> `cacheable=false`, `tiers=[]`, `task_class=random`. Serving one sample forever collapses variety (Fu Bang (2023), *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings*, https://aclanthology.org/2023.nlposs-1.24/, bypass-knob rationale; dossier pitfall).
2. **Time-sensitive** (today, now, current, latest, live, "status of") -> exact tier only, `ttl_s = time_sensitive_ttl_s` (300 s).
3. **Personal data** (my, "my account", "my email", user id, "for me") -> exact tier only, `ttl_s = personal_ttl_s`, tenant-scoped namespace (Gill et al. (2024), *MeanCache: User-Centric Semantic Caching for LLM Web Services*, https://arxiv.org/abs/2403.02694; dossier recommendation 11).

   The vocabularies deliberately avoid single words that are ordinary engineering terms: "roll" (roll axis), "unique" (unique constraint), "alternatives" (design alternatives), "address" (MAC address), "status" (HTTP status code), "personal" (personal protective equipment) and "me" ("tell me") only count inside the phrases listed in `plan.py`.
4. **Analysis vs lookup** -> both cacheable with the 24 h default TTL; long prompts without lookup wording are `generative` (confidence -0.1).
5. **Identifier / numeric density** (>= 2 identifiers or >= 0.08 per word) -> strict threshold 0.98, error budget 1 %, `verify_hits=true` (Zhu et al. (2024); dossier engineering notes: "5 mm vs 50 mm defeats embedding similarity").
6. **Risk level** from `ctx.profile`: `high` -> strict threshold/budget and mandatory verification; `safety_critical` -> semantic tier removed unless `semantic_in_safety_critical=True`.
7. **Prefix TTL** -> `1h` when the profile has standards and document types (long, stable reference material) and the task is lookup/analysis; otherwise `prefix_default_ttl`.
8. **Confidence** = 0.9 minus 0.15 per conflicting signal (e.g. lookup-like *and* time-sensitive), clamped to [0.3, 0.95].

### LLM-driven path (online, one structured call)

When the client is a real model (or `planner_mode="always"` for scripted tests)
and the request is not `deterministic`, `plan()` renders `CACHE_PLAN_TEMPLATE`:
the role, tier definitions, eight numbered rules and two few-shot examples
live in `system_stable` together with `ctx.profile.system_text()` (and
`ENGINEERING_RIGOR` for high/safety-critical profiles); the task, extracted
features, model id and fan-out flag go in the user turn. The call uses
`effort="low"` (`planner_effort`) and `output_schema = sanitize_schema(CachePlan.model_json_schema())`;
the answer is `CachePlan.model_validate`d. The result is then **reconciled**
against the heuristic plan: the LLM may be *more* conservative (drop a tier,
shorten a TTL, raise a threshold) but never less. Any override is recorded in
`CacheReport.plan_overrides`, warned about, and lowers confidence by 0.05
each. Any `LLMError` from the planning call (invalid JSON, refusal,
truncation, connection or API error) and any schema-invalid plan fall back to
the heuristic plan with a warning, so `run()` still succeeds; only
`BudgetExceededError` propagates (the core reports it as a budget failure).
The budget is honoured: with `max_llm_calls=0` or an exhausted deadline the
heuristic plan is used directly.

`effort="low"` for the planning call is a documented deviation from
ARCHITECTURE.md 3.2 ("`high` for planning"): this "plan" is a cacheability
classification over a fixed rule set that the heuristic floor re-checks, which
is the routing/classification work 3.2 assigns to `low` (dossier
recommendation 2: "cheap deterministic rule engine with an optional
small-model fallback"). Set `planner_effort="high"` to follow the spec
literally.

### Status and confidence

`execute()` sets `confidence` from the planner's confidence, the plan source,
overrides, and whether the stable prefix contains volatile content (-0.1). It
compares against `max(ctx.profile.min_confidence_to_accept, request.constraints.min_confidence, config.min_confidence)`
and returns `NEEDS_REVIEW` below that. The same prompt can therefore be
`SUCCESS` for `general` (floor 0.5) and `NEEDS_REVIEW` for
`civil_structural_engineering` (floor 0.85) when its signals conflict.

---

## Quick start

### Python

```python
import asyncio
from pipeline_core import LLMRequest, TaskRequest, make_llm
from caching import CachingModel

async def main():
    llm = make_llm()                      # MockLLM without credentials
    model = CachingModel(llm)             # or CachingModel(llm, semantic_threshold=0.97, backend="sqlite", sqlite_path="cache.db")

    # Plan / explain caching for a prompt
    result = await model.run(TaskRequest(task="What does clause 4.2 of ISO 2768 require for a 120 mm dimension?",
                                         domain="mechanical_engineering", inputs={"expected_reuses": 10}))
    print(result.summary()); print(result.output_text)
    report = result.output                # CacheReport: plan, exact_key, namespace, tags, prefix, savings

    # Middleware for every other component
    cached = model.wrap(llm)
    req = LLMRequest.simple("Define factor of safety.", system=["You are a careful mechanical engineer."])
    a = await cached.complete(req)        # miss -> inner call, stored
    b = await cached.complete(req)        # exact hit: b.cached, b.usage.cache_hits == 1, cost 0
    print(cached.savings_report())

    # ResponseCache protocol
    await model.cache.set("k", {"answer": 1}, ttl_s=600, tags=["doc:spec-1"])
    await model.cache.invalidate(tags=["doc:spec-1"])

asyncio.run(main())
```

Synchronous helpers: `model.explain(task, domain=...)`, `model.plan_for(task, domain=...)`,
`model.run_sync(request)`, `model.key_for(llm_request, profile=None, client=None)`
(the exact key `model.wrap(client or model.llm, profile=profile)` uses for that
request; keys use the client's real model id, so an offline `MockLLM` never
shares entries with a real model), `model.bump_version(namespace)`,
`model.invalidate_documents(ids)`.

### CLI

```bash
cd "<repo root>"
# Windows (Git Bash) uses ';' as the PYTHONPATH separator; POSIX uses ':'
PYTHONPATH="common;caching_model" python -m caching --offline --domain mechanical_engineering \
    --input expected_reuses=12 "What does clause 4.2 of ISO 2768 require for a 120 mm dimension?"
PYTHONPATH="common;caching_model" python -m caching --offline --json "Brainstorm five slogans"
PYTHONPATH="common;caching_model" python -m caching.evaluation          # offline evaluation harness
```

Flags: `--domain NAME`, `--offline` (force the deterministic `MockLLM`),
`--json` (print the full `TaskResult`), `--input key=value` (repeatable),
`--model ID` (model to plan, key and price for; same as `--input model=ID`),
`--expected-reuses N` (reuse assumption for the savings estimate) and
`--trace` (print the span tree after the result).

`--input key=value` is repeatable and JSON-parsed (`--input 'system_blocks=["..."]'`).
Useful inputs: `model`, `expected_reuses`, `expected_output_tokens`,
`system_prompt`, `system_blocks`, `prompt_version`, `prompt_template`,
`corpus_version`, `tenant`, `fan_out`, `llm_request`, `response`.

---

## Configuration (`CachingConfig`)

Where the dossier's literature-backed default differs from the architecture
spec's suggestion, the literature wins and the row says why.

| Field | Default | Why / source |
| --- | --- | --- |
| `backend` | `"memory"` | In-process L1 first (dossier recommendation 1: "L1 exact-match memoization (in-process LRU + Redis)"). |
| `eviction_policy` | `"lru"` | LRU is the starting point; move to ARC/TinyLFU only when trace replay shows a material gap to OPT - Belady (1966), *A Study of Replacement Algorithms for a Virtual-Storage Computer*, https://en.wikipedia.org/wiki/L%C3%A1szl%C3%B3_B%C3%A9l%C3%A1dy; Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention)*, https://arxiv.org/abs/2312.07104 (RadixAttention uses LRU). |
| `max_entries` | `10_000` | Capacity bound; evaluated against OPT in `evaluation.py`. |
| `max_bytes` | `256 MiB` | Size-aware eviction; entry-count eviction is the documented OOM pitfall - Fu Bang (2023), *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings*, https://aclanthology.org/2023.nlposs-1.24/. |
| `sqlite_path`, `file_dir`, `redis_url`, `key_prefix` | `":memory:"`, `".cache/caching"`, `redis://localhost:6379/0`, `"caching"` | Backend locations. |
| `tenant_salt` | `""` | Per-tenant salt in the key chain; prevents cross-tenant hits - Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)*, https://arxiv.org/abs/2309.06180 (`cache_salt`, SHA-256 default). The raw salt only enters the hash chain; namespaces, tags, reports and traces show `salt-<sha256[:12]>`. |
| `normalization.unicode_nfc` | `True` | Dossier technique 1 step (1): NFC, trim, collapse whitespace. |
| `normalization.collapse_whitespace` | `True` | Same. |
| `normalization.lowercase` | `False` | Never lowercase code, identifiers or part numbers (dossier engineering notes: code, BOM). |
| `normalization.strip_labelled_ids` | `True` | Remove *labelled* tracing ids (`request_id: ...`, `trace_id=...`, `correlation_id: ...`): they never change the answer, so true repeats still hit (dossier technique 1). |
| `normalization.strip_volatile` | `False` | Opt-in removal of bare timestamps, clock times, UUIDs and 32-hex ids. Off because a prompt differing only in such a value asks a different question (an incident window, a build id, a commit hash) and the exact tier promises zero false hits; the same patterns are still used to lint the *stable prefix*, where they only break provider caching (dossier incident-log note). |
| `normalization.stable_json` | `True` | Canonical JSON with sorted keys (dossier technique 1 step (2)). |
| `default_ttl_s` | `86_400` (24 h) | Positive TTL for grounded answers over versioned corpora; TTL is a backstop, dependency tags are the primary invalidation (dossier recommendation 8). |
| `time_sensitive_ttl_s` | `300` | Plan rule 2 caps live-data answers at 5 minutes (planner prompt rule 2). |
| `personal_ttl_s` | `3_600` | Exact tier only, tenant-scoped (Gill et al. (2024), *MeanCache: User-Centric Semantic Caching for LLM Web Services*, https://arxiv.org/abs/2403.02694). |
| `negative_ttl_s` | `600` | Negative TTL well below the positive TTL; RFC 2308 reports problems above one day - Andrews (1998), *RFC 2308: Negative Caching of DNS Queries (DNS NCACHE)*, https://www.rfc-editor.org/rfc/rfc2308. |
| `negative_caching` | `True` | Cache deterministic failures briefly (RFC 2308; dossier composition note on verifier rejections). |
| `max_staleness_s` | `None` | Reject hits older than this even inside the TTL (dossier recommendation 8: "every hit returns age ... to enforce max_staleness"). |
| `semantic_enabled` | `True` | Tier 2 of the stack; disabled per prompt by the planner where unsafe. |
| `semantic_threshold` | `0.95` | Conservative initial cosine for general embeddings (dossier technique 2 default; GPTCache example distance 0.2) - Fu Bang (2023); Zhu et al. (2024), *Efficient Prompt Caching via Embedding Similarity*, https://arxiv.org/abs/2402.01173. |
| `strict_semantic_threshold` | `0.98` | Engineering-strict profile for identifier/number-bearing prompts and high-risk domains (dossier recommendation 9). |
| `semantic_top_k` | `5` | Candidates per lookup (dossier technique 2 default). |
| `semantic_error_budget` | `0.02` | Error-budget contract instead of a bare threshold; 1-2 % for engineering - Schroeder et al. (2025), *vCache: Verified Semantic Prompt Caching*, https://arxiv.org/abs/2502.03771. |
| `strict_error_budget` | `0.01` | Engineering-strict budget (dossier recommendation 9: "semantic delta <= 1%"). |
| `semantic_in_safety_critical` | `False` | Mismatch cost is very high for safety-relevant answers - Liu et al. (2025), *Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation*, https://arxiv.org/abs/2508.07675. |
| `verify_borderline_hits` | `True` | Sampled/LLM verification of hits drives the ~3 % incorrect-hit residue down - Regmi and Pun (2024), *GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching*, https://arxiv.org/abs/2411.05276; vCache feedback loop. Each check is one `effort="low"` call whose usage is added to the served response (budgets and traces see it) and to `CacheMetrics.verification_calls` / `verification_cost_usd`; its cost is subtracted from `cost_saved_usd`. |
| `borderline_band` | `0.02` | Width of the verify band above the threshold. |
| `identifier_guards` | `True` | Exact-match guards on identifiers, numbers, units (Zhu et al. (2024); dossier engineering notes). |
| `context_turns` | `3` | Context chain of the previous 2-3 turns (Gill et al. (2024), *MeanCache: User-Centric Semantic Caching for LLM Web Services*, https://arxiv.org/abs/2403.02694). |
| `semantic_max_entries` | `5_000` | Value-aware eviction (fewest *served* hits, oldest first) - Li et al. (2024), *SCALM: Towards Semantic Caching for Automated Chat Services with Large Language Models*, https://arxiv.org/abs/2406.00025. |
| `embedding_dimension` | `256` | Dimension of the offline `HashingEmbedder` that `CachingModel` and `CachedLLMClient` build when no embedder is injected; production injects a real embedder (Ghaffari et al. (2025), *An Ensemble Embedding Approach for Improving Semantic Caching Performance in LLM-based Systems*, https://arxiv.org/abs/2507.07061). Entries written by a different embedder or dimension never match (they are skipped, never compared). |
| `coalesce_inflight` | `True` | Single-flight coalescing of identical in-flight misses (dossier recommendation 7). |
| `xfetch_beta` | `1.0` | Paper default - Vattani et al. (2015), *Optimal Probabilistic Cache Stampede Prevention*, https://iris.uniroma1.it/handle/11573/877847. |
| `bypass_rate` | `0.0` | GPTCache-style diversity bypass knob; off because the planner handles diversity explicitly (Fu Bang (2023)). |
| `prefix_min_tokens` | `None` | Override the model's documented minimum (512-4096) - Anthropic (2026), *Prompt caching (Claude Developer Platform documentation)*, https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching. |
| `prefix_default_ttl` | `"5m"` | 1.25x write pays off from the second use (Anthropic (2026)). |
| `promote_to_1h_gap_s` | `300` | Recommend 1 h when the median reuse gap exceeds 5 minutes (dossier technique 3 default). Gaps up to 60 minutes are bridged by the 1-hour TTL because every read refreshes it; above an hour, `keep-alive` is recommended while a read every ~55 minutes still costs less than an uncached call, otherwise `none`. |
| `promote_to_1h_min_reuses_per_hour` | `3` | Minimum number of observed uses of a prefix before arrival data promotes it to 1 h: the 2x write pays off from the third use (Anthropic (2026) break-even arithmetic). The field name is kept for compatibility. |
| `apply_prefix_ttl` | `True` | The wrapped client sends `cache_ttl="1h"` when the plan's `prefix_ttl` is `1h` (standards-heavy profiles, lookup/analysis tasks); it never demotes a request that already asks for `1h`. |
| `write_multiplier_5m`, `write_multiplier_1h`, `read_multiplier` | `1.25`, `2.0`, `0.1` | Documented multipliers (Anthropic (2026), *Prompt caching (Claude Developer Platform documentation)*, https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching; OpenAI (2026), *Prompt caching (OpenAI API documentation)*, https://developers.openai.com/api/docs/guides/prompt-caching). |
| `planner_effort` | `"low"` | Classification/routing work uses cheap effort (dossier recommendation 2: "cheap deterministic rule engine with an optional small-model fallback"); a documented deviation from ARCHITECTURE.md 3.2's `high` for planning - see "LLM-driven path". |
| `planner_model` | `None` | Optional cheaper model id for the planning and equivalence calls. |
| `default_expected_reuses` | `5` | Reuse assumption for `SavingsEstimate` when the caller gives none. |
| `min_confidence` | `None` | Extra confidence floor on top of the domain profile. |

Spec vs literature: the spec lists "LRU/LFU/TTL" eviction; the dossier adds
byte-aware capacity (GPTCache OOM pitfall) so both `max_entries` and
`max_bytes` are enforced. The dossier's "only cache when temperature == 0 or a
seed is set" cannot be applied literally because the Claude client never
sends sampling parameters; its intent is implemented by the planner's
diversity detection plus the deterministic flag: `Constraints.deterministic`
forces the heuristic planner in `run()`, and `LLMRequest.metadata["deterministic"]`
makes the wrapped client prefer the cache even for diversity tasks. The
middleware only sees the `LLMRequest`, and `pipeline_core.RunContext.llm_call`
does not copy `request.constraints.deterministic` (or `request.domain`) onto
`LLMRequest.metadata`, so a calling component must set
`metadata["deterministic"]` / `metadata["domain"]` itself until the core
stamps them (core change requested).

---

## Techniques implemented

| Technique | Where | Establishing sources |
| --- | --- | --- |
| Canonical exact-match memoization with chained, versioned keys (SHA-256 chain over tools -> system -> volatile -> messages -> extra determinants; tenant salt; namespace of every determinant) | `keys.py`, `backends.py` | Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)*, https://arxiv.org/abs/2309.06180; Gim et al. (2023), *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*, https://arxiv.org/abs/2311.04934 |
| Size-aware LRU / LFU / FIFO eviction, Belady OPT as the evaluation bound | `backends.py`, `evaluation.py` | Belady (1966), *A Study of Replacement Algorithms for a Virtual-Storage Computer*, https://en.wikipedia.org/wiki/L%C3%A1szl%C3%B3_B%C3%A9l%C3%A1dy; Megiddo and Modha (2003), *ARC: A Self-Tuning, Low Overhead Replacement Cache*, https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache; Einziger et al. (2015), *TinyLFU: A Highly Efficient Cache Admission Policy*, https://arxiv.org/abs/1512.00727 (decision rule: adopt only if LRU trails OPT materially) |
| Verified semantic caching: guards, context chain, per-entry thresholds under an error budget, equivalence check on borderline hits, value-aware eviction | `semantic.py`, `middleware.py`, `prompts.py` | Fu Bang (2023), *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings*, https://aclanthology.org/2023.nlposs-1.24/; Gill et al. (2024), *MeanCache: User-Centric Semantic Caching for LLM Web Services*, https://arxiv.org/abs/2403.02694; Zhu et al. (2024), *Efficient Prompt Caching via Embedding Similarity*, https://arxiv.org/abs/2402.01173; Schroeder et al. (2025), *vCache: Verified Semantic Prompt Caching*, https://arxiv.org/abs/2502.03771; Regmi and Pun (2024), *GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching*, https://arxiv.org/abs/2411.05276; Li et al. (2024), *SCALM: Towards Semantic Caching for Automated Chat Services with Large Language Models*, https://arxiv.org/abs/2406.00025; Liu et al. (2025), *Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation*, https://arxiv.org/abs/2508.07675; Ghaffari et al. (2025), *An Ensemble Embedding Approach for Improving Semantic Caching Performance in LLM-based Systems*, https://arxiv.org/abs/2507.07061 |
| Single-flight coalescing and XFetch probabilistic early refresh | `stampede.py`, `middleware.py` | Vattani et al. (2015), *Optimal Probabilistic Cache Stampede Prevention*, https://iris.uniroma1.it/handle/11573/877847 |
| Negative caching with bounded TTL | `middleware.py` | Andrews (1998), *RFC 2308: Negative Caching of DNS Queries (DNS NCACHE)*, https://www.rfc-editor.org/rfc/rfc2308 |
| Provider prefix-cache layout optimisation: static-first linting, minimum-token gate, TTL from inter-arrival gaps, break-even, 1h TTL promotion and prewarm in the wrapped client, serialised warm-up before fan-out, hit verification from `cache_read_input_tokens` | `prefix.py`, `middleware.py` | Anthropic (2026), *Prompt caching (Claude Developer Platform documentation)*, https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching; OpenAI (2026), *Prompt caching (OpenAI API documentation)*, https://developers.openai.com/api/docs/guides/prompt-caching; Gim et al. (2023), *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*, https://arxiv.org/abs/2311.04934; Ye et al. (2024), *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition*, https://aclanthology.org/2024.acl-long.623 |
| Dependency-based invalidation: tags, namespace versions, `DependencyFingerprint` over evidence | `invalidation.py`, `middleware.py` | Jin et al. (2024), *RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation*, https://arxiv.org/abs/2404.12457; Yao et al. (2024), *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*, https://arxiv.org/abs/2405.16444 (versioned, corpus-tied caches); Andrews (1998), *RFC 2308: Negative Caching of DNS Queries (DNS NCACHE)*, https://www.rfc-editor.org/rfc/rfc2308 (negative entries tied to corpus versions) |
| Cache policy planner (rule engine + optional structured LLM call, reconciled) | `plan.py`, `model.py` | Dossier architecture recommendation 2; Schluntz and Zhang (2024), *Building Effective Agents*, https://www.anthropic.com/research/building-effective-agents (deterministic workflows first; pipeline-architecture dossier) |
| Cost / latency accounting from documented multipliers; OTel-style usage counters | `evaluation.py`, `middleware.py` | Chen et al. (2023), *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, https://arxiv.org/abs/2305.05176 (completion cache as "LLM approximation"; pipeline-architecture dossier); OpenTelemetry GenAI SIG (2025), *OpenTelemetry Semantic Conventions for Generative AI (spans, agent spans, metrics, token metrics)*, https://github.com/open-telemetry/semantic-conventions-genai |

Not implemented (out of scope for a hosted-API component, documented for
completeness): engine-side KV reuse (SGLang RadixAttention, Hydragen,
CacheBlend, CacheGen). The prefix advisor's byte-stable, static-first layout is
exactly what those engines also need, so the same prompts benefit when
self-hosted.

---

## Adapting to an engineering domain

Behaviour is driven by `DomainProfile`; no code changes are needed.

```python
from pipeline_core import DomainProfile, register_profile, TaskRequest
from caching import CachingModel

register_profile(DomainProfile(
    name="aerospace_structures",
    description="Airframe structural substantiation.",
    units=["kN", "MPa"], standards=["FAR 25", "MMPDS"], document_types=["stress report", "drawing"],
    risk_level="safety_critical", min_confidence_to_accept=0.9,
))
model = CachingModel()
r = model.run_sync(TaskRequest(task="Which FAR 25.571 criteria apply to a 2.5 mm skin?", domain="aerospace_structures"))
print(r.output.plan.tiers)        # ['exact', 'prefix', 'negative']  - semantic tier removed (safety_critical)
print(r.output.plan.prefix_ttl)   # '1h' - standards + document types => long stable reference material
```

What the profile changes:

| Profile signal | Effect |
| --- | --- |
| `risk_level="high"` | strict threshold 0.98, 1 % error budget, mandatory verification of borderline hits; `ENGINEERING_RIGOR` added to the stable prefix. |
| `risk_level="safety_critical"` | semantic tier disabled (`semantic_in_safety_critical` re-enables it with strict settings). |
| `min_confidence_to_accept` | floor for `NEEDS_REVIEW`. |
| `standards` + `document_types` | `prefix_ttl="1h"` for lookup/analysis tasks. |
| `system_text()` | rendered into the stable system block, so the prefix audit and the exact key reflect the domain. |
| `name` | last segment of the namespace: no cross-domain hits. |

Engineering-specific guards are built in: `extract_guards` treats
`REQ-1234`, `ISO 2768`, `ABC-123-X`, `v1.2.3`, `120 mm`, `12 Nm` and bare
numbers as exact-match tokens, so "5 mm" never serves "50 mm" (see
`examples/engineering_example.py`).

---

## Composing with the other five models

The component never imports another model package; it exposes and accepts
protocol objects from `pipeline_core`.

| Model | How they connect |
| --- | --- |
| **All** | `shared = CachingModel().wrap(make_llm())`; pass `shared` as the `llm` of every component. Each request is planned under its own domain when the caller sets `LLMRequest.metadata["domain"]` (so the single shared client still disables the semantic tier for safety-critical requests); without it the wrap-time profile applies. Hits return `usage.cache_hits=1`, zero cost, and are visible in the trace (`cached=True` span attribute); verification and prewarm calls made by the client are added to the response usage so budgets see them. `CachingModel.cache` is the `ResponseCache` dependency (`get`/`set`/`invalidate`); `CachingModel` itself also satisfies the protocol. |
| **hybrid_retrieval** | Pass `request.metadata["corpus_version"]` / `document_ids` on `LLMRequest.metadata` (or `inputs` of `run`) so keys include the index snapshot and entries carry `doc:<id>` tags; call `model.invalidate_documents(ids)` on re-index. The `Embedder` protocol has the same shape as `hybrid_retrieval.dense.Embedder`, so a retrieval embedder can be injected into `SemanticCache`. |
| **task_decomposition** | Sub-task prompts with identical normalised text hit the exact tier; parallel sub-tasks issuing the same prompt are coalesced by `SingleFlight`. Keep the planner prompt byte-stable so the prefix tier covers every sub-task call. |
| **answer_verification** | The verifier is the feedback signal: `SemanticCache.record_feedback(entry_key, similarity, correct)` tightens or relaxes per-entry thresholds (vCache); a verifier rejection can be stored as a negative entry; store only verified answers via `run(inputs={"llm_request": req, "response": ...})` - the response is stored as an `LLMResponse` under the same key `model.wrap(llm)` computes for `req` (`model.key_for(req)`), so wrapped clients serve it. |
| **parallelization** | Mark fan-out requests with `metadata["fan_out"]=True`: the wrapped client then prewarms the shared prefix once through `AnthropicLLM.prewarm` (concurrent requests wait for it, the write is charged to one response) and skips it while the provider TTL is still running. `PrefixCacheAdvisor.warm_then_fan_out(call, requests)` serialises the first request for clients without `prewarm`; `plan.prewarm` is set when `inputs["fan_out"]` is true; `model.prewarm(llm, request)` is the direct call. |
| **pipeline_rag** | `DependencyFingerprint.from_evidence(request.context)` is folded into the key, so a grounded answer misses as soon as any source changes; `PrefixReport.reference_tokens_in_turn` flags reusable documents that should move into the stable prefix. |

`docs/ARCHITECTURE.md` section 5 specifies that the repository-level
`examples/enterprise_pipeline.py` wires `CachingModel.wrap(llm)` as the shared
client for all six models; this package provides exactly that entry point.

---

## Evaluation

Metrics (from the dossier's evaluation list) and where they come from:

| Metric | Source |
| --- | --- |
| Per-tier hit rate, misses, bypasses, coalesced, early refreshes, stale / version rejections, corrupt entries dropped, negative hits, verification calls and cost, prefix TTL promotions, prewarms | `CachedLLMClient.metrics` / `CachingModel.metrics()` |
| Tokens, dollars and latency saved | `CacheMetrics.tokens_saved`, `cost_saved_usd`, `latency_saved_s`; `evaluation.cost_saved`, `prefix_savings`, `latency_saved` |
| Provider prefix hit ratio `cache_read / (cache_read + cache_creation + input)` | `prefix.prefix_hit_ratio`, `PrefixCacheAdvisor.verify_hit` |
| Semantic false-hit rate, precision / recall / F1, AUC of similarity as a validity predictor | `evaluation.false_hit_audit` (target AUC >= 0.8 before enabling in production, Zhu et al. (2024)) |
| Observed false-hit rate, borderline share, guard rejections | `SemanticCache.false_hit_risk()` |
| Eviction quality vs Belady OPT | `evaluation.compare_policies`, `belady_opt_hits` |
| Invalidations by cause | `InvalidationManager.causes()` |

Run the offline harness (synthetic skewed trace + labelled paraphrase /
identifier-swap pairs):

```bash
PYTHONPATH="common;caching_model" python -m caching.evaluation --trace-length 400 --capacity 24
PYTHONPATH="common;caching_model" python -m caching.evaluation --json
```

It reports LRU/LFU/FIFO hit rates and their gap to OPT, the semantic audit
with guards on (false-hit rate 0.0 on identifier swaps) and off, and the cost
saved per 1000 exact hits for the priced model.

Tests: `python -m pytest caching_model/tests -q` (118 tests, offline, < 3 s; `test_review_regressions.py` holds one or more regression tests per adversarial-review finding).

---

## Limitations and failure modes

From the dossier's pitfalls, and how this package handles (or does not handle) each:

- **Similarity is not equivalence.** General embeddings predict "same answer" at chance level on hard pairs (Zhu et al. (2024)). The `HashingEmbedder` is an offline baseline only; measure AUC on labelled pairs from your domain (`false_hit_audit`) and inject a domain-tuned embedder before enabling the semantic tier in production.
- **Per-entry thresholds need feedback.** vCache-style learning only converges when hit outcomes are reported (`record_feedback`); without a verifier the static initial threshold and the guards are the only protection.
- **Context-blind hits.** Follow-up questions are separated by the context chain over the last `context_turns` turns; conversations longer than that window share a key.
- **Forgetting a determinant.** Model, template version, output schema, effort, `max_tokens`, tools, tenant salt, corpus version and dependency fingerprint are in the key; anything else that changes an answer (e.g. a tool's *behaviour* without a schema change) must be passed via `metadata["cache_namespace"]`/`corpus_version` or a version bump.
- **Volatile prefixes never hit.** The advisor lints for timestamps, dates, UUIDs, request ids and nonces, and reports JSON with unsorted keys as information (it cannot tell a constant literal from one re-rendered from a dict); it cannot see provider-side invalidators such as tool-definition edits or thinking/effort changes; monitor `prefix_hit_ratio` per template version.
- **Exact tier keeps bare timestamps and ids.** Two prompts differing only in a timestamp, clock time, UUID or hex id never share an exact entry (they are different questions); render such values in the user turn, not the stable prefix. Enable `normalization.strip_volatile` only for traffic where those values are known to be noise.
- **Per-request flags rely on the caller.** The wrapped client reads `metadata["domain"]` and `metadata["deterministic"]`; `RunContext.llm_call` does not stamp them from the `TaskRequest` yet, so components must forward them (core change requested).
- **Version registry is process-local.** `bump_version` purges the namespace from the shared backend (also persistent ones) and wrapped clients in the same process reject older versions; another process keeps its own registry and only sees the purge.
- **Cold-prefix concurrency.** Parallel cold requests each pay the write price; use `warm_then_fan_out`.
- **Stale grounded answers.** TTL alone is not freshness; tag entries with document ids and call `invalidate_documents` from ingestion. The exact key already includes the content fingerprint when evidence is supplied.
- **Negative caching hides new data** if the TTL is long; the default is 10 minutes and entries carry document tags.
- **Scan pollution.** Bulk one-off queries flush hot entries under LRU; ARC/W-TinyLFU are not implemented - use `eviction_policy="lfu"`, a separate namespace for batch traffic, or raise capacity when the OPT gap is large.
- **Diversity collapse.** Diversity-requesting prompts are bypassed by the planner; `Constraints.deterministic` deliberately overrides this for reproducible runs.
- **Prefix caching speeds prefill only.** Savings estimates cover input cost; decode-heavy workloads gain little.
- **Local stores only.** `SingleFlight` coalesces within one process; a distributed lease is not provided. `RedisBackend` requires the optional `redis` package.
- **Core accounting quirk.** `pipeline_core.PipelineComponent.run` keeps a component's own `result.usage` when it has zero LLM calls and zero cost, which drops `cache_hits` recorded on `ctx.usage`; components should pass `usage=ctx.usage` to `new_result` (or the core `_max_usage` should merge `cache_hits`; core change requested). `CachedLLMClient.total_usage` and `CachingModel.metrics()` always have the correct counts.

---

## How it was built

1. **Research.** The caching dossier describes a four-layer stack (exact, semantic, provider prefix, engine KV) and insists the component be a *policy engine*: decide per request which layers apply, key on every determinant, bound the semantic false-hit rate explicitly, coalesce misses, and expose hit rate / false-hit rate / savings as first-class metrics. The pipeline-architecture dossier adds: deterministic workflows first, every decision a configuration field with a literature default, structured artifacts between stages, and OTel-style usage counters.
2. **Design decisions.**
   - Keys follow vLLM's block-hash chain with SHA-256 and a tenant salt; the namespace lists every determinant so upgrades miss naturally (dossier recommendation 3).
   - The semantic tier is *verified*: conservative 0.95 initial threshold, 0.98 for identifier-bearing or high-risk prompts, exact-match guards, context chain, per-entry thresholds under an error budget, and an LLM equivalence check only in the borderline band (vCache, MeanCache, Zhu et al.).
   - Safety-critical profiles lose the semantic tier by default because mismatch cost is domain-specific and high (Liu et al. (2025)); the switch is configuration.
   - Eviction starts with LRU and is bounded in bytes; `evaluation.py` measures the gap to Belady OPT so the decision to adopt ARC/TinyLFU is data-driven.
   - Stampedes are handled with XFetch (beta 1) plus single-flight, negative results are cached per RFC 2308 with a TTL far below the positive TTL.
   - Provider prefix caching is treated as a layout problem: static-first, lint for volatile content, gate on the model minimum, choose 5m/1h from observed inter-arrival gaps, serialise the warm-up before fan-out, verify from `cache_read_input_tokens`.
   - The planner is a deterministic rule engine first; the optional structured LLM call can only tighten it (reconciliation), keeping offline and online behaviour consistent and auditable.
3. **Implementation.** Pydantic v2 models for every config, plan, entry, report and metric; async-first `CachedLLMClient` with sync wrappers from `pipeline_core`; injectable clocks and random sources so TTL, XFetch and eviction are unit-tested without sleeping; 118 offline tests covering the heuristic planner, the LLM-planned path with `ScriptedLLM`, every technique module's pure logic, end-to-end `run()`, status/confidence behaviour, domain adaptation, budget enforcement, the direct API, the CLI and a regression test for every adversarial-review finding.
