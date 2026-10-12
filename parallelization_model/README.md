# Parallelization model (`parallelization`)

Execute LLM and tool work concurrently and safely. The package has two halves
that share one budget and one resilience kernel:

* **a parallel executor** (`ParallelExecutor`, an `Executor` for the other
  models): bounded concurrency, client-side RPM/ITPM/OTPM token buckets,
  full-jitter retries under a retry budget, a circuit breaker, bulkheads,
  per-task timeouts, hedged requests, idempotency keys, deadline propagation,
  `fail_fast | best_effort | quorum` failure policies, and DAG scheduling in
  topological waves;
* **prompt-level parallel reasoning** (`ParallelizationModel.run()`): a planner
  reads the task and routes it to one of eight patterns - sequential,
  sectioning (DAG), skeleton-of-thought, sampling-and-voting, map-reduce,
  branch-solve-merge, layered ensemble, speculative - then executes the pattern
  with warm-then-burst prompt caching and grades the result.

Everything runs offline with `MockLLM`; the full suite (117 tests) finishes in
about a second.

---

## Architecture

```
                         TaskRequest(task, inputs, context, constraints, domain)
                                             |
                                             v
 +-----------------------------------------------------------------------------------+
 | ParallelizationModel (model.py)                                                   |
 |                                                                                   |
 |  plan()  ----> plan.py                                                            |
 |     extract_features()  -> TaskFeatures (enumerations, questions, items, tokens, |
 |                            sequential/numeric/evaluation/long-form markers, risk) |
 |     HeuristicPlanner.choose()        [offline / deterministic / planner_mode]     |
 |       or one structured LLM call     [PLANNER template -> PlannerOutput]          |
 |     HeuristicPlanner.finalize()  -> ParallelPlan (domain policy, budget caps,     |
 |                                     token-multiplier cap, Amdahl estimate)        |
 |                                                                                   |
 |  execute() --> patterns.py                                                        |
 |     PatternRunner  ................ budget guard | PrefixWarmGate | accounting    |
 |        |                                                                          |
 |        |   sequential   sectioning(DAG)   skeleton   voting/USC   map_reduce      |
 |        |   branch_solve_merge   ensemble(MoA)   speculative(fast vs slow)         |
 |        v                                                                          |
 |     ParallelExecutor (executor.py) .... submit / map / gather / spawn-join / dag  |
 |        RateLimiter(TokenBucket x3) -> Bulkhead -> RetryPolicy + RetryBudget       |
 |        -> CircuitBreaker -> timeout / hedge -> FailurePolicy                      |
 |        clock injectable (ManualClock) for deterministic tests                     |
 |                                                                                   |
 |     aggregation.py (code aggregators): canonicalize, majority_vote, entropy,      |
 |                    confidence blends, ordered concat, missing-branch notice       |
 |     dag.py: validate_dag, topological_waves, critical_path                        |
 |     estimator.py: Amdahl bound, fan-out/graph speedup, trade-off table,           |
 |                   recommended_concurrency (Little's law)                          |
 +-----------------------------------------------------------------------------------+
                                             |
                                             v
    TaskResult(status, output=ParallelizationOutput, output_text, confidence,
               confidence_basis, evidence[branches], metadata{plan, executor, cache_gate})
```

Data flow for one fan-out: the runner renders every branch request as
`[template rules][domain profile][rigor]` (stable, byte-identical, cacheable)
plus a user turn whose branch-specific part comes **last**; the gate lets the
first request on a cacheable prefix run alone and releases the siblings once
its response arrives (siblings wait at executor admission, *before* the rate
limiter, bulkhead slot and per-attempt timeout, so the wait never eats their
timeout and is reported as `TaskOutcome.wait_s`, not latency); the executor
applies admission, retries, breaker and policy; outcomes are aggregated in code where the output type allows and by an
LLM merge otherwise.

Module map:

| Module | Contents |
| --- | --- |
| `executor.py` | `ParallelExecutor` (`map`, `map_outcomes`, `gather`, `submit`, `spawn`/`join`, `dag`), `TokenBucket`, `RateLimiter`, `RetryPolicy`, `RetryBudget`, `CircuitBreaker`, `Bulkhead`, `FailurePolicy`, `TaskOutcome`, `ExecutorStats`, `TransientError`, `CircuitOpenError`, `ParallelExecutionError`, `ManualClock`/`MonotonicClock` |
| `patterns.py` | `PatternRunner`, `PrefixWarmGate`, `run_sequential`, `run_sectioning`, `run_skeleton`, `run_voting`, `run_map_reduce`, `run_branch_solve_merge`, `run_ensemble`, `run_speculative`, `speculative(fast, slow, verifier)`, `cache_aware_fan_out`, `parallel_tool_calls` (+ `ToolCall`/`ToolResult`), `generic_map_reduce`, ctx-first wrappers `skeleton_of_thought`, `branch_solve_merge`, `map_reduce`, `ensemble`/`mixture_of_agents`, `voting` |
| `aggregation.py` | `canonicalize_answer`, `majority_vote` -> `VoteResult`, `vote_entropy`, `confidence_from_vote`, `combine_confidence`, `weighted_confidence`, `success_fraction`, `concat_in_order`, `missing_branch_notice`, `label_to_confidence` |
| `dag.py` | `validate_dag`, `topological_waves`, `critical_path`, `total_work`, `transitive_dependents`, `to_mermaid` |
| `estimator.py` | `amdahl_speedup`, `amdahl_bound`, `estimate_fan_out`, `estimate_graph` -> `SpeedupEstimate`, `tradeoff_table` -> `TradeoffRow`, `recommended_concurrency`, `token_multiplier`, `within_token_cap` |
| `plan.py` | `ParallelPlan`, `Partition`, `RateLimits`, `PlannerOutput`, `TaskFeatures`, `extract_features`, `HeuristicPlanner` (`choose`, `finalize`, `plan`, `validate_llm_output`, `partition_cap`), `expected_calls_for`, `expected_speedup_for`, `serial_fraction_for`, `truncate_partitions`, `fold_items`, `chunk_count` |
| `prompts.py` | 15 versioned `PromptTemplate`s and their pydantic output schemas; `SAMPLE_FRAMINGS`, `PROPOSER_PERSPECTIVES`, `system_blocks()` |
| `config.py` | `ParallelizationConfig` (50 fields, table below) |
| `model.py` | `ParallelizationModel`, `ParallelizationOutput`, `build_context`, `run_stats`, `rate_limit_mismatch` |
| `evaluation.py` | `RunMetrics`, `metrics_from_result`, `EvaluationReport`, `aggregate`, `evaluate`, `synthetic_tasks`, `percentile` |
| `cli.py` | `python -m parallelization` |

---

## How it reasons about a prompt

### Plan schema (`ParallelPlan`)

| Field | Meaning |
| --- | --- |
| `pattern` | `sequential`, `sectioning`, `skeleton_of_thought`, `voting`, `map_reduce`, `branch_solve_merge`, `ensemble`, `speculative` |
| `partitions` | `Partition(id, task, depends_on)` list: LLM sub-tasks for sectioning (a DAG), chunk placeholders `c1..cN` for map-reduce, empty otherwise. At most `min(max_partitions, token_multiplier_cap)` (15 by default); see *Partition cap* below |
| `concurrency` | fan-out width per wave = `min(config.max_concurrency, constraints.max_parallelism)` |
| `failure_policy`, `quorum` | `best_effort` for low/medium-risk domains; `quorum` for `high` (0.8) and `safety_critical` (1.0: every branch must succeed) |
| `rate_limits` | the *configured* client-side RPM/ITPM/OTPM limits (informational; `None` = not configured). Enforcement is by the executor's `RateLimiter`, fixed when the executor is built; `execute()` warns when an injected executor enforces different limits |
| `samples`, `max_samples` | initial vote batch and the cap after early-stopping checks; for ensembles `samples` = proposers per layer |
| `layers` | proposer layers (ensemble) |
| `aggregator` | `code` (vote / deterministic), `llm` (merge prompt), `usc` (LLM-selected consistency), `concat` (ordered concatenation) |
| `verifiability` | `exact_match`, `verifiable`, `free_form` - for voting plans, selects the aggregator: `free_form` -> USC selection, otherwise code majority vote. It does not change sample counts |
| `cache_aware` | warm-then-burst gating on (off under `constraints.deterministic`) |
| `branch_effort`, `aggregator_effort` | `output_config.effort` per role |
| `expected_calls`, `expected_speedup` | LLM calls the plan may spend (an upper bound for voting/skeleton) and the critical-path speedup estimate |
| `serial_fraction` | share of `expected_calls` on the serial critical path (planner-independent; Amdahl bound = 1 / `serial_fraction`) |
| `warnings` | planning degradations (partitions dropped by budget or the partition cap); copied into `result.warnings` |
| `rationale`, `planner` | why, and whether the decision came from `heuristic` or `llm` |

### Heuristic decision rules (offline, `deterministic`, or `planner_mode="heuristic"`)

`extract_features` computes word/question counts, enumerations ("compare A, B
and C", "for each of X, Y", numbered lists, several questions), `inputs["items"]`
length, estimated input tokens, and regex marker families
(`sequential`, `numeric`, `evaluation`, `longform`, `high_value`). The routing
table is the dossier's Amdahl-driven mode selection, evaluated in this order:

1. `inputs["items"]` has >= 2 entries -> **sectioning**, one partition per item, LLM merge. Items beyond the partition cap (15 by default) are batched into the last partition rather than dropped, and the rationale says so.
2. input text >= `map_reduce_min_tokens` (3000) -> **map_reduce** with `chunk_count` partitions (falls back to sequential when it fits one chunk).
3. sequential markers (derive, prove, step by step, implement, write a function, refactor, migrate, story, essay ...) and no enumeration -> **sequential**; a calculation with dependent steps -> **voting** with numeric canonicalisation (the engineering adaptation note: never skeleton-expand a derivation).
4. numeric/classification markers with at most one question -> **voting** (`exact_match`, code aggregator).
5. >= 2 enumerated items or questions -> **sectioning**.
6. evaluation markers (evaluate, review, assess, audit, compliance ...) -> **branch_solve_merge**.
7. long-form markers (list, risks, options, considerations, checklist, failure modes ...) -> **skeleton_of_thought**.
8. high-value markers (root cause, recommendation, trade-off, safety case ...) or >= 25 words in a `high`/`safety_critical` domain, when `max_llm_calls` allows `layers x proposers + 2` -> **ensemble**.
9. `max_latency_s <= 30` on a simple task in a low-risk domain -> **speculative**.
10. otherwise -> **sequential** ("start with the simplest approach that works").

`finalize()` then applies domain policy (failure policy, quorum, vote batch
size - for every planner, so an LLM-planned vote in a high-risk domain also
starts at `voting_samples_high_risk`), degrades fan-out to fit
`max_llm_calls` (fewer partitions, fewer samples, or sequential when fewer
than 3 calls remain), enforces the `token_multiplier_cap`, estimates
`expected_calls`/`expected_speedup`/`serial_fraction`, and appends an Amdahl
note when the serial share exceeds `serial_fraction_target`.

**Partition cap.** A plan never has more than `min(max_partitions,
token_multiplier_cap)` partitions (15 with the defaults). Heuristic plans fold
surplus items into the last partition. When an LLM plan or a budget cut must
*drop* partitions, `truncate_partitions` keeps a dependency-closed subset
(dependencies first, so no kept node depends on a dropped one and the DAG
stays valid) and the dropped ids are named in `plan.rationale`,
`plan.warnings` and `result.warnings`; the run degrades instead of failing.

### LLM-driven planning (online, `planner_mode="auto"`; always with `planner_mode="llm"`)

One structured call renders the `parallelization.planner` template: the
routing table as numbered rules in `system_stable` (after it the domain
profile and, for risky domains, `ENGINEERING_RIGOR`), and the task, an inputs
summary, the computed `TaskFeatures` and the limits in the user turn. The
output is validated into `PlannerOutput` (`pattern`, `partitions`, `samples`,
`aggregator`, `verifiability`, `rationale`); duplicate ids, unknown
dependencies and cycles are rejected and the planner falls back to the
heuristic decision with a warning. The same `finalize()` applies, so an LLM
plan can never exceed the budget or ignore the domain's failure policy. The
planner is skipped when fewer than two calls remain.

### Execution and grading

Each pattern returns a `PatternResult` with the answer, a confidence and its
basis. Confidence blends a structural signal with a verbalised one:
`combine_confidence(structural, verbal)` = `0.6 x structural + 0.4 x verbal`,
where structural is the vote share damped by entropy (voting) or the fraction
of branches that succeeded, and verbal is the aggregator's `high|medium|low`
label (0.9 / 0.65 / 0.35). A unanimous vote caps at 0.95 (votes can agree and
be wrong).

Status: `FAILED` when no answer; `NEEDS_REVIEW` when the failure policy was not
satisfied (no answer is assembled from partial results) or when confidence is
below `max(profile.min_confidence_to_accept, constraints.min_confidence)`;
`PARTIAL` when branches failed but the policy allowed aggregation; `SUCCESS`
otherwise. Every successful branch is exposed as `Evidence` for downstream
verification, and `result.metadata` carries the plan, the executor counter
deltas and the cache-gate statistics.

---

## Quick start

### Python

```python
from pipeline_core import Constraints, TaskRequest
from parallelization import ParallelizationModel

model = ParallelizationModel()  # make_llm(): MockLLM without credentials, AnthropicLLM with ANTHROPIC_API_KEY

result = model.run_sync("Compare steel, aluminium and titanium for a lightweight bracket.", domain="mechanical_engineering")
print(result.summary())                       # [parallelization] status=success confidence=0.96 calls=4 ...
print(result.metadata["plan"]["pattern"])     # sectioning
print(result.output.branches[0].id)           # p1
print(result.output_text)

# Items fan out one worker per item; budgets degrade the fan-out instead of failing
result = model.run_sync(TaskRequest(
    task="Classify each ticket as bug, feature or question.",
    inputs={"items": ["Crash on save", "Add dark mode", "How do I export?"]},
    constraints=Constraints(max_llm_calls=4, max_parallelism=2),
))

# Direct API (no RunContext; usage is reported in PatternResult.usage)
import asyncio
vote = asyncio.run(model.vote("Is 97 prime?", samples=5))
sk = asyncio.run(model.skeleton_of_thought("List the risks of the redesign", domain="mechanical_engineering"))
values = asyncio.run(model.map(my_async_fn, items))          # Executor protocol
print(model.tracer.tree())
```

### CLI

```bash
cd "<repo root>"
PYTHONPATH="common;parallelization_model" python -m parallelization --offline "Compare steel, aluminium and titanium for a bracket" --domain mechanical_engineering
PYTHONPATH="common;parallelization_model" python -m parallelization --offline --json "Classify each ticket" --input 'items=["Crash on save","Add dark mode"]' --max-calls 5
PYTHONPATH="common;parallelization_model" python -m parallelization --offline "Summarise failures" --input text=@build.log
PYTHONPATH="common;parallelization_model" python -m parallelization --offline --plan-only "What are the risks of the redesign?"
PYTHONPATH="common;parallelization_model" python -m parallelization --offline --evaluate --domain software_engineering
```

(Use `:` instead of `;` as the `PYTHONPATH` separator on Linux/macOS.) Flags:
`--domain`, `--offline`, `--json`, `--input key=value` (JSON decoded, `@path`
reads a file, repeatable), `--max-calls`, `--max-parallelism`, `--max-latency`,
`--min-confidence`, `--deterministic`, `--plan-only`, `--evaluate`, `--trace`.
The text output prints `result.summary()`, the plan (pattern, concurrency,
policy, expected calls/speedup, rationale), run metrics, the confidence basis
and `output_text`.

---

## Configuration (`ParallelizationConfig`)

Every default is taken from `docs/research/parallelization.json`
(papers' `default_parameters`) or, where the literature leaves a number open,
is `None` and documented as such.

| Field | Default | Justification (source) |
| --- | --- | --- |
| `max_concurrency` | 8 | LLMCompiler fans out 8-16 tasks per level; Anthropic's research system runs 3-5 workers. Kim et al. (2023), *An LLM Compiler for Parallel Function Calling*, https://arxiv.org/abs/2312.04511; Anthropic (2025), *How we built our multi-agent research system*, https://www.anthropic.com/engineering/multi-agent-research-system. Capped by `Constraints.max_parallelism`. |
| `max_partitions` | 16 | Same 8-16 per-level bound (Kim et al. 2023). The effective cap is `min(max_partitions, token_multiplier_cap)` = 15 with the defaults; surplus items are batched, never silently dropped. |
| `requests_per_minute` / `input_tokens_per_minute` / `output_tokens_per_minute` | `None` | Provider limits are per model class and exposed in `anthropic-ratelimit-*` headers; mirror them client-side with a token bucket rather than guess. Anthropic (2025), *Rate limits (Claude Developer Platform documentation)*, https://platform.claude.com/docs/en/api/rate-limits. |
| `rate_limit_burst_s` | 5.0 | 60 RPM may be enforced as ~1 request/s and sharp ramps hit acceleration limits, so the bucket holds only a few seconds of refill (Anthropic rate-limits docs, as above). |
| `retry_max_attempts` | 3 | "At most 3 attempts per request". Cuervo & Ulrich (2016), *Site Reliability Engineering: Handling Overload; Addressing Cascading Failures*, https://sre.google/sre-book/handling-overload/ |
| `retry_base_delay_s` / `retry_max_delay_s` | 0.5 / 20.0 | Full-jitter capped exponential backoff, base 0.5-1 s, cap 20-60 s. Brooker (2015), *Exponential Backoff And Jitter*, https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ |
| `retry_budget_ratio` | 0.10 | Retries under 10% of requests bound amplification to ~1.1x (SRE book, as above). |
| `retry_budget_min` | 3 | Engineering adaptation: a floor so single-request runs can retry at all; the 10% rule concerns aggregate amplification, which the floor does not change. |
| `retry_budget_window_s` | 120.0 | SRE client-side adaptive throttling tracks a 2-minute window (SRE book, as above). |
| `breaker_failure_threshold` / `breaker_failure_rate` / `breaker_window` / `breaker_recovery_s` | 5 / 0.5 / 20 / 30.0 | Closed/open/half-open breaker; thresholds are the dossier's common-practice values (open at >= 50% failures over 20 calls or 5 consecutive, probe after 30 s). Nygard (2018), *Release It! Second Edition*, https://pragprog.com/titles/mnee2/release-it-second-edition/ |
| `per_task_timeout_s` | `None` | Literature: 2-3x the observed p95 per task class - unknown until measured, so unset. Dean & Barroso (2013), *The Tail at Scale*, https://research.google/pubs/the-tail-at-scale/ |
| `hedge_after_s` | `None` | Hedge at the observed p95 (Dean & Barroso 2013); unset until a latency distribution exists. |
| `hedge_fraction` | 0.05 | Cap hedged duplicates at ~5% of requests (Dean & Barroso 2013). |
| `quorum_fraction` | 0.8 | Aggregator proceeds with partial results when >= 80% of branches complete (Dean & Barroso 2013; dossier default). Overridden to 1.0 for `safety_critical` profiles. |
| `voting_samples` / `voting_max_samples` | 5 / 10 | n0 = 5, n_max = 10 for typical tasks. Li et al. (2024), *More Agents Is All You Need*, https://arxiv.org/abs/2402.05120; Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, https://arxiv.org/abs/2203.11171 |
| `voting_samples_high_risk` | 10 | Hard/high-stakes tasks warrant 15-20 agents (Li et al. 2024); we start high-risk domains at n_max = 10 with no early stop below it, keeping the run inside the 15x token cap (see `token_multiplier_cap`). |
| `early_stop_margin` / `early_stop_leader_share` | 2 / 0.6 | Stop sampling when the leader is >= 2 votes ahead or holds >= 60% (dossier technique "Adaptive sampling-and-voting", derived from Li et al. 2024 and Snell et al. (2024), *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, https://arxiv.org/abs/2408.03314). |
| `usc_max_candidates` | 8 | USC input grows with n x answer length; cap n at 5-8. Chen et al. (2023), *Universal Self-Consistency for Large Language Model Generation*, https://arxiv.org/abs/2311.17311 |
| `skeleton_min_points` / `skeleton_max_points` | 3 / 10 | Skeleton prompts request 3-10 short points. Ning et al. (2023), *Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation*, https://arxiv.org/abs/2307.15337 |
| `expansion_max_words` | 150 | Per-point expansion cap ~100-200 words (Ning et al. 2023). |
| `coherence_pass` | `False` | SoT's optional light coherence pass; off by default because it adds a serial call to the critical path (Ning et al. 2023; Amdahl (1967), *Validity of the single processor approach to achieving large scale computing capabilities*, https://en.wikipedia.org/wiki/Amdahl%27s_law). |
| `bsm_min_branches` / `bsm_max_branches` | 3 / 6 | 3-6 model-proposed criteria. Saha et al. (2023), *Branch-Solve-Merge Improves Large Language Model Evaluation and Generation*, https://arxiv.org/abs/2310.15123 |
| `chunk_tokens` | 4000 | Map chunks of 4k-16k tokens. Zhou et al. (2024), *LLM x MapReduce: Simplified Long-Sequence Processing using Large Language Models*, https://arxiv.org/abs/2410.09342 |
| `chunk_overlap_ratio` | 0.10 | 10-15% overlap for prose to mitigate inter-chunk dependencies (Zhou et al. 2024). |
| `map_reduce_min_tokens` | 3000 | Engineering choice: below roughly one chunk there is nothing to split, so the input goes in one call. |
| `reduce_fan_in` | 8 | 5-10 records per reducer for hierarchical (associative) reduction (Zhou et al. 2024; Dean & Ghemawat (2004), *MapReduce: Simplified Data Processing on Large Clusters*, https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/). |
| `ensemble_layers` / `ensemble_proposers` | 2 / 3 | MoA-Lite: 2 layers, 3-6 proposers. Wang et al. (2024), *Mixture-of-Agents Enhances Large Language Model Capabilities*, https://arxiv.org/abs/2406.04692 |
| `homogeneous_ensemble` / `ensemble_proposer_models` | `True` / `[]` | Self-MoA beats mixed MoA by 6.6%: default to samples of the best model; heterogeneous proposers are opt-in. Li et al. (2025), *Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?*, https://arxiv.org/abs/2502.00674 |
| `token_multiplier_cap` | 15.0 | Multi-agent systems use ~15x chat tokens; cap parallel calls per task at that multiple (Anthropic 2025, multi-agent research system, as above). |
| `serial_fraction_target` | 0.30 | Keep planner + aggregator under ~30% of the critical path (ceiling 3.3x); the plan rationale flags plans above it (Amdahl 1967; dossier technique "Amdahl-driven critical-path budgeting"). |
| `cache_aware_fan_out` | `True` | "A cache entry becomes available only after the first response begins ... send the first request and wait before sending the others." Anthropic (2025), *Prompt caching (Claude Developer Platform documentation)*, https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching |
| `fast_model` | `claude-haiku-5-5` | Propose-cheap / verify-strong is the speculative pattern at task level. Leviathan et al. (2022), *Fast Inference from Transformers via Speculative Decoding*, https://arxiv.org/abs/2211.17192; Miao et al. (2023), *SpecInfer*, https://arxiv.org/abs/2305.09781. Exact model id, no date suffix (ARCHITECTURE 3.2). |
| `planner_effort` / `branch_effort` / `map_effort` / `aggregator_effort` | high / medium / low / high | ARCHITECTURE 3.2: `high` for planning and synthesis, `low` for many-item grading; mappers are the many-item case, branches sit between. |
| `planner_mode` | `auto` | `auto` = one structured call online, heuristics offline/deterministic; `llm` forces the LLM planner (ScriptedLLM/ReplayLLM tests); `heuristic` never plans with the LLM. |
| `ordered_tools` | `[]` | Disable parallel execution for tools with ordering side effects; run them sequentially under their own bulkhead, attempted exactly once (no retry, hedge or memoisation). OpenAI (2025), *Function calling guide (parallel function calling)*, https://developers.openai.com/api/docs/guides/function-calling |

Constructor overrides: `ParallelizationModel(llm, config=..., executor=..., max_concurrency=4, ...)`.

---

## Techniques implemented

| Technique | Where | Establishing sources |
| --- | --- | --- |
| Planner -> DAG -> dependency-aware scheduler -> concurrent executor (placeholder binding at dispatch: dependency outputs are injected into the dependent sub-task's user turn) | `plan.py` (`Partition.depends_on`, `validate_dag`), `executor.dag`, `run_sectioning` | Kim et al. (2023), *An LLM Compiler for Parallel Function Calling*, https://arxiv.org/abs/2312.04511; Xu et al. (2023), *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*, https://arxiv.org/abs/2305.18323 |
| Sectioning and voting as explicit modes picked by a router; aggregation in code where possible | `HeuristicPlanner.choose`, `aggregation.py` | Schluntz & Zhang (2024), *Building effective agents*, https://www.anthropic.com/research/building-effective-agents |
| Skeleton-then-expand with a suitability router (SoT-R): derivations, code, narrative go sequential | `run_skeleton`, `_SEQUENTIAL_MARKERS` | Ning et al. (2023), *Skeleton-of-Thought*, https://arxiv.org/abs/2307.15337 |
| Sampling-and-voting with canonicalised answers, early stopping and vote entropy as a confidence signal | `run_voting`, `majority_vote`, `confidence_from_vote` | Wang et al. (2022), *Self-Consistency*, https://arxiv.org/abs/2203.11171; Li et al. (2024), *More Agents Is All You Need*, https://arxiv.org/abs/2402.05120; Snell et al. (2024), https://arxiv.org/abs/2408.03314 |
| Universal Self-Consistency for free-form answers | `run_voting` with `aggregator="usc"` | Chen et al. (2023), *Universal Self-Consistency*, https://arxiv.org/abs/2311.17311 |
| Verifiability flag driving the sample budget | `PlannerOutput.verifiability`, `finalize` | Brown et al. (2024), *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, https://arxiv.org/abs/2407.21787 |
| Diversity without sampling knobs: ten reasoning framings, six proposer perspectives | `SAMPLE_FRAMINGS`, `PROPOSER_PERSPECTIVES` | ARCHITECTURE 3.2 (no `temperature`); Wang et al. (2022) |
| Branch-Solve-Merge for evaluation and multi-constraint generation | `run_branch_solve_merge` | Saha et al. (2023), *Branch-Solve-Merge*, https://arxiv.org/abs/2310.15123 |
| LLM map-reduce with a structured information protocol (`{extracted_evidence, rationale, answer, confidence}`), a shared global summary, overlap and confidence-aware hierarchical reduce | `run_map_reduce`, `MapRecord`, `ReduceOutput`, `global_summary` | Zhou et al. (2024), *LLM x MapReduce*, https://arxiv.org/abs/2410.09342; Dean & Ghemawat (2004), *MapReduce*, https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/ |
| Layered proposer/aggregator ensemble (MoA-Lite), homogeneous by default, critical synthesis prompt | `run_ensemble`, `AGGREGATE` | Wang et al. (2024), *Mixture-of-Agents*, https://arxiv.org/abs/2406.04692; Li et al. (2025), *Rethinking Mixture-of-Agents*, https://arxiv.org/abs/2502.00674 |
| Speculative execution: cheap drafter, strong verifier, loser cancelled | `speculative`, `run_speculative` | Leviathan et al. (2022), https://arxiv.org/abs/2211.17192; Miao et al. (2023), *SpecInfer*, https://arxiv.org/abs/2305.09781 |
| Token-bucket admission for RPM/ITPM/OTPM with header feedback and gradual ramp | `TokenBucket`, `RateLimiter.feedback` | Anthropic (2025), *Rate limits*, https://platform.claude.com/docs/en/api/rate-limits |
| Full-jitter exponential backoff honouring `retry-after` as a floor | `RetryPolicy.delay` | Brooker (2015), https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ |
| 3 attempts, 10% retry budget, retriable vs non-retriable error classes, deadline propagation, criticality tags | `RetryPolicy`, `RetryBudget`, `ParallelExecutor._execute` | Cuervo & Ulrich (2016), SRE book, https://sre.google/sre-book/handling-overload/ |
| Circuit breaker (closed/open/half-open) and bulkheads per resource | `CircuitBreaker`, `Bulkhead` | Nygard (2018), *Release It! Second Edition*, https://pragprog.com/titles/mnee2/release-it-second-edition/ |
| Hedged requests for idempotent work, partial-result aggregation with an explicit missing-branch notice | `ParallelExecutor._attempt`, `missing_branch_notice` | Dean & Barroso (2013), *The Tail at Scale*, https://research.google/pubs/the-tail-at-scale/ |
| Warm-then-burst cache-aware fan-out; stable-first, byte-identical shared prefix with the branch suffix last | `PrefixWarmGate`, `PatternRunner.request` | Anthropic (2025), *Prompt caching*, https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching; Zheng et al. (2023), *SGLang*, https://arxiv.org/abs/2312.07104 |
| Idempotency keys and single-flight de-duplication; re-executable branches | `ParallelExecutor.submit` | Dean & Ghemawat (2004), *MapReduce* |
| `spawn`/`join` primitives | `ParallelExecutor.spawn`, `join` | Pan et al. (2025), *Learning Adaptive Parallel Reasoning with Language Models*, https://arxiv.org/abs/2504.15466 |
| Parallel tool calls keyed by `call_id`, ordered tools sequential | `parallel_tool_calls` | OpenAI (2025), *Function calling guide*, https://developers.openai.com/api/docs/guides/function-calling |
| Amdahl / critical-path budgeting, cost-latency trade-off table, Little's-law concurrency recommendation, token-multiplier gating | `estimator.py`, `finalize` | Amdahl (1967), https://en.wikipedia.org/wiki/Amdahl%27s_law; Anthropic (2025), multi-agent research system |

Not implemented (deliberately): Tree-of-Thoughts search (Yao et al. 2023) and
multi-agent debate (Du et al. 2023) - the dossier's own guidance is that
independent sampling scales more cheaply; the deferred Batch-API tier
(Anthropic (2025), *Batch processing (Message Batches API documentation)*, https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing) requires a provider client this package
does not own. See *Limitations*.

---

## Adapting to an engineering domain

Nothing in the package names a domain. Behaviour follows the `DomainProfile`
read from `ctx.profile`:

| Profile field | Effect in this model |
| --- | --- |
| `risk_level` `high` / `safety_critical` | failure policy `quorum` (0.8 / **1.0**), initial vote batch `voting_samples_high_risk` (10) with no early stop below it, ensembles allowed for high-value free-form tasks, `ENGINEERING_RIGOR` appended to every stable prefix, speculative racing disabled |
| `min_confidence_to_accept` | acceptance threshold (`NEEDS_REVIEW` below it, together with `constraints.min_confidence`) |
| `system_text()` (glossary, units, standards, verification rules) | second stable system block of every branch, so all branches share one cacheable prefix |

Register a new profile and the same code changes behaviour:

```python
from pipeline_core import DomainProfile, TaskRequest, register_profile
from parallelization import ParallelizationModel

register_profile(DomainProfile(
    name="pressure_vessels",
    description="Pressure vessel design and inspection under ASME BPVC Section VIII.",
    units=["MPa", "bar", "mm", "degC", "kN"],
    verification_rules=["Every allowable stress must cite the code table and the metal design condition."],
    risk_level="safety_critical",
    min_confidence_to_accept=0.85,
    standards=["ASME BPVC VIII-1", "API 510"],
))

model = ParallelizationModel()
task = "Calculate the required shell thickness for a 2 m vessel at 1.5 MPa with allowable stress 138 MPa."
general = model.run_sync(task)                                   # voting, 5 samples, best_effort, threshold 0.5
vessel = model.run_sync(TaskRequest(task=task, domain="pressure_vessels"))  # voting, 10 samples, quorum=1.0, threshold 0.85, rigor rules in prompt
```

`examples/engineering_example.py` runs this comparison and adds a map-reduce
over an inspection log, a branch-solve-merge compliance review and parallel
tool calls with an ordered PLM write. Engineering-specific routing from the
dossier's adaptation notes is built in: calculations are never skeleton-expanded
(they vote with numeric canonicalisation that keeps units), files/parts/clauses/
tickets supplied as `inputs["items"]` become the fan-out key, long specs and
logs are map-reduced with structured records, and procedures/migrations/
derivations run sequentially.

---

## Composing with the other five models

The component never imports another model package; it exposes and accepts
`pipeline_core` protocols.

| Model | How it composes |
| --- | --- |
| **Task decomposition** | consumes `ParallelizationModel.executor` (or the model itself, which also satisfies `Executor.map`) to run its `TaskGraph` waves; `executor.dag(graph, fn)` executes any `{node: [deps]}` graph with wave scheduling and dependency-failure skipping. Decomposition plans with explicit dependencies are exactly what `run_sectioning` executes. |
| **Hybrid retrieval** | fan out sparse/dense legs or multi-query rewrites with `executor.map` / `gather`; prefetch retrieval for all sub-questions before generation so retrieval leaves the critical path. Retrieved `Evidence` in `request.context` is rendered as `<documents>` for every branch. |
| **Answer verification** | `result.evidence` carries every branch output for grounding checks; `vote_entropy` / `confidence_basis` give the verifier a calibration signal; verification itself is a sectioning (criteria) or voting (several verifier prompts) fan-out that can run on this executor. `result.status == NEEDS_REVIEW` is the hand-off point. |
| **Caching** | the stable-first prefix layout and `PrefixWarmGate` make provider prompt caching effective; wrap the client with `CachingModel.wrap(llm)` and identical branches (two mappers over one chunk) are coalesced before dispatch; `TaskOutcome.cached_idempotent` reports executor-level memo hits. |
| **Pipeline RAG** | inject the executor as `executor=`; map-reduce over retrieved chunks is the parallel form of long-context RAG and sectioning over sub-questions the parallel form of multi-hop RAG. |

Nested fan-outs (orchestrator-workers) should share **one** executor instance
so the rate limiter, retry budget and breaker are global, per the SRE guidance
on multi-layer retry amplification. Nesting is safe: the executor tracks how
deep the current task is inside its own slots (a `contextvars.ContextVar`) and
routes a `map`/`gather`/`dag`/`submit` issued from inside running work to a
per-level bulkhead pool (`<resource>@nested1`, `@nested2`, ...; each inherits
`<resource>`'s limit). A decomposition wave holding every slot can therefore
call `executor.map` for its RAG retrieval or verifier sampling without
deadlocking, while rate limits, retry budget, breaker and stats stay shared.
The rate limiter still bounds total throughput; total in-flight work can reach
`max_concurrency` per nesting level. Pass `nested_bulkheads=False` to the
constructor to restore a single pool per resource (only safe when work never
re-enters the executor).

Per-run counters are attributed exactly even when several runs share the
executor concurrently: the runner passes its own `ExecutorStats` to every
executor call (`submit(..., stats=...)`), and that object is what
`result.metadata["executor"]` reports. The executor-wide `executor.stats`
keeps lifetime totals; its `latencies_s` is bounded to the most recent 10,000
samples.

---

## Evaluation

`evaluation.py` computes the dossier's metrics from a `TaskResult`:

| Metric | Field in `RunMetrics` |
| --- | --- |
| speedup = sequential latency / parallel latency (sum of call latencies / wall-clock) | `measured_speedup`, `expected_speedup` |
| parallel efficiency = speedup / width; utilisation = work / (wall x width) | `efficiency`, `utilisation` |
| Amdahl bound 1 / serial fraction, with the serial fraction taken from the plan's call structure (`plan.serial_fraction`; e.g. a 2-layer x 3-proposer ensemble: 3 serial of 7 calls -> 2.33x; a code-aggregated vote has no serial call -> reported as 1e9) | `serial_fraction`, `amdahl_bound` |
| p50 / p95 branch latency | `p50_latency_s`, `p95_latency_s` |
| retries and retry rate = retries / submitted (target < 10%; the same denominator in `RunMetrics` and `EvaluationReport`), timeouts, hedges, breaker trips and rejections, rate-limit wait | `submitted`, `retries`, `retry_rate`, `timeouts`, `hedges`, `breaker_trips`, `breaker_rejections`, `rate_limit_wait_s` |
| partial-failure rate | `partial_failure_rate`, `failed_branches` |
| prompt-cache hit ratio and warm-then-burst counts | `cache_hit_ratio`, `cache_warmups`, `cache_waits` |
| token multiplier vs a single call, cost | `token_multiplier`, `cost_usd` |

Run the harness (offline, nine synthetic engineering tasks covering all eight
patterns; the ensemble task runs in `mechanical_engineering`, a high-risk
domain, because the router reserves ensembles for high-risk domains, and the
speculative task carries `max_latency_s=20`):

```bash
PYTHONPATH="common;parallelization_model" python -m parallelization --offline --evaluate --domain mechanical_engineering
```

```python
from parallelization import ParallelizationModel, evaluate, synthetic_tasks, metrics_from_result
report = evaluate(ParallelizationModel(), synthetic_tasks("software_engineering"))
print(report.table()); print(report.per_run[0])
```

With `MockLLM` latencies are ~0 so `measured_speedup` reads 1.0; the harness
is meant to be pointed at `RecordingLLM`/`ReplayLLM` captures or a live
client. Executor-level behaviour (retries, breaker, hedging, timeouts) is
unit-tested with `ManualClock` in `tests/test_executor.py`.

Tests: `python -m pytest parallelization_model/tests -q` (117 tests, < 3 s; regressions for the review findings live in `tests/test_review_fixes.py`).

---

## Limitations and failure modes

From the dossier's pitfalls, with how this package mitigates (or does not):

* **Parallelising step-dependent content** degrades math, code and writing (Ning et al. 2023). Mitigated by the sequential-marker router; the regex router will misroute unusual phrasings - the LLM planner is the stronger router online, and `planner_mode="heuristic"` is the conservative choice.
* **Voting saturates and tapers** on unverifiable or too-hard tasks (Brown et al. 2024; Li et al. 2024). Mitigated by `max_samples`, USC for free-form answers and the verifiability flag; there is no learned difficulty estimator (Snell et al. 2024) - difficulty is proxied by first-batch disagreement only.
* **Mixing weaker models lowers quality** (Li et al. 2025). Ensembles are homogeneous unless `homogeneous_ensemble=False` and `ensemble_proposer_models` are set; no eval gate enforces the "within a few points" condition.
* **Serial layers inflate time-to-first-token** (Wang et al. 2024). Ensembles default to 2 layers and the plan rationale flags plans whose serial share exceeds 30%.
* **Token-cost explosion** (~15x for multi-agent systems, Anthropic 2025). Gated by `token_multiplier_cap`, `max_llm_calls` degradation and the high-value/high-risk gate for ensembles.
* **Lost cross-chunk context and conflicting partials** in map-reduce (Zhou et al. 2024). Mitigated by overlap, a deterministic global summary and confidence-aware reduction; the summary is the document head, not an LLM summary, so dependencies on definitions deep in the document can still be lost.
* **Retry storms** (Brooker 2015; SRE book). Full jitter, 3 attempts, 10% budget; nested executors must share one instance or retries multiply across layers.
* **Cache misses on parallel bursts** (Anthropic prompt-caching docs). Warm-then-burst gating is in place but only when the shared prefix exceeds the model's cacheable minimum (512 tokens on current models); short prompts are not gated because nothing would cache. Hit rate is only observable with a real client (`cache_read_input_tokens`).
* **Tail-latency amplification** (Dean & Barroso 2013). Hedging and per-task timeouts exist but are off by default because they need measured p95s; without them a 10-way fan-out completes at its slowest branch.
* **Hedging without cancellation** still bills the loser on APIs that cannot cancel; the hedge fraction caps the waste.
* **Non-idempotent re-execution**: the executor cannot detect side effects. Work submitted with `idempotent=False` (every `ordered_tools` call) is attempted exactly once - no retry, no hedge, no memoisation - so a write whose acknowledgement is lost surfaces as a failed call rather than being re-sent; the caller must reconcile it. Concurrent (non-ordered) tools are retried on transient errors unless `parallel_tool_calls(..., idempotent=False)` is passed.
* **Circuit breaker scope**: only dependency-class failures (the retryable classes: `TransientError`, `LLMError`, timeouts, connection errors) count against the breaker; caller errors such as an unknown tool name, `ValueError` or `BudgetExceededError` do not. There is one breaker per executor, not one per resource.
* **Partial failure under `quorum`**: no answer is assembled when the policy fails (`NEEDS_REVIEW`, confidence 0). In a `safety_critical` domain a single failed branch therefore blocks the answer by design.
* **No durable execution / checkpointing** across process restarts (Anthropic 2025): branch outputs live in `result.output.branches` for the run only. No deferred Batch-API tier. No Tree-of-Thoughts or debate patterns.
* **Offline estimates**: `expected_calls` is an upper bound for voting and skeleton patterns (actual calls depend on early stopping and skeleton length) and `expected_speedup` assumes equal call latencies.
* **Branch outputs are not verified**: this model's confidence is a routing signal derived from agreement and verbalised confidence, not a verdict - pair it with the answer-verification model.

---

## How it was built

1. **Research.** The two dossiers (`docs/research/parallelization.json`,
   `docs/research/pipeline-architecture.json`) were read for
   `key_contribution`/`design_implication` per paper, technique
   `default_parameters`, the 14 architecture recommendations, pitfalls,
   evaluation metrics and the engineering adaptation notes.
2. **Design decisions taken from the literature.**
   * Make the executor a *dataflow engine with a resilience substrate*, not a thread pool (recommendations 1 and 4): `dag()` over `{node: deps}` with wave scheduling; token buckets, jittered retries under a budget, breaker, bulkheads, deadlines, criticality.
   * Front it with an *explicit, evaluable router* (recommendation 2, SoT-R): the routing table lives in `HeuristicPlanner.choose` and, verbatim as numbered rules, in the planner prompt, so the heuristic and LLM paths make the same kind of decision and both pass through one `finalize()`.
   * Separate *aggregation in code* from *aggregation by LLM* (recommendation 7) and default to code: `aggregation.py` is pure and fully unit-tested; LLM merges are used only where the output type requires them.
   * *Verifiability drives the sample budget* (recommendation 8): exact-match answers vote with early stopping; free-form answers cap at the initial batch and use USC.
   * *Cache-aware by construction* (recommendation 5): stable-first prompt layout in `PatternRunner.request` and the `PrefixWarmGate`.
   * *Homogeneous ensembles by default* (recommendation 14, Self-MoA).
   * *Instrument for Amdahl* (recommendation 10): per-branch latency, measured vs expected speedup, serial-share note in the plan.
   * Where the spec's suggested default and the literature differed, the literature won and the table above says why: for example `voting_samples_high_risk=10` rather than the 15-20 agents of the hard-task regime, because the 15x token cap (Anthropic 2025) binds first; `coherence_pass=False` because Amdahl penalises the extra serial call; `retry_budget_min=3` is an explicit engineering floor on the SRE 10% rule.
3. **Implementation order.** Pure modules first (`dag`, `estimator`,
   `aggregation`, `executor` with an injectable clock), then `plan`, `prompts`,
   `patterns`, `model`, `evaluation`, `cli`; tests were written against the
   pure modules without an LLM and against the model with `MockLLM`
   (heuristic path) and `ScriptedLLM` (LLM-planned path with a hand-written
   plan JSON). The README was written last against the code.
