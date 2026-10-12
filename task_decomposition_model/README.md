# Task Decomposition (`task_decomposition`)

Turns a complex natural-language task into a **validated, executable plan**: a
sequence or DAG of sub-tasks with dependencies, inputs, expected outputs and
verification criteria. It then schedules the graph wave by wave with bounded
concurrency, solves each node through a pluggable solver, re-plans the failed
region on external failure signals, and synthesises a traceable answer.
Strategy is **chosen per task** from the task text and the engineering domain
profile: direct answer, Plan-and-Solve, Least-to-Most chain, LLMCompiler-style
parallel DAG, ADaPT-style recursive decomposition, or a bounded
Tree-of-Thoughts search over candidate decompositions.

Everything runs offline: without credentials `make_llm()` returns a `MockLLM`,
the `HeuristicPlanner` chooses the strategy, the heuristic splitter builds the
graph, and the status/confidence logic is exercised end to end. With
`ANTHROPIC_API_KEY` set the same code plans, decomposes, solves, re-plans and
synthesises with Claude.

---

## 1. Purpose

Engineering tasks arrive as one prompt but contain many steps: "for each of the
three pump test logs extract the duty point, flag deviations from the
datasheet, then summarise the fleet status" is three independent extractions
plus an aggregation. One model call either forgets a step or composes the
parts wrongly; the compositionality gap does not close with scale (Press et
al. 2022). Decomposing blindly is also wrong: Least-to-Most is roughly neutral
on two-step problems and pays off at five or more (Zhou et al. 2022), and LLM
planners emit invalid plans often enough that plans must be validated, never
trusted (Shen et al. 2023; Valmeekam et al. 2023).

This package therefore (a) decides **whether and how** to decompose, (b)
validates every graph before execution (acyclic, dependencies exist,
depth/breadth/node caps, no duplicate nodes), (c) runs it under hard budgets
(calls, cost, deadline, retries, replans, recursion depth), (d) triggers
re-planning only from external signals (solver failure, verifier verdict,
schema validation), never from the model re-reading its own output (Huang et
al. 2023), and (e) scores plans step-wise so decomposition failures are not
hidden behind final-answer accuracy (Huang et al. 2024).

## 2. Architecture

```
                 TaskRequest(task, inputs, context, constraints, domain)
                                      |
                                      v
   +---------------------------- plan() ------------------------------+
   | extract_features ---> HeuristicPlanner (offline / deterministic)  |
   |        \----------> PLAN_TEMPLATE, 1 structured LLM call (online) |
   |                      -> DecompositionPlan -> normalize_plan        |
   +------------------------------+-----------------------------------+
                                  v
   +--------------------------- execute() -----------------------------+
   | 1 decompose   strategies.py  Direct | PlanAndSolve | LeastToMost   |
   |               | DAGPlanner | RecursiveDecomposer | TreeOfThoughts  |
   |               LLM graph (validate -> repair x1 -> heuristic)       |
   |               -> TaskGraph (graph.py: waves, critical path)        |
   |                                                                    |
   | 2 schedule    scheduler.py  waves -> Executor.map (bounded)        |
   |               $id.output substitution, retries w/ jitter, budget   |
   |               reservation per node, StepRecord per sub-task        |
   |                    |  solvers.py  LLMSubTaskSolver (SOLVE/PS+)     |
   |                    |             +Retriever (per-sub-question)      |
   |                    |             +Verifier (strict: per node)       |
   |                    |             ProtocolSolver (SubTaskSolver)     |
   |                    |  VerifierNodeSolver (Verifier injected):       |
   |                    |    verify nodes -> Verifier(upstream outputs,  |
   |                    |    upstream evidence) -> Verdict, not solver   |
   |                    |  recursive: expand failed node -> sub-graph    |
   |                                                                    |
   | 3 replan      replanner.py  replannable_failures -> Replanner       |
   |               Reflection -> ReflectionMemory (FIFO cap)             |
   |               Revision -> apply_revision -> re-schedule (<= cap)    |
   |                                                                    |
   | 4 synthesise  synthesis.py  SYNTHESIZE_TEMPLATE (re-ask original)   |
   |               or heuristic sections; aggregate_confidence;         |
   |               verification-node verdicts are never composed        |
   |                                                                    |
   | 5 verify      injected Verifier on claims_text(answer) with the    |
   |               evidence every sub-task collected (optional)         |
   | 6 assess      confidence, confidence_basis, Status                 |
   +--------------------------------------------------------------------+
                                  v
      TaskResult(output=DecompositionOutput(plan, graph, results, synthesis,
                 metrics, waves, replans, reflections, expansions, verdict,
                 node_verdicts),
                 output_text, evidence, confidence, steps, usage, warnings)
```

| Module | Contents |
| --- | --- |
| `graph.py` | `SubTask(id, description, depends_on, inputs, expected_output, verification_criteria, tool_hint, complexity, max_retries)`, `TaskGraph` (`validate_structure`/`ensure_valid`: acyclic, dependencies exist, depth/breadth/node bounds, duplicate detection; `topological_waves`, `critical_path_length`, `depth`, `breadth`, `parallel_speedup_estimate`, `infer_dependencies`, `to_mermaid`, `to_ascii`), placeholder helpers `find_placeholders`, `substitute_placeholders` (`$t1`, `${t1}`, `$t1.output`), pure DAG functions `find_cycle`, `topological_waves`, `critical_path_length` |
| `plan.py` | `DecompositionPlan`, `TaskFeatures`, `extract_features`, `features_text`, `normalize_plan`, `HeuristicPlanner`, `STRATEGIES` |
| `strategies.py` | `Strategy` base (LLM decomposition with bounded repair, heuristic fallback), `DirectStrategy`, `PlanAndSolve`, `LeastToMost`, `DAGPlanner`, `RecursiveDecomposer` (`expand`), `TreeOfThoughtsPlanner` (propose, vote, refine, select), `heuristic_segments`, `heuristic_graph`, `chainify`, `structural_score`, `aggregate_votes`, `get_strategy`, `STRATEGY_CLASSES` |
| `scheduler.py` | `Scheduler` (waves through an `Executor`, default `BoundedExecutor` on `asyncio.gather` with a semaphore; dependency outputs substituted into inputs; budget/deadline stop; per-node budget reservation; per-task retries; `StepRecord`s), `RetryPolicy` (exponential backoff, full jitter, retryable classification), `SubTaskResult`, `ScheduleReport` |
| `solvers.py` | `LLMSubTaskSolver` (default leaf solver; optional `Retriever` and `Verifier`), `ProtocolSolver` (adapts any `SubTaskSolver`), `VerifierNodeSolver` (wraps either when a `Verifier` is injected and answers verification nodes with a verdict), `is_verification_node`/`is_verification_result`/`node_verdict`, `evidence_from_inputs`, `render_prior`, `offline_confidence`, `check_budget_before` (calls, cost and deadline gate before a nested call), `charge_nested_usage`/`charge_verdict_usage` (overshoot is warned about, never raised) |
| `replanner.py` | `Replanner` (`revise`, `heuristic_revision`, `materialise`), `ReflectionMemory`, `replannable_failures`, `apply_revision`, `NON_REPLANNABLE_PREFIXES` |
| `synthesis.py` | `Synthesizer` (`synthesize` LLM path, `heuristic` path; verification-node verdicts are excluded from the composition, the confidence and the coverage and listed under `unresolved` when they did not pass), `aggregate_confidence`, `coverage_ratio`, `claims_text` (the answer without Markdown heading lines, the text the final verifier checks) |
| `schemas.py` | LLM-facing structured-output models: `GraphDraft`/`SubTaskDraft`, `CandidateGraphs`/`CandidateScores`, `SubTaskAnswer`, `Reflection`/`Revision`, `SynthesisOutput`, `CoverageJudgement`, `schema_for` |
| `prompts.py` | Versioned `PromptTemplate`s: `PLAN_TEMPLATE`, `DECOMPOSE_TEMPLATE`, `CANDIDATES_TEMPLATE`, `REFINE_TEMPLATE`, `EVALUATE_TEMPLATE`, `SOLVE_TEMPLATE`, `PLAN_AND_SOLVE_TEMPLATE`, `REPLAN_TEMPLATE`, `SYNTHESIZE_TEMPLATE`, `COVERAGE_JUDGE_TEMPLATE`, `PS_PLUS_TRIGGER` |
| `prompting.py` | `build_request` (stable blocks: template system, few-shot examples, domain profile, `ENGINEERING_RIGOR` for high/safety-critical; volatile user turn), `render_inputs`, `render_documents` |
| `evaluation.py` | `PlanQualityReport`, `evaluate_plan`, `coverage` (keyword judge), `judge_coverage`/`llm_coverage` (LLM judge, optionally through a `RunContext`), `dependency_correctness`, `parallel_speedup`, `compositionality_gap`, `failure_classes`, `subtask_success_rate`, `EvalCase`, `CaseOutcome`, `EvaluationSummary`, `evaluate`, `evaluate_sync`, `sample_cases` |
| `eval_cli.py` | `python -m task_decomposition.eval_cli [--json] [--cases file.json] [--judge keyword\|llm] [--online]` (section 9) |
| `config.py` | `TaskDecompositionConfig` (section 5) |
| `model.py` | `TaskDecompositionModel` (`spec`, `plan()`, `execute()`, direct API), `DecompositionOutput` |
| `cli.py` | `python -m task_decomposition "task" [--domain X] [--offline] [--json] [--input k=v] [--strategy S] [--max-calls N] [--max-parallelism N] [--deterministic] [--plan-only] [--trace]` |

The component imports only `pipeline_core`. Other models reach it through
`run()`/`decompose()` and are reached by it only through the protocol objects
`SubTaskSolver`, `Retriever`, `Verifier` and `Executor`.

## 3. How it reasons about a prompt

### 3.1 The plan (`DecompositionPlan`)

`plan()` answers one question, *how should this task be decomposed?*, and
returns a flat, enum-typed pydantic model that doubles as the LLM planner's
output schema:

| Field | Values | Meaning |
| --- | --- | --- |
| `strategy` | `direct`, `plan_and_solve`, `least_to_most`, `dag`, `recursive`, `tree_of_thoughts` | Which `Strategy` class builds the graph |
| `max_depth` | int >= 1 | Wave depth / recursion bound for this run |
| `max_breadth` | int >= 1 | Maximum sub-tasks per level |
| `parallelism` | int >= 1 | Concurrent sub-tasks per wave |
| `verification` | `none`, `criteria`, `external`, `strict` | `criteria`: every node carries checkable criteria; `external`: an injected `Verifier` checks the composed answer; `strict`: additionally every node is verified and a failed verdict fails the node |
| `complexity_estimate` | `trivial`, `low`, `medium`, `high` | Coarse size class |
| `estimated_steps` | int >= 1 | Estimated reasoning or action steps |
| `replan_on_failure` | bool | Allow bounded re-planning |
| `rationale` | str | One or two sentences, prefixed `[heuristic]` or `[llm]` |

The plan is attached to `result.metadata["plan"]` and to `output.plan`.

### 3.2 Features (`extract_features`)

Cheap deterministic signals computed from the request without a model call:
word/sentence/clause counts, conjunctions, **sequence markers** (`then`,
`after that`, `based on`, `finally`, ...), **independence markers** (`each`,
`every`, `per`, `in parallel`, ...), **exploration markers** (`alternatives`,
`trade-off`, `compare`, `optimal order`, ...), **investigation markers**
(`debug`, `root cause`, `incident`, `fails intermittently`, ...), enumeration
items (numbered/bulleted lines, or three or more comma-separated independent
items; parts that open with an ordering word such as `then size the beam` are
chain steps and do not count), action verbs, questions, presence of documents
(`context` or `inputs.documents`), the profile's `decomposition_hints` count
and `risk_level`. Markers are counted as whole words, longest alternative
first (`each` does not match `teach`, `after that` counts once).
`estimated_steps = max(enumeration, action verbs, clauses, sequence markers +
1)`, raised to the question count when the task asks more than one question,
and capped at 24.

### 3.3 Heuristic planner (offline, `constraints.deterministic`, or LLM failure)

Rules in priority order (`HeuristicPlanner.plan`):

1. **Investigation** markers -> `recursive` (ADaPT: try the whole task in one
   call, decompose only on failure).
2. **Exploration** markers on a task of 8+ words with `enable_search` ->
   `tree_of_thoughts` (candidate decompositions are scorable).
3. **Damping**: fewer than `decompose_min_steps` steps, under 40 words, no
   documents, fewer than three enumerated items -> `direct`.
4. **Independence** markers or an enumeration of 3+ items that outnumbers the
   sequence markers -> `dag`.
5. Sequence markers, or 4+ estimated steps -> `least_to_most`.
6. Otherwise -> `plan_and_solve`.

Then: `deterministic` downgrades `tree_of_thoughts` -> `dag` and `recursive`
-> `least_to_most`; breadth = `clamp(steps, min_subtasks, max_subtasks)`
(1 for single-call strategies); depth = `max_depth`, raised to
`deep_domain_depth` when `steps >= deep_domain_steps` or the profile has 2+
decomposition hints, and at least the breadth for a chain; parallelism =
`min(max_parallelism, constraints.max_parallelism, breadth)` for `dag`,
`tree_of_thoughts` and `recursive`, else 1; verification follows the profile
risk (`safety_critical` -> `strict`, `high` -> `external`, `medium` ->
`criteria`, `low` -> `criteria` when decomposing, `none` for `direct`);
`replan_on_failure` is true unless `direct` or `max_replans == 0`.
`request.metadata["strategy"]` forces a strategy.

### 3.4 LLM planner (online)

One structured call with `PLAN_TEMPLATE` (effort `planner_effort`): the task,
the rendered features and the configured bounds go in the user turn; the
stable system block holds the role, seven numbered rules (prefer the simplest
structure, decomposition pays off from three steps, parallel only when
genuinely independent, recursive when granularity is unknown, search only when
candidates are scorable, verification by risk, stay within bounds), the
strategy catalogue and two few-shot examples. Output is validated with
`DecompositionPlan.model_validate`; any `StructuredOutputError`/`LLMError`/
`BudgetExceededError` falls back to the heuristic planner with a warning.

### 3.5 Normalisation (`normalize_plan`)

Both planners' output is clamped to the configured and requested bounds:
breadth <= `max_subtasks`; depth <= `max(max_depth, deep_domain_depth)`;
parallelism <= `min(max_parallelism, constraints.max_parallelism)`;
`tree_of_thoughts` -> `dag` when search is disabled or the run is
deterministic; `recursive` -> `least_to_most` when deterministic; single-call
strategies get breadth/depth/parallelism 1; verification is **raised** to the
profile's risk floor and never lowered; re-planning is disabled when
`max_replans == 0` or the strategy is `direct`.

### 3.6 Decomposition, execution and status rules

- **Graph building** (`Strategy.build`): online, one `DECOMPOSE_TEMPLATE` call
  returns a `GraphDraft`; `$id.output` placeholders become edges
  (`infer_dependencies`); the graph is validated against the plan bounds and
  `max_nodes`; defects are fed back for at most `plan_repair_attempts` repair
  calls, then the heuristic graph is used. Offline or under `deterministic`,
  `heuristic_graph` splits the task into segments (enumerated lines, or
  sentences split at `;`, `, and`, `then`, `and <verb>`), folds scope clauses
  (`for each of the logs: ...`) into the work items, makes sequence-marked and
  summarising segments depend on everything before them (the PS+ padding node
  included: the "last segment aggregates" rule is applied to the segment index,
  not the node id), pads short tasks with a PS+ "extract the relevant
  variables" node and the profile's `decomposition_hints`, and appends a
  terminal `verify` node for `high` and `safety_critical` profiles.
  `LeastToMost` then chains the nodes (`chainify`): the author's order is kept
  whenever the task states one (sequence markers or an enumerated list, forced
  `--strategy least_to_most` included); only an unordered task is re-sorted
  easiest-first (low complexity, then shorter description). A pre-approved
  graph in `request.metadata["graph"]` (see `run_graph`) bypasses planning and
  is only validated.
- **Scheduling**: nodes run in topological waves; a wave is dispatched through
  the `Executor` with `plan.parallelism`, but never more nodes than the
  remaining call budget (one call reserved per node); dependency outputs are
  substituted into `inputs`/`description` (truncated to
  `max_subtask_output_chars`); a failed node is retried up to `max_retries`
  with jittered exponential backoff unless the failure is a budget stop, a
  refusal or a plan defect; dependents of a failed node are `SKIPPED` with
  `dependency failed: <ids>`; when the deadline, the call cap or the cost cap
  (`Constraints.max_latency_s` / `max_llm_calls` / `max_cost_usd`, one
  `stop_reason`) is exhausted the remaining nodes are `SKIPPED` with
  `not started: budget|deadline`, and their dependents inherit `not started:
  <reason> (upstream <ids>)` so a pure budget stop is never reported as a
  dependency failure. Every gate (planner, decomposer, solver, synthesis,
  verifier) checks the same `stop_reason` *before* calling, so a cost-capped
  run issues no paid call once the cap is reached.
- **Solving**: `LLMSubTaskSolver` makes one `SOLVE_TEMPLATE` call per node with
  only what the node needs (description, inputs, dependency outputs, evidence),
  or one `PLAN_AND_SOLVE_TEMPLATE` call for `plan_and_solve`; with a
  `Retriever` it retrieves `retrieve_k` documents per sub-question; under
  `strict` verification a `Verifier` verdict can fail the node. A
  `ProtocolSolver` wraps any `SubTaskSolver` (typically another component's
  `run`), passing dependency outputs as `Evidence` and charging nested usage;
  when the nested run (or a verifier's `verdict.details["usage"]`) overshoots
  the budget, the completed result is **kept**, the overshoot is charged and a
  warning recorded, and the scheduler stops further dispatch.
- **Verification nodes**: a node whose `tool_hint` is `verify` (or
  `verification`) *and* that has dependencies, such as the terminal `verify`
  node of high-risk profiles, checks other nodes' results; it is not content.
  When a `Verifier` is injected, `VerifierNodeSolver` takes these nodes away
  from the solver (a RAG solver has nothing to retrieve for "verify the
  combined results" and would answer "insufficient evidence"): the verifier
  gets the combined outputs of the completed dependencies as the answer, the
  root task as the task, and the evidence those dependencies gathered (plus
  the request context and `inputs.documents`, never the `result:<id>`
  outputs themselves) as evidence. The node's result carries the verdict:
  `SUCCESS` with `confidence = verdict.score` when it passed; `NEEDS_REVIEW`
  with a `verifier: ...` error when it failed (not `FAILED`, because retrying
  or re-planning the check cannot repair upstream content); `NEEDS_REVIEW`
  with a warning when the verifier raised. `verdict.details["usage"]` is
  charged. The verdict is stored in `result.metadata["verdict"]`,
  `DecompositionOutput.node_verdicts` and `TaskResult.metadata["node_verdicts"]`;
  it is never composed into the answer. A dependency-free node that merely
  starts with "verify" or "check" ("verify the motor rating against the
  measured shaft power") is content work and stays with the solver. Without a
  `Verifier` the solver answers every node, as before.
  `recursive` runs the whole task as one node first and expands a failed or
  low-confidence node (`< recursive_expand_below`) into a sub-graph down to
  `plan.max_depth`.
- **Re-planning**: after each scheduling pass, failures whose error does not
  start with `budget`, `not started` or a refusal are replannable. The
  `Replanner` writes a `Reflection` (root cause, change, affected ids) into a
  FIFO `ReflectionMemory` and proposes replacement sub-tasks for the failed
  region only (`REPLAN_TEMPLATE` online; offline each failed node is split
  once into two shallower halves, then the replanner gives up). Completed
  nodes keep their outputs; dependents are re-pointed to the last replacement.
  At most `max_replans` rounds.
- **Synthesis**: with more than one node and budget left, one
  `SYNTHESIZE_TEMPLATE` call re-asks the **original task** with every sub-answer
  in context and returns the answer, per-sub-task traceability and unresolved
  items; its confidence is bounded by the structural aggregate. With
  `always_append_original_task=False` that call is skipped when every node
  completed (the LLM still composes a partial result, because gaps need an
  explanation). Otherwise the heuristic composer emits ordered sections per
  node. Verification-node results are skipped by both paths: their text is
  not composed, they are not in the traceability, the structural confidence or
  the coverage, and a check that did not pass is listed under `unresolved`.
  `aggregate_confidence = min(mean, weakest + 0.2) x (1 - composition_discount
  if >1 part) x coverage`.
- **Final verification**: the injected `Verifier` checks
  `claims_text(answer)` (the composed answer without its `#`/`##` heading
  lines, which restate the task and sub-task instructions and would otherwise
  be graded as unsupported claims) against the request context plus the
  evidence every sub-task collected (including the children of recursive
  expansions) and the outputs of the content sub-tasks.
- **Status**: `confidence` starts from the synthesis, is capped at
  `offline_confidence_cap` on a `MockLLM`, bounded by each verification-node
  verdict score and by the final verifier score when a `Verifier` ran, and
  capped at `unverified_confidence_cap` when the plan asks
  for `external`/`strict` verification but no verifier is injected. The
  threshold is `max(profile.min_confidence_to_accept,
  constraints.min_confidence)`. Node counts (`n/m sub-tasks completed`) cover
  content nodes only. `FAILED` when no content node completed;
  `NEEDS_REVIEW` when the final verifier rejected the answer, a verification
  node did not pass (or its verifier raised), or confidence is below
  the threshold; `PARTIAL` when some nodes did not complete; else `SUCCESS`.
  `confidence_basis` spells the chain out in words.

## 4. Quick start

### Python

```python
from pipeline_core import Constraints, TaskRequest, make_llm
from task_decomposition import TaskDecompositionModel

model = TaskDecompositionModel(make_llm())       # MockLLM without credentials, Claude with ANTHROPIC_API_KEY

task = "For each of the three pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status."
result = model.run_sync(task, domain="mechanical_engineering")
print(result.summary())                          # [task_decomposition] status=... confidence=... calls=...
print(result.confidence_basis)
out = result.output                              # DecompositionOutput
print(out.plan.describe())                       # strategy=dag steps~4 ... verification=external
print(out.graph.to_ascii())                      # waves with [id] (tool_hint, complexity) description <- deps
print(out.render())                              # plan, graph, per-node status, unresolved items, metrics
print(result.output_text)                        # the composed answer

# Plan and decompose only (human checkpoint): inspect or edit the graph, then execute it
graph = model.decompose_sync("Debug why the nightly ETL job fails intermittently and propose a fix with a regression test.", domain="data_engineering")
print(graph.metadata["plan"]["strategy"], graph.to_mermaid())
edited = graph.with_tasks([t for t in graph.tasks if t.id != "verify"])
approved = model.run_graph_sync(edited, TaskRequest(task=graph.root_task, domain="data_engineering"))

# Plan only, and step-wise plan metrics without any model call
plan = model.plan_for_sync(task, domain="civil_structural_engineering")   # verification == "strict"
report = model.evaluate_plan(graph)                                        # coverage, dependency_correctness, parallel_speedup, valid

# Budgets degrade the run instead of failing it
budgeted = model.run_sync(TaskRequest(task=task, constraints=Constraints(max_llm_calls=2, max_latency_s=30)))
print(budgeted.status.value, budgeted.metadata["stopped_reason"])         # partial/needs_review, budget
```

All `*_sync` methods have `async` counterparts (`run`, `decompose`,
`plan_for`, `run_graph`). `TaskDecompositionModel(llm, config={...},
**overrides)` accepts a `TaskDecompositionConfig`, a dict, or keyword
overrides of individual fields.

### CLI

```bash
# from the repository root; use ";" instead of ":" in PYTHONPATH on Windows
PYTHONPATH="common:task_decomposition_model" python -m task_decomposition --offline \
    "For each of the three pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status." \
    --domain mechanical_engineering

python -m task_decomposition "..." --json --input documents="Log A: 118 m3/h at 31 m" --input k=3 --max-calls 12
python -m task_decomposition "..." --plan-only --strategy least_to_most      # graph as ASCII waves + Mermaid, no execution
python -m task_decomposition "..." --deterministic --max-parallelism 2 --trace
```

Flags: `--domain` (a registered `DomainProfile`), `--offline` (force the
`MockLLM`), `--json`, `--input KEY=VALUE` (repeatable; JSON values are parsed;
`documents=` adds a document), `--strategy`, `--max-calls`,
`--max-parallelism`, `--deterministic`, `--plan-only`, `--trace`. Exit status
is 0 for `success`/`partial`/`cached`/`needs_review` and 1 otherwise.

## 5. Configuration (`TaskDecompositionConfig`)

| Field | Default | Why (source) |
| --- | --- | --- |
| `decompose_min_steps` | `3` | Decomposition is roughly neutral at two steps and pays off at five or more; trivial tasks are answered directly: Zhou et al. (2022), *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models*, https://arxiv.org/abs/2205.10625; damping against over-decomposition: Schluntz & Sutton (2024), *Building effective agents*, https://www.anthropic.com/research/building-effective-agents |
| `min_subtasks` | `3` | ADaPT plans 3-5 sub-tasks per expansion: Prasad et al. (2023), *ADaPT: As-Needed Decomposition and Planning with Language Models*, https://arxiv.org/abs/2311.05772 |
| `max_subtasks` | `8` | Upper bound per level; keeps every step shallow because accuracy decays with computation-graph depth and width: Dziri et al. (2023), *Faith and Fate: Limits of Transformers on Compositionality*, https://arxiv.org/abs/2305.18654 |
| `max_nodes` | `25` | Per-plan node cap; Task Limit Exceeded dominates agent failures: Liu et al. (2023), *AgentBench: Evaluating LLMs as Agents*, https://arxiv.org/abs/2308.03688 |
| `max_depth` | `3` | Recursion bound d_max = 3: Prasad et al. (2023), *ADaPT: As-Needed Decomposition and Planning with Language Models* |
| `deep_domain_depth` | `4` | Depth bound for highly compositional tasks (ADaPT reports gains up to d_max = 4 on deeper problems): Prasad et al. (2023); bounded because the recursion worst case is exponential in depth |
| `deep_domain_steps` | `6` | Step estimate from which the deeper bound applies; decomposition gains are clear from five steps: Zhou et al. (2022) |
| `max_parallelism` | `8` | Concurrent frontier nodes; LLMCompiler's parallel frontier gives 1.8-3.7x latency and 3.4-6.7x cost gains over sequential ReAct, bounded by tool rate limits: Kim et al. (2023), *An LLM Compiler for Parallel Function Calling*, https://arxiv.org/abs/2312.04511 |
| `max_retries_per_task` | `1` | One retry per node, then the failure is handled by reflection and re-planning rather than by repeating the same attempt: Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning*, https://arxiv.org/abs/2303.11366; a retry cap also keeps the node count under the AgentBench task limit: Liu et al. (2023) |
| `retry_base_delay_s` | `0.5` | Exponential backoff base with full jitter: the shared `RetryPolicy` contract in `ARCHITECTURE.md` (retry only retryable error classes, never budget stops or refusals) |
| `retry_factor` | `2.0` | Backoff factor: `ARCHITECTURE.md` `RetryPolicy` |
| `retry_max_delay_s` | `8.0` | Backoff ceiling per attempt: `ARCHITECTURE.md` `RetryPolicy` |
| `max_replans` | `2` | Bounded joiner/replanner iterations; LLMCompiler re-plans a bounded number of times, and verifier back-prompting converges in about 3-4 rounds when it converges at all: Kim et al. (2023); Valmeekam et al. (2023), *On the Planning Abilities of Large Language Models: A Critical Investigation*, https://arxiv.org/abs/2305.15771 |
| `reflection_memory` | `3` | Reflexion's episodic memory is capped at 1-3 reflections: Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning*, https://arxiv.org/abs/2303.11366 |
| `plan_repair_attempts` | `1` | One repair pass for an invalid plan, then the heuristic fallback; LLM plans fail format/structure checks often (planning F1 ~50%): Shen et al. (2023), *HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face*, https://arxiv.org/abs/2303.17580 |
| `recursive_expand_below` | `0.5` | Solver confidence below which a recursive node is expanded; self-reported confidence is a weak prior, so the threshold is permissive and an injected `Verifier` is the real signal: Huang et al. (2023), *Large Language Models Cannot Self-Correct Reasoning Yet*, https://arxiv.org/abs/2310.01798 |
| `tot_breadth` | `5` | Tree-of-Thoughts BFS breadth b = 5: Yao et al. (2023), *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, https://arxiv.org/abs/2305.10601 |
| `tot_depth` | `2` | Search depth 2 (ToT's generation tasks use a two-level plan-then-write search): Yao et al. (2023) |
| `tot_votes` | `3` | Evaluator samples per state, averaged (ToT uses v = 3 votes): Yao et al. (2023) |
| `enable_search` | `True` | Search is allowed but gated on exploration markers because it costs 5-10x a single call: Yao et al. (2023); add complexity only where it demonstrably pays: Schluntz & Sutton (2024), *Building effective agents* |
| `composition_discount` | `0.10` | Confidence discount when composing several sub-answers; the compositionality gap (~40%) means correct parts do not guarantee a correct whole: Press et al. (2022), *Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)*, https://arxiv.org/abs/2210.03350 |
| `offline_confidence_cap` | `0.7` | `MockLLM` answers carry no epistemic signal, so structural confidence cannot clear a high-risk threshold on its own: Huang et al. (2023) |
| `unverified_confidence_cap` | `0.7` | Without the external signal the plan asked for, the run cannot auto-accept; correctness belongs to verifiers, not the planner: Kambhampati et al. (2024), *LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks*, https://arxiv.org/abs/2402.01817 |
| `agreement_threshold` | `0.6` | Averaged candidate vote below which a Tree-of-Thoughts selection is flagged `low_agreement`; agreement ratio as a confidence signal: Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, https://arxiv.org/abs/2203.11171 |
| `planner_effort` | `high` | Planning and decomposition are the calls whose errors cascade: `ARCHITECTURE.md` 3.2 (`high` for planning and synthesis); strongest model/effort for the planner: Xu et al. (2023), *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*, https://arxiv.org/abs/2305.18323 |
| `solver_effort` | `medium` | Leaf sub-tasks are shallow by construction; ReWOO's workers run on a smaller model/effort than the planner: Xu et al. (2023); `ARCHITECTURE.md` 3.2 effort routing |
| `synthesis_effort` | `high` | Composition is where the compositionality gap bites: Press et al. (2022) |
| `safety_critical_verification_effort` | `max` | Used only for the terminal `verify` node in `safety_critical` profiles (`ARCHITECTURE.md` 3.2: `max` only for final verification in safety-critical domains) |
| `solver_model` | `None` | Optional cheaper model for leaf nodes (ReWOO shows a small planner/worker suffices for many leaves): Xu et al. (2023). `None` keeps one model so prompt caches stay model-scoped |
| `max_subtask_output_chars` | `4000` | Dependency outputs passed to a node are truncated; compact per-node context: Xu et al. (2023) |
| `max_context_docs` | `12` | Evidence rendered per node; each worker sees only the observations its sub-task needs (ReWOO decouples reasoning from observations): Xu et al. (2023); sub-agents must compress and bound their context or run out of window: Hadfield et al. (2025), *How we built our multi-agent research system*, https://www.anthropic.com/engineering/multi-agent-research-system |
| `retrieve_k` | `5` | Documents retrieved per sub-question when a `Retriever` is injected (Self-Ask + search; retrieval per hop): Press et al. (2022); Trivedi et al. (2023), *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)*, https://arxiv.org/abs/2212.10509 |
| `always_append_original_task` | `True` | Synthesis re-asks the original task with the sub-answers in context even when every node completed (the last Least-to-Most sub-question is the original question): Zhou et al. (2022). `False` skips that `SYNTHESIZE` call on a fully completed graph and uses the heuristic sections (one call cheaper); partial graphs are still composed by the LLM |
| `force_llm_path` | `False` | Testing knob: drive the LLM-planned path with `ScriptedLLM`/`MockLLM` responders. Heuristics remain the fallback |
| `loop_detection` | `True` | Flag duplicate sub-tasks (same description and inputs) as a plan defect; step repetition is a leading failure mode: Cemri et al. (2025), *Why Do Multi-Agent LLM Systems Fail? (MAST)*, https://arxiv.org/abs/2503.13657; Liu et al. (2023) |

## 6. Techniques implemented

| Technique | Where | Sources |
| --- | --- | --- |
| Damping rule: trivial tasks are one call | `HeuristicPlanner` rule 3, `DirectStrategy`, `PLAN_TEMPLATE` rule 1 | Zhou et al. (2022), *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models*; Schluntz & Sutton (2024), *Building effective agents* |
| Zero-shot Plan-and-Solve (PS+) | `PlanAndSolve`, `PLAN_AND_SOLVE_TEMPLATE` with the PS+ trigger ("extract relevant variables ... make a plan ... carry out the plan"), the PS+ variable-extraction padding node in `heuristic_graph` | Wang et al. (2023), *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models*, https://arxiv.org/abs/2305.04091 |
| Least-to-Most sequential chain with answer carry-forward and the original task re-asked last | `LeastToMost` (author's order when stated, easiest-first otherwise; `chainify`), `SOLVE_TEMPLATE` `<prior_results>`, `Synthesizer` (`always_append_original_task`) | Zhou et al. (2022); Khot et al. (2022), *Decomposed Prompting: A Modular Approach for Solving Complex Tasks*, https://arxiv.org/abs/2210.02406 |
| Self-Ask sub-questions with per-sub-question retrieval and the compositionality gap as a metric and a confidence discount | `LLMSubTaskSolver.gather_evidence`, `composition_discount`, `compositionality_gap` | Press et al. (2022), *Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)*; Trivedi et al. (2023), *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)* |
| DAG planning with `$id.output` placeholders and topological frontier scheduling by pure code | `DAGPlanner`, `DECOMPOSE_TEMPLATE` rule 3, `TaskGraph.infer_dependencies`, `Scheduler` | Kim et al. (2023), *An LLM Compiler for Parallel Function Calling*; Xu et al. (2023), *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*; Shen et al. (2023), *HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face* |
| Machine-validated plans with one bounded repair pass, then a fallback | `TaskGraph.validate_structure`, `Strategy.llm_decompose` (`<previous_attempt_issues>` feedback), `plan_repair_attempts` | Shen et al. (2023); Valmeekam et al. (2023), *On the Planning Abilities of Large Language Models: A Critical Investigation*; Kambhampati et al. (2024), *LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks* |
| As-needed recursive decomposition with a depth bound | `RecursiveDecomposer` (whole task first; `expand` on externally observed failure or low confidence; `plan.max_depth`) | Prasad et al. (2023), *ADaPT: As-Needed Decomposition and Planning with Language Models*; re-planning conditioned on results: Wang et al. (2024), *TDAG: A Multi-Agent Framework based on Dynamic Task Decomposition and Agent Generation*, https://arxiv.org/abs/2402.10178 |
| Tree-of-Thoughts: bounded BFS over candidate decompositions with LLM value voting under varied framings, refinement of the best candidates, structural scoring offline | `TreeOfThoughtsPlanner` (`CANDIDATES_TEMPLATE`, `EVALUATE_TEMPLATE`, `REFINE_TEMPLATE`, `aggregate_votes`, `structural_score`) | Yao et al. (2023), *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*; diversity by prompt framing rather than temperature and agreement as confidence: Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models* |
| Reflexion-style re-planning gated on external signals with bounded episodic memory | `Replanner`, `ReflectionMemory`, `replannable_failures`, `REPLAN_TEMPLATE` | Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning*; gating: Huang et al. (2023), *Large Language Models Cannot Self-Correct Reasoning Yet* |
| Verification as the correctness owner: checkable criteria on every node, external `Verifier` per node (`strict`) and on the composed answer, risk-based verification floor | `DECOMPOSE_TEMPLATE` rule 4, `LLMSubTaskSolver.verify_node`, `VerifierNodeSolver` (verification nodes answered by the external verifier, never by the solver or self-judgement), `execute()` step 5, `normalize_plan` | Kambhampati et al. (2024); Valmeekam et al. (2023); Huang et al. (2023) |
| Hard budgets and graceful degradation: node/depth/replan/retry/memory caps, call *and cost* reservation before fan-out (`stop_reason`), deadline checks, duplicate-node loop detection, nested overshoot charged and warned about rather than failing the node | `TaskDecompositionConfig`, `Scheduler.run`, `stop_reason`/`can_call`, `check_budget_before`, `charge_nested_usage`, `loop_detection` | Liu et al. (2023), *AgentBench: Evaluating LLMs as Agents*; Kim et al. (2023); Cemri et al. (2025), *Why Do Multi-Agent LLM Systems Fail? (MAST)* |
| Jittered exponential backoff, retry only transient failures, never budget stops or refusals; after the retry cap the replanner reflects instead of repeating | `RetryPolicy`, `replannable_failures` | `ARCHITECTURE.md` `RetryPolicy` contract; Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning* |
| Compact per-node context and evidence caps | `render_prior`, `max_subtask_output_chars`, `max_context_docs` | Xu et al. (2023), *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*; Hadfield et al. (2025), *How we built our multi-agent research system* |
| Effort routing by role: high for planning/synthesis, medium for leaves, max only for final safety-critical verification; optional cheaper leaf model | `planner_effort`, `solver_effort`, `synthesis_effort`, `safety_critical_verification_effort`, `solver_model` | Xu et al. (2023); `ARCHITECTURE.md` 3.2 (effort per call, one model for the main loop so caches stay model-scoped) |
| Cache-aware, versioned prompts: role, numbered rules, XML-tagged inputs, explicit output schema, examples in the stable prefix, nothing volatile in system blocks; flat enum-typed schemas | `prompts.py`, `prompting.build_request`, `schemas.py` | Anthropic (2026), *Claude prompting best practices (chain complex prompts, subagent orchestration, parallel tool calling)*, https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices; Schluntz & Sutton (2024), *Building effective agents* |
| Step-wise plan evaluation | `evaluation.py` (section 9) | Huang et al. (2024), *Understanding the planning of LLM agents: A survey*, https://arxiv.org/abs/2402.02716 |

## 7. Adapting to an engineering domain

Nothing in the package names a domain. Behaviour changes through the
`DomainProfile` named by `TaskRequest.domain`:

- `decomposition_hints` are rendered into `DECOMPOSE_TEMPLATE`/
  `CANDIDATES_TEMPLATE`/`REFINE_TEMPLATE` as `<domain_hints>` and pad short
  heuristic graphs; two or more hints raise the depth bound to
  `deep_domain_depth`.
- `risk_level` sets the verification floor (`strict`/`external`/`criteria`),
  adds a terminal `verify` node for `high` and `safety_critical` profiles,
  adds `ENGINEERING_RIGOR` to every stable system block for those levels, and
  gives the terminal `verify` node `safety_critical_verification_effort` when
  the LLM solver answers it (with an injected `Verifier` the verifier owns it).
- `min_confidence_to_accept` is the accept threshold (`NEEDS_REVIEW` below it).
- `units` add a "every quantity carries a unit" criterion to every node;
  `verification_rules` become the `verify` node's criteria and the first rule
  is added to every node in high-risk profiles; `glossary`, `standards`,
  `style_guidance` enter the prompts through `profile.system_text()`.

```python
from pipeline_core import DomainProfile, TaskRequest, register_profile
from task_decomposition import TaskDecompositionModel

register_profile(DomainProfile(
    name="aerospace_structures",
    description="Airframe structural analysis: loads, stress, fatigue and damage tolerance.",
    glossary={"MS": "Margin of safety = allowable / applied - 1", "DUL": "Design ultimate load (1.5 x DLL)"},
    units=["kN", "MPa", "mm", "N/mm"],
    document_types=["loads report", "stress report", "material allowables", "drawing", "test report"],
    verification_rules=[
        "Every margin of safety must be reported against the ultimate allowable with the load case identified.",
        "Material allowables must cite the source table and the temperature/basis (A- or B-basis).",
        "Units must be SI and dimensionally consistent.",
    ],
    risk_level="safety_critical",
    min_confidence_to_accept=0.85,
    decomposition_hints=[
        "Separate load derivation from stress analysis",
        "Check every fastener failure mode (bearing, shear-out, net section) individually",
        "Attach a verification method to each margin",
    ],
    standards=["FAR 25.305", "FAR 25.571", "MMPDS"],
))

model = TaskDecompositionModel()
result = model.run_sync(TaskRequest(
    task="Assess the wing-to-fuselage attachment fitting: identify the limit and ultimate loads, compute the bearing and shear-out stresses, check the margins of safety, and list the assumptions to confirm by test.",
    domain="aerospace_structures",
))
print(result.output.plan.verification)          # strict
print(result.output.graph.ids())                # [..., 'verify']
print(result.status.value)                      # needs_review until a Verifier is injected (no external signal)
```

`examples/engineering_example.py` runs this profile against the `general`
profile, then injects a `Retriever`, a rule-based `Verifier` and a RAG-style
`SubTaskSolver`, and finishes with the evaluation harness.

## 8. Composing with the other five models

| Model | Direction | Interface |
| --- | --- | --- |
| Pipeline RAG | accepts | `TaskDecompositionModel(solver=rag.run)`: any `SubTaskSolver` (`TaskRequest -> TaskResult`) solves the leaves. Each child request carries `task` = the node description, `inputs` = request inputs plus node inputs, `context` = request evidence plus dependency outputs as `Evidence(id="result:<id>")`, and `metadata[subtask_id, root_task, expected_output, verification_criteria, tool_hint]`; the child's `usage` is charged to the parent budget and its `evidence` surfaces in the parent result. With a `verifier` injected as well, verification nodes never reach the solver (section 3.6), so `solver=rag.solver` needs no wrapper such as the `RAGSubTaskSolver` workaround |
| Hybrid Retrieval | accepts | `TaskDecompositionModel(retriever=...)`: `retrieve(query=node.description, k=retrieve_k)` per sub-question for `retrieve`/`llm`/`aggregate` nodes when `constraints.allow_external_tools`; hits are rendered as `<documents>` and cited by id |
| Answer Verification | accepts | `TaskDecompositionModel(verifier=...)`: `verify(answer, task=..., evidence=...)` (1) on every verification node (`tool_hint="verify"` with dependencies): answer = combined upstream outputs, task = root task, evidence = upstream sub-task evidence; the verdict becomes the node's status/confidence and is recorded, not composed; (2) on every content node under `strict` (a failed verdict fails the node: retry, then re-plan); (3) on `claims_text` of the composed answer with all sub-task evidence whenever `plan.verification != "none"` (a failed verdict sets `NEEDS_REVIEW`). `verdict.details["usage"]` (the verification run's `Usage`) is charged each time |
| Parallelization | accepts | `TaskDecompositionModel(executor=...)`: waves are dispatched through `Executor.map(fn, items, concurrency=plan.parallelism)`; the default is `BoundedExecutor` (semaphore-limited `asyncio.gather`) |
| Caching | accepts | `TaskDecompositionModel(CachingModel.wrap(llm))`: every prompt keeps stable blocks first (template system, examples, profile, rigor) and nothing volatile in them, so planner/decomposer/solver prefixes are cacheable; `TaskGraph` JSON keyed by normalised task + domain is a natural plan cache |
| All | exposes | `run(request)` (orchestrator for complex tasks), `decompose(task) -> TaskGraph` (a reviewable plan other components or a human can edit), `run_graph(graph, request)` (execute an approved graph), `plan_for(task) -> DecompositionPlan` (complexity routing), `evaluate_plan(graph)`; `result.as_evidence()` for downstream components |

A single `Tracer` passed to every component yields one tree
(`tracer.tree()`), and nested usage rolls up into the root `usage.cost_usd`.

## 9. Evaluation

Step-wise metrics (`evaluation.py`):

| Metric | Definition |
| --- | --- |
| `coverage` | Fraction of the task's content words that appear in some sub-task description or expected output (keyword judge); `llm_coverage` lists the task's atomic requirements with one low-effort structured call and reports the fraction covered by an existing sub-task, falling back to the keyword judge when the model returns nothing or the call fails. `judge_coverage` returns `(coverage, usage)`; given a `RunContext` the judge call goes through `ctx.llm_call` (traced, budgeted, charged), and the harness adds the judge's usage to each case's `llm_calls`/`cost_usd` (`judge_calls`) so `total_cost_usd` is the true cost of a judged run |
| `dependency_correctness` | 1.0 when every placeholder is backed by an edge, no edge dangles and the graph is acyclic; each defect costs one point over edges + placeholders; a cycle scores 0 |
| `parallel_speedup` | Nodes / critical-path length (the latency ratio of sequential vs wave execution) |
| `valid`, `issues` | `validate_structure` defects before any repair (plan validity rate) |
| `with_criteria` | Share of nodes with at least one verification criterion |
| `estimated_llm_calls` | Nodes + 1 synthesis call |
| `compositionality_gap` | Among cases whose sub-tasks all succeeded, the fraction whose final answer failed the keyword judge |
| `failure_classes` | AgentBench-style buckets: `budget_or_deadline`, `skipped_dependency`, `verification_failed`, `invalid_format`, `refusal`, `solver_failed` |
| cost | LLM calls, tokens, USD and latency per task from `result.usage` |

`EvaluationSummary` aggregates success/needs-review rates, keyword accuracy,
plan validity, strategy accuracy against `EvalCase.expected_strategy`, mean
coverage and speedup, the compositionality gap, calls and cost per task, the
strategy mix and the failure-class distribution.

```bash
# built-in seven-case engineering suite, offline, keyword judge
PYTHONPATH="common:task_decomposition_model" python -m task_decomposition.eval_cli
python -m task_decomposition.eval_cli --json
python -m task_decomposition.eval_cli --judge llm            # one extra low-effort judge call per case, counted in cost
python -m task_decomposition.eval_cli --cases my_cases.json  # [{"task": ..., "domain": ..., "expected_keywords": [...], "expected_strategy": ...}]
python -m task_decomposition.eval_cli --online               # make_llm(): Claude when credentials exist
```

The entry point is `eval_cli.py`, a module the package `__init__` does not
import, so `runpy` executes it once and without Python's double-import
`RuntimeWarning` (`evaluation.main` remains as a thin alias).

```python
from pipeline_core import MockLLM
from task_decomposition import EvalCase, TaskDecompositionModel, evaluate_sync, sample_cases

summary = evaluate_sync(TaskDecompositionModel(MockLLM()), sample_cases(), judge="keyword")
print(summary.table())
```

Offline, the sample suite reports plan validity 1.00, strategy accuracy 1.00
(each strategy family is selected by the heuristic planner for its case),
mean coverage about 0.97 and 2.9 calls per task; `needs_review` outcomes are
the high-risk domains refusing to auto-accept without a verifier.

## 10. Limitations and failure modes

- **Frozen plans cascade.** A `dag` or `least_to_most` plan is fixed before any
  result is seen; an early wrong node propagates (Huang et al. 2024; TDAG
  measured 34.78% cascading errors for static plan-and-execute). Mitigations
  here are re-planning after a pass, dependents skipped on failure and
  `recursive` for investigative tasks; there is no per-node re-planning after
  every result.
- **Self-reported failure is a weak signal.** Without an injected `Verifier`,
  node failure comes from the solver's own `status`/confidence, which Huang et
  al. (2023) show is unreliable; the model therefore caps confidence
  (`unverified_confidence_cap`) and returns `NEEDS_REVIEW` in high-risk
  domains rather than auto-accepting.
- **Planner validity.** LLM plans are invalid often enough (Shen et al. 2023;
  Valmeekam et al. 2023) that one repair pass then the heuristic graph is the
  policy; the heuristic graph is structurally sound but splits on surface
  syntax and can misplace a dependency the text leaves implicit.
- **Compositionality gap.** Correct sub-answers still compose wrongly about
  40% of the time at every scale (Press et al. 2022); the synthesis re-asks
  the original task and confidence is discounted, but the gap is tracked, not
  closed.
- **Depth decay.** Accuracy falls with computation-graph depth (Dziri et al.
  2023); nodes are kept shallow and recursion is bounded, but exact arithmetic
  is still done in natural language. A sandboxed program executor (Gao et al.
  2023, *PAL: Program-aided Language Models*, https://arxiv.org/abs/2211.10435)
  is the natural `compute` handler and is not included.
- **Search cost.** `tree_of_thoughts` costs 1 + votes (+ 2 refinements + votes
  per extra level) calls before any node is solved; it is gated on exploration
  markers and degrades to heuristic candidates under budget, but on small
  tasks it rarely beats `dag` (Yao et al. 2023; Schluntz & Sutton 2024).
- **No interleaved leaf executor.** Leaves are single calls, not ReAct loops
  (Yao et al. 2022, *ReAct: Synergizing Reasoning and Acting in Language
  Models*, https://arxiv.org/abs/2210.03629); observation-dependent sub-tasks
  must be delegated to a `SubTaskSolver` that implements them.
- **Budget reservation assumes one call per node.** The scheduler reserves one
  LLM call per dispatched node and checks calls, cost and deadline before each
  nested call; a nested `SubTaskSolver` that spends several calls can still
  overshoot `max_llm_calls`/`max_cost_usd` by the width of one batch. The
  overshoot is charged to the parent budget and recorded as a warning
  (`budget exceeded after nested sub-task ...`), the nested result is kept,
  and no further node is dispatched (`stopped_reason="budget"`).
- **Offline answers are stubs.** On a `MockLLM` the plan, graph, scheduling,
  status and metrics are real; sub-task answers are schema-conformant
  placeholders and confidence is structural, capped at
  `offline_confidence_cap`.
- **Security.** Sub-task prompts render retrieved documents and dependency
  outputs verbatim; there is no provenance tagging or control-flow integrity
  for untrusted text (Beurer-Kellner et al. 2025, *Design Patterns for
  Securing LLM Agents against Prompt Injections*,
  https://arxiv.org/abs/2506.08837). Use `run_graph` as the human checkpoint
  before executing plans that touch systems of record.

## 11. How it was built

1. **Research** (`docs/research/task-decomposition.json`; the prompting and
   pipeline-architecture dossiers reach this package only through
   `ARCHITECTURE.md` and `pipeline_core`, and every source cited in this README
   is in the model's own dossier): the
   decomposition literature fixed the strategy catalogue and its selector
   (Least-to-Most / Plan-and-Solve chains, LLMCompiler/ReWOO DAGs, ADaPT
   recursion, ToT search), the depth gate (gains from 5+ steps), the caps
   (ADaPT d_max 3-4 and 3-5 children, Reflexion memory 1-3, LLMCompiler
   replans 2-3, AgentBench node cap), the rule that planners propose and
   verifiers decide (Valmeekam, Kambhampati, Huang), the compositionality gap
   as a metric, and step-wise evaluation; `ARCHITECTURE.md` fixed the
   cache-aware template structure, flat enum-typed schemas, effort routing,
   context caps, jittered backoff, retry caps, budget degradation and
   OTel-style tracing through `pipeline_core`.
2. **Design decisions**: a declarative `DecompositionPlan` separates reasoning
   from execution so the heuristic and LLM planners share one normaliser and
   one set of bounds; every strategy produces the same validated `TaskGraph`,
   so the scheduler, replanner, synthesiser and metrics are strategy-agnostic;
   the scheduler is pure code and never calls a model; re-planning is gated on
   external signals only; verification level follows domain risk and is never
   lowered; budgets are checked before fan-outs and degrade sample counts and
   dispatch width rather than fail; `decompose`/`run_graph` give humans and
   orchestrators a checkpoint between planning and execution.
3. **Implementation**: pure-function cores (DAG validation, waves, critical
   path, placeholders, backoff timing with a fake clock and injected sleep,
   confidence aggregation, vote aggregation, revision application, feature
   extraction, metrics) are unit-tested without a model; LLM paths are covered
   with `ScriptedLLM`/`MockLLM` responders (planned DAG run, repair-then-
   fallback, ToT propose/vote/refine, replanner, synthesis); `MockLLM` covers
   the offline end-to-end path, domain adaptation, budgets, protocol injection
   and the CLI. `tests/` holds 85 tests that complete in a few seconds; the
   CLI and both examples run with no credentials.
