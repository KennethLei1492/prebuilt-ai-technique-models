# Answer Verification (`answer_verification`)

Decides whether a candidate answer is **correct, grounded and complete**,
estimates a **calibrated confidence**, and **corrects the answer only when an
external signal backs the correction**. Seven verifiers (self-consistency with
Universal Self-Consistency, factored Chain-of-Verification, FActScore-style
claim grounding, G-Eval-style rubric judging with position-swapped pairwise
mode, deterministic domain rules with a unit parser, sandboxed code execution
and bounded multi-agent debate) are **planned per request** from the task text,
the candidate, the available oracles and the engineering domain profile, then
fused into one `Verdict`. It implements the `pipeline_core.protocols.Verifier`
protocol, so Pipeline RAG and Task Decomposition consume it unchanged.

Everything runs offline: without credentials `make_llm()` returns a `MockLLM`,
the `HeuristicPlanner` decides the plan, and the LLM-free verifiers (grounding
in lexical mode, rules, code execution, self-consistency over pre-sampled
candidates) do the work while the LLM-only verifiers report themselves as
**inconclusive** and carry no weight. A verifier whose inputs are missing
(no evidence, no domain units or rules, no tests) is also inconclusive rather
than a vacuous pass, and a run in which **no verifier of record reached a
conclusion is `NEEDS_REVIEW`**, never `SUCCESS`: the offline path verifies what
it can check and says so when it checked nothing. A one-line quantity answer
(`9.6 mm`) is grounded as a single claim, so the most common engineering answer
shape works offline against a passage that states the value.

---

## 1. Purpose

Generated engineering answers fail in specific ways: a wrong number with the
right words, a missing unit, a claim no document supports, a part number that
does not exist, code that looks right and does not run. The literature is
unambiguous that asking the same model to critique itself does not fix this:
intrinsic self-correction lowers accuracy (Huang et al. (2023), *Large Language
Models Cannot Self-Correct Reasoning Yet*, https://arxiv.org/abs/2310.01798),
self-feedback misses errors (Madaan et al. (2023), *Self-Refine: Iterative
Refinement with Self-Feedback*, https://arxiv.org/abs/2303.17651) and critique
without tools gives negligible or negative gains (Gou et al. (2023), *CRITIC:
Large Language Models Can Self-Correct with Tool-Interactive Critiquing*,
https://arxiv.org/abs/2305.11738).

This package therefore treats verification as the collection of **external
signals** ordered by authority: executed tests and deterministic rules are
terminal; evidence grounding and multi-sample agreement are strong; LLM
judgement is advisory; intrinsic self-critique is never a verifier. A revision
is applied only when such a signal proposes it **and** the plan's external
verifiers confirm it (Section 3.5). Every result carries a confidence with a
human-readable basis, compares it with the domain's acceptance threshold, and
abstains (`NEEDS_REVIEW`, `escalated=True`) when the evidence is not there.

## 2. Architecture

```
          TaskRequest(task, inputs{answer, candidates, documents, tests, reference, rules}, context, domain, constraints)
                                                        │
                                                        ▼
 ┌──────────────────────────────  AnswerVerificationModel.plan()  ─────────────────────────────────────┐
 │  _gather_evidence(): request.context + inputs[documents|evidence]  ─ none? ─▶ Retriever.retrieve()  │
 │  _resolve_answer():  inputs["answer"] | "Verify ...: <answer>" tail | DRAFT (online) | task text     │
 │  policy_for_profile(ctx.profile) ──▶ VerificationPolicy (mandatory verifiers, floors, revision cap)   │
 │                                                                                                      │
 │  offline / deterministic / llm_mode=never ──▶ HeuristicPlanner      online ──▶ PLANNER prompt        │
 │     extract_features() -> TaskFeatures                              (1 structured call: LLMPlan,     │
 │     classify_answer_type() -> answer_type                            sanitised + clamped in Python)  │
 │                                   └──────────────┬────────────────────────┘                          │
 │                           apply_policy()  ──▶  fit_to_budget()  ──▶  VerificationPlan                │
 └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                    ▼
 ┌──────────────────────────────  AnswerVerificationModel.execute()  ──────────────────────────────────┐
 │  VerifyInputs.from_plan()   (one resolved bundle shared by every verifier through the RunContext)    │
 │                                                                                                      │
 │  for name in plan.verifiers:   with ctx.step("verify.<name>")            verifiers.py                │
 │   ┌────────────────────┬──────────────────────┬────────────────┬────────────────┬──────────┬───────┐ │
 │   │ self_consistency   │ chain_of_verification│ grounding      │ rubric_judge   │ rules    │ code_ │ │
 │   │ k derivations,     │ plan questions ->    │ claims ->      │ criteria x N   │ units /  │ exec  │ │
 │   │ adaptive stop,     │ answer blind ->      │ supported /    │ framings,      │ regex /  │ (sub- │ │
 │   │ cluster, USC pick  │ cross-check -> revise│ contradicted / │ pairwise swap  │ range /  │ proc) │ │
 │   │ (uncertainty.py)   │ (evidence-gated)     │ unsupported    │                │ sections │       │ │
 │   └────────────────────┴──────────────────────┴────────────────┴────────────────┴──────────┴───────┘ │
 │   debate (A agents x R rounds, judge)              verbalized_confidence (advisory, 1 call)          │
 │                                                    │                                                 │
 │  aggregate_verdicts(verdicts, weights, calibration) ──▶ (confidence, basis)        uncertainty.py    │
 │  _confirm_revision(): proposed correction re-verified by grounding / rules / code_execution          │
 │  gates: mandatory verifiers, no-contradiction policy, safety-critical evidence requirement,          │
 │         accept threshold, escalation threshold, revision -> NEEDS_REVIEW                             │
 └──────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                    ▼
      TaskResult(status, output=VerificationReport, output_text, evidence, confidence, confidence_basis,
                 metadata{plan, policy, answer_source})          report.to_verdict() -> Verdict (protocol)
```

| Module | Contents |
| --- | --- |
| `model.py` | `AnswerVerificationModel` (`spec`, `plan()`, `execute()`, `verify()`, `verify_report()`, `verify_sync()`, `compare()`), `VerificationReport`, `coerce_evidence` |
| `verifiers.py` | `BaseVerifier`, `SelfConsistencyVerifier`, `ChainOfVerificationVerifier`, `GroundingVerifier`, `RubricJudgeVerifier` (+ `compare`, `pairwise_decision`), `RuleVerifier` (+ `run_rule_checks`, `RuleCheck`), `CodeExecutionVerifier`, `DebateVerifier`, `VerifyInputs`, `inconclusive`, `build_request`, `affordable`, `make_verifiers`, `VERIFIER_CLASSES` |
| `uncertainty.py` | `canonicalize_answer`, `extract_number` (Section 3.7), `jaccard`, `similarity_ratio`, `semantic_agreement` / `AgreementResult`, `majority_stability`, `should_stop_sampling`, `VerbalizedConfidence`, `verbalized_confidence`, `probability_for_candidate`, `platt_scale`, `fit_platt`, `AUTHORITY`, `DEFAULT_WEIGHTS`, `is_inconclusive`, `aggregate_verdicts` |
| `claims.py` | `Claim`, `ClaimVerdict`, `split_claims`, `rank_evidence_for_claim`, `coverage`, `lexical_support`, `grounding_metrics`, `citation_metrics`, `extract_citations` |
| `units.py` | `UNIT_TABLE` (case-sensitive keys: `mW` / `MW`, `mPa` / `MPa`), `Quantity`, `UnitReport`, `normalize_unit`, `unit_family`, `parse_quantities`, `convert`, `check_units` |
| `sandbox.py` | `ExecutionResult`, `extract_code`, `fenced_languages`, `parse_test_summary`, `run_python` (isolated interpreter, temp dir, no inherited environment, hard timeout) |
| `policy.py` | `VerificationPolicy`, `policy_for_profile`, `apply_policy` |
| `plan.py` | `VerificationPlan`, `Thresholds`, `LLMPlan`, `TaskFeatures`, `extract_features`, `code_marker_counts`, `looks_like_code`, `classify_answer_type`, `estimate_calls`, `fit_to_budget`, `effective_accept_threshold`, `sanitize_verifiers`, `HeuristicPlanner` |
| `prompts.py` | 16 versioned `PromptTemplate`s (`PLANNER`, `DRAFT`, `DERIVE` + `DERIVE_FRAMINGS`, `USC_SELECT`, `EQUIVALENCE`, `COVE_PLAN`, `COVE_ANSWER`, `COVE_CROSSCHECK`, `REVISE`, `CLAIM_EXTRACT`, `CLAIM_CLASSIFY`, `RUBRIC` + `JUDGE_FRAMINGS`, `PAIRWISE`, `VERBALIZED_CONFIDENCE`, `DEBATE_TURN` + `DEBATE_FRAMINGS`, `DEBATE_JUDGE`), `DATA_NOT_INSTRUCTIONS` |
| `config.py` | `AnswerVerificationConfig`, `CalibrationParams`, `ALL_VERIFIERS` |
| `evaluation.py` | `accuracy`, `precision_recall_f1`, `brier_score`, `expected_calibration_error`, `auroc`, `VerificationExample`, `EvaluationReport`, `evaluate`, `evaluate_sync`, `synthetic_examples` |
| `cli.py` | `python -m answer_verification "task" [--answer] [--domain] [--offline] [--json] [--input k=v] [--document ...] [--tests] [--evaluate]` |

## 3. How it reasons about a prompt

### 3.1 The plan (`VerificationPlan`)

| Field | Meaning |
| --- | --- |
| `answer_type` | `factual \| numerical \| code \| procedural \| design \| free_text` - dominant type of the candidate |
| `verifiers` | Ordered verifier names to run (subset of `self_consistency, chain_of_verification, grounding, rubric_judge, rules, code_execution, debate`) |
| `samples` | `k_max` for self-consistency / number of candidates for Universal Self-Consistency |
| `thresholds` | `Thresholds(accept, escalate, min_agreement, min_supported_fraction, min_rubric_score)` |
| `use_evidence`, `allow_revision` | Whether evidence is used for grounding; whether LLM-edited revisions (grounding, chain of verification) may be produced |
| `judge_criteria` | Rubric criteria (config defaults + the profile's `verification_rules`) |
| `mandatory` | Verifiers whose failure, or failure to run, forces `NEEDS_REVIEW` |
| `final_effort` | Effort for final-judgement calls (`max` in safety-critical domains, else `high`) |
| `rationale`, `source`, `estimated_calls` | Audit trail: why, who decided (`heuristic \| llm`), the LLM-call estimate used for budget fitting |

The plan is attached to `result.metadata["plan"]`, rendered by the CLI
(`Plan: ...`) and embedded in the `VerificationReport`.

### 3.2 Heuristic planner (offline, `constraints.deterministic`, `llm_mode="never"`, or LLM plan invalid)

`extract_features` computes word/sentence counts, numbers and units in the
answer, code markers, list items, citations, numeric/procedure/design cue words
and whether the task is a factual question; `classify_answer_type` applies, in
order:

1. `code` if the answer has a **strong** code marker (a fence, `def`/`class`, SQL `SELECT ... FROM` in upper case, `=>`, or a line ending in `;`), or at least two **weak** markers (a line that *starts* with `import X`, `from X import Y` or `return`). Prose such as "the pump will return to service" or "the import duty is 5 percent" has no strong marker and at most one weak one, so it is never code; `run_rule_checks` only compiles text that passes the same `looks_like_code` test;
2. `numerical` if the answer has a number **and** (a unit or a quantity word) and <= 250 words;
3. `procedural` if a procedure word occurs and the answer has >= 2 list items or >= 3 sentences;
4. `design` if the task has a design/trade-off word and the answer has >= 40 words;
5. `factual` if the answer has <= 60 words and (the task is a question or the answer has <= 2 sentences);
6. otherwise `free_text`.

`HeuristicPlanner.plan` then selects verifiers:

| `answer_type` | Verifiers | Why |
| --- | --- | --- |
| `numerical` | `self_consistency`, `rules` | majority vote over independent derivations; unit/range rules |
| `code` | `rules`, then `code_execution` when `constraints.allow_code_execution` and `inputs["tests"]` exist, else `self_consistency`; `rubric_judge` | execution is the oracle; USC consensus otherwise |
| `factual` | `self_consistency` | agreement across re-derivations |
| `procedural` | `rubric_judge`, plus `chain_of_verification` when >= 4 sentences | completeness/order by rubric; factual steps by verification questions |
| `design` | `rubric_judge`, plus `debate` in high-risk domains when the call budget is unlimited or >= 12 | open-ended judgement; bounded debate only where it pays |
| `free_text` | `chain_of_verification`, `rubric_judge` | factored verification questions plus rubric |

Then: `grounding` is inserted **first** whenever evidence exists; `rules` is
appended for `high`/`safety_critical` risk; for `safety_critical` the present
`grounding` and `rules` become `mandatory`, `accept` rises to >= 0.85,
`min_supported_fraction` to `grounding_min_supported_safety_critical` (1.0) and
`min_agreement` to >= 0.8. `thresholds.accept` is always the value `execute()`
enforces: `max(release_threshold, profile.min_confidence_to_accept,
constraints.min_confidence)` (`effective_accept_threshold`). `samples` is `samples_default` (5), reduced to
`samples_min` (3) under `constraints.deterministic`, raised to `samples_max`
(10) for high-stakes numerical/factual answers, and capped at `usc_candidates`
(5) for free-form types. `allow_revision` is true only when evidence or tests
exist. Finally `apply_policy` (Section 3.4) and `fit_to_budget` run.

### 3.3 LLM planner (online)

One structured call with the `PLANNER` template (`effort = plan_effort`)
returns an `LLMPlan(answer_type, verifiers, samples, use_evidence,
allow_revision, judge_criteria, rationale)`. The user turn lists the available
oracles (evidence count, tests, execution permission, reference, pre-sampled
candidates, risk level, call budget); the stable system block carries the same
rules the heuristic planner encodes, including "external evidence outranks
model judgement" and "intrinsic self-critique is not a verifier". Python then
sanitises the verifier names, drops `grounding` without evidence and
`code_execution` without tests or permission, clamps `samples` to
`[1, samples_max]`, falls back to heuristic selection when nothing applicable
remains, and applies the same effective thresholds as the heuristic planner. If the call fails or the
JSON does not validate, the heuristic planner is used and a warning is recorded
(`plan.source` tells you which path ran).

### 3.4 Policy (`policy.py`)

`policy_for_profile(profile, config)` derives the domain posture; `apply_policy`
enforces it on either plan (floors only ever rise):

| `risk_level` | Added / mandatory verifiers | Thresholds | Revision | Contradictions |
| --- | --- | --- | --- | --- |
| `low`, `medium` | none | profile `min_confidence_to_accept`; `escalate_below` | `max_revision_rounds` (2) | allowed to lower the score only |
| `high` | `rules` added and mandatory | as above | 2 rounds | any contradicted claim blocks acceptance |
| `safety_critical` | `grounding` (when evidence exists) and `rules` mandatory | `accept >= 0.85`, `min_supported_fraction = 1.0`, `min_agreement >= 0.8`, `escalate >= 0.6`, `final_effort = max` | at most 1 round | block acceptance; external evidence (documents or tests) is **required** |

### 3.5 Execution, fusion and gates

Each verifier returns a `Verdict(method, passed, score, issues,
corrected_answer, evidence_ids, details)` or an **inconclusive** verdict
(`details["inconclusive"] = True`, no weight) when its inputs are missing
(no LLM offline, no evidence, no tests, no applicable rule, insufficient
budget). Scores are:

| Verifier | `score` | `passed` |
| --- | --- | --- |
| `self_consistency` | agreement of samples with the candidate, `(cluster size - 1) / samples` | agreement >= `min_agreement`, or the USC pick lies in the candidate's cluster **and** at least one sample agrees with the candidate |
| `chain_of_verification` | `(agree + 0.5 x unknown) / questions` | no contradicted claim (a failed optional revision call keeps the verdict and reports the contradictions without a correction) |
| `grounding` | supported fraction, halved when any claim is contradicted; a one-line answer is a single claim | fraction >= `min_supported_fraction` and (no contradictions or the policy tolerates them) |
| `rubric_judge` | mean over passes of `((mean criterion + overall) / 2 - 1) / (scale - 1)` | >= `min_rubric_score` |
| `rules` | 0.75 on a clean pass (0.95 when a numeric reference matched) minus 0.1 per soft failure; `max(0.05, 0.3 - 0.1 x (hard failures - 1))` on a hard failure; **inconclusive when no check applies** (domain without `units`, no `inputs["rules"]`/`inputs["reference"]`, not a code answer) | no hard failure |
| `code_execution` | 0.98 pass, `0.3 x passed/total` partial, 0.1 timeout, 0.05 otherwise; an unfenced snippet runs as Python whenever the profile lists Python (a fence naming another language, or `inputs["language"]`, is respected) | tests pass |
| `debate` | judge confidence if `correct`, `1 - confidence` if `incorrect`, 0.5 if `uncertain`; without a judge (no budget or the judge call failed) the final-round vote share decides | judge says `correct` |
| `verbalized_confidence` | probability mass the model assigns to the candidate (advisory: it never gates acceptance on its own) | >= 0.5 |

`aggregate_verdicts` fuses the conclusive verdicts as a weighted mean using
`config.weights` (external-evidence verifiers outrank LLM judgement), then
applies the precedence rules: a **failed authority-3 verifier** (`rules`,
`code_execution`) caps confidence at its own score; a grounding verdict with
**contradicted claims caps at 0.5**; `platt_scale` is applied last; the basis
string names every contributing and every inconclusive verifier.

A **revision** is handled as follows. The highest-authority verifier that
produced a `corrected_answer` proposes it (grounding and chain of verification
only produce one when evidence documents exist and `allow_revision` holds, and
only if the edit preserves >= `revision_min_preservation` of the original;
self-consistency proposes the majority answer when the candidate falls outside
the majority cluster and that cluster holds >= `min_agreement` of the samples). `_confirm_revision` then re-runs the plan's external
verifiers (`grounding`, `rules`, `code_execution`) on the proposal for up to
`policy.max_revision_rounds` rounds; a failing grounding check may itself
supply the next evidence-backed edit. The proposal is applied to
`final_answer` only when every conclusive re-check passes (a re-check that has
nothing to check, such as `rules` in a domain without units, is inconclusive
and does not count as confirmation), or, when no re-check was conclusive, when
the proposer is itself an external signal (authority >= 2:
`revision_verified=False`, "applied on its own authority"). Otherwise the
original answer is kept and the proposal is reported in `issues`.
`max_revision_rounds = 0` disables revisions entirely. An applied revision
always sets `NEEDS_REVIEW` so a human sees the change.

Gates, in order: no verifier of record produced a conclusive verdict (advisory
verbalised confidence does not count; the fused confidence is then the
uninformative prior 0.5 and the answer is reported as **not verified**); a
mandatory verifier that failed or did not produce a verdict; contradicted
claims under a no-contradiction policy; a safety-critical domain with neither
documents nor tests; `confidence < thresholds.escalate` (abstain,
`escalated=True`); `confidence < thresholds.accept` (the effective threshold
above); an applied revision. No reason -> `SUCCESS` (`PARTIAL` if a verifier
errored or the budget ran out mid-way); otherwise `NEEDS_REVIEW`. `FAILED` is
produced only by `run()`'s error containment.

### 3.6 Budget fitting

`estimate_calls` prices a plan (samples for self-consistency plus USC passes
and the equivalence call for free-form types; `2 + questions + revision` for
chain of verification; `2 + revision` for grounding; `judge_samples` for the
rubric; `agents x rounds + 1` for debate; one call for verbalised confidence;
`rules` and `code_execution` are free). `fit_to_budget` first lowers `samples`
in steps of two down to `samples_min`, then to one if the rest fits, then drops
verifiers in the order `debate, chain_of_verification, rubric_judge,
self_consistency, grounding`, never a mandatory one, until the plan fits
`constraints.max_llm_calls` minus the planning call. During execution
`affordable()` is checked before every fan-out, sample counts degrade rather
than fail, and a `BudgetExceededError` inside a verifier yields an inconclusive
verdict and `PARTIAL` instead of an exception.

### 3.7 Which number is "the answer"

Votes (`canonicalize_answer`), `inputs["reference"]` closeness and `range`
rules all compare one number from a text that usually states several
("16 bar at 80 degC", "a 75 kW motor at 1480 rpm", "12.5 kN per M16 bolt").
`extract_number(text, unit=None)` chooses it deterministically:

1. when a unit is known (the reference's unit, or the `range` rule's `unit`
   field), the first quantity of that unit **family** (`unit_family`), preferring
   one introduced by a result marker;
2. else the number after a result marker (`final answer`, `answer`, `result`,
   `therefore`, `thus`, `hence`);
3. else the right-hand side of the **last** `=` (derivation chains end in the result);
4. else the first stand-alone number.

Digits glued to letters (`M16`, `A743`, `Gr 70` identifiers) and numbers inside
fenced code blocks are never the answer; thousands separators are removed.
State the result first or after a marker ("Result: the required thickness is
6.6 mm, to which the 3 mm corrosion allowance is added") and the checks follow.

## 4. Quick start

### Python

```python
from pipeline_core import Constraints, Evidence, TaskRequest, make_llm
from answer_verification import AnswerVerificationModel

model = AnswerVerificationModel(make_llm())        # MockLLM without credentials

spec = [
    Evidence(id="spec-4.2", content="Section 4.2: the pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces.", source="Pump spec rev C"),
    Evidence(id="spec-t2", content="Table 2: discharge piping design pressure 16 bar at 80 degC.", source="Pump spec rev C"),
]

# 1. component API: inputs["answer"] is the candidate
result = model.run_sync(TaskRequest(
    task="What corrosion allowance does the specification require for the pump casing?",
    inputs={"answer": "The specification requires a corrosion allowance of 3 mm on all wetted casing surfaces."},
    context=spec,
    domain="mechanical_engineering",
))
print(result.status.value, result.confidence, result.confidence_basis)
report = result.output                              # VerificationReport
print(report.accepted, report.review_reasons, report.metrics["grounding.supported_fraction"])
print(result.output_text)

# 2. Verifier protocol (what Pipeline RAG and Task Decomposition call); a one-line answer is grounded as one claim
verdict = model.verify_sync("16 bar at 80 degC", task="What is the discharge piping design pressure?", evidence=spec)
print(verdict.passed, verdict.score, verdict.evidence_ids, verdict.corrected_answer)   # True 1.0 ['spec-t2'] None

# 3. oracles: pre-sampled candidates, tests, reference, structured rules
code = "```python\ndef add(a, b):\n    return a + b\n```"
result = model.run_sync(TaskRequest(
    task="Implement add",
    inputs={"answer": code, "tests": "assert add(2, 3) == 5\nprint('1 passed')"},
    domain="software_engineering",
    constraints=Constraints(allow_code_execution=True, max_llm_calls=10),
))
```

`AnswerVerificationModel(llm, config=AnswerVerificationConfig(...) | dict,
retriever=..., executor=..., tracer=..., **field_overrides)`;
`await model.verify_report(...)` returns the full `VerificationReport`;
`await model.compare(a, b, task=...)` is the position-swapped pairwise judge.
Each verifier class is also usable standalone and satisfies the `Verifier`
protocol (`await GroundingVerifier(config).verify(answer, task=..., evidence=...)`).

### CLI

```
python -m answer_verification "Verify this answer: the required NPSH margin is 1.5 m" --offline
python -m answer_verification "What is the design pressure?" --answer "16 bar at 80 degC" \
    --domain mechanical_engineering --document "spec-2: Table 2: discharge piping design pressure 16 bar at 80 degC." --trace
python -m answer_verification "Implement add" --input answer=@answer.md --tests tests_add.py --domain software_engineering
python -m answer_verification "Compute the preload" --answer "12.5 kN" --input 'rules=[{"type":"range","min":10,"max":15}]' --max-calls 6 --json
python -m answer_verification --evaluate --offline --n 12
```

Flags: `--answer`, `--domain`, `--offline`, `--json`, `--input KEY=VALUE`
(JSON parsed, `@file` reads a file), `--document TEXT` (`id: text`, plain text
or `@file`), `--tests PATH` (implies `--allow-code-execution`),
`--allow-code-execution`, `--max-calls`, `--min-confidence`,
`--deterministic`, `--llm-mode auto|force|never`, `--trace`, `--evaluate`,
`--n`. Exit status is 1 only when the run `FAILED`.

## 5. Configuration (`AnswerVerificationConfig`)

| Field | Default | Justification (dossier source) |
| --- | --- | --- |
| `default_domain` | `general` | Domain used by the direct `verify()` API when none is given (ARCHITECTURE.md 3.4: never hard-code a domain) |
| `llm_mode` | `auto` | `auto` uses the LLM unless `ctx.offline`/deterministic; `force` for `ScriptedLLM` tests and replay; `never` for heuristics only (ARCHITECTURE.md 3.1 heuristic fallback) |
| `max_concurrency` | `4` | Fan-out ceiling when no `Executor` is injected (also bounded by `Constraints.max_parallelism`); samples, questions and judge passes are embarrassingly parallel: Aggarwal et al. (2023), *Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs*, https://arxiv.org/abs/2305.11860 (batched sampling) |
| `plan_effort` | `high` | Planning is a reasoning task (ARCHITECTURE.md 3.2: `high` for planning and synthesis) |
| `grade_effort` | `low` | Per-item grading/classification of many items (ARCHITECTURE.md 3.2) |
| `derive_effort` | `medium` | Independent re-derivations and verification answers need reasoning but are repeated k times; Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, https://arxiv.org/abs/2203.11171 shows gains come from diversity, not per-sample depth |
| `judge_model` | `None` | Optional cross-family judge: judges favour their own generations: Panickssery et al. (2024), *LLM Evaluators Recognize and Favor Their Own Generations*, https://arxiv.org/abs/2404.13076; Verga et al. (2024), *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models*, https://arxiv.org/abs/2404.18796 |
| `samples_default` | `5` | 5-10 paths capture most of the 40-path gain: Wang et al. (2022) |
| `samples_min` | `3` | Adaptive-Consistency `k_min = 3` before the stopping rule is consulted: Aggarwal et al. (2023) |
| `samples_max` | `10` | Returns diminish and can reverse past ~10 samples; USC dropped at 16: Chen et al. (2023), *Universal Self-Consistency for Large Language Model Generation*, https://arxiv.org/abs/2311.17311; Wang et al. (2022) |
| `sample_batch` | `3` | Adaptive-Consistency batch size: Aggarwal et al. (2023) |
| `stability_threshold` | `0.95` | Stop when the Dirichlet posterior P(majority stays top) >= 0.95: up to 7.9x fewer samples with < 0.1 % accuracy loss: Aggarwal et al. (2023) |
| `low_agreement_threshold` | `0.6` | Agreement fraction is highly correlated with accuracy; below ~0.6 the answer is flagged uncertain: Wang et al. (2022) |
| `usc_candidates` | `5` | USC gains are flat after 5 candidates: Chen et al. (2023) |
| `usc_selection_passes` | `2` | Selection over a candidate list has position bias, so 2-3 shuffled passes and the modal choice: Chen et al. (2023); Wang et al. (2023), *Large Language Models are not Fair Evaluators*, https://arxiv.org/abs/2305.17926 |
| `equivalence_jaccard` | `0.6` | Lexical stand-in for bidirectional entailment when no LLM/NLI clustering is available: Kuhn et al. (2023), *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*, https://arxiv.org/abs/2302.09664 |
| `llm_equivalence_clustering` | `True` | One call to cluster free-form answers by mutual entailment before computing semantic entropy: Kuhn et al. (2023); Kossen et al. (2024), *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs*, https://arxiv.org/abs/2406.15927 |
| `cove_max_questions` | `10` | One question per atomic claim; CoVe drafts carry ~12-17 facts and precision matters more than count: Dhuliawala et al. (2023), *Chain-of-Verification Reduces Hallucination in Large Language Models*, https://arxiv.org/abs/2309.11495 |
| `allow_revision` | `True` | Evidence-backed revision (RARR research-and-revise): Gao et al. (2022), *RARR: Researching and Revising What Language Models Say, Using Language Models*, https://arxiv.org/abs/2210.08726; never intrinsic: Huang et al. (2023) |
| `max_revision_rounds` | `2` | Cap repairs at 1-2: Olausson et al. (2023), *Is Self-Repair a Silver Bullet for Code Generation?*, https://arxiv.org/abs/2306.09896; Kamoi et al. (2024), *When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs*, https://arxiv.org/abs/2406.01297 |
| `revision_min_preservation` | `0.8` | RARR scores revisions on attribution **and** preservation; large rewrites are rejected: Gao et al. (2022) |
| `grounding_evidence_per_claim` | `5` | Top-5 passages per claim: Min et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*, https://arxiv.org/abs/2305.14251 |
| `grounding_max_claims` | `15` | Atomic-fact decomposition bound; FActScore reports claim count alongside precision: Min et al. (2023) |
| `grounding_min_supported` | `0.9` | Supported fraction required to pass; citation support is incomplete ~50 % of the time for the best models, so the gate must be explicit: Gao et al. (2023), *Enabling Large Language Models to Generate Text with Citations (ALCE)*, https://arxiv.org/abs/2305.14627 |
| `grounding_min_supported_safety_critical` | `1.0` | Compliance / normative statements need full attribution: Rashkin et al. (2021), *Measuring Attribution in Natural Language Generation Models*, https://arxiv.org/abs/2112.12870 |
| `lexical_support_threshold` | `0.5` | Content-token coverage for the offline support heuristic (numbers and identifiers must match exactly): Min et al. (2023) retrieval baseline; FEVER-style claim/evidence matching: Thorne et al. (2018), *FEVER: a large-scale dataset for Fact Extraction and VERification*, https://arxiv.org/abs/1803.05355 |
| `rubric_criteria` | `correctness, completeness, groundedness, clarity` | Explicit fine-grained criteria: Liu et al. (2023), *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*, https://arxiv.org/abs/2303.16634; Kim et al. (2023), *Prometheus: Inducing Fine-grained Evaluation Capability in Language Models*, https://arxiv.org/abs/2310.08491 |
| `rubric_scale` | `5` | G-Eval 1-5 scale: Liu et al. (2023) |
| `judge_samples` | `3` | Several independent passes averaged (framing variation instead of temperature): Liu et al. (2023); Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, https://arxiv.org/abs/2306.05685 |
| `judge_min_score` | `0.7` | Judges reach ~85 % agreement with experts only with explicit criteria and references; a 0.7 normalised bar keeps the judge advisory: Zheng et al. (2023) |
| `pairwise_swap` | `True` | Evaluate both orderings, count only consistent wins: Wang et al. (2023), *Large Language Models are not Fair Evaluators*, https://arxiv.org/abs/2305.17926; Zheng et al. (2023) |
| `debate_agents` | `3` | Three agents: Du et al. (2023), *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, https://arxiv.org/abs/2305.14325 |
| `debate_rounds` | `2` | Two rounds: Du et al. (2023) |
| `debate_max_rounds` | `4` | Longer debates degenerate: Liang et al. (2023), *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate*, https://arxiv.org/abs/2305.19118 |
| `debate_adaptive_stop` | `True` | Adaptive break once reviewers agree: Liang et al. (2023) |
| `retrieve_k` | `8` | Passages fetched from an injected `Retriever` when the request carries none; ALCE grounds on the top-5 retrieved passages and FActScore scores each claim against its top-5, so 8 leaves headroom for ranking by claim: Gao et al. (2023); Min et al. (2023) |
| `code_timeout_s` | `30.0` | Execution with a hard timeout is the code oracle: Chen et al. (2022), *CodeT: Code Generation with Generated Tests*, https://arxiv.org/abs/2207.10397 |
| `code_max_output_chars` | `4000` | Failure trace kept as repair feedback, bounded: Olausson et al. (2023) |
| `elicit_verbalized_confidence` | `True` | One-call top-k verbalised confidence as an extra signal: Tian et al. (2023), *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback*, https://arxiv.org/abs/2305.14975 |
| `verbalized_top_k` | `4` | Top-k verbalisation beats single-answer confidence: Tian et al. (2023) |
| `verbalized_weight` | `0.5` | Verbalised confidence is overconfident on professional tasks, so it is advisory (lowest weight): Xiong et al. (2023), *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs*, https://arxiv.org/abs/2306.13063 |
| `weights` | `code_execution 2.0, rules 1.5, grounding 1.5, self_consistency 1.0, chain_of_verification 1.0, rubric_judge 0.75, debate 0.5` | Precedence "external evidence over LLM judgement over self-critique": Huang et al. (2023); Gou et al. (2023); Chen et al. (2022) |
| `release_threshold` | `None` | Extra acceptance **floor**: the enforced threshold is `max(release_threshold, DomainProfile.min_confidence_to_accept, constraints.min_confidence)`, reported in `plan.thresholds.accept`; a value below the profile minimum has no effect and `None` adds no floor (ARCHITECTURE.md 3.1 rule 5; 0.85 or more for safety-critical profiles) |
| `escalate_below` | `0.5` | Human escalation threshold for selective prediction: Xiong et al. (2023); Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*, https://arxiv.org/abs/2207.05221 |
| `calibration` | `CalibrationParams(a=1.0, b=0.0)` | Identity until a labelled set is available; Platt / temperature scaling with stored parameters: Guo et al. (2017), *On Calibration of Modern Neural Networks*, https://arxiv.org/abs/1706.04599 |
| `draft_when_missing` | `True` | Without `inputs["answer"]` the model drafts one (online) so the CLI always has something to verify (ARCHITECTURE.md 6: `--offline` must produce a meaningful result; offline the task text is verified with a warning) |
| `max_answer_chars` | `20000` | Candidate truncation bound: factual precision collapses for long outputs, and claim decomposition is bounded by `grounding_max_claims` anyway: Min et al. (2023) |

## 6. Techniques implemented

| Technique | Where | Citation(s) |
| --- | --- | --- |
| Self-consistency over independent re-derivations with varied framing (no temperature), canonical-key voting | `SelfConsistencyVerifier`, `canonicalize_answer`, `DERIVE_FRAMINGS` | Wang et al. (2022); diversity by prompt variation: Chen et al. (2023) |
| Adaptive-Consistency early stopping (Dirichlet posterior over answer counts) | `majority_stability`, `should_stop_sampling` | Aggarwal et al. (2023) |
| Universal Self-Consistency: LLM-selected consensus over shuffled candidates, modal choice over passes | `SelfConsistencyVerifier._usc_select`, `USC_SELECT` | Chen et al. (2023) |
| Discrete semantic entropy over equivalence clusters (lexical or LLM bidirectional entailment) | `semantic_agreement`, `EQUIVALENCE` | Kuhn et al. (2023); Kossen et al. (2024); Manakul et al. (2023), *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*, https://arxiv.org/abs/2303.08896 |
| Factored Chain-of-Verification (questions answered without seeing the draft, cross-check, evidence-gated revision) | `ChainOfVerificationVerifier`, `COVE_*`, `REVISE` | Dhuliawala et al. (2023) |
| Research-and-revise with preservation check | `REVISE`, `similarity_ratio`, `revision_min_preservation` | Gao et al. (2022); Zhao et al. (2023), *Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework*, https://arxiv.org/abs/2305.03268 |
| Atomic-claim grounding: supported / contradicted / unsupported, supported fraction with claim count and contradictions | `GroundingVerifier`, `claims.py` | Min et al. (2023); Chern et al. (2023), *FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios*, https://arxiv.org/abs/2307.13528 |
| Citation precision / recall | `citation_metrics` | Gao et al. (2023); Rashkin et al. (2021) |
| G-Eval-style rubric judging with explicit criteria, anti-verbosity rule, reference-guided grading, averaged passes | `RubricJudgeVerifier`, `RUBRIC`, `JUDGE_FRAMINGS` | Liu et al. (2023); Zheng et al. (2023); Dubois et al. (2024), *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*, https://arxiv.org/abs/2404.04475 |
| Pairwise judging with position swapping (balanced position calibration) | `RubricJudgeVerifier.compare`, `pairwise_decision`, `PAIRWISE` | Wang et al. (2023); Zheng et al. (2023) |
| Deterministic rules: units (case-sensitive parser with SI families), regex, forbidden patterns, numeric range (unit-family aware), required sections, max words, JSON keys, Python syntax, numeric reference closeness; inconclusive when nothing applies | `RuleVerifier`, `run_rule_checks`, `units.py`, `extract_number` | Tool-grounded checks over self-critique: Gou et al. (2023); calculator/units arm: Chern et al. (2023); backward substitution against a reference: Weng et al. (2022), *Large Language Models are Better Reasoners with Self-Verification*, https://arxiv.org/abs/2212.09561 |
| Sandboxed test execution as the code oracle | `CodeExecutionVerifier`, `sandbox.py` | Chen et al. (2022); Olausson et al. (2023) |
| Bounded multi-agent debate with per-agent reviewer personas and peer order (framing diversity, no temperature), peer summaries, adaptive termination and an evidence-weighing judge | `DebateVerifier`, `DEBATE_FRAMINGS`, `DEBATE_*` | Du et al. (2023); Liang et al. (2023); used sparingly because it loses to self-consistency at matched budget: Huang et al. (2023) |
| Top-k verbalised confidence (advisory) | `verbalized_confidence`, `probability_for_candidate` | Tian et al. (2023); Xiong et al. (2023); Lin et al. (2022), *Teaching Models to Express Their Uncertainty in Words*, https://arxiv.org/abs/2205.14334 |
| Precedence-aware confidence fusion with hard caps and Platt scaling | `aggregate_verdicts`, `AUTHORITY`, `platt_scale`, `fit_platt` | Huang et al. (2023); Kamoi et al. (2024); Guo et al. (2017) |
| Selective prediction: acceptance and escalation thresholds, domain-mandatory verifiers | `policy.py`, `execute()` gates | Kadavath et al. (2022); Xiong et al. (2023); Mohri et al. (2024), *Language Models with Conformal Factuality Guarantees*, https://arxiv.org/abs/2402.10978 |
| Cache-aware prompt layout and injection hygiene (documents marked as data, never instructions) | `build_request`, `DATA_NOT_INSTRUCTIONS` | Cross-dossier sources from `docs/research/prompting-techniques.json` (the answer-verification dossier has no injection paper): Greshake et al. (2023), *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*, https://arxiv.org/abs/2302.12173; Hines et al. (2024), *Defending Against Indirect Prompt Injection Attacks With Spotlighting*, https://arxiv.org/abs/2403.14720 |

## 7. Adapting to an engineering domain

Behaviour is driven by the `DomainProfile` named in `TaskRequest.domain`; no
code changes are needed. The fields this model reads: `units` (unit hygiene
in `RuleVerifier`; a bare decimal as the final number of a numerical answer is
a hard failure once the domain declares units; without `units`, rules or a
reference the rule verifier is inconclusive, so a domain that wants rule
checks must declare its units), `verification_rules`
(appended to the rubric criteria and rendered in every stable system block),
`code_languages` (Python syntax check and the sandbox runtime hint),
`risk_level` (`high` adds mandatory rules and blocks contradictions;
`safety_critical` adds mandatory grounding, requires external evidence and
raises every threshold), `min_confidence_to_accept` (acceptance threshold),
`glossary`/`standards`/`style_guidance` (prompt context through
`profile.system_text()`).

```python
from pipeline_core import DomainProfile, Evidence, TaskRequest, register_profile
from answer_verification import AnswerVerificationModel

register_profile(DomainProfile(
    name="pressure_vessels",
    description="ASME BPVC Section VIII Division 1 pressure vessel design.",
    units=["MPa", "mm", "bar", "degC", "kN"],
    verification_rules=[
        "Every thickness must state the corrosion allowance included.",
        "Allowable stress must cite the material table and design temperature.",
    ],
    risk_level="safety_critical",
    min_confidence_to_accept=0.85,
    standards=["ASME BPVC VIII-1", "ASME II-D"],
))

model = AnswerVerificationModel()
result = model.run_sync(TaskRequest(
    task="What minimum shell thickness is required for the SA-516 Gr 70 vessel at the design case?",
    inputs={
        "answer": "Assumptions: P = 1.5 MPa, R = 600 mm, S = 138 MPa at 150 degC, E = 1.\nResult: the required thickness is 6.6 mm, to which the 3 mm corrosion allowance is added.",
        "rules": [
            {"type": "required_sections", "sections": ["Assumptions", "Result"]},
            {"type": "regex", "pattern": r"corrosion allowance", "description": "states the corrosion allowance"},
            {"type": "range", "min": 5, "max": 15, "unit": "mm", "description": "plausible thickness band"},  # 'unit' picks the mm result, not 1.5 MPa or 138 MPa
        ],
    },
    context=[Evidence(id="ug-27", content="UG-27(c)(1): ... P = 1.5 MPa, R = 600 mm, S = 138 MPa, E = 1.0 the required thickness is 6.6 mm plus 3 mm corrosion allowance, 9.6 mm in total.", source="calc note CN-12")],
    domain="pressure_vessels",
))
# plan: ['grounding', 'self_consistency', 'rules'], mandatory ['grounding', 'rules'], accept >= 0.85,
# min_supported_fraction 1.0, final_effort 'max'; without evidence the same request is NEEDS_REVIEW with
# "safety-critical domain requires external evidence (documents or tests); none supplied".
```

`examples/engineering_example.py` runs this profile end to end, including a
wrong answer (contradicted and rule-failing), the same answer under `general`,
pre-sampled candidates, a majority revision re-verified by grounding and rules,
a general-domain answer with nothing to check (inconclusive rules, `NEEDS_REVIEW`)
and a code answer executed against tests. Structured rules accepted in
`inputs["rules"]`: `regex` (`pattern`, `must_match`), `forbidden` (`pattern`),
`range` (`min`, `max`, optional `unit` selecting the quantity by unit family),
`required_sections` (`sections`), `max_words` (`value`, soft), `json_schema`
(`required` keys); plus `inputs["reference"]` with optional
`inputs["tolerance"]` (default 1 %). A reference that carries a unit (`"16
bar"`) is compared with the answer's quantity of the same family, so "16 bar at
80 degC" matches it (Section 3.7). Unit tokens are case-sensitive (`mW` is a
milliwatt, `MW` a megawatt); unambiguous spellings (`bar`, `PSI`, `rpm`) and
the conventional dropped-case `mpa` are accepted, counts followed by ordinary
words ("4 days", "7 bolts") are not units.

## 8. Composing with the other five models

| Model | Direction | Interface |
| --- | --- | --- |
| Pipeline RAG | provides | `AnswerVerificationModel` **is a** `Verifier`: `await verify(answer, task=..., evidence=...) -> Verdict` with `passed`, calibrated `score`, `issues`, `corrected_answer` (only an applied, signal-backed revision), `evidence_ids` (supporting documents) and `details{status, answer_type, confidence_basis, verifiers, metrics, escalated, revision_verified, usage, component_status, request_id}`; callers roll `details["usage"]` (the verification run's `Usage` as a dict) into their own result with `ctx.charge(Usage(**details["usage"]))` |
| Task Decomposition | provides | The same `verify()` at sub-task boundaries and on the recomposed answer; `details["answer_type"]` and `metrics` let the decomposer route low-agreement sub-tasks to heavier verification |
| Hybrid Retrieval | accepts | `AnswerVerificationModel(retriever=...)`: when a request carries no evidence and `constraints.allow_external_tools`, `retriever.retrieve(task, k=retrieve_k)` supplies claim-level evidence (step `retrieve`); `unsupported` claims and low `citation_recall` in `details["metrics"]` are retriever-gap signals |
| Parallelization | accepts | `AnswerVerificationModel(executor=...)`: samples, verification questions, judge passes and debate turns fan out through `Executor.map(fn, items, concurrency=...)`, bounded by `max_concurrency` and `Constraints.max_parallelism`; a bounded `asyncio.gather` is used otherwise |
| Caching | accepts | `AnswerVerificationModel(CachingModel.wrap(llm))`; every prompt keeps stable blocks first (rules, few-shot, domain profile, engineering rigor) with task/answer/documents in the user turn and no timestamps or ids, so planner, derivation, grading and judge prefixes are cacheable; `VerificationReport` and per-claim verdicts are the natural cache values |

The package never imports another model package; it accepts protocol objects
(`Retriever`, `Executor`, any `BaseLLMClient`) and exposes `Verifier`. Nested
work is charged through the shared `RunContext`, so `result.usage` rolls up
and one `Tracer` shows `answer_verification.run -> plan -> verify.* ->
confidence.verbalized -> fuse -> revision.confirm`.

## 9. Evaluation

Metrics (`evaluation.py`, pure functions): decision `accuracy` and
`precision_recall_f1` of accept/reject against labels; calibration of the fused
confidence with `brier_score` and `expected_calibration_error` (10 equal-width
bins; dossier target ECE <= 0.05); `auroc` of confidence for separating correct
from incorrect answers (target >= 0.8); mean confidence; abstention
(escalation) rate and `NEEDS_REVIEW` rate; LLM calls, total and per-answer
cost, mean latency. `fit_platt(scores, labels)` in `uncertainty.py` fits
`CalibrationParams` from such a run.

`evaluate(model, examples, domain=..., constraints=...) -> EvaluationReport`
runs the model over `VerificationExample(task, answer, correct, evidence,
inputs, domain)` items and returns the metrics plus `per_item` rows;
`evaluate_sync` wraps it. `synthetic_examples(n)` builds a deterministic
pump-specification set (half correct and grounded, half with a wrong value
against the same evidence) so the harness runs offline and in CI:

```
python -m answer_verification --evaluate --offline --n 12
Evaluated 12 labelled answers
  decision accuracy      1.000   precision 1.000  recall 1.000  F1 1.000
  calibration            Brier 0.000  ECE 0.000  AUROC 1.000
  mean confidence        0.500
  abstention / review    50.00% / 50.00%
  ...
```

Offline in the `general` domain the only conclusive signal is lexical
grounding (the rule verifier has no units to check and reports itself
inconclusive), so every item is scored 1.0 or 0.0 and the calibration numbers
are trivially perfect on a set whose wrong answers all contradict the evidence;
they say nothing about the fused confidence a live client produces, where the
self-consistency, chain-of-verification and judge signals spread the scores.
Run the harness with `--domain mechanical_engineering` to add the unit rules,
and replace the synthetic set with labelled answers from your own pipeline
before storing fitted `CalibrationParams` in the config.

## 10. Limitations and failure modes

- **Consistency is not correctness.** Shared misconceptions pass
  self-consistency and USC (oracle 96.2 vs USC 90.2 on GSM8K: Chen et al.
  (2023)); the planner therefore pairs agreement with grounding or rules
  whenever their inputs exist, and the weights rank them higher.
- **Revisions are only as good as the signal that backs them.** Intrinsic
  self-correction regresses accuracy (Huang et al. (2023); Kamoi et al.
  (2024)); this model never revises on model judgement alone, re-verifies every
  proposal, caps rounds at `max_revision_rounds`, and still routes applied
  revisions to `NEEDS_REVIEW`. Code self-repair is often not worth the cost at
  equal budget (Olausson et al. (2023)): prefer more candidates filtered by tests.
- **Judges are biased.** Position bias, verbosity bias and self-preference
  (Zheng et al. (2023); Wang et al. (2023); Dubois et al. (2024); Panickssery
  et al. (2024)) are mitigated by swapping, anti-verbosity rules, averaged
  framings and an optional cross-family `judge_model`, but the rubric judge
  stays advisory (weight 0.75) and cannot by itself pass a safety-critical answer.
- **Precision-only factuality rewards saying less** (Min et al. (2023);
  Dhuliawala et al. (2023)): the grounding verdict always reports claim count
  and contradictions next to the supported fraction, and `grounding_metrics`
  / `citation_metrics` are exposed for recall-oriented monitoring.
- **The offline lexical grounder is a heuristic.** It needs the claim's
  numbers and identifiers to appear verbatim in a passage; derived values
  (`6.6 + 3 = 9.6 mm` when the evidence never states 9.6) or unit conversions
  in the answer are labelled `contradicted`/`unsupported` until the LLM
  classifier runs. Treat its verdicts as conservative.
- **Nothing conclusive means not verified.** Offline, a bare answer in a domain
  without `units`, with no evidence, candidates, rules, reference or tests,
  leaves every verifier inconclusive; the run is `NEEDS_REVIEW` at the prior
  0.5 and says so. Supply one external signal (documents, `candidates`,
  `rules`/`reference`, or a profile with `units`) to get a verdict offline.
- **Verbalised confidence is overconfident on professional-knowledge tasks**
  (Xiong et al. (2023); Tian et al. (2023)); it carries the lowest weight and
  the identity calibration is a placeholder until `fit_platt` is run on
  labelled data.
- **Debate is expensive and can degenerate** (Liang et al. (2023)); at matched
  budget it loses to self-consistency (Huang et al. (2023)), so it is planned
  only for open-ended `design` answers in high-risk domains with budget.
- **The sandbox is containment, not a security boundary.** `run_python` uses a
  fresh interpreter with `-I`, an empty environment and a temp directory, but
  untrusted code should run in a container or microVM; only Python is executed.
- **Long outputs and rare entities** collapse factuality (Min et al. (2023)):
  answers are capped at `max_answer_chars`, claims at `grounding_max_claims`,
  and long-tail identifiers (part numbers, clause ids) must be matched exactly
  by the rule and grounding checks rather than paraphrased.
- **Prompt injection via documents** is mitigated, not solved: every template
  states that `<documents>`, `<answer>` and `<task>` content is data (Greshake
  et al. (2023); Hines et al. (2024), both cited from the prompting-techniques
  dossier); keep the structured-output schemas and Python-side validation in
  place.

## 11. How it was built

1. **Research** (`docs/research/answer-verification.json`; the two prompt
   injection papers are cross-dossier citations from
   `prompting-techniques.json` and are marked as such where they appear): the
   self-correction literature (Huang,
   Madaan, Gou, Tyen, Olausson, Kamoi) fixed the central design constraint -
   revisions only on external signals - and the precedence rule for fusion;
   Self-Consistency, Adaptive-Consistency and USC fixed `k`, batch size,
   stopping posterior and candidate count; CoVe fixed the factored variant;
   FActScore, ALCE and RARR fixed the grounding metrics, the top-5 passages per
   claim and the preservation check; the judge literature (Zheng, Wang, Dubois,
   Panickssery, G-Eval, Prometheus) fixed criteria, swapping, averaged passes and
   the cross-family option; the debate papers fixed agents, rounds, adaptive
   stopping and persona-based diversity; Tian/Xiong/Guo fixed top-k
   verbalisation, its low weight and Platt scaling; the prompting dossier fixed
   the data-not-instructions marking and the stable-prefix layout.
2. **Design decisions**: a declarative `VerificationPlan` separates reasoning
   from execution and is inspectable in `metadata["plan"]`; the heuristic planner
   encodes the same rules the LLM planner is prompted with, so offline and
   online runs agree; domain posture is a `VerificationPolicy` derived from the
   `DomainProfile`, so a new field changes behaviour without code; verdicts carry
   an explicit **inconclusive** state so missing inputs never masquerade as
   evidence (a rule verifier with nothing to check, a grounder with no evidence),
   and a run with no conclusive verdict is reported as unverified rather than
   accepted at the prior; every LLM output is a strict JSON schema validated and clamped in
   Python; costs are estimated before execution and degraded to fit the budget;
   proposed corrections pass through `_confirm_revision` before they touch the
   final answer; the direct `verify()` API is the `Verifier` protocol and runs
   the full plan so other models get the same guarantees.
3. **Implementation**: pure-function cores (canonicalisation, clustering and
   entropy, stopping posterior, Platt fit, fusion caps, claim splitting and
   lexical support, unit parsing, rule checks, test-summary parsing, budget
   fitting, metrics) are unit-tested without an LLM; LLM paths are covered with
   a template-dispatching `MockLLM` and `ScriptedLLM`; `MockLLM` covers the
   offline end-to-end path, domain adaptation, budget containment, protocols and
   the CLI; both examples and the CLI run with no credentials. `tests/` holds 97
   tests that complete in about five seconds, including a regression test for
   each finding of the adversarial review (vacuous rule passes, acceptance at
   the prior, prose classified as code, last-number extraction, SI prefix case,
   one-line grounding, identical debate prompts, lost CoVe verdicts).
