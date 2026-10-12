"""Versioned prompt templates for the task decomposition model.

Every template follows the cache-aware structure from ``pipeline_core.prompts``:
stable role, rules and few-shot examples first (cacheable), the volatile task,
documents and prior results in the user turn. Bump ``version`` whenever
wording changes; downstream caches key on the template fingerprint.

Prompting conventions encoded here (prompting-techniques dossier): explicit
role and goal, numbered rules, XML-delimited inputs, explicit output
specification, examples before the volatile request, a damping rule against
over-decomposition (Anthropic), explicit independence marking for parallelism
(LLMCompiler) and compact per-node context (ReWOO). No timestamps, ids or
request-specific text appear in any stable block.
"""
from __future__ import annotations

from pipeline_core import ENGINEERING_RIGOR, STRUCTURED_OUTPUT_RULES, PromptTemplate

__all__ = [
    "PLAN_TEMPLATE",
    "DECOMPOSE_TEMPLATE",
    "CANDIDATES_TEMPLATE",
    "REFINE_TEMPLATE",
    "EVALUATE_TEMPLATE",
    "SOLVE_TEMPLATE",
    "PLAN_AND_SOLVE_TEMPLATE",
    "REPLAN_TEMPLATE",
    "SYNTHESIZE_TEMPLATE",
    "COVERAGE_JUDGE_TEMPLATE",
    "PS_PLUS_TRIGGER",
    "ENGINEERING_RIGOR",
]

PS_PLUS_TRIGGER = (
    "Let's first understand the problem, extract relevant variables and their corresponding values, and make a plan. "
    "Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and "
    "commonsense), solve the problem step by step, and show the answer."
)

_STRATEGY_CATALOGUE = """<strategies>
- direct: answer in one call. Use for tasks of one or two steps; never decompose a trivial task.
- plan_and_solve: one call that first writes a plan then executes it. Use for moderate tasks without clear structure.
- least_to_most: ordered sub-questions from easy to hard, each solved with all prior answers in context, the original task re-asked last. Use for long sequential chains (5+ steps) and when later steps need earlier results.
- dag: a dependency graph where independent sub-tasks have no dependencies and run concurrently; dependent ones reference earlier outputs. Use when several lookups or analyses are independent.
- recursive: try the whole task first, decompose only on failure, recurse to a depth bound. Use for investigative tasks whose right granularity is unknown (debugging, root cause, incidents).
- tree_of_thoughts: propose several candidate decompositions, score them, keep the best. Use when alternatives, orderings or trade-offs must be compared and intermediate states are scorable; it costs several times a single call.
</strategies>"""

PLAN_TEMPLATE = PromptTemplate(
    name="task_decomposition.plan",
    version="1.0.0",
    description="Choose the decomposition strategy and its bounds for a task.",
    system="\n\n".join(
        [
            "You are the planning stage of an engineering task-decomposition pipeline. You read a task and decide how it "
            "should be broken down and executed; you do not solve it.",
            "Rules:\n"
            "1. Prefer the simplest structure that will work. Trivial tasks must not be decomposed (strategy direct).\n"
            "2. Estimate the number of reasoning or action steps; decomposition pays off from about three steps and clearly from five.\n"
            "3. Mark work as parallel (dag) only when sub-tasks are genuinely independent; dependent work stays sequential.\n"
            "4. Choose recursive when the right granularity cannot be known before trying, and tree_of_thoughts only when "
            "alternatives must be compared and candidates can be scored.\n"
            "5. Set verification to strict for safety-critical domains, external when an independent check is warranted, "
            "criteria when sub-tasks should carry checkable criteria, none for trivial tasks.\n"
            "6. Keep depth, breadth and parallelism within the stated bounds.\n"
            "7. The rationale is one or two sentences; do not restate the task.",
            _STRATEGY_CATALOGUE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    examples=[
        (
            "<task>Convert 25 degC to kelvin.</task>\n<features>steps~1 words=5 enumeration=0 independence=0 sequence=0</features>",
            '{"strategy":"direct","max_depth":1,"max_breadth":1,"parallelism":1,"verification":"none","complexity_estimate":"trivial",'
            '"estimated_steps":1,"replan_on_failure":false,"rationale":"Single arithmetic step; decomposition would only add cost."}',
        ),
        (
            "<task>For each of the five pump test logs, extract the duty point, flag deviations from the datasheet, then summarise the fleet status.</task>\n"
            "<features>steps~5 words=24 enumeration=5 independence=1 sequence=1</features>",
            '{"strategy":"dag","max_depth":3,"max_breadth":6,"parallelism":5,"verification":"external","complexity_estimate":"medium",'
            '"estimated_steps":6,"replan_on_failure":true,"rationale":"Five independent per-log extractions can run concurrently; a final aggregation depends on all of them."}',
        ),
    ],
    instructions=(
        "<task>\n{task}\n</task>\n\n<features>\n{features}\n</features>\n\n<bounds>\n{bounds}\n</bounds>\n\n"
        "Decide the decomposition plan for this task and return it as JSON."
    ),
)

DECOMPOSE_TEMPLATE = PromptTemplate(
    name="task_decomposition.decompose",
    version="1.0.0",
    description="Produce a validated dependency graph of sub-tasks.",
    system="\n\n".join(
        [
            "You are the decomposer of an engineering task pipeline. You turn one task into a small set of self-contained "
            "sub-tasks with explicit dependencies, so that a scheduler can run independent sub-tasks concurrently and a "
            "separate solver can answer each one without seeing the others.",
            "Rules:\n"
            "1. Emit between the minimum and maximum number of sub-tasks requested; do not pad and do not merge distinct steps.\n"
            "2. Each sub-task must be solvable in isolation from its description plus its inputs; keep each one shallow "
            "(one lookup, one computation, one judgement) so that no step requires long chains of reasoning.\n"
            "3. Independent sub-tasks have an empty depends_on list. A sub-task that needs an earlier result lists that id in "
            "depends_on and references the result in an input value as $id.output.\n"
            "4. Give every sub-task an expected_output and at least one verification criterion that an external reviewer or "
            "tool could check (a unit, a range, a cross-reference, a test, a schema).\n"
            "5. Use tool_hint retrieve for sub-tasks that look information up, compute for numeric work, verify for checks, "
            "aggregate for combining earlier outputs, llm otherwise.\n"
            "6. Apply the domain decomposition hints when they are relevant; they describe how experts split work in this field.\n"
            "7. Ids are short and stable (t1, t2, ...). Do not create cycles. Do not repeat a sub-task.\n"
            "8. Do not include a final 'write the answer' sub-task; synthesis is performed separately with the original task in context.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    examples=[
        (
            "<task>Check the pump station design: verify hydraulic sizing, verify the motor rating against the duty point, and list open risks.</task>\n<bounds>min=3 max=6</bounds>",
            '{"tasks":[{"id":"t1","description":"Verify the hydraulic sizing of the pump station against the design flow and head.",'
            '"depends_on":[],"inputs":[],"expected_output":"Required flow and head with a pass/fail against the selected pump curve","verification_criteria":["Flow and head carry units","Duty point lies on the pump curve"],"tool_hint":"compute","complexity":"medium"},'
            '{"id":"t2","description":"Verify the motor rating against the pump duty point power including service factor.","depends_on":[],"inputs":[],'
            '"expected_output":"Required shaft power, motor rating and margin","verification_criteria":["Power in kW","Margin stated as a percentage"],"tool_hint":"compute","complexity":"medium"},'
            '{"id":"t3","description":"List open design risks from the hydraulic and motor checks.","depends_on":["t1","t2"],'
            '"inputs":[{"name":"hydraulic_check","value":"$t1.output"},{"name":"motor_check","value":"$t2.output"}],'
            '"expected_output":"Ranked list of risks with owner and mitigation","verification_criteria":["Each risk cites the check that raised it"],"tool_hint":"aggregate","complexity":"low"}],'
            '"notes":"Datasheet values assumed at rated temperature."}',
        ),
    ],
    instructions=(
        "<task>\n{task}\n</task>\n\n<bounds>\n{bounds}\n</bounds>\n\n<domain_hints>\n{hints}\n</domain_hints>\n\n"
        "<inputs>\n{inputs}\n</inputs>\n\n{documents}\n\n{feedback}\n\nDecompose the task into sub-tasks and return the graph as JSON."
    ),
)

CANDIDATES_TEMPLATE = PromptTemplate(
    name="task_decomposition.candidates",
    version="1.0.0",
    description="Propose several alternative decompositions for Tree-of-Thoughts selection.",
    system="\n\n".join(
        [
            "You propose alternative decompositions of an engineering task so that a separate evaluator can choose the "
            "best one. Candidates must differ materially: in granularity, in ordering, or in which sub-problems are solved "
            "first. Each candidate follows the same rules as a single decomposition: self-contained shallow sub-tasks, "
            "explicit dependencies, $id.output references, checkable verification criteria, no cycles, no duplicates.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<task>\n{task}\n</task>\n\n<bounds>\n{bounds}\n</bounds>\n\n<domain_hints>\n{hints}\n</domain_hints>\n\n{documents}\n\n"
        "Propose {n} materially different candidate decompositions and return them as JSON."
    ),
)

REFINE_TEMPLATE = PromptTemplate(
    name="task_decomposition.refine",
    version="1.0.0",
    description="Tree-of-Thoughts expansion: propose improved variants of one promising candidate decomposition.",
    system="\n\n".join(
        [
            "You expand one promising decomposition of an engineering task into a few improved variants so that a "
            "separate evaluator can choose among them. Keep what the parent gets right; fix what it misses: uncovered "
            "requirements, steps that are not solvable in isolation, dependencies that do not match the real data flow, "
            "redundant or trivial steps. Each variant follows the same rules as a single decomposition: self-contained "
            "shallow sub-tasks, explicit dependencies, $id.output references, checkable verification criteria, no cycles, "
            "no duplicates.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<task>\n{task}\n</task>\n\n<bounds>\n{bounds}\n</bounds>\n\n<domain_hints>\n{hints}\n</domain_hints>\n\n{documents}\n\n"
        "<parent_candidate>\n{parent}\n</parent_candidate>\n\n"
        "Propose {n} improved variants of the parent candidate and return them as JSON."
    ),
)

EVALUATE_TEMPLATE = PromptTemplate(
    name="task_decomposition.evaluate",
    version="1.0.0",
    description="Score candidate decompositions (Tree-of-Thoughts state evaluation).",
    system="\n\n".join(
        [
            "You evaluate candidate decompositions of a task. For each candidate judge whether executing its sub-tasks "
            "would solve the original task completely and correctly. Score 1.0 when success is sure, around 0.6 when "
            "likely, and 0.0 when a candidate is impossible (misses a requirement, has an unsolvable step, or depends on "
            "information that no sub-task produces). Penalise missing coverage more than extra steps.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<task>\n{task}\n</task>\n\n<candidates>\n{candidates}\n</candidates>\n\n{framing}\n\nScore every candidate and return JSON."
    ),
)

SOLVE_TEMPLATE = PromptTemplate(
    name="task_decomposition.solve",
    version="1.0.0",
    description="Solve one sub-task with its inputs, prior results and evidence.",
    system="\n\n".join(
        [
            "You solve one sub-task of a larger engineering task. You receive only what this sub-task needs: its "
            "description, its inputs, the outputs of the sub-tasks it depends on, and any documents. Answer it completely "
            "and state what you could not determine.",
            "Rules:\n"
            "1. Use the provided inputs and documents; cite documents by id in square brackets when they support a claim.\n"
            "2. Do not invent values. If information is missing, say so in gaps and lower your confidence; set status to "
            "failed when a correct answer is not possible from the inputs.\n"
            "3. Report criteria_met with one boolean per verification criterion, in order.\n"
            "4. Keep units explicit and calculations visible.\n"
            "5. Confidence is your probability that the answer is correct and complete.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<sub_task id=\"{task_id}\">\n{description}\n</sub_task>\n\n<expected_output>\n{expected_output}\n</expected_output>\n\n"
        "<verification_criteria>\n{criteria}\n</verification_criteria>\n\n<inputs>\n{inputs}\n</inputs>\n\n"
        "<prior_results>\n{prior_results}\n</prior_results>\n\n{documents}\n\n<original_task>\n{root_task}\n</original_task>\n\n"
        "Solve the sub-task and return JSON."
    ),
)

PLAN_AND_SOLVE_TEMPLATE = PromptTemplate(
    name="task_decomposition.plan_and_solve",
    version="1.0.0",
    description="Zero-shot Plan-and-Solve (PS+) single call.",
    system="\n\n".join(
        [
            "You solve engineering tasks by planning before executing. " + PS_PLUS_TRIGGER,
            "Rules:\n"
            "1. Work through the plan before writing the answer; the answer field must contain the plan followed by the result.\n"
            "2. Keep units explicit; cite documents by id when they support a claim; do not invent values.\n"
            "3. List missing information in gaps and reflect it in confidence.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n\n<inputs>\n{inputs}\n</inputs>\n\n{documents}\n\nPlan, then solve the task and return JSON.",
)

REPLAN_TEMPLATE = PromptTemplate(
    name="task_decomposition.replan",
    version="1.0.0",
    description="Reflect on externally observed failures and revise the failed part of the plan.",
    system="\n\n".join(
        [
            "You are the replanner of an engineering task pipeline. Some sub-tasks failed according to external signals "
            "(solver errors, failed checks, verifier verdicts). Write a short reflection (root cause, what to change) and "
            "propose replacement sub-tasks for the failed ones only. Completed sub-tasks keep their outputs and may be "
            "referenced by id with $id.output.",
            "Rules:\n"
            "1. Change the approach, not just the wording: split an overloaded step, add a missing lookup, or route to a "
            "different kind of step.\n"
            "2. Replacement sub-tasks follow the usual rules: self-contained, shallow, explicit depends_on, checkable criteria, no cycles.\n"
            "3. Set give_up to true when the failure is caused by missing inputs that no sub-task can obtain.\n"
            "4. Use fresh ids that do not collide with existing ones.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<task>\n{task}\n</task>\n\n<current_graph>\n{graph}\n</current_graph>\n\n<completed>\n{completed}\n</completed>\n\n"
        "<failures>\n{failures}\n</failures>\n\n<memory>\n{memory}\n</memory>\n\n<bounds>\n{bounds}\n</bounds>\n\n"
        "Reflect on the failures and return the revision as JSON."
    ),
)

SYNTHESIZE_TEMPLATE = PromptTemplate(
    name="task_decomposition.synthesize",
    version="1.0.0",
    description="Compose sub-task results into the final answer with traceability.",
    system="\n\n".join(
        [
            "You compose the results of several sub-tasks into one final answer to the original task. Sub-tasks were "
            "solved independently; composing them correctly is your responsibility.",
            "Rules:\n"
            "1. Answer the original task directly and completely, using only the sub-task results and documents provided.\n"
            "2. Resolve conflicts between sub-answers explicitly; when they cannot be resolved, list them under unresolved.\n"
            "3. Record traceability: for each sub-task, what it contributed. Do not cite a sub-task that contributed nothing.\n"
            "4. Confidence is your probability that the composed answer is correct and complete, never higher than the weakest essential sub-answer.\n"
            "5. Keep units explicit and keep the structure the task asked for.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<original_task>\n{task}\n</original_task>\n\n<sub_task_results>\n{results}\n</sub_task_results>\n\n{documents}\n\n"
        "Compose the final answer to the original task and return JSON."
    ),
)

COVERAGE_JUDGE_TEMPLATE = PromptTemplate(
    name="task_decomposition.coverage_judge",
    version="1.0.0",
    description="LLM judge for plan coverage: which requirements of the task does the graph address?",
    system="\n\n".join(
        [
            "You audit a decomposition of an engineering task. First list the atomic requirements the task states or "
            "clearly implies (one per deliverable, check, computation or constraint). Then, for each requirement, list the "
            "ids of the sub-tasks that address it, or an empty list when no sub-task does. You judge coverage only; do not "
            "solve the task and do not reward sub-tasks that are unrelated to the task.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions=(
        "<task>\n{task}\n</task>\n\n<graph>\n{graph}\n</graph>\n\nList the requirements and the sub-task ids covering each, as JSON."
    ),
)
