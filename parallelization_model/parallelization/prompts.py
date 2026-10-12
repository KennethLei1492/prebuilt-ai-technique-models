"""Versioned prompt templates and the structured-output schemas they fill.

Every template keeps the stable text (role, numbered rules, output rules) in
``system`` so it is byte-identical across all branches of a fan-out and lands
in the cacheable prefix; everything that varies per request (task, shared
context, branch-specific suffix) is in ``instructions`` and rendered into the
user turn, with the branch-specific part *last* so sibling branches share the
longest possible prefix (SGLang / Anthropic prompt-caching guidance).

Bump ``version`` whenever wording changes: caches key on the fingerprint.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import GROUNDING_RULES, STRUCTURED_OUTPUT_RULES, PromptTemplate

ConfidenceLabel = Literal["high", "medium", "low"]

_CONF_RULE = "Report confidence as one of high, medium or low: high only when the answer is complete and well supported; low when evidence is thin or the task is ambiguous."

# --------------------------------------------------------------------------
# Output schemas (validated with model_validate after every structured call)
# --------------------------------------------------------------------------


class AnswerOutput(BaseModel):
    """Single sequential answer."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The complete answer to the task")
    confidence: ConfidenceLabel = Field(description="high, medium or low")


class SkeletonOutput(BaseModel):
    """Skeleton-of-Thought stage 1: concise, independent answer points."""

    model_config = ConfigDict(extra="forbid")

    points: list[str] = Field(min_length=3, max_length=10, description="3-10 short point headers, each self-contained and independent of the others")


class ExpansionOutput(BaseModel):
    """Skeleton-of-Thought stage 2: one expanded point."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The expanded point, concise and self-contained")


class BranchOutput(BaseModel):
    """Branch-Solve-Merge branch stage."""

    model_config = ConfigDict(extra="forbid")

    branches: list[str] = Field(min_length=3, max_length=6, description="3-6 independent sub-tasks or evaluation criteria that together cover the task")


class BranchSolution(BaseModel):
    """Structured solution to one branch / partition (also the sectioning worker output)."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Answer for this sub-task only")
    evidence: list[str] = Field(default_factory=list, description="Short quotes or facts the answer rests on")
    rationale: str = Field(description="Why the answer follows from the evidence")
    confidence: ConfidenceLabel


class MergeOutput(BaseModel):
    """Merge / synthesis of several branch solutions."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The merged final answer")
    unresolved: list[str] = Field(default_factory=list, description="Conflicts or gaps that could not be resolved")
    confidence: ConfidenceLabel


class MapRecord(BaseModel):
    """LLM x MapReduce structured information protocol for one chunk."""

    model_config = ConfigDict(extra="forbid")

    extracted_evidence: list[str] = Field(default_factory=list, description="Quotes from this chunk relevant to the task")
    rationale: str = Field(description="How the evidence bears on the task")
    answer: str = Field(description="Partial answer from this chunk alone, or 'no relevant information'")
    confidence: ConfidenceLabel


class ReduceOutput(BaseModel):
    """Confidence-aware reduction of map records."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Combined answer that resolves conflicts using the records' confidence")
    conflicts: list[str] = Field(default_factory=list, description="Contradictions between records and how they were resolved")
    confidence: ConfidenceLabel


class SampleOutput(BaseModel):
    """One self-consistency sample: reasoning plus an extractable final answer."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="Step-by-step reasoning")
    final_answer: str = Field(description="The final answer only, as short as possible (number with unit, label, yes/no, or one sentence)")


class USCOutput(BaseModel):
    """Universal Self-Consistency selection."""

    model_config = ConfigDict(extra="forbid")

    selected_index: int = Field(ge=1, description="1-based index of the most consistent candidate")
    justification: str
    confidence: ConfidenceLabel


class ProposerOutput(BaseModel):
    """Mixture-of-Agents proposer."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="A complete, improved response")


class AggregateOutput(BaseModel):
    """Mixture-of-Agents aggregator."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Critically synthesised final response")
    corrections: list[str] = Field(default_factory=list, description="Errors found in proposals and how they were corrected")
    confidence: ConfidenceLabel


class VerifyOutput(BaseModel):
    """Speculative pattern: is the fast answer acceptable?"""

    model_config = ConfigDict(extra="forbid")

    acceptable: bool
    issues: list[str] = Field(default_factory=list)


class CoherenceOutput(BaseModel):
    """Optional light coherence pass after skeleton expansion."""

    model_config = ConfigDict(extra="forbid")

    text: str


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

PLANNER = PromptTemplate(
    name="parallelization.planner",
    version="1.0.0",
    description="Parallelism router: picks the execution pattern and partitions for a task.",
    system="\n".join(
        [
            "You are the parallelism router of an AI pipeline. You read a task and decide how to execute it so that independent work runs concurrently and dependent work stays sequential.",
            "Rules:",
            "1. Choose exactly one pattern using this routing table:",
            "   sequential - the task is a derivation, procedure, narrative or code where later steps depend on earlier ones; parallel generation would hurt quality.",
            "   sectioning - the task contains enumerable, independent sub-questions or items; list them as partitions.",
            "   skeleton_of_thought - a long-form answer whose points are independent (options, risks, considerations, checklist, review).",
            "   voting - a single short verifiable answer (number, classification, yes/no, calculation) where several independent attempts can be compared.",
            "   map_reduce - the input is a long document or many records that must be processed chunk by chunk.",
            "   branch_solve_merge - an evaluation, assessment or multi-constraint generation that decomposes into criteria.",
            "   ensemble - a high-value free-form answer where several proposals and a critical synthesis are worth the cost.",
            "2. Partitions are required for sectioning only; each partition has a unique short id, a self-contained task, and depends_on ids (usually empty).",
            "3. Never split work whose parts depend on each other. Prefer the simplest pattern that fits; parallelism costs tokens.",
            "4. Set verifiability to exact_match when answers can be compared literally, verifiable when an external check exists, free_form otherwise.",
            "5. Respect the stated limits on partitions and samples.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="\n".join(
        [
            "<task>\n{task}\n</task>",
            "<inputs_summary>\n{inputs_summary}\n</inputs_summary>",
            "<task_features>\n{features}\n</task_features>",
            "<limits>\n{limits}\n</limits>",
            "Decide the execution plan.",
        ]
    ),
)

SEQUENTIAL = PromptTemplate(
    name="parallelization.sequential",
    version="1.0.0",
    description="Single complete answer (the no-parallelism path).",
    system="\n".join(
        [
            "You are a careful expert assistant. Answer the task completely and precisely.",
            "Rules:",
            "1. Work through the task in order; keep dependent steps in sequence.",
            "2. State assumptions explicitly and prefer uncertainty over guessing.",
            "3. " + GROUNDING_RULES,
            "4. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>",
)

SKELETON = PromptTemplate(
    name="parallelization.skeleton",
    version="1.0.0",
    description="Skeleton-of-Thought stage 1.",
    system="\n".join(
        [
            "You are drafting the skeleton of a long-form answer.",
            "Rules:",
            "1. Produce between {min_points} and {max_points} short point headers (3-8 words each).",
            "2. Points must be independent: each can be expanded without reading the others.",
            "3. Together the points must cover the task completely with no overlap.",
            "4. Do not expand the points; headers only.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>",
)

EXPAND = PromptTemplate(
    name="parallelization.expand",
    version="1.0.0",
    description="Skeleton-of-Thought stage 2: expand one point.",
    system="\n".join(
        [
            "You are expanding one point of a skeleton into its final text.",
            "Rules:",
            "1. Write only about the assigned point; do not repeat other points.",
            "2. Keep it under {max_words} words, concrete and self-contained.",
            "3. Start directly with content; no preamble, no heading.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>\n<skeleton>\n{skeleton}\n</skeleton>\n<assigned_point index=\"{index}\">\n{point}\n</assigned_point>",
)

COHERENCE = PromptTemplate(
    name="parallelization.coherence",
    version="1.0.0",
    description="Optional light coherence pass over the concatenated expansions.",
    system="\n".join(
        [
            "You lightly edit a document assembled from independently written sections.",
            "Rules:",
            "1. Fix transitions, repeated phrases and inconsistent terminology only.",
            "2. Do not add, remove or reorder substantive content.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n<draft>\n{draft}\n</draft>",
)

BRANCH = PromptTemplate(
    name="parallelization.branch",
    version="1.0.0",
    description="Branch-Solve-Merge: propose independent sub-tasks or criteria.",
    system="\n".join(
        [
            "You decompose a task into independent branches that can be solved in parallel and merged.",
            "Rules:",
            "1. Propose between {min_branches} and {max_branches} branches.",
            "2. For evaluation tasks, branches are evaluation criteria; for generation tasks, branches are constraints or sections.",
            "3. Branches must not depend on each other and must jointly cover the task.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>",
)

SOLVE = PromptTemplate(
    name="parallelization.solve",
    version="1.0.0",
    description="Solve one branch or partition with evidence and rationale.",
    system="\n".join(
        [
            "You solve exactly one sub-task of a larger task.",
            "Rules:",
            "1. Address only the assigned sub-task; other branches are handled elsewhere.",
            "2. Quote the evidence you rely on and give the rationale.",
            "3. " + GROUNDING_RULES,
            "4. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<overall_task>\n{task}\n</overall_task>\n<sub_task id=\"{branch_id}\">\n{branch}\n</sub_task>",
)

MERGE = PromptTemplate(
    name="parallelization.merge",
    version="1.0.0",
    description="Merge branch solutions into one answer.",
    system="\n".join(
        [
            "You merge independently produced partial answers into one final answer.",
            "Rules:",
            "1. Preserve every supported finding; resolve conflicts by preferring higher-confidence, better-evidenced solutions.",
            "2. If a branch is marked missing, say what is missing rather than inventing it.",
            "3. Keep the structure of the original task (criteria, sections or sub-questions) visible in the answer.",
            "4. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n<solutions>\n{solutions}\n</solutions>\n{notice}",
)

MAP = PromptTemplate(
    name="parallelization.map",
    version="1.0.0",
    description="Map over one chunk with the structured information protocol.",
    system="\n".join(
        [
            "You process one chunk of a long input on behalf of a map-reduce pipeline.",
            "Rules:",
            "1. Extract only evidence present in this chunk; quote it.",
            "2. If the chunk is irrelevant, say 'no relevant information' with low confidence.",
            "3. Your rationale and confidence will be compared against other chunks, so be calibrated.",
            "4. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n<global_summary>\n{summary}\n</global_summary>\n<chunk index=\"{index}\" of=\"{total}\">\n{chunk}\n</chunk>",
)

REDUCE = PromptTemplate(
    name="parallelization.reduce",
    version="1.0.0",
    description="Confidence-aware reduce over map records.",
    system="\n".join(
        [
            "You combine structured records produced from chunks of a long input.",
            "Rules:",
            "1. Weigh records by their stated confidence and evidence; discard 'no relevant information' records.",
            "2. Name contradictions between records and say how you resolved them.",
            "3. Cite chunk indices in square brackets, for example [chunk-3].",
            "4. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n<records>\n{records}\n</records>",
)

SAMPLE = PromptTemplate(
    name="parallelization.sample",
    version="1.0.0",
    description="One self-consistency sample with an extractable final answer.",
    system="\n".join(
        [
            "You solve a task that has a single checkable answer.",
            "Rules:",
            "1. Reason step by step, then state the final answer alone in final_answer.",
            "2. final_answer must be minimal: a number with its unit, a label, yes/no, or one short sentence.",
            "3. Keep units explicit and consistent.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>\n<approach>\n{framing}\n</approach>",
)

#: Framing variants give independent samples diverse reasoning paths without sampling knobs.
SAMPLE_FRAMINGS: tuple[str, ...] = (
    "Solve it directly, checking each intermediate value.",
    "First restate the givens and the unknown, then solve.",
    "Solve it, then verify by working backwards from your answer.",
    "Consider the most common mistake for this kind of task and avoid it while solving.",
    "Solve it with a different method than the most obvious one if a second method exists.",
    "Estimate the order of magnitude first, then compute precisely.",
    "Identify every assumption, then solve under those assumptions.",
    "Solve as a careful reviewer would, double-checking units and signs.",
    "Break the problem into the smallest sub-steps and solve each.",
    "Solve it as if explaining to a colleague, justifying each step.",
)

USC_SELECT = PromptTemplate(
    name="parallelization.usc_select",
    version="1.0.0",
    description="Universal Self-Consistency: select the most consistent candidate.",
    system="\n".join(
        [
            "You are shown several candidate answers to the same task produced independently.",
            "Rules:",
            "1. Select the candidate most consistent with the majority of the others in substance.",
            "2. Do not write a new answer; choose one by index.",
            "3. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n<candidates>\n{candidates}\n</candidates>",
)

PROPOSE = PromptTemplate(
    name="parallelization.propose",
    version="1.0.0",
    description="Mixture-of-Agents proposer (layer 1 has no previous outputs).",
    system="\n".join(
        [
            "You are one of several proposers answering the same task.",
            "Rules:",
            "1. Give a complete, high-quality answer.",
            "2. When previous proposals are shown, use their strengths and correct their errors; do not merely average them.",
            "3. " + GROUNDING_RULES,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>\n<previous_proposals>\n{previous}\n</previous_proposals>\n<perspective>\n{perspective}\n</perspective>",
)

#: Perspectives diversify homogeneous proposers (Self-MoA) without changing the model.
PROPOSER_PERSPECTIVES: tuple[str, ...] = (
    "Prioritise completeness: cover every part of the task.",
    "Prioritise correctness: verify each claim before stating it.",
    "Prioritise clarity: structure the answer so a reviewer can check it quickly.",
    "Prioritise risks: surface what could go wrong and how to detect it.",
    "Prioritise practicality: concrete steps, numbers and references.",
    "Prioritise alternatives: compare options before recommending one.",
)

AGGREGATE = PromptTemplate(
    name="parallelization.aggregate",
    version="1.0.0",
    description="Mixture-of-Agents aggregator.",
    system="\n".join(
        [
            "You are the final aggregator over several proposals for the same task.",
            "Rules:",
            "1. Critically evaluate the proposals; identify errors, omissions and contradictions.",
            "2. Synthesise one answer that keeps what is correct and well supported; do not average or list the proposals.",
            "3. Record each correction you made.",
            "4. " + _CONF_RULE,
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<context>\n{context}\n</context>\n<task>\n{task}\n</task>\n<proposals>\n{proposals}\n</proposals>",
)

VERIFY_FAST = PromptTemplate(
    name="parallelization.verify_fast",
    version="1.0.0",
    description="Speculative execution: check whether the fast-path answer is acceptable.",
    system="\n".join(
        [
            "You check whether a quickly produced answer is acceptable as the final answer.",
            "Rules:",
            "1. Accept only if the answer is complete, consistent with the task and free of visible errors.",
            "2. List concrete issues when rejecting.",
            STRUCTURED_OUTPUT_RULES,
        ]
    ),
    instructions="<task>\n{task}\n</task>\n<candidate_answer>\n{answer}\n</candidate_answer>",
)

def system_blocks(template: PromptTemplate, **values: object) -> list[str]:
    """Render a template's stable system blocks, filling configuration placeholders.

    :class:`PromptTemplate.render` substitutes placeholders only in the user
    instructions. Several templates here carry configuration constants in the
    system text (``{min_points}``, ``{max_words}`` ...); those are stable for
    a deployment, so substituting them keeps the block byte-identical across
    every branch of a fan-out and therefore cacheable.
    """
    blocks = template.render(extra_system=(), **{k: "" for k in template.placeholders()}).system_blocks
    out: list[str] = []
    for block in blocks:
        for k, v in values.items():
            block = block.replace("{" + k + "}", str(v))
        out.append(block)
    return out


ALL_TEMPLATES: tuple[PromptTemplate, ...] = (
    PLANNER,
    SEQUENTIAL,
    SKELETON,
    EXPAND,
    COHERENCE,
    BRANCH,
    SOLVE,
    MERGE,
    MAP,
    REDUCE,
    SAMPLE,
    USC_SELECT,
    PROPOSE,
    AGGREGATE,
    VERIFY_FAST,
)
