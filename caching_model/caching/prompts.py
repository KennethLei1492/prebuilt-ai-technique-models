"""Versioned prompt templates for the caching model.

Two templates:

* :data:`CACHE_PLAN_TEMPLATE` - the single structured planning call made by
  :meth:`caching.CachingModel.plan` when online. Stable system text (role,
  numbered rules, few-shot examples) is cache-friendly; the task, extracted
  features and domain risk arrive in the user turn.
* :data:`EQUIVALENCE_TEMPLATE` - the borderline-hit check used by
  :class:`caching.CachedLLMClient`: does the cached answer to prompt A validly
  answer prompt B?

Rules: no timestamps, ids or random values in the stable blocks; bump
``version`` whenever wording changes (caches key on the fingerprint).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import STRUCTURED_OUTPUT_RULES, PromptTemplate, sanitize_schema

from .plan import CachePlan


class EquivalenceVerdict(BaseModel):
    """Structured output of the borderline-hit equivalence check."""

    model_config = ConfigDict(extra="forbid")

    equivalent: bool = Field(description="True if the cached answer fully and correctly answers the new prompt")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability the judgement is right")
    reason: str = Field(description="One sentence; name any identifier, number or constraint that differs")


CACHE_PLAN_SCHEMA = sanitize_schema(CachePlan.model_json_schema())
EQUIVALENCE_SCHEMA = sanitize_schema(EquivalenceVerdict.model_json_schema())

_PLAN_SYSTEM = """You are the cache policy planner for an LLM pipeline. Given a prompt that will be sent to a language model, decide which response-cache tiers may serve it safely and how long an answer may be reused.

Tiers:
- exact: byte-identical (normalised) prompt repeats. Zero false-hit risk. Safe for any deterministic prompt.
- semantic: paraphrase matches by embedding similarity. Risky: embeddings blur identifiers, numbers, units and dates. Only for stable, wording-insensitive lookups.
- prefix: provider prompt caching of the stable system prefix. Always useful when a stable prefix exists.
- negative: cache deterministic failures (schema violations, refusals) briefly.

Rules:
1. If the prompt asks for diverse, random or creative output (brainstorming, variations, jokes, stories), set cacheable=false and tiers=[]; serving one cached sample would collapse variety.
2. If the answer depends on the present moment (today, now, current, latest, live data), use only the exact tier with a TTL of at most 300 seconds.
3. If the prompt contains personal or account-specific data, use only the exact tier; the namespace must be tenant-scoped.
4. If the prompt contains part numbers, requirement or clause ids, version strings, tolerances or other numbers whose small differences change the answer, keep identifier_guards=true, raise semantic_threshold to at least 0.98 and set verify_hits=true, or drop the semantic tier.
5. For high or safety-critical risk levels, raise semantic_threshold to at least 0.98, set error_budget to at most 0.01 and verify_hits=true; for safety-critical, drop the semantic tier unless the question is a pure definition lookup.
6. Prefer prefix_ttl="1h" when the stable prefix is long reference material (standards, specifications, schemas) reused over an hour; otherwise "5m". Set prewarm=true only when the prompt will be fanned out in parallel.
7. The exact tier is always listed first when cacheable. Default TTL for grounded answers over versioned material is 86400 seconds.
8. Explain the decision in one or two sentences in `reason` and list the risk flags you relied on.

""" + STRUCTURED_OUTPUT_RULES

CACHE_PLAN_TEMPLATE = PromptTemplate(
    name="caching.plan",
    version="1.0.0",
    description="Classify the cacheability of a prompt and emit a CachePlan.",
    system=_PLAN_SYSTEM,
    instructions=(
        "<task>\n{task}\n</task>\n"
        "<features>\n{features}\n</features>\n"
        "<model>{model}</model>\n"
        "<has_stable_prefix>{has_stable_prefix}</has_stable_prefix>\n"
        "<fan_out>{fan_out}</fan_out>\n"
        "Produce the CachePlan JSON object."
    ),
    examples=[
        (
            "<task>\nWhat does ISO 2768-m specify as the general tolerance for a 120 mm linear dimension?\n</task>\n"
            "<features>\n{\"lookup_like\": true, \"identifier_count\": 1, \"number_count\": 2, \"time_sensitive\": false}\n</features>\n"
            "<model>claude-opus-5-5</model>\n<has_stable_prefix>true</has_stable_prefix>\n<fan_out>false</fan_out>",
            "{\"cacheable\": true, \"tiers\": [\"exact\", \"semantic\", \"prefix\", \"negative\"], \"ttl_s\": 86400, "
            "\"semantic_threshold\": 0.98, \"error_budget\": 0.01, \"verify_hits\": true, \"identifier_guards\": true, "
            "\"namespace\": \"claude-opus-5-5:-:-:-:-:-:-:mechanical_engineering\", \"prefix_ttl\": \"1h\", \"prewarm\": false, "
            "\"negative_ttl_s\": 600, \"task_class\": \"lookup\", \"risk_flags\": [\"identifier_or_numeric_sensitive\"], "
            "\"reason\": \"Stable standards lookup; identifiers and a dimension are present so semantic hits need identical guards and a strict threshold.\", "
            "\"confidence\": 0.9}",
        ),
        (
            "<task>\nBrainstorm five different names for our new torque sensor product line.\n</task>\n"
            "<features>\n{\"randomness\": true, \"lookup_like\": false}\n</features>\n"
            "<model>claude-opus-5-5</model>\n<has_stable_prefix>true</has_stable_prefix>\n<fan_out>false</fan_out>",
            "{\"cacheable\": false, \"tiers\": [], \"ttl_s\": null, \"semantic_threshold\": 0.95, \"error_budget\": 0.02, "
            "\"verify_hits\": false, \"identifier_guards\": true, \"namespace\": \"claude-opus-5-5:-:-:-:-:-:-:general\", "
            "\"prefix_ttl\": \"5m\", \"prewarm\": false, \"negative_ttl_s\": 600, \"task_class\": \"random\", "
            "\"risk_flags\": [\"diversity_requested\"], \"reason\": \"The user wants varied creative output; a cached sample would remove the variety they asked for.\", "
            "\"confidence\": 0.9}",
        ),
    ],
    output_schema=CACHE_PLAN_SCHEMA,
)

_EQUIVALENCE_SYSTEM = """You judge whether a cached answer may be reused for a new prompt.

Rules:
1. The cached answer may be reused only if it answers the new prompt completely and correctly as written.
2. Any difference in identifiers (part numbers, requirement or clause ids), numbers, units, tolerances, versions, dates, names or constraints means the prompts are NOT equivalent.
3. A difference only in wording, word order, politeness or punctuation does not matter.
4. When unsure, answer equivalent=false; a false hit is worse than a miss.

""" + STRUCTURED_OUTPUT_RULES

EQUIVALENCE_TEMPLATE = PromptTemplate(
    name="caching.equivalence",
    version="1.0.0",
    description="Borderline semantic-hit verification.",
    system=_EQUIVALENCE_SYSTEM,
    instructions=(
        "<cached_prompt>\n{cached_prompt}\n</cached_prompt>\n"
        "<cached_answer>\n{cached_answer}\n</cached_answer>\n"
        "<new_prompt>\n{new_prompt}\n</new_prompt>\n"
        "Decide whether the cached answer may be served for the new prompt."
    ),
    output_schema=EQUIVALENCE_SCHEMA,
)

TEMPLATES: dict[str, PromptTemplate] = {t.name: t for t in (CACHE_PLAN_TEMPLATE, EQUIVALENCE_TEMPLATE)}

__all__ = [
    "CACHE_PLAN_SCHEMA",
    "CACHE_PLAN_TEMPLATE",
    "EQUIVALENCE_SCHEMA",
    "EQUIVALENCE_TEMPLATE",
    "EquivalenceVerdict",
    "TEMPLATES",
]
