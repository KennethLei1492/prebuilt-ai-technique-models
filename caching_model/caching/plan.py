"""Cache plan schema and the heuristic planner.

The planner is the "cache policy engine" the dossier calls for: it reads the
natural-language task (and the domain profile) and decides which tiers apply,
which TTL, which semantic threshold and error budget, and which namespace.
Per the dossier's architecture recommendation 2, the planner is a cheap,
deterministic rule engine first; :class:`caching.CachingModel.plan` adds one
optional structured LLM call online and reconciles it against these rules.

Decision rules (heuristic)
--------------------------
1. **Randomness / diversity requested** (brainstorm, random, "N different",
   creative, surprise) -> not cacheable (GPTCache bypass knob rationale).
2. **Time-sensitive** (today, now, current, latest, real-time, "as of",
   "status of") -> exact tier only, short TTL (``time_sensitive_ttl_s``), no
   semantic tier.
3. **Personal data markers** (my, our account, user id, "my email", "for me")
   -> exact tier only, tenant-scoped namespace, moderate TTL.

The vocabularies avoid single words that are ordinary engineering terms
("roll" axis, "unique" constraint, design "alternatives", MAC "address",
HTTP "status" code, "personal" protective equipment, "tell me"); such words
only count inside the multi-word phrases listed in the sets below.
4. **Identifier / numeric density** (part numbers, clause ids, tolerances) ->
   semantic tier only with guards and the strict threshold; disabled when the
   profile is high risk.
5. **Risk level** - ``safety_critical`` disables the semantic tier unless the
   config allows it; ``high`` uses the strict threshold, the strict error
   budget and mandatory verification of borderline hits.
6. **Prefix tier** is recommended whenever a stable system prefix exists; the
   provider TTL is 1 h for domains with long, stable reference material
   (standards, specifications) and 5 min otherwise.
"""
from __future__ import annotations

import hashlib
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import DomainProfile, LLMRequest, TaskRequest, get_profile, tokenize_words

from .config import CachingConfig
from .keys import extract_guards, namespace_for

CacheTier = Literal["exact", "semantic", "prefix", "negative"]
TaskClass = Literal["lookup", "analysis", "generative", "time_sensitive", "personal", "random"]
PrefixTTLChoice = Literal["5m", "1h", "none"]

TIME_SENSITIVE_WORDS = {
    "today", "tonight", "tomorrow", "yesterday", "now", "currently", "current", "latest", "newest", "recent",
    "live", "real-time", "realtime", "right now", "this week", "this month", "this year", "as of", "so far",
    "at the moment", "up to date", "breaking", "status of", "current status", "weather", "stock price",
    "exchange rate",
}
PERSONAL_WORDS = {
    "my", "mine", "our account", "my account", "user id", "customer id", "my email", "my e-mail", "my phone",
    "my address", "home address", "phone number of", "password", "ssn", "date of birth", "personal data",
    "personal information", "for me", "i am", "i'm", "we are",
}
RANDOM_WORDS = {
    "random", "randomly", "brainstorm", "brainstorming", "creative", "surprise me", "shuffle", "roll a die",
    "roll the dice", "different ideas", "unique ideas", "novel ideas", "variety of ideas", "generate 3",
    "generate 5", "generate 10", "poem", "write a story", "short story", "joke", "diverse",
}
LOOKUP_WORDS = {
    "what is", "what does", "what are", "define", "definition", "explain", "meaning", "require", "requires",
    "requirement", "clause", "section", "standard", "specify", "specifies", "which", "how many", "list", "summarize",
    "summarise", "according to", "look up", "lookup",
}
ANALYSIS_WORDS = {
    "analyze", "analyse", "compare", "evaluate", "assess", "review", "design", "derive", "calculate", "compute",
    "estimate", "diagnose", "root cause", "why", "trade-off", "tradeoff", "recommend", "plan", "verify",
}
_NUM_WORDS = re.compile(r"\b\d+(?:\.\d+)?\b")


class CacheabilityFeatures(BaseModel):
    """Features extracted from the task text that drive the heuristic planner."""

    model_config = ConfigDict(extra="forbid")

    length_words: int = 0
    time_sensitive: bool = False
    personal: bool = False
    randomness: bool = False
    lookup_like: bool = False
    analysis_like: bool = False
    identifier_count: int = 0
    number_count: int = 0
    identifier_density: float = Field(default=0.0, description="identifiers+numbers per word")
    matched_terms: list[str] = Field(default_factory=list)
    guards: list[str] = Field(default_factory=list)

    def conflicts(self) -> list[str]:
        """Signals that pull in different directions (lower planner confidence)."""
        out: list[str] = []
        if self.lookup_like and self.time_sensitive:
            out.append("lookup-like but time-sensitive")
        if self.lookup_like and self.randomness:
            out.append("lookup-like but asks for diversity")
        if self.analysis_like and self.randomness:
            out.append("analysis but asks for diversity")
        if self.personal and self.lookup_like:
            out.append("lookup-like but personalised")
        return out


def analyze_task(text: str) -> CacheabilityFeatures:
    """Pure feature extraction from a prompt or task string."""
    lowered = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    words = tokenize_words(text)

    def any_in(vocab: set[str]) -> list[str]:
        hits = []
        for term in vocab:
            pat = r"(?<![a-z0-9'])" + re.escape(term) + r"(?![a-z0-9])"
            if re.search(pat, lowered):
                hits.append(term)
        return sorted(hits)

    ts = any_in(TIME_SENSITIVE_WORDS)
    pe = any_in(PERSONAL_WORDS)
    ra = any_in(RANDOM_WORDS)
    lo = any_in(LOOKUP_WORDS)
    an = any_in(ANALYSIS_WORDS)
    guards = extract_guards(text)
    numbers = _NUM_WORDS.findall(text)
    identifiers = [g for g in guards if not re.fullmatch(r"\d+(?:\.\d+)?", g)]
    n_words = max(1, len(words))
    return CacheabilityFeatures(
        length_words=len(words),
        time_sensitive=bool(ts),
        personal=bool(pe),
        randomness=bool(ra),
        lookup_like=bool(lo),
        analysis_like=bool(an),
        identifier_count=len(identifiers),
        number_count=len(numbers),
        identifier_density=round((len(identifiers) + len(numbers)) / n_words, 3),
        matched_terms=ts + pe + ra + lo + an,
        guards=sorted(guards),
    )


class CachePlan(BaseModel):
    """What the caching model decided for a prompt.

    Produced by :class:`HeuristicPlanner` offline or by one structured LLM
    call online (then reconciled with the heuristics). Attached to
    ``TaskResult.metadata["plan"]`` and consumed by :class:`caching.CachedLLMClient`.
    """

    model_config = ConfigDict(extra="forbid")

    cacheable: bool = Field(description="Whether any response cache tier should serve this prompt")
    tiers: list[CacheTier] = Field(default_factory=list, description="Tiers to consult, in order")
    ttl_s: Optional[float] = Field(default=None, ge=0, description="TTL in seconds; null = no expiry; 0 = do not retain (nothing is stored)")
    semantic_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Cosine threshold for the semantic tier")
    error_budget: float = Field(default=0.02, ge=0.0, le=1.0, description="Maximum tolerated incorrect semantic hit rate")
    verify_hits: bool = Field(default=True, description="Run an LLM equivalence check on borderline semantic hits")
    identifier_guards: bool = Field(default=True, description="Require identical identifiers/numbers for semantic hits")
    namespace: str = Field(default="default", description="Key namespace: model, template, corpus version, tenant")
    prefix_ttl: PrefixTTLChoice = Field(default="5m", description="Provider prefix-cache TTL to request")
    prewarm: bool = Field(default=False, description="Warm the provider prefix cache before a fan-out")
    negative_ttl_s: float = Field(default=600.0, ge=0, description="TTL for cached deterministic failures")
    task_class: TaskClass = Field(default="lookup")
    risk_flags: list[str] = Field(default_factory=list, description="Why caution is warranted (time_sensitive, personal, numeric, ...)")
    reason: str = Field(default="", description="One or two sentences explaining the decision")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Planner's confidence that the plan is safe and useful")

    def uses(self, tier: CacheTier) -> bool:
        return self.cacheable and tier in self.tiers


class HeuristicPlanner:
    """Deterministic cacheability classifier used offline and as the online floor."""

    def __init__(self, config: CachingConfig | None = None) -> None:
        self.config = config or CachingConfig()

    # ------------------------------------------------------------- helpers
    def _namespace(self, model: str, profile: DomainProfile, *, tenant: str | None, prompt_version: str | None, corpus_version: str | None) -> str:
        # the tenant salt is a secret-like isolation value: only a hash of it may appear in namespaces,
        # tags, reports and traces (the raw salt is folded into the exact key's hash chain instead)
        salt = self.config.tenant_salt
        tenant_segment = tenant or ("salt-" + hashlib.sha256(salt.encode("utf-8")).hexdigest()[:12] if salt else None)
        return namespace_for(model, prompt_version=prompt_version, corpus_version=corpus_version, tenant=tenant_segment) + f":{profile.name}"

    @staticmethod
    def _long_stable_reference(profile: DomainProfile) -> bool:
        return bool(profile.standards) and bool(profile.document_types)

    # ---------------------------------------------------------------- plan
    def plan_text(
        self,
        text: str,
        profile: DomainProfile,
        *,
        model: str = "default",
        tenant: str | None = None,
        prompt_version: str | None = None,
        corpus_version: str | None = None,
        has_stable_prefix: bool = True,
        fan_out: bool = False,
        features: CacheabilityFeatures | None = None,
    ) -> CachePlan:
        cfg = self.config
        f = features or analyze_task(text)
        strict = profile.risk_level in ("high", "safety_critical")
        risk_flags: list[str] = []
        tiers: list[CacheTier] = []
        reason_parts: list[str] = []

        if f.randomness:
            risk_flags.append("diversity_requested")
            return CachePlan(
                cacheable=False,
                tiers=[],
                ttl_s=None,
                semantic_threshold=cfg.semantic_threshold,
                error_budget=cfg.semantic_error_budget,
                verify_hits=False,
                identifier_guards=cfg.identifier_guards,
                namespace=self._namespace(model, profile, tenant=tenant, prompt_version=prompt_version, corpus_version=corpus_version),
                prefix_ttl="5m" if has_stable_prefix else "none",
                prewarm=fan_out and has_stable_prefix,
                negative_ttl_s=cfg.negative_ttl_s,
                task_class="random",
                risk_flags=risk_flags,
                reason="The prompt asks for diverse or random output; serving a cached sample would collapse variety (GPTCache bypass rationale). Only provider prefix caching of the stable system prompt applies.",
                confidence=0.85 if not f.conflicts() else 0.6,
            )

        task_class: TaskClass
        ttl: Optional[float]
        semantic_ok = cfg.semantic_enabled
        threshold = cfg.semantic_threshold
        budget = cfg.semantic_error_budget
        verify = cfg.verify_borderline_hits

        if f.time_sensitive:
            task_class = "time_sensitive"
            ttl = cfg.time_sensitive_ttl_s
            semantic_ok = False
            risk_flags.append("time_sensitive")
            reason_parts.append(f"time-sensitive wording ({', '.join(t for t in f.matched_terms if t in TIME_SENSITIVE_WORDS)}) limits caching to exact matches with a {int(ttl)}s TTL")
        elif f.personal:
            task_class = "personal"
            ttl = cfg.personal_ttl_s
            semantic_ok = False
            risk_flags.append("personal_data")
            reason_parts.append("personal-data markers restrict caching to exact matches inside a tenant-scoped namespace")
        elif f.analysis_like and not f.lookup_like:
            task_class = "analysis"
            ttl = cfg.default_ttl_s
            reason_parts.append("analytical task: deterministic repeats hit the exact tier; semantic reuse only with guards")
        elif f.lookup_like:
            task_class = "lookup"
            ttl = cfg.default_ttl_s
            reason_parts.append("lookup-style question over stable material: response caching applies (semantic tier where the profile permits)")
        else:
            task_class = "generative" if f.length_words > 60 else "lookup"
            ttl = cfg.default_ttl_s
            reason_parts.append("no risk markers found; default policy applies")

        if f.identifier_density >= 0.08 or f.identifier_count >= 2:
            risk_flags.append("identifier_or_numeric_sensitive")
            threshold = max(threshold, cfg.strict_semantic_threshold)
            budget = min(budget, cfg.strict_error_budget)
            verify = True
            reason_parts.append("identifiers/numbers present: exact-match guards and the strict semantic threshold are enforced")
        if strict:
            risk_flags.append(f"risk_level={profile.risk_level}")
            threshold = max(threshold, cfg.strict_semantic_threshold)
            budget = min(budget, cfg.strict_error_budget)
            verify = True
            if profile.risk_level == "safety_critical" and not cfg.semantic_in_safety_critical:
                semantic_ok = False
                reason_parts.append("safety-critical profile: semantic tier disabled (mismatch cost too high)")
            else:
                reason_parts.append("high-risk profile: strict threshold, 1% error budget, borderline hits verified")

        tiers.append("exact")
        if semantic_ok:
            tiers.append("semantic")
        if has_stable_prefix:
            tiers.append("prefix")
        if cfg.negative_caching:
            tiers.append("negative")

        prefix_ttl: PrefixTTLChoice
        if not has_stable_prefix:
            prefix_ttl = "none"
        elif self._long_stable_reference(profile) and task_class in ("lookup", "analysis"):
            prefix_ttl = "1h"
        else:
            prefix_ttl = cfg.prefix_default_ttl

        conflicts = f.conflicts()
        confidence = 0.9 - 0.15 * len(conflicts)
        if task_class == "generative":
            confidence -= 0.1
        return CachePlan(
            cacheable=True,
            tiers=tiers,
            ttl_s=ttl,
            semantic_threshold=threshold,
            error_budget=budget,
            verify_hits=verify,
            identifier_guards=cfg.identifier_guards,
            namespace=self._namespace(model, profile, tenant=tenant, prompt_version=prompt_version, corpus_version=corpus_version),
            prefix_ttl=prefix_ttl,
            prewarm=fan_out and has_stable_prefix,
            negative_ttl_s=cfg.negative_ttl_s,
            task_class=task_class,
            risk_flags=risk_flags,
            reason="; ".join(reason_parts) + ".",
            confidence=max(0.3, min(0.95, confidence)),
        )

    def plan(self, request: TaskRequest, profile: DomainProfile | None = None, *, model: str = "default") -> CachePlan:
        """Plan for a :class:`TaskRequest` (used by :meth:`caching.CachingModel.plan`)."""
        prof = profile or get_profile(request.domain)
        inputs = request.inputs
        return self.plan_text(
            request.task,
            prof,
            model=str(inputs.get("model", model)),
            tenant=inputs.get("tenant"),
            prompt_version=inputs.get("prompt_version"),
            corpus_version=inputs.get("corpus_version"),
            has_stable_prefix=True,  # the model always renders the domain profile as a stable system block
            fan_out=bool(inputs.get("fan_out", False)),
        )

    def plan_for_llm_request(self, request: LLMRequest, profile: DomainProfile | None = None, *, model: str = "default") -> CachePlan:
        """Plan for an :class:`LLMRequest` intercepted by :class:`caching.CachedLLMClient`."""
        meta = request.metadata or {}
        prof = profile or get_profile(meta.get("domain"))
        return self.plan_text(
            request.last_user_text(),
            prof,
            model=request.model or model,
            tenant=meta.get("tenant"),
            prompt_version=meta.get("prompt_version"),
            corpus_version=meta.get("corpus_version"),
            has_stable_prefix=bool(request.system_stable),
            fan_out=bool(meta.get("fan_out", False)),
        )

    # ------------------------------------------------------------ reconcile
    def reconcile(self, llm_plan: CachePlan, heuristic: CachePlan, profile: DomainProfile) -> tuple[CachePlan, list[str]]:
        """Merge an LLM-produced plan with the heuristic floor.

        The LLM may be *more* conservative than the heuristics (disable a tier,
        shorten a TTL, raise a threshold) but never less: safety rules from the
        heuristics win. Returns the merged plan and the list of overrides applied.
        """
        cfg = self.config
        overrides: list[str] = []
        merged = llm_plan.model_copy(deep=True)
        if not heuristic.cacheable and merged.cacheable:
            merged.cacheable = False
            merged.tiers = []
            overrides.append("heuristics mark the prompt non-cacheable (diversity requested)")
        if "semantic" in merged.tiers and "semantic" not in heuristic.tiers:
            merged.tiers = [t for t in merged.tiers if t != "semantic"]
            overrides.append("semantic tier removed: heuristics/profile forbid it")
        if merged.semantic_threshold < heuristic.semantic_threshold:
            merged.semantic_threshold = heuristic.semantic_threshold
            overrides.append(f"semantic threshold raised to {heuristic.semantic_threshold}")
        if merged.error_budget > heuristic.error_budget:
            merged.error_budget = heuristic.error_budget
            overrides.append(f"error budget tightened to {heuristic.error_budget}")
        if heuristic.verify_hits and not merged.verify_hits:
            merged.verify_hits = True
            overrides.append("borderline-hit verification enforced")
        if heuristic.ttl_s is not None and (merged.ttl_s is None or merged.ttl_s > heuristic.ttl_s) and heuristic.task_class in ("time_sensitive", "personal"):
            merged.ttl_s = heuristic.ttl_s
            overrides.append(f"TTL capped at {heuristic.ttl_s}s for {heuristic.task_class} content")
        if merged.cacheable and (not merged.tiers or merged.tiers[0] != "exact"):
            merged.tiers = ["exact", *[t for t in merged.tiers if t != "exact"]]
            overrides.append("exact tier is always first")
        if cfg.identifier_guards and not merged.identifier_guards:
            merged.identifier_guards = True
            overrides.append("identifier guards enforced by configuration")
        merged.namespace = heuristic.namespace
        merged.negative_ttl_s = min(merged.negative_ttl_s or cfg.negative_ttl_s, cfg.negative_ttl_s)
        for flag in heuristic.risk_flags:
            if flag not in merged.risk_flags:
                merged.risk_flags.append(flag)
        return merged, overrides


__all__ = [
    "ANALYSIS_WORDS",
    "CachePlan",
    "CacheTier",
    "CacheabilityFeatures",
    "HeuristicPlanner",
    "LOOKUP_WORDS",
    "PERSONAL_WORDS",
    "RANDOM_WORDS",
    "TIME_SENSITIVE_WORDS",
    "TaskClass",
    "analyze_task",
]
