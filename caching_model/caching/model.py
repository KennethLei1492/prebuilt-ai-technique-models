"""``CachingModel``: the caching component of the pipeline.

Two ways to use it:

* **Middleware** - :meth:`CachingModel.wrap` returns a
  :class:`caching.CachedLLMClient` that any other component can use as its
  ``llm``; every call then passes through the per-request plan -> provider
  prefix policy (1h TTL promotion, prewarm before a fan-out) -> exact ->
  semantic -> single-flight -> store. :attr:`CachingModel.cache` is the
  :class:`pipeline_core.ResponseCache` other models accept as a dependency.
* **Component** - ``run(request)`` *plans and explains* caching for a prompt:
  it classifies cacheability (heuristics offline, one structured LLM call
  online reconciled against the heuristic floor), constructs the exact key,
  namespace, tags and dependency fingerprint, audits the provider prefix
  layout, estimates savings and returns a :class:`CacheReport`.

The plan is a :class:`caching.CachePlan`; its decision rules are documented in
:mod:`caching.plan` and in the README.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline_core import (
    DEFAULT_MODEL,
    ENGINEERING_RIGOR,
    PRICING_USD_PER_MTOK,
    BaseLLMClient,
    ComponentSpec,
    DomainProfile,
    Evidence,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    PipelineComponent,
    RunContext,
    Status,
    TaskRequest,
    TaskResult,
    Tracer,
    Usage,
    estimate_cost,
    estimate_tokens,
    get_profile,
    register,
    render_evidence,
    run_sync,
)

from .backends import CacheBackend, CacheEntry, make_backend
from .config import CachingConfig
from .evaluation import latency_saved, prefix_savings
from .invalidation import DependencyFingerprint, InvalidationEvent, InvalidationManager
from .keys import exact_key, tagged_keys
from .middleware import CacheMetrics, CacheStore, CachedLLMClient
from .plan import CacheabilityFeatures, CachePlan, HeuristicPlanner, analyze_task
from .prefix import PrefixCacheAdvisor, PrefixReport
from .prompts import CACHE_PLAN_TEMPLATE
from .semantic import Embedder, HashingEmbedder, SemanticCache

Clock = Callable[[], float]
PlannerMode = Literal["auto", "always", "never"]


class SavingsEstimate(BaseModel):
    """Expected savings for a prompt under the plan (upper bounds; every reuse assumed a repeat)."""

    model_config = ConfigDict(extra="forbid")

    priced_model: str = Field(description="Model id used for pricing (falls back to the default model for unpriced clients)")
    expected_reuses: int = Field(ge=1)
    prompt_tokens: int = Field(ge=0)
    expected_output_tokens: int = Field(ge=0)
    cost_per_uncached_call_usd: float = Field(ge=0.0)
    exact_tier_saved_usd: float = Field(ge=0.0, description="Cost avoided if every reuse after the first is an exact repeat")
    prefix_saved_usd: float = Field(ge=0.0, description="Input cost avoided by provider prefix caching over the reuses")
    prefix_saved_pct: float = Field(ge=0.0, le=100.0)
    latency_saved_s: float = Field(ge=0.0, description="Latency avoided by exact hits, assuming the given mean miss latency")
    assumed_miss_latency_s: float = Field(ge=0.0)


class CacheReport(BaseModel):
    """Structured output of :meth:`CachingModel.execute`."""

    model_config = ConfigDict(extra="forbid")

    task: str
    domain: str
    plan: CachePlan
    plan_source: Literal["heuristic", "llm"] = "heuristic"
    plan_overrides: list[str] = Field(default_factory=list, description="Safety overrides applied to the LLM plan")
    features: CacheabilityFeatures
    exact_key: str
    namespace: str
    tags: list[str]
    dependency_fingerprint: Optional[DependencyFingerprint] = None
    prefix: PrefixReport
    savings: SavingsEstimate
    cached_entry_exists: bool = Field(default=False, description="An exact-tier entry already exists for this key")
    cached_entry_age_s: Optional[float] = None
    stored: bool = Field(default=False, description="A response from request.inputs['response'] was stored under the key")
    recommendations: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Human-readable explanation used for ``TaskResult.output_text``."""
        p = self.plan
        lines = [
            f"Cache plan for: {self.task[:120]}",
            f"  domain={self.domain}  class={p.task_class}  cacheable={p.cacheable}  tiers={p.tiers}",
            f"  ttl={'none' if p.ttl_s is None else f'{p.ttl_s:g}s'}  semantic_threshold={p.semantic_threshold}  error_budget={p.error_budget}  verify_hits={p.verify_hits}",
            f"  namespace={p.namespace}",
            f"  prefix_ttl={p.prefix_ttl}  prewarm={p.prewarm}  risk_flags={p.risk_flags}",
            f"  reason: {p.reason}",
            f"  plan source: {self.plan_source}" + (f" (overrides: {'; '.join(self.plan_overrides)})" if self.plan_overrides else ""),
            f"Exact key: {self.exact_key[:16]}...  tags={self.tags}",
        ]
        if self.dependency_fingerprint is not None and self.dependency_fingerprint.document_ids:
            lines.append(f"Grounded on documents {self.dependency_fingerprint.document_ids} (fingerprint {self.dependency_fingerprint.fingerprint[:12]})")
        pr = self.prefix
        lines.append(
            f"Prefix: ~{pr.prefix_tokens} stable tokens (min {pr.min_cacheable_tokens} for {pr.model}), "
            f"meets_minimum={pr.meets_minimum}, invalidators={len(pr.invalidators)}, recommended_ttl={pr.recommended_ttl}, score={pr.score}"
        )
        s = self.savings
        lines.append(
            f"Savings over {s.expected_reuses} reuses ({s.priced_model}): exact tier up to ${s.exact_tier_saved_usd:.4f}, "
            f"prefix cache ${s.prefix_saved_usd:.4f} ({s.prefix_saved_pct:.0f}% of prefix input), latency {s.latency_saved_s:.1f}s"
        )
        if self.cached_entry_exists:
            lines.append(f"An exact-tier entry already exists (age {self.cached_entry_age_s:.0f}s)")
        if self.stored:
            lines.append("Stored the provided response under the exact key.")
        if self.recommendations:
            lines.append("Recommendations:")
            lines.extend(f"  - {r}" for r in self.recommendations)
        return "\n".join(lines)


@register
class CachingModel(PipelineComponent):
    """Plan, explain and provide response caching for LLM pipelines."""

    spec = ComponentSpec(
        name="caching",
        version="1.0.0",
        description=(
            "Exact-match response caching, verified semantic caching with identifier guards, single-flight "
            "coalescing, negative caching, provider prefix-cache optimisation and dependency-based invalidation."
        ),
        capabilities=[
            "cache_planning",
            "exact_match_cache",
            "semantic_cache",
            "request_coalescing",
            "negative_cache",
            "prefix_cache_advice",
            "invalidation",
            "llm_middleware",
        ],
        consumes=[
            "task (the prompt to classify)",
            "inputs.system_blocks | inputs.system_prompt (stable prefix to audit)",
            "inputs.model, inputs.expected_reuses, inputs.prompt_version, inputs.corpus_version, inputs.tenant, inputs.fan_out",
            "inputs.llm_request (a full LLMRequest to audit instead of the task)",
            "inputs.response (a response to store under the computed key)",
            "context (Evidence the answer depends on -> dependency fingerprint)",
        ],
        produces=["CacheReport (plan, key, namespace, tags, prefix audit, savings estimate)"],
        composes_with=["hybrid_retrieval", "task_decomposition", "answer_verification", "parallelization", "pipeline_rag"],
    )

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        config: CachingConfig | dict[str, Any] | None = None,
        backend: CacheBackend | None = None,
        semantic: SemanticCache | None = None,
        embedder: Embedder | None = None,
        planner_mode: PlannerMode = "auto",
        tracer: Tracer | None = None,
        raise_on_error: bool = False,
        clock: Clock = time.time,
        **config_overrides: Any,
    ) -> None:
        super().__init__(llm, tracer=tracer, raise_on_error=raise_on_error)
        if isinstance(config, CachingConfig):
            cfg = config.model_copy(update=config_overrides) if config_overrides else config
        else:
            cfg = CachingConfig(**{**(config or {}), **config_overrides})
        self.config: CachingConfig = cfg  # type: ignore[assignment]
        self.clock = clock
        self.planner_mode: PlannerMode = planner_mode
        self.backend: CacheBackend = backend if backend is not None else make_backend(
            cfg.backend,
            max_entries=cfg.max_entries,
            max_bytes=cfg.max_bytes,
            policy=cfg.eviction_policy,
            sqlite_path=cfg.sqlite_path,
            file_dir=cfg.file_dir,
            redis_url=cfg.redis_url,
            key_prefix=cfg.key_prefix,
            clock=clock,
        )
        self.semantic: SemanticCache | None = semantic
        if self.semantic is None and cfg.semantic_enabled:
            self.semantic = SemanticCache(
                embedder if embedder is not None else HashingEmbedder(cfg.embedding_dimension),
                threshold=cfg.semantic_threshold,
                top_k=cfg.semantic_top_k,
                borderline_band=cfg.borderline_band,
                error_budget=cfg.semantic_error_budget,
                identifier_guards=cfg.identifier_guards,
                max_entries=cfg.semantic_max_entries,
                normalization=cfg.normalization,
                clock=clock,
            )
        self.planner = HeuristicPlanner(cfg)
        self.advisor = PrefixCacheAdvisor(cfg, clock=clock)
        self.invalidation = InvalidationManager(self.backend, self.semantic, clock=clock)
        self.store = CacheStore(self.backend, self.semantic, default_ttl_s=cfg.default_ttl_s)
        self.clients: list[CachedLLMClient] = []

    # ------------------------------------------------------------------ plan
    def _should_use_llm(self, request: TaskRequest, ctx: RunContext) -> bool:
        if request.constraints.deterministic or self.planner_mode == "never":
            return False
        remaining = ctx.budget.remaining_calls()
        if remaining is not None and remaining <= 0:
            return False
        if ctx.deadline_exceeded():
            return False
        if self.planner_mode == "always":
            return True
        return not ctx.offline

    def _model_id(self, request: TaskRequest) -> str:
        """Model id used for *pricing* and the prefix minimum (falls back to the default model for unpriced clients)."""
        m = request.inputs.get("model")
        if m:
            return str(m)
        return self.llm.model if self.llm.model in PRICING_USD_PER_MTOK else DEFAULT_MODEL

    def _key_model_id(self, request: TaskRequest) -> str:
        """Model id used for *keys and namespaces*: ``inputs.model`` or this component's client model.

        Same rule as :class:`CachedLLMClient` (``request.model or inner.model``),
        so a report's key matches the key a client wrapping the same model
        computes; an unpriced offline client keeps its own id, so mock answers
        can never be served to a real model through a shared store.
        """
        m = request.inputs.get("model")
        return str(m) if m else (self.llm.model or DEFAULT_MODEL)

    async def plan(self, request: TaskRequest, ctx: RunContext) -> CachePlan:
        """Classify cacheability: heuristics always; one structured LLM call online, reconciled against them."""
        model_id = self._model_id(request)
        heuristic = self.planner.plan(request, ctx.profile, model=self._key_model_id(request))
        ctx.scratch["plan_source"] = "heuristic"
        ctx.scratch["plan_overrides"] = []
        if not self._should_use_llm(request, ctx):
            return heuristic
        features = analyze_task(request.task)
        rendered = CACHE_PLAN_TEMPLATE.render(
            extra_system=[ctx.profile.system_text()] + ([ENGINEERING_RIGOR] if ctx.profile.risk_level in ("high", "safety_critical") else []),
            task=request.task,
            features=json.dumps(features.model_dump(), sort_keys=True),
            model=model_id,
            has_stable_prefix="true",
            fan_out="true" if request.inputs.get("fan_out") else "false",
        )
        req = LLMRequest(
            system_stable=rendered.system_blocks,
            messages=[Message(role="user", content=rendered.user)],
            output_schema=rendered.output_schema,
            effort=self.config.planner_effort,
            model=self.config.planner_model,
            max_tokens=1500,
            tags=["caching.plan"],
            metadata={"no_cache": True, "prompt_template": rendered.template, "prompt_version": rendered.version},
        )
        try:
            resp = await ctx.llm_call(req, step="plan.llm")
            llm_plan = CachePlan.model_validate(resp.parsed)
        except LLMError as exc:
            # refusal, truncation, invalid JSON, connection/API errors: the heuristic plan is the floor.
            # BudgetExceededError is not an LLMError and propagates (the core reports it).
            ctx.warn(f"LLM plan unavailable ({type(exc).__name__}: {str(exc)[:200]}); using heuristic plan")
            return heuristic
        except ValidationError as exc:
            ctx.warn(f"LLM plan failed validation ({str(exc)[:200]}); using heuristic plan")
            return heuristic
        merged, overrides = self.planner.reconcile(llm_plan, heuristic, ctx.profile)
        ctx.scratch["plan_source"] = "llm"
        ctx.scratch["plan_overrides"] = overrides
        for o in overrides:
            ctx.warn(f"plan override: {o}")
        return merged

    # --------------------------------------------------------------- execute
    def _representative_request(self, request: TaskRequest, plan: CachePlan, profile: DomainProfile, model_id: str) -> LLMRequest:
        """The LLMRequest the prompt would be sent as (what the key, prefix audit and estimate are computed on)."""
        inputs = request.inputs
        raw = inputs.get("llm_request")
        if raw is not None:
            req = raw if isinstance(raw, LLMRequest) else LLMRequest.model_validate(raw)
        else:
            blocks: list[str] = [profile.system_text()]
            if profile.risk_level in ("high", "safety_critical"):
                blocks.append(ENGINEERING_RIGOR)
            if inputs.get("system_prompt"):
                blocks.append(str(inputs["system_prompt"]))
            blocks.extend(str(b) for b in inputs.get("system_blocks", []) if b)
            user = request.task
            if request.context:
                user = render_evidence(request.context) + "\n<task>\n" + request.task + "\n</task>"
            req = LLMRequest(system_stable=blocks, messages=[Message(role="user", content=user)], model=model_id)
        meta = dict(req.metadata)
        for k in ("prompt_version", "prompt_template", "corpus_version", "tenant"):
            if k in inputs and k not in meta:
                meta[k] = inputs[k]
        req = req.model_copy(update={"metadata": meta, "cache_ttl": "1h" if plan.prefix_ttl == "1h" else "5m", "model": req.model or model_id})
        return req

    def _savings(self, req: LLMRequest, plan: CachePlan, prefix: PrefixReport, request: TaskRequest, model_id: str) -> SavingsEstimate:
        cfg = self.config
        reuses = int(request.inputs.get("expected_reuses", cfg.default_expected_reuses))
        reuses = max(1, reuses)
        out_tokens = int(request.inputs.get("expected_output_tokens", 400))
        miss_latency = float(request.inputs.get("mean_miss_latency_s", 4.0))
        prompt_tokens = req.prefix_tokens() + estimate_tokens(req.system_volatile or "") + sum(estimate_tokens(m.text()) for m in req.messages)
        per_call = estimate_cost(model_id, Usage(input_tokens=prompt_tokens, output_tokens=out_tokens))
        exact_saved = per_call * (reuses - 1) if plan.uses("exact") else 0.0
        if prefix.meets_minimum and plan.uses("prefix") and plan.prefix_ttl != "none":
            ps = prefix_savings(
                model_id,
                prefix.prefix_tokens,
                reuses,
                ttl="1h" if plan.prefix_ttl == "1h" else "5m",
                write_multiplier_5m=cfg.write_multiplier_5m,
                write_multiplier_1h=cfg.write_multiplier_1h,
                read_multiplier=cfg.read_multiplier,
            )
            prefix_saved, prefix_pct = max(0.0, ps["saved_usd"]), max(0.0, ps["saved_pct"])
        else:
            prefix_saved, prefix_pct = 0.0, 0.0
        lat = latency_saved(reuses - 1, miss_latency) if plan.uses("exact") else 0.0
        return SavingsEstimate(
            priced_model=model_id,
            expected_reuses=reuses,
            prompt_tokens=prompt_tokens,
            expected_output_tokens=out_tokens,
            cost_per_uncached_call_usd=round(per_call, 8),
            exact_tier_saved_usd=round(exact_saved, 8),
            prefix_saved_usd=round(prefix_saved, 8),
            prefix_saved_pct=round(min(100.0, prefix_pct), 2),
            latency_saved_s=round(lat, 3),
            assumed_miss_latency_s=miss_latency,
        )

    async def execute(self, request: TaskRequest, plan: CachePlan, ctx: RunContext) -> TaskResult:
        """Explain the plan: key, namespace, tags, dependency fingerprint, prefix audit, savings."""
        cfg = self.config
        model_id = self._model_id(request)
        key_model = self._key_model_id(request)
        inputs = request.inputs

        with ctx.step("analyze") as st:
            features = analyze_task(request.task)
            req = self._representative_request(request, plan, ctx.profile, model_id)
            st.summary = f"class={plan.task_class} cacheable={plan.cacheable}"

        with ctx.step("keys") as st:
            fingerprint: DependencyFingerprint | None = None
            doc_ids: list[str] = []
            if request.context:
                fingerprint = DependencyFingerprint.from_evidence(request.context, corpus_version=inputs.get("corpus_version"))
                doc_ids = fingerprint.document_ids
            key = exact_key(
                req,
                key_model,
                inputs.get("prompt_version"),
                normalization=cfg.normalization,
                namespace=plan.namespace,
                tenant_salt=cfg.tenant_salt,
                extra_keys={"corpus_version": inputs.get("corpus_version"), "dependency_fingerprint": fingerprint.fingerprint if fingerprint else None},
            )
            tags = tagged_keys(
                model=key_model,
                prompt_template=inputs.get("prompt_template"),
                prompt_version=inputs.get("prompt_version"),
                document_ids=doc_ids,
                namespace=plan.namespace,
            )
            st.summary = f"key={key[:12]} tags={len(tags)}"

        with ctx.step("prefix_audit") as st:
            prefix = self.advisor.analyze(req, model=model_id, expected_reuses=inputs.get("expected_reuses"))
            if prefix.invalidators:
                ctx.warn(f"stable prefix contains volatile content: {sorted({i.kind for i in prefix.invalidators})}")
            st.summary = f"prefix_tokens={prefix.prefix_tokens} ttl={prefix.recommended_ttl}"

        with ctx.step("estimate") as st:
            savings = self._savings(req, plan, prefix, request, model_id)
            st.summary = f"exact=${savings.exact_tier_saved_usd:.4f} prefix=${savings.prefix_saved_usd:.4f}"

        with ctx.step("store") as st:
            entry: CacheEntry | None = self.backend.peek(key)  # does not touch the entry or skew hit/miss stats
            stored = False
            if plan.cacheable and "response" in inputs and inputs["response"] is not None:
                value = self._as_response_value(inputs["response"], key_model)
                written = self.backend.set(
                    key, value, ttl_s=plan.ttl_s, tags=tags, namespace=plan.namespace, version=self.invalidation.versions.current(plan.namespace)
                )
                stored = written is not None
                entry = None
                if not stored:
                    ctx.warn(f"plan ttl_s={plan.ttl_s} means 'do not retain'; the provided response was not stored")
            st.summary = f"exists={entry is not None} stored={stored}"

        recs = list(prefix.recommendations)
        if not plan.cacheable:
            recs.append("Do not response-cache this prompt; keep only provider prefix caching of the stable system prompt.")
        elif "semantic" not in plan.tiers:
            recs.append("Semantic tier disabled for this prompt; only normalised exact repeats will hit.")
        elif plan.verify_hits:
            recs.append(f"Semantic hits within {cfg.borderline_band} of the {plan.semantic_threshold} threshold are verified by a cheap equivalence check before serving.")
        if fingerprint is not None and doc_ids:
            recs.append(f"Answer depends on {len(doc_ids)} document(s); the key includes their content fingerprint and the entry is tagged doc:<id> for purge on re-index.")
        if plan.task_class in ("time_sensitive", "personal"):
            recs.append(f"Short TTL ({plan.ttl_s}s) because the content is {plan.task_class.replace('_', '-')}.")

        report = CacheReport(
            task=request.task,
            domain=ctx.profile.name,
            plan=plan,
            plan_source=ctx.scratch.get("plan_source", "heuristic"),
            plan_overrides=list(ctx.scratch.get("plan_overrides", [])),
            features=features,
            exact_key=key,
            namespace=plan.namespace,
            tags=tags,
            dependency_fingerprint=fingerprint,
            prefix=prefix,
            savings=savings,
            cached_entry_exists=entry is not None,
            cached_entry_age_s=None if entry is None else round(entry.age_s(self.clock()), 3),
            stored=stored,
            recommendations=recs,
        )

        confidence, basis = self._confidence(plan, prefix, report)
        threshold = max(ctx.profile.min_confidence_to_accept, request.constraints.min_confidence, cfg.min_confidence or 0.0)
        status = Status.SUCCESS
        if confidence < threshold:
            status = Status.NEEDS_REVIEW
            ctx.warn(f"plan confidence {confidence:.2f} below required {threshold:.2f}; review before enabling caching")
        return self.new_result(
            request,
            status=status,
            output=report,
            output_text=report.render(),
            confidence=confidence,
            confidence_basis=basis,
            evidence=list(request.context),
            metadata={"plan": plan.model_dump(mode="json"), "exact_key": key, "namespace": plan.namespace, "confidence_threshold": threshold},
        )

    @staticmethod
    def _as_response_value(value: Any, model: str) -> dict[str, Any]:
        """Coerce ``inputs['response']`` into a stored :class:`LLMResponse` so wrapped clients can serve it.

        An ``LLMResponse`` or a dict with ``text`` is validated as one; a string
        becomes the response text; any other value becomes ``parsed`` with its
        canonical JSON as the text.
        """
        resp: LLMResponse
        if isinstance(value, LLMResponse):
            resp = value
        elif isinstance(value, dict) and "text" in value:
            try:
                resp = LLMResponse.model_validate(value)
            except ValidationError:
                resp = LLMResponse(text=json.dumps(value, sort_keys=True, default=str), parsed=value)
        elif isinstance(value, str):
            resp = LLMResponse(text=value)
        else:
            resp = LLMResponse(text=json.dumps(value, sort_keys=True, default=str), parsed=value)
        if not resp.model:
            resp = resp.model_copy(update={"model": model})
        return resp.model_dump(mode="json", exclude={"raw"})

    @staticmethod
    def _confidence(plan: CachePlan, prefix: PrefixReport, report: CacheReport) -> tuple[float, str]:
        c = plan.confidence
        parts = [f"planner confidence {plan.confidence:.2f} ({report.plan_source})"]
        conflicts = report.features.conflicts()
        if conflicts:
            parts.append(f"{len(conflicts)} conflicting signal(s): {'; '.join(conflicts)}")
        if report.plan_source == "llm":
            if report.plan_overrides:
                c -= 0.05 * len(report.plan_overrides)
                parts.append(f"{len(report.plan_overrides)} safety override(s) applied to the LLM plan")
            else:
                c = min(0.98, c + 0.05)
                parts.append("LLM plan agrees with heuristic floor")
        if prefix.invalidators:
            c -= 0.1
            parts.append("volatile content found in the stable prefix")
        c = max(0.0, min(1.0, c))
        return round(c, 3), "; ".join(parts)

    # ----------------------------------------------------------- direct API
    def wrap(self, llm: BaseLLMClient, *, profile: DomainProfile | str | None = None) -> CachedLLMClient:
        """Wrap ``llm`` so every call goes through this model's cache tiers (shared backend and semantic index)."""
        client = CachedLLMClient(
            llm,
            backend=self.backend,
            semantic=self.semantic,
            config=self.config,
            planner=self.planner,
            profile=profile,
            versions=self.invalidation.versions,
            clock=self.clock,
        )
        self.clients.append(client)
        return client

    @property
    def cache(self) -> CacheStore:
        """The :class:`pipeline_core.ResponseCache` other components receive as ``cache``."""
        return self.store

    async def get(self, key: str) -> Optional[Any]:
        """ResponseCache protocol: exact-tier lookup."""
        return await self.store.get(key)

    async def set(self, key: str, value: Any, *, ttl_s: Optional[float] = None, tags: Sequence[str] = ()) -> None:
        """ResponseCache protocol: exact-tier store with the configured default TTL."""
        await self.store.set(key, value, ttl_s=ttl_s, tags=tags)

    async def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int:
        """ResponseCache protocol: remove entries by tag or key across tiers."""
        ev = self.invalidation.invalidate(tags=tags, keys=keys, cause="manual")
        return ev.removed_exact + ev.removed_semantic

    def invalidate_documents(self, document_ids: Sequence[str], *, cause: str = "document_updated") -> InvalidationEvent:
        """Purge every entry grounded on any of ``document_ids`` (tag ``doc:<id>``)."""
        return self.invalidation.invalidate_documents(document_ids, cause=cause)

    def bump_version(self, namespace: str, *, cause: str = "version_bump") -> int:
        """Coarse invalidation: bump the namespace version and purge its exact and semantic entries.

        Wrapped clients share the version registry and also reject entries
        stamped with an older version of the namespace.
        """
        return self.invalidation.bump_version(namespace, cause=cause)

    def fingerprint(self, evidence: Sequence[Evidence], *, corpus_version: str | None = None) -> DependencyFingerprint:
        """Dependency fingerprint of the documents an answer is grounded on."""
        return DependencyFingerprint.from_evidence(evidence, corpus_version=corpus_version)

    def plan_for(self, task: str, *, domain: str = "general", **inputs: Any) -> CachePlan:
        """Heuristic plan for a task string (no LLM, no run context)."""
        req = TaskRequest(task=task, inputs=inputs, domain=domain)
        return self.planner.plan(req, get_profile(domain), model=self._key_model_id(req))

    def key_for(self, request: LLMRequest, *, profile: DomainProfile | str | None = None, client: BaseLLMClient | None = None) -> str:
        """Exact-tier key that ``self.wrap(client or self.llm, profile=profile)`` uses for ``request``.

        Same rules as :meth:`CachedLLMClient.key_for`: the model id is
        ``request.model`` or the client's model, and a registered
        ``request.metadata['domain']`` overrides ``profile``.
        """
        probe = CachedLLMClient(
            client or self.llm, backend=self.backend, semantic=self.semantic, config=self.config, planner=self.planner, profile=profile, clock=self.clock
        )
        return probe.key_for(request)

    def analyze_prefix(self, request: LLMRequest, *, model: str | None = None) -> PrefixReport:
        """Audit a request's provider-cacheable prefix (minimum size, invalidators, TTL, break-even)."""
        return self.advisor.analyze(request, model=model or request.model or self._model_id(TaskRequest(task="-")))

    async def prewarm(self, llm: BaseLLMClient, request: LLMRequest) -> Usage:
        """Write the provider prefix cache for ``request`` before a fan-out (no-op for offline clients)."""
        return await self.advisor.prewarm(llm, request)

    def explain(self, task: str, *, domain: str = "general", **inputs: Any) -> CacheReport:
        """Synchronous convenience: run the component and return the :class:`CacheReport`."""
        result = run_sync(self.run(TaskRequest(task=task, inputs=inputs, domain=domain)))
        if not isinstance(result.output, CacheReport):
            raise RuntimeError(f"caching run failed: {result.errors}")
        return result.output

    def metrics(self) -> CacheMetrics:
        """Aggregate middleware metrics over every client created by :meth:`wrap`."""
        total = CacheMetrics()
        for c in self.clients:
            for k, v in c.metrics.model_dump().items():
                setattr(total, k, getattr(total, k) + v)
        return total

    def savings_report(self) -> dict[str, Any]:
        """Hit counts, cost and latency saved, backend and semantic statistics."""
        m = self.metrics()
        return {
            **m.model_dump(),
            "hits": m.hits,
            "hit_rate": round(m.hit_rate, 4),
            "clients": len(self.clients),
            "backend": self.backend.snapshot_stats().model_dump(),
            "semantic": self.semantic.false_hit_risk() if self.semantic is not None else None,
            "invalidations_by_cause": self.invalidation.causes(),
        }

    def describe(self) -> dict[str, Any]:
        return {**self.spec.model_dump(), "llm": self.llm.describe(), "config": self.config.model_dump(), "backend": self.backend.name, "semantic": self.semantic is not None}


__all__ = ["CacheReport", "CachingModel", "PlannerMode", "SavingsEstimate"]
