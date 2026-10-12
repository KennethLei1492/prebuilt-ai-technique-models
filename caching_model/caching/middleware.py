"""``CachedLLMClient``: transparent response caching for any ``BaseLLMClient``.

Lookup order per request::

    plan (cacheability, per-request domain) -> provider prefix policy (1h TTL
    promotion, one-time prewarm before a fan-out) -> exact tier -> semantic
    tier (guards, context chain, optional LLM equivalence check on borderline
    hits) -> single-flight coalescing -> inner client -> store (exact +
    semantic) / negative-cache

Behaviour:

* hits return a copy of the stored :class:`LLMResponse` with ``cached=True``,
  ``usage = Usage(cache_hits=1)`` (no tokens, no cost, no LLM call) and
  ``metadata`` describing the tier, key and age;
* any LLM work the client does *besides* the inner call (the borderline-hit
  equivalence check, a provider prefix prewarm) is added to the usage of the
  response it belongs to, so ``RunContext`` budgets and the tracer see it;
* ``request.metadata`` controls the client per call: ``no_cache`` (bypass),
  ``deterministic`` (always prefer the cache, never bypass for diversity),
  ``domain`` (domain profile for this request; overrides the profile given
  to ``wrap``), ``cache_ttl_s`` (``0`` = do not retain), ``cache_tags``,
  ``cache_namespace``, ``tenant``, ``prompt_version``, ``prompt_template``,
  ``corpus_version``, ``document_ids``, ``dependency_fingerprint``, ``fan_out``;
* XFetch early refresh recomputes a hot entry shortly before it expires,
  using the recompute cost stored with the entry (beta = 1);
* deterministic failures (schema violations, refusals) are negative-cached for
  a short TTL so repeated failing prompts do not re-run the model;
* entries that do not validate as an :class:`LLMResponse` (for example a raw
  value stored through the ``ResponseCache`` protocol under a client key) are
  dropped and treated as a miss, never raised to the caller;
* entries written before a namespace version bump are rejected;
* every decision is counted in :class:`CacheMetrics`.

The class also exposes :class:`CacheStore`, an async :class:`ResponseCache`
façade over a backend, which is what other models receive as ``cache``.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from pipeline_core import (
    BaseLLMClient,
    BudgetExceededError,
    DomainProfile,
    LLMError,
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    LLMTruncatedError,
    PRICING_USD_PER_MTOK,
    StructuredOutputError,
    Usage,
    estimate_cost,
    get_profile,
    list_profiles,
)

from .backends import CacheBackend, CacheEntry, InMemoryBackend
from .config import CachingConfig
from .invalidation import VersionRegistry
from .keys import context_chain_key, exact_key, previous_turn_texts, semantic_text, tagged_keys
from .plan import CachePlan, HeuristicPlanner
from .prompts import EQUIVALENCE_TEMPLATE, EquivalenceVerdict
from .semantic import HashingEmbedder, SemanticCache
from .stampede import SingleFlight, xfetch_should_refresh

log = logging.getLogger(__name__)

Clock = Callable[[], float]

_NEGATIVE_ERRORS: dict[str, type[LLMError]] = {
    "StructuredOutputError": StructuredOutputError,
    "LLMRefusalError": LLMRefusalError,
    "LLMTruncatedError": LLMTruncatedError,
}

_PROVIDER_TTL_S = {"5m": 300.0, "1h": 3600.0}


class CacheMetrics(BaseModel):
    """Counters maintained by :class:`CachedLLMClient`."""

    model_config = ConfigDict(extra="forbid")

    requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    semantic_verified: int = 0
    semantic_rejected_by_verifier: int = 0
    negative_hits: int = 0
    coalesced: int = 0
    misses: int = 0
    bypassed: int = 0
    early_refreshes: int = 0
    stale_rejections: int = 0
    version_rejections: int = 0
    corrupt_entries: int = 0
    stores: int = 0
    negative_stores: int = 0
    verification_calls: int = 0
    verification_cost_usd: float = 0.0
    prefix_ttl_promotions: int = 0
    prewarms: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    latency_saved_s: float = 0.0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits + self.negative_hits + self.coalesced

    @property
    def hit_rate(self) -> float:
        served = self.requests - self.bypassed
        return self.hits / served if served else 0.0


class CacheStore:
    """Async :class:`pipeline_core.ResponseCache` façade over a :class:`CacheBackend` (+ optional semantic tier).

    ``set(..., ttl_s=0)`` (or any non-positive TTL) means "do not retain": nothing
    is stored and an existing entry under ``key`` is dropped.
    """

    def __init__(self, backend: CacheBackend, semantic: SemanticCache | None = None, *, default_ttl_s: float | None = None) -> None:
        self.backend = backend
        self.semantic = semantic
        self.default_ttl_s = default_ttl_s

    async def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)

    async def set(self, key: str, value: Any, *, ttl_s: Optional[float] = None, tags: Sequence[str] = ()) -> None:
        self.backend.set(key, value, ttl_s=self.default_ttl_s if ttl_s is None else ttl_s, tags=tags)

    async def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int:
        n = self.backend.invalidate(tags=tags, keys=keys)
        if self.semantic is not None:
            n += self.semantic.invalidate(tags=tags, keys=keys)
        return n

    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        return self.backend.get_entry(key)

    def __len__(self) -> int:
        return len(self.backend)


class CachedLLMClient(BaseLLMClient):
    """Wrap any client with exact, semantic, coalescing and negative caching."""

    name = "cached"

    def __init__(
        self,
        inner: BaseLLMClient,
        *,
        backend: CacheBackend | None = None,
        semantic: SemanticCache | None = None,
        config: CachingConfig | None = None,
        planner: HeuristicPlanner | None = None,
        profile: DomainProfile | str | None = None,
        versions: VersionRegistry | None = None,
        clock: Clock = time.time,
        rand: Callable[[], float] = random.random,
    ) -> None:
        super().__init__(inner.model, effort=inner.effort)
        self.inner = inner
        self.config = config or CachingConfig()
        # ``backend or ...`` would be wrong: an empty backend is falsy because it defines __len__
        self.backend = backend if backend is not None else InMemoryBackend(max_entries=self.config.max_entries, max_bytes=self.config.max_bytes, policy=self.config.eviction_policy, clock=clock)
        self.semantic = semantic
        if self.semantic is None and self.config.semantic_enabled:
            self.semantic = SemanticCache(
                HashingEmbedder(self.config.embedding_dimension),
                threshold=self.config.semantic_threshold,
                top_k=self.config.semantic_top_k,
                borderline_band=self.config.borderline_band,
                error_budget=self.config.semantic_error_budget,
                identifier_guards=self.config.identifier_guards,
                max_entries=self.config.semantic_max_entries,
                normalization=self.config.normalization,
                clock=clock,
            )
        self.planner = planner or HeuristicPlanner(self.config)
        self.profile: DomainProfile = profile if isinstance(profile, DomainProfile) else get_profile(profile)
        self.versions = versions
        self.clock = clock
        self.rand = rand
        self.flight = SingleFlight()
        self.metrics = CacheMetrics()
        self.store = CacheStore(self.backend, self.semantic, default_ttl_s=self.config.default_ttl_s)
        self.last_plan: CachePlan | None = None
        self._warm_until: dict[str, float] = {}

    # ------------------------------------------------------------- helpers
    def describe(self) -> dict[str, Any]:
        return {"client": self.name, "inner": self.inner.describe(), "backend": self.backend.name, "semantic": self.semantic is not None, "metrics": self.metrics.model_dump()}

    def _model_for(self, request: LLMRequest) -> str:
        return request.model or self.inner.model

    def profile_for(self, request: LLMRequest) -> DomainProfile:
        """The profile governing ``request``: ``metadata['domain']`` when it names a registered profile, else the wrap-time profile."""
        dom = (request.metadata or {}).get("domain")
        if dom and str(dom) in list_profiles():
            return get_profile(str(dom))
        if dom:
            log.warning("unknown domain '%s' in request metadata; using the client profile '%s'", dom, self.profile.name)
        return self.profile

    def plan_for(self, request: LLMRequest) -> CachePlan:
        """The heuristic :class:`CachePlan` this client applies to ``request``."""
        return self.planner.plan_for_llm_request(request, self.profile_for(request), model=self.inner.model)

    def _namespace(self, request: LLMRequest, plan: CachePlan) -> str:
        return str(request.metadata.get("cache_namespace") or plan.namespace)

    def _version(self, namespace: str) -> int:
        return self.versions.current(namespace) if self.versions is not None else 1

    def _tags(self, request: LLMRequest, plan: CachePlan) -> list[str]:
        meta = request.metadata or {}
        return tagged_keys(
            model=self._model_for(request),
            prompt_template=meta.get("prompt_template"),
            prompt_version=meta.get("prompt_version"),
            document_ids=meta.get("document_ids", ()),
            namespace=self._namespace(request, plan),
            extra=meta.get("cache_tags", ()),
        )

    def _ttl(self, request: LLMRequest, plan: CachePlan) -> Optional[float]:
        meta = request.metadata or {}
        if "cache_ttl_s" in meta:
            v = meta["cache_ttl_s"]
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                log.warning("ignoring non-numeric metadata cache_ttl_s=%r; using the plan TTL", v)
        return plan.ttl_s

    def key_for(self, request: LLMRequest, plan: CachePlan | None = None) -> str:
        """The exact-tier key this client would use for ``request``."""
        p = plan or self.plan_for(request)
        meta = request.metadata or {}
        return exact_key(
            request,
            self._model_for(request),
            meta.get("prompt_version"),
            normalization=self.config.normalization,
            namespace=self._namespace(request, p),
            tenant_salt=self.config.tenant_salt,
            extra_keys={"corpus_version": meta.get("corpus_version"), "dependency_fingerprint": meta.get("dependency_fingerprint")},
        )

    @staticmethod
    def _cost_of(usage: Usage, model: str) -> float:
        return usage.cost_usd or estimate_cost(model, usage)

    def _cached_response(self, entry_value: Any, *, tier: str, key: str, age_s: float, extra: dict[str, Any] | None = None) -> Optional[LLMResponse]:
        """Rebuild a served response from a stored value; ``None`` when the value is not an ``LLMResponse``."""
        try:
            resp = LLMResponse.model_validate(entry_value)
        except ValidationError as exc:
            self.metrics.corrupt_entries += 1
            log.warning("dropping %s cache entry %s: stored value is not an LLMResponse (%s)", tier, key[:16], str(exc).splitlines()[0])
            return None
        resp.cached = True
        resp.latency_s = 0.0
        orig_usage = resp.usage
        resp.usage = Usage(cache_hits=1)
        saved_cost = self._cost_of(orig_usage, resp.model or self.inner.model)
        self.metrics.tokens_saved += orig_usage.input_tokens + orig_usage.output_tokens + orig_usage.cache_read_input_tokens + orig_usage.cache_creation_input_tokens
        self.metrics.cost_saved_usd += saved_cost
        self.metrics.latency_saved_s += orig_usage.latency_s
        info = {"cache_tier": tier, "cache_key": key, "age_s": round(age_s, 3), "saved_cost_usd": round(saved_cost, 8)}
        if extra:
            info.update(extra)
        setattr(resp, "cache", info)
        return resp

    @staticmethod
    def _with_usage(resp: LLMResponse, extra: Usage) -> LLMResponse:
        """Fold side-call usage (verification, prewarm) into the response the caller receives."""
        if extra.llm_calls or extra.cost_usd or extra.input_tokens or extra.output_tokens or extra.cache_creation_input_tokens or extra.cache_read_input_tokens:
            resp.usage = resp.usage + extra
        return resp

    def _stale(self, entry: CacheEntry) -> bool:
        ms = self.config.max_staleness_s
        return ms is not None and entry.age_s(self.clock()) > ms

    # ------------------------------------------------------- provider prefix
    def _apply_prefix_policy(self, request: LLMRequest, plan: CachePlan) -> LLMRequest:
        """Request the 1-hour provider TTL when the plan recommends it (never demotes an explicit ``1h``)."""
        if self.config.apply_prefix_ttl and plan.prefix_ttl == "1h" and request.system_stable and request.cache and request.cache_ttl != "1h":
            self.metrics.prefix_ttl_promotions += 1
            return request.model_copy(update={"cache_ttl": "1h"})
        return request

    async def _maybe_prewarm(self, request: LLMRequest, plan: CachePlan) -> Usage:
        """Write the provider prefix cache once before a fan-out (``plan.prewarm``); concurrent callers wait for it."""
        fn = getattr(self.inner, "prewarm", None)
        if not plan.prewarm or fn is None or not request.system_stable or not request.cache:
            return Usage()
        fp = request.prefix_fingerprint(self._model_for(request))
        if self._warm_until.get(fp, float("-inf")) > self.clock():
            return Usage()

        async def _warm() -> Usage:
            usage = await fn(request)
            self._warm_until[fp] = self.clock() + _PROVIDER_TTL_S.get(request.cache_ttl, 300.0)
            self.metrics.prewarms += 1
            return usage

        try:
            usage, coalesced = await self.flight.run("prewarm:" + fp, _warm)
        except BudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed prewarm only costs the cache write it was meant to save
            log.warning("provider prefix prewarm failed (%s: %s); continuing without it", type(exc).__name__, exc)
            return Usage()
        return Usage() if coalesced else usage

    # ------------------------------------------------------------- compose
    async def _complete(self, request: LLMRequest) -> LLMResponse:
        self.metrics.requests += 1
        meta = request.metadata or {}
        deterministic = bool(meta.get("deterministic", False))
        profile = self.profile_for(request)
        plan = self.planner.plan_for_llm_request(request, profile, model=self.inner.model)
        self.last_plan = plan
        request = self._apply_prefix_policy(request, plan)
        side_usage = await self._maybe_prewarm(request, plan)

        if meta.get("no_cache"):
            self.metrics.bypassed += 1
            return self._with_usage(await self.inner.complete(request), side_usage)
        if not plan.cacheable and not deterministic:
            self.metrics.bypassed += 1
            return self._with_usage(await self.inner.complete(request), side_usage)
        if not deterministic and self.config.bypass_rate > 0 and self.rand() < self.config.bypass_rate:
            self.metrics.bypassed += 1
            return self._with_usage(await self.inner.complete(request), side_usage)

        key = self.key_for(request, plan)
        namespace = self._namespace(request, plan)
        version = self._version(namespace)
        now = self.clock()

        # 1. exact tier --------------------------------------------------------------
        refresh_exact = False  # an exact entry exists but must be recomputed (stale or XFetch early refresh)
        entry = self.backend.get_entry(key)
        if entry is not None and entry.version != version:
            # written before a namespace version bump (coarse invalidation)
            self.metrics.version_rejections += 1
            self.backend.delete(key)
            if self.semantic is not None:
                self.semantic.invalidate(keys=[key])  # the semantic copy of the same answer is just as stale
            entry = None
        if entry is not None:
            if entry.negative:
                self.metrics.negative_hits += 1
                value = entry.value if isinstance(entry.value, dict) else {}
                err_cls = _NEGATIVE_ERRORS.get(str(value.get("error_type")), LLMError)
                raise err_cls(f"{value.get('message', 'cached failure')} (served from negative cache, age {entry.age_s(now):.0f}s)")
            if self._stale(entry):
                self.metrics.stale_rejections += 1
                refresh_exact = True
            elif xfetch_should_refresh(now, entry.expires_at(), entry.recompute_cost_s, beta=self.config.xfetch_beta, rand=self.rand):
                self.metrics.early_refreshes += 1
                refresh_exact = True
            else:
                served = self._cached_response(entry.value, tier="exact", key=key, age_s=entry.age_s(now))
                if served is not None:
                    self.metrics.exact_hits += 1
                    return self._with_usage(served, side_usage)
                self.backend.delete(key)  # corrupt entry: drop it and recompute

        # 2. semantic tier -----------------------------------------------------------
        # When the exact entry was rejected for staleness/refresh, the semantic tier must not
        # resurrect the very same stored answer; the recompute below refreshes both tiers.
        use_semantic = self.semantic is not None and plan.uses("semantic") and not request.tools
        consult_semantic = use_semantic and not refresh_exact
        ctx_key = context_chain_key(previous_turn_texts(request), self.config.context_turns, self.config.normalization)
        query_text = semantic_text(request, self.config.normalization)
        if consult_semantic and query_text:
            assert self.semantic is not None
            hit = self.semantic.lookup(query_text, namespace=namespace, context_key=ctx_key)
            if hit is not None and self.config.max_staleness_s is not None and hit.age_s > self.config.max_staleness_s:
                self.metrics.stale_rejections += 1
                hit = None
            if hit is not None and hit.entry.version != version:
                self.metrics.version_rejections += 1
                self.semantic.invalidate(keys=[hit.entry.key])
                hit = None
            if hit is not None and hit.similarity >= plan.semantic_threshold:
                serve = True
                verified: Optional[bool] = None
                if hit.borderline and plan.verify_hits and self.config.verify_borderline_hits:
                    verified, verify_usage = await self._verify_equivalence(hit.entry.text, hit.entry.value, query_text, request, profile)
                    side_usage = side_usage + verify_usage
                    serve = verified
                    if verified:
                        self.metrics.semantic_verified += 1
                    else:
                        self.metrics.semantic_rejected_by_verifier += 1
                        self.semantic.record_feedback(hit.entry.key, similarity=hit.similarity, correct=False)
                if serve:
                    served = self._cached_response(
                        hit.entry.value,
                        tier="semantic",
                        key=hit.entry.exact_key or hit.entry.key,
                        age_s=hit.age_s,
                        extra={"similarity": round(hit.similarity, 4), "borderline": hit.borderline, "verified": verified, "semantic_key": hit.entry.key},
                    )
                    if served is not None:
                        self.metrics.semantic_hits += 1
                        self.semantic.confirm_hit(hit.entry.key)
                        return self._with_usage(served, side_usage)
                    self.semantic.invalidate(keys=[hit.entry.key])

        # 3. single flight + inner call ----------------------------------------------
        async def _call() -> LLMResponse:
            t0 = self.clock()
            try:
                resp = await self.inner.complete(request)
            except (StructuredOutputError, LLMRefusalError, LLMTruncatedError) as exc:
                if plan.uses("negative") and self.config.negative_caching:
                    stored_neg = self.backend.set(
                        key,
                        {"error_type": type(exc).__name__, "message": str(exc)[:500]},
                        ttl_s=min(plan.negative_ttl_s or self.config.negative_ttl_s, self.config.negative_ttl_s),
                        tags=self._tags(request, plan),
                        namespace=namespace,
                        version=version,
                        negative=True,
                    )
                    if stored_neg is not None:
                        self.metrics.negative_stores += 1
                raise
            cost_s = max(0.0, self.clock() - t0) or resp.latency_s
            value = resp.model_dump(mode="json", exclude={"raw"})
            ttl = self._ttl(request, plan)
            stored = self.backend.set(key, value, ttl_s=ttl, tags=self._tags(request, plan), namespace=namespace, version=version, recompute_cost_s=cost_s)
            if stored is not None:  # ttl_s <= 0 means "do not retain"
                self.metrics.stores += 1
                if use_semantic and query_text:
                    assert self.semantic is not None
                    self.semantic.add(
                        query_text,
                        value,
                        namespace=namespace,
                        context_key=ctx_key,
                        tags=self._tags(request, plan),
                        ttl_s=ttl,
                        threshold=plan.semantic_threshold,
                        exact_key=key,
                        version=version,
                    )
            return resp

        if self.config.coalesce_inflight:
            resp, coalesced = await self.flight.run(key, _call)
            if coalesced:
                self.metrics.coalesced += 1
                out = resp.model_copy(deep=True)
                out.cached = True
                out.usage = Usage(cache_hits=1)
                setattr(out, "cache", {"cache_tier": "coalesced", "cache_key": key, "age_s": 0.0})
                return self._with_usage(out, side_usage)
        else:
            resp = await _call()
        self.metrics.misses += 1
        return self._with_usage(resp, side_usage)

    async def _verify_equivalence(
        self,
        cached_prompt: str,
        cached_value: Any,
        new_prompt: str,
        request: LLMRequest,
        profile: DomainProfile | None = None,
    ) -> tuple[bool, Usage]:
        """One cheap structured call: may the cached answer be served for the new prompt?

        Returns ``(verdict, usage)``. The usage is folded into the response the
        caller receives (so budgets and traces see the call) and counted in
        ``metrics.verification_calls`` / ``verification_cost_usd``; its cost is
        subtracted from ``cost_saved_usd``. Any failure of the check means "do
        not serve" and is logged; :class:`BudgetExceededError` propagates.
        """
        prof = profile or self.profile
        try:
            cached_answer = str(LLMResponse.model_validate(cached_value).text)[:4000]
        except ValidationError:
            cached_answer = str(cached_value)[:4000]
        rendered = EQUIVALENCE_TEMPLATE.render(
            extra_system=[prof.system_text()],
            cached_prompt=cached_prompt[:4000],
            cached_answer=cached_answer,
            new_prompt=new_prompt[:4000],
        )
        req = LLMRequest(
            system_stable=rendered.system_blocks,
            messages=[{"role": "user", "content": rendered.user}],  # type: ignore[list-item]
            output_schema=rendered.output_schema,
            effort="low",
            model=self.config.planner_model or request.model,
            max_tokens=400,
            tags=["caching.equivalence"],
            metadata={"no_cache": True},
        )
        self.metrics.verification_calls += 1
        try:
            resp = await self.inner.complete(req)
        except BudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001 - verification failure means "do not serve"
            log.warning("borderline-hit verification failed (%s: %s); not serving the semantic hit", type(exc).__name__, exc)
            return False, Usage()
        usage = resp.usage
        cost = self._cost_of(usage, resp.model or req.model or self.inner.model)
        self.metrics.verification_cost_usd += cost
        self.metrics.cost_saved_usd -= cost
        try:
            verdict = EquivalenceVerdict.model_validate(resp.parsed)
        except ValidationError as exc:
            log.warning("borderline-hit verifier returned an invalid verdict (%s); not serving the semantic hit", str(exc).splitlines()[0])
            return False, usage
        return bool(verdict.equivalent and verdict.confidence >= 0.5), usage

    # ------------------------------------------------------------- passthrough
    async def prewarm(self, request: LLMRequest) -> Usage:
        fn = getattr(self.inner, "prewarm", None)
        return await fn(request) if fn is not None else Usage()

    async def count_tokens(self, request: LLMRequest) -> int:
        fn = getattr(self.inner, "count_tokens", None)
        if fn is None:
            return request.prefix_tokens() + sum(len(m.text()) // 4 for m in request.messages)
        return await fn(request)

    def invalidate(self, *, tags: Sequence[str] = (), keys: Sequence[str] = ()) -> int:
        n = self.backend.invalidate(tags=tags, keys=keys)
        if self.semantic is not None:
            n += self.semantic.invalidate(tags=tags, keys=keys)
        return n

    def savings_report(self) -> dict[str, Any]:
        m = self.metrics
        return {
            **m.model_dump(),
            "hits": m.hits,
            "hit_rate": round(m.hit_rate, 4),
            "backend": self.backend.snapshot_stats().model_dump(),
            "semantic": self.semantic.false_hit_risk() if self.semantic is not None else None,
            "priced_model": self.inner.model if self.inner.model in PRICING_USD_PER_MTOK else None,
        }


__all__ = ["CacheMetrics", "CacheStore", "CachedLLMClient"]
