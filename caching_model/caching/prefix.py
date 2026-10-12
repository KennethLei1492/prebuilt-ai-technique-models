"""Provider prefix-cache advisor.

Anthropic's prompt cache is a prefix match: tools -> system -> messages up to
a ``cache_control`` breakpoint; writes cost 1.25x (5-minute TTL) or 2x
(1-hour TTL) of base input, reads 0.1x; the minimum cacheable prefix is
512-4096 tokens depending on the model; the entry only becomes available after
the first response begins, so cold-prefix fan-outs must be serialised.

:class:`PrefixCacheAdvisor` turns those rules into checks and recommendations:

* **lint** a request or template for silent invalidators in the stable blocks
  (timestamps, dates, UUIDs, request ids, clock times, nonces); JSON objects
  with unsorted keys are reported only as an informational recommendation,
  because a byte-constant JSON literal never breaks the prefix - only JSON
  re-rendered from a dict at runtime can;
* **gate** on the model's minimum cacheable token count;
* **recommend a TTL** from observed inter-arrival gaps of the same prefix
  (5m when the median gap is at most five minutes; 1h for gaps of 5-60
  minutes once the prefix has been seen at least three times, because every
  hit refreshes the 1-hour TTL and the 2x write pays off from the third use;
  keep-alive for gaps over an hour while cheap keep-alive reads still beat
  uncached calls; none beyond that);
* compute **break-even** reuse counts and estimated savings from the pricing
  table;
* **prewarm** through ``AnthropicLLM.prewarm`` and **verify** hits from
  ``usage.cache_read_input_tokens``.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import DEFAULT_MODEL, MIN_CACHEABLE_TOKENS, PRICING_USD_PER_MTOK, BaseLLMClient, LLMRequest, LLMResponse, PromptTemplate, Usage, estimate_tokens

from .config import CachingConfig
from .keys import VOLATILE_PATTERNS

Clock = Callable[[], float]
TTLRecommendation = Literal["5m", "1h", "keep-alive", "none"]

_DATE_ONLY = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_DATE_WORDS = re.compile(r"(?i)\b(?:today is|current date|the date is|as of \w+ \d{1,2},? \d{4}|generated at|generated on)\b")
_NONCE = re.compile(r"(?i)\b(?:nonce|random seed|seed\s*[:=]\s*\d+)\b")
_JSON_OBJECT = re.compile(r"\{[^{}]{2,}\}")


class Invalidator(BaseModel):
    """A fragment inside a stable block that changes between requests."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    block_index: int = Field(description="Index into system_stable (-1 = tools)")
    snippet: str
    advice: str


class PrefixReport(BaseModel):
    """Result of analysing a request's cacheable prefix."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prefix_tokens: int
    min_cacheable_tokens: int
    meets_minimum: bool
    stable_blocks: int
    volatile_tokens: int = Field(description="Tokens after the breakpoint (volatile system + messages)")
    invalidators: list[Invalidator] = Field(default_factory=list)
    cache_friendly: bool
    recommended_ttl: TTLRecommendation
    breakeven_uses: int
    write_cost_usd: float = Field(description="Cost of one cache write of the prefix at the recommended TTL")
    savings_per_reuse_usd: float = Field(description="Input-cost saved by one cache read versus an uncached call")
    reference_tokens_in_turn: int = Field(default=0, description="Document tokens found in the user turn that could be promoted to the prefix")
    score: float = Field(ge=0.0, le=1.0, description="0 = cannot cache, 1 = ideal layout")
    recommendations: list[str] = Field(default_factory=list)


class ArrivalStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix_fingerprint: str
    arrivals: int
    median_gap_s: Optional[float]
    reuses_per_hour: Optional[float]
    recommended_ttl: TTLRecommendation


def breakeven_uses(write_multiplier: float, read_multiplier: float = 0.1) -> int:
    """Smallest N such that ``write + (N-1) * read < N`` (uncached cost N).

    With 1.25x write / 0.1x read caching pays from N=2; with the 2x one-hour
    write it pays from N=3.
    """
    n = 1
    while n < 10_000:
        if write_multiplier + (n - 1) * read_multiplier < n:
            return n
        n += 1
    return n


_ONE_HOUR_S = 3600.0
_KEEPALIVE_INTERVAL_S = 3300.0  # ping a little inside the 1-hour TTL


def recommend_ttl_from_gaps(
    gaps_s: Sequence[float],
    *,
    promote_gap_s: float = 300.0,
    min_reuses_per_hour: int = 3,
    read_multiplier: float = 0.1,
) -> TTLRecommendation:
    """Pick a provider TTL from inter-arrival gaps of the same prefix.

    Cache reads refresh the TTL, so a TTL only has to bridge one gap at a time.

    * median gap <= ``promote_gap_s`` (5 min): the 5-minute cache bridges every
      gap, so ``5m``;
    * 5 min < median gap <= 60 min: the 1-hour cache bridges every gap and the
      2x write pays off from the third use, so ``1h`` once the prefix has been
      observed at least ``min_reuses_per_hour`` times (``len(gaps) + 1``;
      the name is kept for compatibility, it is the minimum number of
      observed uses); with fewer observations stay on ``5m``;
    * median gap > 60 min: no TTL bridges the gap. ``keep-alive`` (a cheap
      read every ~55 minutes) is recommended while ``(1 + pings) * read``
      stays below the uncached cost of one use, otherwise ``none``.
    """
    if not gaps_s:
        return "5m"
    med = statistics.median(gaps_s)
    if med <= promote_gap_s:
        return "5m"
    enough_uses = len(gaps_s) + 1 >= min_reuses_per_hour
    if med <= _ONE_HOUR_S:
        return "1h" if enough_uses else "5m"
    pings = max(1, math.ceil(med / _KEEPALIVE_INTERVAL_S) - 1)
    if enough_uses and (1 + pings) * read_multiplier < 1.0:
        return "keep-alive"
    return "none"


def prefix_hit_ratio(usage: Usage) -> float:
    """``cache_read / (cache_read + cache_creation + input)`` - the provider hit metric."""
    total = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    return usage.cache_read_input_tokens / total if total else 0.0


def verify_hit(usage: Usage) -> bool:
    """True when the provider reported cached prefix tokens."""
    return usage.cache_read_input_tokens > 0


def find_invalidators(blocks: Sequence[str], *, tools: Sequence[dict[str, Any]] = ()) -> list[Invalidator]:
    """Lint stable blocks (and tool definitions) for content that silently breaks prefix matching."""
    found: list[Invalidator] = []

    def add(kind: str, idx: int, snippet: str, advice: str) -> None:
        found.append(Invalidator(kind=kind, block_index=idx, snippet=snippet[:80], advice=advice))

    for i, block in enumerate(blocks):
        for kind, pat in VOLATILE_PATTERNS.items():
            m = pat.search(block)
            if m:
                add(kind, i, m.group(0), "move this value after the cache breakpoint (user turn or system_volatile)")
        m = _DATE_ONLY.search(block)
        if m and not VOLATILE_PATTERNS["iso_timestamp"].search(block):
            add("date", i, m.group(0), "dates change daily; render them in the volatile turn")
        m = _DATE_WORDS.search(block)
        if m:
            add("date_phrase", i, m.group(0), "remove the current-date phrase from the stable prefix")
        m = _NONCE.search(block)
        if m:
            add("nonce", i, m.group(0), "random values in the prefix guarantee a miss every call")
    if tools:
        raw = json.dumps(list(tools), sort_keys=False)
        for kind in ("iso_timestamp", "uuid", "request_id"):
            m = VOLATILE_PATTERNS[kind].search(raw)
            if m:
                add(kind, -1, m.group(0), "tool definitions sit before the system prompt; any change invalidates everything")
    return found


def find_unsorted_json(blocks: Sequence[str]) -> list[tuple[int, str]]:
    """``(block_index, snippet)`` of embedded JSON objects whose keys are not sorted.

    Informational only: a constant JSON literal is byte-stable and never breaks
    the prefix (the package's own few-shot examples contain such literals).
    It matters only when the block is re-rendered from a dict at runtime, which
    a linter cannot see, so it never counts as an invalidator.
    """
    out: list[tuple[int, str]] = []
    for i, block in enumerate(blocks):
        for jm in _JSON_OBJECT.finditer(block):
            try:
                obj = json.loads(jm.group(0))
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and len(obj) >= 2 and list(obj) != sorted(obj):
                out.append((i, jm.group(0)[:80]))
                break
    return out


class PrefixCacheAdvisor:
    """Analyse, recommend, prewarm and verify provider prefix caching."""

    def __init__(self, config: CachingConfig | None = None, *, clock: Clock = time.time) -> None:
        self.config = config or CachingConfig()
        self.clock = clock
        self._arrivals: dict[str, list[float]] = {}

    # ----------------------------------------------------------- analysis
    def analyze(self, request: LLMRequest, *, model: str | None = None, expected_reuses: int | None = None) -> PrefixReport:
        cfg = self.config
        model_id = model or request.model or DEFAULT_MODEL
        prefix_tokens = request.prefix_tokens()
        min_tokens = cfg.prefix_min_tokens or MIN_CACHEABLE_TOKENS.get(model_id, 512)
        meets = prefix_tokens >= min_tokens
        volatile_tokens = estimate_tokens(request.system_volatile or "") + sum(estimate_tokens(m.text()) for m in request.messages)
        invalidators = find_invalidators(request.system_stable, tools=request.tools)
        ref_tokens = self._reference_tokens_in_turn(request)
        fp = request.prefix_fingerprint(model_id)
        gaps = self._gaps(fp)
        ttl: TTLRecommendation = (
            recommend_ttl_from_gaps(gaps, promote_gap_s=cfg.promote_to_1h_gap_s, min_reuses_per_hour=cfg.promote_to_1h_min_reuses_per_hour, read_multiplier=cfg.read_multiplier)
            if gaps
            else cfg.prefix_default_ttl
        )
        if not meets:
            ttl_for_cost = ttl
            ttl = "none"
        else:
            ttl_for_cost = ttl
        write_mult = cfg.write_multiplier_1h if ttl_for_cost == "1h" else cfg.write_multiplier_5m
        be = breakeven_uses(write_mult, cfg.read_multiplier)
        price = PRICING_USD_PER_MTOK.get(model_id, {"input": 0.0, "cache_read": 0.0, "cache_write": 0.0})
        write_cost = prefix_tokens * (price["input"] * write_mult) / 1_000_000 if meets else 0.0
        savings = prefix_tokens * (price["input"] - price["input"] * cfg.read_multiplier) / 1_000_000 if meets else 0.0

        recs: list[str] = []
        if not request.system_stable:
            recs.append("No stable system block: move role, rules, domain profile and examples into system_stable so a cache breakpoint can be placed.")
        if not meets:
            recs.append(f"Stable prefix is ~{prefix_tokens} tokens, below the {min_tokens}-token minimum for {model_id}; cache_control would be inert. Add reference material or few-shot examples to the prefix, or skip prefix caching.")
        if invalidators:
            kinds = sorted({i.kind for i in invalidators})
            recs.append(f"Stable blocks contain volatile content ({', '.join(kinds)}); every request would write a fresh cache entry at {write_mult}x and never hit.")
        unsorted = find_unsorted_json(request.system_stable)
        if unsorted:
            recs.append(
                f"Stable block(s) {sorted({i for i, _ in unsorted})} embed JSON with unsorted keys; harmless if the text is a constant, "
                "but if it is rendered from a dict at runtime serialise it with sort_keys=True so the bytes never change."
            )
        if ref_tokens >= min_tokens // 2:
            recs.append(f"About {ref_tokens} tokens of documents sit in the user turn; if they are reused across queries, render them as a stable system block (1h TTL) so prefix caching covers them.")
        if meets and not invalidators:
            recs.append(f"Prefix of ~{prefix_tokens} tokens is cacheable; break-even at {be} uses within the TTL (write {write_mult}x, read {cfg.read_multiplier}x).")
        if ttl == "1h":
            recs.append("Observed reuse gaps exceed five minutes with regular reuse: request cache_ttl='1h'.")
        elif ttl == "keep-alive":
            recs.append("Reuse gaps exceed one hour: request cache_ttl='1h' and send a cheap keep-alive read about every 55 minutes (each read refreshes the TTL), or accept misses.")
        if not request.cache:
            recs.append("request.cache is False: enable it to place the breakpoint.")

        score = 0.0
        if request.system_stable:
            score += 0.3
        if meets:
            score += 0.4
        if not invalidators:
            score += 0.2
        if request.cache:
            score += 0.1
        return PrefixReport(
            model=model_id,
            prefix_tokens=prefix_tokens,
            min_cacheable_tokens=min_tokens,
            meets_minimum=meets,
            stable_blocks=len(request.system_stable),
            volatile_tokens=volatile_tokens,
            invalidators=invalidators,
            cache_friendly=meets and not invalidators and request.cache,
            recommended_ttl=ttl,
            breakeven_uses=be,
            write_cost_usd=round(write_cost, 8),
            savings_per_reuse_usd=round(savings, 8),
            reference_tokens_in_turn=ref_tokens,
            score=round(score, 3),
            recommendations=recs,
        )

    def analyze_template(self, template: PromptTemplate, *, extra_system: Sequence[str] = (), model: str | None = None, **values: Any) -> PrefixReport:
        """Analyse a :class:`PromptTemplate` rendered with placeholder ``values`` (missing ones are filled with their names)."""
        filled = {p: values.get(p, f"<{p}>") for p in template.placeholders()}
        rendered = template.render(extra_system=extra_system, **filled)
        req = LLMRequest.simple(rendered.user, system=rendered.system_blocks, schema=rendered.output_schema)
        return self.analyze(req, model=model)

    @staticmethod
    def _reference_tokens_in_turn(request: LLMRequest) -> int:
        text = request.last_user_text()
        docs = re.findall(r"<document\b.*?</document>", text, flags=re.DOTALL)
        return sum(estimate_tokens(d) for d in docs)

    # ------------------------------------------------------------ arrivals
    def observe(self, request: LLMRequest, *, model: str | None = None, at: float | None = None) -> str:
        """Record an arrival of this request's prefix; returns the prefix fingerprint."""
        fp = request.prefix_fingerprint(model or request.model or DEFAULT_MODEL)
        self._arrivals.setdefault(fp, []).append(self.clock() if at is None else at)
        return fp

    def _gaps(self, fp: str) -> list[float]:
        ts = sorted(self._arrivals.get(fp, []))
        return [b - a for a, b in zip(ts, ts[1:]) if b > a]

    def arrival_stats(self, request_or_fp: LLMRequest | str, *, model: str | None = None) -> ArrivalStats:
        fp = request_or_fp if isinstance(request_or_fp, str) else request_or_fp.prefix_fingerprint(model or request_or_fp.model or DEFAULT_MODEL)
        gaps = self._gaps(fp)
        med = statistics.median(gaps) if gaps else None
        return ArrivalStats(
            prefix_fingerprint=fp,
            arrivals=len(self._arrivals.get(fp, [])),
            median_gap_s=med,
            reuses_per_hour=(3600.0 / med) if med else None,
            recommended_ttl=recommend_ttl_from_gaps(
                gaps,
                promote_gap_s=self.config.promote_to_1h_gap_s,
                min_reuses_per_hour=self.config.promote_to_1h_min_reuses_per_hour,
                read_multiplier=self.config.read_multiplier,
            ),
        )

    def recommend_ttl(self, request: LLMRequest, *, model: str | None = None) -> TTLRecommendation:
        return self.arrival_stats(request, model=model).recommended_ttl

    # ----------------------------------------------------- prewarm / verify
    async def prewarm(self, llm: BaseLLMClient, request: LLMRequest) -> Usage:
        """Write the provider cache for ``request``'s prefix.

        Uses ``llm.prewarm`` when the client offers it (``AnthropicLLM``);
        otherwise returns an empty :class:`Usage` (mock/offline clients have no
        provider cache to warm).
        """
        fn = getattr(llm, "prewarm", None)
        if fn is None:
            return Usage()
        return await fn(request)

    @staticmethod
    def verify_hit(response: LLMResponse | Usage) -> bool:
        usage = response if isinstance(response, Usage) else response.usage
        return verify_hit(usage)

    @staticmethod
    def hit_ratio(usage: Usage) -> float:
        return prefix_hit_ratio(usage)

    async def warm_then_fan_out(
        self,
        call: Callable[[LLMRequest], Awaitable[LLMResponse]],
        requests: Sequence[LLMRequest],
        *,
        concurrency: int = 8,
    ) -> list[LLMResponse]:
        """Send the first request alone (so the provider cache entry exists), then the rest concurrently.

        This is the Anthropic guidance that a cache entry only becomes
        available after the first response begins; parallel cold requests each
        pay the write price and never hit.
        """
        from pipeline_core import gather_limited

        if not requests:
            return []
        first = await call(requests[0])
        rest = await gather_limited((call(r) for r in requests[1:]), max(1, concurrency), return_exceptions=False)
        return [first, *rest]  # type: ignore[list-item]


__all__ = [
    "ArrivalStats",
    "Invalidator",
    "PrefixCacheAdvisor",
    "PrefixReport",
    "TTLRecommendation",
    "breakeven_uses",
    "find_invalidators",
    "find_unsorted_json",
    "prefix_hit_ratio",
    "recommend_ttl_from_gaps",
    "verify_hit",
]
