"""Language-model client abstraction.

Components never talk to a provider SDK directly. They build an
:class:`LLMRequest` and hand it to any :class:`BaseLLMClient`:

* :class:`AnthropicLLM` - production client on the official ``anthropic`` SDK.
  Defaults to ``claude-opus-5-5`` with adaptive thinking, explicit effort,
  structured outputs through ``output_config.format``, prompt caching through
  ``cache_control`` on the last stable system block, and server-side refusal
  fallbacks.
* :class:`MockLLM` - deterministic offline client. It fabricates
  schema-conformant JSON when a schema is requested, so every component's
  tests and demos run without credentials.
* :class:`ScriptedLLM`, :class:`RecordingLLM`, :class:`ReplayLLM` - test
  doubles for golden-file testing of prompts.

Why requests separate ``system_stable`` from ``system_volatile``
---------------------------------------------------------------
Claude's prompt cache is a prefix match. Stable blocks (role, rules, domain
profile, few-shot examples) go first and get a ``cache_control`` breakpoint;
anything that changes per request (retrieved documents, the question) goes
after it. Getting this ordering right is worth up to 90% of input cost on
repeated calls; getting it wrong silently disables caching.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Usage
from .errors import LLMError, LLMRefusalError, LLMTruncatedError, StructuredOutputError
from .utils import canonical_json, estimate_tokens, safe_json_loads, stable_hash

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
Effort = Literal["low", "medium", "high", "xhigh", "max"]

# USD per million tokens. Cache write = 1.25x input (5-minute TTL).
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25, "cache_write": 12.50},
}

# Minimum cacheable prefix per model (tokens). Shorter prefixes silently do not cache.
MIN_CACHEABLE_TOKENS: dict[str, int] = {
    "claude-opus-5-5": 512,
    "claude-opus-5": 512,
    "claude-sonnet-5-5": 512,
    "claude-haiku-5-5": 512,
    "claude-fable-5-1": 512,
    "claude-sonnet-5": 1024,
    "claude-opus-4-8": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-haiku-4-5": 4096,
}


def estimate_cost(model: str, usage: Usage) -> float:
    p = PRICING_USD_PER_MTOK.get(model)
    if p is None:
        return 0.0
    return (
        usage.input_tokens * p["input"]
        + usage.output_tokens * p["output"]
        + usage.cache_read_input_tokens * p["cache_read"]
        + usage.cache_creation_input_tokens * p["cache_write"]
    ) / 1_000_000


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: Any = Field(..., description="str or list of provider content blocks")

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(b.get("text", "") for b in self.content if isinstance(b, dict) and b.get("type") == "text")


class LLMRequest(BaseModel):
    """Provider-neutral request."""

    model_config = ConfigDict(extra="forbid")

    system_stable: list[str] = Field(default_factory=list, description="Cacheable system blocks, most stable first")
    system_volatile: Optional[str] = Field(default=None, description="Per-request system text placed after the cache breakpoint")
    messages: list[Message] = Field(default_factory=list)
    output_schema: Optional[dict[str, Any]] = Field(default=None, description="JSON schema for structured output")
    max_tokens: int = Field(default=16000, ge=1)
    effort: Optional[Effort] = None
    model: Optional[str] = Field(default=None, description="Override the client's default model")
    tools: list[dict[str, Any]] = Field(default_factory=list)
    cache: bool = Field(default=True, description="Place a cache_control breakpoint on the last stable system block")
    cache_ttl: Literal["5m", "1h"] = "5m"
    stream: Optional[bool] = Field(default=None, description="None = stream automatically when max_tokens > 16000")
    tags: list[str] = Field(default_factory=list, description="Free-form labels for tracing")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def simple(
        cls,
        prompt: str,
        *,
        system: Union[str, Sequence[str], None] = None,
        schema: Optional[dict[str, Any]] = None,
        history: Optional[Sequence[Message]] = None,
        **kw: Any,
    ) -> "LLMRequest":
        stable: list[str]
        if system is None:
            stable = []
        elif isinstance(system, str):
            stable = [system]
        else:
            stable = [s for s in system if s]
        msgs = list(history or []) + [Message(role="user", content=prompt)]
        return cls(system_stable=stable, messages=msgs, output_schema=schema, **kw)

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.text()
        return ""

    def fingerprint(self, model: str | None = None) -> str:
        """Hash of everything that influences the answer (for exact-match caches)."""
        return stable_hash(
            model or self.model,
            self.system_stable,
            self.system_volatile,
            [m.model_dump() for m in self.messages],
            self.output_schema,
            self.effort,
            self.max_tokens,
            self.tools,
        )

    def prefix_fingerprint(self, model: str | None = None) -> str:
        """Hash of the provider-cacheable prefix (model, tools, stable system)."""
        return stable_hash(model or self.model, self.tools, self.system_stable)

    def prefix_tokens(self) -> int:
        return estimate_tokens("\n".join(self.system_stable)) + estimate_tokens(canonical_json(self.tools))


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    text: str
    parsed: Any = None
    stop_reason: str = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    response_id: Optional[str] = None
    cached: bool = False
    fallback_used: bool = False
    latency_s: float = 0.0
    request_fingerprint: Optional[str] = None
    raw: Any = Field(default=None, exclude=True)


ResponseHook = Callable[[LLMRequest, LLMResponse], None]


def _schema_from(schema: Any) -> Optional[dict[str, Any]]:
    """Accept a dict JSON schema or a pydantic model class."""
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return sanitize_schema(schema.model_json_schema())
    raise TypeError(f"unsupported schema type: {type(schema)!r}")


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON schema acceptable for strict structured output.

    Sets ``additionalProperties: false`` on every object, drops ``title`` and
    ``default`` keys (which constrained decoders reject or ignore), and leaves
    ``$defs``/``$ref`` intact.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k, v in node.items():
                if k in ("title", "default", "examples"):
                    continue
                out[k] = walk(v) if k not in ("enum", "const", "required") else v
            if out.get("type") == "object" and "additionalProperties" not in out:
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def example_from_schema(schema: dict[str, Any], *, root: dict[str, Any] | None = None, depth: int = 0) -> Any:
    """Build a deterministic instance that satisfies ``schema`` (used by :class:`MockLLM`)."""
    root = root or schema
    if depth > 12:
        return None
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/"):
            node: Any = root
            for part in ref[2:].split("/"):
                node = node[part]
            return example_from_schema(node, root=root, depth=depth + 1)
        return None
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and schema[key]:
            options = [o for o in schema[key] if o.get("type") != "null"] or schema[key]
            return example_from_schema(options[0], root=root, depth=depth + 1)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0])
    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = schema.get("required", list(props))
        return {k: example_from_schema(v, root=root, depth=depth + 1) for k, v in props.items() if k in required or depth == 0}
    if t == "array":
        n = int(schema.get("minItems", 0))
        item = schema.get("items", {"type": "string"})
        return [example_from_schema(item, root=root, depth=depth + 1) for _ in range(n)]
    if t == "string":
        fmt = schema.get("format")
        if fmt == "date":
            return "2026-01-01"
        if fmt == "date-time":
            return "2026-01-01T00:00:00Z"
        desc = schema.get("description")
        return f"mock {desc[:60]}" if desc else "mock"
    if t == "integer":
        return int(schema.get("minimum", 0))
    if t == "number":
        return float(schema.get("minimum", 0.0))
    if t == "boolean":
        return True
    if t == "null":
        return None
    return None


class BaseLLMClient(ABC):
    """Common behaviour: timing, parsing, validation, hooks."""

    name: str = "llm"

    def __init__(self, model: str = DEFAULT_MODEL, *, effort: Effort = "high") -> None:
        self.model = model
        self.effort: Effort = effort
        self.hooks: list[ResponseHook] = []
        self.total_usage = Usage()

    # --------------------------------------------------------------- public
    async def complete(self, request: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        resp = await self._complete(request)
        resp.latency_s = resp.latency_s or (time.perf_counter() - t0)
        resp.usage.latency_s = resp.latency_s
        resp.request_fingerprint = request.fingerprint(self.model)
        if request.output_schema is not None and resp.parsed is None and resp.text:
            try:
                resp.parsed = safe_json_loads(resp.text)
            except (json.JSONDecodeError, ValueError) as exc:
                raise StructuredOutputError(f"model output is not valid JSON: {exc}; text[:200]={resp.text[:200]!r}") from exc
        self.total_usage = self.total_usage + resp.usage
        for hook in self.hooks:
            try:
                hook(request, resp)
            except Exception:  # noqa: BLE001 - hooks must never break the call
                log.exception("LLM response hook failed")
        return resp

    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        from .utils import run_sync

        return run_sync(self.complete(request))

    async def generate(
        self,
        prompt: str,
        *,
        system: Union[str, Sequence[str], None] = None,
        schema: Any = None,
        history: Optional[Sequence[Message]] = None,
        **kw: Any,
    ) -> LLMResponse:
        req = LLMRequest.simple(prompt, system=system, schema=_schema_from(schema), history=history, **kw)
        return await self.complete(req)

    async def structured(self, prompt: str, schema: Any, *, system: Union[str, Sequence[str], None] = None, **kw: Any) -> Any:
        """Return parsed structured output; validates into ``schema`` when it is a pydantic class."""
        resp = await self.generate(prompt, system=system, schema=schema, **kw)
        return validate_structured(resp.parsed, schema)

    # ------------------------------------------------------------ abstract
    @abstractmethod
    async def _complete(self, request: LLMRequest) -> LLMResponse: ...

    def describe(self) -> dict[str, Any]:
        return {"client": self.name, "model": self.model, "effort": self.effort}


def is_offline_client(llm: Any, *, max_depth: int = 8) -> bool:
    """True when ``llm`` is, or wraps, a deterministic offline client.

    Wrapper clients (response caches, recorders, middleware) expose the wrapped client
    as ``inner``. Offline detection must look through them, otherwise a cached
    :class:`MockLLM` looks "online" and components parse empty schema stubs instead of
    using their heuristic planners. A client may also declare ``offline = True``.
    """
    seen = 0
    while llm is not None and seen <= max_depth:
        if isinstance(llm, MockLLM) or getattr(llm, "offline", None) is True:
            return True
        llm = getattr(llm, "inner", None)
        seen += 1
    return False


def validate_structured(parsed: Any, schema: Any) -> Any:
    if hasattr(schema, "model_validate"):
        try:
            return schema.model_validate(parsed)
        except Exception as exc:  # pydantic.ValidationError
            raise StructuredOutputError(f"structured output failed validation: {exc}") from exc
    return parsed


# --------------------------------------------------------------------------
# Production client
# --------------------------------------------------------------------------


class AnthropicLLM(BaseLLMClient):
    """Claude via the official ``anthropic`` SDK (async).

    Parameters
    ----------
    model:
        Defaults to ``claude-opus-5-5``. Use the exact IDs from the pricing
        table; never append date suffixes.
    effort:
        ``output_config.effort``. Opus 5.5 defaults to ``medium`` server-side,
        so we always send it explicitly. ``high`` is the quality/cost sweet
        spot; use ``low`` for cheap sub-agents and ``max`` when correctness
        dominates cost.
    use_fallbacks:
        Sends ``fallbacks="default"`` with the ``server-side-fallback-2026-07-01``
        beta so a safety-classifier refusal is retried server-side on a fallback
        model. Disable on platforms that reject the parameter (Bedrock, Vertex).
    thinking:
        ``"adaptive"`` (default) or ``None`` to omit the parameter. Thinking is
        always on for Opus 5.5; disabling it returns HTTP 400.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: Effort = "high",
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 600.0,
        use_fallbacks: bool = True,
        thinking: Optional[Literal["adaptive"]] = "adaptive",
        thinking_display: Optional[Literal["summarized", "omitted"]] = None,
        client: Any = None,
        pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        super().__init__(model, effort=effort)
        try:
            import anthropic  # noqa: WPS433 - optional dependency resolved lazily
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise LLMError("the 'anthropic' package is required for AnthropicLLM (pip install anthropic)") from exc
        self._anthropic = anthropic
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self.use_fallbacks = use_fallbacks
        self.thinking = thinking
        self.thinking_display = thinking_display
        self.pricing = pricing or PRICING_USD_PER_MTOK

    # ---------------------------------------------------------- request build
    def build_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model or self.model
        system_blocks: list[dict[str, Any]] = []
        stable = [s for s in request.system_stable if s]
        for i, text in enumerate(stable):
            block: dict[str, Any] = {"type": "text", "text": text}
            if request.cache and i == len(stable) - 1:
                cc: dict[str, Any] = {"type": "ephemeral"}
                if request.cache_ttl == "1h":
                    cc["ttl"] = "1h"
                block["cache_control"] = cc
            system_blocks.append(block)
        if request.system_volatile:
            system_blocks.append({"type": "text", "text": request.system_volatile})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [m.model_dump() for m in request.messages],
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        output_config: dict[str, Any] = {"effort": request.effort or self.effort}
        if request.output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.output_schema}
        kwargs["output_config"] = output_config
        if self.thinking:
            thinking: dict[str, Any] = {"type": self.thinking}
            if self.thinking_display:
                thinking["display"] = self.thinking_display
            kwargs["thinking"] = thinking
        if request.tools:
            kwargs["tools"] = request.tools
        if request.cache and request.prefix_tokens() < MIN_CACHEABLE_TOKENS.get(model, 512):
            log.debug("prefix (~%d tokens) is below the cacheable minimum for %s; cache_control is harmless but inert", request.prefix_tokens(), model)
        return kwargs

    # ------------------------------------------------------------- execution
    async def _complete(self, request: LLMRequest) -> LLMResponse:
        kwargs = self.build_kwargs(request)
        stream = request.stream if request.stream is not None else request.max_tokens > 16000
        anthropic = self._anthropic
        if self.use_fallbacks:
            api = self._client.beta.messages
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        else:
            api = self._client.messages
        try:
            if stream:
                async with api.stream(**kwargs) as s:
                    msg = await s.get_final_message()
            else:
                msg = await api.create(**kwargs)
        except anthropic.BadRequestError as exc:
            raise LLMError(f"bad request: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("authentication failed: set ANTHROPIC_API_KEY or run 'ant auth login'") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited after SDK retries: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc

        text_parts: list[str] = []
        fallback_used = False
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "fallback":
                fallback_used = True
        text = "".join(text_parts)

        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise LLMRefusalError(
                "model refused the request",
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
            )
        if msg.stop_reason == "max_tokens" and request.output_schema is not None:
            raise LLMTruncatedError(f"structured output truncated at max_tokens={request.max_tokens}; raise max_tokens or stream")

        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            llm_calls=1,
        )
        usage.cost_usd = estimate_cost(msg.model, usage) if msg.model in self.pricing else estimate_cost(kwargs["model"], usage)
        parsed = None
        if request.output_schema is not None and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None  # BaseLLMClient.complete retries with the tolerant parser
        return LLMResponse(
            text=text,
            parsed=parsed,
            stop_reason=msg.stop_reason or "end_turn",
            usage=usage,
            model=msg.model,
            response_id=msg.id,
            fallback_used=fallback_used,
            raw=msg,
        )

    async def count_tokens(self, request: LLMRequest) -> int:
        kwargs = self.build_kwargs(request)
        kwargs.pop("max_tokens", None)
        kwargs.pop("output_config", None)
        res = await self._client.messages.count_tokens(**kwargs)
        return int(res.input_tokens)

    async def prewarm(self, request: LLMRequest) -> Usage:
        """Write the cache for ``request``'s stable prefix with a ``max_tokens=0`` call."""
        kwargs = self.build_kwargs(request)
        kwargs["max_tokens"] = 0
        kwargs.pop("output_config", None)
        kwargs["output_config"] = {"effort": request.effort or self.effort}
        kwargs["messages"] = [{"role": "user", "content": "warmup"}]
        kwargs.pop("stream", None)
        msg = await self._client.messages.create(**kwargs)
        u = msg.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            llm_calls=1,
        )
        usage.cost_usd = estimate_cost(kwargs["model"], usage)
        return usage


# --------------------------------------------------------------------------
# Offline / test clients
# --------------------------------------------------------------------------

Responder = Callable[[LLMRequest], Union[str, dict, list, LLMResponse]]


class MockLLM(BaseLLMClient):
    """Deterministic offline client.

    ``responder(request)`` may return a string, a JSON-serialisable object
    (used as the parsed structured output) or a full :class:`LLMResponse`.
    Without a responder, structured requests receive a schema-conformant
    example and free-text requests receive a short echo. Usage is estimated
    so budgeting code paths are exercised; cost is reported as zero.
    """

    name = "mock"

    def __init__(
        self,
        responder: Optional[Responder] = None,
        *,
        model: str = "mock-llm",
        effort: Effort = "high",
        latency_s: float = 0.0,
        simulate_cost_as: str | None = None,
    ) -> None:
        super().__init__(model, effort=effort)
        self.responder = responder
        self.latency_s = latency_s
        self.simulate_cost_as = simulate_cost_as
        self.calls: list[LLMRequest] = []

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        out: Any
        if self.responder is not None:
            out = self.responder(request)
        elif request.output_schema is not None:
            out = example_from_schema(request.output_schema)
        else:
            out = f"[mock:{self.model}] {request.last_user_text()[:300]}"
        if isinstance(out, LLMResponse):
            resp = out
        elif isinstance(out, str):
            resp = LLMResponse(text=out, parsed=None)
        else:
            resp = LLMResponse(text=json.dumps(out, ensure_ascii=False), parsed=out)
        prompt_tokens = estimate_tokens("\n".join(request.system_stable) + (request.system_volatile or "") + "\n".join(m.text() for m in request.messages))
        resp.usage = Usage(input_tokens=prompt_tokens, output_tokens=estimate_tokens(resp.text), llm_calls=1)
        if self.simulate_cost_as:
            resp.usage.cost_usd = estimate_cost(self.simulate_cost_as, resp.usage)
        resp.model = resp.model or self.model
        resp.stop_reason = resp.stop_reason or "end_turn"
        return resp

    def reset(self) -> None:
        self.calls.clear()
        self.total_usage = Usage()


class ScriptedLLM(MockLLM):
    """Returns queued responses in order; useful for deterministic multi-step tests."""

    def __init__(self, responses: Sequence[Union[str, dict, list, LLMResponse]], *, loop: bool = False, **kw: Any) -> None:
        self._responses = list(responses)
        self._i = 0
        self._loop = loop
        super().__init__(responder=self._next, **kw)

    def _next(self, request: LLMRequest) -> Union[str, dict, list, LLMResponse]:
        if self._i >= len(self._responses):
            if not self._loop:
                raise LLMError(f"ScriptedLLM exhausted after {len(self._responses)} responses")
            self._i = 0
        out = self._responses[self._i]
        self._i += 1
        return out


class RecordingLLM(BaseLLMClient):
    """Wrap a real client and persist (fingerprint -> response) for later replay."""

    name = "recording"

    def __init__(self, inner: BaseLLMClient, path: str | Path) -> None:
        super().__init__(inner.model, effort=inner.effort)
        self.inner = inner
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        resp = await self.inner.complete(request)
        self.records[request.fingerprint(self.inner.model)] = resp.model_dump(mode="json", exclude={"raw"})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=1, ensure_ascii=False), encoding="utf-8")
        return resp


class ReplayLLM(BaseLLMClient):
    """Serve recorded responses; fall back to :class:`MockLLM` (or raise) on a miss."""

    name = "replay"

    def __init__(self, path: str | Path, *, model: str = DEFAULT_MODEL, strict: bool = False) -> None:
        super().__init__(model)
        self.records: dict[str, dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
        self.strict = strict
        self._fallback = MockLLM(model=model)

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        rec = self.records.get(request.fingerprint(self.model))
        if rec is None:
            if self.strict:
                raise LLMError("no recorded response for request")
            return await self._fallback._complete(request)
        resp = LLMResponse.model_validate(rec)
        resp.cached = True
        return resp
