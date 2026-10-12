"""Tracing and metrics for pipeline runs.

A :class:`Tracer` records a tree of :class:`Span` objects (one per logical
operation) plus rolled-up usage. Attribute names follow the OpenTelemetry
GenAI semantic conventions (``gen_ai.request.model``,
``gen_ai.usage.input_tokens`` ...) so exports can be forwarded to any OTel
collector with a thin adapter.

The tracer is safe to share across concurrently running components: the
"current span" is tracked with :mod:`contextvars`, which asyncio copies into
every task.
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Usage
from .utils import now_iso


class Span(BaseModel):
    model_config = ConfigDict(extra="allow")

    span_id: str
    trace_id: str
    parent_id: Optional[str] = None
    name: str
    component: str = ""
    start_time: str = Field(default_factory=now_iso)
    end_time: Optional[str] = None
    duration_s: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


_CURRENT_SPAN: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar("pipeline_core_current_span", default=None)


class Tracer:
    """Collects spans, events and usage for one trace."""

    def __init__(self, trace_id: str | None = None, *, name: str = "pipeline") -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.name = name
        self.spans: list[Span] = []
        self.metrics: dict[str, float] = {}
        self.total_usage = Usage()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ spans
    @contextmanager
    def span(self, name: str, component: str = "", **attributes: Any) -> Iterator[Span]:
        parent = _CURRENT_SPAN.get()
        sp = Span(
            span_id=uuid.uuid4().hex[:16],
            trace_id=self.trace_id,
            parent_id=parent.span_id if parent else None,
            name=name,
            component=component or (parent.component if parent else ""),
            attributes=dict(attributes),
        )
        token = _CURRENT_SPAN.set(sp)
        t0 = time.perf_counter()
        try:
            yield sp
        except BaseException as exc:
            sp.status = "error"
            sp.attributes["error.type"] = type(exc).__name__
            sp.attributes["error.message"] = str(exc)[:500]
            raise
        finally:
            sp.duration_s = time.perf_counter() - t0
            sp.end_time = now_iso()
            _CURRENT_SPAN.reset(token)
            with self._lock:
                self.spans.append(sp)
                if parent is not None:
                    parent.usage = parent.usage + sp.usage

    def current_span(self) -> Optional[Span]:
        return _CURRENT_SPAN.get()

    # ------------------------------------------------------------ recording
    def record_usage(self, usage: Usage, *, model: str | None = None, span: Span | None = None) -> None:
        sp = span or _CURRENT_SPAN.get()
        with self._lock:
            self.total_usage = self.total_usage + usage
            self.metrics["gen_ai.usage.input_tokens"] = self.metrics.get("gen_ai.usage.input_tokens", 0) + usage.input_tokens
            self.metrics["gen_ai.usage.output_tokens"] = self.metrics.get("gen_ai.usage.output_tokens", 0) + usage.output_tokens
            self.metrics["gen_ai.usage.cache_read_input_tokens"] = (
                self.metrics.get("gen_ai.usage.cache_read_input_tokens", 0) + usage.cache_read_input_tokens
            )
            self.metrics["gen_ai.usage.cache_creation_input_tokens"] = (
                self.metrics.get("gen_ai.usage.cache_creation_input_tokens", 0) + usage.cache_creation_input_tokens
            )
            self.metrics["pipeline.llm_calls"] = self.metrics.get("pipeline.llm_calls", 0) + usage.llm_calls
            self.metrics["pipeline.cache_hits"] = self.metrics.get("pipeline.cache_hits", 0) + usage.cache_hits
            self.metrics["pipeline.cost_usd"] = self.metrics.get("pipeline.cost_usd", 0.0) + usage.cost_usd
        if sp is not None:
            sp.usage = sp.usage + usage
            if model:
                sp.attributes.setdefault("gen_ai.request.model", model)

    def event(self, name: str, **attributes: Any) -> None:
        sp = _CURRENT_SPAN.get()
        ev = {"name": name, "time": now_iso(), **attributes}
        if sp is not None:
            sp.events.append(ev)
        else:
            with self._lock:
                self.spans.append(
                    Span(span_id=uuid.uuid4().hex[:16], trace_id=self.trace_id, name=f"event:{name}", events=[ev], end_time=now_iso())
                )

    def increment(self, metric: str, value: float = 1.0) -> None:
        with self._lock:
            self.metrics[metric] = self.metrics.get(metric, 0.0) + value

    # --------------------------------------------------------------- export
    def export(self, path: str | Path | None = None) -> list[dict[str, Any]]:
        """Return spans as plain dicts; optionally write JSON lines to ``path``."""
        rows = [s.model_dump(mode="json") for s in self.spans]
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows

    def summary(self) -> dict[str, Any]:
        by_component: dict[str, dict[str, float]] = {}
        for s in self.spans:
            if s.parent_id is not None:
                continue  # only roots, to avoid double counting
            d = by_component.setdefault(s.component or "unknown", {"spans": 0, "duration_s": 0.0, "cost_usd": 0.0, "llm_calls": 0})
            d["spans"] += 1
            d["duration_s"] += s.duration_s
            d["cost_usd"] += s.usage.cost_usd
            d["llm_calls"] += s.usage.llm_calls
        return {
            "trace_id": self.trace_id,
            "spans": len(self.spans),
            "errors": sum(1 for s in self.spans if s.status == "error"),
            "usage": self.total_usage.model_dump(),
            "metrics": dict(self.metrics),
            "by_component": by_component,
        }

    def tree(self) -> str:
        """ASCII rendering of the span tree (useful in logs and READMEs)."""
        children: dict[Optional[str], list[Span]] = {}
        for s in self.spans:
            children.setdefault(s.parent_id, []).append(s)
        for lst in children.values():
            lst.sort(key=lambda s: s.start_time)
        lines: list[str] = []

        def walk(parent: Optional[str], depth: int) -> None:
            for s in children.get(parent, []):
                mark = "x" if s.status == "error" else "-"
                lines.append(
                    f"{'  ' * depth}{mark} {s.name} [{s.component}] {s.duration_s * 1000:.0f}ms"
                    f" calls={s.usage.llm_calls} cost=${s.usage.cost_usd:.4f}"
                )
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)


class NullTracer(Tracer):
    """Tracer that records nothing; for hot paths where tracing is disabled."""

    @contextmanager
    def span(self, name: str, component: str = "", **attributes: Any) -> Iterator[Span]:  # type: ignore[override]
        yield Span(span_id="0", trace_id=self.trace_id, name=name, component=component)

    def record_usage(self, usage: Usage, *, model: str | None = None, span: Span | None = None) -> None:  # type: ignore[override]
        self.total_usage = self.total_usage + usage

    def event(self, name: str, **attributes: Any) -> None:  # type: ignore[override]
        return None
