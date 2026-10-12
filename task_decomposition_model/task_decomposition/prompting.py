"""Helpers that turn a :class:`PromptTemplate` plus run state into an :class:`LLMRequest`.

Centralising this keeps every call cache-friendly: the template's system
blocks, the domain profile text and (for high-risk domains) the engineering
rigor rules are the stable prefix; the rendered user turn carries everything
volatile. Nothing here injects timestamps or identifiers into stable blocks.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from pipeline_core import ENGINEERING_RIGOR, DomainProfile, Evidence, LLMRequest, PromptTemplate, render_evidence

from .config import Effort


def stable_system(template: PromptTemplate, profile: DomainProfile, *, extra: Sequence[str] = ()) -> list[str]:
    """System blocks in cache order: template system, few-shot examples, profile, rigor, extras."""
    rendered = template.render(**{p: "" for p in template.placeholders()})
    blocks = list(rendered.system_blocks)
    blocks.append(profile.system_text())
    if profile.risk_level in ("high", "safety_critical"):
        blocks.append(ENGINEERING_RIGOR)
    blocks.extend(b for b in extra if b)
    return blocks


def build_request(
    template: PromptTemplate,
    *,
    profile: DomainProfile,
    values: dict[str, Any],
    schema: Optional[dict[str, Any]] = None,
    effort: Optional[Effort] = None,
    model: Optional[str] = None,
    tags: Sequence[str] = (),
    extra_system: Sequence[str] = (),
    max_tokens: int = 8000,
) -> LLMRequest:
    """Render ``template`` with ``values`` into a cache-aware request."""
    rendered = template.render(**values)
    return LLMRequest.simple(
        rendered.user,
        system=stable_system(template, profile, extra=extra_system),
        schema=schema,
        effort=effort,
        model=model,
        tags=[template.name, f"v{template.version}", *tags],
        max_tokens=max_tokens,
    )


def render_inputs(inputs: dict[str, Any], *, max_chars: int = 4000) -> str:
    """Render structured inputs as compact JSON lines (sorted keys for stability)."""
    if not inputs:
        return "(none)"
    lines: list[str] = []
    for k in sorted(inputs):
        v = inputs[k]
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        lines.append(f"{k}: {text}")
    return "\n".join(lines)


def render_documents(evidence: Sequence[Evidence], *, max_items: int = 12, max_chars_each: int = 2000) -> str:
    if not evidence:
        return ""
    return render_evidence(list(evidence), max_items=max_items, max_chars_each=max_chars_each)
