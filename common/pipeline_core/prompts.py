"""Prompt templates with cache-aware structure.

The single most important rule for cost on Claude is that prompt caching is a
*prefix match*: any byte that changes early in the prompt invalidates every
cached token after it. Templates therefore separate:

* ``system`` - stable instructions, rendered first, cacheable;
* ``instructions`` - the per-request task, rendered last in the user turn.

Templates are versioned and fingerprinted so that caches can be keyed on the
exact prompt text that produced a result (see the caching model).

Prompting conventions encoded here follow Anthropic's prompt engineering
guidance and the literature on structured prompting: explicit role and task
statement, XML-delimited inputs, numbered procedure, explicit output format,
and examples placed before the volatile request.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Evidence
from .utils import stable_hash

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def xml(tag: str, content: str, **attrs: Any) -> str:
    """Wrap ``content`` in an XML element. Attribute values are escaped minimally."""
    attr_s = "".join(f' {k}="{str(v).replace(chr(34), "&quot;")}"' for k, v in attrs.items() if v is not None)
    return f"<{tag}{attr_s}>\n{content}\n</{tag}>"


def bullet_list(items: Iterable[str], marker: str = "-") -> str:
    return "\n".join(f"{marker} {it}" for it in items)


def numbered_list(items: Iterable[str]) -> str:
    return "\n".join(f"{i}. {it}" for i, it in enumerate(items, 1))


def render_evidence(evidence: Sequence[Evidence], *, max_chars_each: int = 2000, max_items: int | None = None) -> str:
    """Render evidence as ``<documents>`` so the model can cite by id."""
    items = list(evidence)[:max_items] if max_items else list(evidence)
    parts = []
    for i, ev in enumerate(items, 1):
        body = ev.content if len(ev.content) <= max_chars_each else ev.content[: max_chars_each - 3] + "..."
        parts.append(xml("document", body, index=i, id=ev.id, source=ev.source or None, score=f"{ev.score:.3f}" if ev.score else None))
    return xml("documents", "\n".join(parts)) if parts else "<documents></documents>"


class RenderedPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_blocks: list[str]
    user: str
    output_schema: Optional[dict[str, Any]] = None
    template: str
    version: str
    fingerprint: str


class PromptTemplate(BaseModel):
    """A versioned prompt with a stable system part and a templated user part."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0.0"
    system: str = Field(default="", description="Stable instructions; rendered first and cached")
    instructions: str = Field(..., description="User-turn template with {placeholders}")
    examples: list[tuple[str, str]] = Field(default_factory=list, description="(input, output) few-shot pairs, rendered after system")
    output_schema: Optional[dict[str, Any]] = Field(default=None, description="JSON schema for structured output")
    description: str = ""

    def placeholders(self) -> list[str]:
        return sorted(set(_PLACEHOLDER.findall(self.instructions)))

    def fingerprint(self) -> str:
        return stable_hash(self.name, self.version, self.system, self.instructions, self.examples, self.output_schema)

    def render(self, *, extra_system: Sequence[str] = (), **values: Any) -> RenderedPrompt:
        missing = [p for p in self.placeholders() if p not in values]
        if missing:
            raise KeyError(f"prompt '{self.name}' missing values for {missing}")
        user = self.instructions
        for k, v in values.items():
            user = user.replace("{" + k + "}", str(v))
        blocks: list[str] = []
        if self.system:
            blocks.append(self.system)
        if self.examples:
            ex = "\n\n".join(xml("example", xml("input", a) + "\n" + xml("output", b)) for a, b in self.examples)
            blocks.append(xml("examples", ex))
        blocks.extend(b for b in extra_system if b)
        return RenderedPrompt(
            system_blocks=blocks,
            user=user,
            output_schema=self.output_schema,
            template=self.name,
            version=self.version,
            fingerprint=self.fingerprint(),
        )


# Reusable building blocks -------------------------------------------------

REASONING_PREAMBLE = (
    "Work through the problem carefully before answering. Consider alternative interpretations of the task, "
    "check your intermediate results, and prefer stating uncertainty over guessing."
)

GROUNDING_RULES = (
    "Ground every factual claim in the provided documents. Cite supporting documents by id in square brackets, "
    "for example [doc-3]. If the documents do not contain the information, say so explicitly instead of inventing it."
)

STRUCTURED_OUTPUT_RULES = (
    "Respond with a single JSON object that conforms exactly to the requested schema. Do not include prose, "
    "markdown fences or commentary outside the JSON."
)

ENGINEERING_RIGOR = (
    "Apply engineering rigor: keep units explicit and consistent, state assumptions, reference the governing "
    "standard or specification when one applies, distinguish measured values from estimates, and flag any "
    "safety-relevant implication."
)
