"""Exact-match key construction, normalisation and tagging.

Keys follow the vLLM block-hash design: a SHA-256 chain over the ordered
prompt segments (tools -> stable system -> volatile system -> messages) in
which every segment hash commits to the hash of everything before it, plus
``extra_keys`` for the determinants that are not literally in the prompt
(model id, prompt-template version, output schema, effort, max_tokens,
corpus snapshot, tenant salt). Forgetting a determinant is the main way a
cache serves wrong answers, so :func:`exact_key` takes them explicitly.

Normalisation is deliberately conservative: whitespace is collapsed and
labelled tracing ids (``request_id: ...``) are stripped so that true repeats
hit, but bare timestamps, clock times, UUIDs and hex ids are **kept** by
default (a different time window or object id is a different question - the
exact tier must never serve it), text is never lower-cased inside code blocks
and lower-casing of prose is opt-in because identifiers and part numbers are
case-sensitive. :data:`VOLATILE_PATTERNS` is primarily for
:func:`caching.prefix.find_invalidators`, which lints the *stable prefix*
where such values do belong to no question and only break provider caching.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Optional, Sequence

from pipeline_core import LLMRequest, canonical_json

from .config import KeyNormalization

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_WS = re.compile(r"\s+")

# Values that silently break provider prefix caching when they sit in the stable prefix.
# Only ``request_id`` (a *labelled* tracing id) is stripped from exact keys by default.
VOLATILE_PATTERNS: dict[str, re.Pattern[str]] = {
    "iso_timestamp": re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    "hex_id": re.compile(r"\b[0-9a-f]{32}\b"),
    "request_id": re.compile(r"(?i)\b(?:request|trace|session|run|correlation)[ _-]?id\s*[:=]\s*[\w-]+"),
    "clock_time": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm|UTC|Z)\b"),
}

# Engineering identifiers that embeddings blur but that must match exactly.
IDENTIFIER_PATTERNS: dict[str, re.Pattern[str]] = {
    "identifier": re.compile(r"\b[A-Z]{2,}[-_ ]?\d{2,}[A-Za-z0-9.-]*\b"),  # REQ-1234, ISO 2768, IEC 61511, ASTM A36
    "part_number": re.compile(r"\b[A-Z0-9]{2,}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b"),  # ABC-123-X
    "version": re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b"),  # 1.2.3
    "quantity": re.compile(
        r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m|km|kN|N|Nm|MPa|GPa|kPa|Pa|bar|psi|V|mV|kV|A|mA|Hz|kHz|MHz|W|kW|MW|degC|K|kg|g|rpm|%|ppm|dB|ohm|uF|nF|F)\b"
    ),
    "number": re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])"),
}

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def _stable_json_blocks(text: str) -> str:
    """Re-serialise flat JSON objects embedded in ``text`` with sorted keys."""

    def repl(m: re.Match[str]) -> str:
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return m.group(0)
        if not isinstance(obj, dict):
            return m.group(0)
        return canonical_json(obj)

    return _JSON_OBJECT.sub(repl, text)


def strip_volatile(text: str, replacement: str = "<volatile>") -> str:
    """Replace timestamps, UUIDs, hex ids, request ids and clock times (opt-in; see :class:`KeyNormalization`)."""
    out = text
    for pat in VOLATILE_PATTERNS.values():
        out = pat.sub(replacement, out)
    return out


def strip_labelled_ids(text: str, replacement: str = "<id>") -> str:
    """Replace only explicitly labelled tracing ids (``request_id: x``, ``trace_id=y``)."""
    return VOLATILE_PATTERNS["request_id"].sub(replacement, text)


def normalize_prompt_text(text: str, options: KeyNormalization | None = None) -> str:
    """Normalise prompt text for exact-match hashing.

    Code blocks (``` fenced ```) are preserved byte-for-byte; the options apply
    only to the prose between them.
    """
    opts = options or KeyNormalization()
    if not text:
        return ""
    blocks = _CODE_BLOCK.findall(text)
    parts = _CODE_BLOCK.split(text)
    out_parts: list[str] = []
    for i, part in enumerate(parts):
        p = part
        if opts.unicode_nfc:
            p = unicodedata.normalize("NFC", p)
        if opts.strip_volatile:
            p = strip_volatile(p)
        elif opts.strip_labelled_ids:
            p = strip_labelled_ids(p)
        if opts.stable_json:
            p = _stable_json_blocks(p)
        if opts.collapse_whitespace:
            p = _WS.sub(" ", p).strip()
        if opts.lowercase:
            p = p.lower()
        out_parts.append(p)
        if i < len(blocks):
            out_parts.append(blocks[i])
    return " ".join(x for x in out_parts if x)


def _h(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(canonical_json(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def namespace_for(
    model: str,
    *,
    prompt_version: str | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    output_schema: dict[str, Any] | None = None,
    corpus_version: str | None = None,
    embedding_model: str | None = None,
    tenant: str | None = None,
) -> str:
    """Build the versioned namespace ``{model}:{template}:{tools}:{schema}:{corpus}:{embedding}:{tenant}``.

    Every determinant of an output is part of the namespace so that a model
    upgrade, template bump, tool-schema change or re-index naturally misses.
    """
    parts = [
        model or "default",
        (prompt_version or "-"),
        _h(list(tools or []))[:12] if tools else "-",
        _h(output_schema)[:12] if output_schema else "-",
        corpus_version or "-",
        embedding_model or "-",
        tenant or "-",
    ]
    return ":".join(parts)


def exact_key(
    request: LLMRequest,
    model: str | None = None,
    prompt_version: str | None = None,
    *,
    normalization: KeyNormalization | None = None,
    namespace: str | None = None,
    tenant_salt: str = "",
    extra_keys: dict[str, Any] | None = None,
) -> str:
    """SHA-256 chain key for ``request`` (vLLM-style parent-hash chaining).

    The chain commits, in order, to the namespace, the tool definitions, each
    stable system block, the volatile system text and each message (all
    normalised), and finally to the extra determinants (model, template
    version, output schema, effort, max_tokens, tenant salt and caller-supplied
    ``extra_keys`` such as a corpus snapshot id).
    """
    opts = normalization or KeyNormalization()
    model_id = model or request.model or "default"
    ns = namespace or namespace_for(model_id, prompt_version=prompt_version, tools=request.tools, output_schema=request.output_schema)
    h = _h("caching:v1", ns)
    h = _h(h, "tools", request.tools)
    for block in request.system_stable:
        h = _h(h, "system", normalize_prompt_text(block, opts))
    h = _h(h, "volatile", normalize_prompt_text(request.system_volatile or "", opts))
    for msg in request.messages:
        h = _h(h, "msg", msg.role, normalize_prompt_text(msg.text(), opts) if isinstance(msg.content, str) else msg.content)
    h = _h(
        h,
        "extra",
        {
            "model": model_id,
            "prompt_version": prompt_version,
            "output_schema": request.output_schema,
            "effort": request.effort,
            "max_tokens": request.max_tokens,
            "tenant_salt": tenant_salt,
            **(extra_keys or {}),
        },
    )
    return h


def tagged_keys(
    *,
    model: str | None = None,
    prompt_template: str | None = None,
    prompt_version: str | None = None,
    document_ids: Iterable[str] = (),
    namespace: str | None = None,
    extra: Iterable[str] = (),
) -> list[str]:
    """Tags for invalidation by model, prompt template or document id."""
    tags: list[str] = []
    if model:
        tags.append(f"model:{model}")
    if prompt_template:
        tags.append(f"template:{prompt_template}" + (f"@{prompt_version}" if prompt_version else ""))
    for d in document_ids:
        tags.append(f"doc:{d}")
    if namespace:
        tags.append(f"ns:{namespace}")
    tags.extend(extra)
    return list(dict.fromkeys(tags))


def extract_guards(text: str) -> frozenset[str]:
    """Identifiers, part numbers, versions, quantities and numbers in ``text``.

    Semantic hits require the guard sets of the query and the cached entry to
    be identical: embeddings blur ``5 mm`` versus ``50 mm`` and ``REQ-123``
    versus ``REQ-124`` (Zhu et al. 2024; dossier engineering notes).
    """
    found: set[str] = set()
    for kind, pat in IDENTIFIER_PATTERNS.items():
        for m in pat.finditer(text):
            tok = _WS.sub("", m.group(0))
            if kind == "quantity":
                tok = tok.lower()
            found.add(tok)
    return frozenset(found)


def context_chain_key(previous_turns: Sequence[str], turns: int = 3, options: KeyNormalization | None = None) -> str:
    """Hash of the last ``turns`` prior turns (MeanCache context chain)."""
    if turns <= 0:
        return _h("ctx", [])[:16]
    tail = [normalize_prompt_text(t, options) for t in list(previous_turns)[-turns:]]
    return _h("ctx", tail)[:16]


def semantic_text(request: LLMRequest, options: KeyNormalization | None = None) -> str:
    """The text the semantic tier embeds: the normalised last user turn."""
    return normalize_prompt_text(request.last_user_text(), options)


def previous_turn_texts(request: LLMRequest) -> list[str]:
    """All message texts before the last user turn (for the context chain)."""
    texts = [m.text() for m in request.messages]
    for i in range(len(request.messages) - 1, -1, -1):
        if request.messages[i].role == "user":
            return texts[:i]
    return texts


__all__ = [
    "IDENTIFIER_PATTERNS",
    "VOLATILE_PATTERNS",
    "context_chain_key",
    "exact_key",
    "extract_guards",
    "namespace_for",
    "normalize_prompt_text",
    "previous_turn_texts",
    "semantic_text",
    "strip_labelled_ids",
    "strip_volatile",
    "tagged_keys",
]
