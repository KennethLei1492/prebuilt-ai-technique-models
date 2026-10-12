"""Atomic-claim utilities for grounding (FActScore / ALCE style), all LLM-free.

* :func:`split_claims` - sentence-level fallback decomposition.
* :func:`rank_evidence_for_claim` - pick the top-k passages per claim.
* :func:`lexical_support` - supported / contradicted / unsupported heuristic.
* :func:`grounding_metrics` and :func:`citation_metrics` - the three numbers
  the literature insists on reporting together (supported fraction, claim
  count, contradictions) plus citation precision/recall.
"""
from __future__ import annotations

import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, tokenize_words

from .uncertainty import _NUM_RE, _STOPWORDS

ClaimLabel = Literal["supported", "contradicted", "unsupported"]
ClaimType = Literal["factual", "numerical", "code", "procedural", "citation"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n+")
_HEDGE = re.compile(r"^(?:in summary|overall|note that|as (?:shown|mentioned|discussed)|i think|i believe|it seems)", re.I)
_CITE_RE = re.compile(r"\[([\w\-:.#/]+)\]")  # chunk ids such as SPEC-1#0 or docs/a.md#2 included


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    type: ClaimType = "factual"


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    label: ClaimLabel
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""
    overlap: float = Field(default=0.0, ge=0.0, le=1.0)


def split_claims(text: str, *, max_claims: int = 15, min_words: int = 3) -> list[Claim]:
    """Sentence-level decomposition used offline or when LLM extraction returns nothing."""
    out: list[Claim] = []
    for raw in _SENT_SPLIT.split(text.strip()):
        s = raw.strip().lstrip("-*• ").strip()
        if not s or len(s.split()) < min_words or _HEDGE.match(s):
            continue
        ctype: ClaimType = "factual"
        if "```" in s or re.search(r"\bdef \w+\(|\breturn\b|;\s*$", s):
            ctype = "code"
        elif _NUM_RE.search(s):
            ctype = "numerical"
        elif re.match(r"^(first|then|next|finally|step \d|\d+[.)])", s, re.I):
            ctype = "procedural"
        elif _CITE_RE.search(s) and len(s.split()) <= 8:
            ctype = "citation"
        out.append(Claim(text=s, type=ctype))
        if len(out) >= max_claims:
            break
    return out


#: Stand-alone numbers (a glued short unit such as ``12mm`` is allowed); digits inside
#: identifiers (``A216``, ``spec-1``, ``CA6NM``) are not quantities.
_CLAIM_NUM_RE = re.compile(r"(?<![\w.\-])[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:e[-+]?\d+)?(?=[a-zA-Zµ°%]{0,4}\b)", re.I)
#: Alphanumeric identifiers (material grades, part numbers, clause ids) that must match exactly.
_IDENT_RE = re.compile(r"^(?=[a-z0-9/\-]*\d)(?=[a-z0-9/\-]*[a-z])[a-z0-9]+(?:[-/][a-z0-9]+)*$")
_NUMBER_WITH_UNIT_RE = re.compile(r"^\d+(?:[.,]\d+)?[a-zµ°%]{1,4}$")


def _content_tokens(text: str) -> set[str]:
    """Content tokens of ``text`` with stopwords and ``[doc-id]`` citation markers removed."""
    return {w for w in tokenize_words(_CITE_RE.sub(" ", text)) if w not in _STOPWORDS and len(w) > 1}


def _identifiers(text: str) -> set[str]:
    """Alphanumeric identifier tokens (``a743``, ``ca6nm``, ``req-12a``), excluding number+unit glue like ``12mm``."""
    return {w for w in tokenize_words(_CITE_RE.sub(" ", text)) if _IDENT_RE.match(w) and not _NUMBER_WITH_UNIT_RE.match(w)}


def _keys(text: str) -> set[str]:
    """Numbers and identifiers: the tokens a supporting passage must reproduce exactly."""
    return _numbers(text) | _identifiers(text)


def _numbers(text: str) -> set[str]:
    """Stand-alone numbers in ``text`` (citations removed) normalised so that ``3.0`` and ``3`` compare equal."""
    out: set[str] = set()
    for n in _CLAIM_NUM_RE.findall(_CITE_RE.sub(" ", text)):
        try:
            out.add(format(float(n.replace(",", "")), "g"))
        except ValueError:
            continue
    return out


def coverage(claim: str, passage: str) -> float:
    """Fraction of the claim's content tokens that appear in the passage."""
    ct = _content_tokens(claim)
    if not ct:
        return 0.0
    pt = _content_tokens(passage)
    return len(ct & pt) / len(ct)


def rank_evidence_for_claim(claim: str, evidence: Sequence[Evidence], k: int = 5) -> list[Evidence]:
    """Top-k evidence by content-token coverage (ties broken by retriever score)."""
    scored = [(coverage(claim, ev.content), ev.score, i, ev) for i, ev in enumerate(evidence)]
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [t[3] for t in scored[:k]]


def lexical_support(claim: str, evidence: Sequence[Evidence], *, threshold: float = 0.5) -> ClaimVerdict:
    """Heuristic entailment.

    * ``supported``: a passage covers at least ``threshold`` of the claim's content
      tokens and the passages given (the claim's top-k) between them reproduce
      *every* number and identifier (grade, part number, clause id) the claim states;
    * ``contradicted``: the claim's words match a passage that states its own
      numbers/identifiers, but at least one of the claim's keys appears in none of
      the passages (same statement, different value);
    * otherwise ``unsupported``.
    """
    best_cov, best_ev = 0.0, None
    claim_tokens = _content_tokens(claim)
    claim_keys = _keys(claim)
    word_tokens = {t for t in claim_tokens if t not in claim_keys and not _NUMBER_WITH_UNIT_RE.match(t) and not t.replace(".", "").isdigit()}
    keys_by_ev = {ev.id: _keys(ev.content) for ev in evidence}
    union_keys: set[str] = set().union(*keys_by_ev.values()) if keys_by_ev else set()
    missing = claim_keys - union_keys
    contradiction: Evidence | None = None
    for ev in evidence:
        ev_tokens = _content_tokens(ev.content)
        cov = len(claim_tokens & ev_tokens) / len(claim_tokens) if claim_tokens else 0.0
        if cov > best_cov:
            best_cov, best_ev = cov, ev
        if missing and contradiction is None:
            words_only = (len(word_tokens & ev_tokens) / len(word_tokens)) if word_tokens else cov
            if words_only >= threshold and keys_by_ev[ev.id]:
                contradiction = ev
    if best_ev is not None and best_cov >= threshold and not missing:
        ids = [best_ev.id]
        uncovered = claim_keys - keys_by_ev[best_ev.id]
        for ev in evidence:  # passages that supply keys the best passage lacks
            if ev.id != best_ev.id and uncovered & keys_by_ev[ev.id]:
                ids.append(ev.id)
                uncovered -= keys_by_ev[ev.id]
        return ClaimVerdict(claim=claim, label="supported", evidence_ids=ids, overlap=best_cov, note="lexical coverage")
    if contradiction is not None:
        return ClaimVerdict(
            claim=claim, label="contradicted", evidence_ids=[contradiction.id], overlap=best_cov, note="matching statement with different number or identifier"
        )
    return ClaimVerdict(claim=claim, label="unsupported", evidence_ids=[best_ev.id] if best_ev else [], overlap=best_cov, note="no covering passage")


def grounding_metrics(verdicts: Sequence[ClaimVerdict]) -> dict[str, float]:
    """FActScore-style numbers that must always be reported together."""
    n = len(verdicts)
    supported = sum(1 for v in verdicts if v.label == "supported")
    contradicted = sum(1 for v in verdicts if v.label == "contradicted")
    unsupported = n - supported - contradicted
    return {
        "claim_count": float(n),
        "supported": float(supported),
        "contradicted": float(contradicted),
        "unsupported": float(unsupported),
        "supported_fraction": (supported / n) if n else 0.0,
    }


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(_CITE_RE.findall(text)))


def citation_metrics(answer: str, verdicts: Sequence[ClaimVerdict], evidence_ids: Sequence[str]) -> dict[str, float]:
    """ALCE-style citation precision/recall.

    * precision: cited ids that exist and support at least one claim / cited ids;
    * recall: claims whose supporting evidence is cited somewhere in the answer / claims.
    """
    cited = extract_citations(answer)
    known = set(evidence_ids)
    supporting = {eid for v in verdicts if v.label == "supported" for eid in v.evidence_ids}
    if cited:
        precision = sum(1 for c in cited if c in known and c in supporting) / len(cited)
    else:
        precision = 0.0
    if verdicts:
        recall = sum(1 for v in verdicts if v.label == "supported" and any(e in cited for e in v.evidence_ids)) / len(verdicts)
    else:
        recall = 0.0
    return {"citation_precision": precision, "citation_recall": recall, "citations": float(len(cited))}
