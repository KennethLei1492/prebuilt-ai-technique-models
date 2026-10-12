"""Deterministic aggregation in code.

The dossier (architecture recommendation 7) separates "aggregation in code"
from "aggregation by LLM". Everything here is the code half: answer
canonicalisation, majority vote with early-stopping test, vote entropy as a
confidence signal, confidence-label mapping and ordered concatenation.
All functions are pure and unit-tested without an LLM.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

ConfidenceLabel = Literal["high", "medium", "low"]

#: Fixed-scale verbalised confidence (LLM x MapReduce asks mappers for calibrated
#: confidence on a fixed scale; we use three labels so a reducer can compare them).
CONFIDENCE_SCALE: dict[str, float] = {"high": 0.9, "medium": 0.65, "low": 0.35}

_NUMERIC_RE = re.compile(
    r"^\s*(?:[$€£])?\s*([-+]?(?:\d[\d,]*)?\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z%/°μ]*(?:\^?\d)?(?:/[a-zA-Z]+\d?)?)\s*$"
)
_ANSWER_PREFIX_RE = re.compile(r"^(?:the\s+)?(?:final\s+)?(?:answer|result|value)\s*(?:is|=|:)?\s*[:=]?\s*", re.IGNORECASE)
_YES = {"yes", "true", "correct", "y", "affirmative"}
_NO = {"no", "false", "incorrect", "n", "negative"}


def label_to_confidence(label: str | None, default: float = 0.5) -> float:
    """Map a verbalised label to a probability-like score."""
    if label is None:
        return default
    return CONFIDENCE_SCALE.get(str(label).strip().lower(), default)


def canonicalize_answer(text: str, *, significant_digits: int = 6) -> str:
    """Normalise a short answer so that equal answers compare equal.

    Rules (in order): strip whitespace, trailing punctuation and quotes;
    lower-case; remove "the answer is"-style prefixes; collapse whitespace;
    normalise numbers (thousands separators, ``1e3`` notation, trailing
    zeros) and keep their unit token; map yes/true and no/false to ``yes``
    and ``no``.
    """
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    s = _ANSWER_PREFIX_RE.sub("", s)
    s = s.strip(" .\"'`“”")
    s = s.lower()
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    m = _NUMERIC_RE.match(s)
    if m:
        try:
            num = float(m.group(1).replace(",", ""))
        except ValueError:
            return s
        unit = (m.group(2) or "").strip()
        formatted = f"{num:.{significant_digits}g}"
        if "e" in formatted:
            formatted = f"{num:.{significant_digits}e}".replace("e+0", "e").replace("e-0", "e-")
        return f"{formatted} {unit}".strip()
    return s


class VoteResult(BaseModel):
    """Outcome of a majority vote over canonicalised answers."""

    model_config = ConfigDict(extra="forbid")

    winner: str = Field(default="", description="Canonical form of the winning answer")
    winner_raw: str = Field(default="", description="First raw answer in the winning cluster")
    counts: dict[str, int] = Field(default_factory=dict)
    n: int = 0
    share: float = Field(default=0.0, description="Leader votes / n")
    margin: int = Field(default=0, description="Leader votes minus runner-up votes")
    entropy: float = Field(default=0.0, description="Normalised Shannon entropy of the vote distribution (0 = unanimous)")
    decisive: bool = False

    @property
    def unanimous(self) -> bool:
        return self.n > 0 and self.share == 1.0


def vote_entropy(counts: Mapping[str, int]) -> float:
    """Normalised entropy in [0, 1]; 0 when unanimous, 1 when uniform over >= 2 answers."""
    total = sum(counts.values())
    k = len([c for c in counts.values() if c > 0])
    if total == 0 or k <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    return h / math.log(k)


def majority_vote(
    answers: Sequence[str],
    *,
    margin: int = 2,
    leader_share: float = 0.6,
    canonicalize: bool = True,
) -> VoteResult:
    """Tally answers (canonicalised unless disabled) and apply the early-stop rule.

    ``decisive`` is True when the leader is ahead by at least ``margin`` votes
    or holds at least ``leader_share`` of all votes (the dossier's default
    early-stopping test: margin 2 or leader >= 60%). Ties are broken by first
    appearance so the result is deterministic.
    """
    if not answers:
        return VoteResult()
    canon = [canonicalize_answer(a) if canonicalize else str(a) for a in answers]
    counts = Counter(canon)
    order = {c: i for i, c in reversed(list(enumerate(canon)))}  # first index per canonical answer
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    winner, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    n = len(canon)
    share = top / n
    vote_margin = top - runner_up
    decisive = vote_margin >= margin or share >= leader_share
    raw = next(a for a, c in zip(answers, canon) if c == winner)
    return VoteResult(
        winner=winner,
        winner_raw=str(raw),
        counts=dict(ranked),
        n=n,
        share=share,
        margin=vote_margin,
        entropy=vote_entropy(counts),
        decisive=decisive,
    )


def confidence_from_vote(vote: VoteResult) -> float:
    """Confidence from vote agreement: leader share damped by entropy.

    A unanimous 5/5 gives 0.95 (never 1.0: votes can agree and be wrong), a
    3/5 split gives roughly 0.5, a uniform spread tends to 1/k.
    """
    if vote.n == 0:
        return 0.0
    base = vote.share * (1.0 - 0.25 * vote.entropy)
    return round(min(0.95, base), 4)


def weighted_confidence(items: Iterable[tuple[float, float]]) -> float:
    """Weighted mean of (confidence, weight) pairs; 0 when empty."""
    pairs = [(c, w) for c, w in items if w > 0]
    if not pairs:
        return 0.0
    return sum(c * w for c, w in pairs) / sum(w for _, w in pairs)


def success_fraction(ok_flags: Sequence[bool]) -> float:
    """Fraction of branches that succeeded (0 when no branches)."""
    return sum(1 for f in ok_flags if f) / len(ok_flags) if ok_flags else 0.0


def combine_confidence(structural: float, verbalized: float, *, structural_weight: float = 0.6) -> float:
    """Blend a structural signal (vote agreement, branch success) with a verbalised label."""
    w = min(max(structural_weight, 0.0), 1.0)
    return round(min(1.0, max(0.0, w * structural + (1 - w) * verbalized)), 4)


def concat_in_order(parts: Sequence[tuple[int, str]], *, separator: str = "\n\n") -> str:
    """Deterministic merge: order by index, drop empties, join."""
    ordered = sorted(parts, key=lambda p: p[0])
    return separator.join(p[1].strip() for p in ordered if p[1] and p[1].strip())


def missing_branch_notice(failed_ids: Sequence[str]) -> str:
    """Text appended when aggregation proceeds with partial results (Tail at Scale guidance)."""
    if not failed_ids:
        return ""
    return "Note: the following branches did not complete and are missing from this answer: " + ", ".join(failed_ids) + "."
