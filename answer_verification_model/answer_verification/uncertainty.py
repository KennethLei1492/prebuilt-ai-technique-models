"""Uncertainty estimation, calibration and verdict fusion.

Everything in this module is a pure function or a small pydantic model so it
can be unit-tested without an LLM:

* :func:`canonicalize_answer` - normalise answers so votes can be counted
  (numbers, labels, code, short text).
* :func:`semantic_agreement` - cluster sampled answers by equivalence and
  compute agreement and discrete semantic entropy (Kuhn et al.; Farquhar et al.).
* :func:`majority_stability` / :func:`should_stop_sampling` - the
  Adaptive-Consistency stopping rule (Aggarwal et al.).
* :func:`platt_scale` / :func:`fit_platt` - calibration with stored parameters.
* :func:`aggregate_verdicts` - precedence-aware weighted fusion of
  :class:`pipeline_core.Verdict` objects into ``(confidence, basis)``.
* :func:`probability_for_candidate` - read a verbalised top-k distribution;
  :func:`verbalized_confidence` - the one-call elicitation itself (Tian et al.).
"""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import RunContext, Verdict, normalize_text, tokenize_words

from .config import AnswerVerificationConfig, CalibrationParams

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime (verifiers imports this module)
    from .verifiers import VerifyInputs

# --------------------------------------------------------------------------
# Canonicalisation
# --------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:e[-+]?\d+)?", re.I)
_STOPWORDS = frozenset(
    "the a an of to in on for and or is are was were be been it this that these those with as by at from which who whom "
    "its their there here than then so such into onto about over under between".split()
)
_CODE_COMMENT = re.compile(r"(#.*?$)|(//.*?$)|(/\*.*?\*/)", re.M | re.S)
_FENCED_BLOCK = re.compile(r"```.*?```", re.S)
#: A stand-alone number: not glued to a letter on the left (``M16``, ``A743``, ``CA6NM`` are identifiers, not values).
_ANSWER_NUM = r"(?<![A-Za-z0-9.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?"
_ANSWER_NUM_RE = re.compile(_ANSWER_NUM, re.I)
#: ``final answer: 12.5``, ``the result is 6.6 mm``, ``therefore t = 9.6 mm`` - the number stated as the result.
_RESULT_MARK_RE = re.compile(r"\b(?:final answer|answer|result|therefore|thus|hence)\b[^\d\n.]{0,40}?(" + _ANSWER_NUM + r")", re.I)
#: ``F = 187.5 kN`` - the right-hand side of the last equation in a derivation chain.
_EQUALS_RE = re.compile(r"=\s*(?:approximately|approx\.?|about|roughly|~|≈)?\s*(" + _ANSWER_NUM + r")", re.I)


def _answer_text(text: str) -> str:
    """Text with thousands separators and fenced code removed (numbers inside code blocks are not the answer)."""
    t = text.replace(",", "")
    stripped = _FENCED_BLOCK.sub(" ", t)
    return stripped if _ANSWER_NUM_RE.search(stripped) else t


def extract_number(text: str, *, unit: str | None = None) -> Optional[float]:
    """Return the number ``text`` states as its answer, or None.

    Selection order (deterministic, documented in the README):

    1. a quantity whose unit family matches ``unit`` (``unit="bar"`` picks ``16`` in
       ``"16 bar at 80 degC"``); among several, the one introduced by a result marker,
       else the first;
    2. the number after a result marker (``final answer``, ``result``, ``therefore`` ...);
    3. the right-hand side of the last ``=`` (derivation chains end in the result);
    4. the first stand-alone number. Digits glued to letters (``M16``, ``A743``) and
       numbers inside fenced code blocks are never the answer.
    """
    t = _answer_text(text)
    if unit:
        from .units import parse_quantities, unit_family  # local import: units does not depend on this module

        fam = unit_family(unit)
        if fam is not None:
            qs = [q for q in parse_quantities(t) if q.family == fam]
            if qs:
                marked = _RESULT_MARK_RE.search(t)
                if marked is not None:
                    pos = marked.start(1)
                    for q in qs:
                        if q.start <= pos < q.end:
                            return q.value
                return qs[0].value
    m = _RESULT_MARK_RE.search(t)
    if m is None:
        eqs = list(_EQUALS_RE.finditer(t))
        m = eqs[-1] if eqs else None
    raw = m.group(1) if m is not None else None
    if raw is None:
        first = _ANSWER_NUM_RE.search(t)
        raw = first.group(0) if first else None
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def canonicalize_answer(text: str, answer_type: str = "factual") -> str:
    """Normalise an answer into a vote key.

    * numerical: the answer number chosen by :func:`extract_number`, rounded to 6
      significant digits (units are ignored for the vote; the rule verifier checks them separately);
    * code: whitespace- and comment-insensitive source;
    * otherwise: the sorted bag of lower-cased content tokens (stopwords and
      punctuation removed), so paraphrases that reorder the same facts vote together.
    """
    t = text.strip()
    if answer_type == "numerical":
        n = extract_number(t)
        if n is not None:
            if n == 0:
                return "0"
            digits = 6 - int(math.floor(math.log10(abs(n)))) - 1
            return format(round(n, max(0, digits)), "g")
    if answer_type == "code":
        src = _CODE_COMMENT.sub("", t)
        return re.sub(r"\s+", "", src)
    toks = sorted(w for w in tokenize_words(t) if w not in _STOPWORDS)
    return " ".join(toks) if toks else normalize_text(t)


def similarity_ratio(a: str, b: str) -> float:
    """Sequence similarity in [0, 1] (difflib ratio) used as the RARR-style preservation score."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over content tokens."""
    sa = {w for w in tokenize_words(a) if w not in _STOPWORDS}
    sb = {w for w in tokenize_words(b) if w not in _STOPWORDS}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------
# Agreement / semantic entropy
# --------------------------------------------------------------------------


class AgreementResult(BaseModel):
    """Clusters of equivalent answers plus the derived uncertainty statistics."""

    model_config = ConfigDict(extra="forbid")

    clusters: list[list[int]] = Field(description="Indices of samples per cluster, largest first")
    representatives: list[str] = Field(description="One sample text per cluster")
    agreement: float = Field(ge=0.0, le=1.0, description="Largest cluster size / samples")
    entropy: float = Field(ge=0.0, description="Discrete semantic entropy over clusters (nats)")
    normalized_entropy: float = Field(ge=0.0, le=1.0, description="entropy / log(n_samples); 0 = unanimous")
    majority_index: int = Field(ge=0, description="Index of the representative of the largest cluster")

    @property
    def counts(self) -> list[int]:
        return [len(c) for c in self.clusters]


def _entropy(counts: Sequence[int]) -> float:
    n = sum(counts)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in counts if c > 0)


def semantic_agreement(
    samples: Sequence[str],
    *,
    equivalent: Callable[[str, str], bool] | None = None,
    answer_type: str = "factual",
    jaccard_threshold: float = 0.6,
) -> AgreementResult:
    """Cluster ``samples`` by meaning and compute agreement and semantic entropy.

    The default equivalence is canonical-key equality (exact for numbers and
    code) falling back to a Jaccard threshold for free text; an LLM-judged
    equivalence function can be injected for true bidirectional entailment.
    """
    if not samples:
        return AgreementResult(clusters=[], representatives=[], agreement=0.0, entropy=0.0, normalized_entropy=0.0, majority_index=0)

    keys = [canonicalize_answer(s, answer_type) for s in samples]

    def default_equivalent(i: int, j: int) -> bool:
        if keys[i] == keys[j]:
            return True
        if answer_type in ("numerical", "code"):
            return False
        return jaccard(samples[i], samples[j]) >= jaccard_threshold

    clusters: list[list[int]] = []
    for i in range(len(samples)):
        placed = False
        for cl in clusters:
            rep = cl[0]
            same = equivalent(samples[rep], samples[i]) if equivalent else default_equivalent(rep, i)
            if same:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    clusters.sort(key=lambda c: (-len(c), c[0]))
    counts = [len(c) for c in clusters]
    n = len(samples)
    ent = _entropy(counts)
    max_ent = math.log(n) if n > 1 else 0.0
    return AgreementResult(
        clusters=clusters,
        representatives=[samples[c[0]] for c in clusters],
        agreement=counts[0] / n,
        entropy=ent,
        normalized_entropy=(ent / max_ent) if max_ent > 0 else 0.0,
        majority_index=clusters[0][0],
    )


def _regularized_incomplete_beta_half(alpha: int, beta: int) -> float:
    """``I_0.5(alpha, beta)`` for positive integer parameters via the binomial identity."""
    n = alpha + beta - 1
    total = 0.0
    for j in range(alpha, n + 1):
        total += math.comb(n, j)
    return total * (0.5**n)


def majority_stability(counts: Sequence[int], *, prior: int = 1) -> float:
    """Posterior probability that the current majority answer stays the majority.

    Uses the Adaptive-Consistency criterion with a Dirichlet(prior) posterior
    over answer counts, reduced to the two leading clusters: with ``n1`` votes
    for the leader and ``n2`` for the runner-up, ``P(p1 > p2) = 1 - I_0.5(n1 +
    prior, n2 + prior)``. With a single cluster the runner-up count is 0.
    """
    if not counts:
        return 0.0
    ordered = sorted(counts, reverse=True)
    n1 = ordered[0]
    n2 = ordered[1] if len(ordered) > 1 else 0
    return 1.0 - _regularized_incomplete_beta_half(n1 + prior, n2 + prior)


def should_stop_sampling(counts: Sequence[int], *, threshold: float = 0.95, min_samples: int = 3) -> bool:
    """Adaptive-Consistency stopping rule: stop once the majority is stable enough."""
    if sum(counts) < min_samples:
        return False
    return majority_stability(counts) >= threshold


# --------------------------------------------------------------------------
# Verbalised confidence
# --------------------------------------------------------------------------


class ConfidenceCandidate(BaseModel):
    """One candidate answer with its verbalised probability (bounds are enforced in code, not the schema)."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    probability: float


class VerbalizedConfidence(BaseModel):
    """Top-k candidates with probabilities (Tian et al. one-stage elicitation)."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[ConfidenceCandidate]


def probability_for_candidate(dist: VerbalizedConfidence, answer: str, answer_type: str = "factual") -> float:
    """Probability mass the model assigned to ``answer`` (matched by canonical key, else best Jaccard >= 0.6)."""
    if not dist.candidates:
        return 0.0
    key = canonicalize_answer(answer, answer_type)
    for c in dist.candidates:
        if canonicalize_answer(c.answer, answer_type) == key:
            return min(1.0, max(0.0, c.probability))
    best = max(dist.candidates, key=lambda c: jaccard(c.answer, answer))
    if jaccard(best.answer, answer) >= 0.6:
        return min(1.0, max(0.0, best.probability))
    return 0.0


async def verbalized_confidence(ctx: RunContext, vi: "VerifyInputs", config: AnswerVerificationConfig) -> Verdict:
    """One-call top-k verbalised confidence elicitation (Tian et al.; Xiong et al.).

    Asks for the top-``config.verbalized_top_k`` candidate answers with
    probabilities, renormalises a distribution whose mass exceeds 1, and reads
    the mass assigned to the candidate under review (canonical-key match, else
    best Jaccard >= 0.6). The result is an *advisory* verdict
    (``details["advisory"] = True``) weighted by ``config.verbalized_weight``
    in :func:`aggregate_verdicts`; it never gates acceptance on its own because
    verbalised confidence is overconfident on professional-knowledge tasks.
    """
    from . import prompts as P  # local imports: verifiers imports this module at load time
    from .verifiers import build_request, documents_block

    req = build_request(
        ctx,
        P.VERBALIZED_CONFIDENCE,
        schema=VerbalizedConfidence,
        effort=config.grade_effort,
        documents=documents_block(vi.evidence),
        task=vi.task,
        answer=vi.answer,
        k=str(config.verbalized_top_k),
    )
    resp = await ctx.llm_call(req, step=P.VERBALIZED_CONFIDENCE.name)
    dist = VerbalizedConfidence.model_validate(resp.parsed)
    total = sum(c.probability for c in dist.candidates)
    if total > 1.0:
        dist = VerbalizedConfidence(candidates=[c.model_copy(update={"probability": c.probability / total}) for c in dist.candidates])
    p = probability_for_candidate(dist, vi.answer, vi.answer_type)
    return Verdict(
        method="verbalized_confidence",
        passed=p >= 0.5,
        score=round(p, 4),
        details={"advisory": True, "candidates": [c.model_dump() for c in dist.candidates], "k": config.verbalized_top_k},
    )


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def platt_scale(p: float, params: CalibrationParams | None = None) -> float:
    """Apply stored Platt parameters ``sigmoid(a * logit(p) + b)``; identity by default."""
    params = params or CalibrationParams()
    if params.a == 1.0 and params.b == 0.0:
        return min(1.0, max(0.0, p))
    return _sigmoid(params.a * _logit(p) + params.b)


def fit_platt(scores: Sequence[float], labels: Sequence[int], *, iterations: int = 500, lr: float = 0.1) -> CalibrationParams:
    """Fit Platt parameters by gradient descent on log loss (small, dependency-free).

    Intended for the offline evaluation harness; production users should fit on
    >= 500 labelled items and store the parameters in the config.
    """
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and the same length")
    xs = [_logit(s) for s in scores]
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(iterations):
        ga = gb = 0.0
        for x, y in zip(xs, labels):
            p = _sigmoid(a * x + b)
            ga += (p - y) * x
            gb += p - y
        a -= lr * ga / n
        b -= lr * gb / n
    return CalibrationParams(a=a, b=b)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

#: Precedence from the literature: executed tests and deterministic rules are terminal
#: authorities; evidence grounding and multi-sample agreement are strong external signals;
#: LLM judgement is advisory.
AUTHORITY: dict[str, int] = {
    "code_execution": 3,
    "rules": 3,
    "grounding": 2,
    "self_consistency": 2,
    "chain_of_verification": 1,
    "rubric_judge": 1,
    "debate": 1,
    "verbalized_confidence": 0,
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "code_execution": 2.0,
    "rules": 1.5,
    "grounding": 1.5,
    "self_consistency": 1.0,
    "chain_of_verification": 1.0,
    "rubric_judge": 0.75,
    "debate": 0.5,
    "verbalized_confidence": 0.5,
}


def is_inconclusive(v: Verdict) -> bool:
    return bool(v.details.get("inconclusive"))


def aggregate_verdicts(
    verdicts: Iterable[Verdict],
    weights: dict[str, float] | None = None,
    *,
    calibration: CalibrationParams | None = None,
) -> tuple[float, str]:
    """Fuse verdicts into one calibrated confidence and a human-readable basis.

    Rules:

    1. Inconclusive verdicts (``details['inconclusive']``) carry no weight but are named in the basis.
    2. Confidence is the weight-averaged score of conclusive verdicts.
    3. A failed authority-3 verifier (tests, rules) caps confidence at its own score (hard evidence wins).
    4. A grounding verdict with contradictions caps confidence at 0.5.
    5. Platt scaling is applied last.
    """
    verdicts = list(verdicts)  # the argument may be a generator; it is traversed twice below
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    conclusive = [v for v in verdicts if not is_inconclusive(v)]
    skipped = [v for v in verdicts if is_inconclusive(v)]
    parts: list[str] = []
    if not conclusive:
        names = ", ".join(v.method for v in skipped) or "none"
        return 0.5, f"no conclusive verifier ran (inconclusive: {names}); confidence is the uninformative prior 0.5"
    num = den = 0.0
    for v in conclusive:
        wt = w.get(v.method, 1.0)
        num += wt * v.score
        den += wt
        parts.append(f"{v.method}={v.score:.2f}{'' if v.passed else ' FAIL'}")
    conf = num / den if den else 0.5
    caps: list[str] = []
    for v in conclusive:
        if AUTHORITY.get(v.method, 1) >= 3 and not v.passed and v.score < conf:
            conf = v.score
            caps.append(f"capped by failed {v.method}")
        if v.method == "grounding" and v.details.get("contradicted", 0) > 0 and conf > 0.5:
            conf = 0.5
            caps.append("capped at 0.5: contradicted claims")
    conf = platt_scale(conf, calibration)
    basis = "weighted fusion of " + ", ".join(parts)
    if caps:
        basis += "; " + "; ".join(dict.fromkeys(caps))
    if skipped:
        basis += "; inconclusive: " + ", ".join(v.method for v in skipped)
    return round(min(1.0, max(0.0, conf)), 4), basis
