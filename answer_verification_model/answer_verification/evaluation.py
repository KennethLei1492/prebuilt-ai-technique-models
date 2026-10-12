"""Verification quality metrics and an offline evaluation harness.

Metrics (all pure functions):

* :func:`accuracy`, :func:`precision_recall_f1` - pass/fail decisions vs labels;
* :func:`brier_score`, :func:`expected_calibration_error` - calibration of the
  fused confidence (targets from the dossier: ECE <= 0.05);
* :func:`auroc` - how well confidence separates correct from incorrect answers
  (target >= 0.8 for failure prediction);
* cost per verified answer, LLM calls, abstention (escalation) rate.

:func:`evaluate` runs a model over labelled :class:`VerificationExample`
items; :func:`synthetic_examples` generates a small engineering set (correct
answers with supporting evidence vs. answers with wrong numbers or missing
units) so the harness can run offline and in CI.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Constraints, Evidence, Status, TaskRequest, run_sync

from .model import AnswerVerificationModel, VerificationReport

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def accuracy(predictions: Sequence[bool], labels: Sequence[bool]) -> float:
    """Fraction of pass/fail decisions that match the labels."""
    if not predictions or len(predictions) != len(labels):
        return 0.0
    return sum(1 for p, y in zip(predictions, labels) if bool(p) == bool(y)) / len(predictions)


def precision_recall_f1(predictions: Sequence[bool], labels: Sequence[bool]) -> dict[str, float]:
    """Precision/recall/F1 of predicting *correct* answers (positive class = accepted & correct)."""
    tp = sum(1 for p, y in zip(predictions, labels) if p and y)
    fp = sum(1 for p, y in zip(predictions, labels) if p and not y)
    fn = sum(1 for p, y in zip(predictions, labels) if not p and y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error between confidence and the 0/1 outcome (lower is better)."""
    if not probabilities or len(probabilities) != len(labels):
        return 0.0
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[bool], *, bins: int = 10) -> float:
    """ECE with equal-width bins: sum over bins of |accuracy - confidence| weighted by bin mass."""
    if not probabilities or len(probabilities) != len(labels):
        return 0.0
    n = len(probabilities)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probabilities) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probabilities[i] for i in idx) / len(idx)
        acc = sum(1.0 for i in idx if labels[i]) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-based AUROC of ``scores`` for predicting ``labels``; None when one class is missing."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class VerificationExample(BaseModel):
    """One labelled item: is ``answer`` a correct answer to ``task``?"""

    model_config = ConfigDict(extra="forbid")

    task: str
    answer: str
    correct: bool
    evidence: list[Evidence] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    domain: Optional[str] = None
    note: str = ""


class EvaluationReport(BaseModel):
    """Aggregate metrics of one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float
    ece: float
    auroc: Optional[float]
    mean_confidence: float
    abstention_rate: float = Field(description="Fraction of items escalated (confidence below the escalation threshold)")
    needs_review_rate: float
    llm_calls: int
    cost_usd: float
    cost_per_answer_usd: float
    mean_latency_s: float
    per_item: list[dict[str, Any]] = Field(default_factory=list)

    def render(self) -> str:
        auc = f"{self.auroc:.3f}" if self.auroc is not None else "n/a"
        return "\n".join(
            [
                f"Evaluated {self.n} labelled answers",
                f"  decision accuracy      {self.accuracy:.3f}   precision {self.precision:.3f}  recall {self.recall:.3f}  F1 {self.f1:.3f}",
                f"  calibration            Brier {self.brier:.3f}  ECE {self.ece:.3f}  AUROC {auc}",
                f"  mean confidence        {self.mean_confidence:.3f}",
                f"  abstention / review    {self.abstention_rate:.2%} / {self.needs_review_rate:.2%}",
                f"  cost                   {self.llm_calls} LLM calls, ${self.cost_usd:.4f} total, ${self.cost_per_answer_usd:.4f} per answer, {self.mean_latency_s * 1000:.0f} ms mean",
            ]
        )


async def evaluate(
    model: AnswerVerificationModel,
    examples: Sequence[VerificationExample],
    *,
    domain: str | None = None,
    constraints: Constraints | None = None,
) -> EvaluationReport:
    """Run ``model`` over ``examples`` and compute decision, calibration and cost metrics."""
    preds: list[bool] = []
    confs: list[float] = []
    labels: list[bool] = []
    rows: list[dict[str, Any]] = []
    calls = 0
    cost = 0.0
    latency = 0.0
    escalated = 0
    review = 0
    for ex in examples:
        req = TaskRequest(
            task=ex.task,
            inputs={**ex.inputs, "answer": ex.answer},
            context=list(ex.evidence),
            domain=ex.domain or domain or model.config.default_domain,
            constraints=constraints or Constraints(),
        )
        res = await model.run(req)
        report = res.output if isinstance(res.output, VerificationReport) else None
        accepted = bool(report.accepted) if report else False
        conf = float(res.confidence)
        preds.append(accepted)
        confs.append(conf)
        labels.append(ex.correct)
        calls += res.usage.llm_calls
        cost += res.usage.cost_usd
        latency += res.usage.latency_s
        if report and report.escalated:
            escalated += 1
        if res.status == Status.NEEDS_REVIEW:
            review += 1
        rows.append(
            {
                "task": ex.task[:80],
                "correct": ex.correct,
                "accepted": accepted,
                "confidence": round(conf, 4),
                "status": res.status.value,
                "verifiers": [v.method for v in report.verdicts] if report else [],
                "note": ex.note,
            }
        )
    prf = precision_recall_f1(preds, labels)
    n = len(examples)
    return EvaluationReport(
        n=n,
        accuracy=accuracy(preds, labels),
        precision=prf["precision"],
        recall=prf["recall"],
        f1=prf["f1"],
        brier=brier_score(confs, labels),
        ece=expected_calibration_error(confs, labels),
        auroc=auroc(confs, labels),
        mean_confidence=(sum(confs) / n) if n else 0.0,
        abstention_rate=(escalated / n) if n else 0.0,
        needs_review_rate=(review / n) if n else 0.0,
        llm_calls=calls,
        cost_usd=cost,
        cost_per_answer_usd=(cost / n) if n else 0.0,
        mean_latency_s=(latency / n) if n else 0.0,
        per_item=rows,
    )


def evaluate_sync(model: AnswerVerificationModel, examples: Sequence[VerificationExample], **kw: Any) -> EvaluationReport:
    """Synchronous wrapper around :func:`evaluate`."""
    return run_sync(evaluate(model, examples, **kw))


_SPECS: list[tuple[str, str, str, str]] = [
    # (task, evidence sentence, correct answer, wrong answer)
    (
        "What corrosion allowance does the specification require for the pump casing?",
        "Section 4.2: the pump casing shall include a corrosion allowance of 3 mm on all wetted surfaces.",
        "The specification requires a corrosion allowance of 3 mm on all wetted casing surfaces.",
        "The specification requires a corrosion allowance of 6 mm on all wetted casing surfaces.",
    ),
    (
        "What is the design pressure of the discharge piping?",
        "Table 2: discharge piping design pressure 16 bar at 80 degC.",
        "The discharge piping design pressure is 16 bar at 80 degC.",
        "The discharge piping design pressure is 25 bar at 80 degC.",
    ),
    (
        "What motor rating is specified for the duty point?",
        "Datasheet line 14: motor rating 75 kW, 1480 rpm, 400 V three phase.",
        "The specified motor rating is 75 kW at 1480 rpm.",
        "The specified motor rating is 90 kW at 1480 rpm.",
    ),
    (
        "Which material grade is required for the impeller?",
        "Clause 5.1: the impeller shall be cast in ASTM A743 CA6NM martensitic stainless steel.",
        "The impeller shall be ASTM A743 CA6NM martensitic stainless steel.",
        "The impeller shall be ASTM A216 WCB carbon steel.",
    ),
    (
        "What minimum NPSH margin does the standard require?",
        "Clause 6.1.9: the NPSH available shall exceed NPSH required by a margin of at least 1 m at the rated flow.",
        "A minimum NPSH margin of 1 m is required at the rated flow.",
        "A minimum NPSH margin of 3 m is required at the rated flow.",
    ),
    (
        "What is the allowable bearing housing vibration limit?",
        "Table 8: unfiltered bearing housing vibration shall not exceed 3.0 mm/s rms at rated conditions.",
        "The bearing housing vibration limit is 3.0 mm/s rms at rated conditions.",
        "The bearing housing vibration limit is 7.1 mm/s rms at rated conditions.",
    ),
]


def synthetic_examples(n: int = 12) -> list[VerificationExample]:
    """Deterministic engineering QA items: half correct (grounded), half with a wrong value."""
    out: list[VerificationExample] = []
    i = 0
    while len(out) < n:
        task, ev_text, good, bad = _SPECS[i % len(_SPECS)]
        ev = [Evidence(id=f"spec-{i % len(_SPECS) + 1}", content=ev_text, source="pump specification", score=1.0, retriever="provided")]
        if len(out) % 2 == 0:
            out.append(VerificationExample(task=task, answer=good, correct=True, evidence=ev, note="grounded, correct value"))
        else:
            out.append(VerificationExample(task=task, answer=bad, correct=False, evidence=ev, note="wrong value vs evidence"))
        i += 1
    return out


__all__ = [
    "EvaluationReport",
    "VerificationExample",
    "accuracy",
    "auroc",
    "brier_score",
    "evaluate",
    "evaluate_sync",
    "expected_calibration_error",
    "precision_recall_f1",
    "synthetic_examples",
]
