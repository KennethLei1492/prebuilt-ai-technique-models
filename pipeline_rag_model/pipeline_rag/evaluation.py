"""Evaluation: RAGAS-style metrics, citation metrics, failure-point attribution and an offline harness.

Metrics (all pure functions, unit-tested without a model):

* answer correctness against a reference: :func:`exact_match`, :func:`token_f1`;
* retrieval: :func:`context_precision`, :func:`context_recall` (Es et al.
  2024; Barnett et al. 2024 FP2/FP3);
* grounding: :func:`faithfulness` (fraction of statements supported, RAGAS),
  :func:`citation_metrics` (ALCE citation recall / precision, Gao et al. 2023);
* abstention: :func:`abstention_metrics` (rate, precision, recall);
* :func:`attribute_failure_point` maps one record onto Barnett's FP1-FP7.

:class:`RAGEvaluator` runs a :class:`PipelineRAGModel` over
:class:`QAExample` items, judges faithfulness and answer relevance with the
LLM when one is available (``JUDGE_*`` prompts) or with the lexical proxies
otherwise, and returns an :class:`EvaluationReport` with release gates
(faithfulness >= ``min_faithfulness``, citation recall >=
``min_citation_coverage``). :func:`synthetic_qa` generates in-domain
questions from a corpus (ARES-style, heuristic), :func:`sample_corpus` is a
small synthetic engineering corpus for demos and self-tests.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, MockLLM, Status, TaskRequest, TaskResult, Usage, is_offline_client, render_evidence, run_sync, sanitize_schema, tokenize_words

from .context import content_terms, lexical_relevance, split_sentences
from .generator import Statement, split_statements
from .indexing import chunk_document, normalize_documents
from .iterate import heuristic_support
from .model import PipelineRAGModel, RAGOutput
from .prompts import JUDGE_FAITHFULNESS_PROMPT, JUDGE_RELEVANCE_PROMPT, SYNTHETIC_QA_PROMPT, make_request

# ----------------------------------------------------------------- schemas


class QAExample(BaseModel):
    """One evaluation item."""

    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str = ""
    relevant_ids: list[str] = Field(default_factory=list, description="Ids of evidence units that answer the question")
    answerable: bool = True
    domain: str = "general"
    inputs: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(BaseModel):
    """Per-example scores."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str = ""
    status: str = ""
    route: str = ""
    abstained: bool = False
    answerable: bool = True
    confidence: float = 0.0
    exact_match: Optional[float] = None
    token_f1: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    citation_recall: Optional[float] = None
    citation_precision: Optional[float] = None
    failure_point: Optional[str] = None
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    escalated: bool = False


class EvaluationReport(BaseModel):
    """Aggregate of an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    n: int
    means: dict[str, float] = Field(default_factory=dict)
    abstention: dict[str, float] = Field(default_factory=dict)
    route_distribution: dict[str, int] = Field(default_factory=dict)
    escalation_rate: float = 0.0
    failure_points: dict[str, int] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    judge: str = "heuristic"
    usage: Usage = Field(default_factory=Usage)
    records: list[EvalRecord] = Field(default_factory=list)

    def summary(self) -> str:
        """One-paragraph human summary."""
        parts = [f"n={self.n}", f"judge={self.judge}"]
        for k in ("exact_match", "token_f1", "faithfulness", "answer_relevance", "context_recall", "citation_recall"):
            if k in self.means:
                parts.append(f"{k}={self.means[k]:.2f}")
        parts.append(f"abstention_rate={self.abstention.get('rate', 0.0):.2f}")
        parts.append("gates=" + ",".join(f"{k}:{'pass' if v else 'FAIL'}" for k, v in self.gates.items()))
        return " ".join(parts)


class FaithfulnessJudgement(BaseModel):
    """Structured output of the faithfulness judge."""

    model_config = ConfigDict(extra="forbid")

    statements: list["JudgedStatement"] = Field(default_factory=list)


class JudgedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    supported: bool


class RelevanceJudgement(BaseModel):
    """Structured output of the answer-relevance judge."""

    model_config = ConfigDict(extra="forbid")

    score: float = 0.0
    reason: str = ""


class SyntheticQA(BaseModel):
    """Structured output of the synthetic question generator."""

    model_config = ConfigDict(extra="forbid")

    question: str
    reference_answer: str


FaithfulnessJudgement.model_rebuild()


# ------------------------------------------------------------ pure metrics


def normalize_answer(text: str) -> str:
    """Lowercase, strip citation markers, punctuation and articles (SQuAD-style)."""
    t = re.sub(r"\[[^\[\]\n]{1,80}?\]", " ", text.lower())
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^a-z0-9%.\- ]", " ", t)
    return " ".join(t.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 when the normalised prediction equals or contains the normalised reference."""
    p, r = normalize_answer(prediction), normalize_answer(reference)
    if not r:
        return 0.0
    return 1.0 if p == r or r in p else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 between prediction and reference."""
    p = Counter(tokenize_words(normalize_answer(prediction)))
    r = Counter(tokenize_words(normalize_answer(reference)))
    if not p or not r:
        return 0.0
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    precision = common / sum(p.values())
    recall = common / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def context_precision(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of retrieved units that are relevant (parent-document match counts)."""
    if not retrieved_ids:
        return 0.0
    rel = set(relevant_ids)
    return sum(1 for i in retrieved_ids if _matches_any(i, rel)) / len(retrieved_ids)


def context_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of relevant units that were retrieved (Barnett FP2 when 0)."""
    rel = list(dict.fromkeys(relevant_ids))
    if not rel:
        return 0.0
    got = set(retrieved_ids)
    return sum(1 for r in rel if any(_matches_any(g, {r}) for g in got)) / len(rel)


def _matches_any(unit_id: str, relevant: set[str]) -> bool:
    if unit_id in relevant:
        return True
    parent = unit_id.split("#", 1)[0]
    return parent in relevant or any(r.split("#", 1)[0] == unit_id for r in relevant)


def faithfulness(supported: Sequence[bool]) -> float:
    """RAGAS faithfulness: fraction of answer statements supported by the context."""
    if not supported:
        return 0.0
    return sum(1 for s in supported if s) / len(supported)


def citation_metrics(statements: Sequence[Statement], supports: Callable[[str, str], bool]) -> tuple[float, float]:
    """ALCE citation recall and precision.

    ``supports(statement_text, doc_id)`` says whether a cited unit entails the
    statement. Recall: fraction of statements supported by at least one of
    their citations. Precision: fraction of individual citations that support
    their statement. Statements without citations count against recall.
    """
    if not statements:
        return (0.0, 0.0)
    recall_hits = 0
    cit_total = 0
    cit_hits = 0
    for st in statements:
        hits = [c for c in st.citation_ids if supports(st.text, c)]
        if hits:
            recall_hits += 1
        cit_total += len(st.citation_ids)
        cit_hits += len(hits)
    recall = recall_hits / len(statements)
    precision = cit_hits / cit_total if cit_total else 0.0
    return (recall, precision)


def abstention_metrics(records: Sequence[tuple[bool, bool]]) -> dict[str, float]:
    """From ``(abstained, answerable)`` pairs: rate, precision (abstained and unanswerable) and recall of unanswerables."""
    if not records:
        return {"rate": 0.0, "precision": 0.0, "recall": 0.0}
    abstained = [a for a, _ in records]
    unanswerable = [not ans for _, ans in records]
    tp = sum(1 for a, u in zip(abstained, unanswerable) if a and u)
    rate = sum(abstained) / len(records)
    precision = tp / sum(abstained) if any(abstained) else 0.0
    recall = tp / sum(unanswerable) if any(unanswerable) else 0.0
    return {"rate": rate, "precision": precision, "recall": recall}


def attribute_failure_point(
    *,
    answerable: bool,
    abstained: bool,
    correct: Optional[bool],
    gold_retrieved: Optional[bool],
    gold_in_context: Optional[bool],
    status: str,
    complete: Optional[bool] = None,
) -> Optional[str]:
    """Map one outcome to Barnett et al.'s failure points (None when the answer was fine).

    FP1 missing content answered anyway; FP2 gold not in top-k; FP3 gold
    retrieved but lost in consolidation; FP4 gold in context but not
    extracted; FP5 wrong format (run failed); FP7 incomplete answer.
    """
    if status == "failed":
        return "FP5"
    if not answerable:
        return None if abstained else "FP1"
    if abstained:
        if gold_retrieved is False:
            return "FP2"
        if gold_in_context is False:
            return "FP3"
        return "FP4"
    if correct is False:
        if gold_retrieved is False:
            return "FP2"
        if gold_in_context is False:
            return "FP3"
        return "FP4"
    if complete is False:
        return "FP7"
    return None


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values (None when empty)."""
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# -------------------------------------------------------------------- judges


class Judge:
    """Faithfulness and answer-relevance judges: LLM when available, lexical proxies otherwise."""

    def __init__(self, llm: Any, *, use_llm: Optional[bool] = None) -> None:
        self.llm = llm
        self.use_llm = (not is_offline_client(llm)) if use_llm is None else use_llm
        self.usage = Usage()

    @property
    def name(self) -> str:
        return "llm" if self.use_llm else "heuristic"

    async def faithfulness(self, answer: str, evidence: Sequence[Evidence]) -> Optional[float]:
        statements = split_statements(answer, [e.id for e in evidence])
        if not statements:
            return None
        if not self.use_llm or not evidence:
            judged = heuristic_support(statements, evidence)
            return faithfulness([s.support != "no" for s in judged.statements])
        try:
            req = make_request(
                JUDGE_FAITHFULNESS_PROMPT,
                None,
                schema=sanitize_schema(FaithfulnessJudgement.model_json_schema()),
                effort="low",
                documents=render_evidence(evidence),
                answer=answer,
            )
            resp = await self.llm.complete(req)
            self.usage = self.usage + resp.usage
            j = FaithfulnessJudgement.model_validate(resp.parsed)
            if not j.statements:
                raise ValueError("empty judgement")
            return faithfulness([s.supported for s in j.statements])
        except Exception:  # noqa: BLE001 - fall back to the proxy
            judged = heuristic_support(statements, evidence)
            return faithfulness([s.support != "no" for s in judged.statements])

    async def answer_relevance(self, question: str, answer: str) -> Optional[float]:
        if not answer.strip():
            return 0.0
        if not self.use_llm:
            return max(0.0, min(1.0, (lexical_relevance(question, answer) + 1.0) / 2.0))
        try:
            req = make_request(
                JUDGE_RELEVANCE_PROMPT,
                None,
                schema=sanitize_schema(RelevanceJudgement.model_json_schema()),
                effort="low",
                question=question,
                answer=answer,
            )
            resp = await self.llm.complete(req)
            self.usage = self.usage + resp.usage
            return max(0.0, min(1.0, float(RelevanceJudgement.model_validate(resp.parsed).score)))
        except Exception:  # noqa: BLE001
            return max(0.0, min(1.0, (lexical_relevance(question, answer) + 1.0) / 2.0))


# ------------------------------------------------------------------- harness


class RAGEvaluator:
    """Run a model over examples and compute the metrics above."""

    def __init__(self, model: PipelineRAGModel, *, judge: Optional[Judge] = None, use_llm_judge: Optional[bool] = None) -> None:
        self.model = model
        self.judge = judge or Judge(model.llm, use_llm=use_llm_judge)

    async def evaluate(self, examples: Sequence[QAExample], *, documents: Any = None) -> EvaluationReport:
        """Evaluate ``examples``; ``documents`` is used when an example carries none in ``inputs``."""
        records: list[EvalRecord] = []
        usage = Usage()
        for ex in examples:
            inputs = dict(ex.inputs)
            if "documents" not in inputs and documents is not None:
                inputs["documents"] = documents
            result = await self.model.run(TaskRequest(task=ex.question, inputs=inputs, domain=ex.domain))
            usage = usage + result.usage
            records.append(await self.score(ex, result))
        return self.report(records, usage=usage + self.judge.usage)

    def evaluate_sync(self, examples: Sequence[QAExample], *, documents: Any = None) -> EvaluationReport:
        """Synchronous :meth:`evaluate`."""
        return run_sync(self.evaluate(examples, documents=documents))

    async def score(self, ex: QAExample, result: TaskResult) -> EvalRecord:
        """Score one result against its example."""
        out = result.output if isinstance(result.output, RAGOutput) else None
        answer = out.answer if out else (result.output_text or "")
        abstained = bool(out.abstained) if out else False
        rec = EvalRecord(
            question=ex.question,
            answer=answer,
            status=result.status.value,
            route=out.route if out else "",
            abstained=abstained,
            answerable=ex.answerable,
            confidence=result.confidence,
            llm_calls=result.usage.llm_calls,
            cost_usd=result.usage.cost_usd,
            latency_s=result.usage.latency_s,
            escalated=bool(out and len(out.routes_tried) > 1),
        )
        retrieved = [e.id for e in result.evidence]  # what reached the generator (after grading, hops and compression)
        # what retrieval produced before consolidation: every retrieve step's ids (one per Self-Route pass) plus the final context
        retrieved_any = list(dict.fromkeys([*(i for s in result.steps if s.name.startswith("retrieve") for i in s.details.get("evidence_ids", [])), *retrieved]))
        if ex.relevant_ids:
            rec.context_precision = context_precision(retrieved, ex.relevant_ids)
            rec.context_recall = context_recall(retrieved, ex.relevant_ids)
        if ex.reference_answer and not abstained:
            rec.exact_match = exact_match(answer, ex.reference_answer)
            rec.token_f1 = token_f1(answer, ex.reference_answer)
        if answer and not abstained:
            rec.faithfulness = await self.judge.faithfulness(answer, result.evidence)
            rec.answer_relevance = await self.judge.answer_relevance(ex.question, answer)
            statements = out.statements if out else split_statements(answer, retrieved)
            by_id = {e.id: e for e in result.evidence}

            def supports(text: str, doc_id: str) -> bool:
                ev = by_id.get(doc_id)
                if ev is None:
                    return False
                return any(lexical_relevance(text, s) >= -0.2 for s in (split_sentences(ev.content) or [ev.content]))

            rec.citation_recall, rec.citation_precision = citation_metrics(statements, supports)
        # FP2 (gold never retrieved) versus FP3 (retrieved, then lost in grading / compression)
        gold_retrieved = context_recall(retrieved_any, ex.relevant_ids) > 0 if ex.relevant_ids else None
        gold_in_context = (rec.context_recall or 0.0) > 0 if ex.relevant_ids else None
        correct: Optional[bool] = None
        if rec.exact_match is not None:
            correct = rec.exact_match > 0 or (rec.token_f1 or 0.0) >= 0.5
        complete: Optional[bool] = None if rec.token_f1 is None else rec.token_f1 >= 0.3
        rec.failure_point = attribute_failure_point(
            answerable=ex.answerable,
            abstained=abstained,
            correct=correct,
            gold_retrieved=gold_retrieved,
            gold_in_context=gold_in_context,
            status=result.status.value,
            complete=complete,
        )
        return rec

    def report(self, records: Sequence[EvalRecord], *, usage: Optional[Usage] = None) -> EvaluationReport:
        """Aggregate records into a report with release gates."""
        cfg = self.model.cfg
        means: dict[str, float] = {}
        for key in ("exact_match", "token_f1", "faithfulness", "answer_relevance", "context_precision", "context_recall", "citation_recall", "citation_precision", "confidence", "llm_calls", "latency_s"):
            m = mean([getattr(r, key) for r in records])
            if m is not None:
                means[key] = round(m, 4)
        routes = Counter(r.route for r in records if r.route)
        fps = Counter(r.failure_point for r in records if r.failure_point)
        gates = {}
        if "faithfulness" in means:
            gates["faithfulness"] = means["faithfulness"] >= cfg.min_faithfulness
        if "citation_recall" in means:
            gates["citation_recall"] = means["citation_recall"] >= cfg.min_citation_coverage
        if "answer_relevance" in means:
            gates["answer_relevance"] = means["answer_relevance"] >= 0.8
        return EvaluationReport(
            n=len(records),
            means=means,
            abstention=abstention_metrics([(r.abstained, r.answerable) for r in records]),
            route_distribution=dict(routes),
            escalation_rate=(sum(1 for r in records if r.escalated) / len(records)) if records else 0.0,
            failure_points=dict(fps),
            gates=gates,
            judge=self.judge.name,
            usage=usage or Usage(),
            records=list(records),
        )


# ------------------------------------------------------- synthetic data


def synthetic_qa(documents: Any, n: int = 10, *, seed: int = 0, chunk_size_tokens: int = 512) -> list[QAExample]:
    """ARES-style synthetic questions from a corpus, without an LLM.

    For each chunk, the sentence with the most numbers/identifiers becomes the
    reference answer and a keyword question is built from its content terms,
    so the retriever must return that chunk at rank 1 (the ARES filter).
    """
    rng = random.Random(seed)
    docs = normalize_documents(documents)
    chunks = [c for d in docs for c in chunk_document(d, chunk_size_tokens, 64)]
    rng.shuffle(chunks)
    out: list[QAExample] = []
    for ch in chunks:
        sents = [s for s in split_sentences(ch.text) if len(content_terms(s)) >= 4]
        if not sents:
            continue
        sents.sort(key=lambda s: (-len(re.findall(r"\d", s)), -len(s)))
        ref = sents[0]
        terms = [t for t in tokenize_words(ref, lowercase=False) if t.lower() in content_terms(ref) and not t.isdigit()]
        if len(terms) < 3:
            continue
        subject = " ".join(terms[:4])
        title = ch.source or ch.doc_id
        question = f"According to {title}, what is stated about {subject}?"
        out.append(QAExample(question=question, reference_answer=ref, relevant_ids=[ch.id]))
        if len(out) >= n:
            break
    return out


async def synthetic_qa_llm(model: PipelineRAGModel, documents: Any, n: int = 10, *, domain: str = "general") -> list[QAExample]:
    """Synthetic questions written by the LLM (one call per chunk); falls back to :func:`synthetic_qa` per chunk."""
    from pipeline_core import get_profile

    docs = normalize_documents(documents)
    chunks = [c for d in docs for c in chunk_document(d, model.cfg.chunk_size_tokens, model.cfg.chunk_overlap_tokens)][:n]
    out: list[QAExample] = []
    profile = get_profile(domain)
    for ch in chunks:
        try:
            req = make_request(SYNTHETIC_QA_PROMPT, profile, schema=sanitize_schema(SyntheticQA.model_json_schema()), effort="low", doc_id=ch.id, passage=ch.text)
            resp = await model.llm.complete(req)
            model.direct_usage = model.direct_usage + resp.usage
            qa = SyntheticQA.model_validate(resp.parsed)
            out.append(QAExample(question=qa.question, reference_answer=qa.reference_answer, relevant_ids=[ch.id], domain=domain))
        except Exception:  # noqa: BLE001
            out.extend(synthetic_qa([{"id": ch.doc_id, "text": ch.text, "source": ch.source}], 1))
    return out


def sample_corpus() -> list[dict[str, Any]]:
    """A small synthetic engineering corpus (pump station) for demos and self-tests."""
    return [
        {
            "id": "PS-100",
            "source": "Pump Specification PS-100 Rev C",
            "text": (
                "Section 1 Scope. This specification covers the horizontal centrifugal seawater lift pumps for the cooling water system. "
                "Section 4.2 Casing. The pump casing shall be ASTM A216 WCB carbon steel with a 3 mm corrosion allowance for the pump casing in seawater service. "
                "Casing design pressure is 16 bar at 80 degC and the hydrostatic test pressure shall be 1.5 times design pressure. "
                "Section 4.5 Impeller. The impeller material shall be duplex stainless steel ASTM A890 Grade 4A (CD3MN). "
                "Section 6.1 Performance. Rated flow is 450 m3/h at 42 m head; the NPSH margin at the duty point shall be at least 1.5 m. "
                "Section 7 Testing. Each pump shall undergo a hydrostatic test and a performance test per ISO 9906 grade 2B."
            ),
        },
        {
            "id": "MD-7",
            "source": "Motor Data Sheet MD-7",
            "text": (
                "The drive motor for pump P-101 is rated 75 kW at 1480 rpm, 400 V, 50 Hz, IP55. Service factor 1.15. "
                "Insulation class F with class B temperature rise at 40 degC ambient. The motor shall be suitable for VFD operation with an output filter. "
                "Full load current is 135 A; locked rotor current is 6.5 times full load current. Bearing lubrication interval is 8000 hours."
            ),
        },
        {
            "id": "HT-22",
            "source": "Hydrostatic Test Report HT-22",
            "text": (
                "Hydrostatic test of casing serial 0451 for pump P-101. Test pressure 24 bar held for 30 minutes with no visible leakage or pressure drop. "
                "Test water temperature 18 degC. Gauge calibration certificate G-1182 valid. Witnessed by the client inspector on behalf of the purchaser. Result: accepted."
            ),
        },
        {
            "id": "FMEA-CW",
            "source": "FMEA Cooling Water System Rev B",
            "text": (
                "Failure mode CW-03: seawater pump casing corrosion leading to through-wall leak. Cause: corrosion allowance consumed, coating damage. "
                "Effect: loss of cooling water, forced shutdown. Severity 8, occurrence 3, detection 4, RPN 96. Mitigation: 3 mm corrosion allowance, annual UT thickness survey, cathodic protection. "
                "Failure mode CW-07: motor overheating under VFD operation. Cause: inadequate cooling at low speed. Severity 6, occurrence 2, detection 3, RPN 36. Mitigation: minimum speed 30 percent, winding temperature monitoring."
            ),
        },
        {
            "id": "INC-2041",
            "source": "Incident Report INC-2041",
            "text": (
                "On commissioning, pump P-102 tripped on high vibration (12 mm/s RMS at the drive end bearing). Root cause: misalignment of 0.35 mm offset exceeding the 0.05 mm tolerance after the hot alignment check was skipped. "
                "Corrective action: realignment and procedure update MP-14 Rev D requiring a hot alignment check within 4 hours of first run. No damage to the casing or impeller was found."
            ),
        },
        {
            "id": "INC-2057",
            "source": "Incident Report INC-2057",
            "text": (
                "Pump P-101 strainer differential pressure alarm after 3 weeks of operation. Root cause: marine growth fouling the suction strainer. "
                "Corrective action: strainer cleaning interval reduced from 8 weeks to 3 weeks and chlorination dosing increased to 0.5 ppm residual. Vibration remained below 4.5 mm/s RMS throughout."
            ),
        },
        {
            "id": "MP-14",
            "source": "Maintenance Procedure MP-14 Rev D",
            "text": (
                "Step 1. Isolate the pump and verify zero energy. Step 2. Check coupling alignment cold; offset shall not exceed 0.05 mm and angularity 0.05 mm per 100 mm. "
                "Step 3. Run the pump for 4 hours and repeat the alignment check hot. Step 4. Measure vibration at both bearings; the acceptance limit is 4.5 mm/s RMS per ISO 10816-3 zone B. "
                "Step 5. Record ultrasonic thickness at the four casing survey points; minimum remaining wall is nominal thickness minus the 3 mm corrosion allowance."
            ),
        },
        {
            "id": "REQ-CW",
            "source": "Cooling Water Requirements REQ-CW-001",
            "text": (
                "REQ-CW-010 The cooling water system shall deliver 450 m3/h at 42 m head with one pump in standby. "
                "REQ-CW-020 Seawater-wetted pressure parts shall have a minimum 3 mm corrosion allowance or be of corrosion-resistant alloy. "
                "REQ-CW-030 Rotating equipment shall meet ISO 10816-3 zone B vibration limits. REQ-CW-040 Motors shall be rated for VFD operation."
            ),
        },
    ]


def sample_qa() -> list[QAExample]:
    """Labelled questions over :func:`sample_corpus` (answerable and unanswerable)."""
    return [
        QAExample(question="What corrosion allowance does the specification require for the pump casing?", reference_answer="3 mm corrosion allowance for the pump casing", relevant_ids=["PS-100", "REQ-CW"]),
        QAExample(question="What is the rated power of the drive motor for pump P-101?", reference_answer="75 kW at 1480 rpm", relevant_ids=["MD-7"]),
        QAExample(question="What pressure was held during the hydrostatic test of casing serial 0451?", reference_answer="24 bar held for 30 minutes", relevant_ids=["HT-22"]),
        QAExample(question="What was the root cause of the high vibration trip on pump P-102 and which procedure was updated?", reference_answer="misalignment of 0.35 mm offset; procedure MP-14 Rev D", relevant_ids=["INC-2041", "MP-14"]),
        QAExample(question="What is the vibration acceptance limit for the pumps?", reference_answer="4.5 mm/s RMS per ISO 10816-3 zone B", relevant_ids=["MP-14", "REQ-CW"]),
        QAExample(question="What is the warranty period offered by the pump vendor?", reference_answer="", relevant_ids=[], answerable=False),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m pipeline_rag.evaluation [--documents PATH ...] [--offline] [--json] [--domain NAME] [--n N]``.

    Without ``--documents`` the labelled :func:`sample_qa` set is run over
    :func:`sample_corpus`; with documents, :func:`synthetic_qa` builds ``n``
    questions from them. The exit code is 0 whenever the run completes; the
    release gates are reported, not enforced, because the lexical judges used
    offline are proxies (see the README's evaluation section).
    """
    import argparse
    import json
    import os

    from pipeline_core import make_llm

    from .cli import load_documents

    ap = argparse.ArgumentParser(prog="python -m pipeline_rag.evaluation", description="Offline RAG evaluation harness (RAGAS-style metrics, citation metrics, failure points).")
    ap.add_argument("--documents", action="append", default=[], metavar="PATH", help="File, JSON list or directory of documents; repeatable")
    ap.add_argument("--domain", default="general", help="DomainProfile name for the generated questions")
    ap.add_argument("--n", type=int, default=10, help="Synthetic questions to generate when --documents is given")
    ap.add_argument("--offline", action="store_true", help="Use the deterministic MockLLM and heuristic judges")
    ap.add_argument("--json", action="store_true", help="Print the full EvaluationReport as JSON")
    args = ap.parse_args(argv)
    if args.offline:
        os.environ["PIPELINE_OFFLINE"] = "1"
    llm = MockLLM() if args.offline else make_llm()
    model = PipelineRAGModel(llm)
    if args.documents:
        documents: Any = load_documents(args.documents)
        examples = synthetic_qa(documents, n=args.n)
        for ex in examples:
            ex.domain = args.domain
    else:
        documents = sample_corpus()
        examples = sample_qa()
    report = RAGEvaluator(model).evaluate_sync(examples, documents=documents)
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
        print("routes:", report.route_distribution, "escalation_rate:", round(report.escalation_rate, 2))
        print("failure points:", report.failure_points or "none")
        for rec in report.records:
            flag = "abstained" if rec.abstained else f"f1={rec.token_f1 if rec.token_f1 is not None else '-'}"
            print(f"  [{rec.status}] {rec.route:<11} {flag:<14} {rec.question[:80]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EvalRecord",
    "EvaluationReport",
    "FaithfulnessJudgement",
    "Judge",
    "JudgedStatement",
    "QAExample",
    "RAGEvaluator",
    "RelevanceJudgement",
    "SyntheticQA",
    "abstention_metrics",
    "attribute_failure_point",
    "citation_metrics",
    "context_precision",
    "context_recall",
    "exact_match",
    "faithfulness",
    "main",
    "mean",
    "normalize_answer",
    "sample_corpus",
    "sample_qa",
    "synthetic_qa",
    "synthetic_qa_llm",
    "token_f1",
]
