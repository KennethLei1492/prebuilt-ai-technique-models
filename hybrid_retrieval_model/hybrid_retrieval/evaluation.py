"""Retrieval metrics, an offline evaluation harness and a synthetic engineering corpus.

Metrics follow the dossier's evaluation guidance: nDCG@10 as the headline
graded metric (Järvelin & Kekäläinen 2002; BEIR standard), Recall@k for
first-stage headroom, MRR for single-answer tasks, Precision@k and Success@k.
All metric functions are pure and operate on ``ranked_ids`` plus a
``qrels: {doc_id: grade}`` mapping (grade > 0 means relevant).

:func:`evaluate` runs any ``Retriever`` over a list of :class:`QueryJudgment`
and returns an :class:`EvaluationReport`; :func:`ablation` runs the
sparse-only, dense-only and hybrid plans so the BM25 baseline is always
reported next to the hybrid result (Ma et al. 2021 pitfall).
:func:`synthetic_engineering_corpus` generates a deterministic corpus of
specifications, incidents, runbooks and datasheets with judged queries, used
by the CLI demo, the examples and the self-tests.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import Evidence, Usage, run_sync
from pipeline_core.protocols import Retriever

# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


def _relevant(qrels: Mapping[str, float]) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the log2(i+1) position discount (1-based positions)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    """Normalised DCG over graded judgments; 0 when nothing is relevant."""
    if k <= 0 or not qrels:
        return 0.0
    gains = [float(qrels.get(d, 0.0)) for d in list(ranked_ids)[:k]]
    ideal = sorted((float(g) for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    rel = _relevant(qrels)
    if not rel or k <= 0:
        return 0.0
    return len(rel & set(list(ranked_ids)[:k])) / len(rel)


def precision_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0
    rel = _relevant(qrels)
    return sum(1 for d in top if d in rel) / k


def mrr(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: Optional[int] = None) -> float:
    """Reciprocal rank of the first relevant result; ``k=None`` means no cutoff, ``k <= 0`` yields 0 like the other metrics."""
    rel = _relevant(qrels)
    if k is not None and k <= 0:
        return 0.0
    ids = list(ranked_ids) if k is None else list(ranked_ids)[:k]
    for i, d in enumerate(ids, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def success_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    """1.0 when at least one relevant document is in the top-k (DPR top-k accuracy)."""
    return 1.0 if _relevant(qrels) & set(list(ranked_ids)[:k]) else 0.0


def collapse_to_documents(evidence: Sequence[Evidence]) -> list[str]:
    """Chunk-level evidence -> ordered unique parent document ids (first occurrence wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for ev in evidence:
        doc = str(ev.metadata.get("doc_id") or ev.id.split("#")[0])
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class QueryJudgment(BaseModel):
    """One evaluation query with graded relevance labels keyed by document id."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    qrels: dict[str, float] = Field(description="document id -> grade (0 = not relevant, 3 = directly answers)")
    filters: Optional[dict[str, Any]] = None
    kind: str = Field(default="", description="identifier | natural_language | exploratory ... (for breakdowns)")


class QueryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    kind: str = ""
    ndcg: float
    recall: float
    precision: float
    mrr: float
    success: float
    n_results: int
    latency_s: float
    top_ids: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Aggregated metrics over a judged query set."""

    model_config = ConfigDict(extra="forbid")

    name: str = "hybrid"
    k: int
    n_queries: int
    ndcg_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    success_at_k: float
    mean_latency_s: float
    by_kind: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_query: list[QueryMetrics] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    def summary(self) -> str:
        line = (
            f"[{self.name}] n={self.n_queries} nDCG@{self.k}={self.ndcg_at_k:.3f} Recall@{self.k}={self.recall_at_k:.3f} "
            f"P@{self.k}={self.precision_at_k:.3f} MRR={self.mrr:.3f} Success@{self.k}={self.success_at_k:.3f} "
            f"latency={self.mean_latency_s * 1000:.1f}ms/query"
        )
        for kind, m in sorted(self.by_kind.items()):
            line += f"\n    {kind:<18} n={int(m['n']):>3} nDCG={m['ndcg']:.3f} Recall={m['recall']:.3f} MRR={m['mrr']:.3f}"
        return line

    def worst(self, n: int = 3) -> list[QueryMetrics]:
        return sorted(self.per_query, key=lambda q: (q.ndcg, q.recall))[:n]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows: Sequence[QueryMetrics], *, k: int, name: str = "hybrid", usage: Usage | None = None) -> EvaluationReport:
    """Pure aggregation of per-query metrics into a report (with per-kind breakdown)."""
    kinds: dict[str, list[QueryMetrics]] = {}
    for r in rows:
        kinds.setdefault(r.kind or "unlabelled", []).append(r)
    by_kind = {
        kd: {"n": float(len(rs)), "ndcg": _mean([r.ndcg for r in rs]), "recall": _mean([r.recall for r in rs]), "mrr": _mean([r.mrr for r in rs])}
        for kd, rs in kinds.items()
    }
    return EvaluationReport(
        name=name,
        k=k,
        n_queries=len(rows),
        ndcg_at_k=_mean([r.ndcg for r in rows]),
        recall_at_k=_mean([r.recall for r in rows]),
        precision_at_k=_mean([r.precision for r in rows]),
        mrr=_mean([r.mrr for r in rows]),
        success_at_k=_mean([r.success for r in rows]),
        mean_latency_s=_mean([r.latency_s for r in rows]),
        by_kind=by_kind if len(by_kind) > 1 else {},
        per_query=list(rows),
        usage=usage or Usage(),
    )


def score_ranking(j: QueryJudgment, ranked_ids: Sequence[str], *, k: int, latency_s: float = 0.0) -> QueryMetrics:
    return QueryMetrics(
        query_id=j.query_id,
        kind=j.kind,
        ndcg=ndcg_at_k(ranked_ids, j.qrels, k),
        recall=recall_at_k(ranked_ids, j.qrels, k),
        precision=precision_at_k(ranked_ids, j.qrels, k),
        mrr=mrr(ranked_ids, j.qrels, k),
        success=success_at_k(ranked_ids, j.qrels, k),
        n_results=len(ranked_ids),
        latency_s=latency_s,
        top_ids=list(ranked_ids)[:k],
    )


async def evaluate(
    retriever: Retriever,
    judgments: Sequence[QueryJudgment] | Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
    document_level: bool = True,
    chunk_factor: int = 3,
    name: str = "hybrid",
) -> EvaluationReport:
    """Run ``retriever.retrieve`` for every judged query and aggregate the metrics.

    ``judgments`` may be a list of :class:`QueryJudgment` or a ``{query: qrels}``
    mapping. With ``document_level`` the chunk-level evidence is collapsed to
    parent document ids before scoring (``chunk_factor * k`` chunks are
    requested so that enough distinct documents surface).
    """
    if isinstance(judgments, Mapping):
        judgments = [QueryJudgment(query_id=f"q{i + 1}", query=q, qrels=dict(r)) for i, (q, r) in enumerate(judgments.items())]
    rows: list[QueryMetrics] = []
    usage = Usage()
    before = getattr(retriever, "usage", None)
    for j in judgments:
        t0 = time.perf_counter()
        evidence = await retriever.retrieve(j.query, k=k * chunk_factor if document_level else k, filters=j.filters)
        dt = time.perf_counter() - t0
        ranked = collapse_to_documents(evidence) if document_level else [e.id for e in evidence]
        rows.append(score_ranking(j, ranked, k=k, latency_s=dt))
    after = getattr(retriever, "usage", None)
    if isinstance(before, Usage) and isinstance(after, Usage):
        usage = Usage(llm_calls=after.llm_calls - before.llm_calls, input_tokens=after.input_tokens - before.input_tokens, output_tokens=after.output_tokens - before.output_tokens, cost_usd=after.cost_usd - before.cost_usd)
    return aggregate(rows, k=k, name=name, usage=usage)


def evaluate_sync(retriever: Retriever, judgments: Any, **kw: Any) -> EvaluationReport:
    """Synchronous wrapper around :func:`evaluate`."""
    return run_sync(evaluate(retriever, judgments, **kw))


async def ablation(retriever: Any, judgments: Sequence[QueryJudgment], *, k: int = 10, fusion: str | None = None) -> dict[str, EvaluationReport]:
    """Evaluate sparse-only, dense-only and the hybrid plan on a :class:`HybridRetriever` (or model exposing ``.retriever``).

    Reports the lexical baseline next to the hybrid result, as Ma et al. (2021)
    recommend, so fusion gains are never claimed against an unmeasured baseline.
    """
    hr = getattr(retriever, "retriever", retriever)
    reports: dict[str, EvaluationReport] = {}
    for name, use_sparse, use_dense in (("sparse", True, False), ("dense", False, True), ("hybrid", True, True)):
        rows: list[QueryMetrics] = []
        for j in judgments:
            plan = hr.plan_for(j.query, k=k * 3, filters=j.filters)
            plan.use_sparse, plan.use_dense = use_sparse, use_dense
            plan.rerank = "none"
            if fusion:
                plan.fusion = fusion  # type: ignore[assignment]
            t0 = time.perf_counter()
            res = await hr.search(plan)
            rows.append(score_ranking(j, collapse_to_documents(res.evidence), k=k, latency_s=time.perf_counter() - t0))
        reports[name] = aggregate(rows, k=k, name=name)
    return reports


# ---------------------------------------------------------------------------
# Synthetic engineering corpus
# ---------------------------------------------------------------------------

_COMPONENTS = [
    ("pump casing", "PS-4471"), ("pump impeller", "PS-4472"), ("pump shaft", "PS-4473"), ("bearing housing", "PS-4474"),
    ("valve body", "VS-2101"), ("heat exchanger tubes", "HX-3310"), ("vessel shell", "VS-7700"), ("nozzle flange", "VS-7701"),
    ("compressor rotor", "CS-5120"), ("gearbox housing", "GS-6001"), ("agitator blade", "AG-1180"), ("expansion joint", "EJ-0420"),
]
_MATERIALS = [
    ("ASTM A216 WCB carbon steel", "WCB"), ("duplex stainless steel ASTM A890 Grade 4A", "A890 4A"), ("austenitic stainless steel ASTM A351 CF8M", "CF8M"),
    ("17-4PH precipitation hardening steel", "17-4PH"), ("nodular cast iron EN-GJS-400-15", "GJS-400"), ("nickel alloy UNS N06625", "N06625"),
    ("titanium grade 2 ASTM B265", "Ti Gr2"), ("aluminium bronze ASTM B148 C95800", "C95800"),
]
_ALLOWANCES = ["1.5 mm", "3 mm", "4.5 mm", "6 mm"]
_PRESSURES = ["10 bar", "16 bar", "25 bar", "40 bar", "63 bar"]
_TEMPS = ["80", "120", "150", "200", "250"]
_SERVICES = ["cooling water", "lube oil", "instrument air", "hydraulic", "firewater", "boiler feedwater"]
_MODES = ["cavitation", "overheating", "seal leak", "high vibration", "blocked strainer", "bearing failure"]
_ERR_CAUSES = [
    ("heartbeat from the field bus is lost for longer than the configured timeout", "restore communication, check the bus terminator and clear the fault"),
    ("the configuration file fails schema validation on start-up", "fix the offending key reported in the log and restart the service"),
    ("the licence server cannot be reached within 30 seconds", "verify DNS and firewall rules for port 27000 and retry"),
    ("a write to the historian database is rejected because the tag does not exist", "create the tag in the historian or correct the mapping table"),
    ("the actuator feedback disagrees with the commanded position by more than 5 percent", "recalibrate the positioner and inspect the linkage"),
    ("the redundant controller fails to synchronise after a firmware update", "update both controllers to the same firmware revision and resynchronise"),
]
_PROPS = [("yield strength", "MPa", (200, 900)), ("tensile strength", "MPa", (400, 1100)), ("hardness", "HB", (120, 400)), ("maximum service temperature", "degC", (150, 650))]


def synthetic_engineering_corpus(n_docs: int = 24, *, seed: int = 7) -> tuple[list[dict[str, Any]], list[QueryJudgment]]:
    """Deterministic corpus of specifications, incidents, runbooks and datasheets with judged queries.

    Returns ``(documents, judgments)``; documents are dicts accepted by
    ``HybridRetrievalModel.index`` with ``metadata`` carrying ``type``,
    ``severity`` and (for incidents) ``service`` so filters can be exercised.
    """
    rng = random.Random(seed)
    docs: list[dict[str, Any]] = []
    judgments: list[QueryJudgment] = []
    incidents: list[tuple[str, str, str]] = []
    mats_used: dict[str, list[str]] = {}
    kinds = ["spec", "incident", "runbook", "datasheet"]
    counters = {k: 0 for k in kinds}
    for i in range(n_docs):
        kind = kinds[i % len(kinds)]
        n = counters[kind]
        counters[kind] += 1
        if kind == "spec":
            comp, code = _COMPONENTS[n % len(_COMPONENTS)]
            mat, short = _MATERIALS[(n * 3 + rng.randrange(2)) % len(_MATERIALS)]
            allowance = _ALLOWANCES[n % len(_ALLOWANCES)]
            pressure, temp = _PRESSURES[n % len(_PRESSURES)], _TEMPS[n % len(_TEMPS)]
            rev = "ABC"[n % 3]
            doc_id = f"spec-{code}"
            text = (
                f"# {comp.capitalize()} specification {code} rev {rev}\n\n"
                f"## 1 Scope\nThis specification covers the design, materials and inspection of the {comp} for the process unit.\n\n"
                f"## 2 Materials\nThe {comp} shall be {mat}. A corrosion allowance of {allowance} shall be applied to all wetted surfaces. "
                f"Material certificates to EN 10204 3.1 are required.\n\n"
                f"## 3 Design conditions\nDesign pressure {pressure} at {temp} degC. Hydrostatic test at 1.5 times the design pressure for 30 minutes.\n\n"
                f"## 4 Inspection\nDimensional check per drawing; surface examination of all weld seams; hardness test on {short} components."
            )
            docs.append({"id": doc_id, "text": text, "metadata": {"title": f"{comp.capitalize()} specification {code}", "type": "specification", "revision": rev, "severity": 2}})
            mats_used.setdefault(short, []).append(doc_id)
            judgments.append(QueryJudgment(query_id=f"spec-ca-{n}", query=f"What corrosion allowance is specified for the {comp}?", qrels={doc_id: 3.0}, kind="natural_language"))
            judgments.append(QueryJudgment(query_id=f"spec-id-{n}", query=f"{code} material", qrels={doc_id: 3.0}, kind="identifier"))
        elif kind == "incident":
            inc = f"INC-{4800 + n}"
            service = _SERVICES[n % len(_SERVICES)]
            mode = _MODES[(n * 2) % len(_MODES)]
            doc_id = f"incident-{inc}"
            text = (
                f"# Incident {inc}: {service} pump {mode}\n\n"
                f"Severity {1 + n % 3} incident on the {service} system. Pump P-{200 + n} showed {mode} during the night shift and the "
                f"standby unit was started. Root cause: {mode} caused by inadequate maintenance interval and a missing alarm. "
                f"Corrective action: revise the inspection interval, add a condition monitoring alarm and update the operating procedure."
            )
            docs.append({"id": doc_id, "text": text, "metadata": {"title": f"Incident {inc}", "type": "incident", "severity": 1 + n % 3, "service": service}})
            incidents.append((doc_id, service, mode))
            judgments.append(QueryJudgment(query_id=f"inc-id-{n}", query=f"{inc} root cause", qrels={doc_id: 3.0}, kind="identifier"))
        elif kind == "runbook":
            code = f"ERR-{1200 + n}"
            cause, fix = _ERR_CAUSES[n % len(_ERR_CAUSES)]
            doc_id = f"runbook-{code}"
            text = (
                f"# Runbook: fault {code}\n\n## Symptom\n{code} is raised when {cause}.\n\n## Resolution\nTo clear the fault, {fix}. "
                f"If the fault persists, escalate to second-line support with the controller diagnostics export.\n\n"
                f"```text\nalarm {code} severity=major source=controller\n```"
            )
            docs.append({"id": doc_id, "text": text, "metadata": {"title": f"Runbook {code}", "type": "runbook", "severity": 3}})
            judgments.append(QueryJudgment(query_id=f"err-id-{n}", query=code, qrels={doc_id: 3.0}, kind="identifier"))
            judgments.append(QueryJudgment(query_id=f"err-nl-{n}", query=f"what causes {code} and how is it cleared", qrels={doc_id: 3.0}, kind="natural_language"))
        else:
            mat, short = _MATERIALS[n % len(_MATERIALS)]
            doc_id = f"datasheet-{short.replace(' ', '-')}"
            props = ", ".join(f"{p} {rng.randrange(lo, hi, 5)} {u}" for p, u, (lo, hi) in _PROPS)
            text = (
                f"# Material datasheet: {mat}\n\n"
                f"Designation {short}. Typical properties at room temperature: {props}. "
                f"Suitable for {['seawater', 'steam', 'hydrocarbon', 'acid'][n % 4]} service; weldable with matching consumables and post-weld heat treatment where specified."
            )
            docs.append({"id": doc_id, "text": text, "metadata": {"title": f"Datasheet {short}", "type": "datasheet", "severity": 3}})
            qrels = {doc_id: 3.0}
            for sid in mats_used.get(short, []):
                qrels[sid] = 1.0
            judgments.append(QueryJudgment(query_id=f"ds-nl-{n}", query=f"yield strength of {short}", qrels=qrels, kind="natural_language"))
    # exploratory incident queries: same failure mode across services
    modes_seen: dict[str, dict[str, float]] = {}
    for doc_id, service, mode in incidents:
        modes_seen.setdefault(mode, {})[doc_id] = 3.0
    for i, (mode, qrels) in enumerate(sorted(modes_seen.items())):
        judgments.append(QueryJudgment(query_id=f"inc-ex-{i}", query=f"have there been previous {mode} incidents on pumps?", qrels=qrels, kind="exploratory"))
    return docs, judgments
