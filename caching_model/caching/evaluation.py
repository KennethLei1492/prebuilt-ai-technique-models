"""Metrics and an offline evaluation harness for the caching model.

Metrics follow the dossier's evaluation list: per-tier hit rate, cost and
latency saved from the pricing table, provider prefix hit ratio, semantic
false-hit audit (precision/recall/AUC on labelled same-answer pairs) and
eviction quality against Belady's offline optimum (OPT, 1966).

Run the harness::

    python -m caching.evaluation            # text summary
    python -m caching.evaluation --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import PRICING_USD_PER_MTOK, Usage

from .backends import InMemoryBackend
from .config import EvictionPolicy
from .keys import extract_guards
from .semantic import HashingEmbedder, SemanticCache, cosine


class EvaluationReport(BaseModel):
    """Everything :func:`run_offline_evaluation` produces."""

    model_config = ConfigDict(extra="forbid")

    trace_length: int
    unique_keys: int
    capacity: int
    hit_rate_by_policy: dict[str, float]
    belady_opt_hit_rate: float
    opt_gap_pp: dict[str, float] = Field(description="Percentage points below OPT per policy")
    semantic_audit: dict[str, float]
    semantic_audit_without_guards: dict[str, float]
    cost_saved_usd_per_1000_hits: float
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Cost / latency
# --------------------------------------------------------------------------


def cost_saved(model: str, usage_of_uncached_call: Usage) -> float:
    """Dollars saved by serving one cached response instead of re-running ``usage_of_uncached_call``."""
    p = PRICING_USD_PER_MTOK.get(model)
    if p is None:
        return usage_of_uncached_call.cost_usd
    return (usage_of_uncached_call.input_tokens * p["input"] + usage_of_uncached_call.output_tokens * p["output"] + usage_of_uncached_call.cache_read_input_tokens * p["cache_read"] + usage_of_uncached_call.cache_creation_input_tokens * p["cache_write"]) / 1_000_000


def prefix_savings(model: str, prefix_tokens: int, uses: int, *, ttl: Literal["5m", "1h"] = "5m", write_multiplier_5m: float = 1.25, write_multiplier_1h: float = 2.0, read_multiplier: float = 0.1) -> dict[str, float]:
    """Compare ``uses`` uncached calls against one write plus ``uses-1`` reads of a prefix."""
    p = PRICING_USD_PER_MTOK.get(model, {"input": 0.0})
    base = prefix_tokens * p["input"] / 1_000_000
    write = base * (write_multiplier_1h if ttl == "1h" else write_multiplier_5m)
    uncached = base * uses
    cached = write + base * read_multiplier * max(0, uses - 1)
    return {"uncached_usd": uncached, "cached_usd": cached, "saved_usd": uncached - cached, "saved_pct": ((uncached - cached) / uncached * 100.0) if uncached else 0.0}


def latency_saved(hit_count: int, mean_miss_latency_s: float, mean_hit_latency_s: float = 0.002) -> float:
    return max(0.0, hit_count * (mean_miss_latency_s - mean_hit_latency_s))


# --------------------------------------------------------------------------
# Eviction quality: policy simulation vs Belady OPT
# --------------------------------------------------------------------------


def belady_opt_hits(trace: Sequence[str], capacity: int) -> int:
    """Hits achieved by Belady's OPT (evict the key whose next use is furthest away)."""
    if capacity <= 0:
        return 0
    next_use: list[int] = [0] * len(trace)
    last: dict[str, int] = {}
    inf = len(trace) + 1
    for i in range(len(trace) - 1, -1, -1):
        next_use[i] = last.get(trace[i], inf)
        last[trace[i]] = i
    cache: dict[str, int] = {}
    hits = 0
    for i, k in enumerate(trace):
        if k in cache:
            hits += 1
            cache[k] = next_use[i]
            continue
        if len(cache) >= capacity:
            victim = max(cache.items(), key=lambda kv: kv[1])[0]
            del cache[victim]
        cache[k] = next_use[i]
    return hits


def simulate_policy(trace: Sequence[str], capacity: int, policy: EvictionPolicy = "lru") -> int:
    """Hits of :class:`InMemoryBackend` with ``policy`` on a key trace (fake clock)."""
    t = [0.0]

    def clock() -> float:
        t[0] += 1.0
        return t[0]

    backend = InMemoryBackend(max_entries=capacity, max_bytes=10**12, policy=policy, clock=clock)
    hits = 0
    for k in trace:
        if backend.get(k) is not None:
            hits += 1
        else:
            backend.set(k, 1)
    return hits


def compare_policies(trace: Sequence[str], capacity: int, policies: Iterable[EvictionPolicy] = ("lru", "lfu", "fifo")) -> dict[str, float]:
    n = max(1, len(trace))
    out = {p: simulate_policy(trace, capacity, p) / n for p in policies}
    out["opt"] = belady_opt_hits(trace, capacity) / n
    return out


# --------------------------------------------------------------------------
# Semantic false-hit audit
# --------------------------------------------------------------------------


def auc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """Rank-based AUC (probability a positive outscores a negative)."""
    if not scores_pos or not scores_neg:
        return 0.5
    wins = 0.0
    for p in scores_pos:
        for n in scores_neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(scores_pos) * len(scores_neg))


def false_hit_audit(
    pairs: Sequence[tuple[str, str, bool]],
    *,
    embedder: Any = None,
    threshold: float = 0.95,
    guards: bool = True,
) -> dict[str, float]:
    """Audit a threshold rule on labelled pairs ``(query, cached_query, same_answer)``.

    Returns precision/recall/F1 of the hit decision, the false-hit rate among
    served hits, and the AUC of cosine similarity as a validity predictor.
    """
    emb = embedder or HashingEmbedder()
    tp = fp = fn = tn = 0
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for q, c, same in pairs:
        vq, vc = emb.embed([q, c])
        sim = cosine(vq, vc)
        (pos_scores if same else neg_scores).append(sim)
        hit = sim >= threshold
        if guards and extract_guards(q) != extract_guards(c):
            hit = False
        if hit and same:
            tp += 1
        elif hit and not same:
            fp += 1
        elif not hit and same:
            fn += 1
        else:
            tn += 1
    served = tp + fp
    precision = tp / served if served else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "pairs": float(len(pairs)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_hit_rate": (fp / served) if served else 0.0,
        "auc": auc(pos_scores, neg_scores),
        "hits_served": float(served),
    }


# --------------------------------------------------------------------------
# Synthetic engineering workload
# --------------------------------------------------------------------------

_CLAUSES = ["4.2", "5.1", "6.3", "7.4", "8.1"]
_PARTS = ["ABC-1234", "ABC-1235", "XYZ-77", "REQ-120", "REQ-121"]
_TEMPLATES = [
    "What does clause {c} of ISO 2768 require for a {n} mm dimension?",
    "Which torque applies to fastener {p} at {n} Nm?",
    "Summarize the acceptance criteria for requirement {p}.",
    "Explain the derating rule for component {p} at {n} degC.",
]


def synthetic_trace(n: int = 400, *, hot_fraction: float = 0.3, seed: int = 7) -> list[str]:
    """Skewed key trace: a hot set reused often plus one-off scans (ARC/TinyLFU motivation)."""
    rng = random.Random(seed)
    hot = [f"hot-{i}" for i in range(20)]
    out: list[str] = []
    for i in range(n):
        if rng.random() < hot_fraction + 0.3:
            out.append(rng.choice(hot))
        else:
            out.append(f"scan-{i}")
    return out


def synthetic_pairs(n: int = 60, *, seed: int = 11) -> list[tuple[str, str, bool]]:
    """Labelled same-answer pairs with paraphrases (positive) and identifier/number swaps (negative)."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str, bool]] = []
    for _ in range(n):
        t = rng.choice(_TEMPLATES)
        c, p, num = rng.choice(_CLAUSES), rng.choice(_PARTS), rng.choice([5, 12, 50, 120])
        q = t.format(c=c, p=p, n=num)
        if rng.random() < 0.5:
            # positive: punctuation / politeness paraphrase
            para = q.rstrip("?.") + (" please?" if rng.random() < 0.5 else " ?")
            pairs.append((para, q, True))
        else:
            # negative: same wording, different identifier or number
            c2 = rng.choice([x for x in _CLAUSES if x != c])
            p2 = rng.choice([x for x in _PARTS if x != p])
            n2 = num * 10
            pairs.append((t.format(c=c2, p=p2, n=n2), q, False))
    return pairs


def run_offline_evaluation(*, trace_length: int = 400, capacity: int = 24, model: str = "claude-opus-5-5", seed: int = 7) -> EvaluationReport:
    """Run the full offline harness: eviction vs OPT, semantic audit with/without guards, cost model."""
    trace = synthetic_trace(trace_length, seed=seed)
    rates = compare_policies(trace, capacity)
    opt = rates.pop("opt")
    pairs = synthetic_pairs(seed=seed + 4)
    audit = false_hit_audit(pairs, threshold=0.95, guards=True)
    audit_ng = false_hit_audit(pairs, threshold=0.80, guards=False)
    typical = Usage(input_tokens=1500, output_tokens=400)
    saved = cost_saved(model, typical) * 1000
    notes = [
        "Policies: adopt ARC/W-TinyLFU only if LRU trails OPT by more than ~10 pp at the deployed capacity (dossier decision rule).",
        "Semantic audit: identifier guards drive the false-hit rate on identifier swaps to zero; the hashing embedder is an offline baseline, not a production embedding.",
    ]
    return EvaluationReport(
        trace_length=len(trace),
        unique_keys=len(set(trace)),
        capacity=capacity,
        hit_rate_by_policy={k: round(v, 4) for k, v in rates.items()},
        belady_opt_hit_rate=round(opt, 4),
        opt_gap_pp={k: round((opt - v) * 100.0, 2) for k, v in rates.items()},
        semantic_audit={k: round(v, 4) for k, v in audit.items()},
        semantic_audit_without_guards={k: round(v, 4) for k, v in audit_ng.items()},
        cost_saved_usd_per_1000_hits=round(saved, 4),
        notes=notes,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m caching.evaluation", description="Offline evaluation harness for the caching model")
    ap.add_argument("--trace-length", type=int, default=400)
    ap.add_argument("--capacity", type=int, default=24)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = run_offline_evaluation(trace_length=args.trace_length, capacity=args.capacity, model=args.model)
    if args.json:
        print(json.dumps(rep.model_dump(), indent=2))
        return 0
    print(f"Trace: {rep.trace_length} requests, {rep.unique_keys} unique keys, capacity {rep.capacity}")
    print(f"Belady OPT hit rate: {rep.belady_opt_hit_rate:.3f}")
    for k, v in rep.hit_rate_by_policy.items():
        print(f"  {k:<5} hit rate {v:.3f}  (gap to OPT {rep.opt_gap_pp[k]:.1f} pp)")
    print("Semantic audit (threshold 0.95, guards on):", {k: round(v, 3) for k, v in rep.semantic_audit.items()})
    print("Semantic audit (threshold 0.80, guards off):", {k: round(v, 3) for k, v in rep.semantic_audit_without_guards.items()})
    print(f"Cost saved per 1000 exact hits ({args.model}): ${rep.cost_saved_usd_per_1000_hits:.2f}")
    for n in rep.notes:
        print("-", n)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EvaluationReport",
    "auc",
    "belady_opt_hits",
    "compare_policies",
    "cost_saved",
    "false_hit_audit",
    "latency_saved",
    "prefix_savings",
    "run_offline_evaluation",
    "simulate_policy",
    "synthetic_pairs",
    "synthetic_trace",
]
