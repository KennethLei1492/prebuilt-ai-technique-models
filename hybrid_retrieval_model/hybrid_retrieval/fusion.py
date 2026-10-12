"""Rank and score fusion: pure functions over ``list[list[(doc_id, score)]]``.

* :func:`reciprocal_rank_fusion` - rank-only, zero configuration, k = 60
  (Cormack et al. 2009). Robust when channel scores are incomparable.
* :func:`convex_combination` - normalised weighted sum; theoretical min-max
  with semantic alpha in [0.6, 0.8] beat RRF on every dataset tested when a few
  labelled queries exist (Bruch et al. 2023).
* :func:`distribution_based_score_fusion` - normalises each channel by its
  mean +/- 3 standard deviations before summing; robust to heavy tails.
* :func:`interleave` - round-robin merge, useful as a diagnostic baseline.
* :func:`channel_overlap` - overlap@k between channels, a drift and
  complementarity signal used in the confidence estimate.

Every function is deterministic: ties break on ``doc_id``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

ScoredId = tuple[str, float]
Ranking = Sequence[ScoredId]


def _sorted(scores: dict[str, float]) -> list[ScoredId]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _weights(n: int, weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0] * n
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    return list(weights)


def reciprocal_rank_fusion(rankings: Sequence[Ranking], k: int = 60, *, weights: Sequence[float] | None = None) -> list[ScoredId]:
    """``score(d) = sum_i w_i / (k + rank_i(d))`` over the lists containing ``d`` (ranks are 1-based)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    ws = _weights(len(rankings), weights)
    fused: dict[str, float] = {}
    for w, ranking in zip(ws, rankings):
        seen: set[str] = set()
        for rank, (doc_id, _score) in enumerate(ranking, 1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return _sorted(fused)


def normalize_scores(scores: Sequence[float], method: str = "minmax", *, score_range: tuple[float, float] | None = None) -> list[float]:
    """Normalise a channel's scores.

    ``minmax`` maps observed [min, max] to [0, 1] (all-equal -> 1.0 if positive else 0.0).
    ``zscore`` returns standard scores (std 0 -> zeros).
    ``theoretical`` maps a known ``score_range`` to [0, 1] with clipping; without a
    range it falls back to ``minmax`` (Bruch et al. 2023 use the theoretical range
    for cosine and the observed range for BM25).
    """
    vals = list(scores)
    if not vals:
        return []
    if method == "theoretical":
        if score_range is None:
            method = "minmax"
        else:
            lo, hi = score_range
            if hi <= lo:
                raise ValueError("score_range must have hi > lo")
            return [min(1.0, max(0.0, (s - lo) / (hi - lo))) for s in vals]
    if method == "minmax":
        lo, hi = min(vals), max(vals)
        if hi - lo <= 1e-12:
            return [1.0 if hi > 0 else 0.0 for _ in vals]
        return [(s - lo) / (hi - lo) for s in vals]
    if method == "zscore":
        mean = sum(vals) / len(vals)
        var = sum((s - mean) ** 2 for s in vals) / len(vals)
        std = math.sqrt(var)
        if std <= 1e-12:
            return [0.0 for _ in vals]
        return [(s - mean) / std for s in vals]
    raise ValueError(f"unknown normalization {method!r}")


def convex_combination(
    score_lists: Sequence[Ranking],
    weights: Sequence[float],
    normalization: str = "theoretical",
    *,
    ranges: Sequence[Optional[tuple[float, float]]] | None = None,
) -> list[ScoredId]:
    """``score(d) = sum_i w_i * norm_i(s_i(d))``; a document missing from a channel scores that channel's minimum (0).

    ``weights`` should sum to 1 (they are renormalised if not). ``ranges`` gives the
    theoretical score range per channel (``None`` entries fall back to observed min-max).
    """
    ws = _weights(len(score_lists), weights)
    total = sum(ws)
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    ws = [w / total for w in ws]
    rngs = list(ranges) if ranges is not None else [None] * len(score_lists)
    if len(rngs) != len(score_lists):
        raise ValueError("ranges must align with score_lists")
    fused: dict[str, float] = {}
    for w, ranking, rng in zip(ws, score_lists, rngs):
        if not ranking:
            continue
        ids = [d for d, _ in ranking]
        normed = normalize_scores([s for _, s in ranking], normalization, score_range=rng)
        if normalization == "zscore":  # shift to non-negative so missing docs (0) are the minimum
            lo = min(normed)
            normed = [n - lo for n in normed]
        for d, n in zip(ids, normed):
            fused[d] = fused.get(d, 0.0) + w * n
    return _sorted(fused)


def distribution_based_score_fusion(score_lists: Sequence[Ranking], *, weights: Sequence[float] | None = None, sigmas: float = 3.0) -> list[ScoredId]:
    """Normalise each channel to ``[mean - sigmas*std, mean + sigmas*std]`` clipped to [0, 1], then weighted-sum."""
    ws = _weights(len(score_lists), weights)
    fused: dict[str, float] = {}
    for w, ranking in zip(ws, score_lists):
        if not ranking:
            continue
        vals = [s for _, s in ranking]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((s - mean) ** 2 for s in vals) / len(vals))
        lo, hi = mean - sigmas * std, mean + sigmas * std
        for d, s in ranking:
            n = 1.0 if hi - lo <= 1e-12 else min(1.0, max(0.0, (s - lo) / (hi - lo)))
            fused[d] = fused.get(d, 0.0) + w * n
    return _sorted(fused)


def interleave(rankings: Sequence[Ranking]) -> list[ScoredId]:
    """Round-robin merge; the score is ``1/position`` in the merged list."""
    seen: set[str] = set()
    out: list[ScoredId] = []
    iters = [list(r) for r in rankings]
    i = 0
    while any(i < len(r) for r in iters):
        for r in iters:
            if i < len(r):
                d = r[i][0]
                if d not in seen:
                    seen.add(d)
                    out.append((d, 1.0 / (len(out) + 1)))
        i += 1
    return out


def channel_overlap(a: Ranking, b: Ranking, k: int) -> float:
    """``|top-k(a) & top-k(b)| / k`` (0 when either list is empty)."""
    if k <= 0 or not a or not b:
        return 0.0
    sa = {d for d, _ in list(a)[:k]}
    sb = {d for d, _ in list(b)[:k]}
    return len(sa & sb) / k


def fuse(method: str, rankings: Sequence[Ranking], *, weights: Sequence[float] | None = None, rrf_k: int = 60, normalization: str = "theoretical", ranges: Sequence[Optional[tuple[float, float]]] | None = None) -> list[ScoredId]:
    """Dispatch on the plan's fusion method name (an empty channel simply contributes nothing)."""
    if method == "rrf":
        return reciprocal_rank_fusion(rankings, rrf_k, weights=weights)
    if method == "convex":
        ws = list(weights) if weights is not None else [1.0] * len(rankings)
        return convex_combination(rankings, ws, normalization, ranges=ranges)
    if method == "dbsf":
        return distribution_based_score_fusion(rankings, weights=weights)
    if method == "interleave":
        return interleave(rankings)
    raise ValueError(f"unknown fusion method {method!r}")
