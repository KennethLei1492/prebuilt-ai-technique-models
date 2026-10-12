"""Pure fusion maths: RRF, normalisation, convex combination, DBSF, interleave, overlap."""
from __future__ import annotations

import math

import pytest

from hybrid_retrieval.fusion import (
    channel_overlap,
    convex_combination,
    distribution_based_score_fusion,
    fuse,
    interleave,
    normalize_scores,
    reciprocal_rank_fusion,
)

SPARSE = [("a", 12.0), ("b", 9.0), ("c", 1.0)]
DENSE = [("b", 0.9), ("d", 0.8), ("a", 0.7)]


def test_rrf_scores_and_order() -> None:
    fused = reciprocal_rank_fusion([SPARSE, DENSE], k=60)
    scores = dict(fused)
    assert math.isclose(scores["a"], 1 / 61 + 1 / 63)
    assert math.isclose(scores["b"], 1 / 62 + 1 / 61)
    assert math.isclose(scores["d"], 1 / 62)
    assert [d for d, _ in fused][:2] == ["b", "a"]  # b: ranks 2+1 beats a: ranks 1+3
    assert fused == sorted(fused, key=lambda kv: (-kv[1], kv[0]))


def test_rrf_weights_and_validation() -> None:
    fused = dict(reciprocal_rank_fusion([SPARSE, DENSE], k=60, weights=[1.0, 0.0]))
    assert fused["a"] > fused["b"] > fused["c"] > fused["d"] == 0.0  # zero-weight channel contributes nothing
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([SPARSE, DENSE], k=-1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([SPARSE, DENSE], weights=[1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([SPARSE], weights=[-1.0])


def test_rrf_duplicates_within_a_list_count_once() -> None:
    fused = dict(reciprocal_rank_fusion([[("a", 1.0), ("a", 0.5)]], k=60))
    assert math.isclose(fused["a"], 1 / 61)


def test_normalize_minmax_zscore_theoretical() -> None:
    assert normalize_scores([1.0, 3.0, 5.0], "minmax") == [0.0, 0.5, 1.0]
    assert normalize_scores([2.0, 2.0], "minmax") == [1.0, 1.0]
    assert normalize_scores([0.0, 0.0], "minmax") == [0.0, 0.0]
    z = normalize_scores([1.0, 2.0, 3.0], "zscore")
    assert math.isclose(sum(z), 0.0, abs_tol=1e-9) and z[0] < z[1] < z[2]
    assert normalize_scores([5.0, 5.0], "zscore") == [0.0, 0.0]
    th = normalize_scores([-1.0, 0.0, 1.0, 2.0], "theoretical", score_range=(-1.0, 1.0))
    assert th == [0.0, 0.5, 1.0, 1.0]
    assert normalize_scores([1.0, 3.0], "theoretical") == [0.0, 1.0]  # no range -> minmax fallback
    assert normalize_scores([], "minmax") == []
    with pytest.raises(ValueError):
        normalize_scores([1.0], "bogus")
    with pytest.raises(ValueError):
        normalize_scores([1.0], "theoretical", score_range=(1.0, 0.0))


def test_convex_combination_weights_and_missing_docs() -> None:
    # alpha extremes reproduce the single-channel order
    sparse_only = [d for d, _ in convex_combination([SPARSE, DENSE], [1.0, 0.0], "minmax")]
    assert sparse_only[:3] == ["a", "b", "c"]
    dense_only = [d for d, _ in convex_combination([SPARSE, DENSE], [0.0, 1.0], "minmax")]
    assert dense_only[:3] == ["b", "d", "a"]
    # weights are renormalised; missing docs contribute 0 on that channel
    fused = dict(convex_combination([SPARSE, DENSE], [2.0, 2.0], "minmax"))
    assert math.isclose(fused["c"], 0.0) and math.isclose(fused["d"], 0.5 * 0.5)
    assert math.isclose(fused["b"], 0.5 * (8 / 11) + 0.5 * 1.0)
    with pytest.raises(ValueError):
        convex_combination([SPARSE, DENSE], [0.0, 0.0])
    with pytest.raises(ValueError):
        convex_combination([SPARSE, DENSE], [0.5, 0.5], ranges=[None])


def test_convex_theoretical_range_for_dense_channel() -> None:
    fused = dict(convex_combination([SPARSE, DENSE], [0.3, 0.7], "theoretical", ranges=[None, (-1.0, 1.0)]))
    # dense 0.9 -> 0.95, sparse b 9 -> 8/11 ; a: sparse 1.0, dense 0.7 -> 0.85
    assert math.isclose(fused["b"], 0.3 * (8 / 11) + 0.7 * 0.95)
    assert math.isclose(fused["a"], 0.3 * 1.0 + 0.7 * 0.85)
    assert fused["a"] > fused["b"] > fused["d"] > fused["c"] == 0.0


def test_convex_zscore_is_shifted_non_negative() -> None:
    fused = dict(convex_combination([SPARSE, DENSE], [0.5, 0.5], "zscore"))
    assert all(v >= 0 for v in fused.values())
    assert fused["c"] < fused["a"]


def test_dbsf_bounds_and_order() -> None:
    fused = distribution_based_score_fusion([SPARSE, DENSE])
    scores = dict(fused)
    assert all(0.0 <= v <= 2.0 for v in scores.values())
    assert fused == sorted(fused, key=lambda kv: (-kv[1], kv[0]))
    assert scores["c"] < scores["a"]
    assert distribution_based_score_fusion([[], []]) == []


def test_interleave_round_robin_dedupes() -> None:
    merged = interleave([SPARSE, DENSE])
    assert [d for d, _ in merged] == ["a", "b", "d", "c"]
    assert [s for _, s in merged] == [1.0, 0.5, 1 / 3, 0.25]


def test_channel_overlap() -> None:
    assert channel_overlap(SPARSE, DENSE, 3) == pytest.approx(2 / 3)
    assert channel_overlap(SPARSE, DENSE, 2) == pytest.approx(1 / 2)
    assert channel_overlap([], DENSE, 3) == 0.0
    assert channel_overlap(SPARSE, DENSE, 0) == 0.0


def test_fuse_dispatch() -> None:
    assert fuse("rrf", [SPARSE, DENSE])[0][0] == "b"
    assert fuse("convex", [SPARSE, DENSE], weights=[1.0, 0.0], normalization="minmax")[0][0] == "a"
    assert fuse("dbsf", [SPARSE, DENSE])
    assert fuse("interleave", [SPARSE, DENSE])[0][0] == "a"
    with pytest.raises(ValueError):
        fuse("borda", [SPARSE, DENSE])
