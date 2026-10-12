"""Pure logic: DAG validation and waves, speedup estimates, aggregation maths."""
from __future__ import annotations

import math

import pytest

from pipeline_core import PlanValidationError

from parallelization.aggregation import (
    canonicalize_answer,
    combine_confidence,
    concat_in_order,
    confidence_from_vote,
    majority_vote,
    missing_branch_notice,
    vote_entropy,
    weighted_confidence,
)
from parallelization.dag import critical_path, to_mermaid, topological_waves, total_work, transitive_dependents, validate_dag
from parallelization.estimator import amdahl_bound, amdahl_speedup, estimate_fan_out, estimate_graph, recommended_concurrency, tradeoff_table, waves_for


# -------------------------------------------------------------------- dag


def test_validate_dag_rejects_cycles_unknown_and_self_dependencies():
    validate_dag({"a": [], "b": ["a"]})
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_dag({"a": ["b"], "b": ["a"]})
    with pytest.raises(PlanValidationError, match="unknown"):
        validate_dag({"a": ["zzz"]})
    with pytest.raises(PlanValidationError, match="itself"):
        validate_dag({"a": ["a"]})
    with pytest.raises(PlanValidationError, match="maximum"):
        validate_dag({"a": [], "b": []}, max_nodes=1)


def test_topological_waves_and_critical_path():
    g = {"a": [], "b": [], "c": ["a", "b"], "d": ["c"], "e": ["a"]}
    assert topological_waves(g) == [["a", "b"], ["c", "e"], ["d"]]
    length, path = critical_path(g, {"a": 1, "b": 5, "c": 1, "d": 1, "e": 10})
    assert length == 11 and path == ["a", "e"]
    assert total_work(g) == 5
    assert transitive_dependents(g, ["a"]) == {"c", "d", "e"}
    assert "a --> c" in to_mermaid(g)


# -------------------------------------------------------------- estimator


def test_amdahl_formulas():
    assert amdahl_speedup(0.5, 1e9) == pytest.approx(2.0, rel=1e-6)
    assert amdahl_bound(0.3) == pytest.approx(1 / 0.3)
    assert math.isinf(amdahl_bound(0.0))
    assert waves_for(10, 4) == 3 and waves_for(0, 4) == 0


def test_estimate_fan_out_matches_hand_computation():
    est = estimate_fan_out(8, serial_calls=2, concurrency=4)
    # work = 2 + 8 = 10 ; critical = 2 + ceil(8/4) = 4 ; speedup 2.5 ; serial fraction 0.2 ; bound 5
    assert est.total_work == 10 and est.critical_path == 4
    assert est.speedup == pytest.approx(2.5) and est.serial_fraction == pytest.approx(0.2)
    assert est.amdahl_bound == pytest.approx(5.0) and est.waves == 2 and not est.near_bound
    # 5 serial + 8 parallel in one wave: speedup 13/6 = 2.17 against a bound of 2.6 -> widening is pointless
    assert estimate_fan_out(8, serial_calls=5, concurrency=8).near_bound


def test_estimate_graph_respects_concurrency_ceiling():
    g = {f"n{i}": [] for i in range(6)}
    g["sink"] = list(g)
    est = estimate_graph(g, concurrency=3)
    assert est.total_work == 7 and est.critical_path == 3 and est.speedup == pytest.approx(7 / 3, abs=1e-3)


def test_tradeoff_table_and_recommended_concurrency():
    rows = tradeoff_table(10, task_latency_s=2.0, task_cost_usd=0.01, serial_latency_s=1.0, candidates=(1, 5, 10, 20))
    assert [r.waves for r in rows] == [10, 2, 1, 1]
    assert rows[0].latency_s == 21.0 and rows[2].latency_s == 3.0
    assert all(r.cost_usd == pytest.approx(0.1) for r in rows)  # cost is flat in width
    assert recommended_concurrency(requests_per_minute=60, task_latency_s=5.0, hard_cap=8) == 5
    assert recommended_concurrency(input_tokens_per_minute=60_000, est_input_tokens_per_request=1000, task_latency_s=2.0, hard_cap=8) == 2
    assert recommended_concurrency(hard_cap=8, n_tasks=3) == 3


# ------------------------------------------------------------ aggregation


@pytest.mark.parametrize(
    "raw, canon",
    [
        ("The answer is 1,250 kN.", "1250 kN".lower()),
        ("1250kN", "1250 kN".lower()),
        ("Yes.", "yes"),
        ("FALSE", "no"),
        ("Final answer: 3.14159265", "3.14159"),
        ("  Steel ", "steel"),
    ],
)
def test_canonicalize_answer(raw: str, canon: str):
    assert canonicalize_answer(raw) == canon


def test_majority_vote_early_stop_rule_and_tie_break():
    v = majority_vote(["42 mm", "42mm", "41 mm", "The answer is 42 mm", "40"], margin=2, leader_share=0.6)
    assert v.winner == "42 mm" and v.counts["42 mm"] == 3 and v.margin == 2 and v.decisive and v.share == 0.6
    v2 = majority_vote(["a", "b", "c", "d"])
    assert not v2.decisive and v2.winner == "a" and v2.entropy == pytest.approx(1.0)
    assert majority_vote([]).n == 0 and not majority_vote([]).decisive
    v3 = majority_vote(["x", "x", "x"])
    assert v3.unanimous and v3.entropy == 0.0 and confidence_from_vote(v3) == 0.95


def test_entropy_and_confidence_blends():
    assert vote_entropy({"a": 2, "b": 2}) == pytest.approx(1.0)
    assert vote_entropy({"a": 5}) == 0.0
    v = majority_vote(["a", "a", "a", "b", "c"])
    assert 0.45 < confidence_from_vote(v) < 0.6
    assert weighted_confidence([(0.9, 1), (0.5, 1)]) == pytest.approx(0.7)
    assert weighted_confidence([]) == 0.0
    assert combine_confidence(1.0, 0.9) == pytest.approx(0.96)
    assert combine_confidence(0.5, 0.9, structural_weight=0.5) == pytest.approx(0.7)


def test_concat_and_missing_notice():
    assert concat_in_order([(2, "c"), (0, " a "), (1, "")]) == "a\n\nc"
    assert missing_branch_notice([]) == ""
    assert "p2, p4" in missing_branch_notice(["p2", "p4"])
