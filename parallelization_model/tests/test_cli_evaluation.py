"""CLI and evaluation harness, all offline."""
from __future__ import annotations

import json

import pytest

from pipeline_core import MockLLM, TaskRequest

from parallelization import ParallelizationModel
from parallelization.cli import main, parse_input
from parallelization.evaluation import aggregate, evaluate, metrics_from_result, percentile, synthetic_tasks
from parallelization.executor import ManualClock, ParallelExecutor


def test_parse_input_variants(tmp_path):
    assert parse_input("k=v") == ("k", "v")
    assert parse_input('items=["a","b"]') == ("items", ["a", "b"])
    assert parse_input("items=a, b ,c") == ("items", ["a", "b", "c"])
    assert parse_input("n=3") == ("n", 3)
    f = tmp_path / "log.txt"
    f.write_text("hello", encoding="utf-8")
    assert parse_input(f"text=@{f}") == ("text", "hello")


def test_cli_text_output(capsys, monkeypatch):
    monkeypatch.delenv("PIPELINE_OFFLINE", raising=False)
    code = main(["--offline", "Compare steel, aluminium and titanium for a bracket", "--domain", "mechanical_engineering"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[parallelization] status=success" in out and "pattern=sectioning" in out and "Output:" in out


def test_cli_json_output_with_items(capsys, monkeypatch):
    monkeypatch.delenv("PIPELINE_OFFLINE", raising=False)
    code = main(["--offline", "--json", "Classify each ticket", "--input", 'items=["Crash on save","Add dark mode"]', "--max-calls", "5"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["component"] == "parallelization"
    assert data["metadata"]["plan"]["pattern"] == "sectioning" and len(data["metadata"]["plan"]["partitions"]) == 2
    assert data["usage"]["llm_calls"] <= 5


def test_cli_plan_only_and_evaluate(capsys):
    assert main(["--offline", "--plan-only", "What are the risks of the redesign?"]) == 0
    assert "pattern=skeleton_of_thought" in capsys.readouterr().out
    assert main(["--offline", "--evaluate", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runs"] == 9 and report["by_status"] == {"success": 9}


def test_cli_requires_task(capsys):
    assert main(["--offline"]) == 2


def test_percentile_nearest_rank():
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5
    assert percentile([10, 1], 0) == 1


def test_metrics_from_result_and_aggregate():
    model = ParallelizationModel(MockLLM(), executor=ParallelExecutor.from_config(ParallelizationModel().config, clock=ManualClock()))
    res = model.run_sync(TaskRequest(task="Compare steel, aluminium and titanium for a bracket."))
    m = metrics_from_result(res)
    assert m.pattern == "sectioning" and m.llm_calls == 4 and m.branches == 4 and m.failed_branches == 0
    assert m.expected_speedup == pytest.approx(2.0) and m.amdahl_bound == pytest.approx(4.0)
    assert m.token_multiplier == 4.0 and m.retry_rate == 0.0
    rep = aggregate([m, m], elapsed_s=0.5)
    assert rep.runs == 2 and rep.by_pattern == {"sectioning": 2} and rep.mean_llm_calls == 4.0
    assert "runs=2" in rep.table()


def test_evaluate_harness_covers_every_pattern_family():
    model = ParallelizationModel(MockLLM())
    report = evaluate(model, synthetic_tasks("mechanical_engineering"))
    assert report.runs == 9
    assert set(report.by_pattern) == {"sectioning", "skeleton_of_thought", "voting", "branch_solve_merge", "map_reduce", "sequential", "ensemble", "speculative"}
    assert report.by_status == {"success": 9}
    assert report.mean_expected_speedup > 1.5
