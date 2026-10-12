"""Evaluation metrics/harness and the CLI."""
from __future__ import annotations

import json

import pytest

from pipeline_core import MockLLM, Status

from task_decomposition import (
    EvalCase,
    SubTask,
    SubTaskResult,
    TaskDecompositionModel,
    TaskGraph,
    compositionality_gap,
    coverage,
    dependency_correctness,
    evaluate,
    evaluate_plan,
    failure_classes,
    sample_cases,
)
from task_decomposition.cli import main, parse_inputs
from task_decomposition.evaluation import keyword_judge, main as eval_main

from td_fixtures import PUMP_TASK


def test_plan_metrics_on_hand_built_graph():
    g = TaskGraph(
        root_task="extract the duty point and summarise the fleet status",
        tasks=[
            SubTask(id="t1", description="extract the duty point", verification_criteria=["units"]),
            SubTask(id="t2", description="summarise the fleet status using $t1.output", depends_on=["t1"]),
        ],
    )
    rep = evaluate_plan(g)
    assert rep.valid and rep.n_tasks == 2 and rep.depth == 2 and rep.critical_path == 2 and rep.parallel_speedup == 1.0
    assert rep.coverage == 1.0 and rep.dependency_correctness == 1.0 and rep.with_criteria == 0.5 and rep.estimated_llm_calls == 3
    assert 0 < rep.score() <= 1
    loose = TaskGraph(root_task="r", tasks=[SubTask(id="t1", description="a"), SubTask(id="t2", description="use $t1.output")])
    assert dependency_correctness(loose) == 0.0 and not evaluate_plan(loose).valid
    cyc = TaskGraph(root_task="r", tasks=[SubTask(id="a", description="x", depends_on=["b"]), SubTask(id="b", description="y", depends_on=["a"])])
    assert dependency_correctness(cyc) == 0.0 and evaluate_plan(cyc).depth == 0
    assert coverage(TaskGraph(root_task="the of and", tasks=[])) == 1.0


def test_gap_failure_classes_and_keyword_judge():
    assert compositionality_gap([(True, True), (True, False), (False, False)]) == 0.5
    assert compositionality_gap([(False, False)]) == 0.0
    results = {
        "a": SubTaskResult(task_id="a", status=Status.FAILED, error="budget: x"),
        "b": SubTaskResult(task_id="b", status=Status.SKIPPED, error="dependency failed: a"),
        "c": SubTaskResult(task_id="c", status=Status.FAILED, error="verifier: bad units"),
        "d": SubTaskResult(task_id="d", status=Status.FAILED, error="StructuredOutputError: not valid JSON"),
        "e": SubTaskResult(task_id="e", status=Status.FAILED, error="RuntimeError: boom"),
        "f": SubTaskResult(task_id="f", output_text="ok"),
    }
    assert failure_classes(results) == {"budget_or_deadline": 1, "skipped_dependency": 1, "verification_failed": 1, "invalid_format": 1, "solver_failed": 1}
    assert keyword_judge("Duty Point is 120", ["duty point"]) and not keyword_judge("x", ["y"]) and keyword_judge("x", [])


async def test_evaluation_harness_offline():
    model = TaskDecompositionModel(MockLLM(), retry_base_delay_s=0.0)
    cases = sample_cases()
    assert len(cases) >= 6
    summary = await evaluate(model, cases[:3])
    assert summary.cases == 3 and 0 <= summary.success_rate <= 1 and summary.plan_validity_rate == 1.0
    assert summary.strategy_accuracy == 1.0
    assert summary.mean_llm_calls > 0 and summary.strategies and len(summary.outcomes) == 3
    assert "cases=3" in summary.table()
    custom = await evaluate(model, [EvalCase(task=PUMP_TASK, expected_keywords=["zzz-not-there"])])
    assert custom.keyword_accuracy == 0.0


def test_evaluation_main_prints_table(capsys):
    assert eval_main([]) == 0
    out = capsys.readouterr().out
    assert "cases=" in out and "strategies=" in out
    assert eval_main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cases"] >= 6
    from task_decomposition.eval_cli import main as cli_main

    assert cli_main([]) == 0 and "cases=" in capsys.readouterr().out


def test_eval_cli_module_runs_without_runtime_warning():
    """Regression: the documented harness command must not emit Python's double-import RuntimeWarning."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "common"), str(root / "task_decomposition_model")])}
    proc = subprocess.run([sys.executable, "-m", "task_decomposition.eval_cli"], cwd=str(root), env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "cases=" in proc.stdout
    assert "RuntimeWarning" not in proc.stderr and "found in sys.modules" not in proc.stderr


def test_cli_offline_text_and_json(capsys):
    code = main([PUMP_TASK, "--offline", "--domain", "general"])
    out = capsys.readouterr().out
    assert code == 0 and "[task_decomposition] status=success" in out and "PLAN" in out and "wave 1:" in out and "OUTPUT" in out
    code = main([PUMP_TASK, "--offline", "--json", "--input", "documents=Log A: 118 m3/h", "--input", "k=3", "--max-calls", "20"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["status"] == "success" and data["plan"]["strategy"] == "dag" and data["output"]["graph"]["tasks"]
    assert data["usage"]["llm_calls"] <= 20


def test_cli_plan_only_and_strategy_override(capsys):
    assert main([PUMP_TASK, "--offline", "--plan-only", "--strategy", "least_to_most"]) == 0
    out = capsys.readouterr().out
    assert "strategy=least_to_most" in out and "graph TD" in out
    assert main([PUMP_TASK, "--offline", "--plan-only", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["metrics"]["valid"] is True


def test_cli_needs_review_exit_code_and_trace(capsys):
    code = main([PUMP_TASK, "--offline", "--domain", "civil_structural_engineering", "--trace"])
    out = capsys.readouterr().out
    assert code == 0 and "status=needs_review" in out and "task_decomposition.run" in out


def test_parse_inputs():
    assert parse_inputs(["a=1", "b=text", 'c={"x": 2}', "documents=d1", 'documents=["d2","d3"]']) == {"a": 1, "b": "text", "c": {"x": 2}, "documents": ["d1", "d2", "d3"]}
    with pytest.raises(SystemExit):
        parse_inputs(["novalue"])


async def test_llm_coverage_judge_and_fallbacks():
    from pipeline_core import LLMError, get_profile

    from task_decomposition.evaluation import llm_coverage

    g = TaskGraph(
        root_task="extract the duty point and summarise the fleet status",
        tasks=[SubTask(id="t1", description="extract the duty point"), SubTask(id="t2", description="summarise the fleet status", depends_on=["t1"])],
    )
    judged = MockLLM(
        responder=lambda req: {
            "requirements": [
                {"requirement": "extract the duty point", "covered_by": ["t1"]},
                {"requirement": "summarise the fleet status", "covered_by": ["t2"]},
                {"requirement": "flag deviations", "covered_by": []},
                {"requirement": "phantom", "covered_by": ["t9"]},  # unknown id does not count
            ]
        }
    )
    assert await llm_coverage(g, llm=judged, profile=get_profile("mechanical_engineering")) == 0.5
    assert judged.calls and "task_decomposition.coverage_judge" in judged.calls[0].tags and "wave 1:" in judged.calls[0].last_user_text()
    with pytest.raises(ValueError):
        await llm_coverage(g)
    # offline MockLLM returns no requirements -> keyword judge; errors -> keyword judge
    assert await llm_coverage(g, llm=MockLLM()) == coverage(g)

    def boom(req):  # noqa: ANN001
        raise LLMError("down")

    assert await llm_coverage(g, llm=MockLLM(responder=boom)) == coverage(g)
    summary = await evaluate(TaskDecompositionModel(MockLLM(), retry_base_delay_s=0.0), sample_cases()[:2], judge="llm")
    assert summary.cases == 2 and 0.0 <= summary.mean_coverage <= 1.0
    assert eval_main(["--judge", "llm"]) == 0


async def test_llm_coverage_judge_is_accounted_and_can_run_through_a_run_context():
    """Regression: the judge call was invisible to cost accounting and to the tracer."""
    from pipeline_core import RunContext, TaskRequest, Tracer

    from task_decomposition.evaluation import judge_coverage, llm_coverage

    g = TaskGraph(root_task="extract the duty point and summarise", tasks=[SubTask(id="t1", description="extract the duty point"), SubTask(id="t2", description="summarise", depends_on=["t1"])])
    judged = MockLLM(responder=lambda req: {"requirements": [{"requirement": "extract", "covered_by": ["t1"]}, {"requirement": "summarise", "covered_by": ["t2"]}]}, simulate_cost_as="claude-opus-5-5")
    value, usage = await judge_coverage(g, llm=judged)
    assert value == 1.0 and usage.llm_calls == 1 and usage.cost_usd > 0
    # the harness adds the judge's usage to the case and the total
    keyword = await evaluate(TaskDecompositionModel(MockLLM(), retry_base_delay_s=0.0), sample_cases()[:1], judge="keyword")
    with_judge = await evaluate(TaskDecompositionModel(MockLLM(), retry_base_delay_s=0.0), sample_cases()[:1], judge="llm")
    assert keyword.outcomes[0].judge_calls == 0 and with_judge.outcomes[0].judge_calls == 1
    assert with_judge.outcomes[0].llm_calls == keyword.outcomes[0].llm_calls + 1 and with_judge.mean_llm_calls == keyword.mean_llm_calls + 1
    # with a RunContext the call is traced, budgeted and charged
    tracer = Tracer()
    ctx = RunContext(TaskRequest(task=g.root_task), tracer, judged, "task_decomposition")
    assert await llm_coverage(g, ctx=ctx) == 1.0
    assert ctx.usage.llm_calls == 1 and "coverage_judge.llm" in tracer.tree()
    from pipeline_core import Constraints

    capped = RunContext(TaskRequest(task=g.root_task, constraints=Constraints(max_llm_calls=0)), Tracer(), judged, "task_decomposition")
    assert await llm_coverage(g, ctx=capped) == coverage(g) and capped.usage.llm_calls == 0  # budget stop -> keyword judge
