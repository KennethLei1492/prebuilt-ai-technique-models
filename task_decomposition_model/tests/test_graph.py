"""Pure DAG logic: validation, waves, critical path, placeholders."""
from __future__ import annotations

import pytest

from pipeline_core import PlanValidationError

from task_decomposition import SubTask, TaskGraph, critical_path_length, find_cycle, find_placeholders, substitute_placeholders, topological_waves


def _graph() -> TaskGraph:
    return TaskGraph(
        root_task="root",
        tasks=[
            SubTask(id="t1", description="look up A"),
            SubTask(id="t2", description="look up B"),
            SubTask(id="t3", description="combine", depends_on=["t1", "t2"], inputs={"a": "$t1.output", "b": "${t2}"}),
            SubTask(id="t4", description="report", depends_on=["t3"]),
        ],
    )


def test_topological_waves_and_critical_path():
    g = _graph()
    assert g.topological_waves() == [["t1", "t2"], ["t3"], ["t4"]]
    assert g.depth() == 3 and g.breadth() == 2 and g.critical_path_length() == 3
    assert g.parallel_speedup_estimate() == pytest.approx(4 / 3)
    assert critical_path_length({"a": [], "b": ["a"], "c": ["a"]}, {"a": 2.0, "b": 1.0, "c": 5.0}) == 7.0


def test_cycle_and_missing_dependency_are_rejected():
    assert find_cycle({"a": ["b"], "b": ["a"]}) in (["a", "b", "a"], ["b", "a", "b"])
    with pytest.raises(PlanValidationError):
        topological_waves({"a": ["b"], "b": ["a"]})
    with pytest.raises(PlanValidationError):
        topological_waves({"a": ["zzz"]})
    bad = TaskGraph(root_task="r", tasks=[SubTask(id="a", description="x", depends_on=["b"]), SubTask(id="b", description="y", depends_on=["a"])])
    issues = bad.validate_structure()
    assert any("cycle" in i for i in issues)
    with pytest.raises(PlanValidationError):
        bad.ensure_valid()


def test_validate_structure_reports_bounds_and_duplicates():
    g = _graph()
    assert g.validate_structure() == []
    assert any("max_depth" in i for i in g.validate_structure(max_depth=2))
    assert any("breadth" in i for i in g.validate_structure(max_breadth=1))
    assert any("node cap" in i for i in g.validate_structure(max_nodes=3))
    dup = g.with_tasks(g.tasks + [SubTask(id="t5", description="look up A")])
    assert any("duplicates" in i for i in dup.validate_structure())
    dupid = g.with_tasks(g.tasks + [SubTask(id="t1", description="again")])
    assert any("duplicate sub-task ids" in i for i in dupid.validate_structure())


def test_placeholders_found_substituted_and_inferred():
    assert find_placeholders({"a": "$t1.output and ${t2} plus $t3", "n": ["$t4"]}) == ["t1", "t2", "t3", "t4"]
    out = substitute_placeholders({"a": "cost $t1.output, keep $5 literal", "b": ["${t2}"]}, {"t1": "X", "t2": "Y"})
    assert out == {"a": "cost X, keep $5 literal", "b": ["Y"]}
    g = TaskGraph(root_task="r", tasks=[SubTask(id="t1", description="a"), SubTask(id="t2", description="use $t1.output", depends_on=["t2"])])
    g.infer_dependencies()
    assert g.get("t2").depends_on == ["t1"]  # self-edge dropped, placeholder edge added
    missing = TaskGraph(root_task="r", tasks=[SubTask(id="t1", description="a"), SubTask(id="t2", description="use $t1.output")])
    assert any("references $t1" in i for i in missing.validate_structure())


def test_renderings():
    g = _graph()
    mermaid = g.to_mermaid()
    assert mermaid.startswith("graph TD") and "t1 --> t3" in mermaid and "t3 --> t4" in mermaid
    ascii_ = g.to_ascii()
    assert "wave 1:" in ascii_ and "[t3]" in ascii_ and "<- t1, t2" in ascii_
    assert TaskGraph(root_task="r").to_ascii() == "(empty graph)"
    assert g.descendants("t1") == {"t3", "t4"} and g.dependents("t3") == ["t4"]
