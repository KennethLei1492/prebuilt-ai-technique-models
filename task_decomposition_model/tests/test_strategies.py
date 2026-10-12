"""Strategy modules: heuristic graphs per strategy, ToT voting, LLM decomposition with repair."""
from __future__ import annotations

from pipeline_core import Constraints, MockLLM, RunContext, TaskRequest, Tracer, get_profile

from task_decomposition import (
    DAGPlanner,
    DecompositionPlan,
    DirectStrategy,
    HeuristicPlanner,
    LeastToMost,
    PlanAndSolve,
    RecursiveDecomposer,
    TaskDecompositionConfig,
    TreeOfThoughtsPlanner,
    aggregate_votes,
    chainify,
    get_strategy,
    heuristic_graph,
    heuristic_segments,
    structural_score,
)

from td_fixtures import BRIDGE_TASK, CHAIN_TASK, PUMP_TASK, TRIVIAL_TASK


def _plan(task: str, domain: str = "general") -> DecompositionPlan:
    return HeuristicPlanner().plan(task, profile=get_profile(domain))


def test_segments_fold_scope_and_preamble():
    pre, segs = heuristic_segments(PUMP_TASK)
    assert pre == ""
    assert len(segs) == 3
    assert segs[0].lower().startswith("for each of the three pump test logs: extract")
    assert segs[-1].lower().startswith("then summarise")
    pre2, segs2 = heuristic_segments(BRIDGE_TASK)
    assert pre2 == "Review the bridge deck design" and len(segs2) == 4
    _, enum = heuristic_segments("Do the following:\n1. measure the flow\n2. compute the head\n3. select the pump")
    assert enum == ["Measure the flow", "Compute the head", "Select the pump"]
    _, capped = heuristic_segments(BRIDGE_TASK, max_segments=2)
    assert len(capped) == 2


def test_dag_heuristic_dependencies_and_domain_adaptation():
    general = heuristic_graph(PUMP_TASK, plan=_plan(PUMP_TASK), profile=get_profile("general"), config=TaskDecompositionConfig())
    assert general.validate_structure() == []
    ids = general.ids()
    assert "verify" not in ids
    last = general.get(ids[-1])
    assert last.depends_on and last.tool_hint == "aggregate"  # "then summarise" depends on everything before
    assert general.depth() == 2 and general.breadth() >= 2
    civil = heuristic_graph(PUMP_TASK, plan=_plan(PUMP_TASK, "civil_structural_engineering"), profile=get_profile("civil_structural_engineering"), config=TaskDecompositionConfig())
    assert "verify" in civil.ids()
    v = civil.get("verify")
    assert v.tool_hint == "verify" and set(v.depends_on) == set(civil.ids()) - {"verify"}
    assert any("code" in c.lower() or "load" in c.lower() for c in v.verification_criteria)
    assert all("unit" in " ".join(t.verification_criteria).lower() for t in civil.tasks)


def test_padding_uses_domain_hints_when_task_is_short():
    cfg = TaskDecompositionConfig(min_subtasks=4)
    plan = _plan("Size the shaft and check the bearings", "mechanical_engineering")
    plan = plan.model_copy(update={"strategy": "dag", "max_breadth": 6})
    g = heuristic_graph("Size the shaft and check the bearings", plan=plan, profile=get_profile("mechanical_engineering"), config=cfg)
    descs = " | ".join(t.description for t in g.tasks)
    assert "Extract the relevant variables" in descs and "Identify loads and boundary conditions" in descs
    assert g.validate_structure() == []


def test_padded_graph_last_segment_still_aggregates():
    """Regression: with the PS+ padding node inserted (fewer than min_subtasks segments) the summarising last
    segment must still depend on everything before it; the old rule compared the node index to the segment count."""
    task = "Compute the bearing stress and summarise the margin"
    plan = _plan(task).model_copy(update={"strategy": "dag", "max_breadth": 4})
    g = heuristic_graph(task, plan=plan, profile=get_profile("general"), config=TaskDecompositionConfig())
    assert g.validate_structure() == []
    assert g.tasks[0].description.startswith("Extract the relevant variables")  # padding node t1
    compute = g.get("t2")
    summary = g.get("t3")
    assert compute.description.startswith("Compute") and compute.tool_hint == "compute" and compute.depends_on == []
    assert summary.description.startswith("Summarise") and summary.depends_on == ["t1", "t2"] and summary.tool_hint == "aggregate"
    assert g.topological_waves() == [["t1", "t2"], ["t3"]]
    # without padding (three segments) the same rule holds, as before
    g3 = heuristic_graph(PUMP_TASK, plan=_plan(PUMP_TASK), profile=get_profile("general"), config=TaskDecompositionConfig())
    assert g3.get(g3.ids()[-1]).depends_on == g3.ids()[:-1]


def test_least_to_most_keeps_an_explicit_enumerated_order():
    """Regression: a numbered list is an explicit order; the easiest-first proxy must not re-sort it."""
    task = "1. Measure the flow\n2. Compute the head from the flow\n3. Select the pump"
    plan = HeuristicPlanner().plan(task, profile=get_profile("general"), strategy_override="least_to_most")
    g = LeastToMost().heuristic(task, plan=plan, profile=get_profile("general"))
    assert [t.description for t in g.tasks] == ["Measure the flow", "Compute the head from the flow", "Select the pump"]
    assert g.tasks[-1].depends_on == ["t1", "t2"] and g.depth() == 3
    # no order stated anywhere: the proxy still sorts easiest (shortest, low complexity) first
    loose = "Write a detailed commissioning report for the whole installation and size the pump"
    plan2 = HeuristicPlanner().plan(loose, profile=get_profile("general"), strategy_override="least_to_most")
    g2 = LeastToMost().heuristic(loose, plan=plan2, profile=get_profile("general"))
    descs = [t.description for t in g2.tasks]
    assert descs.index("Size the pump") < descs.index("Write a detailed commissioning report for the whole installation")


def test_least_to_most_is_a_chain_with_carry_forward():
    plan = _plan(CHAIN_TASK)
    g = LeastToMost().heuristic(CHAIN_TASK, plan=plan, profile=get_profile("general"))
    n = len(g.tasks)
    assert n >= 3 and g.depth() == n and g.breadth() == 1
    assert g.tasks[-1].depends_on == [t.id for t in g.tasks[:-1]]
    short = chainify(heuristic_graph(BRIDGE_TASK, plan=_plan(BRIDGE_TASK), profile=get_profile("general"), config=TaskDecompositionConfig()))
    assert short.depth() == len(short.tasks)


def test_direct_plan_and_solve_recursive_single_node():
    plan = _plan(TRIVIAL_TASK)
    d = DirectStrategy().heuristic(TRIVIAL_TASK, plan=plan, profile=get_profile("general"))
    assert len(d.tasks) == 1 and d.tasks[0].description == TRIVIAL_TASK and d.strategy == "direct"
    ps = PlanAndSolve().heuristic(TRIVIAL_TASK, plan=plan, profile=get_profile("general"))
    assert ps.tasks[0].tool_hint == "plan_and_solve" and ps.strategy == "plan_and_solve"
    r = RecursiveDecomposer().heuristic("Debug the flaky job", plan=plan, profile=get_profile("general"))
    assert len(r.tasks) == 1 and r.metadata["recursive"] is True


async def test_recursive_expand_offline_splits_node():
    cfg = TaskDecompositionConfig()
    req = TaskRequest(task="Debug why the nightly ETL job fails intermittently and propose a fix", domain="data_engineering")
    ctx = RunContext(req, Tracer(), MockLLM(), "task_decomposition")
    plan = HeuristicPlanner(cfg).plan(req.task, profile=ctx.profile)
    strat = RecursiveDecomposer(cfg)
    root = strat.heuristic(req.task, plan=plan, profile=ctx.profile)
    sub = await strat.expand(root.tasks[0], reason="solver reported failure", depth=0, request=req, plan=plan, ctx=ctx)
    assert len(sub.tasks) >= 2 and sub.validate_structure() == [] and sub.metadata["depth"] == 1
    tiny = await strat.expand(root.tasks[0].model_copy(update={"description": "Fix the broken nightly job quickly please"}), reason="x", depth=1, request=req, plan=plan, ctx=ctx)
    assert len(tiny.tasks) >= 2 and tiny.validate_structure() == []


def test_tree_of_thoughts_offline_selects_best_candidate_and_votes_aggregate():
    plan = _plan("Compare three alternative gearbox layouts and select the best trade-off between mass, efficiency and cost")
    tot = TreeOfThoughtsPlanner()
    g = tot.heuristic(plan.rationale and "Compare three alternative gearbox layouts and select the best trade-off between mass, efficiency and cost", plan=plan, profile=get_profile("general"))
    assert g.validate_structure() == [] and "candidate_scores" in g.metadata and g.metadata["selected_index"] == g.metadata["candidate_scores"].index(max(g.metadata["candidate_scores"]))
    assert 0.0 <= structural_score(g, g.root_task) <= 1.0
    assert aggregate_votes([[0.9, 0.2, 0.5], [0.7, 0.0, 0.6]], 3) == [0.8, 0.0, 0.55]
    assert aggregate_votes([], 2) == [0.0, 0.0]


async def test_llm_decompose_repairs_invalid_graph_then_falls_back():
    bad = {"tasks": [{"id": "t1", "description": "a", "depends_on": ["t2"]}, {"id": "t2", "description": "b", "depends_on": ["t1"]}], "notes": ""}
    good = {"tasks": [{"id": "t1", "description": "look up the datasheet duty point"}, {"id": "t2", "description": "compare the measured duty point", "depends_on": ["t1"], "inputs": [{"name": "ds", "value": "$t1.output"}]}], "notes": "ok"}
    answers = [bad, good]
    llm = MockLLM(responder=lambda req: answers.pop(0))
    cfg = TaskDecompositionConfig(force_llm_path=True)
    req = TaskRequest(task=PUMP_TASK)
    ctx = RunContext(req, Tracer(), llm, "task_decomposition")
    plan = HeuristicPlanner(cfg).plan(PUMP_TASK, profile=ctx.profile)
    g = await DAGPlanner(cfg).build(PUMP_TASK, request=req, plan=plan, ctx=ctx)
    assert g.metadata["planner"] == "llm" and g.ids() == ["t1", "t2"] and ctx.usage.llm_calls == 2
    assert any("repairing" in w for w in ctx.warnings)
    assert "<previous_attempt_issues>" in llm.calls[1].last_user_text()
    # exhausted repairs -> heuristic fallback
    always_bad = MockLLM(responder=lambda req: bad)
    ctx2 = RunContext(req, Tracer(), always_bad, "task_decomposition")
    g2 = await DAGPlanner(cfg).build(PUMP_TASK, request=req, plan=plan, ctx=ctx2)
    assert g2.metadata["planner"] == "heuristic" and ctx2.usage.llm_calls == 2


async def test_deterministic_constraint_skips_llm_decomposition():
    llm = MockLLM()
    req = TaskRequest(task=PUMP_TASK, constraints=Constraints(deterministic=True))
    ctx = RunContext(req, Tracer(), llm, "task_decomposition")
    plan = HeuristicPlanner().plan(PUMP_TASK, profile=ctx.profile, constraints=req.constraints)
    g = await get_strategy("dag", TaskDecompositionConfig(force_llm_path=True)).build(PUMP_TASK, request=req, plan=plan, ctx=ctx)
    assert g.metadata["planner"] == "heuristic" and llm.calls == []
    assert isinstance(get_strategy("unknown"), DAGPlanner)


_TOT_CANDIDATES = {
    "candidates": [
        {"tasks": [{"id": "t1", "description": "Compare the three gearbox layouts on mass, efficiency and cost"}, {"id": "t2", "description": "Select the best trade-off", "depends_on": ["t1"], "inputs": [{"name": "c", "value": "$t1.output"}]}], "notes": ""},
        {"tasks": [{"id": "t1", "description": "List the three alternative gearbox layouts for the envelope"}, {"id": "t2", "description": "Score each layout on mass, efficiency and cost"}, {"id": "t3", "description": "Select the best trade-off", "depends_on": ["t1", "t2"], "inputs": [{"name": "a", "value": "$t1.output"}, {"name": "b", "value": "$t2.output"}]}], "notes": ""},
    ]
}


def _tot_responder(score_a: float, score_b: float):
    def responder(req):  # noqa: ANN001
        if "task_decomposition.candidates" in req.tags or "task_decomposition.refine" in req.tags:
            return _TOT_CANDIDATES
        if "task_decomposition.evaluate" in req.tags:
            return {"scores": [{"index": 0, "score": score_a, "verdict": "likely", "issue": ""}, {"index": 1, "score": score_b, "verdict": "likely", "issue": ""}]}
        raise AssertionError(f"unexpected call {req.tags}")

    return responder


async def test_tree_of_thoughts_llm_search_depth_refinement_and_agreement_flag():
    from conftest import EXPLORE_TASK

    # depth 1: propose + 3 votes; candidate 1 wins; agreement is fine
    llm = MockLLM(responder=_tot_responder(0.4, 0.8))
    cfg = TaskDecompositionConfig(force_llm_path=True, tot_depth=1)
    req = TaskRequest(task=EXPLORE_TASK, domain="mechanical_engineering")
    ctx = RunContext(req, Tracer(), llm, "task_decomposition")
    plan = HeuristicPlanner(cfg).plan(EXPLORE_TASK, profile=ctx.profile)
    assert plan.strategy == "tree_of_thoughts"
    g = await TreeOfThoughtsPlanner(cfg).build(EXPLORE_TASK, request=req, plan=plan, ctx=ctx)
    assert g.ids() == ["t1", "t2", "t3"] and g.metadata["selected_index"] == 1 and g.metadata["search_depth"] == 1
    assert "low_agreement" not in g.metadata and ctx.usage.llm_calls == 1 + cfg.tot_votes
    assert sum(1 for c in llm.calls if "refine" in c.tags) == 0

    # depth 2: the two best parents are refined, parents + children are voted on again
    llm2 = MockLLM(responder=_tot_responder(0.3, 0.35))
    cfg2 = TaskDecompositionConfig(force_llm_path=True, tot_depth=2, tot_votes=2)
    ctx2 = RunContext(req, Tracer(), llm2, "task_decomposition")
    g2 = await TreeOfThoughtsPlanner(cfg2).build(EXPLORE_TASK, request=req, plan=plan, ctx=ctx2)
    refine_calls = [c for c in llm2.calls if "refine" in c.tags]
    assert len(refine_calls) == 2 and all("<parent_candidate>" in c.last_user_text() for c in refine_calls)
    assert g2.metadata["search_depth"] == 2 and g2.validate_structure() == []
    assert g2.metadata["low_agreement"] is True and any("agreement_threshold" in w for w in ctx2.warnings)
    # propose(1) + votes(2) + refine(2) + votes(2)
    assert ctx2.usage.llm_calls == 1 + 2 + 2 + 2

    # a tight budget degrades the search instead of failing: no refinement level is started
    llm3 = MockLLM(responder=_tot_responder(0.4, 0.8))
    req3 = TaskRequest(task=EXPLORE_TASK, domain="mechanical_engineering", constraints=Constraints(max_llm_calls=4))
    ctx3 = RunContext(req3, Tracer(), llm3, "task_decomposition")
    g3 = await TreeOfThoughtsPlanner(cfg2).build(EXPLORE_TASK, request=req3, plan=plan, ctx=ctx3)
    assert g3.validate_structure() == [] and ctx3.usage.llm_calls <= 3 and g3.metadata["search_depth"] == 1
