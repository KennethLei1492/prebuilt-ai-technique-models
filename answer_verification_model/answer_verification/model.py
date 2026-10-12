"""The ``answer_verification`` pipeline component.

:class:`AnswerVerificationModel` decides whether a candidate answer is correct,
grounded and complete, estimates a calibrated confidence and - only when an
external signal (evidence, executed tests, rules, strong multi-sample
agreement) backs it - proposes a minimal correction.

Life cycle
----------
``plan()`` reads the task, the candidate answer, the available oracles
(evidence, tests, reference, code execution permission) and the domain
profile, then produces a :class:`~answer_verification.plan.VerificationPlan`
through one structured LLM call (online) or the
:class:`~answer_verification.plan.HeuristicPlanner` (offline / deterministic).
The domain :class:`~answer_verification.policy.VerificationPolicy` is applied
on top of either plan and the plan is fitted to the call budget.

``execute()`` runs the planned verifiers in order through a shared
:class:`pipeline_core.RunContext`, optionally elicits a verbalised confidence,
fuses every verdict with precedence-aware weights, applies the acceptance
gates (thresholds, mandatory verifiers, contradictions, safety-critical
evidence requirement) and returns a :class:`VerificationReport`.

Direct API
----------
``await model.verify(answer, task=..., evidence=...) -> Verdict`` implements the
:class:`pipeline_core.Verifier` protocol for other models.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    BaseLLMClient,
    BudgetExceededError,
    ComponentSpec,
    Constraints,
    Evidence,
    Executor,
    LLMError,
    PipelineComponent,
    Retriever,
    RunContext,
    Status,
    TaskRequest,
    TaskResult,
    Tracer,
    Verdict,
    register,
    run_sync,
)

from . import prompts as P
from .config import AnswerVerificationConfig
from .plan import HeuristicPlanner, LLMPlan, Thresholds, VerificationPlan, effective_accept_threshold, fit_to_budget, sanitize_verifiers
from .policy import VerificationPolicy, apply_policy, policy_for_profile
from .uncertainty import AUTHORITY, aggregate_verdicts, is_inconclusive
from .verifiers import BaseVerifier, VerifyInputs, affordable, build_request, elicit_verbalized_confidence, inconclusive, make_verifiers

_VERIFY_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:verify|check|validate|review|confirm|is this (?:answer|claim) (?:correct|right))\b[^:]{0,80}:\s*(.+)$",
    re.I | re.S,
)


class VerificationReport(BaseModel):
    """Structured output of one verification run (``TaskResult.output``)."""

    model_config = ConfigDict(extra="forbid")

    task: str
    answer: str = Field(description="Candidate answer as verified")
    answer_source: str = Field(default="inputs", description="inputs | task | draft")
    final_answer: str = Field(description="Accepted revision when one exists, otherwise the original answer")
    revised: bool = False
    revision_source: Optional[str] = None
    revision_verified: bool = Field(default=False, description="The applied revision passed re-verification by the plan's external verifiers")
    revision_rounds: int = Field(default=0, ge=0, description="Re-verification rounds spent on proposed revisions")
    revision_verdicts: list[Verdict] = Field(default_factory=list, description="Verdicts produced while re-verifying revisions")
    answer_type: str
    domain: str
    accepted: bool
    status: str
    escalated: bool = Field(default=False, description="Confidence fell below the escalation threshold: abstain / human review")
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_basis: str
    accept_threshold: float
    escalate_threshold: float
    verdicts: list[Verdict] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    plan: VerificationPlan
    policy: str = ""

    def to_verdict(self) -> Verdict:
        """Collapse the report into the single :class:`Verdict` other models consume."""
        return Verdict(
            method="answer_verification",
            passed=self.accepted,
            score=self.confidence,
            issues=list(self.review_reasons) + [i for i in self.issues if i not in self.review_reasons][:20],
            corrected_answer=self.final_answer if self.revised else None,
            evidence_ids=list(self.evidence_ids),
            details={
                "status": self.status,
                "answer_type": self.answer_type,
                "confidence_basis": self.confidence_basis,
                "verifiers": {v.method: {"passed": v.passed, "score": v.score, "inconclusive": is_inconclusive(v)} for v in self.verdicts},
                "metrics": dict(self.metrics),
                "escalated": self.escalated,
                "revision_source": self.revision_source,
                "revision_verified": self.revision_verified,
            },
        )

    def render(self) -> str:
        """Human-readable rendering used for ``TaskResult.output_text``."""
        lines = [
            f"Verification of candidate answer (type: {self.answer_type}, domain: {self.domain})",
            f"Status: {self.status.upper()} | confidence {self.confidence:.2f} (accept >= {self.accept_threshold:.2f}, escalate < {self.escalate_threshold:.2f})",
            f"Basis: {self.confidence_basis}",
            "Verifiers:",
        ]
        for v in self.verdicts:
            if is_inconclusive(v):
                lines.append(f"  - {v.method}: inconclusive ({v.details.get('reason', '')})")
            else:
                tag = "PASS" if v.passed else "FAIL"
                lines.append(f"  - {v.method}: {tag} score={v.score:.2f}" + (f" | {v.issues[0]}" if v.issues else ""))
        if self.review_reasons:
            lines.append("Needs review because:")
            lines.extend(f"  - {r}" for r in self.review_reasons)
        if self.revised:
            how = "confirmed by re-verification" if self.revision_verified else "backed by the proposing verifier"
            lines.append(f"Revision accepted from {self.revision_source} ({how}, {self.revision_rounds} round(s)); see final answer.")
        lines.append("Final answer:")
        lines.append(self.final_answer)
        return "\n".join(lines)


def coerce_evidence(items: Any) -> list[Evidence]:
    """Turn ``inputs['documents']`` (strings, dicts or Evidence) into :class:`Evidence`."""
    out: list[Evidence] = []
    if not items:
        return out
    if isinstance(items, (str, dict, Evidence)):
        items = [items]
    for i, it in enumerate(items, 1):
        if isinstance(it, Evidence):
            out.append(it)
        elif isinstance(it, str):
            out.append(Evidence(id=f"doc-{i}", content=it, retriever="provided"))
        elif isinstance(it, dict):
            content = str(it.get("content") or it.get("text") or "")
            if content:
                out.append(Evidence(id=str(it.get("id") or f"doc-{i}"), content=content, source=str(it.get("source", "")), retriever="provided", metadata=dict(it.get("metadata", {}))))
    return out


@register
class AnswerVerificationModel(PipelineComponent):
    """Verify a candidate answer with external-evidence-first verification strategies.

    Parameters
    ----------
    llm:
        Any :class:`pipeline_core.BaseLLMClient`; ``None`` uses :func:`pipeline_core.make_llm`.
    config:
        :class:`AnswerVerificationConfig` (or a dict of its fields); keyword
        overrides are applied on top.
    retriever:
        Optional :class:`pipeline_core.Retriever` used to fetch evidence for
        grounding when the request carries none.
    executor:
        Optional :class:`pipeline_core.Executor` used for sample / question /
        judge fan-out; a bounded ``asyncio.gather`` is used otherwise.
    """

    spec = ComponentSpec(
        name="answer_verification",
        version="1.0.0",
        description="Decide whether an answer is correct, grounded and complete; estimate calibrated confidence; correct it only on external signals.",
        capabilities=[
            "self_consistency",
            "universal_self_consistency",
            "chain_of_verification",
            "grounding",
            "citation_metrics",
            "rubric_judge",
            "pairwise_judge",
            "rules",
            "code_execution",
            "debate",
            "verbalized_confidence",
            "confidence_fusion",
        ],
        consumes=[
            "inputs.answer (candidate; else extracted from 'Verify ...: <answer>' or drafted)",
            "inputs.candidates (pre-sampled alternatives)",
            "inputs.documents | context (Evidence for grounding)",
            "inputs.tests (+ constraints.allow_code_execution)",
            "inputs.reference",
            "inputs.rules",
        ],
        produces=["VerificationReport (output), Verdict via verify()"],
        composes_with=["pipeline_rag", "task_decomposition", "hybrid_retrieval", "parallelization", "caching"],
    )

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        config: AnswerVerificationConfig | dict[str, Any] | None = None,
        retriever: Retriever | None = None,
        executor: Executor | None = None,
        tracer: Tracer | None = None,
        raise_on_error: bool = False,
        **overrides: Any,
    ) -> None:
        if isinstance(config, dict):
            config = AnswerVerificationConfig(**config)
        base = config or AnswerVerificationConfig()
        cfg = AnswerVerificationConfig(**{**base.model_dump(), **overrides}) if overrides else base
        super().__init__(llm, tracer=tracer, raise_on_error=raise_on_error, **cfg.model_dump())
        self.config: AnswerVerificationConfig = cfg  # type: ignore[assignment]
        self.retriever = retriever
        self.executor = executor
        self.planner = HeuristicPlanner(cfg)
        self.verifiers: dict[str, BaseVerifier] = make_verifiers(cfg, llm=self.llm, executor=executor)

    # ------------------------------------------------------------ helpers
    def describe(self) -> dict[str, Any]:
        return {**self.spec.model_dump(), "llm": self.llm.describe(), "config": self.config.model_dump()}

    def use_llm(self, ctx: RunContext, request: TaskRequest) -> bool:
        """Whether the LLM may be used for planning/verification in this run."""
        mode = self.config.llm_mode
        if mode == "force":
            return True
        if mode == "never":
            return False
        return not ctx.offline and not request.constraints.deterministic

    async def _resolve_answer(self, request: TaskRequest, ctx: RunContext, evidence: Sequence[Evidence]) -> tuple[str, str]:
        """Find the candidate: ``inputs['answer']``, the task's 'Verify ...: X' tail, or a draft."""
        if request.inputs.get("answer"):
            return str(request.inputs["answer"])[: self.config.max_answer_chars], "inputs"
        m = _VERIFY_PREFIX.match(request.task)
        if m and m.group(1).strip():
            return m.group(1).strip()[: self.config.max_answer_chars], "task"
        if self.config.draft_when_missing and self.use_llm(ctx, request) and affordable(ctx, 1, reserve=1) > 0:
            with ctx.step("draft"):
                from .verifiers import documents_block

                req = build_request(ctx, P.DRAFT, schema=None, effort=self.config.derive_effort, documents=documents_block(evidence), task=request.task)
                resp = await ctx.llm_call(req, step=P.DRAFT.name)
            ctx.warn("no candidate answer supplied; a draft was generated and verified")
            return resp.text.strip()[: self.config.max_answer_chars], "draft"
        ctx.warn("no candidate answer supplied; verifying the task text itself")
        return request.task[: self.config.max_answer_chars], "task"

    async def _gather_evidence(self, request: TaskRequest, ctx: RunContext) -> list[Evidence]:
        evidence = list(request.context) + coerce_evidence(request.inputs.get("documents") or request.inputs.get("evidence"))
        seen: set[str] = set()
        unique = [e for e in evidence if not (e.id in seen or seen.add(e.id))]  # type: ignore[func-returns-value]
        if not unique and self.retriever is not None and request.constraints.allow_external_tools:
            with ctx.step("retrieve"):
                try:
                    unique = list(await self.retriever.retrieve(request.task, k=self.config.retrieve_k))
                except Exception as exc:  # noqa: BLE001 - retrieval is best-effort here
                    ctx.warn(f"retriever failed: {exc}")
        return unique

    def _plan_from_llm(self, lp: LLMPlan, request: TaskRequest, ctx: RunContext, *, has_evidence: bool, has_tests: bool) -> VerificationPlan:
        cfg = self.config
        verifiers = sanitize_verifiers(lp.verifiers, allow_code_execution=request.constraints.allow_code_execution)
        verifiers = [v for v in verifiers if not (v == "grounding" and not has_evidence) and not (v == "code_execution" and not has_tests)]
        if not verifiers:
            ctx.warn("LLM plan selected no applicable verifiers; using heuristic selection")
            verifiers = self.planner.plan(request, ctx.profile, answer=ctx.scratch.get("answer", ""), evidence_count=int(has_evidence)).verifiers
        risk = ctx.profile.risk_level
        thresholds = Thresholds(
            accept=effective_accept_threshold(cfg, ctx.profile, request.constraints),
            escalate=cfg.escalate_below,
            min_agreement=cfg.low_agreement_threshold,
            min_supported_fraction=cfg.grounding_min_supported,
            min_rubric_score=cfg.judge_min_score,
        )
        return VerificationPlan(
            answer_type=lp.answer_type,
            verifiers=verifiers,
            samples=max(1, min(int(lp.samples), cfg.samples_max)),
            thresholds=thresholds,
            use_evidence=lp.use_evidence,
            allow_revision=lp.allow_revision and cfg.allow_revision,
            judge_criteria=[c for c in lp.judge_criteria if c.strip()] or list(cfg.rubric_criteria) + list(ctx.profile.verification_rules),
            mandatory=[],
            final_effort="max" if risk == "safety_critical" else "high",
            rationale=lp.rationale.strip() or "LLM verification strategist",
            source="llm",
        )

    # ------------------------------------------------------------ life cycle
    async def plan(self, request: TaskRequest, ctx: RunContext) -> VerificationPlan:
        """Decide verifiers, samples and thresholds for this request."""
        cfg = self.config
        evidence = await self._gather_evidence(request, ctx)
        answer, source = await self._resolve_answer(request, ctx, evidence)
        ctx.scratch.update({"answer": answer, "answer_source": source, "evidence": evidence})
        policy = policy_for_profile(ctx.profile, cfg)
        ctx.scratch["policy"] = policy
        has_tests = bool(request.inputs.get("tests"))
        plan: VerificationPlan | None = None
        planning_calls = 0
        if self.use_llm(ctx, request) and affordable(ctx, 1) > 0:
            oracles = "\n".join(
                [
                    f"- evidence documents: {len(evidence)}",
                    f"- tests supplied: {'yes' if has_tests else 'no'}",
                    f"- code execution allowed: {'yes' if request.constraints.allow_code_execution else 'no'}",
                    f"- reference answer supplied: {'yes' if request.inputs.get('reference') else 'no'}",
                    f"- pre-sampled candidates: {len(request.inputs.get('candidates', []) or [])}",
                    f"- domain risk level: {ctx.profile.risk_level}",
                    f"- LLM call budget: {request.constraints.max_llm_calls if request.constraints.max_llm_calls is not None else 'unlimited'}",
                ]
            )
            try:
                req = build_request(ctx, P.PLANNER, schema=LLMPlan, effort=cfg.plan_effort, task=request.task, answer=answer, oracles=oracles)
                resp = await ctx.llm_call(req, step="plan.llm")
                planning_calls = 1
                plan = self._plan_from_llm(LLMPlan.model_validate(resp.parsed), request, ctx, has_evidence=bool(evidence), has_tests=has_tests)
            except BudgetExceededError:
                raise
            except (LLMError, ValueError) as exc:
                ctx.warn(f"LLM planning failed ({type(exc).__name__}); using heuristic planner")
                plan = None
        if plan is None:
            plan = self.planner.plan(request, ctx.profile, answer=answer, evidence_count=len(evidence))
        plan = apply_policy(plan, policy, has_evidence=bool(evidence))
        return fit_to_budget(plan, request.constraints, cfg, planning_calls=planning_calls)

    async def execute(self, request: TaskRequest, plan: VerificationPlan, ctx: RunContext) -> TaskResult:
        """Run the planned verifiers, fuse confidence and apply the acceptance gates."""
        cfg = self.config
        evidence: list[Evidence] = ctx.scratch.get("evidence") if "evidence" in ctx.scratch else await self._gather_evidence(request, ctx)
        if "answer" in ctx.scratch:
            answer, source = ctx.scratch["answer"], ctx.scratch.get("answer_source", "inputs")
        else:
            answer, source = await self._resolve_answer(request, ctx, evidence)
        policy: VerificationPolicy = ctx.scratch.get("policy") or policy_for_profile(ctx.profile, cfg)
        vi = VerifyInputs.from_plan(
            task=request.task,
            answer=answer,
            evidence=evidence,
            plan=plan,
            inputs=request.inputs,
            constraints=request.constraints,
            require_no_contradictions=policy.require_no_contradictions,
        )

        verdicts: list[Verdict] = []
        errors: list[str] = []
        budget_hit = False
        for name in plan.verifiers:
            verifier = self.verifiers.get(name)
            if verifier is None:
                ctx.warn(f"unknown verifier '{name}' skipped")
                continue
            if budget_hit or ctx.deadline_exceeded():
                verdicts.append(inconclusive(name, "skipped: budget exhausted"))
                continue
            with ctx.step(f"verify.{name}") as rec:
                try:
                    v = await verifier._verify(vi, ctx)
                except BudgetExceededError as exc:
                    budget_hit = True
                    ctx.warn(f"{name}: {exc}")
                    v = inconclusive(name, f"budget exhausted: {exc}")
                except (LLMError, ValueError, OSError) as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
                    v = inconclusive(name, f"error: {type(exc).__name__}: {exc}")
                rec.summary = f"{name}: " + ("inconclusive" if is_inconclusive(v) else f"{'pass' if v.passed else 'fail'} score={v.score:.2f}")
                rec.details.update({"passed": v.passed, "score": v.score, "issues": len(v.issues)})
            verdicts.append(v)

        if cfg.elicit_verbalized_confidence and self.use_llm(ctx, request) and not budget_hit and affordable(ctx, 1) > 0:
            with ctx.step("confidence.verbalized"):
                try:
                    verdicts.append(await elicit_verbalized_confidence(ctx, vi, cfg))
                except BudgetExceededError as exc:
                    budget_hit = True
                    ctx.warn(f"verbalized confidence skipped: {exc}")
                except (LLMError, ValueError) as exc:
                    ctx.warn(f"verbalized confidence failed: {exc}")

        with ctx.step("fuse"):
            weights = {**cfg.weights, "verbalized_confidence": cfg.verbalized_weight, **policy.weights}
            confidence, basis = aggregate_verdicts(verdicts, weights, calibration=cfg.calibration)

        # revision: the highest-authority verifier that produced an external-signal-backed correction proposes it
        # (LLM-edited revisions by grounding / chain_of_verification are already gated by plan.allow_revision and by
        # the presence of evidence inside those verifiers; self_consistency proposes the majority answer, a selection,
        # not an edit). The proposal is applied only after re-verification by the plan's external verifiers
        # (policy.max_revision_rounds rounds) or, when none can run, on the proposer's own authority if it is an
        # external signal (grounding, multi-sample agreement). Never on LLM judgement alone.
        corrections = [(AUTHORITY.get(v.method, 1), -i, v) for i, v in enumerate(verdicts) if v.corrected_answer and not is_inconclusive(v)]
        final_answer, revised, revision_source = answer, False, None
        revision_verified, revision_rounds, revision_verdicts = False, 0, []
        revision_notes: list[str] = []
        if corrections and not budget_hit:
            _, _, best = max(corrections, key=lambda t: (t[0], t[1]))
            proposed = str(best.corrected_answer)
            if policy.max_revision_rounds <= 0:
                revision_notes.append(f"{best.method} proposed a correction but revisions are disabled by policy (max_revision_rounds=0); original answer kept")
            else:
                with ctx.step("revision.confirm") as rec:
                    outcome = await self._confirm_revision(vi, plan, proposed, ctx, policy)
                    revision_rounds, revision_verdicts = outcome["rounds"], outcome["verdicts"]
                    rec.summary = f"{best.method}: {outcome['reason'] or 'confirmed'}"
                    rec.details.update({"confirmed": outcome["confirmed"], "rounds": revision_rounds})
                if outcome["confirmed"]:
                    final_answer, revised, revision_source, revision_verified = outcome["answer"], True, best.method, True
                elif outcome["checked"] == 0 and AUTHORITY.get(best.method, 1) >= 2:
                    final_answer, revised, revision_source = proposed, True, best.method
                    revision_notes.append(f"revision by {best.method} applied on its own authority (no external verifier available to confirm it)")
                else:
                    revision_notes.append(f"proposed correction by {best.method} not applied: {outcome['reason']}")
        elif corrections:
            revision_notes.append("a correction was proposed but the call budget is exhausted; original answer kept")

        reasons: list[str] = []
        ran = {v.method for v in verdicts if not is_inconclusive(v)}
        # Advisory signals (verbalised confidence) never gate acceptance on their own: if no verifier of record produced a
        # conclusive verdict, the fused confidence is the uninformative prior and the answer was not verified at all.
        verified_by = {v.method for v in verdicts if not is_inconclusive(v) and not v.details.get("advisory")}
        if not verified_by:
            skipped = ", ".join(v.method for v in verdicts if is_inconclusive(v)) or "none planned"
            reasons.append(f"no verifier produced a conclusive verdict (inconclusive: {skipped}); the answer was not verified and confidence is the uninformative prior")
        for m in plan.mandatory:
            if m not in ran:
                reasons.append(f"mandatory verifier '{m}' did not produce a verdict")
            elif not next(v for v in verdicts if v.method == m).passed:
                reasons.append(f"mandatory verifier '{m}' failed")
        if policy.require_no_contradictions:
            for v in verdicts:
                if v.method == "grounding" and not is_inconclusive(v) and int(v.details.get("contradicted", 0)) > 0:
                    reasons.append(f"{v.details['contradicted']} claim(s) contradicted by evidence (policy: no contradictions)")
        if policy.risk_level == "safety_critical" and not evidence and not request.inputs.get("tests"):
            reasons.append("safety-critical domain requires external evidence (documents or tests); none supplied")
        accept_threshold = max(plan.thresholds.accept, ctx.profile.min_confidence_to_accept, request.constraints.min_confidence)
        escalated = confidence < plan.thresholds.escalate
        if escalated:
            reasons.append(f"confidence {confidence:.2f} below escalation threshold {plan.thresholds.escalate:.2f}: abstain and escalate to a human")
        elif confidence < accept_threshold:
            reasons.append(f"confidence {confidence:.2f} below acceptance threshold {accept_threshold:.2f}")
        if revised:
            how = f"confirmed by re-verification in {revision_rounds} round(s)" if revision_verified else "on the proposer's authority"
            reasons.append(f"answer was revised by {revision_source} ({how}); review the correction")
        accepted = not reasons
        if accepted:
            status = Status.PARTIAL if (errors or budget_hit) else Status.SUCCESS
        else:
            status = Status.NEEDS_REVIEW

        metrics = self._metrics(verdicts, ctx)
        metrics["revision.rounds"] = float(revision_rounds)
        issues = [i for v in verdicts if not is_inconclusive(v) for i in v.issues] + revision_notes
        report = VerificationReport(
            task=request.task,
            answer=answer,
            answer_source=source,
            final_answer=final_answer,
            revised=revised,
            revision_source=revision_source,
            revision_verified=revision_verified,
            revision_rounds=revision_rounds,
            revision_verdicts=revision_verdicts,
            answer_type=plan.answer_type,
            domain=ctx.profile.name,
            accepted=accepted,
            status=status.value,
            escalated=escalated,
            confidence=confidence,
            confidence_basis=basis,
            accept_threshold=accept_threshold,
            escalate_threshold=plan.thresholds.escalate,
            verdicts=verdicts,
            issues=issues,
            review_reasons=reasons,
            evidence_ids=sorted({eid for v in verdicts for eid in v.evidence_ids}),
            metrics=metrics,
            plan=plan,
            policy=policy.description,
        )
        return self.new_result(
            request,
            status=status,
            output=report,
            output_text=report.render(),
            evidence=[e for e in evidence if e.id in set(report.evidence_ids)] or list(evidence)[:5],
            confidence=confidence,
            confidence_basis=basis,
            errors=errors,
            metadata={"plan": plan.model_dump(mode="json"), "policy": policy.model_dump(mode="json"), "answer_source": source},
        )

    async def _confirm_revision(
        self, vi: VerifyInputs, plan: VerificationPlan, proposed: str, ctx: RunContext, policy: VerificationPolicy
    ) -> dict[str, Any]:
        """Re-verify a proposed revision with the plan's external verifiers, up to ``policy.max_revision_rounds`` rounds.

        Each round runs ``grounding``, ``rules`` and ``code_execution`` (those present in the plan) on the current
        text. All conclusive checks passing confirms it. A failing grounding check may itself propose a further
        evidence-backed edit, which starts the next round while rounds remain. Returns ``confirmed``, the final
        ``answer``, the ``verdicts`` produced, the number of ``rounds``, how many checks were conclusive (``checked``)
        and a ``reason`` when not confirmed.
        """
        external = [n for n in ("grounding", "rules", "code_execution") if n in plan.verifiers and n in self.verifiers]
        out: dict[str, Any] = {"confirmed": False, "answer": proposed, "verdicts": [], "rounds": 0, "checked": 0, "reason": ""}
        if not external:
            out["reason"] = "no external verifier in the plan can confirm it"
            return out
        current = proposed
        for rnd in range(1, policy.max_revision_rounds + 1):
            out["rounds"] = rnd
            rvi = vi.model_copy(update={"answer": current, "allow_revision": rnd < policy.max_revision_rounds})
            checks: list[Verdict] = []
            for name in external:
                try:
                    checks.append(await self.verifiers[name]._verify(rvi, ctx))
                except BudgetExceededError as exc:
                    out["reason"] = f"budget exhausted during re-verification ({exc})"
                    return out
                except (LLMError, ValueError, OSError) as exc:
                    out["reason"] = f"{name} failed during re-verification ({type(exc).__name__}: {exc})"
                    return out
            out["verdicts"].extend(checks)
            conclusive = [c for c in checks if not is_inconclusive(c)]
            out["checked"] += len(conclusive)
            if not conclusive:
                out["reason"] = "no conclusive external check could be run on the revision"
                return out
            if all(c.passed for c in conclusive):
                out["confirmed"], out["answer"] = True, current
                return out
            failed = [c for c in conclusive if not c.passed]
            follow_up = next((c.corrected_answer for c in failed if c.corrected_answer and c.corrected_answer.strip() != current.strip()), None)
            if follow_up is None:
                out["reason"] = "revision still fails " + ", ".join(c.method for c in failed) + f" after {rnd} round(s)"
                return out
            current = str(follow_up)
        out["reason"] = f"revision not confirmed within {policy.max_revision_rounds} round(s)"
        return out

    @staticmethod
    def _metrics(verdicts: Sequence[Verdict], ctx: RunContext) -> dict[str, float]:
        m: dict[str, float] = {"verifiers_planned": float(len(verdicts)), "verifiers_conclusive": float(sum(1 for v in verdicts if not is_inconclusive(v)))}
        for v in verdicts:
            if is_inconclusive(v):
                continue
            d = v.details
            if v.method == "grounding":
                for k in ("supported_fraction", "claim_count", "contradicted", "citation_precision", "citation_recall"):
                    if k in d:
                        m[f"grounding.{k}"] = float(d[k])
            elif v.method == "self_consistency":
                for k in ("agreement_with_candidate", "majority_agreement", "normalized_entropy", "samples"):
                    if k in d:
                        m[f"self_consistency.{k}"] = float(d[k])
            elif v.method == "rubric_judge":
                m["rubric_judge.score"] = float(v.score)
            elif v.method == "verbalized_confidence":
                m["verbalized_confidence"] = float(v.score)
            elif v.method == "code_execution":
                m["code_execution.passed"] = 1.0 if v.passed else 0.0
            elif v.method == "rules":
                m["rules.hard_failures"] = float(d.get("hard_failures", 0))
        m["llm_calls"] = float(ctx.usage.llm_calls)
        return m

    # ------------------------------------------------------------ direct API
    async def verify_report(
        self,
        answer: str,
        *,
        task: str,
        evidence: Sequence[Evidence] = (),
        domain: str | None = None,
        inputs: dict[str, Any] | None = None,
        constraints: Constraints | None = None,
    ) -> VerificationReport:
        """Full report for a candidate answer (the :class:`Verdict` is ``report.to_verdict()``)."""
        report, _ = await self._verify_with_result(
            answer, task=task, evidence=evidence, domain=domain, inputs=inputs, constraints=constraints
        )
        return report

    async def _verify_with_result(
        self,
        answer: str,
        *,
        task: str,
        evidence: Sequence[Evidence] = (),
        domain: str | None = None,
        inputs: dict[str, Any] | None = None,
        constraints: Constraints | None = None,
    ) -> tuple[VerificationReport, TaskResult]:
        """Run verification and return both the report and the component's :class:`TaskResult`."""
        request = TaskRequest(
            task=task,
            inputs={**(inputs or {}), "answer": answer},
            context=list(evidence),
            domain=domain or self.config.default_domain,
            constraints=constraints or Constraints(),
        )
        result = await self.run(request)
        if isinstance(result.output, VerificationReport):
            return result.output, result
        plan = VerificationPlan(answer_type="free_text", verifiers=["rules"], rationale="run failed")
        return VerificationReport(
            task=task,
            answer=answer,
            final_answer=answer,
            answer_type="free_text",
            domain=request.domain,
            accepted=False,
            status=result.status.value,
            confidence=0.0,
            confidence_basis=result.confidence_basis or "run failed",
            accept_threshold=1.0,
            escalate_threshold=0.0,
            review_reasons=list(result.errors) or ["verification run failed"],
            plan=plan,
        ), result

    async def verify(
        self,
        answer: str,
        *,
        task: str,
        evidence: Sequence[Evidence] = (),
        domain: str | None = None,
        inputs: dict[str, Any] | None = None,
        constraints: Constraints | None = None,
    ) -> Verdict:
        """:class:`pipeline_core.Verifier` protocol: one fused verdict for ``answer``.

        ``verdict.details`` carries ``usage`` (the verification run's :class:`Usage` as a dict),
        ``component_status`` and ``request_id`` so callers (pipeline RAG, task decomposition)
        can roll verification cost into their own results with ``ctx.charge``.
        """
        report, result = await self._verify_with_result(
            answer, task=task, evidence=evidence, domain=domain, inputs=inputs, constraints=constraints
        )
        verdict = report.to_verdict()
        verdict.details["usage"] = result.usage.model_dump()
        verdict.details["component_status"] = result.status.value
        verdict.details["request_id"] = result.request_id
        return verdict

    def verify_sync(self, answer: str, *, task: str, evidence: Sequence[Evidence] = (), **kw: Any) -> Verdict:
        """Synchronous wrapper around :meth:`verify`."""
        return run_sync(self.verify(answer, task=task, evidence=evidence, **kw))

    async def compare(self, answer_a: str, answer_b: str, *, task: str, evidence: Sequence[Evidence] = (), reference: str | None = None) -> dict[str, Any]:
        """Position-swapped pairwise judgement (``winner`` is 'A', 'B' or 'tie')."""
        judge = self.verifiers["rubric_judge"]
        assert hasattr(judge, "compare")
        return await judge.compare(answer_a, answer_b, task=task, evidence=evidence, reference=reference)  # type: ignore[attr-defined]


__all__ = ["AnswerVerificationModel", "VerificationReport", "coerce_evidence"]
