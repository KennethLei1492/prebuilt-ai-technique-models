"""Verifier implementations.

Every class here satisfies the :class:`pipeline_core.Verifier` protocol
(``await v.verify(answer, task=..., evidence=...) -> Verdict``) and can be used
standalone, but inside :class:`~answer_verification.model.AnswerVerificationModel`
they share one :class:`pipeline_core.RunContext` so budget, tracing and usage
roll up into a single result.

The precedence rule from the literature is baked into the verdicts: executed
tests and deterministic rules are terminal authorities, evidence grounding and
multi-sample agreement are strong external signals, LLM judgement is advisory,
and intrinsic self-critique is never a verifier. Each LLM-backed verifier has
an explicit offline behaviour: it either falls back to a pure heuristic
(grounding, rules, code execution) or returns an *inconclusive* verdict that
carries no weight in confidence fusion (self-consistency without samples,
chain of verification, rubric judge, debate).
"""
from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core import (
    ENGINEERING_RIGOR,
    BaseLLMClient,
    BudgetExceededError,
    Constraints,
    Evidence,
    Executor,
    LLMError,
    LLMRequest,
    PromptTemplate,
    RunContext,
    TaskRequest,
    Tracer,
    Verdict,
    gather_limited,
    render_evidence,
    sanitize_schema,
    stable_hash,
)

from . import prompts as P
from .claims import (
    Claim,
    ClaimLabel,
    ClaimVerdict,
    citation_metrics,
    grounding_metrics,
    lexical_support,
    rank_evidence_for_claim,
    split_claims,
)
from .config import AnswerVerificationConfig
from .plan import AnswerType, Thresholds, VerificationPlan, classify_answer_type, extract_features, looks_like_code
from .sandbox import extract_code, fenced_languages, run_python
from .uncertainty import (
    AgreementResult,
    canonicalize_answer,
    extract_number,
    semantic_agreement,
    should_stop_sampling,
    similarity_ratio,
    verbalized_confidence,
)
from .units import check_units, parse_quantities

# --------------------------------------------------------------------------
# Structured-output schemas for LLM calls (flat, required-all, no numeric bounds)
# --------------------------------------------------------------------------


class DerivedAnswer(BaseModel):
    """One independent re-derivation."""

    model_config = ConfigDict(extra="forbid")

    final_answer: str
    key_steps: list[str]


class USCSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_index: int = Field(description="1-based index of the most consistent candidate")
    rationale: str


class EquivalenceGroups(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups: list[list[int]] = Field(description="Clusters of 1-based candidate indices")


class CoVeQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    claim_excerpt: str


class CoVeQuestions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[CoVeQuestion]


class CoVeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    confident: bool
    evidence_ids: list[str]


class CoVeLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[Literal["agree", "contradict", "unknown"]]


class Revision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revised_answer: str
    changes: list[str]


class ExtractedClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim]


class ClaimLabelOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: ClaimLabel
    evidence_ids: list[str]
    note: str


class ClaimLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[ClaimLabelOut]


class CriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str
    score: int
    rationale: str


class RubricScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[CriterionScore]
    overall: int


class PairwiseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B", "tie"]
    rationale: str


class DebatePosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acceptable: bool
    position: str
    key_issue: str


class DebateJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "incorrect", "uncertain"]
    confidence: float
    unresolved_issues: list[str]


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


class VerifyInputs(BaseModel):
    """Everything a verifier needs for one call, resolved once by the model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    task: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    answer_type: AnswerType = "free_text"
    thresholds: Thresholds = Field(default_factory=Thresholds)
    samples: int = 5
    allow_revision: bool = True
    judge_criteria: list[str] = Field(default_factory=list)
    final_effort: str = "high"
    require_no_contradictions: bool = False
    inputs: dict[str, Any] = Field(default_factory=dict)
    constraints: Constraints = Field(default_factory=Constraints)

    @classmethod
    def from_plan(
        cls,
        *,
        task: str,
        answer: str,
        evidence: Sequence[Evidence],
        plan: VerificationPlan,
        inputs: dict[str, Any] | None = None,
        constraints: Constraints | None = None,
        require_no_contradictions: bool = False,
    ) -> "VerifyInputs":
        return cls(
            task=task,
            answer=answer,
            evidence=list(evidence),
            answer_type=plan.answer_type,
            thresholds=plan.thresholds,
            samples=plan.samples,
            allow_revision=plan.allow_revision,
            judge_criteria=list(plan.judge_criteria),
            final_effort=plan.final_effort,
            require_no_contradictions=require_no_contradictions,
            inputs=dict(inputs or {}),
            constraints=constraints or Constraints(),
        )


def inconclusive(method: str, reason: str, **details: Any) -> Verdict:
    """A verdict that carries no weight in fusion but is reported."""
    return Verdict(method=method, passed=False, score=0.5, issues=[], details={"inconclusive": True, "reason": reason, **details})


def build_request(
    ctx: RunContext,
    template: PromptTemplate,
    *,
    schema: type[BaseModel] | None,
    effort: str,
    model: str | None = None,
    **values: Any,
) -> LLMRequest:
    """Render a template into an :class:`LLMRequest` with the cache-aware layout.

    Stable blocks: template system text, few-shot examples, the domain profile
    and (for high-risk domains) the engineering-rigor rules. Volatile content
    (task, answer, documents) goes in the user turn through the template's
    ``instructions``.
    """
    extra: list[str] = [ctx.profile.system_text()]
    if ctx.profile.risk_level in ("high", "safety_critical"):
        extra.append(ENGINEERING_RIGOR)
    rendered = template.render(extra_system=extra, **values)
    return LLMRequest(
        system_stable=rendered.system_blocks,
        messages=[{"role": "user", "content": rendered.user}],  # type: ignore[list-item]
        output_schema=sanitize_schema(schema.model_json_schema()) if schema is not None else None,
        effort=effort,  # type: ignore[arg-type]
        model=model,
        tags=[template.name, f"v{template.version}"],
    )


def documents_block(evidence: Sequence[Evidence], *, max_items: int = 12) -> str:
    """Render evidence for the user turn, or an empty string when there is none."""
    return render_evidence(evidence, max_items=max_items) if evidence else ""


def affordable(ctx: RunContext, wanted: int, *, reserve: int = 0) -> int:
    """How many LLM calls can be made now without breaching the call budget."""
    remaining = ctx.budget.remaining_calls()
    if remaining is None:
        return max(0, wanted)
    return max(0, min(wanted, remaining - reserve))


def _seeded_shuffle(items: Sequence[Any], seed_material: str) -> list[Any]:
    out = list(items)
    random.Random(stable_hash(seed_material)).shuffle(out)
    return out


class BaseVerifier(ABC):
    """Common behaviour for all verifiers.

    Subclasses implement :meth:`_verify` against a :class:`VerifyInputs` and a
    :class:`RunContext`. The public :meth:`verify` matches the
    :class:`pipeline_core.Verifier` protocol and builds an ad-hoc context when
    the verifier is used standalone.
    """

    name: str = "base"
    authority: int = 1

    def __init__(
        self,
        config: AnswerVerificationConfig | None = None,
        *,
        llm: BaseLLMClient | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.config = config or AnswerVerificationConfig()
        self._llm = llm
        self.executor = executor

    # ------------------------------------------------------------ protocol
    async def verify(
        self,
        answer: str,
        *,
        task: str,
        evidence: Sequence[Evidence] = (),
        ctx: RunContext | None = None,
        plan: VerificationPlan | None = None,
        inputs: dict[str, Any] | None = None,
        require_no_contradictions: bool = False,
    ) -> Verdict:
        """Verify ``answer`` for ``task``; standalone use creates its own context."""
        if ctx is None:
            llm = self._llm
            if llm is None:
                from pipeline_core import make_llm

                llm = make_llm()
            request = TaskRequest(task=task, inputs=dict(inputs or {}), context=list(evidence), domain=self.config.default_domain)
            ctx = RunContext(request, Tracer(name=f"verifier.{self.name}"), llm, f"verifier.{self.name}")
        if plan is None:
            f = extract_features(task, answer, evidence_count=len(evidence), has_tests=bool((inputs or {}).get("tests")))
            plan = VerificationPlan(answer_type=classify_answer_type(f), verifiers=[self.name], samples=self.config.samples_default)
        vi = VerifyInputs.from_plan(
            task=task,
            answer=answer,
            evidence=evidence,
            plan=plan,
            inputs=inputs if inputs is not None else dict(ctx.request.inputs),
            constraints=ctx.request.constraints,
            require_no_contradictions=require_no_contradictions,
        )
        return await self._verify(vi, ctx)

    @abstractmethod
    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict: ...

    # ------------------------------------------------------------- helpers
    def llm_enabled(self, ctx: RunContext) -> bool:
        """Whether this verifier may call the LLM under the configured mode."""
        mode = self.config.llm_mode
        if mode == "force":
            return True
        if mode == "never":
            return False
        return not ctx.offline

    async def fan_out(self, ctx: RunContext, fn: Callable[[Any], Awaitable[Any]], items: Sequence[Any]) -> list[Any]:
        """Run ``fn`` over ``items`` concurrently through the injected Executor or a bounded gather."""
        if not items:
            return []
        limit = max(1, min(self.config.max_concurrency, ctx.request.constraints.max_parallelism))
        if self.executor is not None:
            return list(await self.executor.map(fn, items, concurrency=limit))
        return list(await gather_limited((fn(it) for it in items), limit, return_exceptions=True))

    async def structured(self, ctx: RunContext, request: LLMRequest, schema: type[BaseModel]) -> BaseModel:
        """One structured call validated into ``schema``."""
        resp = await ctx.llm_call(request, step=request.tags[0] if request.tags else None)
        return schema.model_validate(resp.parsed)


# --------------------------------------------------------------------------
# Self-consistency (adaptive) + Universal Self-Consistency
# --------------------------------------------------------------------------


class SelfConsistencyVerifier(BaseVerifier):
    """N independent re-derivations with varied framing; agreement is the signal.

    * canonicalizable answers (numerical, factual, code): majority vote over
      canonical keys with Adaptive-Consistency early stopping;
    * free-form answers: clusters by lexical (or LLM-judged) equivalence and a
      Universal-Self-Consistency selection pass over shuffled candidates.

    Pre-sampled alternatives in ``inputs["candidates"]`` are used before any
    LLM call, which also gives the verifier an LLM-free path.
    """

    name = "self_consistency"
    authority = 2

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        cfg = self.config
        samples: list[str] = [str(c) for c in vi.inputs.get("candidates", []) if str(c).strip()]
        sampled_llm = 0
        stopped_early = False
        if self.llm_enabled(ctx):
            wanted = max(0, vi.samples - len(samples))
            batch = max(1, cfg.sample_batch)
            framings = list(P.DERIVE_FRAMINGS)
            while wanted > 0:
                n = affordable(ctx, min(batch, wanted), reserve=1)
                if n <= 0 or ctx.deadline_exceeded():
                    ctx.warn("self_consistency: sampling stopped by budget")
                    break
                idx = list(range(sampled_llm, sampled_llm + n))
                outs = await self.fan_out(ctx, lambda i: self._derive(ctx, vi, framings[i % len(framings)]), idx)
                for o in outs:
                    if isinstance(o, BudgetExceededError):
                        raise o
                    if isinstance(o, BaseException):
                        ctx.warn(f"self_consistency: sample failed: {o}")
                        continue
                    samples.append(o)
                sampled_llm += n
                wanted -= n
                counts = semantic_agreement(samples, answer_type=vi.answer_type, jaccard_threshold=cfg.equivalence_jaccard).counts
                if vi.answer_type in ("numerical", "factual", "code") and should_stop_sampling(
                    counts, threshold=cfg.stability_threshold, min_samples=cfg.samples_min
                ):
                    stopped_early = wanted > 0
                    break
        if not samples:
            return inconclusive(self.name, "no samples available (offline and no inputs['candidates'])")

        pool = samples + [vi.answer]
        agreement = await self._cluster(ctx, vi, pool)
        cand_idx = len(pool) - 1
        cand_cluster = next(c for c in agreement.clusters if cand_idx in c)
        agree_with_candidate = (len(cand_cluster) - 1) / len(samples)
        majority = agreement.clusters[0]
        majority_rep = pool[majority[0]]
        usc_index: Optional[int] = None
        usc_passes: list[int] = []
        if vi.answer_type in ("free_text", "design", "procedural") and self.llm_enabled(ctx) and len(pool) >= 2:
            usc_index, usc_passes = await self._usc_select(ctx, vi, pool)
        issues: list[str] = []
        passed = agree_with_candidate >= vi.thresholds.min_agreement
        if usc_index is not None:
            usc_agrees = usc_index in cand_cluster
            passed = passed or (usc_agrees and agree_with_candidate > 0)
            if not usc_agrees:
                issues.append("USC selected a different candidate as most consistent")
        corrected: Optional[str] = None
        if cand_idx not in majority and len(majority) / len(samples) >= vi.thresholds.min_agreement:
            corrected = majority_rep
            issues.append(f"majority of {len(majority)}/{len(samples)} independent derivations disagree with the candidate")
        score = agree_with_candidate
        if agreement.normalized_entropy > 0.8:
            issues.append("high semantic entropy across samples: the task may be underspecified")
        return Verdict(
            method=self.name,
            passed=passed,
            score=round(score, 4),
            issues=issues,
            corrected_answer=corrected,
            details={
                "samples": len(samples),
                "sampled_llm": sampled_llm,
                "stopped_early": stopped_early,
                "agreement_with_candidate": round(agree_with_candidate, 4),
                "majority_agreement": round(agreement.agreement, 4),
                "entropy": round(agreement.entropy, 4),
                "normalized_entropy": round(agreement.normalized_entropy, 4),
                "clusters": agreement.counts,
                "majority_answer": majority_rep,
                "usc_selected_index": usc_index,
                "usc_passes": usc_passes,
            },
        )

    async def _derive(self, ctx: RunContext, vi: VerifyInputs, framing: str) -> str:
        req = build_request(
            ctx,
            P.DERIVE,
            schema=DerivedAnswer,
            effort=self.config.derive_effort,
            documents=documents_block(vi.evidence),
            task=vi.task,
            framing=framing,
        )
        out = await self.structured(ctx, req, DerivedAnswer)
        return out.final_answer.strip()  # type: ignore[attr-defined]

    async def _cluster(self, ctx: RunContext, vi: VerifyInputs, pool: Sequence[str]) -> AgreementResult:
        cfg = self.config
        if (
            vi.answer_type in ("free_text", "design", "procedural")
            and cfg.llm_equivalence_clustering
            and self.llm_enabled(ctx)
            and affordable(ctx, 1, reserve=1) > 0
        ):
            try:
                cands = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(pool))
                req = build_request(ctx, P.EQUIVALENCE, schema=EquivalenceGroups, effort=cfg.grade_effort, task=vi.task, candidates=cands)
                groups = (await self.structured(ctx, req, EquivalenceGroups)).groups  # type: ignore[attr-defined]
                membership: dict[int, int] = {}
                for gi, g in enumerate(groups):
                    for one_based in g:
                        if 1 <= one_based <= len(pool):
                            membership.setdefault(one_based - 1, gi)
                if len(membership) == len(pool):
                    return semantic_agreement(pool, equivalent=lambda a, b: membership[pool.index(a)] == membership[pool.index(b)])
                ctx.warn("self_consistency: LLM equivalence groups incomplete; using lexical clustering")
            except (LLMError, ValueError) as exc:
                ctx.warn(f"self_consistency: equivalence call failed ({exc}); using lexical clustering")
        return semantic_agreement(pool, answer_type=vi.answer_type, jaccard_threshold=cfg.equivalence_jaccard)

    async def _usc_select(self, ctx: RunContext, vi: VerifyInputs, pool: Sequence[str]) -> tuple[Optional[int], list[int]]:
        passes = affordable(ctx, self.config.usc_selection_passes, reserve=1)
        choices: list[int] = []
        for p in range(passes):
            order = _seeded_shuffle(range(len(pool)), f"{vi.task}|{p}")
            cands = "\n".join(f"[{i + 1}] {pool[j]}" for i, j in enumerate(order))
            req = build_request(ctx, P.USC_SELECT, schema=USCSelection, effort=self.config.grade_effort, task=vi.task, candidates=cands)
            try:
                sel = await self.structured(ctx, req, USCSelection)
            except (LLMError, ValueError) as exc:
                ctx.warn(f"self_consistency: USC pass failed ({exc})")
                continue
            k = int(sel.selected_index) - 1  # type: ignore[attr-defined]
            if 0 <= k < len(order):
                choices.append(order[k])
        if not choices:
            return None, []
        modal = max(set(choices), key=lambda c: (choices.count(c), -c))
        return modal, choices


# --------------------------------------------------------------------------
# Chain of Verification (factored, with evidence-gated minimal revision)
# --------------------------------------------------------------------------


class ChainOfVerificationVerifier(BaseVerifier):
    """Plan verification questions, answer each without the draft, cross-check, revise.

    The factored variant (Dhuliawala et al.) is used: the question-answering
    calls never see the draft, so they cannot copy its hallucinations. A
    revision is produced only when contradictions were found *and* evidence
    documents were available to answer the questions from (an external signal);
    without documents the contradictions are reported, not auto-corrected.
    """

    name = "chain_of_verification"
    authority = 1

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        if not self.llm_enabled(ctx):
            return inconclusive(self.name, "requires an LLM (offline)")
        cfg = self.config
        if affordable(ctx, 3, reserve=1) < 3:
            return inconclusive(self.name, "insufficient call budget")
        plan_req = build_request(ctx, P.COVE_PLAN, schema=CoVeQuestions, effort=cfg.derive_effort, task=vi.task, answer=vi.answer, max_questions=cfg.cove_max_questions)
        questions = (await self.structured(ctx, plan_req, CoVeQuestions)).questions[: cfg.cove_max_questions]  # type: ignore[attr-defined]
        questions = [q for q in questions if q.question.strip()]
        if not questions:
            return inconclusive(self.name, "no verification questions were generated")
        n = affordable(ctx, len(questions), reserve=2)
        if n < len(questions):
            ctx.warn(f"chain_of_verification: budget allows {n}/{len(questions)} verification questions")
            questions = questions[:n]
        if not questions:
            return inconclusive(self.name, "insufficient call budget for verification questions")
        docs = documents_block(vi.evidence)

        async def answer_one(q: CoVeQuestion) -> CoVeAnswer:
            req = build_request(ctx, P.COVE_ANSWER, schema=CoVeAnswer, effort=cfg.derive_effort, documents=docs, task=vi.task, question=q.question)
            return await self.structured(ctx, req, CoVeAnswer)  # type: ignore[return-value]

        outs = await self.fan_out(ctx, answer_one, questions)
        answers: list[Optional[CoVeAnswer]] = []
        for o in outs:
            if isinstance(o, BudgetExceededError):
                raise o
            answers.append(None if isinstance(o, BaseException) else o)
        items = [
            {"claim": q.claim_excerpt, "question": q.question, "verified_answer": a.answer if a else "unknown", "confident": bool(a and a.confident)}
            for q, a in zip(questions, answers)
        ]
        labels: list[str]
        if affordable(ctx, 1, reserve=0) > 0:
            items_txt = "\n".join(f"{i + 1}. claim: {it['claim']} | question: {it['question']} | verified_answer: {it['verified_answer']} | confident: {it['confident']}" for i, it in enumerate(items))
            cross = await self.structured(ctx, build_request(ctx, P.COVE_CROSSCHECK, schema=CoVeLabels, effort=cfg.grade_effort, items=items_txt), CoVeLabels)
            labels = list(cross.labels)[: len(items)]  # type: ignore[attr-defined]
        else:
            labels = []
        while len(labels) < len(items):
            labels.append("unknown")
        agree = sum(1 for l in labels if l == "agree")
        contra = [items[i] for i, l in enumerate(labels) if l == "contradict"]
        unknown = sum(1 for l in labels if l == "unknown")
        total = len(items)
        score = (agree + 0.5 * unknown) / total if total else 0.5
        issues = [f"contradicted: '{c['claim']}' (verified: {c['verified_answer']})" for c in contra]
        corrected: Optional[str] = None
        preservation: Optional[float] = None
        if contra and vi.allow_revision and vi.evidence and affordable(ctx, 1) > 0:
            corrections = "\n".join(f"- claim: {c['claim']} -> verified answer: {c['verified_answer']}" for c in contra)
            try:
                rev = await self.structured(ctx, build_request(ctx, P.REVISE, schema=Revision, effort=vi.final_effort, documents=docs, answer=vi.answer, corrections=corrections), Revision)
                preservation = similarity_ratio(vi.answer, rev.revised_answer)  # type: ignore[attr-defined]
                if preservation >= cfg.revision_min_preservation and rev.revised_answer.strip():  # type: ignore[attr-defined]
                    corrected = rev.revised_answer  # type: ignore[attr-defined]
                else:
                    issues.append(f"revision rejected: preservation {preservation:.2f} < {cfg.revision_min_preservation}")
            except BudgetExceededError:
                raise
            except (LLMError, ValueError) as exc:
                # the plan / answer / cross-check calls already produced the verdict; a failed optional revision must not erase it
                ctx.warn(f"chain_of_verification: revision failed ({exc}); contradictions reported without a correction")
                issues.append("revision call failed; contradictions reported without a correction")
        elif contra and not vi.evidence:
            issues.append("contradictions found but no evidence documents: not auto-corrected (intrinsic self-correction is unreliable)")
        return Verdict(
            method=self.name,
            passed=not contra,
            score=round(score, 4),
            issues=issues,
            corrected_answer=corrected,
            evidence_ids=sorted({eid for a in answers if a for eid in a.evidence_ids}),
            details={
                "questions": total,
                "agree": agree,
                "contradict": len(contra),
                "unknown": unknown,
                "items": [{**it, "label": labels[i]} for i, it in enumerate(items)],
                "preservation": preservation,
            },
        )


# --------------------------------------------------------------------------
# Grounding (FActScore / ALCE style)
# --------------------------------------------------------------------------


class GroundingVerifier(BaseVerifier):
    """Atomic claims -> supported / contradicted / unsupported against evidence.

    Online: LLM claim extraction and classification. Offline: sentence-level
    claims and lexical support (content-token coverage plus number agreement).
    Always reports supported fraction, claim count and contradiction count
    together, plus ALCE-style citation precision/recall.
    """

    name = "grounding"
    authority = 2

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        if not vi.evidence:
            return inconclusive(self.name, "no evidence available")
        cfg = self.config
        use_llm = self.llm_enabled(ctx) and affordable(ctx, 2, reserve=0) >= 2
        claims: list[Claim] = []
        if use_llm:
            try:
                req = build_request(ctx, P.CLAIM_EXTRACT, schema=ExtractedClaims, effort=cfg.grade_effort, task=vi.task, answer=vi.answer, max_claims=cfg.grounding_max_claims)
                claims = list((await self.structured(ctx, req, ExtractedClaims)).claims)[: cfg.grounding_max_claims]  # type: ignore[attr-defined]
            except (LLMError, ValueError) as exc:
                ctx.warn(f"grounding: claim extraction failed ({exc}); using sentence split")
        claims = [c for c in claims if c.text.strip()] or split_claims(vi.answer, max_claims=cfg.grounding_max_claims)
        if not claims and vi.answer.strip():
            # A one-line answer ("9.6 mm", "ASTM A743 CA6NM") is below the sentence splitter's minimum length but is
            # itself one checkable claim; grounding it is the most common engineering case, not an edge case.
            text = vi.answer.strip()
            claims = [Claim(text=text, type="numerical" if extract_number(text) is not None else "factual")]
        if not claims:
            return inconclusive(self.name, "answer contains no checkable claims")
        per_claim_ev = [rank_evidence_for_claim(c.text, vi.evidence, k=cfg.grounding_evidence_per_claim) for c in claims]
        verdicts: list[ClaimVerdict] = []
        mode = "lexical"
        if use_llm and affordable(ctx, 1) > 0:
            seen: dict[str, Evidence] = {}
            for evs in per_claim_ev:
                for e in evs:
                    seen.setdefault(e.id, e)
            claims_txt = "\n".join(f"{i + 1}. [{c.type}] {c.text}" for i, c in enumerate(claims))
            try:
                labelled = await self.structured(
                    ctx,
                    build_request(ctx, P.CLAIM_CLASSIFY, schema=ClaimLabels, effort=cfg.grade_effort, documents=render_evidence(list(seen.values()), max_items=cfg.grounding_evidence_per_claim * 3), claims=claims_txt),
                    ClaimLabels,
                )
                outs = list(labelled.verdicts)  # type: ignore[attr-defined]
                if len(outs) == len(claims):
                    known = {e.id for e in vi.evidence}
                    for c, o in zip(claims, outs):
                        ids = [i for i in o.evidence_ids if i in known]
                        verdicts.append(ClaimVerdict(claim=c.text, label=o.label, evidence_ids=ids, note=o.note))
                    mode = "llm"
                else:
                    ctx.warn("grounding: classifier returned a different number of verdicts; using lexical support")
            except (LLMError, ValueError) as exc:
                ctx.warn(f"grounding: classification failed ({exc}); using lexical support")
        if not verdicts:
            verdicts = [lexical_support(c.text, evs, threshold=cfg.lexical_support_threshold) for c, evs in zip(claims, per_claim_ev)]
        gm = grounding_metrics(verdicts)
        cm = citation_metrics(vi.answer, verdicts, [e.id for e in vi.evidence])
        contradicted = int(gm["contradicted"])
        passed = gm["supported_fraction"] >= vi.thresholds.min_supported_fraction and (contradicted == 0 or not vi.require_no_contradictions)
        issues = [f"{v.label}: {v.claim}" for v in verdicts if v.label != "supported"]
        corrected: Optional[str] = None
        preservation: Optional[float] = None
        if contradicted and vi.allow_revision and use_llm and affordable(ctx, 1) > 0:
            by_id = {e.id: e for e in vi.evidence}
            corrections = "\n".join(
                f"- claim: {v.claim} -> contradicted by {', '.join(v.evidence_ids) or 'evidence'}: "
                + " | ".join(by_id[i].short(300) for i in v.evidence_ids if i in by_id)
                for v in verdicts
                if v.label == "contradicted"
            )
            try:
                rev = await self.structured(ctx, build_request(ctx, P.REVISE, schema=Revision, effort=vi.final_effort, documents=documents_block(vi.evidence), answer=vi.answer, corrections=corrections), Revision)
                preservation = similarity_ratio(vi.answer, rev.revised_answer)  # type: ignore[attr-defined]
                if preservation >= cfg.revision_min_preservation and rev.revised_answer.strip():  # type: ignore[attr-defined]
                    corrected = rev.revised_answer  # type: ignore[attr-defined]
                else:
                    issues.append(f"revision rejected: preservation {preservation:.2f} < {cfg.revision_min_preservation}")
            except (LLMError, ValueError) as exc:
                ctx.warn(f"grounding: revision failed ({exc})")
        score = gm["supported_fraction"] * (0.5 if contradicted else 1.0)
        return Verdict(
            method=self.name,
            passed=passed,
            score=round(score, 4),
            issues=issues,
            corrected_answer=corrected,
            evidence_ids=sorted({eid for v in verdicts if v.label == "supported" for eid in v.evidence_ids}),
            details={
                "mode": mode,
                "claim_count": int(gm["claim_count"]),
                "supported": int(gm["supported"]),
                "contradicted": contradicted,
                "unsupported": int(gm["unsupported"]),
                "supported_fraction": round(gm["supported_fraction"], 4),
                "citation_precision": round(cm["citation_precision"], 4),
                "citation_recall": round(cm["citation_recall"], 4),
                "claims": [v.model_dump() for v in verdicts],
                "preservation": preservation,
            },
        )


# --------------------------------------------------------------------------
# Rubric judge (G-Eval style) with position-swapped pairwise mode
# --------------------------------------------------------------------------


class RubricJudgeVerifier(BaseVerifier):
    """Criteria scoring averaged over several framings; pairwise mode swaps positions.

    Follows the debiasing checklist from the judge literature: explicit
    criteria, reference-guided when a reference exists, anti-verbosity rule,
    several independent passes averaged (framing variation instead of
    temperature), optional cross-family ``judge_model``, and in pairwise mode a
    win only counts when both orderings agree.
    """

    name = "rubric_judge"
    authority = 1

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        if not self.llm_enabled(ctx):
            return inconclusive(self.name, "requires an LLM (offline)")
        cfg = self.config
        passes = affordable(ctx, cfg.judge_samples, reserve=0)
        if passes <= 0:
            return inconclusive(self.name, "insufficient call budget")
        criteria = vi.judge_criteria or list(cfg.rubric_criteria)
        reference = vi.inputs.get("reference")
        ref_block = f"<reference>\n{reference}\n</reference>" if reference else ""
        docs = documents_block(vi.evidence)
        framings = list(P.JUDGE_FRAMINGS)

        async def judge_once(i: int) -> RubricScores:
            req = build_request(
                ctx,
                P.RUBRIC,
                schema=RubricScores,
                effort=cfg.grade_effort,
                model=cfg.judge_model,
                documents=docs,
                task=vi.task,
                criteria="\n".join(f"- {c}" for c in criteria),
                scale=str(cfg.rubric_scale),
                reference=ref_block,
                answer=vi.answer,
                framing=framings[i % len(framings)],
            )
            return await self.structured(ctx, req, RubricScores)  # type: ignore[return-value]

        outs = await self.fan_out(ctx, judge_once, list(range(passes)))
        results: list[RubricScores] = []
        for o in outs:
            if isinstance(o, BudgetExceededError):
                raise o
            if isinstance(o, BaseException):
                ctx.warn(f"rubric_judge: pass failed: {o}")
            else:
                results.append(o)
        if not results:
            return inconclusive(self.name, "all judge passes failed")
        normalized = [self.normalize(r, cfg.rubric_scale) for r in results]
        score = sum(normalized) / len(normalized)
        per_criterion: dict[str, list[float]] = {}
        issues: list[str] = []
        for r in results:
            for s in r.scores:
                per_criterion.setdefault(s.criterion, []).append(float(s.score))
                if s.score <= cfg.rubric_scale / 2:
                    issues.append(f"{s.criterion}: {s.score}/{cfg.rubric_scale} - {s.rationale}")
        return Verdict(
            method=self.name,
            passed=score >= vi.thresholds.min_rubric_score,
            score=round(score, 4),
            issues=list(dict.fromkeys(issues))[:10],
            details={
                "passes": len(results),
                "criteria": {k: round(sum(v) / len(v), 3) for k, v in per_criterion.items()},
                "overall": [r.overall for r in results],
                "reference_guided": bool(reference),
                "judge_model": cfg.judge_model,
            },
        )

    @staticmethod
    def normalize(r: RubricScores, scale: int) -> float:
        """Map a rubric result to [0, 1]: mean of criterion mean and overall, rescaled from 1..scale."""
        vals = [min(max(float(s.score), 1.0), float(scale)) for s in r.scores]
        mean_c = sum(vals) / len(vals) if vals else float(r.overall)
        overall = min(max(float(r.overall), 1.0), float(scale))
        raw = (mean_c + overall) / 2.0
        return (raw - 1.0) / (scale - 1.0) if scale > 1 else 1.0

    async def compare(
        self,
        answer_a: str,
        answer_b: str,
        *,
        task: str,
        evidence: Sequence[Evidence] = (),
        reference: str | None = None,
        ctx: RunContext | None = None,
    ) -> dict[str, Any]:
        """Pairwise judgement with position swapping; a win counts only if both orders agree."""
        if ctx is None:
            llm = self._llm
            if llm is None:
                from pipeline_core import make_llm

                llm = make_llm()
            request = TaskRequest(task=task, context=list(evidence), domain=self.config.default_domain)
            ctx = RunContext(request, Tracer(name="verifier.pairwise"), llm, "verifier.pairwise")
        docs = documents_block(evidence)
        ref_block = f"<reference>\n{reference}\n</reference>" if reference else ""

        async def once(a: str, b: str) -> str:
            req = build_request(ctx, P.PAIRWISE, schema=PairwiseVerdict, effort=self.config.grade_effort, model=self.config.judge_model, documents=docs, task=task, reference=ref_block, answer_a=a, answer_b=b)
            return (await self.structured(ctx, req, PairwiseVerdict)).winner  # type: ignore[attr-defined]

        first = await once(answer_a, answer_b)
        if not self.config.pairwise_swap:
            return {"winner": first, "position_consistent": None, "orders": [first]}
        second_raw = await once(answer_b, answer_a)
        second = {"A": "B", "B": "A", "tie": "tie"}[second_raw]
        return pairwise_decision(first, second)


def pairwise_decision(first: str, second_mapped: str) -> dict[str, Any]:
    """Balanced position calibration: both orders (mapped to the original labels) must agree."""
    consistent = first == second_mapped
    winner = first if consistent else "tie"
    return {"winner": winner, "position_consistent": consistent, "orders": [first, second_mapped]}


# --------------------------------------------------------------------------
# Rules (deterministic checks from the domain profile and request inputs)
# --------------------------------------------------------------------------


class RuleCheck(BaseModel):
    """One deterministic check result."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    passed: bool
    severity: Literal["hard", "soft"] = "hard"
    detail: str = ""


class RuleVerifier(BaseVerifier):
    """Schema / regex / units / numeric-range / required-sections / reference checks.

    Rules come from two places and need no LLM:

    * the :class:`pipeline_core.DomainProfile` (``units`` drives unit hygiene;
      ``code_languages`` gates syntax checks);
    * ``inputs["rules"]``: a list of dicts such as ``{"type": "regex", "pattern":
      "...", "must_match": true}``, ``{"type": "range", "min": 0, "max": 10}``,
      ``{"type": "required_sections", "sections": [...]}``, ``{"type":
      "forbidden", "pattern": "..."}``, ``{"type": "max_words", "value": 200}``,
      ``{"type": "json_schema", "required": [...]}``; plus ``inputs["reference"]``
      (numeric closeness within ``inputs["tolerance"]``, default 1%).

    Passing rules are necessary, not sufficient, so a clean pass scores 0.75
    (0.95 when a numeric reference matched); any hard failure caps fusion. When
    no check applies at all (no domain units, no rules, no reference, not code)
    the verdict is *inconclusive*: nothing was verified, so nothing is claimed.

    Numeric comparisons (``reference``, ``range``) use the number the answer
    states as its result (:func:`~answer_verification.uncertainty.extract_number`),
    selected by unit family when the reference carries a unit or the ``range``
    rule names one (``{"type": "range", "min": 50, "max": 100, "unit": "kW"}``).
    """

    name = "rules"
    authority = 3

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        checks = run_rule_checks(
            vi.answer,
            rules=list(vi.inputs.get("rules", []) or []),
            expected_units=list(ctx.profile.units),
            answer_type=vi.answer_type,
            reference=vi.inputs.get("reference"),
            tolerance=float(vi.inputs.get("tolerance", 0.01) or 0.01),
            code_languages=list(ctx.profile.code_languages),
        )
        if not checks:
            # Nothing was checked, so nothing was verified: a vacuous pass must not count as an external signal
            # (it would otherwise accept any answer in a domain without ``units`` and "confirm" revisions).
            return inconclusive(
                self.name,
                "no deterministic check applies (domain declares no units; no inputs['rules'] or inputs['reference']; not a code answer)",
                checks=[],
            )
        hard_fail = [c for c in checks if not c.passed and c.severity == "hard"]
        soft_fail = [c for c in checks if not c.passed and c.severity == "soft"]
        ref_ok = any(c.rule == "reference_match" and c.passed for c in checks)
        if hard_fail:
            score = max(0.05, 0.3 - 0.1 * (len(hard_fail) - 1))
        else:
            score = 0.95 if ref_ok else 0.75
            score -= 0.1 * len(soft_fail)
        return Verdict(
            method=self.name,
            passed=not hard_fail,
            score=round(max(0.0, min(1.0, score)), 4),
            issues=[f"{c.rule}: {c.detail}" for c in hard_fail + soft_fail],
            details={"checks": [c.model_dump() for c in checks], "hard_failures": len(hard_fail), "soft_failures": len(soft_fail), "reference_matched": ref_ok},
        )


def run_rule_checks(
    answer: str,
    *,
    rules: Sequence[dict[str, Any]] = (),
    expected_units: Sequence[str] = (),
    answer_type: str = "free_text",
    reference: Any = None,
    tolerance: float = 0.01,
    code_languages: Sequence[str] = (),
) -> list[RuleCheck]:
    """Pure function behind :class:`RuleVerifier` (unit-tested without an LLM)."""
    import json
    import re

    checks: list[RuleCheck] = []
    # units: hygiene is checked whenever the domain declares units; missing units are hard only for numerical answers
    if expected_units:
        rep = check_units(answer, expected_units=list(expected_units), require_units=answer_type == "numerical")
        if rep.unknown_units:
            checks.append(RuleCheck(rule="units.known", passed=False, severity="soft", detail="unknown unit token(s): " + ", ".join(sorted(set(rep.unknown_units)))))
        else:
            checks.append(RuleCheck(rule="units.known", passed=True, detail=f"{len(rep.quantities)} quantities parsed"))
        if answer_type == "numerical":
            # The final answer number must carry a unit (hard); other bare decimals (factors such as E = 1.0) are soft.
            last = rep.quantities[-1] if rep.quantities else None
            final_missing = last is not None and last.unit is None and last.raw in rep.bare_decimals
            checks.append(
                RuleCheck(
                    rule="units.present",
                    passed=not rep.bare_decimals,
                    severity="hard" if final_missing else "soft",
                    detail=("final answer number has no unit; " if final_missing else "") + "quantities without units: " + ", ".join(rep.bare_decimals[:5])
                    if rep.bare_decimals
                    else "all decimal quantities carry units",
                )
            )
    # reference value closeness: compare the answer's result number, selected by the reference's unit family
    if reference is not None and answer_type in ("numerical", "factual"):
        ref_text = str(reference)
        ref_unit = next((q.unit for q in parse_quantities(ref_text) if q.family is not None), None)
        ref_num = extract_number(ref_text, unit=ref_unit)
        got = extract_number(answer, unit=ref_unit)
        if ref_num is not None and got is not None:
            tol = abs(ref_num) * tolerance if ref_num != 0 else tolerance
            ok = abs(got - ref_num) <= tol
            unit_note = f" [{ref_unit}]" if ref_unit else ""
            checks.append(RuleCheck(rule="reference_match", passed=ok, severity="hard", detail=f"answer {got:g} vs reference {ref_num:g}{unit_note} (tolerance {tolerance:.2%})"))
        elif ref_num is None and canonicalize_answer(ref_text) == canonicalize_answer(answer):
            checks.append(RuleCheck(rule="reference_match", passed=True, detail="canonical text match"))
    # python syntax for code answers: only when the text actually looks like code (a planner may still label prose 'code')
    if answer_type == "code" and looks_like_code(answer) and (not code_languages or "python" in [c.lower() for c in code_languages]):
        code, lang = extract_code(answer, language_hint="python")
        if lang == "python":
            try:
                compile(code, "<candidate>", "exec")
                checks.append(RuleCheck(rule="code.parses", passed=True, detail="python source compiles"))
            except SyntaxError as exc:
                checks.append(RuleCheck(rule="code.parses", passed=False, detail=f"SyntaxError: {exc.msg} (line {exc.lineno})"))
    # explicit rules
    for r in rules:
        kind = str(r.get("type", "")).lower()
        desc = str(r.get("description") or kind)
        if kind == "regex":
            m = re.search(str(r.get("pattern", "")), answer, re.I | re.M | re.S) is not None
            want = bool(r.get("must_match", True))
            checks.append(RuleCheck(rule=f"regex:{desc}", passed=m == want, detail=f"pattern {'matched' if m else 'not matched'}; must_match={want}"))
        elif kind == "forbidden":
            m = re.search(str(r.get("pattern", "")), answer, re.I | re.M) is not None
            checks.append(RuleCheck(rule=f"forbidden:{desc}", passed=not m, detail="forbidden pattern present" if m else "absent"))
        elif kind == "range":
            unit = str(r["unit"]) if r.get("unit") else None
            n = extract_number(answer, unit=unit)
            lo, hi = r.get("min"), r.get("max")
            ok = n is not None and (lo is None or n >= float(lo)) and (hi is None or n <= float(hi))
            unit_note = f" {unit}" if unit else ""
            checks.append(RuleCheck(rule=f"range:{desc}", passed=ok, detail=f"value {n}{unit_note} within [{lo}, {hi}]" if ok else f"value {n}{unit_note} outside [{lo}, {hi}]"))
        elif kind == "required_sections":
            missing = [s for s in r.get("sections", []) if not re.search(rf"\b{re.escape(str(s))}\b", answer, re.I)]
            checks.append(RuleCheck(rule=f"required_sections:{desc}", passed=not missing, detail="missing: " + ", ".join(missing) if missing else "all sections present"))
        elif kind == "max_words":
            n_words = len(answer.split())
            lim = int(r.get("value", 0))
            checks.append(RuleCheck(rule=f"max_words:{desc}", passed=n_words <= lim, severity="soft", detail=f"{n_words} words (limit {lim})"))
        elif kind == "json_schema":
            try:
                obj = json.loads(extract_code(answer, language_hint="json")[0])
                req_keys = [k for k in r.get("required", []) if not (isinstance(obj, dict) and k in obj)]
                checks.append(RuleCheck(rule=f"json:{desc}", passed=not req_keys, detail="missing keys: " + ", ".join(req_keys) if req_keys else "valid JSON with required keys"))
            except (ValueError, TypeError) as exc:
                checks.append(RuleCheck(rule=f"json:{desc}", passed=False, detail=f"not valid JSON: {exc}"))
        else:
            checks.append(RuleCheck(rule=f"unknown:{desc}", passed=True, severity="soft", detail="rule type not understood; skipped"))
    return checks


# --------------------------------------------------------------------------
# Code execution (sandboxed tests)
# --------------------------------------------------------------------------


class CodeExecutionVerifier(BaseVerifier):
    """Run caller-supplied tests against the candidate code in a subprocess sandbox.

    Executes only when ``constraints.allow_code_execution`` is true and
    ``inputs["tests"]`` is present; Python is the supported runtime. Execution
    is the terminal authority for code: failing tests cap fused confidence.
    """

    name = "code_execution"
    authority = 3

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        if not vi.constraints.allow_code_execution:
            return inconclusive(self.name, "code execution not allowed by constraints")
        tests = vi.inputs.get("tests")
        if not tests:
            return inconclusive(self.name, "no tests supplied in inputs['tests']")
        explicit = vi.inputs.get("language")
        langs = [c.lower() for c in ctx.profile.code_languages]
        hint = str(explicit or (langs[0] if langs else "python"))
        code, lang = extract_code(vi.answer, language_hint=hint)
        if not explicit and not fenced_languages(vi.answer) and (not langs or "python" in langs):
            # Unfenced (or untagged) snippet, no caller hint: the profile's *first* language is not evidence of anything,
            # so use the one runtime available whenever the profile lists Python at all.
            lang = "python"
        if lang != "python":
            return inconclusive(self.name, f"no sandbox runtime for language '{lang}'")
        res = await asyncio.to_thread(run_python, code, str(tests), timeout_s=self.config.code_timeout_s, max_output_chars=self.config.code_max_output_chars)
        if res.timed_out:
            score = 0.1
        elif res.passed:
            score = 0.98
        elif res.tests_passed is not None and (res.tests_passed + (res.tests_failed or 0)) > 0:
            score = 0.3 * res.tests_passed / (res.tests_passed + (res.tests_failed or 0))
        else:
            score = 0.05
        issues: list[str] = []
        if not res.passed:
            tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
            issues.append("tests failed: " + (" | ".join(tail) if tail else f"exit code {res.exit_code}"))
        return Verdict(method=self.name, passed=res.passed, score=round(score, 4), issues=issues, details=res.model_dump())


# --------------------------------------------------------------------------
# Debate (bounded, optional)
# --------------------------------------------------------------------------


class DebateVerifier(BaseVerifier):
    """A agents x R rounds of position/rebuttal, then a judge; stops early on convergence.

    Each agent argues from a different reviewer persona (:data:`prompts.DEBATE_FRAMINGS`)
    and sees its peers' positions in its own order, so the agents are independent
    samples rather than copies of one prompt. Peers see a one-line summary of each
    other's positions rather than full transcripts (Du et al.), and the judge
    weighs arguments by evidence rather than headcount. Used only for open-ended judgement in high-risk domains,
    because at matched budget debate underperforms self-consistency on
    canonicalizable tasks (Huang et al.).
    """

    name = "debate"
    authority = 1

    async def _verify(self, vi: VerifyInputs, ctx: RunContext) -> Verdict:
        if not self.llm_enabled(ctx):
            return inconclusive(self.name, "requires an LLM (offline)")
        cfg = self.config
        agents = cfg.debate_agents
        if affordable(ctx, agents + 1) < agents + 1:
            return inconclusive(self.name, "insufficient call budget")
        docs = documents_block(vi.evidence)
        positions: list[DebatePosition] = []
        transcript: list[str] = []
        rounds_run = 0
        converged = False
        for rnd in range(min(cfg.debate_rounds, cfg.debate_max_rounds)):
            if affordable(ctx, agents, reserve=1) < agents:
                ctx.warn("debate: stopping early, budget")
                break
            framings = list(P.DEBATE_FRAMINGS)
            peer_lines = [f"- reviewer {i + 1}: {'accepts' if p.acceptable else 'rejects'}: {p.position}" for i, p in enumerate(positions)]

            async def turn(i: int) -> DebatePosition:
                # Diversity comes from framing, not temperature (ARCHITECTURE.md 3.2): each agent gets its own reviewer
                # persona and sees the peers' positions in its own order, so the N agents are not N copies of one prompt.
                peers_txt = "\n".join(_seeded_shuffle(peer_lines, f"{vi.task}|{rnd}|{i}")) or "(none yet)"
                req = build_request(
                    ctx,
                    P.DEBATE_TURN,
                    schema=DebatePosition,
                    effort=cfg.derive_effort,
                    documents=docs,
                    task=vi.task,
                    answer=vi.answer,
                    peers=peers_txt,
                    framing=framings[(rnd * agents + i) % len(framings)],
                )
                return await self.structured(ctx, req, DebatePosition)  # type: ignore[return-value]

            outs = await self.fan_out(ctx, turn, list(range(agents)))
            new_positions: list[DebatePosition] = []
            for o in outs:
                if isinstance(o, BudgetExceededError):
                    raise o
                if isinstance(o, BaseException):
                    ctx.warn(f"debate: agent turn failed: {o}")
                else:
                    new_positions.append(o)
            if not new_positions:
                break
            positions = new_positions
            rounds_run += 1
            transcript.append(f"round {rnd + 1}:\n" + "\n".join(f"  reviewer {i + 1} ({'accept' if p.acceptable else 'reject'}): {p.position} [issue: {p.key_issue}]" for i, p in enumerate(positions)))
            if len({p.acceptable for p in positions}) == 1 and rounds_run >= 1 and cfg.debate_adaptive_stop:
                converged = True
                break
        if not positions:
            return inconclusive(self.name, "no debate positions produced")
        judged = False
        verdict, conf, unresolved = "uncertain", 0.0, []
        if affordable(ctx, 1) > 0:
            try:
                judge = await self.structured(ctx, build_request(ctx, P.DEBATE_JUDGE, schema=DebateJudgement, effort=vi.final_effort, model=cfg.judge_model, task=vi.task, answer=vi.answer, transcript="\n".join(transcript)), DebateJudgement)
                verdict, conf, unresolved = judge.verdict, float(judge.confidence), list(judge.unresolved_issues)  # type: ignore[attr-defined]
                judged = True
            except BudgetExceededError:
                raise
            except (LLMError, ValueError) as exc:
                ctx.warn(f"debate: judge call failed ({exc}); using the reviewers' vote share")
        if not judged:
            # Judge-less fallback (no budget left or the judge call failed): the final-round vote share decides.
            acc = sum(1 for p in positions if p.acceptable) / len(positions)
            verdict, conf, unresolved = ("correct" if acc > 0.5 else "incorrect" if acc < 0.5 else "uncertain"), abs(acc - 0.5) * 2, [p.key_issue for p in positions if not p.acceptable]
        conf = min(1.0, max(0.0, conf))
        score = conf if verdict == "correct" else (1.0 - conf) if verdict == "incorrect" else 0.5
        return Verdict(
            method=self.name,
            passed=verdict == "correct",
            score=round(score, 4),
            issues=unresolved[:10],
            details={"agents": agents, "rounds": rounds_run, "converged": converged, "verdict": verdict, "judge_confidence": conf, "judged": judged, "accept_votes": sum(1 for p in positions if p.acceptable)},
        )


# --------------------------------------------------------------------------
# Verbalised confidence (advisory signal, not a verifier of record)
# --------------------------------------------------------------------------

#: Kept under its historical name; the implementation lives in :mod:`.uncertainty`.
elicit_verbalized_confidence = verbalized_confidence


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

VERIFIER_CLASSES: dict[str, type[BaseVerifier]] = {
    SelfConsistencyVerifier.name: SelfConsistencyVerifier,
    ChainOfVerificationVerifier.name: ChainOfVerificationVerifier,
    GroundingVerifier.name: GroundingVerifier,
    RubricJudgeVerifier.name: RubricJudgeVerifier,
    RuleVerifier.name: RuleVerifier,
    CodeExecutionVerifier.name: CodeExecutionVerifier,
    DebateVerifier.name: DebateVerifier,
}


def make_verifiers(config: AnswerVerificationConfig, *, llm: BaseLLMClient | None = None, executor: Executor | None = None) -> dict[str, BaseVerifier]:
    """Instantiate every registered verifier with shared config, client and executor."""
    return {name: cls(config, llm=llm, executor=executor) for name, cls in VERIFIER_CLASSES.items()}


__all__ = [
    "BaseVerifier",
    "ChainOfVerificationVerifier",
    "CodeExecutionVerifier",
    "DebateVerifier",
    "GroundingVerifier",
    "RubricJudgeVerifier",
    "RuleCheck",
    "RuleVerifier",
    "SelfConsistencyVerifier",
    "VERIFIER_CLASSES",
    "VerifyInputs",
    "affordable",
    "build_request",
    "elicit_verbalized_confidence",
    "inconclusive",
    "make_verifiers",
    "pairwise_decision",
    "run_rule_checks",
]
