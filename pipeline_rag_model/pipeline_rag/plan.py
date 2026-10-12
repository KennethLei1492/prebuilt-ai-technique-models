"""Plan schema and the heuristic planner.

``plan()`` is where the model *reasons about the prompt*: it reads the task,
looks at what evidence is available and decides the execution plan. Online
this is one structured LLM call (see :mod:`pipeline_rag.model`); offline
(``ctx.offline``) or when the request asks for determinism,
:class:`HeuristicPlanner` derives the plan from task features and the domain
profile.

Decision rules encoded here (see the README for the citations):

* Routing follows Adaptive-RAG / Self-Route (``router.route_for_features``).
* Identifier-heavy queries keep the literal rewrite only (HyDE and step-back
  hallucinate or over-generalise private tokens); "why/how" questions add a
  step-back variant; multi-faceted questions add multi-query.
* ``k`` defaults to the configured evidence cap (reader saturation, Lost in
  the Middle); multi-hop plans reserve ``k_per_hop`` new units per hop.
* High or safety-critical domains force grading, verification, inline
  citations and abstention, regardless of route.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline_core import DomainProfile, TaskRequest

from .config import PipelineRAGConfig
from .router import Route, TaskFeatures, extract_task_features, route_for_features

QueryTransform = Literal["rewrite", "multi_query", "step_back", "hyde"]
CitationStyle = Literal["inline", "footnote", "none"]
AnswerType = Literal["factual", "numerical", "procedural", "explanatory", "list", "comparison", "summary", "code"]


class RAGPlan(BaseModel):
    """What the pipeline will do for one request.

    The fields required by the architecture contract come first; the rest
    refine execution. No numeric bounds are declared on the schema because
    constrained decoders reject them; validators clamp instead.
    """

    model_config = ConfigDict(extra="forbid")

    route: Route = Field(description="no_retrieval | single_step | multi_step | agentic (Adaptive-RAG routing)")
    query_transforms: list[QueryTransform] = Field(default_factory=lambda: ["rewrite"], description="Query variants to generate before retrieval; results are fused with RRF")
    k: int = Field(default=10, description="Evidence units kept after fusion (1-100)")
    grade_context: bool = Field(default=True, description="Run the CRAG-style evidence gate before generation")
    compress: bool = Field(default=True, description="Extractive compression of evidence to the context token budget")
    max_hops: int = Field(default=1, description="Maximum retrieve-and-reason hops (1 for single_step, 2-10 otherwise)")
    verify: bool = Field(default=True, description="Run the injected verifier or the built-in grounding check after generation")
    citation_style: CitationStyle = Field(default="inline", description="inline [doc-id] markers | footnote list | none")
    rationale: str = Field(default="", description="One or two sentences explaining the plan")
    context_token_budget: int = Field(default=2000, description="Evidence token budget passed to the generator")
    abstain_when_insufficient: bool = Field(default=True, description="Answer 'insufficient evidence' rather than guess")
    escalate_on_abstention: bool = Field(default=True, description="Self-Route: escalate to the next richer plan when the cheap plan abstains")
    use_hierarchy: bool = Field(default=False, description="Build and search RAPTOR-style document summaries (global questions)")
    expected_answer_type: AnswerType = Field(default="factual", description="Shape of the expected answer; guides the generator's format")
    planner: Literal["heuristic", "llm"] = Field(default="heuristic", description="Which planner produced the plan")

    @field_validator("k")
    @classmethod
    def _clamp_k(cls, v: int) -> int:
        return max(1, min(100, int(v)))

    @field_validator("max_hops")
    @classmethod
    def _clamp_hops(cls, v: int) -> int:
        return max(1, min(10, int(v)))

    @field_validator("context_token_budget")
    @classmethod
    def _clamp_budget(cls, v: int) -> int:
        return max(100, min(200_000, int(v)))

    @field_validator("query_transforms")
    @classmethod
    def _dedupe_transforms(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for t in v:
            if t not in out:
                out.append(t)
        return out  # type: ignore[return-value]

    def normalized(self, config: PipelineRAGConfig) -> "RAGPlan":
        """Make a plan (possibly LLM-produced) internally consistent with its route and the config."""
        data = self.model_dump()
        if data["route"] == "no_retrieval":
            data.update(query_transforms=[], max_hops=1, grade_context=False, compress=False, use_hierarchy=False)
        elif data["route"] == "single_step":
            data["max_hops"] = 1
        else:
            # at least two hops when the config allows them, never more than the configured cap
            data["max_hops"] = max(1, min(max(2, data["max_hops"]), config.max_hops))
        if data["route"] == "agentic" and config.build_hierarchy != "never":
            data["use_hierarchy"] = True
        if config.build_hierarchy == "never":
            data["use_hierarchy"] = False
        if config.build_hierarchy == "always" and data["route"] != "no_retrieval":
            data["use_hierarchy"] = True
        if "hyde" in data["query_transforms"] and not config.enable_hyde:
            data["query_transforms"] = [t for t in data["query_transforms"] if t != "hyde"]
        data["query_transforms"] = data["query_transforms"][: config.max_query_variants]
        if data["route"] != "no_retrieval" and not data["query_transforms"]:
            data["query_transforms"] = ["rewrite"]
        if config.max_escalations == 0:
            data["escalate_on_abstention"] = False
        return RAGPlan(**data)

    def describe(self) -> str:
        """Short human-readable rendering for CLI output and logs."""
        parts = [
            f"route={self.route}",
            f"transforms={','.join(self.query_transforms) or '-'}",
            f"k={self.k}",
            f"hops={self.max_hops}",
            f"grade={'y' if self.grade_context else 'n'}",
            f"compress={'y' if self.compress else 'n'}",
            f"verify={'y' if self.verify else 'n'}",
            f"citations={self.citation_style}",
            f"answer={self.expected_answer_type}",
            f"planner={self.planner}",
        ]
        line = " ".join(parts)
        return f"{line}\n  rationale: {self.rationale}" if self.rationale else line


class HeuristicPlanner:
    """Deterministic planner used when the LLM is unavailable or determinism is requested."""

    def __init__(self, config: PipelineRAGConfig) -> None:
        self.config = config

    def plan(self, request: TaskRequest, profile: DomainProfile, *, has_retriever: bool = False) -> RAGPlan:
        """Produce a :class:`RAGPlan` from the request's task features and the domain profile."""
        features = extract_task_features(
            request.task, request.inputs, context=request.context, has_retriever=has_retriever, profile=profile
        )
        return self.plan_from_features(features, profile)

    def plan_from_features(self, features: TaskFeatures, profile: DomainProfile) -> RAGPlan:
        """Apply the decision rules to pre-computed features (pure; unit-testable)."""
        cfg = self.config
        decision = route_for_features(features)
        route = decision.route
        high_risk = profile.risk_level in ("high", "safety_critical")
        reasons = [decision.reason]

        transforms: list[str] = ["rewrite"]
        if features.why_how and not features.has_identifiers:
            transforms.append("step_back")
        if route in ("multi_step", "agentic") or features.enumerations >= 2 or features.question_count >= 2:
            transforms.append("multi_query")
        if features.has_identifiers:
            reasons.append("identifiers present: literal query kept, no HyDE/step-back over private tokens")
        elif cfg.enable_hyde and not high_risk and route == "single_step":
            transforms.append("hyde")

        k = cfg.default_k if route == "single_step" else max(cfg.default_k, cfg.k_per_hop * 2)
        if cfg.full_corpus_gate_tokens and 0 < features.corpus_tokens <= cfg.full_corpus_gate_tokens:
            k = 100
            reasons.append(f"corpus of ~{features.corpus_tokens} tokens is under the full-corpus gate; every chunk is used")

        if high_risk:
            reasons.append(f"{profile.risk_level} domain: grading, verification, inline citations and abstention are mandatory")

        citation_style: str = cfg.citation_style if route != "no_retrieval" else "none"
        if high_risk and route != "no_retrieval":
            citation_style = "inline"

        plan = RAGPlan(
            route=route,
            query_transforms=transforms if route != "no_retrieval" else [],  # type: ignore[arg-type]
            k=k,
            grade_context=route != "no_retrieval",
            compress=route != "no_retrieval",
            max_hops=1 if route in ("no_retrieval", "single_step") else cfg.max_hops,
            verify=route != "no_retrieval" or high_risk,
            citation_style=citation_style,  # type: ignore[arg-type]
            rationale="; ".join(reasons),
            context_token_budget=cfg.context_token_budget,
            abstain_when_insufficient=cfg.abstain_when_insufficient or high_risk,
            escalate_on_abstention=cfg.max_escalations > 0,
            use_hierarchy=route == "agentic",
            expected_answer_type=answer_type_for_features(features),
            planner="heuristic",
        )
        return plan.normalized(cfg)


def answer_type_for_features(features: TaskFeatures) -> AnswerType:
    """Guess the expected answer shape from task features (pure)."""
    if features.global_signals:
        return "summary"
    if any(("compar" in s) or ("differen" in s) or ("versus" in s) or ("vs" in s) for s in features.multi_hop_signals):
        return "comparison"
    qw = features.question_words[0] if features.question_words else ""
    if qw == "list" or features.enumerations >= 3:
        return "list"
    if features.numeric:
        return "numerical"
    if qw in ("how",) and not features.why_how:
        return "procedural"
    if features.why_how:
        return "explanatory" if qw != "how" else "procedural"
    return "factual"


__all__ = ["AnswerType", "CitationStyle", "HeuristicPlanner", "QueryTransform", "RAGPlan", "answer_type_for_features"]
