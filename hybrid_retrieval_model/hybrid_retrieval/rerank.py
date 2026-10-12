"""Rerankers: the single largest quality lever in the cascade (Thakur et al. 2021).

* :class:`LLMListwiseReranker` - RankGPT-style sliding-window permutation
  generation (window 20, stride 10, bottom-up) with structured output
  (Sun et al. 2023; Pradeep et al. 2023). Invalid or partial permutations are
  repaired deterministically.
* :class:`LLMPointwiseReranker` - independent 0-10 grades with a rationale,
  issued in parallel (monoBERT/monoT5 pattern with an LLM; Nogueira et al. 2019, 2020).
* :class:`CrossEncoderReranker` - adapter for a sentence-transformers
  ``CrossEncoder`` (lazy import; any object with ``predict(pairs)`` works).
* :class:`NoopReranker` - passthrough.

Rerankers accept a ``RunContext`` so calls are budgeted and traced; without one
(direct API) they call their own client and accumulate ``self.usage``.

Degradation contract (spec 3.1.4 / 3.2): before every window or grading fan-out
the reranker consults :func:`~hybrid_retrieval.plan.budget_exhausted`; when the
latency, call or cost budget is spent, or when every window/grade fails, the
candidates are returned **unchanged** (fused scores, ranks and ``retriever``
field intact) and ``reranker.applied`` is ``False`` so callers do not claim a
rerank that did not happen.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from pipeline_core import BaseLLMClient, Evidence, LLMRequest, Usage, gather_limited, xml
from pipeline_core.component import RunContext
from pipeline_core.errors import BudgetExceededError, ConfigurationError, LLMError, StructuredOutputError
from pipeline_core.protocols import Executor

from .plan import budget_exhausted
from .prompts import LISTWISE_RERANK_PROMPT, POINTWISE_RERANK_PROMPT, ListwiseRankingOutput, PointwiseScoreOutput


@runtime_checkable
class Reranker(Protocol):
    """Structural interface for rerankers."""

    name: str
    applied: bool
    """``True`` after :meth:`rerank` when at least one window/grade changed the order; ``False`` when the fused order was kept."""

    async def rerank(self, query: str, candidates: Sequence[Evidence], *, k: int | None = None, ctx: RunContext | None = None) -> list[Evidence]: ...


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without an LLM)
# ---------------------------------------------------------------------------


def sliding_windows(n: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Bottom-up ``(start, end)`` windows so strong candidates bubble to the top (RankGPT)."""
    if n <= 0:
        return []
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if n <= window:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    end = n
    while True:
        start = max(0, end - window)
        out.append((start, end))
        if start == 0:
            break
        end -= stride
    return out


def repair_permutation(perm: Sequence[int], n: int) -> list[int]:
    """Turn a (possibly invalid) 1-based permutation into a valid 0-based one.

    Out-of-range and duplicate entries are dropped; missing positions are
    appended in their original order, so an empty answer leaves the order unchanged.
    """
    seen: set[int] = set()
    out: list[int] = []
    for p in perm:
        try:
            i = int(p) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    out.extend(i for i in range(n) if i not in seen)
    return out


def apply_window_permutation(order: list[int], start: int, end: int, perm: Sequence[int]) -> list[int]:
    """Reorder ``order[start:end]`` according to a 0-based ``perm`` over that window."""
    window = order[start:end]
    fixed = repair_permutation([p + 1 for p in perm], len(window))
    return order[:start] + [window[i] for i in fixed] + order[end:]


def _with_rerank(ev: Evidence, *, score: float, rank: int, method: str, **extra: Any) -> Evidence:
    meta = dict(ev.metadata)
    meta.setdefault("fused_rank", ev.rank)
    meta.setdefault("fused_score", ev.score)
    meta.update({"rerank_score": score, "rerank_method": method, **extra})
    return ev.model_copy(update={"score": score, "rank": rank, "retriever": "reranked", "metadata": meta})


# ---------------------------------------------------------------------------
# LLM rerankers
# ---------------------------------------------------------------------------


class _LLMReranker:
    def __init__(self, llm: BaseLLMClient | None = None, *, passage_chars: int = 1200, effort: str = "low", model: str | None = None, extra_system: Sequence[str] = ()) -> None:
        self.llm = llm
        self.passage_chars = passage_chars
        self.effort = effort
        self.model = model
        self.extra_system = [s for s in extra_system if s]
        self.usage = Usage()
        self.warnings: list[str] = []
        self.applied = False

    async def _call(self, request: LLMRequest, ctx: RunContext | None, step: str):
        if ctx is not None:
            reason = budget_exhausted(ctx)
            if reason:  # never spend a call the budget cannot cover (spec 3.2)
                raise BudgetExceededError(reason)
            return await ctx.llm_call(request, step=step)
        if self.llm is None:
            raise ConfigurationError(f"{type(self).__name__} needs an llm when used without a RunContext")
        resp = await self.llm.complete(request)
        self.usage = self.usage + resp.usage
        return resp

    def _truncate(self, text: str) -> str:
        t = text.replace("\n", " ").strip()
        return t if len(t) <= self.passage_chars else t[: self.passage_chars - 3] + "..."

    def _warn(self, ctx: RunContext | None, msg: str) -> None:
        """Record a warning on the reranker (direct API) and on the context when one exists."""
        self.warnings.append(msg)
        if ctx is not None:
            ctx.warn(msg)


class LLMListwiseReranker(_LLMReranker):
    """Sliding-window permutation reranker (RankGPT)."""

    name = "listwise"

    def __init__(self, llm: BaseLLMClient | None = None, *, window: int = 20, stride: int = 10, passes: int = 1, **kw: Any) -> None:
        super().__init__(llm, **kw)
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.stride = stride
        self.passes = max(1, passes)

    def _render(self, query: str, passages: Sequence[Evidence]) -> LLMRequest:
        body = "\n".join(xml("passage", self._truncate(p.content), index=i + 1) for i, p in enumerate(passages))
        rp = LISTWISE_RERANK_PROMPT.render(extra_system=self.extra_system, passages=body, query=query, n=len(passages))
        return LLMRequest(system_stable=rp.system_blocks, messages=[{"role": "user", "content": rp.user}], output_schema=rp.output_schema, effort=self.effort, model=self.model, tags=["rerank", "listwise"], max_tokens=2000)  # type: ignore[list-item]

    async def rerank(self, query: str, candidates: Sequence[Evidence], *, k: int | None = None, ctx: RunContext | None = None) -> list[Evidence]:
        """Rerank ``candidates``; returns them unchanged (``applied=False``) when no window could be applied."""
        cands = list(candidates)
        n = len(cands)
        self.applied = False
        self.windows_applied = 0
        if n <= 1:  # nothing to order; no call is spent and no rerank is claimed
            return cands[: k or n]
        order = list(range(n))
        stopped = False
        for _ in range(self.passes):
            if stopped:
                break
            for start, end in sliding_windows(n, self.window, self.stride):
                window = [cands[i] for i in order[start:end]]
                try:
                    resp = await self._call(self._render(query, window), ctx, "rerank.listwise")
                    perm = ListwiseRankingOutput.model_validate(resp.parsed).ranking
                except BudgetExceededError as exc:
                    self._warn(ctx, f"listwise rerank stopped at window {start}-{end}: {exc}; keeping fused order for the remaining windows")
                    stopped = True
                    break
                except (LLMError, StructuredOutputError, ValueError) as exc:
                    self._warn(ctx, f"listwise rerank window {start}-{end} failed ({type(exc).__name__}); keeping fused order")
                    continue
                if not perm:
                    self._warn(ctx, "listwise reranker returned an empty ranking; keeping fused order for this window")
                    continue
                order = apply_window_permutation(order, start, end, repair_permutation(perm, len(window)))
                self.windows_applied += 1
        if not self.windows_applied:
            return cands[: k or n]
        self.applied = True
        out: list[Evidence] = []
        for pos, idx in enumerate(order):
            score = 1.0 - pos / n
            out.append(_with_rerank(cands[idx], score=round(score, 6), rank=pos + 1, method=self.name))
        return out[: k or n]


class LLMPointwiseReranker(_LLMReranker):
    """Parallel 0-10 relevance grading with rationale."""

    name = "pointwise"

    def __init__(self, llm: BaseLLMClient | None = None, *, concurrency: int = 8, executor: Executor | None = None, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self.concurrency = max(1, concurrency)
        self.executor = executor
        self._budget_warned = False

    def _render(self, query: str, passage: Evidence) -> LLMRequest:
        rp = POINTWISE_RERANK_PROMPT.render(extra_system=self.extra_system, passage=self._truncate(passage.content), query=query)
        return LLMRequest(system_stable=rp.system_blocks, messages=[{"role": "user", "content": rp.user}], output_schema=rp.output_schema, effort=self.effort, model=self.model, tags=["rerank", "pointwise"], max_tokens=400)  # type: ignore[list-item]

    async def _grade(self, query: str, ev: Evidence, ctx: RunContext | None) -> tuple[float, str]:
        try:
            resp = await self._call(self._render(query, ev), ctx, "rerank.pointwise")
            out = PointwiseScoreOutput.model_validate(resp.parsed)
            return float(min(10, max(0, int(out.score)))) / 10.0, out.rationale
        except BudgetExceededError as exc:
            if not self._budget_warned:  # one warning for the whole fan-out; later grades short-circuit in _call
                self._budget_warned = True
                self._warn(ctx, f"pointwise rerank stopped: {exc}; ungraded candidates keep the fused order")
            return -1.0, ""
        except (LLMError, StructuredOutputError, ValueError) as exc:
            self._warn(ctx, f"pointwise grade failed for {ev.id} ({type(exc).__name__}); using fused score")
            return -1.0, ""

    async def rerank(self, query: str, candidates: Sequence[Evidence], *, k: int | None = None, ctx: RunContext | None = None) -> list[Evidence]:
        """Grade ``candidates`` in parallel; returns them unchanged (``applied=False``) when no grade succeeded."""
        cands = list(candidates)
        self.applied = False
        self._budget_warned = False
        if not cands:
            return []
        reason = budget_exhausted(ctx)
        if reason:  # spec 3.2: check the budget before the fan-out
            self._warn(ctx, f"pointwise rerank skipped: {reason}; keeping fused order")
            return cands[: k or len(cands)]
        conc = self.concurrency
        if ctx is not None:
            conc = min(conc, ctx.request.constraints.max_parallelism)
            rem = ctx.budget.remaining_calls()
            if rem is not None and rem < len(cands):
                self._warn(ctx, f"pointwise rerank limited to {rem} of {len(cands)} candidates by the call budget")
                cands_graded, cands_rest = cands[: max(0, rem)], cands[max(0, rem) :]
            else:
                cands_graded, cands_rest = cands, []
        else:
            cands_graded, cands_rest = cands, []
        if self.executor is not None:
            graded = await self.executor.map(lambda ev: self._grade(query, ev, ctx), cands_graded, concurrency=conc)
        else:
            graded = await gather_limited((self._grade(query, ev, ctx) for ev in cands_graded), conc, return_exceptions=False)
        rows: list[tuple[float, int, Evidence, str]] = []
        for i, (ev, g) in enumerate(zip(cands_graded, graded)):
            score, rationale = g if isinstance(g, tuple) else (-1.0, "")
            rows.append((score, i, ev, rationale))
        for j, ev in enumerate(cands_rest):
            rows.append((-1.0, len(cands_graded) + j, ev, ""))
        if not any(s >= 0 for s, _i, _ev, _r in rows):
            return cands[: k or len(cands)]
        self.applied = True
        rows.sort(key=lambda r: (-r[0], r[1]))
        out = [_with_rerank(ev, score=max(0.0, s), rank=pos + 1, method=self.name, rerank_rationale=r, graded=s >= 0) for pos, (s, _i, ev, r) in enumerate(rows)]
        return out[: k or len(out)]


class CrossEncoderReranker:
    """Adapter for a cross-encoder (``predict(list[(query, passage)]) -> scores``)."""

    name = "cross_encoder"

    def __init__(self, model: Any = None, *, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", passage_chars: int = 2000) -> None:
        if model is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: WPS433
            except ImportError as exc:  # pragma: no cover
                raise ImportError("sentence-transformers is not installed") from exc
            model = CrossEncoder(model_name)
        self._model = model
        self.model_name = model_name
        self.passage_chars = passage_chars
        self.usage = Usage()
        self.warnings: list[str] = []
        self.applied = False

    async def rerank(self, query: str, candidates: Sequence[Evidence], *, k: int | None = None, ctx: RunContext | None = None) -> list[Evidence]:
        cands = list(candidates)
        self.applied = False
        if not cands:
            return []
        pairs = [(query, c.content[: self.passage_chars]) for c in cands]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        self.applied = True
        rows = sorted(zip([float(s) for s in scores], range(len(cands)), cands), key=lambda r: (-r[0], r[1]))
        out = [_with_rerank(ev, score=s, rank=pos + 1, method=self.name) for pos, (s, _i, ev) in enumerate(rows)]
        return out[: k or len(out)]


class NoopReranker:
    """Passthrough reranker."""

    name = "none"

    def __init__(self) -> None:
        self.usage = Usage()
        self.warnings: list[str] = []
        self.applied = False

    async def rerank(self, query: str, candidates: Sequence[Evidence], *, k: int | None = None, ctx: RunContext | None = None) -> list[Evidence]:
        return list(candidates)[: k or len(candidates)]


def make_reranker(mode: str, llm: BaseLLMClient | None = None, *, window: int = 20, stride: int = 10, passage_chars: int = 1200, concurrency: int = 8, executor: Executor | None = None, extra_system: Sequence[str] = (), cross_encoder: Any = None) -> Reranker:
    """Factory keyed on the plan's ``rerank`` mode."""
    if mode == "listwise":
        return LLMListwiseReranker(llm, window=window, stride=stride, passage_chars=passage_chars, extra_system=extra_system)
    if mode == "pointwise":
        return LLMPointwiseReranker(llm, concurrency=concurrency, executor=executor, passage_chars=passage_chars, extra_system=extra_system)
    if mode == "cross_encoder":
        return CrossEncoderReranker(cross_encoder)
    return NoopReranker()
