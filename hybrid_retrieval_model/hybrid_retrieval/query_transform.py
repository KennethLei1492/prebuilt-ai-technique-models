"""Query-side transforms, router-gated and drift-guarded.

LLM query expansion adds 10-15 nDCG points to lexical search in zero-shot
settings but costs seconds per query and can steer retrieval toward
plausible-but-wrong content (Gao et al. 2023; Wang et al. 2023). The planner
therefore enables these transforms only for short or exploratory queries, and
every generated text passes a *drift guard*: it must share at least
``min_overlap`` of the query's content tokens, otherwise it is dropped and a
warning is recorded.

All transforms accept a ``RunContext`` for budgeted, traced calls; without one
they use their own client and accumulate ``self.usage``. When the context's
latency, call or cost budget is exhausted (``budget_exhausted``), the transform
is skipped with a warning and the original query is used (spec 3.2); nothing raises.
"""
from __future__ import annotations

from typing import Optional, Sequence

from pipeline_core import BaseLLMClient, LLMRequest, Message, Usage
from pipeline_core.component import RunContext
from pipeline_core.errors import BudgetExceededError, ConfigurationError, LLMError, StructuredOutputError

from .plan import budget_exhausted
from .prompts import HYDE_PROMPT, KEYWORDS_PROMPT, MULTI_QUERY_PROMPT, QUERY2DOC_PROMPT, HyDEOutput, KeywordsOutput, MultiQueryOutput, Query2DocOutput
from .sparse import DEFAULT_STOPWORDS, Tokenizer, whole_tokens

_TOK = Tokenizer()


def content_overlap(query: str, text: str, tokenizer: Tokenizer = _TOK) -> float:
    """Fraction of the query's content tokens that also occur in ``text`` (drift guard)."""
    q = tokenizer.content_tokens(query)
    if not q:
        return 1.0
    t = tokenizer.content_tokens(text)
    return len(q & t) / len(q)


class _LLMTransform:
    def __init__(self, llm: BaseLLMClient | None = None, *, min_overlap: float = 0.1, effort: str = "low", model: str | None = None, extra_system: Sequence[str] = ()) -> None:
        self.llm = llm
        self.min_overlap = min_overlap
        self.effort = effort
        self.model = model
        self.extra_system = [s for s in extra_system if s]
        self.usage = Usage()
        self.warnings: list[str] = []

    def _warn(self, ctx: RunContext | None, msg: str) -> None:
        """Record a warning on the transform (direct API) and on the context when one exists."""
        self.warnings.append(msg)
        if ctx is not None:
            ctx.warn(msg)

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

    def _request(self, rendered, *, max_tokens: int, tag: str) -> LLMRequest:
        return LLMRequest(system_stable=rendered.system_blocks, messages=[Message(role="user", content=rendered.user)], output_schema=rendered.output_schema, effort=self.effort, model=self.model, tags=["query_transform", tag], max_tokens=max_tokens)  # type: ignore[arg-type]

    def _guard(self, query: str, texts: Sequence[str], ctx: RunContext | None, what: str) -> list[str]:
        kept: list[str] = []
        dropped = 0
        seen = {query.strip().lower()}
        for t in texts:
            s = (t or "").strip()
            key = s.lower()
            if not s or key in seen:
                continue
            if content_overlap(query, s) < self.min_overlap:
                dropped += 1
                continue
            seen.add(key)
            kept.append(s)
        if dropped:
            self._warn(ctx, f"{what}: dropped {dropped} generated text(s) that drifted from the query")
        return kept


class MultiQueryExpander(_LLMTransform):
    """Generate ``n`` paraphrases; the retriever runs each and fuses the lists with RRF."""

    def __init__(self, llm: BaseLLMClient | None = None, *, n: int = 3, **kw) -> None:
        super().__init__(llm, **kw)
        self.n = max(1, n)

    async def expand(self, query: str, ctx: RunContext | None = None) -> list[str]:
        rp = MULTI_QUERY_PROMPT.render(extra_system=self.extra_system, query=query, n=self.n)
        try:
            resp = await self._call(self._request(rp, max_tokens=600, tag="multi_query"), ctx, "transform.multi_query")
            out = MultiQueryOutput.model_validate(resp.parsed)
        except BudgetExceededError as exc:
            self._warn(ctx, f"multi-query expansion skipped: {exc}; using the original query only")
            return []
        except (LLMError, StructuredOutputError, ValueError) as exc:
            self._warn(ctx, f"multi-query expansion failed ({type(exc).__name__}); using the original query only")
            return []
        return self._guard(query, out.queries, ctx, "multi_query")[: self.n]


class HyDE(_LLMTransform):
    """Hypothetical Document Embeddings: generate ``n`` passages, embed, average with the query vector (Gao et al. 2023)."""

    def __init__(self, llm: BaseLLMClient | None = None, *, n: int = 4, **kw) -> None:
        super().__init__(llm, **kw)
        self.n = max(1, n)

    async def generate(self, query: str, ctx: RunContext | None = None) -> list[str]:
        rp = HYDE_PROMPT.render(extra_system=self.extra_system, query=query, n=self.n)
        try:
            resp = await self._call(self._request(rp, max_tokens=1200, tag="hyde"), ctx, "transform.hyde")
            out = HyDEOutput.model_validate(resp.parsed)
        except BudgetExceededError as exc:
            self._warn(ctx, f"HyDE skipped: {exc}; dense leg uses the query embedding only")
            return []
        except (LLMError, StructuredOutputError, ValueError) as exc:
            self._warn(ctx, f"HyDE generation failed ({type(exc).__name__}); dense leg uses the query embedding only")
            return []
        return self._guard(query, out.passages, ctx, "hyde")[: self.n]


class Query2Doc(_LLMTransform):
    """Lexical expansion: ``query x repeat`` + pseudo-document (Wang et al. 2023)."""

    def __init__(self, llm: BaseLLMClient | None = None, *, repeat: int = 5, max_tokens: int = 128, **kw) -> None:
        super().__init__(llm, **kw)
        self.repeat = max(1, repeat)
        self.max_tokens = max_tokens

    @staticmethod
    def compose(query: str, pseudo_document: str, repeat: int = 5) -> str:
        """Pure composition rule used by the expander and by tests."""
        return " ".join([query.strip()] * repeat + [pseudo_document.strip()]).strip()

    async def expand(self, query: str, ctx: RunContext | None = None) -> str:
        rp = QUERY2DOC_PROMPT.render(extra_system=self.extra_system, query=query)
        try:
            resp = await self._call(self._request(rp, max_tokens=max(64, self.max_tokens * 2), tag="query2doc"), ctx, "transform.query2doc")
            out = Query2DocOutput.model_validate(resp.parsed)
        except BudgetExceededError as exc:
            self._warn(ctx, f"Query2doc skipped: {exc}; lexical leg uses the original query")
            return query
        except (LLMError, StructuredOutputError, ValueError) as exc:
            self._warn(ctx, f"Query2doc failed ({type(exc).__name__}); lexical leg uses the original query")
            return query
        kept = self._guard(query, [out.pseudo_document], ctx, "query2doc")
        if not kept:
            return query
        words = kept[0].split()
        pseudo = " ".join(words[: self.max_tokens])
        return self.compose(query, pseudo, self.repeat)


class KeywordExtractor(_LLMTransform):
    """Key terms for the sparse leg: heuristic by default (free), LLM when ``use_llm=True``."""

    def __init__(self, llm: BaseLLMClient | None = None, *, use_llm: bool = False, max_terms: int = 10, **kw) -> None:
        super().__init__(llm, **kw)
        self.use_llm = use_llm
        self.max_terms = max_terms

    def extract_heuristic(self, query: str) -> list[str]:
        """Identifiers first, then content words (stopwords removed), preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for t in whole_tokens(query):
            if t in DEFAULT_STOPWORDS or len(t) <= 1 or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out[: self.max_terms]

    async def extract(self, query: str, ctx: RunContext | None = None) -> list[str]:
        if not self.use_llm:
            return self.extract_heuristic(query)
        rp = KEYWORDS_PROMPT.render(extra_system=self.extra_system, query=query)
        try:
            resp = await self._call(self._request(rp, max_tokens=300, tag="keywords"), ctx, "transform.keywords")
            out = KeywordsOutput.model_validate(resp.parsed)
        except BudgetExceededError as exc:
            self._warn(ctx, f"LLM keyword extraction skipped: {exc}; using heuristic terms")
            return self.extract_heuristic(query)
        except (LLMError, StructuredOutputError, ValueError) as exc:
            self._warn(ctx, f"keyword extraction failed ({type(exc).__name__}); using heuristic terms")
            return self.extract_heuristic(query)
        terms = [t.strip() for t in out.keywords if t and t.strip()]
        return (terms or self.extract_heuristic(query))[: self.max_terms]
