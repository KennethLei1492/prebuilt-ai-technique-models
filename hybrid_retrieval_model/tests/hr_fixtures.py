"""Shared test helpers for the hybrid retrieval package (imported via conftest.py).

``OnlineScriptedLLM`` is a scripted :class:`BaseLLMClient` that is **not** a
``MockLLM`` subclass, so ``RunContext.offline`` is ``False`` and the component
takes the LLM-planned path (spec 3.1.2 reserves the heuristic planner for
``ctx.offline``). It replays queued responses exactly like ``ScriptedLLM``.
"""
from __future__ import annotations

from typing import Any, Sequence, Union

from pipeline_core import BaseLLMClient, LLMRequest, LLMResponse, ScriptedLLM


class OnlineScriptedLLM(BaseLLMClient):
    """Deterministic scripted client that the core treats as *online* (not a ``MockLLM``)."""

    name = "scripted-online"

    def __init__(self, responses: Sequence[Union[str, dict, list, LLMResponse]], *, loop: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self._inner = ScriptedLLM(responses, loop=loop)

    @property
    def calls(self) -> list[LLMRequest]:
        return self._inner.calls

    async def _complete(self, request: LLMRequest) -> LLMResponse:
        return await self._inner._complete(request)


__all__ = ["OnlineScriptedLLM"]
