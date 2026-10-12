"""Exception hierarchy shared by every pipeline component.

Components raise these instead of bare exceptions so that orchestrators can
distinguish recoverable conditions (budget exhaustion, refusals, truncation)
from programming errors.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ConfigurationError(PipelineError):
    """A component was configured inconsistently (bad weights, missing index, ...)."""


class LLMError(PipelineError):
    """The language model call failed in a way the SDK did not already classify."""


class LLMRefusalError(LLMError):
    """The model declined the request (stop_reason == "refusal")."""

    def __init__(self, message: str, category: str | None = None, explanation: str | None = None):
        super().__init__(message)
        self.category = category
        self.explanation = explanation


class LLMTruncatedError(LLMError):
    """The response hit max_tokens before completing structured output."""


class StructuredOutputError(LLMError):
    """The model's output could not be parsed or validated against the requested schema."""


class BudgetExceededError(PipelineError):
    """A cost, call-count or latency constraint was exhausted."""


class VerificationFailedError(PipelineError):
    """An answer failed verification and no acceptable correction was found."""


class PlanValidationError(PipelineError):
    """A generated plan is structurally invalid (cycle, missing dependency, depth bound)."""


class CacheError(PipelineError):
    """A cache backend failed; callers should treat this as a miss."""


class RetrievalError(PipelineError):
    """Retrieval failed (empty index, embedder failure, filter error)."""
