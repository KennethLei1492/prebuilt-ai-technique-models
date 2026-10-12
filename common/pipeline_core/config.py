"""Environment-driven settings and the LLM factory.

Environment variables
---------------------
``PIPELINE_MODEL``          default ``claude-opus-5-5``
``PIPELINE_EFFORT``         low | medium | high | xhigh | max (default ``high``)
``PIPELINE_OFFLINE``        ``1`` forces the deterministic :class:`MockLLM`
``PIPELINE_USE_FALLBACKS``  ``0`` disables server-side refusal fallbacks
``PIPELINE_TIMEOUT_S``      per-request timeout (default 600)
``PIPELINE_MAX_RETRIES``    SDK retries (default 3)
``ANTHROPIC_API_KEY``       credential; an ``ant auth login`` profile also works
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .llm import DEFAULT_MODEL, AnthropicLLM, BaseLLMClient, Effort, MockLLM

log = logging.getLogger(__name__)
_warned_offline = False


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_MODEL
    effort: Effort = "high"
    offline: bool = False
    use_fallbacks: bool = True
    timeout_s: float = 600.0
    max_retries: int = 3
    api_key: Optional[str] = Field(default=None, repr=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        env = os.environ
        data: dict[str, Any] = {
            "model": env.get("PIPELINE_MODEL", DEFAULT_MODEL),
            "effort": env.get("PIPELINE_EFFORT", "high"),
            "offline": env.get("PIPELINE_OFFLINE", "0").lower() in ("1", "true", "yes"),
            "use_fallbacks": env.get("PIPELINE_USE_FALLBACKS", "1").lower() not in ("0", "false", "no"),
            "timeout_s": float(env.get("PIPELINE_TIMEOUT_S", "600")),
            "max_retries": int(env.get("PIPELINE_MAX_RETRIES", "3")),
            "api_key": env.get("ANTHROPIC_API_KEY") or None,
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def credentials_available() -> bool:
    """True when the SDK will be able to resolve a credential."""
    env = os.environ
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_PROFILE"):
        return True
    if env.get("ANTHROPIC_IDENTITY_TOKEN_FILE") or env.get("ANTHROPIC_IDENTITY_TOKEN"):
        return True
    cfg = Path(os.path.expanduser("~")) / ".config" / "anthropic"
    return cfg.exists() and any(cfg.iterdir())


def make_llm(settings: Settings | None = None, **overrides: Any) -> BaseLLMClient:
    """Return a production client when credentials exist, otherwise a MockLLM.

    The fallback is deliberate: every component, demo and test must run in a
    credential-less CI environment. A single warning is logged the first time
    the mock is selected implicitly.
    """
    global _warned_offline
    s = settings or Settings.from_env(**overrides)
    if s.offline:
        return MockLLM(model=s.model, effort=s.effort)
    if not credentials_available():
        if not _warned_offline:
            log.warning("no Anthropic credentials found; using deterministic MockLLM (set ANTHROPIC_API_KEY for real inference)")
            _warned_offline = True
        return MockLLM(model=s.model, effort=s.effort)
    return AnthropicLLM(
        s.model,
        effort=s.effort,
        api_key=s.api_key,
        max_retries=s.max_retries,
        timeout=s.timeout_s,
        use_fallbacks=s.use_fallbacks,
    )
