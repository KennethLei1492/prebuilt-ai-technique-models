"""Component registry.

Models register themselves with the :func:`register` decorator so that
orchestrators and routers can discover them by name without importing each
package explicitly. :func:`discover` imports the six built-in model packages
if they are installed.
"""
from __future__ import annotations

import importlib
import logging
from typing import Iterable, Type

from .component import ComponentSpec, PipelineComponent

log = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[PipelineComponent]] = {}

BUILTIN_PACKAGES: tuple[str, ...] = (
    "hybrid_retrieval",
    "task_decomposition",
    "answer_verification",
    "parallelization",
    "caching",
    "pipeline_rag",
)


def register(cls: Type[PipelineComponent]) -> Type[PipelineComponent]:
    spec = getattr(cls, "spec", None)
    if not isinstance(spec, ComponentSpec):
        raise TypeError(f"{cls.__name__} must define a ComponentSpec as 'spec'")
    _REGISTRY[spec.name] = cls
    return cls


def get(name: str) -> Type[PipelineComponent]:
    if name not in _REGISTRY:
        discover()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"no component registered as '{name}'; available: {sorted(_REGISTRY)}") from exc


def available() -> list[ComponentSpec]:
    return [cls.spec for cls in _REGISTRY.values()]


def discover(packages: Iterable[str] = BUILTIN_PACKAGES) -> list[str]:
    loaded: list[str] = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            loaded.append(pkg)
        except ImportError as exc:
            log.debug("component package %s not importable: %s", pkg, exc)
    return loaded
