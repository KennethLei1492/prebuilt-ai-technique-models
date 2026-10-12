"""Stampede prevention: XFetch probabilistic early expiration and single-flight coalescing.

* :func:`xfetch_should_refresh` implements Vattani, Chierichetti and
  Lowenstein (VLDB 2015): a reader recomputes early when
  ``now - delta * beta * ln(rand()) >= expiry`` where ``delta`` is the measured
  recompute cost stored with the entry and ``beta`` (default 1.0) tunes
  eagerness. Expensive entries refresh earlier; no locks are needed.
* :class:`SingleFlight` coalesces concurrent misses for the same key so one
  caller performs the expensive call while the others await its result.

Both take injectable clocks / random sources so they are deterministic in tests.
"""
from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


def xfetch_should_refresh(
    now: float,
    expires_at: Optional[float],
    recompute_cost_s: float,
    *,
    beta: float = 1.0,
    rand: Callable[[], float] = random.random,
) -> bool:
    """Return True when the caller should recompute the entry ahead of expiry.

    ``rand`` returns a float in ``(0, 1]``; ``ln(rand)`` is therefore <= 0, so
    the effective expiry is pulled earlier by a random amount proportional to
    the recompute cost. ``beta=0`` disables early refresh.
    """
    if expires_at is None or beta <= 0 or recompute_cost_s <= 0:
        return expires_at is not None and now >= expires_at
    r = rand()
    r = min(max(r, 1e-12), 1.0)
    return now - recompute_cost_s * beta * math.log(r) >= expires_at


class SingleFlight:
    """Per-key in-flight registry: identical concurrent misses share one call."""

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self.coalesced = 0
        self.leaders = 0

    def inflight(self, key: str) -> bool:
        return key in self._inflight

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Run ``fn`` once per key; returns ``(result, coalesced)``.

        ``coalesced`` is True for followers that received the leader's result.
        Exceptions raised by the leader propagate to every waiter.
        """
        fut = self._inflight.get(key)
        if fut is not None:
            self.coalesced += 1
            return await asyncio.shield(fut), True
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._inflight[key] = fut
        self.leaders += 1
        try:
            result = await fn()
        except BaseException as exc:  # noqa: BLE001 - propagate to followers too
            if not fut.done():
                fut.set_exception(exc)
                # Mark the exception as retrieved: when no follower awaited the future, asyncio
                # would otherwise log "Future exception was never retrieved" at garbage collection.
                fut.exception()
            raise
        else:
            if not fut.done():
                fut.set_result(result)
            return result, False
        finally:
            self._inflight.pop(key, None)


__all__ = ["SingleFlight", "xfetch_should_refresh"]
