"""Workload governor — keep foreground chat responsive while four background
loops (dream, nightly, single-study, deep-dive) plus heavy endpoints all share
one 16GB card.

The problem isn't a single op; it's contention — several background cycles
hitting the GPU/CPU at the same moment as an active chat. A plain Semaphore(1)
around everything would be worse: a chat would have to WAIT for a running deep
dive to release it. So this governor gives foreground priority instead:

  • FOREGROUND (chat, user-initiated heavy endpoints) never waits on background;
    it just marks "busy" so background work holds off.
  • BACKGROUND serializes to ONE at a time AND defers while any foreground work
    is in flight (no NEW background cycle starts during a chat). Running work
    can't be preempted, but not piling on is what bounds latency.

Default-ON; ERIS_GOVERNOR=off makes both context managers inert (the previous
behavior, for troubleshooting).
"""
from __future__ import annotations
from contextlib import asynccontextmanager
import asyncio
import os


def _enabled() -> bool:
    return os.environ.get("ERIS_GOVERNOR", "on").strip().lower() not in ("0", "off", "false", "no")


class Governor:
    def __init__(self):
        self._bg_lock = asyncio.Lock()          # at most one background op at a time
        self._fg = 0                            # active foreground count
        self._idle = asyncio.Event()
        self._idle.set()
        self._enabled = _enabled()

    @asynccontextmanager
    async def foreground(self):
        """Wrap chat / user-initiated heavy work. Never blocks on background;
        signals activity so background defers."""
        if not self._enabled:
            yield
            return
        self._fg += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._fg -= 1
            if self._fg <= 0:
                self._fg = 0
                self._idle.set()

    @asynccontextmanager
    async def background(self):
        """Wrap a background cycle. Waits until no foreground is active, then
        takes the single background slot; re-checks after acquiring so a chat
        that arrived mid-wait still wins."""
        if not self._enabled:
            yield
            return
        while True:
            await self._idle.wait()
            async with self._bg_lock:
                if self._fg == 0:
                    yield
                    return
            # A foreground turn appeared while we waited for the slot — yield the
            # loop and wait for idle again before retrying.
            await asyncio.sleep(0)

    @property
    def foreground_active(self) -> bool:
        return self._fg > 0
