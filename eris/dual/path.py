"""DualPath — run every novel subsystem on trial above a proven SOTA floor.

The generalization of Eris's flag+ruler pattern: one wrapper holds a
(novel, traditional) pair, a mode, and an optional independent arbiter. The floor
always answers, so Eris never depends on an unproven piece; in SHADOW mode every
turn quietly produces the data that teaches the novel path whether it's winning,
losing, or merely copying the floor.

Build it once; every later subsystem (retrieval, chunking, vision, …) reuses it.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Optional


class Mode(Enum):
    TRADITIONAL_ONLY = "traditional_only"   # floor only — the safe default per subsystem
    SHADOW = "shadow"                       # traditional authoritative; novel observed + scored
    NOVEL_PRIMARY = "novel_primary"         # novel leads; traditional catches empty/low-confidence
    NOVEL_ONLY = "novel_only"               # novel only (post-proof / experiments)

    @classmethod
    def parse(cls, value: str, default: "Mode" = None) -> "Mode":
        v = (value or "").strip().lower()
        for m in cls:
            if m.value == v:
                return m
        return default or cls.TRADITIONAL_ONLY


class DualPath:
    """Holds a proven floor and an on-trial novel path for ONE capability.

    novel, traditional : callables (query, **kw) -> Result  (same output type).
    arbiter            : optional Arbiter (§3) — scores task success, NOT agreement.
    logger             : optional DivergenceLog (§4).
    accept             : optional predicate(result) -> bool for NOVEL_PRIMARY fallback;
                         defaults to "non-empty result".
    """

    def __init__(self, novel: Callable, traditional: Callable,
                 mode: Mode = Mode.TRADITIONAL_ONLY, arbiter=None, logger=None,
                 name: str = "unnamed", accept: Optional[Callable[[Any], bool]] = None):
        self.novel = novel
        self.traditional = traditional
        self.mode = mode
        self.arbiter = arbiter
        self.logger = logger
        self.name = name
        self._accept = accept

    def run(self, query, *, gold=None, **kw):
        if self.mode is Mode.TRADITIONAL_ONLY:
            return self.traditional(query, **kw)
        if self.mode is Mode.NOVEL_ONLY:
            return self.novel(query, **kw)
        if self.mode is Mode.SHADOW:
            t = self.traditional(query, **kw)
            # The novel path failing must NEVER break the turn — the floor stands.
            try:
                n = self.novel(query, **kw)
                if self.logger is not None:
                    self.logger.record(self.name, query, t, n, self.arbiter, gold=gold, kw=kw)
            except Exception as e:                      # noqa: BLE001 — observe, don't raise
                if self.logger is not None:
                    self.logger.record_error(self.name, query, e)
            return t                                    # floor is authoritative in shadow
        # NOVEL_PRIMARY
        try:
            n = self.novel(query, **kw)
        except Exception:                               # noqa: BLE001
            return self.traditional(query, **kw)
        return n if self._acceptable(n) else self.traditional(query, **kw)

    def _acceptable(self, result) -> bool:
        """Confidence/empties gate for NOVEL_PRIMARY. Custom predicate if given;
        else fall back to the floor when the novel result is empty/low-confidence."""
        if self._accept is not None:
            try:
                return bool(self._accept(result))
            except Exception:
                return False
        recs = getattr(result, "records", result)
        return bool(recs)
