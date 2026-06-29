"""Request-size caps + a simple per-client rate limiter for the reachable endpoints, so a single
caller can't exhaust RAM/CPU/disk with an oversized body or a request flood.

Caps are ON by default (generous, env-overridable). The rate limiter is OPT-IN
(ERIS_RATE_PER_MIN, default 0 = off) so default local/cockpit use is unchanged. Kept in its own
module so it's unit-testable without importing the heavy server app.
"""
import os
import time
import threading
from typing import Optional


def char_cap(env_key: str, default_chars: int) -> int:
    """Max allowed characters for a text field (env-overridable)."""
    try:
        return max(1, int(os.environ.get(env_key, str(default_chars))))
    except ValueError:
        return default_chars


def byte_cap(env_key: str, default_mb: int) -> int:
    """Max allowed bytes for a request body, from an MB env var (env-overridable)."""
    try:
        return max(1, int(os.environ.get(env_key, str(default_mb)))) * 1024 * 1024
    except ValueError:
        return default_mb * 1024 * 1024


class RateLimiter:
    """Fixed-window, per-client rate limiter. Thread-safe. `per_min <= 0` disables it entirely
    (allow() always True) — the default, so local use is unaffected until the owner opts in."""

    def __init__(self, per_min: int, window: float = 60.0):
        self.per_min = per_min
        self.window = window
        self._hits: dict = {}
        self._lock = threading.Lock()

    def allow(self, client: str, now: Optional[float] = None) -> bool:
        if self.per_min <= 0:
            return True
        now = time.time() if now is None else now
        with self._lock:
            q = self._hits.setdefault(client or "anon", [])
            cutoff = now - self.window
            q[:] = [t for t in q if t >= cutoff]      # drop hits outside the window
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True


def client_of(request) -> str:
    """Best-effort client identity for rate limiting (the peer IP)."""
    return getattr(getattr(request, "client", None), "host", "") or "anon"
