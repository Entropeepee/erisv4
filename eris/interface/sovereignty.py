"""
eris/interface/sovereignty.py
=============================
Sovereignty routing for the Contractor Layer (contractor_gateway_layer_spec §5).

A hard boundary, enforced in ROUTING, between two classes of LLM call:

  • SOVEREIGN  — IP-sensitive work. Runs ONLY on a direct local backend
                 (Ollama / vLLM). Structurally incapable of naming a gateway or
                 cloud model-group. If a sovereign call is ever handed a
                 non-local backend, we FAIL CLOSED — raise, never silently
                 downgrade to a cloud route.
  • OPEN       — non-sensitive work. May route across the cost-tiered pool
                 (local → free cloud → frontier-synthesis → cheap-paid) via the
                 gateway.

This module is layer 1 of the defense-in-depth in §5 (routing). Layer 2 is the
gateway policy (no sovereign→cloud route exists in LiteLLM); layer 3 is the OS
egress block (egress_guard). All three are required; this is the one that lives
in erisv4's own process and is unit-testable with no network.

Pure and dependency-light on purpose: the sovereign path must not depend on the
gateway, the SDK, or anything that could pull in egress.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class Sensitivity(str, Enum):
    """The sovereignty tag every routed LLM call carries. `str` mixin so it
    serializes/compares as its value ('sovereign'/'open') in logs and configs."""
    SOVEREIGN = "sovereign"   # IP-sensitive → local only, no egress
    OPEN = "open"             # non-sensitive → may use the gateway/cloud pool

    @classmethod
    def coerce(cls, value: Any, default: "Sensitivity" = None) -> "Sensitivity":
        """Parse a tag from a str/enum; fail CLOSED to SOVEREIGN on anything
        unrecognized (an unknown tag must never accidentally open an egress path)."""
        if isinstance(value, cls):
            return value
        if value is None:
            return default if default is not None else cls.SOVEREIGN
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.SOVEREIGN


class SovereigntyError(RuntimeError):
    """Raised when a SOVEREIGN call is routed to a non-local backend. This is a
    fail-closed safety stop, NOT a recoverable error — callers must fix the route,
    never catch-and-downgrade to cloud."""


# Backend identities that are local-only (no egress). Matched against the
# backend's `.name`. The gateway/cloud backends use distinct names ("gateway-*",
# "openai", "anthropic", "gemini", "claude-agent-sdk") and are NEVER in this set.
_LOCAL_BACKEND_NAMES = frozenset({"ollama", "vllm", "local", "local-mirror"})


def is_local_backend(backend: Any) -> bool:
    """True iff `backend` is a direct local backend with no egress surface.

    Conservative / fail-closed: a backend with no recognizable local name is
    treated as NON-local (so an unknown backend can never satisfy a sovereign
    call). A local-named backend that points at a non-loopback base_url is also
    rejected — 'ollama' aimed at a remote host is not sovereign-safe.
    """
    name = str(getattr(backend, "name", "")).strip().lower()
    if name not in _LOCAL_BACKEND_NAMES:
        return False
    base_url = str(getattr(backend, "base_url", "") or "")
    if base_url and not _is_loopback_url(base_url):
        return False
    return True


def _is_loopback_url(url: str) -> bool:
    """Loopback / on-box host check for a backend base_url. Empty host (relative)
    counts as local. Anything resolving to an external host does not."""
    from urllib.parse import urlparse
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return True
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0",
                    "host.docker.internal"} or host.endswith(".local")


def assert_backend_allowed(sensitivity: Any, backend: Any) -> None:
    """Gate a chosen backend against the call's sovereignty tag — the §5 layer-1
    enforcement. SOVEREIGN calls may use ONLY a local backend; anything else
    raises SovereigntyError (fail closed). OPEN calls may use any backend.

    Call this at the point a backend is selected, BEFORE generate()."""
    sens = Sensitivity.coerce(sensitivity)
    if sens is Sensitivity.SOVEREIGN and not is_local_backend(backend):
        name = getattr(backend, "name", backend.__class__.__name__)
        raise SovereigntyError(
            f"SOVEREIGN call routed to non-local backend {name!r} — refusing "
            f"(fail-closed). IP-sensitive work runs only on the direct local model.")


def select_sovereign_backend(backends) -> Any:
    """Pick the local backend for a sovereign call from a list, or raise if none
    exists (fail closed — never fall through to a cloud backend)."""
    for b in backends:
        if is_local_backend(b):
            return b
    raise SovereigntyError(
        "No local backend available for a SOVEREIGN call — refusing to route "
        "(fail-closed). A direct Ollama/vLLM backend must be configured.")
