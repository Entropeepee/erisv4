"""Resolve the host uvicorn binds to.

Default is 127.0.0.1 (localhost-only) — the box is NOT reachable from the LAN or your phone unless
you opt in explicitly. Reaching her remotely should go over an AUTHENTICATED tunnel (Tailscale /
WireGuard), not a raw public bind. An externally-reachable bind is therefore REFUSED unless
ERIS_AUTH_TOKEN is set, because an unauthenticated externally-bound server is owned by anyone on the
network (the surface includes /sandbox file read/write/exec and a live stream of the cognitive field).

Kept in its own tiny module (only depends on `os`) so it is unit-testable without importing the heavy
server app, which builds an orchestrator at import time.

NOTE: a token is necessary but not yet sufficient for safe external exposure — the WebSocket
endpoints bypass the HTTP auth gate until that fix lands, so even with a token a raw external bind
still leaks /ws/field. Prefer the tunnel.
"""
import os

_LOCALHOST = {"127.0.0.1", "localhost", "::1", ""}


def resolve_bind_host() -> str:
    """Return the bind host. 127.0.0.1 by default; ERIS_BIND_HOST overrides; a non-localhost host
    raises SystemExit unless ERIS_AUTH_TOKEN is set."""
    host = (os.environ.get("ERIS_BIND_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    if host not in _LOCALHOST and not os.environ.get("ERIS_AUTH_TOKEN", "").strip():
        raise SystemExit(
            f"[eris] REFUSING to bind {host} (externally reachable) with no ERIS_AUTH_TOKEN set — "
            "an unauthenticated external bind lets anyone on the network read/write files on the box "
            "and exfiltrate your IP. Set ERIS_AUTH_TOKEN (and prefer an authenticated tunnel like "
            "Tailscale/WireGuard over a raw LAN bind), or use the default 127.0.0.1.")
    return host
