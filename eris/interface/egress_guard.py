"""
eris/interface/egress_guard.py
==============================
Layer 3 of the §5 sovereignty defense-in-depth: an OS-level egress self-check for
the truly sovereign worker process.

The sovereign worker is meant to run behind a Windows Firewall outbound-block on
its venv `python.exe`. This module *verifies* that block at startup: it attempts
an outbound TCP connection to an external IP and treats SUCCESS as a fatal
misconfiguration — if the sovereign process can reach the open internet, the
isolation has failed and IP-sensitive work must not run.

Loopback is unaffected, so a local Ollama/vLLM call still works while external
egress is blocked. The probe target/port and the connect function are injectable
so this is unit-testable with no real network.
"""
from __future__ import annotations

import os
import socket
from typing import Callable, Optional


class EgressNotBlockedError(RuntimeError):
    """Raised by assert_isolated() when the sovereign worker CAN reach an external
    host — the firewall isolation is not in place. Fail closed."""


# A well-known external IP:port to probe. We never send data; we only see whether
# the TCP handshake completes. Default is a public DNS resolver on 53.
_DEFAULT_PROBE_HOST = os.environ.get("ERIS_EGRESS_PROBE_HOST", "1.1.1.1")
_DEFAULT_PROBE_PORT = int(os.environ.get("ERIS_EGRESS_PROBE_PORT", "53"))


def egress_reachable(host: str = _DEFAULT_PROBE_HOST, port: int = _DEFAULT_PROBE_PORT,
                     timeout: float = 2.0,
                     connect: Optional[Callable[[str, int, float], bool]] = None) -> bool:
    """Return True iff an outbound TCP connection to (host, port) succeeds — i.e.
    egress is OPEN. False means the connection was refused/blocked/timed out, i.e.
    egress is BLOCKED (the desired state for the sovereign worker).

    `connect` is injectable for tests; the default does a real, dataless TCP probe."""
    probe = connect or _tcp_connects
    try:
        return bool(probe(host, port, timeout))
    except Exception:
        # Any failure to even attempt the probe is treated as 'blocked' (fail closed
        # toward isolation — we never report egress as open on an inconclusive probe).
        return False


def _tcp_connects(host: str, port: int, timeout: float) -> bool:
    """Real probe: open a TCP socket, no payload. True if the handshake completes."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def isolation_status(**kw) -> str:
    """'blocked' (sovereign-safe) or 'open' (egress reachable). Convenience for logs
    and the §9 test that asserts the sovereign worker reports 'blocked'."""
    return "open" if egress_reachable(**kw) else "blocked"


def assert_isolated(**kw) -> None:
    """Fail-closed startup check for the sovereign worker: raise EgressNotBlockedError
    if external egress is reachable. Loopback is untouched, so local model calls still
    work. Call once at sovereign-worker startup (per §5.3).

    Honors ERIS_SOVEREIGN_REQUIRE_ISOLATION=0 to skip (dev only) — but logs loudly so
    a skipped check is never silent."""
    if os.environ.get("ERIS_SOVEREIGN_REQUIRE_ISOLATION", "1") == "0":
        print("[egress_guard] WARNING: isolation check SKIPPED "
              "(ERIS_SOVEREIGN_REQUIRE_ISOLATION=0) — sovereign egress is NOT verified.")
        return
    if egress_reachable(**kw):
        raise EgressNotBlockedError(
            "Sovereign worker can reach the external network — firewall isolation is "
            "NOT in place. Refusing to run IP-sensitive work (fail-closed). Block "
            "outbound TCP on this venv python.exe, or set "
            "ERIS_SOVEREIGN_REQUIRE_ISOLATION=0 to override (dev only).")
