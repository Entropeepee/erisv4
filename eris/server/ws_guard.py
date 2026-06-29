"""WebSocket access helpers — kept out of app.py so they are unit-testable without importing the
heavy server app (which builds an orchestrator at import).

The HTTP auth middleware is a Starlette BaseHTTPMiddleware, which NEVER runs for the websocket scope
(`scope["type"] == "websocket"` is forwarded straight to the app). So /ws and /ws/field must check
the token IN-ENDPOINT, before accept(), or the access gate is bypassed entirely for both sockets.
"""
import re
from typing import Optional

from eris.server.auth import token_ok

_FIELD_AGENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def ws_authorized(expected_token: str, websocket) -> bool:
    """True iff the websocket handshake carries the expected token (header `x-eris-token`, `?token=`,
    or the `eris_token` cookie the HTTP gate sets), OR no token is configured (gate disabled — local
    use is unchanged). Call this BEFORE websocket.accept()."""
    return token_ok(expected_token,
                    header=websocket.headers.get("x-eris-token"),
                    query=websocket.query_params.get("token"),
                    cookie=websocket.cookies.get("eris_token"))


def sanitize_field_agent(agent: Optional[str], allowlist: str = "") -> str:
    """Whitelist the /ws/field `?agent=` param so it can't be used to probe arbitrary node names.
    A value that isn't a short [A-Za-z0-9_-] token falls back to 'eris' (the default OverSoul field).
    If `allowlist` (comma-separated, e.g. ERIS_WS_FIELD_AGENTS) is non-empty, the agent must be in it,
    else it also falls back to 'eris'."""
    a = agent or "eris"
    if not _FIELD_AGENT_RE.match(a):
        return "eris"
    allow = {x.strip() for x in (allowlist or "").split(",") if x.strip()}
    if allow and a not in allow:
        return "eris"
    return a
