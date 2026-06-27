"""Optional access token for reaching Eris from off-LAN (phone over Tailscale/5G).

Default-OFF: with no ERIS_AUTH_TOKEN set, nothing changes (local/LAN use). When a
token is set, every request must carry it — header `X-Eris-Token`, `?token=...`,
or the `eris_token` cookie that visiting `/?token=SECRET` once sets (so a phone
browser stays logged in). The Tailscale/Cloudflare tunnel is the transport
security; this just stops anyone else who reaches the port from using her.
"""
from __future__ import annotations
from typing import Optional
import secrets


def token_ok(expected: str, *, header: Optional[str] = None,
             query: Optional[str] = None, cookie: Optional[str] = None) -> bool:
    """True if any presented credential matches the expected token. If no token is
    configured (expected falsy), everything is allowed (gate disabled).

    B7: constant-time comparison (secrets.compare_digest) so the check doesn't
    leak the token byte-by-byte via timing. (The earlier 'substring auth' finding
    was WRONG — this was tuple equality, never a substring match.)"""
    expected = (expected or "").strip()
    if not expected:
        return True
    return any(secrets.compare_digest(expected, cred or "")
               for cred in (header, query, cookie))


def make_auth_middleware(token: str):
    """Build a Starlette middleware class that enforces `token` on every request."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import PlainTextResponse

    class _AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not token_ok(token,
                            header=request.headers.get("x-eris-token"),
                            query=request.query_params.get("token"),
                            cookie=request.cookies.get("eris_token")):
                return PlainTextResponse("Unauthorized", status_code=401)
            resp = await call_next(request)
            if request.query_params.get("token") == token:
                resp.set_cookie("eris_token", token, httponly=True,
                                samesite="lax", max_age=60 * 60 * 24 * 365)
            return resp

    return _AuthMiddleware
