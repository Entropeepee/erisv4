"""Accelerator service health probe + status (Phase 7, cross-cutting).

Reports, for each optional external service (embeddings / rerank / tts / stt),
whether it is configured and reachable — so the cockpit can show at a glance
which accelerators are live vs falling back to in-process behavior. Probing never
blocks boot and never raises; a down service simply reports unreachable.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import os
from urllib.parse import urlparse

# (status key, CONFIG url attr, CONFIG model attr)
SERVICES = [
    ("embeddings", "embed_base_url", "embed_model"),
    ("rerank", "rerank_base_url", "rerank_model"),
    ("tts", "tts_base_url", "tts_model"),
    ("stt", "stt_base_url", "stt_model"),
]


# ── Egress guard for the optional accelerator services (Codex r3 #10) ──────────
# These services receive Eris's RAW CONTENT — text to embed/rerank, audio to
# transcribe, images for the VLM. That content is the owner's IP. A loopback URL
# keeps it on-box; a REMOTE URL ships it off-box. #89 added this consent for
# edge_tts only; the same discipline must cover every accelerator URL so a single
# misconfigured remote endpoint can't quietly exfiltrate. Default-DENY remote,
# consistent with the localhost-bind / sandbox / edge_tts posture.

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "ip6-localhost", ""}


def is_loopback_url(base_url: str) -> bool:
    """True if `base_url` targets the local machine (loopback / unspecified host) — so sending
    content there does NOT leave the box. Anything else (a LAN IP, a public host) is remote."""
    host = (urlparse(base_url if "://" in (base_url or "") else f"//{base_url}").hostname or "")
    host = host.strip().lower()
    if host in _LOOPBACK_HOSTS or host.endswith(".localhost"):
        return True
    return host.startswith("127.")            # all of 127.0.0.0/8 is loopback


def _consent_on(*keys: str) -> bool:
    for k in keys:
        if os.environ.get(k, "").strip().lower() in ("1", "on", "true", "yes"):
            return True
    return False


def egress_allowed(name: str, base_url: str,
                   *, consent_var: str = "ERIS_ALLOW_REMOTE_ACCEL") -> Tuple[bool, str]:
    """Decide whether content may be sent to accelerator `name` at `base_url`. Returns
    (allowed, reason). Policy:
      • no URL set            → allowed (in-process; nothing leaves the box);
      • loopback URL          → allowed (local service);
      • REMOTE URL            → DENIED unless the owner opts in via the per-service
                                ERIS_ALLOW_REMOTE_<NAME> or the global ERIS_ALLOW_REMOTE_ACCEL.
    A denied remote call must fall back to in-process / unavailable — never silently ship."""
    base = (base_url or "").strip()
    if not base:
        return True, "in-process (no URL configured)"
    if is_loopback_url(base):
        return True, "loopback (local — content stays on-box)"
    per = f"ERIS_ALLOW_REMOTE_{name.upper()}"
    if _consent_on(per, consent_var):
        return True, f"remote egress consented ({per} or {consent_var})"
    return (False,
            f"REFUSED: accelerator '{name}' URL is REMOTE ({base}) — sending content there would "
            f"ship it off-box (possible IP exfiltration). Use a loopback URL, or set {per}=1 / "
            f"{consent_var}=1 to consent.")


def check_egress_or_warn(name: str, base_url: str, logger=None) -> bool:
    """Convenience wrapper for call sites: returns True if egress is allowed, else logs/prints a
    loud one-line warning and returns False (the caller then falls back). Never raises."""
    ok, reason = egress_allowed(name, base_url)
    if not ok:
        msg = f"[egress] {reason}"
        try:
            (logger.warning if logger else print)(msg)
        except Exception:
            pass
    return ok


def _reachable(base_url: str, timeout: float = 2.0) -> bool:
    """True if the service answers at all (any HTTP status = up). A connection
    error/timeout = down. Tries /models (OpenAI-style) then the bare base."""
    base = (base_url or "").rstrip("/")
    if not base:
        return False
    try:
        import httpx
    except Exception:
        return False
    for url in (f"{base}/models", base):
        try:
            with httpx.Client(timeout=timeout) as c:
                c.get(url)
            return True
        except Exception:
            continue
    return False


def accelerator_status(probe: bool = True, timeout: float = 2.0) -> Dict[str, dict]:
    """Per-service {configured, base_url, model, reachable, status}. With
    `probe=False` it reports config only (no network)."""
    from eris.config import CONFIG
    out: Dict[str, dict] = {}
    for name, url_attr, model_attr in SERVICES:
        base = getattr(CONFIG, url_attr, "") or ""
        model = getattr(CONFIG, model_attr, "") or ""
        configured = bool(base)
        reachable = _reachable(base, timeout) if (configured and probe) else False
        if not configured:
            status = "off (in-process)"
        elif reachable:
            status = "live"
        else:
            status = "unreachable → fallback"
        out[name] = {"configured": configured, "base_url": base, "model": model,
                     "reachable": reachable, "status": status}
    return out
