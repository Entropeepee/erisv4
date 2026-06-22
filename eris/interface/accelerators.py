"""Accelerator service health probe + status (Phase 7, cross-cutting).

Reports, for each optional external service (embeddings / rerank / tts / stt),
whether it is configured and reachable — so the cockpit can show at a glance
which accelerators are live vs falling back to in-process behavior. Probing never
blocks boot and never raises; a down service simply reports unreachable.
"""
from __future__ import annotations
from typing import Dict

# (status key, CONFIG url attr, CONFIG model attr)
SERVICES = [
    ("embeddings", "embed_base_url", "embed_model"),
    ("rerank", "rerank_base_url", "rerank_model"),
    ("tts", "tts_base_url", "tts_model"),
    ("stt", "stt_base_url", "stt_model"),
]


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
