"""Speech-to-text seam (Phase 4) — optional Whisper-on-NPU via a local service.

Opt-in voice input: if ERIS_STT_BASE_URL is set, audio is transcribed by a local
OpenAI-compatible /audio/transcriptions service (Whisper on the NPU is the
research's flagship win). Unset => the voice path is simply unavailable; Eris adds
no dependency and never imports openvino.
"""
from __future__ import annotations


def is_configured() -> bool:
    from eris.config import CONFIG
    return bool(CONFIG.stt_base_url)


def _post_audio(url: str, audio_bytes: bytes, filename: str, content_type: str,
                model: str, timeout: float) -> dict:
    """Thin HTTP seam (multipart) so tests can monkeypatch it."""
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            url,
            files={"file": (filename, audio_bytes, content_type)},
            data={"model": model})
        r.raise_for_status()
        return r.json()


def transcribe(audio_bytes: bytes, *, filename: str = "audio.wav",
               content_type: str = "audio/wav") -> str:
    """Transcribe audio via the configured STT service. Returns the text.
    Raises RuntimeError if no service is configured."""
    from eris.config import CONFIG
    base = (CONFIG.stt_base_url or "").rstrip("/")
    if not base:
        raise RuntimeError("No STT service configured (set ERIS_STT_BASE_URL).")
    data = _post_audio(
        f"{base}/audio/transcriptions", audio_bytes, filename, content_type,
        CONFIG.stt_model or "whisper-base", CONFIG.accel_timeout_s)
    return (data.get("text") or "").strip()
