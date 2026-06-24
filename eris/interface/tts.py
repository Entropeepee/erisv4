import os
import asyncio
import tempfile
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _post_speech(url: str, payload: dict, timeout: float) -> bytes:
    """Thin HTTP seam (so tests can monkeypatch). OpenAI /audio/speech shape."""
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.content


def _provider_speech(text: str, voice_id: str, CONFIG) -> Optional[bytes]:
    """Synthesize via the configured local TTS service (iGPU). Returns audio bytes
    or None to fall back to edge-tts."""
    base = (CONFIG.tts_base_url or "").rstrip("/")
    if not base:
        return None
    payload = {"model": CONFIG.tts_model or "tts", "input": text,
               "voice": voice_id or "default", "response_format": "wav"}
    data = _post_speech(f"{base}/audio/speech", payload, CONFIG.accel_timeout_s)
    return data or None


class TTSEngine:
    def __init__(self):
        self.voices = [
            {"id": "en-IE-EmilyNeural", "name": "Emily (Irish Female, 20s)", "engine": "edge-tts"},
            {"id": "en-GB-LibbyNeural", "name": "Libby (British Female, Soft/Pleasing)", "engine": "edge-tts"},
            {"id": "en-GB-SoniaNeural", "name": "Sonia (British Female, Confident)", "engine": "edge-tts"},
            {"id": "en-US-JennyNeural", "name": "Jenny (US Female, Comforting)", "engine": "edge-tts"},
            {"id": "en-US-AriaNeural", "name": "Aria (US Female, News/Novel)", "engine": "edge-tts"},
            {"id": "en-AU-NatashaNeural", "name": "Natasha (Australian Female, Friendly)", "engine": "edge-tts"}
        ]

    def get_voices(self) -> List[Dict[str, str]]:
        return self.voices

    def generate_audio(self, text: str, voice_id: str) -> Optional[bytes]:
        """Synchronous wrapper for Edge-TTS generation. Routes through the async
        bridge (run_blocking) instead of bare asyncio.run, so calling it from
        within a running event loop does not raise (A6)."""
        from eris.interface.mediator import run_blocking
        return run_blocking(self._generate_audio_async(text, voice_id))

    async def _generate_audio_async(self, text: str, voice_id: str) -> Optional[bytes]:
        # Phase 3b: optional local TTS provider (iGPU Kokoro, etc.) via
        # ERIS_TTS_BASE_URL — for fully-offline/private synthesis. Never the NPU
        # (dynamic STFT shapes). Falls back to edge-tts on any error/unset.
        from eris.config import CONFIG
        if CONFIG.tts_base_url:
            try:
                audio = await asyncio.to_thread(
                    _provider_speech, text, voice_id, CONFIG)
                if audio:
                    return audio
            except Exception as e:
                logger.warning(f"TTS provider failed ({e}); falling back to edge-tts")

        import edge_tts

        if not voice_id:
            voice_id = "en-IE-EmilyNeural"
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name
            
        try:
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(tmp_path)
            
            with open(tmp_path, 'rb') as f:
                data = f.read()
            return data
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
