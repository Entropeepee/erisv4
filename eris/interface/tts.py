import os
import asyncio
import tempfile
import logging
import edge_tts
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

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
        """Synchronous wrapper for Edge-TTS generation."""
        return asyncio.run(self._generate_audio_async(text, voice_id))

    async def _generate_audio_async(self, text: str, voice_id: str) -> Optional[bytes]:
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
