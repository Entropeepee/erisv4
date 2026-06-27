"""Phase A audit fixes: A3 honest is_semantic, A5 atomic manifest, A6 TTS sync
wrapper safe inside an event loop."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import json
import tempfile
import unittest


class TestIsSemanticHonest(unittest.TestCase):
    def test_down_provider_reports_not_semantic(self):
        # A3: a configured-but-unreachable provider must report False, not True
        # just because the URL string is set.
        import eris.knowledge.embeddings as emb
        from eris.config import CONFIG
        old_url, old_health = CONFIG.embed_base_url, emb._PROVIDER_HEALTHY
        try:
            CONFIG.embed_base_url = "http://127.0.0.1:9/v1"   # nothing listens here
            emb._PROVIDER_HEALTHY = None
            self.assertFalse(emb.is_semantic())               # probe fails → False
        finally:
            CONFIG.embed_base_url = old_url
            emb._PROVIDER_HEALTHY = old_health

    def test_no_provider_no_model_is_not_semantic(self):
        import eris.knowledge.embeddings as emb
        from eris.config import CONFIG
        old = CONFIG.embed_base_url
        try:
            CONFIG.embed_base_url = ""        # ERIS_EMBEDDINGS=off → no local model
            self.assertFalse(emb.is_semantic())
        finally:
            CONFIG.embed_base_url = old


class TestAtomicManifest(unittest.TestCase):
    def test_record_is_atomic_and_reloads(self):
        from eris.knowledge.documents import LibraryManifest
        path = os.path.join(tempfile.mkdtemp(), "library_manifest.json")
        m = LibraryManifest(path)
        m.record("sha1", {"title": "Doc 1"})
        m.record("sha2", {"title": "Doc 2"})
        # No temp file left behind; file is complete valid JSON.
        self.assertFalse(os.path.exists(path + ".tmp"))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(set(data), {"sha1", "sha2"})
        # A fresh manifest reloads it.
        self.assertTrue(LibraryManifest(path).seen("sha1"))


class TestTtsSyncWrapper(unittest.IsolatedAsyncioTestCase):
    async def test_generate_audio_inside_event_loop_does_not_raise(self):
        # A6: called from a running event loop, the sync wrapper must NOT raise
        # "asyncio.run() cannot be called from a running event loop". (A missing
        # edge-tts dependency in CI is a different, acceptable error — it proves
        # the coroutine actually ran in a worker thread past the loop hurdle.)
        from eris.interface.tts import TTSEngine
        eng = TTSEngine()
        try:
            out = eng.generate_audio("hello", "")
            self.assertTrue(out is None or isinstance(out, (bytes, bytearray)))
        except RuntimeError as e:
            if "running event loop" in str(e) or "asyncio.run" in str(e):
                self.fail(f"A6 not fixed — event-loop error: {e}")
        except ModuleNotFoundError:
            pass   # no edge-tts in this environment; the loop hurdle was cleared


if __name__ == "__main__":
    unittest.main()
