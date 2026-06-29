"""Request caps + opt-in rate limit + edge_tts egress consent (Phase 1, corrected #6). One caller
must not be able to exhaust RAM/CPU/disk via an oversized body or a flood, and the edge-tts fallback
(which ships text to Microsoft) must be OFF by default."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import importlib.util
import unittest

from eris.server.limits import RateLimiter, char_cap, byte_cap, client_of


class TestCaps(unittest.TestCase):
    def test_char_and_byte_cap_env_parsing(self):
        os.environ["ERIS_TEST_CAP"] = "1234"
        try:
            self.assertEqual(char_cap("ERIS_TEST_CAP", 5), 1234)
            self.assertEqual(char_cap("ERIS_NOPE_CAP", 77), 77)          # unset → default
            os.environ["ERIS_TEST_CAP"] = "notanint"
            self.assertEqual(char_cap("ERIS_TEST_CAP", 9), 9)            # bad value → default
            self.assertEqual(byte_cap("ERIS_NOPE_MB", 10), 10 * 1024 * 1024)
        finally:
            os.environ.pop("ERIS_TEST_CAP", None)


class TestRateLimiter(unittest.TestCase):
    def test_fixed_window_blocks_then_recovers(self):
        rl = RateLimiter(per_min=3, window=60)
        t = 1000.0
        self.assertTrue(rl.allow("ip", now=t))
        self.assertTrue(rl.allow("ip", now=t))
        self.assertTrue(rl.allow("ip", now=t))
        self.assertFalse(rl.allow("ip", now=t))          # 4th in the window → blocked
        self.assertTrue(rl.allow("ip", now=t + 61))      # window elapsed → allowed again
        self.assertTrue(rl.allow("other", now=t))        # a different client is independent

    def test_disabled_when_zero(self):
        rl = RateLimiter(per_min=0)
        for _ in range(100):
            self.assertTrue(rl.allow("ip"))              # 0 = disabled (default → local use unchanged)

    def test_client_of_reads_peer_ip(self):
        class _C:
            host = "10.0.0.9"

        class _Req:
            client = _C()
        self.assertEqual(client_of(_Req()), "10.0.0.9")
        self.assertEqual(client_of(object()), "anon")


class TestEdgeTtsConsent(unittest.TestCase):
    def test_fallback_off_by_default(self):
        import asyncio
        from eris.interface.tts import TTSEngine
        os.environ.pop("ERIS_TTS_ALLOW_CLOUD", None)
        os.environ.pop("ERIS_TTS_BASE_URL", None)
        # No local provider + no cloud consent → returns None WITHOUT shipping text to Microsoft
        # (returns before `import edge_tts`, so this passes even if edge_tts isn't installed).
        out = asyncio.run(TTSEngine()._generate_audio_async("hello world", ""))
        self.assertIsNone(out)


class TestEndpointsWired(unittest.TestCase):
    def test_five_endpoints_capped_and_rate_limited(self):
        spec = importlib.util.find_spec("eris.server.app")
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        # /chat, /v1, /ingest, /api/tts/generate, /api/stt each gate on the rate limiter
        self.assertGreaterEqual(src.count("_rate_or_429(request)"), 5)
        for cap in ("_max_chat", "_max_ingest", "_max_tts", "_max_stt"):
            self.assertIn(cap, src)
        self.assertIn("content-length", src)             # STT rejects oversize before buffering
        self.assertIn("asyncio.to_thread(extractor.extract_text", src)  # /ingest off the event loop


if __name__ == "__main__":
    unittest.main()
