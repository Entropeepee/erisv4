"""Codex PR#94 audit follow-ups (#2, #5), reusing the merged accelerators host helper.

#2 — the TTS provider POST ships raw speech text off-box BEFORE the edge_tts cloud guard; a remote
     ERIS_TTS_BASE_URL with no consent must refuse (fall back), never POST.
#5 — sovereignty's local-backend check treated *.local (mDNS) / docker.internal as local, so a
     "local"-named LLM backend at evil.local passed the sovereign gate. Now delegated to the shared
     fail-closed classifier → .local is remote / non-sovereign.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import types
import unittest
from unittest import mock

from eris.interface.sovereignty import is_local_backend


def _clear_consent(env):
    for k in list(env):
        if k.startswith("ERIS_ALLOW_REMOTE"):
            del env[k]


def _cfg(base):
    return types.SimpleNamespace(tts_base_url=base, tts_model="tts", accel_timeout_s=5.0)


class TestTTSProviderEgress(unittest.TestCase):
    def test_remote_tts_url_no_consent_refuses_without_posting(self):
        import eris.interface.tts as tts
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            with mock.patch.object(tts, "_post_speech",
                                   side_effect=AssertionError("must not POST raw text off-box")):
                out = tts._provider_speech("secret IP text", "v", _cfg("http://10.0.0.5:8001/v1"))
        self.assertIsNone(out)                       # refused → fall back, no POST

    def test_loopback_tts_url_is_allowed_to_post(self):
        import eris.interface.tts as tts
        posted = {"n": 0}

        def fake_post(url, payload, timeout):
            posted["n"] += 1
            return b"AUDIO"

        with mock.patch.object(tts, "_post_speech", side_effect=fake_post):
            out = tts._provider_speech("hello", "v", _cfg("http://127.0.0.1:8001/v1"))
        self.assertEqual(out, b"AUDIO")
        self.assertEqual(posted["n"], 1)

    def test_remote_tts_url_with_consent_posts(self):
        import eris.interface.tts as tts
        with mock.patch.dict(os.environ, {"ERIS_ALLOW_REMOTE_TTS": "1"}, clear=False):
            with mock.patch.object(tts, "_post_speech", return_value=b"AUDIO") as p:
                out = tts._provider_speech("hi", "v", _cfg("http://10.0.0.5:8001/v1"))
        self.assertEqual(out, b"AUDIO")
        p.assert_called_once()


class _Backend:
    def __init__(self, name, base_url=""):
        self.name = name
        self.base_url = base_url


class TestSovereigntyDotLocal(unittest.TestCase):
    def test_dot_local_backend_is_not_sovereign(self):
        # the #5 repro: a "local"-named backend at evil.local must NOT count as local
        self.assertFalse(is_local_backend(_Backend("local", "http://evil.local:8000/v1")))
        self.assertFalse(is_local_backend(_Backend("ollama", "http://evil.local:11434")))

    def test_docker_internal_now_remote(self):
        self.assertFalse(is_local_backend(_Backend("local", "http://host.docker.internal:8000")))

    def test_genuine_loopback_still_sovereign(self):
        self.assertTrue(is_local_backend(_Backend("ollama")))                       # no url
        self.assertTrue(is_local_backend(_Backend("local", "http://localhost:8000")))
        self.assertTrue(is_local_backend(_Backend("vllm", "http://127.0.0.1:8000")))

    def test_remote_ip_still_rejected(self):
        self.assertFalse(is_local_backend(_Backend("ollama", "http://10.0.0.5:11434")))


if __name__ == "__main__":
    unittest.main()
