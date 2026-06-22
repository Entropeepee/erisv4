"""Tests for the accelerator provider seams (embeddings/rerank) — Phase 1/2."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.config import CONFIG
from eris.knowledge import embeddings as emb
from eris.knowledge.embeddings import EMBED_DIM


class TestEmbeddingProvider(unittest.TestCase):
    def setUp(self):
        emb._CACHE.clear()
        emb._PROVIDER_WARNED = False
        self._base = CONFIG.embed_base_url
        self._post = emb._post_json

    def tearDown(self):
        CONFIG.embed_base_url = self._base
        emb._post_json = self._post
        emb._CACHE.clear()

    def test_provider_used_when_configured(self):
        CONFIG.embed_base_url = "http://localhost:9/v1"
        sentinel = np.arange(EMBED_DIM, dtype=np.float32)
        calls = {"n": 0}
        def fake(url, payload, timeout):
            calls["n"] += 1
            return {"data": [{"embedding": sentinel.tolist()} for _ in payload["input"]]}
        emb._post_json = fake
        v = emb.get_embedding("hello")
        self.assertEqual(v.shape[0], EMBED_DIM)
        self.assertTrue(np.allclose(v, sentinel))
        self.assertEqual(calls["n"], 1)

    def test_batch_one_call_and_cache(self):
        CONFIG.embed_base_url = "http://localhost:9/v1"
        calls = {"n": 0}
        def fake(url, payload, timeout):
            calls["n"] += 1
            return {"data": [{"embedding": [float(i)] * EMBED_DIM} for i in range(len(payload["input"]))]}
        emb._post_json = fake
        out = emb.get_embeddings(["a", "b", "c"])
        self.assertEqual(len(out), 3)
        self.assertEqual(calls["n"], 1)                 # one HTTP call for the batch
        emb.get_embedding("a")                          # cached -> no new call
        self.assertEqual(calls["n"], 1)

    def test_falls_back_on_provider_error(self):
        CONFIG.embed_base_url = "http://localhost:9/v1"
        def boom(url, payload, timeout):
            raise ConnectionError("service down")
        emb._post_json = boom
        v = emb.get_embedding("hello")                  # must not raise
        self.assertEqual(v.shape[0], EMBED_DIM)         # in-process fallback used

    def test_dim_mismatch_falls_back(self):
        CONFIG.embed_base_url = "http://localhost:9/v1"
        def wrong(url, payload, timeout):
            return {"data": [{"embedding": [0.1] * (EMBED_DIM + 7)}]}
        emb._post_json = wrong
        v = emb.get_embedding("hello")
        self.assertEqual(v.shape[0], EMBED_DIM)         # fell back, correct dim

    def test_unset_uses_in_process(self):
        CONFIG.embed_base_url = ""
        def fail(url, payload, timeout):
            raise AssertionError("provider should not be called when unset")
        emb._post_json = fail
        v = emb.get_embedding("hello")
        self.assertEqual(v.shape[0], EMBED_DIM)


class _Rec:
    def __init__(self, text):
        self.text = text
        self.embedding = None


class TestRerankProvider(unittest.TestCase):
    def setUp(self):
        self._base = CONFIG.rerank_base_url
        self._post = emb._post_json

    def tearDown(self):
        CONFIG.rerank_base_url = self._base
        emb._post_json = self._post

    def test_none_when_unset(self):
        from eris.retrieval.hybrid import http_reranker
        CONFIG.rerank_base_url = ""
        self.assertIsNone(http_reranker())

    def test_reranker_reorders_via_hybrid(self):
        from eris.retrieval.hybrid import http_reranker, hybrid_search
        CONFIG.rerank_base_url = "http://localhost:9/v1"
        def fake(url, payload, timeout):
            docs = payload["documents"]
            return {"results": [{"index": i, "relevance_score": (1.0 if i == len(docs) - 1 else 0.0)}
                                for i in range(len(docs))]}
        emb._post_json = fake
        out = hybrid_search("q", [_Rec("a"), _Rec("b"), _Rec("c")],
                            top_k=3, reranker=http_reranker())
        self.assertEqual(out[0].text, "c")              # endpoint preferred the last

    def test_reranker_error_is_rrf_only(self):
        from eris.retrieval.hybrid import http_reranker, hybrid_search
        CONFIG.rerank_base_url = "http://localhost:9/v1"
        def boom(url, payload, timeout):
            raise ConnectionError("down")
        emb._post_json = boom
        out = hybrid_search("q", [_Rec("a"), _Rec("b")], top_k=2,
                            reranker=http_reranker())
        self.assertEqual(len(out), 2)                   # no error; fused order kept


class TestAcceleratorStatus(unittest.TestCase):
    def setUp(self):
        self._e = CONFIG.embed_base_url

    def tearDown(self):
        CONFIG.embed_base_url = self._e

    def test_unset_is_off_in_process(self):
        from eris.interface.accelerators import accelerator_status
        CONFIG.embed_base_url = ""
        st = accelerator_status(probe=False)
        self.assertFalse(st["embeddings"]["configured"])
        self.assertIn("in-process", st["embeddings"]["status"])

    def test_configured_is_reported(self):
        from eris.interface.accelerators import accelerator_status
        CONFIG.embed_base_url = "http://localhost:9/v1"
        st = accelerator_status(probe=False)
        self.assertTrue(st["embeddings"]["configured"])
        self.assertEqual(st["embeddings"]["base_url"], "http://localhost:9/v1")


class TestSTTSeam(unittest.TestCase):
    def setUp(self):
        self._b = CONFIG.stt_base_url

    def tearDown(self):
        CONFIG.stt_base_url = self._b

    def test_unconfigured(self):
        from eris.interface import stt
        CONFIG.stt_base_url = ""
        self.assertFalse(stt.is_configured())
        with self.assertRaises(RuntimeError):
            stt.transcribe(b"x")

    def test_transcribe_via_mock(self):
        from eris.interface import stt
        CONFIG.stt_base_url = "http://localhost:9/v1"
        orig = stt._post_audio
        stt._post_audio = lambda *a, **k: {"text": "hello world"}
        try:
            self.assertEqual(stt.transcribe(b"audio"), "hello world")
        finally:
            stt._post_audio = orig


class TestTTSProvider(unittest.TestCase):
    def setUp(self):
        self._b = CONFIG.tts_base_url

    def tearDown(self):
        CONFIG.tts_base_url = self._b

    def test_provider_unset_returns_none(self):
        from eris.interface import tts
        CONFIG.tts_base_url = ""
        self.assertIsNone(tts._provider_speech("hi", "v", CONFIG))

    def test_provider_returns_bytes(self):
        from eris.interface import tts
        CONFIG.tts_base_url = "http://localhost:9/v1"
        orig = tts._post_speech
        tts._post_speech = lambda url, payload, timeout: b"RIFF....wav"
        try:
            self.assertEqual(tts._provider_speech("hi", "v", CONFIG), b"RIFF....wav")
        finally:
            tts._post_speech = orig


if __name__ == "__main__":
    unittest.main()
