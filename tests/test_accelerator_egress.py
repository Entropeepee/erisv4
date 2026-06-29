"""Codex r3 #10: the optional accelerator services (embed / rerank / STT / VLM) receive Eris's RAW
CONTENT — texts, audio, images = the owner's IP. #89 added egress consent for edge_tts only; the
same loopback-guard / explicit-consent discipline must cover EVERY accelerator URL, so one
misconfigured remote endpoint can't quietly exfiltrate. Default-DENY remote; loopback always OK.

Offline: the guard is pure; the integration checks short-circuit BEFORE any network call.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
from unittest import mock

from eris.interface.accelerators import is_loopback_url, egress_allowed, check_egress_or_warn


def _clear_consent(env):
    for k in list(env):
        if k.startswith("ERIS_ALLOW_REMOTE"):
            del env[k]


class TestLoopbackDetection(unittest.TestCase):
    def test_loopback_hosts(self):
        for u in ("http://localhost:8013/v1", "http://127.0.0.1:8000", "http://127.0.0.5:9",
                  "http://127.0.0.7", "http://[::1]:8000/v1", "localhost:8013"):
            self.assertTrue(is_loopback_url(u), u)

    def test_remote_hosts(self):
        for u in ("http://10.0.0.5:8013/v1", "http://192.168.1.50:8000", "https://api.openai.com/v1",
                  "http://embeddings.example.com", "http://my-gpu-box.lan:8013",
                  "http://0.0.0.0:8013"):       # 0.0.0.0 is not is_loopback → remote (fail closed)
            self.assertFalse(is_loopback_url(u), u)

    def test_codex1_bypass_cases_are_remote(self):
        # The P0: a public DNS name that merely embeds/decorates a loopback token must NOT pass.
        for u in ("http://127.0.0.1@evil.com/v1",        # userinfo trick — real host is evil.com
                  "http://127.0.0.1.evil.com:8000",      # DNS name that starts with 127.
                  "http://evil.localhost/v1",            # *.localhost is a public name, not loopback
                  "http://%31%32%37.0.0.1:8000",         # %-encoded "127" — not a valid IP literal
                  "http://127.0.0.1.example.com"):
            self.assertFalse(is_loopback_url(u), u)

    def test_exact_localhost_variants_are_local(self):
        for u in ("http://localhost.:8013", "http://LOCALHOST:8013", "LocalHost:9"):
            self.assertTrue(is_loopback_url(u), u)

    def test_ipv4_mapped_ipv6_judged_by_embedded_v4(self):
        # Documented choice: ::ffff:127.0.0.1 is local (embedded v4 is loopback);
        # ::ffff:10.0.0.5 is remote (embedded v4 is public).
        self.assertTrue(is_loopback_url("http://[::ffff:127.0.0.1]:8000"))
        self.assertFalse(is_loopback_url("http://[::ffff:10.0.0.5]:8000"))

    def test_invalid_ip_fails_closed_remote(self):
        # a bracketed-but-invalid IPv6 host is not a valid IP and isn't "localhost" → remote
        self.assertFalse(is_loopback_url("http://[gggg::1]:8000"))
        self.assertFalse(is_loopback_url("http://[::ffff:10.0.0.5.junk]"))


class TestEgressDecision(unittest.TestCase):
    def test_no_url_is_allowed_in_process(self):
        ok, why = egress_allowed("embeddings", "")
        self.assertTrue(ok)
        self.assertIn("in-process", why)

    def test_loopback_allowed(self):
        ok, _ = egress_allowed("embeddings", "http://127.0.0.1:8013/v1")
        self.assertTrue(ok)

    def test_remote_denied_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            ok, why = egress_allowed("embeddings", "http://10.0.0.5:8013/v1")
            self.assertFalse(ok)
            self.assertIn("REFUSED", why)

    def test_remote_allowed_with_per_service_consent(self):
        with mock.patch.dict(os.environ, {"ERIS_ALLOW_REMOTE_EMBEDDINGS": "1"}, clear=False):
            ok, _ = egress_allowed("embeddings", "http://10.0.0.5:8013/v1")
            self.assertTrue(ok)

    def test_remote_allowed_with_global_consent(self):
        with mock.patch.dict(os.environ, {"ERIS_ALLOW_REMOTE_ACCEL": "1"}, clear=False):
            ok, _ = egress_allowed("rerank", "http://10.0.0.5:8013/v1")
            self.assertTrue(ok)

    def test_per_service_consent_is_scoped(self):
        # consenting to remote embeddings must NOT consent remote STT
        with mock.patch.dict(os.environ, {"ERIS_ALLOW_REMOTE_EMBEDDINGS": "1"}, clear=False):
            os.environ.pop("ERIS_ALLOW_REMOTE_ACCEL", None)
            os.environ.pop("ERIS_ALLOW_REMOTE_STT", None)
            self.assertFalse(egress_allowed("stt", "http://10.0.0.5:9000")[0])

    def test_check_warns_and_returns_false(self):
        msgs = []
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            ok = check_egress_or_warn("rerank", "http://10.0.0.5:8013",
                                      logger=mock.Mock(warning=msgs.append))
        self.assertFalse(ok)
        self.assertTrue(any("REFUSED" in m for m in msgs))


class TestEmbeddingsGuard(unittest.TestCase):
    def test_remote_embeddings_refused_falls_back_without_network(self):
        import eris.knowledge.embeddings as emb
        import eris.config as cfg
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            with mock.patch.object(cfg.CONFIG, "embed_base_url", "http://10.0.0.5:8013/v1"), \
                 mock.patch.object(emb, "_post_json",
                                   side_effect=AssertionError("must not hit the network")):
                self.assertIsNone(emb._provider_embeddings(["hello"]))   # refused → fall back


class TestRerankGuard(unittest.TestCase):
    def test_remote_rerank_refused_returns_none(self):
        from eris.retrieval.hybrid import http_reranker
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            self.assertIsNone(http_reranker(base_url="http://10.0.0.5:8013"))

    def test_loopback_rerank_constructs(self):
        from eris.retrieval.hybrid import http_reranker
        rr = http_reranker(base_url="http://127.0.0.1:8013")
        self.assertIsNotNone(rr)


class TestSTTGuard(unittest.TestCase):
    def test_remote_stt_refused_raises(self):
        import eris.interface.stt as stt
        import eris.config as cfg
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            with mock.patch.object(cfg.CONFIG, "stt_base_url", "http://10.0.0.5:9000"), \
                 mock.patch.object(stt, "_post_audio",
                                   side_effect=AssertionError("must not hit the network")):
                with self.assertRaises(RuntimeError):
                    stt.transcribe(b"audio")


class TestVisionGuard(unittest.TestCase):
    def test_remote_vision_refused_raises(self):
        import asyncio
        from eris.interface.vision import see
        with mock.patch.dict(os.environ, {"ERIS_VISION_BASE_URL": "http://10.0.0.5:8000/v1"},
                             clear=False):
            _clear_consent(os.environ)
            with self.assertRaises(RuntimeError):
                asyncio.run(see("what is this?", []))


class TestStatusProbeGuard(unittest.TestCase):
    """Codex #3: accelerator_status probes configured URLs over HTTP, leaking source IP/UA. A remote
    URL with no consent must NOT be probed."""

    def test_remote_url_not_probed_without_consent(self):
        import eris.interface.accelerators as acc
        import eris.config as cfg
        with mock.patch.dict(os.environ, {}, clear=False):
            _clear_consent(os.environ)
            with mock.patch.object(cfg.CONFIG, "embed_base_url", "http://10.0.0.5:8013/v1"), \
                 mock.patch.object(acc, "_reachable",
                                   side_effect=AssertionError("must not probe a denied remote URL")):
                st = acc.accelerator_status(probe=True, timeout=0.1)
        emb = st["embeddings"]
        self.assertFalse(emb["egress_allowed"])
        self.assertFalse(emb["reachable"])
        self.assertIn("not probed", emb["status"])

    def test_remote_url_probed_when_consented(self):
        import eris.interface.accelerators as acc
        import eris.config as cfg
        probed = {"n": 0}
        with mock.patch.dict(os.environ, {"ERIS_ALLOW_REMOTE_EMBEDDINGS": "1"}, clear=False):
            with mock.patch.object(cfg.CONFIG, "embed_base_url", "http://10.0.0.5:8013/v1"), \
                 mock.patch.object(acc, "_reachable",
                                   side_effect=lambda *a, **k: probed.__setitem__("n", 1) or True):
                st = acc.accelerator_status(probe=True, timeout=0.1)
        self.assertEqual(probed["n"], 1)                  # consent → probe runs
        self.assertTrue(st["embeddings"]["egress_allowed"])


if __name__ == "__main__":
    unittest.main()
