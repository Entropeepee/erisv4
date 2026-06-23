"""fetch_wikipedia must (a) identify itself with a Wikimedia-policy-compliant
User-Agent (browser-spoofing the API is what silently broke study), and (b)
raise on an API rejection / empty extract so '0 passages' carries a real reason
instead of failing silently."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import io
import json
import unittest

import eris.knowledge.web_reader as wr


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _patched(payload):
    captured = {}
    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")
        captured["url"] = req.full_url
        return _FakeResp(payload)
    return fake_urlopen, captured


class TestWikipediaFetch(unittest.TestCase):
    def setUp(self):
        self._orig = wr.urlopen

    def tearDown(self):
        wr.urlopen = self._orig

    def test_uses_descriptive_user_agent_not_browser(self):
        fake, cap = _patched({"query": {"pages": {"1": {"extract": "Hello world."}}}})
        wr.urlopen = fake
        text = wr.fetch_wikipedia("Cognitive science")
        self.assertEqual(text, "Hello world.")
        self.assertIn("ErisResearchBot", cap["ua"])
        self.assertNotIn("Mozilla", cap["ua"])          # not a browser spoof

    def test_api_error_raises(self):
        fake, _ = _patched({"error": {"code": "blocked", "info": "UA blocked"}})
        wr.urlopen = fake
        with self.assertRaises(RuntimeError) as ctx:
            wr.fetch_wikipedia("Cognitive science")
        self.assertIn("UA blocked", str(ctx.exception))

    def test_empty_extract_raises_not_silent(self):
        fake, _ = _patched({"query": {"pages": {"1": {"extract": ""}}}})
        wr.urlopen = fake
        with self.assertRaises(RuntimeError):
            wr.fetch_wikipedia("Cognitive science")

    def test_missing_article_returns_empty_not_error(self):
        fake, _ = _patched({"query": {"pages": {"-1": {"missing": ""}}}})
        wr.urlopen = fake
        self.assertEqual(wr.fetch_wikipedia("Nonexistent topic xyzzy"), "")


if __name__ == "__main__":
    unittest.main()
