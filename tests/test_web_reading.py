"""Tests for Fix A — clean extraction + reader-proxy fallback on 403."""
import os
os.environ.setdefault("ERIS_GPU", "0")

import asyncio
import unittest
from urllib.error import HTTPError

from eris.config import CONFIG
from eris.retrieval import web_search


class TestCleanExtraction(unittest.TestCase):
    def test_strips_jump_to_content_and_chrome(self):
        html = """
        <html><body>
          <a class="mw-jump-link" href="#content">Jump to content</a>
          <nav id="site-nav"><a href="/">Home</a><a href="/x">Menu</a></nav>
          <div id="article-body"><p>The Kuramoto model describes synchronization
          of coupled oscillators.</p></div>
          <div class="footer-links">Privacy policy and terms</div>
        </body></html>
        """
        text = web_search._extract_text_from_html(html)
        self.assertIn("Kuramoto model describes synchronization", text)
        self.assertNotIn("Jump to content", text)
        self.assertNotIn("Home", text)
        self.assertNotIn("Privacy policy", text)

    def test_nested_boilerplate_skipped_content_kept(self):
        # Well-formed: a closed nav with nested chrome, then the article body.
        html = ("<nav><ul><li><a>Menu item</a></li></ul></nav>"
                "<article><h1>Title</h1><p>Real body text here.</p></article>")
        text = web_search._extract_text_from_html(html)
        self.assertIn("Real body text here.", text)
        self.assertIn("Title", text)
        self.assertNotIn("Menu item", text)


class TestProxyFallback(unittest.TestCase):
    def _force_403(self, *a, **k):
        raise HTTPError("http://blocked.example", 403, "Forbidden", {}, None)

    def test_403_uses_proxy_when_enabled(self):
        orig_direct = web_search._direct_fetch
        orig_proxy = web_search._proxy_fetch
        web_search._direct_fetch = self._force_403
        web_search._proxy_fetch = lambda url, mc: "CLEAN PROXY TEXT"
        CONFIG.web_reader_proxy = True
        try:
            out = asyncio.run(web_search.fetch_content("http://blocked.example"))
            self.assertEqual(out, "CLEAN PROXY TEXT")
        finally:
            web_search._direct_fetch = orig_direct
            web_search._proxy_fetch = orig_proxy
            CONFIG.web_reader_proxy = False

    def test_403_returns_empty_when_proxy_disabled(self):
        orig_direct = web_search._direct_fetch
        web_search._direct_fetch = self._force_403
        CONFIG.web_reader_proxy = False
        try:
            out = asyncio.run(web_search.fetch_content("http://blocked.example"))
            self.assertEqual(out, "")
        finally:
            web_search._direct_fetch = orig_direct


if __name__ == "__main__":
    unittest.main()
