"""Optional access-token gate for off-LAN (phone) access."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.server.auth import token_ok, make_auth_middleware


class TestTokenOk(unittest.TestCase):
    def test_disabled_when_no_token(self):
        # No configured token → gate is open (unchanged local behavior).
        self.assertTrue(token_ok(""))
        self.assertTrue(token_ok("", header=None, query=None, cookie=None))

    def test_accepts_any_channel(self):
        self.assertTrue(token_ok("secret", header="secret"))
        self.assertTrue(token_ok("secret", query="secret"))
        self.assertTrue(token_ok("secret", cookie="secret"))

    def test_rejects_wrong_or_missing(self):
        self.assertFalse(token_ok("secret"))
        self.assertFalse(token_ok("secret", header="nope", query="", cookie=""))


class TestMiddlewareIntegration(unittest.TestCase):
    def _client(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi/testclient not installed")
        app = FastAPI()
        app.add_middleware(make_auth_middleware("s3cret"))

        @app.get("/ping")
        def ping():
            return {"ok": True}
        return TestClient(app)

    def test_blocks_without_token(self):
        c = self._client()
        self.assertEqual(c.get("/ping").status_code, 401)

    def test_allows_with_header(self):
        c = self._client()
        r = c.get("/ping", headers={"X-Eris-Token": "s3cret"})
        self.assertEqual(r.status_code, 200)

    def test_query_sets_cookie_then_persists(self):
        c = self._client()
        r = c.get("/ping?token=s3cret")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(c.cookies.get("eris_token"), "s3cret")
        # Cookie now carries auth on a plain request.
        self.assertEqual(c.get("/ping").status_code, 200)


if __name__ == "__main__":
    unittest.main()
