"""The server binds localhost-only by default, and refuses an externally-reachable bind unless an
auth token is set — so a default launch can't silently expose the IP box to the whole LAN."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.server.bind import resolve_bind_host


class TestResolveBindHost(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("ERIS_BIND_HOST", "ERIS_AUTH_TOKEN")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_default_is_localhost(self):
        self.assertEqual(resolve_bind_host(), "127.0.0.1")            # not 0.0.0.0

    def test_external_bind_refused_without_token(self):
        os.environ["ERIS_BIND_HOST"] = "0.0.0.0"
        with self.assertRaises(SystemExit):                          # fail loud, don't expose
            resolve_bind_host()

    def test_external_bind_allowed_with_token(self):
        os.environ["ERIS_BIND_HOST"] = "0.0.0.0"
        os.environ["ERIS_AUTH_TOKEN"] = "secret"
        self.assertEqual(resolve_bind_host(), "0.0.0.0")             # explicit opt-in + token

    def test_localhost_variants_ok_without_token(self):
        for h in ("127.0.0.1", "localhost", "::1"):
            os.environ.pop("ERIS_AUTH_TOKEN", None)
            os.environ["ERIS_BIND_HOST"] = h
            self.assertEqual(resolve_bind_host(), h)

    def test_blank_bind_host_falls_back_to_localhost(self):
        os.environ["ERIS_BIND_HOST"] = "   "
        self.assertEqual(resolve_bind_host(), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
