"""WebSocket auth (Phase 1, corrected #3): /ws and /ws/field must check the token IN-ENDPOINT and
close(1008) BEFORE accept() — the HTTP middleware never runs for the websocket scope — and the
/ws/field ?agent= param must be whitelisted so it can't probe arbitrary node names."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import importlib.util
import unittest

from eris.server.ws_guard import sanitize_field_agent, ws_authorized


class TestSanitizeFieldAgent(unittest.TestCase):
    def test_known_agent_passes(self):
        self.assertEqual(sanitize_field_agent("willow"), "willow")
        self.assertEqual(sanitize_field_agent("eris"), "eris")

    def test_garbage_falls_back_to_eris(self):
        for bad in ("", None, "../etc/passwd", "a b", "x" * 50, "a;rm -rf", "{}"):
            self.assertEqual(sanitize_field_agent(bad), "eris")

    def test_allowlist_enforced_when_set(self):
        self.assertEqual(sanitize_field_agent("willow", allowlist="eris,willow"), "willow")
        self.assertEqual(sanitize_field_agent("npc_c", allowlist="eris,willow"), "eris")  # not allowed
        self.assertEqual(sanitize_field_agent("willow", allowlist=""), "willow")           # no allowlist → any valid name


class TestWsAuthMechanism(unittest.TestCase):
    """Prove the close-before-accept pattern actually rejects an unauthenticated handshake, using a
    minimal app (no heavy orchestrator) that mirrors the real endpoints' guard."""

    def _app(self, token):
        from fastapi import FastAPI, WebSocket
        app = FastAPI()

        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            if not ws_authorized(token, websocket):
                await websocket.close(code=1008)
                return
            await websocket.accept()
            await websocket.send_json({"ok": True})

        return app

    def test_rejects_without_token_accepts_with(self):
        from starlette.testclient import TestClient
        client = TestClient(self._app("secret"))
        with self.assertRaises(Exception):                       # closed before accept → connect raises
            with client.websocket_connect("/ws") as c:
                c.receive_json()
        with client.websocket_connect("/ws?token=secret") as c:  # query token authorizes
            self.assertEqual(c.receive_json(), {"ok": True})

    def test_no_token_configured_allows_local_use(self):
        from starlette.testclient import TestClient
        client = TestClient(self._app(""))                       # gate disabled → unchanged local use
        with client.websocket_connect("/ws") as c:
            self.assertEqual(c.receive_json(), {"ok": True})


class TestRealEndpointsGuarded(unittest.TestCase):
    def test_both_ws_endpoints_check_auth_before_accept(self):
        # Source-level regression guard (cheap — no orchestrator import): both sockets must call the
        # auth helper and close(1008) before accept(). Stops a future edit from re-opening the hole.
        spec = importlib.util.find_spec("eris.server.app")
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("_ws_authorized(_auth_token, websocket)", src)
        self.assertGreaterEqual(src.count("close(code=1008)"), 2)         # /ws AND /ws/field
        self.assertIn("sanitize_field_agent(agent,", src)                 # agent whitelist wired
        # ordering: the first 1008-close must precede the first accept() in the file's WS region
        i_auth = src.find("_ws_authorized(_auth_token, websocket)")
        i_accept = src.find("await websocket.accept()", i_auth)
        self.assertGreater(i_accept, i_auth, "auth check must come before accept()")


if __name__ == "__main__":
    unittest.main()
