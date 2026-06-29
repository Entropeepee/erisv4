"""/ws/field connection cap (Phase 1, corrected #5): /ws had an ERIS_WS_MAX cap; /ws/field lacked
it, so a client could open unbounded field sockets and starve the event loop. This guards that the
cap is wired into /ws/field (separate bucket, close 1013 over limit, cleanup on disconnect)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import importlib.util
import unittest


class TestWsFieldCap(unittest.TestCase):
    def test_field_socket_cap_is_wired(self):
        # Source-level guard (cheap — no orchestrator import). The cap must live in the /ws/field
        # region: a separate bucket, the ERIS_WS_MAX limit, a 1013 close over-limit, and cleanup.
        spec = importlib.util.find_spec("eris.server.app")
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("field_ws_connections", src)
        i = src.find('@app.websocket("/ws/field")')
        self.assertGreater(i, 0)
        region = src[i:i + 3200]
        self.assertIn("field_ws_connections.append", region)
        self.assertIn('os.environ.get("ERIS_WS_MAX"', region)
        self.assertIn("close(code=1013)", region)
        self.assertIn("field_ws_connections.remove", region)      # cleanup on disconnect

    def test_cap_pattern_closes_over_limit(self):
        # Verify the cap MECHANISM (a held connection blocks an over-limit one) on a minimal app,
        # mirroring the endpoint's logic without the heavy orchestrator. cap=1 for the test.
        import asyncio
        from fastapi import FastAPI, WebSocket
        from starlette.testclient import TestClient
        conns = []
        app = FastAPI()

        @app.websocket("/ws/field")
        async def f(ws: WebSocket):
            await ws.accept()
            if len(conns) >= 1:                    # cap = 1
                await ws.close(code=1013)
                return
            conns.append(ws)
            try:
                await ws.send_json({"ok": True})
                while True:
                    await asyncio.sleep(0.05)
            finally:
                if ws in conns:
                    conns.remove(ws)

        client = TestClient(app)
        with client.websocket_connect("/ws/field") as c1:
            self.assertEqual(c1.receive_json(), {"ok": True})       # first holds the only slot
            with self.assertRaises(Exception):                      # second is capped (closed 1013)
                with client.websocket_connect("/ws/field") as c2:
                    c2.receive_json()


if __name__ == "__main__":
    unittest.main()
