"""Phase B audit fixes (web-exposure hardening): B1 deep-read root confinement,
B2 sandbox import bypass, B3 conversation-id validation, B7 constant-time auth."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest


class TestDeepReadRoots(unittest.TestCase):
    def test_path_outside_roots_rejected(self):
        from eris.knowledge.deep_read import _iter_segments, DeepReadConfig
        allowed = tempfile.mkdtemp()
        outside = tempfile.mkdtemp()
        old = os.environ.get("ERIS_DEEPREAD_ROOTS")
        os.environ["ERIS_DEEPREAD_ROOTS"] = allowed
        try:
            ok = os.path.join(allowed, "a.txt")
            with open(ok, "w") as f:
                f.write("allowed content here")
            bad = os.path.join(outside, "secret.txt")
            with open(bad, "w") as f:
                f.write("secret content")
            cfg = DeepReadConfig()
            segs = _iter_segments(None, ok, cfg)
            self.assertTrue(segs and "allowed content" in segs[0][1])
            self.assertEqual(_iter_segments(None, bad, cfg), [])   # rejected
        finally:
            if old is None:
                os.environ.pop("ERIS_DEEPREAD_ROOTS", None)
            else:
                os.environ["ERIS_DEEPREAD_ROOTS"] = old

    def test_ltm_and_raw_text_still_work(self):
        from eris.knowledge.deep_read import _iter_segments, DeepReadConfig
        cfg = DeepReadConfig()
        # raw text (not a path) passes through untouched
        self.assertEqual(_iter_segments(None, "just some text", cfg),
                         [("(text)", "just some text")])


class TestSandboxBlocksEris(unittest.TestCase):
    def test_import_eris_blocked(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("import eris.sandbox.executor\nx=1")
        self.assertFalse(ok)
        self.assertIn("eris", msg)

    def test_from_eris_blocked(self):
        from eris.sandbox.validator import validate_code
        ok, _ = validate_code("from eris.config import xp")
        self.assertFalse(ok)

    def test_numpy_still_allowed(self):
        from eris.sandbox.validator import validate_code
        ok, _ = validate_code("import numpy as np\nnp.zeros(3)")
        self.assertTrue(ok)


class TestConversationIdValidation(unittest.TestCase):
    def test_traversal_rejected(self):
        from eris.memory.conversations import ConversationStore
        store = ConversationStore(data_dir=tempfile.mkdtemp())
        with self.assertRaises(ValueError):
            store._path("../../etc/passwd")
        with self.assertRaises(ValueError):
            store._path("a/b")

    def test_valid_id_ok(self):
        from eris.memory.conversations import ConversationStore
        store = ConversationStore(data_dir=tempfile.mkdtemp())
        p = store._path("abc123DEF_-")
        self.assertTrue(p.endswith("abc123DEF_-.json"))


class TestConstantTimeAuth(unittest.TestCase):
    def test_still_matches_each_channel(self):
        from eris.server.auth import token_ok
        self.assertTrue(token_ok("secret", header="secret"))
        self.assertTrue(token_ok("secret", query="secret"))
        self.assertTrue(token_ok("secret", cookie="secret"))
        self.assertFalse(token_ok("secret", header="nope"))

    def test_substring_does_not_authenticate(self):
        # The 'substring auth' finding was WRONG — verify it stays wrong-proof.
        from eris.server.auth import token_ok
        self.assertFalse(token_ok("abc", header="xxxabcxxx"))

    def test_disabled_when_unset(self):
        from eris.server.auth import token_ok
        self.assertTrue(token_ok(""))


if __name__ == "__main__":
    unittest.main()
