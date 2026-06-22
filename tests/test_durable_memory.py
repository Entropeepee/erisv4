"""Tests for the durable fact store (roadmap 1.4)."""
import os
os.environ.setdefault("ERIS_GPU", "0")

import tempfile
import unittest

from eris.memory.durable import LocalFactStore, get_durable_memory, DurableMemory


class TestLocalFactStore(unittest.TestCase):
    def _store(self):
        d = tempfile.mkdtemp()
        return LocalFactStore(os.path.join(d, "facts.json"))

    def test_add_and_search(self):
        s = self._store()
        s.add("The user's name is David.")
        s.add("Eris runs on an RTX 5080.")
        # Lexical store (exact recall of names/IDs/values); query the actual token.
        hits = s.search("RTX 5080", k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("5080", hits[0]["text"])

    def test_self_edits_duplicate(self):
        s = self._store()
        a = s.add("favorite color is blue", source="turn1")
        b = s.add("Favorite color is blue", source="turn2")   # same fact, diff case
        self.assertEqual(a, b)                                  # updated, not duplicated
        self.assertEqual(len(s.all()), 1)
        self.assertEqual(s.all()[0]["metadata"]["source"], "turn2")

    def test_persists_across_instances(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "facts.json")
        LocalFactStore(path).add("durable across restarts")
        reopened = LocalFactStore(path)
        self.assertEqual(len(reopened.all()), 1)

    def test_forget(self):
        s = self._store()
        fid = s.add("temporary fact")
        self.assertTrue(s.forget(fid))
        self.assertEqual(len(s.all()), 0)

    def test_factory_default_is_local_and_satisfies_protocol(self):
        d = tempfile.mkdtemp()
        m = get_durable_memory(os.path.join(d, "f.json"))
        self.assertIsInstance(m, LocalFactStore)
        self.assertIsInstance(m, DurableMemory)        # runtime Protocol check

    def test_unknown_backend_raises(self):
        os.environ["ERIS_MEMORY_BACKEND"] = "bogus"
        try:
            with self.assertRaises(ValueError):
                get_durable_memory()
        finally:
            os.environ.pop("ERIS_MEMORY_BACKEND", None)


if __name__ == "__main__":
    unittest.main()
