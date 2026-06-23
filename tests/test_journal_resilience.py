"""The dream journal must survive a disk-full truncated line: a corrupt/partial
JSONL line (incl. broken UTF-8) must not throw during load and lose every other
entry — that was the 'clicks open nothing' failure after the disk filled."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.metacognition.dream_journal import DreamJournal


class TestJournalCorruptionResilience(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "dream_journal.jsonl")

    def test_good_entries_survive_truncated_and_broken_utf8_line(self):
        j = DreamJournal(path=self.path)
        a = j.record(kind="ponder", topic="first", summary="one", detail="d1")
        b = j.record(kind="ponder", topic="second", summary="two", detail="d2")
        # Simulate a disk-full write: append a partial line with a broken
        # multibyte byte sequence and NO trailing newline.
        with open(self.path, "ab") as f:
            f.write(b'{"id":"deadbeef","topic":"trunc","det\xff\xfe')
        c = j.record(kind="ponder", topic="third", summary="three", detail="d3")

        ids = {e["id"] for e in j.list(limit=50)}
        # The two clean early entries AND the one written after the corruption
        # are all still loadable (the corrupt line is skipped, not fatal).
        self.assertIn(a["id"], ids)
        self.assertIn(b["id"], ids)
        self.assertIn(c["id"], ids)
        # get-by-id (what the click handler calls) still finds them.
        self.assertEqual(j.get(a["id"])["detail"], "d1")
        self.assertEqual(j.get(c["id"])["detail"], "d3")

    def test_get_missing_returns_none(self):
        j = DreamJournal(path=self.path)
        self.assertIsNone(j.get("nope"))


if __name__ == "__main__":
    unittest.main()
