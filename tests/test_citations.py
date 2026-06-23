"""§9a: library sources that ground a turn are surfaced as citations; her own
reflections / plain conversation are not cited as external sources."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.orchestrator import _collect_citations


class _Rec:
    def __init__(self, source, title="", url=""):
        self.source = source
        self.metadata = {"title": title}
        if url:
            self.metadata["url"] = url


class TestCitations(unittest.TestCase):
    def test_library_sources_cited(self):
        recs = [_Rec("reading:SGT Patent", "SGT Patent"),
                _Rec("study:Kuramoto model", "Kuramoto model"),
                _Rec("exploration:https://example.org/x", "Example", "https://example.org/x")]
        cites = _collect_citations(recs)
        titles = {c["title"] for c in cites}
        self.assertEqual(titles, {"SGT Patent", "Kuramoto model", "Example"})
        ex = [c for c in cites if c["title"] == "Example"][0]
        self.assertEqual(ex["url"], "https://example.org/x")

    def test_conversation_and_reflection_not_cited(self):
        recs = [_Rec("conversation", "chat"),
                _Rec("introspection", "my reflection"),
                _Rec("reflection", "thought")]
        self.assertEqual(_collect_citations(recs), [])

    def test_dedup_and_limit(self):
        recs = [_Rec("reading:Same", "Same") for _ in range(5)] + \
               [_Rec(f"reading:Doc{i}", f"Doc{i}") for i in range(10)]
        cites = _collect_citations(recs, limit=6)
        self.assertLessEqual(len(cites), 6)
        self.assertEqual(len({c["title"] for c in cites}), len(cites))  # deduped


if __name__ == "__main__":
    unittest.main()
