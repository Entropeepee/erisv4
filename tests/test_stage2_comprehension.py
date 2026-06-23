"""Stage-2 comprehension: knowledge-graph triples + multi-hop, propositions,
and the multi-source parser router (all default-OFF, graceful fallback)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import json
import tempfile
import unittest

from eris.knowledge.graph import (
    KnowledgeGraph, extract_triples, parse_triples,
)
from eris.knowledge.comprehend import propositions, _parse_props
from eris.knowledge import parsers


class TestTripleExtraction(unittest.TestCase):
    def test_parse_triples(self):
        raw = '[{"s":"Kuramoto model","r":"describes","o":"synchronization"}]'
        t = parse_triples(raw)
        self.assertEqual(t, [{"s": "Kuramoto model", "r": "describes",
                              "o": "synchronization"}])

    def test_extract_with_retry(self):
        calls = []
        def gen(p):
            calls.append(p)
            return ("garbage" if len(calls) == 1 else
                    '[{"s":"A","r":"relates to","o":"B"}]')
        out = extract_triples("text", gen, retries=1)
        self.assertEqual(len(calls), 2)             # retried after bad JSON
        self.assertEqual(out[0]["s"], "A")

    def test_no_model_safe(self):
        self.assertEqual(extract_triples("text", None), [])


class TestKnowledgeGraph(unittest.TestCase):
    def _kg(self):
        return KnowledgeGraph(path=os.path.join(tempfile.mkdtemp(), "kg.jsonl"))

    def test_add_persist_reload(self):
        path = os.path.join(tempfile.mkdtemp(), "kg.jsonl")
        kg = KnowledgeGraph(path=path)
        kg.add_triples([{"s": "X", "r": "causes", "o": "Y"}], source="s1")
        self.assertEqual(kg.size(), 1)
        self.assertEqual(KnowledgeGraph(path=path).size(), 1)  # reloads from disk

    def test_multihop_expand(self):
        kg = self._kg()
        kg.add_triples([
            {"s": "entropy", "r": "relates to", "o": "information"},
            {"s": "information", "r": "relates to", "o": "computation"},
            {"s": "computation", "r": "relates to", "o": "thermodynamics"},
        ])
        # From 'entropy', 2 hops should reach 'information' and 'computation'.
        reached = kg.expand(["entropy"], hops=2, limit=10)
        self.assertIn("information", reached)
        self.assertIn("computation", reached)
        self.assertNotIn("entropy", reached)        # seeds excluded
        self.assertNotIn("thermodynamics", reached) # 3 hops away


class TestPropositions(unittest.TestCase):
    def test_parse_props(self):
        self.assertEqual(_parse_props('["fact one", "fact two"]'),
                         ["fact one", "fact two"])

    def test_propositions(self):
        out = propositions("text", lambda p: '["A is B.", "C depends on D."]', n=6)
        self.assertEqual(out, ["A is B.", "C depends on D."])

    def test_no_model_safe(self):
        self.assertEqual(propositions("text", None), [])


class TestParserRouter(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(parsers.classify("https://arxiv.org/pdf/2401.00001.pdf"),
                         "paper_pdf")
        self.assertEqual(parsers.classify("https://example.org/article"), "web")
        self.assertEqual(parsers.classify("notes.docx"), "local_file")

    def test_local_file_fallback_reads_plain(self):
        p = os.path.join(tempfile.mkdtemp(), "n.txt")
        with open(p, "w") as f:
            f.write("# Title\n\nbody content")
        doc = parsers.fetch_and_parse(p)
        self.assertIn("body content", doc.markdown)
        self.assertEqual(doc.kind, "local_file")

    def test_disabled_by_default(self):
        # ERIS_PARSERS unset → heavy parsers not used (router still works via fallback).
        self.assertFalse(parsers.parsers_enabled())


if __name__ == "__main__":
    unittest.main()
