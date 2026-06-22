"""Tests for the standalone hybrid retrieval module (roadmap 1.3)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.retrieval.hybrid import BM25, reciprocal_rank_fusion, hybrid_search


class _Rec:
    def __init__(self, text, embedding=None):
        self.text = text
        self.embedding = embedding


class TestBM25(unittest.TestCase):
    def test_exact_token_ranks_first(self):
        corpus = [
            "the resonant field evolves over time",
            "sgtpatent describes statistical gating technology",
            "a discussion about coherence and dynamics",
        ]
        scores = BM25(corpus).scores("sgtpatent")
        self.assertEqual(int(np.argmax(scores)), 1)   # only doc 1 has the rare token
        self.assertGreater(scores[1], 0.0)

    def test_idf_is_nonnegative(self):
        bm = BM25(["a a a b", "a c", "a d"])
        self.assertTrue(all(v >= 0 for v in bm.idf.values()))


class TestRRF(unittest.TestCase):
    def test_fuses_two_rankings(self):
        # item 2 is top of list A and second of list B -> should win overall
        fused = reciprocal_rank_fusion([[2, 0, 1], [0, 2, 1]])
        self.assertEqual(fused[0], 2)


class TestHybridSearch(unittest.TestCase):
    def test_lexical_catches_what_dense_misses(self):
        # An exact identifier the (deterministic) embedding won't privilege, but
        # BM25 will — the whole point of hybrid retrieval.
        recs = [
            _Rec("notes on weather and clouds today"),
            _Rec("the file sgtpatent contains the gating proof"),
            _Rec("a recipe for bread and butter"),
        ]
        out = hybrid_search("sgtpatent", recs, top_k=1)
        self.assertEqual(out[0].text, recs[1].text)

    def test_returns_records_and_respects_top_k(self):
        from eris.knowledge.embeddings import get_embedding
        recs = [_Rec(t, get_embedding(t)) for t in
                ("alpha topic one", "beta topic two", "gamma topic three",
                 "delta topic four")]
        out = hybrid_search("beta topic", recs,
                            query_embedding=get_embedding("beta topic"), top_k=2)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(r, _Rec) for r in out))

    def test_reranker_reorders(self):
        recs = [_Rec("one"), _Rec("two"), _Rec("three")]
        # reranker that always prefers the LAST candidate text
        def rr(query, texts):
            return [1.0 if t == "three" else 0.0 for t in texts]
        out = hybrid_search("anything", recs, top_k=3, reranker=rr)
        self.assertEqual(out[0].text, "three")

    def test_empty_records(self):
        self.assertEqual(hybrid_search("q", []), [])


if __name__ == "__main__":
    unittest.main()
