"""§3: the arbiter rewards the gold passage and PENALIZES the plausible-but-wrong
near-neighbour — i.e. it scores task success, not cosine agreement."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.dual.arbiter import Arbiter, citation_resolves, gold_passage_at_k
from eris.dual.types import RetrievalResult


class _Rec:
    def __init__(self, text, sha):
        self.text = text
        self.metadata = {"sha256": sha}
        self.source = "reading:paper"


def _res(*recs):
    return RetrievalResult(records=list(recs), scores=[1.0] * len(recs))


QUERY = "what causes Kuramoto synchronization above critical coupling?"
GOLD = _Rec("Kuramoto coupling causes synchronization above a critical coupling K.", "gold")
# Lexically close, but the WRONG fact (no answer) — a cosine would rank it well.
NEIGHBOUR = _Rec("Kuramoto coupling and synchronization appear widely in biology overviews.", "near")
GOLD_ID = "sha:gold"


class TestArbiterIndependence(unittest.TestCase):
    def test_gold_scores_high_neighbour_scores_low(self):
        arb = Arbiter()
        s_gold = arb.score(QUERY, _res(GOLD, NEIGHBOUR), gold=GOLD_ID)
        s_near = arb.score(QUERY, _res(NEIGHBOUR), gold=GOLD_ID)
        self.assertEqual(s_gold["gold_at_k"], 1.0)
        self.assertEqual(s_near["gold_at_k"], 0.0)
        self.assertGreater(s_gold["success"], 0.4)
        self.assertLess(s_near["success"], 0.1)        # killed despite lexical closeness
        # The neighbour IS lexically close — proving success is NOT just `cite`.
        self.assertGreater(s_near["cite"], 0.3)

    def test_citation_resolves_is_lexical(self):
        self.assertGreater(citation_resolves(QUERY, _res(GOLD)), 0.7)
        self.assertEqual(citation_resolves(QUERY, _res()), 0.0)

    def test_gold_at_k_rank(self):
        hit, rank = gold_passage_at_k(_res(NEIGHBOUR, GOLD), GOLD_ID, k=8)
        self.assertEqual(hit, 1.0)
        self.assertEqual(rank, 2)
        miss, r = gold_passage_at_k(_res(NEIGHBOUR), GOLD_ID, k=8)
        self.assertEqual(miss, 0.0)
        self.assertIsNone(r)

    def test_no_gold_uses_lexical_only(self):
        arb = Arbiter()
        s = arb.score(QUERY, _res(GOLD))     # no gold given
        self.assertNotIn("gold_at_k", s)
        self.assertIn("success", s)


if __name__ == "__main__":
    unittest.main()
